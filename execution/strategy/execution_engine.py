"""
执行引擎模块
- TWAP时间加权平均价格
- VWAP成交量加权平均价格
- 冰山订单
- 智能路由
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from abc import ABC, abstractmethod
import time
import random
from collections import deque

from ..exchange.exchange_client import (
    BaseExchange, Order, OrderType, OrderSide, OrderStatus
)
from ..order.order_manager import OrderManager, OrderRequest
from ..slippage.slippage_model import SlippageModel, MarketImpactModel

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """执行策略类型"""
    TWAP = "twap"           # 时间加权
    VWAP = "vwap"           # 成交量加权
    ICEBERG = "iceberg"     # 冰山订单
    SMART = "smart"         # 智能路由
    DIRECT = "direct"       # 直接执行


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionConfig:
    """执行配置"""
    strategy: ExecutionStrategy
    symbol: str
    side: OrderSide
    total_amount: Decimal
    price_limit: Optional[Decimal] = None
    time_limit: float = 300.0  # 执行时间限制(秒)
    min_interval: float = 1.0  # 最小执行间隔
    max_slippage: Decimal = Decimal("0.01")  # 最大滑点1%
    urgency: float = 0.5  # 紧急程度 0-1
    allow_partial: bool = True
    hidden: bool = False


@dataclass
class ExecutionSlice:
    """执行切片"""
    index: int
    amount: Decimal
    target_time: float
    executed: bool = False
    order_id: Optional[str] = None
    executed_price: Optional[Decimal] = None
    executed_time: Optional[float] = None


@dataclass
class ExecutionReport:
    """执行报告"""
    execution_id: str
    strategy: ExecutionStrategy
    symbol: str
    side: OrderSide
    status: ExecutionStatus
    total_amount: Decimal
    filled_amount: Decimal
    avg_price: Decimal
    total_cost: Decimal
    slippage: Decimal
    start_time: float
    end_time: Optional[float]
    slices: List[ExecutionSlice]
    orders: List[str]


class BaseExecutionStrategy(ABC):
    """执行策略基类"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        slippage_model: Optional[SlippageModel] = None
    ):
        self.order_manager = order_manager
        self.slippage_model = slippage_model or SlippageModel()
        self.impact_model = MarketImpactModel()
    
    @abstractmethod
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """执行订单"""
        pass
    
    @abstractmethod
    def generate_slices(
        self,
        config: ExecutionConfig
    ) -> List[ExecutionSlice]:
        """生成执行切片"""
        pass


class TWAPStrategy(BaseExecutionStrategy):
    """TWAP时间加权平均价格策略"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        slippage_model: Optional[SlippageModel] = None,
        num_slices: int = 10
    ):
        super().__init__(order_manager, slippage_model)
        self.num_slices = num_slices
    
    def generate_slices(self, config: ExecutionConfig) -> List[ExecutionSlice]:
        """生成TWAP切片"""
        slices = []
        slice_amount = config.total_amount / self.num_slices
        interval = config.time_limit / self.num_slices
        
        for i in range(self.num_slices):
            # 添加随机扰动，避免模式识别
            jitter = random.uniform(-0.1, 0.1) * interval
            target_time = time.time() + i * interval + jitter
            
            # 最后一个切片处理余数
            if i == self.num_slices - 1:
                amount = config.total_amount - sum(s.amount for s in slices)
            else:
                amount = slice_amount
            
            slices.append(ExecutionSlice(
                index=i,
                amount=amount,
                target_time=target_time
            ))
        
        return slices
    
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """执行TWAP订单"""
        execution_id = f"twap_{int(time.time() * 1000)}"
        start_time = time.time()
        
        slices = self.generate_slices(config)
        orders = []
        filled_amount = Decimal("0")
        total_cost = Decimal("0")
        
        report = ExecutionReport(
            execution_id=execution_id,
            strategy=ExecutionStrategy.TWAP,
            symbol=config.symbol,
            side=config.side,
            status=ExecutionStatus.RUNNING,
            total_amount=config.total_amount,
            filled_amount=Decimal("0"),
            avg_price=Decimal("0"),
            total_cost=Decimal("0"),
            slippage=Decimal("0"),
            start_time=start_time,
            end_time=None,
            slices=slices,
            orders=[]
        )
        
        for slice_info in slices:
            # 检查是否超时
            if time.time() - start_time > config.time_limit:
                logger.warning(f"TWAP执行超时: {execution_id}")
                report.status = ExecutionStatus.FAILED
                break
            
            # 等待到目标时间
            wait_time = slice_info.target_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            try:
                # 获取当前市场价格
                ticker = await self.order_manager.exchange.get_ticker(config.symbol)
                current_price = Decimal(str(ticker.get('last', 0)))
                
                # 估算滑点
                estimated_slippage = self.slippage_model.estimate_slippage(
                    config.symbol,
                    slice_info.amount,
                    config.side,
                    current_price
                )
                
                # 检查滑点限制
                if estimated_slippage > config.max_slippage:
                    logger.warning(f"滑点超过限制: {estimated_slippage} > {config.max_slippage}")
                    if not config.allow_partial:
                        report.status = ExecutionStatus.FAILED
                        break
                    continue
                
                # 下市价单
                order = await self.order_manager.place_market_order(
                    symbol=config.symbol,
                    side=config.side,
                    amount=slice_info.amount
                )
                
                if order:
                    slice_info.executed = True
                    slice_info.order_id = order.id
                    slice_info.executed_price = order.cost / order.filled if order.filled > 0 else current_price
                    slice_info.executed_time = time.time()
                    
                    orders.append(order.id)
                    filled_amount += order.filled
                    total_cost += order.cost
                    
                    # 更新报告
                    report.filled_amount = filled_amount
                    report.total_cost = total_cost
                    report.avg_price = total_cost / filled_amount if filled_amount > 0 else Decimal("0")
                    report.orders = orders
                    
                    if progress_callback:
                        progress_callback(report)
                
            except Exception as e:
                logger.error(f"TWAP切片执行失败: {e}")
                if not config.allow_partial:
                    report.status = ExecutionStatus.FAILED
                    break
        
        # 完成报告
        if report.status == ExecutionStatus.RUNNING:
            if filled_amount >= config.total_amount:
                report.status = ExecutionStatus.COMPLETED
            elif filled_amount > 0:
                report.status = ExecutionStatus.COMPLETED if config.allow_partial else ExecutionStatus.FAILED
            else:
                report.status = ExecutionStatus.FAILED
        
        report.end_time = time.time()
        
        # 计算实际滑点
        if filled_amount > 0:
            ticker = await self.order_manager.exchange.get_ticker(config.symbol)
            vwap = Decimal(str(ticker.get('vwap', ticker.get('last', 0))))
            if vwap > 0:
                report.slippage = (report.avg_price - vwap) / vwap
        
        logger.info(f"TWAP执行完成: {execution_id}, 状态: {report.status.value}")
        return report


class VWAPStrategy(BaseExecutionStrategy):
    """VWAP成交量加权平均价格策略"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        slippage_model: Optional[SlippageModel] = None,
        volume_history_size: int = 100
    ):
        super().__init__(order_manager, slippage_model)
        self.volume_history: Dict[str, deque] = {}
        self.volume_history_size = volume_history_size
    
    def update_volume_profile(self, symbol: str, volume: Decimal):
        """更新成交量分布"""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.volume_history_size)
        self.volume_history[symbol].append(volume)
    
    def get_volume_profile(self, symbol: str) -> List[Decimal]:
        """获取成交量分布"""
        if symbol not in self.volume_history:
            return []
        return list(self.volume_history[symbol])
    
    def generate_slices(self, config: ExecutionConfig) -> List[ExecutionSlice]:
        """生成VWAP切片"""
        volume_profile = self.get_volume_profile(config.symbol)
        
        if not volume_profile:
            # 没有历史数据，使用均匀分布
            return TWAPStrategy(self.order_manager).generate_slices(config)
        
        slices = []
        total_volume = sum(volume_profile)
        
        if total_volume == 0:
            return TWAPStrategy(self.order_manager).generate_slices(config)
        
        # 根据成交量分布分配订单
        current_time = time.time()
        for i, volume in enumerate(volume_profile):
            ratio = volume / total_volume
            amount = config.total_amount * Decimal(str(ratio))
            target_time = current_time + i * (config.time_limit / len(volume_profile))
            
            slices.append(ExecutionSlice(
                index=i,
                amount=amount,
                target_time=target_time
            ))
        
        return slices
    
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """执行VWAP订单"""
        execution_id = f"vwap_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # 获取历史成交量数据
        try:
            trades = await self.order_manager.exchange.get_recent_trades(
                config.symbol, limit=100
            )
            for trade in trades:
                self.update_volume_profile(
                    config.symbol,
                    Decimal(str(trade.get('amount', 0)))
                )
        except Exception as e:
            logger.warning(f"获取历史成交数据失败: {e}")
        
        slices = self.generate_slices(config)
        orders = []
        filled_amount = Decimal("0")
        total_cost = Decimal("0")
        
        report = ExecutionReport(
            execution_id=execution_id,
            strategy=ExecutionStrategy.VWAP,
            symbol=config.symbol,
            side=config.side,
            status=ExecutionStatus.RUNNING,
            total_amount=config.total_amount,
            filled_amount=Decimal("0"),
            avg_price=Decimal("0"),
            total_cost=Decimal("0"),
            slippage=Decimal("0"),
            start_time=start_time,
            end_time=None,
            slices=slices,
            orders=[]
        )
        
        for slice_info in slices:
            if time.time() - start_time > config.time_limit:
                logger.warning(f"VWAP执行超时: {execution_id}")
                report.status = ExecutionStatus.FAILED
                break
            
            wait_time = slice_info.target_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            try:
                # 检查当前成交量
                recent_trades = await self.order_manager.exchange.get_recent_trades(
                    config.symbol, limit=10
                )
                recent_volume = sum(
                    Decimal(str(t.get('amount', 0))) for t in recent_trades
                )
                
                # 根据成交量调整切片大小
                avg_volume = sum(self.get_volume_profile(config.symbol)) / len(self.volume_history[config.symbol]) if config.symbol in self.volume_history else recent_volume
                
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    adjusted_amount = slice_info.amount * Decimal(str(min(volume_ratio * 1.5, 2.0)))
                else:
                    adjusted_amount = slice_info.amount
                
                # 下限价单
                ticker = await self.order_manager.exchange.get_ticker(config.symbol)
                current_price = Decimal(str(ticker.get('last', 0)))
                
                # 根据方向调整价格
                if config.side == OrderSide.BUY:
                    limit_price = current_price * (1 + config.max_slippage * Decimal("0.5"))
                else:
                    limit_price = current_price * (1 - config.max_slippage * Decimal("0.5"))
                
                order = await self.order_manager.place_limit_order(
                    symbol=config.symbol,
                    side=config.side,
                    amount=adjusted_amount,
                    price=limit_price
                )
                
                if order:
                    # 等待成交
                    filled_order = await self.order_manager.wait_for_fill(
                        order.id, timeout=30
                    )
                    
                    if filled_order:
                        slice_info.executed = True
                        slice_info.order_id = filled_order.id
                        slice_info.executed_price = filled_order.cost / filled_order.filled if filled_order.filled > 0 else current_price
                        slice_info.executed_time = time.time()
                        
                        orders.append(filled_order.id)
                        filled_amount += filled_order.filled
                        total_cost += filled_order.cost
                        
                        report.filled_amount = filled_amount
                        report.total_cost = total_cost
                        report.avg_price = total_cost / filled_amount if filled_amount > 0 else Decimal("0")
                        report.orders = orders
                        
                        if progress_callback:
                            progress_callback(report)
                
            except Exception as e:
                logger.error(f"VWAP切片执行失败: {e}")
                if not config.allow_partial:
                    report.status = ExecutionStatus.FAILED
                    break
        
        if report.status == ExecutionStatus.RUNNING:
            if filled_amount >= config.total_amount:
                report.status = ExecutionStatus.COMPLETED
            elif filled_amount > 0:
                report.status = ExecutionStatus.COMPLETED if config.allow_partial else ExecutionStatus.FAILED
            else:
                report.status = ExecutionStatus.FAILED
        
        report.end_time = time.time()
        
        # 计算VWAP滑点
        if filled_amount > 0:
            try:
                trades = await self.order_manager.exchange.get_recent_trades(
                    config.symbol, limit=len(slices)
                )
                market_vwap = sum(
                    Decimal(str(t.get('amount', 0))) * Decimal(str(t.get('price', 0)))
                    for t in trades
                ) / sum(Decimal(str(t.get('amount', 1))) for t in trades) if trades else Decimal("0")
                
                if market_vwap > 0:
                    report.slippage = (report.avg_price - market_vwap) / market_vwap
            except Exception as e:
                logger.warning(f"计算VWAP滑点失败: {e}")
        
        logger.info(f"VWAP执行完成: {execution_id}, 状态: {report.status.value}")
        return report


class IcebergStrategy(BaseExecutionStrategy):
    """冰山订单策略"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        slippage_model: Optional[SlippageModel] = None,
        display_size: Optional[Decimal] = None,
        variance: float = 0.2
    ):
        super().__init__(order_manager, slippage_model)
        self.display_size = display_size
        self.variance = variance
    
    def generate_slices(self, config: ExecutionConfig) -> List[ExecutionSlice]:
        """生成冰山切片"""
        # 确定显示数量
        if self.display_size:
            display = self.display_size
        else:
            # 默认显示1-5%的订单
            display = config.total_amount * Decimal(str(random.uniform(0.01, 0.05)))
        
        slices = []
        remaining = config.total_amount
        index = 0
        
        while remaining > 0:
            # 添加随机变化
            variance = Decimal(str(random.uniform(1 - self.variance, 1 + self.variance)))
            slice_amount = min(display * variance, remaining)
            
            slices.append(ExecutionSlice(
                index=index,
                amount=slice_amount,
                target_time=time.time() + index * config.min_interval
            ))
            
            remaining -= slice_amount
            index += 1
        
        return slices
    
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """执行冰山订单"""
        execution_id = f"iceberg_{int(time.time() * 1000)}"
        start_time = time.time()
        
        slices = self.generate_slices(config)
        orders = []
        filled_amount = Decimal("0")
        total_cost = Decimal("0")
        
        report = ExecutionReport(
            execution_id=execution_id,
            strategy=ExecutionStrategy.ICEBERG,
            symbol=config.symbol,
            side=config.side,
            status=ExecutionStatus.RUNNING,
            total_amount=config.total_amount,
            filled_amount=Decimal("0"),
            avg_price=Decimal("0"),
            total_cost=Decimal("0"),
            slippage=Decimal("0"),
            start_time=start_time,
            end_time=None,
            slices=slices,
            orders=[]
        )
        
        for slice_info in slices:
            if time.time() - start_time > config.time_limit:
                logger.warning(f"冰山订单执行超时: {execution_id}")
                report.status = ExecutionStatus.FAILED
                break
            
            try:
                # 获取订单簿
                orderbook = await self.order_manager.exchange.get_orderbook(config.symbol)
                
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if config.side == OrderSide.BUY:
                    # 买价略低于最优卖价
                    best_price = Decimal(str(asks[0][0])) if asks else Decimal("0")
                    limit_price = best_price * Decimal("0.9999")
                else:
                    # 卖价略高于最优买价
                    best_price = Decimal(str(bids[0][0])) if bids else Decimal("0")
                    limit_price = best_price * Decimal("1.0001")
                
                # 检查价格限制
                if config.price_limit:
                    if config.side == OrderSide.BUY and limit_price > config.price_limit:
                        limit_price = config.price_limit
                    elif config.side == OrderSide.SELL and limit_price < config.price_limit:
                        limit_price = config.price_limit
                
                # 下限价单
                order = await self.order_manager.place_limit_order(
                    symbol=config.symbol,
                    side=config.side,
                    amount=slice_info.amount,
                    price=limit_price
                )
                
                if order:
                    # 等待成交或刷新
                    filled_order = await self.order_manager.wait_for_fill(
                        order.id, timeout=config.min_interval
                    )
                    
                    if filled_order and filled_order.filled > 0:
                        slice_info.executed = True
                        slice_info.order_id = filled_order.id
                        slice_info.executed_price = filled_order.cost / filled_order.filled
                        slice_info.executed_time = time.time()
                        
                        orders.append(filled_order.id)
                        filled_amount += filled_order.filled
                        total_cost += filled_order.cost
                        
                        report.filled_amount = filled_amount
                        report.total_cost = total_cost
                        report.avg_price = total_cost / filled_amount if filled_amount > 0 else Decimal("0")
                        report.orders = orders
                        
                        if progress_callback:
                            progress_callback(report)
                    
                    # 取消未成交部分
                    if filled_order and filled_order.remaining > 0:
                        await self.order_manager.cancel_order(
                            filled_order.id, config.symbol
                        )
                
                # 添加随机延迟，模拟人工操作
                delay = config.min_interval * random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"冰山切片执行失败: {e}")
                if not config.allow_partial:
                    report.status = ExecutionStatus.FAILED
                    break
        
        if report.status == ExecutionStatus.RUNNING:
            if filled_amount >= config.total_amount:
                report.status = ExecutionStatus.COMPLETED
            elif filled_amount > 0:
                report.status = ExecutionStatus.COMPLETED if config.allow_partial else ExecutionStatus.FAILED
            else:
                report.status = ExecutionStatus.FAILED
        
        report.end_time = time.time()
        
        logger.info(f"冰山订单执行完成: {execution_id}, 状态: {report.status.value}")
        return report


class SmartRouter(BaseExecutionStrategy):
    """智能路由策略"""
    
    def __init__(
        self,
        order_manager: OrderManager,
        slippage_model: Optional[SlippageModel] = None
    ):
        super().__init__(order_manager, slippage_model)
        self.exchange_scores: Dict[str, float] = {}
    
    def generate_slices(self, config: ExecutionConfig) -> List[ExecutionSlice]:
        """生成智能路由切片"""
        # 智能路由通常直接执行
        return [ExecutionSlice(
            index=0,
            amount=config.total_amount,
            target_time=time.time()
        )]
    
    async def analyze_market_conditions(self, symbol: str) -> Dict:
        """分析市场条件"""
        try:
            # 获取订单簿
            orderbook = await self.order_manager.exchange.get_orderbook(symbol)
            
            # 计算买卖价差
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if bids and asks:
                best_bid = Decimal(str(bids[0][0]))
                best_ask = Decimal(str(asks[0][0]))
                spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
            else:
                spread = Decimal("0")
            
            # 计算订单簿深度
            bid_depth = sum(Decimal(str(b[1])) for b in bids[:10])
            ask_depth = sum(Decimal(str(a[1])) for a in asks[:10])
            
            # 计算波动率（简化版）
            ticker = await self.order_manager.exchange.get_ticker(symbol)
            high = Decimal(str(ticker.get('high', 0)))
            low = Decimal(str(ticker.get('low', 0)))
            volatility = (high - low) / low if low > 0 else Decimal("0")
            
            return {
                'spread': float(spread),
                'bid_depth': float(bid_depth),
                'ask_depth': float(ask_depth),
                'volatility': float(volatility),
                'best_bid': float(best_bid),
                'best_ask': float(best_ask)
            }
            
        except Exception as e:
            logger.error(f"分析市场条件失败: {e}")
            return {}
    
    def select_strategy(self, market_conditions: Dict, config: ExecutionConfig) -> ExecutionStrategy:
        """选择最优执行策略"""
        spread = market_conditions.get('spread', 0)
        volatility = market_conditions.get('volatility', 0)
        depth = min(market_conditions.get('bid_depth', 0), market_conditions.get('ask_depth', 0))
        
        # 根据市场条件选择策略
        if spread < 0.001 and volatility < 0.02 and depth > float(config.total_amount * 2):
            # 市场条件好，直接执行
            return ExecutionStrategy.DIRECT
        elif config.total_amount > Decimal('10000') and config.time_limit > 60:
            # 大额订单，使用TWAP或VWAP
            if volatility > 0.05:
                return ExecutionStrategy.VWAP
            else:
                return ExecutionStrategy.TWAP
        else:
            # 默认使用冰山订单
            return ExecutionStrategy.ICEBERG
    
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """智能执行订单"""
        execution_id = f"smart_{int(time.time() * 1000)}"
        
        # 分析市场条件
        market_conditions = await self.analyze_market_conditions(config.symbol)
        logger.info(f"市场条件: {market_conditions}")
        
        # 选择最优策略
        selected_strategy = self.select_strategy(market_conditions, config)
        logger.info(f"选择策略: {selected_strategy.value}")
        
        # 使用选定的策略执行
        if selected_strategy == ExecutionStrategy.DIRECT:
            # 直接执行
            return await self._execute_direct(config, progress_callback)
        elif selected_strategy == ExecutionStrategy.TWAP:
            strategy = TWAPStrategy(self.order_manager, self.slippage_model)
        elif selected_strategy == ExecutionStrategy.VWAP:
            strategy = VWAPStrategy(self.order_manager, self.slippage_model)
        else:
            strategy = IcebergStrategy(self.order_manager, self.slippage_model)
        
        return await strategy.execute(config, progress_callback)
    
    async def _execute_direct(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """直接执行"""
        execution_id = f"direct_{int(time.time() * 1000)}"
        start_time = time.time()
        
        order = await self.order_manager.place_market_order(
            symbol=config.symbol,
            side=config.side,
            amount=config.total_amount
        )
        
        if order:
            status = ExecutionStatus.COMPLETED if order.filled >= config.total_amount else ExecutionStatus.FAILED
            
            report = ExecutionReport(
                execution_id=execution_id,
                strategy=ExecutionStrategy.DIRECT,
                symbol=config.symbol,
                side=config.side,
                status=status,
                total_amount=config.total_amount,
                filled_amount=order.filled,
                avg_price=order.cost / order.filled if order.filled > 0 else Decimal("0"),
                total_cost=order.cost,
                slippage=Decimal("0"),
                start_time=start_time,
                end_time=time.time(),
                slices=[ExecutionSlice(
                    index=0,
                    amount=config.total_amount,
                    target_time=start_time,
                    executed=True,
                    order_id=order.id,
                    executed_price=order.cost / order.filled if order.filled > 0 else None,
                    executed_time=time.time()
                )],
                orders=[order.id]
            )
        else:
            report = ExecutionReport(
                execution_id=execution_id,
                strategy=ExecutionStrategy.DIRECT,
                symbol=config.symbol,
                side=config.side,
                status=ExecutionStatus.FAILED,
                total_amount=config.total_amount,
                filled_amount=Decimal("0"),
                avg_price=Decimal("0"),
                total_cost=Decimal("0"),
                slippage=Decimal("0"),
                start_time=start_time,
                end_time=time.time(),
                slices=[],
                orders=[]
            )
        
        return report


class ExecutionEngine:
    """执行引擎"""
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.slippage_model = SlippageModel()
        self.strategies: Dict[ExecutionStrategy, BaseExecutionStrategy] = {
            ExecutionStrategy.TWAP: TWAPStrategy(order_manager, self.slippage_model),
            ExecutionStrategy.VWAP: VWAPStrategy(order_manager, self.slippage_model),
            ExecutionStrategy.ICEBERG: IcebergStrategy(order_manager, self.slippage_model),
            ExecutionStrategy.SMART: SmartRouter(order_manager, self.slippage_model)
        }
        self.active_executions: Dict[str, ExecutionReport] = {}
    
    async def execute(
        self,
        config: ExecutionConfig,
        progress_callback: Optional[Callable[[ExecutionReport], None]] = None
    ) -> ExecutionReport:
        """执行订单"""
        strategy = self.strategies.get(config.strategy)
        if not strategy:
            raise ValueError(f"未知执行策略: {config.strategy}")
        
        report = await strategy.execute(config, progress_callback)
        self.active_executions[report.execution_id] = report
        
        return report
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        if execution_id not in self.active_executions:
            return False
        
        report = self.active_executions[execution_id]
        report.status = ExecutionStatus.CANCELLED
        
        # 取消所有相关订单
        for order_id in report.orders:
            await self.order_manager.cancel_order(order_id, report.symbol)
        
        return True
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionReport]:
        """获取执行报告"""
        return self.active_executions.get(execution_id)


# 便捷函数
def create_execution_engine(order_manager: OrderManager) -> ExecutionEngine:
    """创建执行引擎"""
    return ExecutionEngine(order_manager)


# 示例用法
if __name__ == "__main__":
    async def test():
        from ..exchange.exchange_client import SimulatedExchange
        
        # 创建模拟交易所
        sim = SimulatedExchange({'USDT': Decimal('100000')})
        await sim.connect()
        sim.set_price('BTC/USDT', Decimal('50000'))
        
        # 创建订单管理器和执行引擎
        order_manager = OrderManager(sim)
        await order_manager.start()
        
        engine = ExecutionEngine(order_manager)
        
        # 配置TWAP执行
        config = ExecutionConfig(
            strategy=ExecutionStrategy.TWAP,
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            total_amount=Decimal('1.0'),
            time_limit=60.0,
            num_slices=5
        )
        
        # 执行
        def on_progress(report: ExecutionReport):
            print(f"进度: {report.filled_amount}/{report.total_amount} @ {report.avg_price}")
        
        report = await engine.execute(config, on_progress)
        print(f"执行完成: {report.status.value}")
        print(f"平均价格: {report.avg_price}")
        print(f"滑点: {report.slippage}")
        
        await order_manager.stop()
        await sim.disconnect()
    
    asyncio.run(test())
