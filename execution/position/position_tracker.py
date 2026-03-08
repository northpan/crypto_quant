"""
持仓跟踪模块
- 现货持仓跟踪
- 合约持仓和保证金管理
- 多币种持仓管理
- 自动对冲
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import defaultdict
import time

from ..exchange.exchange_client import (
    BaseExchange, Order, OrderSide, OrderStatus, Position, PositionSide, Balance
)
from ..order.order_manager import OrderManager, OrderRequest

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """持仓类型"""
    SPOT = "spot"           # 现货
    MARGIN = "margin"       # 杠杆
    FUTURES = "futures"     # 期货
    PERPETUAL = "perpetual" # 永续合约


class HedgeMode(Enum):
    """对冲模式"""
    NONE = "none"           # 不对冲
    FULL = "full"           # 完全对冲
    DELTA = "delta"         # Delta对冲
    BETA = "beta"           # Beta对冲


@dataclass
class PositionUpdate:
    """持仓更新"""
    symbol: str
    position_type: PositionType
    side: PositionSide
    amount: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: int


@dataclass
class Portfolio:
    """投资组合"""
    total_value: Decimal
    cash_value: Decimal
    position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin_used: Decimal
    margin_available: Decimal
    leverage: Decimal
    positions: Dict[str, 'PositionDetail']
    timestamp: int


@dataclass
class PositionDetail:
    """持仓详情"""
    symbol: str
    position_type: PositionType
    side: PositionSide
    amount: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Optional[Decimal]
    margin: Decimal
    leverage: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    pnl_percent: Decimal
    value: Decimal
    weight: Decimal


@dataclass
class HedgeConfig:
    """对冲配置"""
    enabled: bool = False
    mode: HedgeMode = HedgeMode.NONE
    target_hedge_ratio: Decimal = Decimal("1.0")
    hedge_symbols: List[str] = field(default_factory=list)
    rebalance_threshold: Decimal = Decimal("0.05")
    max_hedge_deviation: Decimal = Decimal("0.1")


class PositionTracker:
    """持仓跟踪器"""
    
    def __init__(
        self,
        exchange: BaseExchange,
        order_manager: Optional[OrderManager] = None
    ):
        self.exchange = exchange
        self.order_manager = order_manager
        
        # 持仓数据
        self.positions: Dict[str, PositionDetail] = {}
        self.spot_positions: Dict[str, PositionDetail] = {}
        self.futures_positions: Dict[str, PositionDetail] = {}
        
        # 余额数据
        self.balances: Dict[str, Balance] = {}
        
        # 价格数据
        self.prices: Dict[str, Decimal] = {}
        
        # 对冲配置
        self.hedge_config = HedgeConfig()
        self.hedge_positions: Dict[str, Decimal] = {}
        
        # 回调
        self.position_callbacks: List[Callable[[PositionUpdate], None]] = []
        self.portfolio_callbacks: List[Callable[[Portfolio], None]] = []
        
        # 同步任务
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._sync_interval = 1.0
        
        # 历史记录
        self.position_history: List[Portfolio] = []
        self.max_history_size = 1000
        
        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def add_position_callback(self, callback: Callable[[PositionUpdate], None]):
        """添加持仓回调"""
        self.position_callbacks.append(callback)
    
    def add_portfolio_callback(self, callback: Callable[[Portfolio], None]):
        """添加投资组合回调"""
        self.portfolio_callbacks.append(callback)
    
    async def start(self):
        """启动持仓跟踪器"""
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_positions())
        logger.info("持仓跟踪器已启动")
    
    async def stop(self):
        """停止持仓跟踪器"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("持仓跟踪器已停止")
    
    async def _sync_positions(self):
        """同步持仓数据"""
        while self._running:
            try:
                await self._update_balances()
                await self._update_positions()
                await self._update_prices()
                await self._check_hedge()
                await asyncio.sleep(self._sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"持仓同步错误: {e}")
                await asyncio.sleep(self._sync_interval)
    
    async def _update_balances(self):
        """更新余额"""
        try:
            self.balances = await self.exchange.get_balance()
        except Exception as e:
            logger.error(f"更新余额失败: {e}")
    
    async def _update_positions(self):
        """更新持仓"""
        try:
            positions = await self.exchange.get_positions()
            
            for pos in positions:
                position_type = PositionType.PERPETUAL  # 默认永续合约
                
                detail = PositionDetail(
                    symbol=pos.symbol,
                    position_type=position_type,
                    side=pos.side,
                    amount=pos.amount,
                    entry_price=pos.entry_price,
                    mark_price=pos.mark_price,
                    liquidation_price=pos.liquidation_price,
                    margin=pos.margin,
                    leverage=pos.leverage,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    pnl_percent=self._calculate_pnl_percent(pos),
                    value=pos.amount * pos.mark_price,
                    weight=Decimal("0")
                )
                
                self.positions[pos.symbol] = detail
                self.futures_positions[pos.symbol] = detail
                
                # 通知回调
                update = PositionUpdate(
                    symbol=pos.symbol,
                    position_type=position_type,
                    side=pos.side,
                    amount=pos.amount,
                    entry_price=pos.entry_price,
                    mark_price=pos.mark_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    timestamp=int(time.time() * 1000)
                )
                
                for callback in self.position_callbacks:
                    try:
                        callback(update)
                    except Exception as e:
                        logger.error(f"持仓回调执行失败: {e}")
            
        except Exception as e:
            logger.error(f"更新持仓失败: {e}")
    
    async def _update_prices(self):
        """更新价格数据"""
        symbols = list(self.positions.keys()) + list(self.spot_positions.keys())
        
        for symbol in set(symbols):
            try:
                ticker = await self.exchange.get_ticker(symbol)
                self.prices[symbol] = Decimal(str(ticker.get('last', 0)))
            except Exception as e:
                logger.debug(f"更新价格失败 {symbol}: {e}")
    
    def _calculate_pnl_percent(self, position: Position) -> Decimal:
        """计算盈亏百分比"""
        if position.entry_price == 0:
            return Decimal("0")
        
        if position.side == PositionSide.LONG:
            return (position.mark_price - position.entry_price) / position.entry_price * 100
        else:
            return (position.entry_price - position.mark_price) / position.entry_price * 100
    
    async def _check_hedge(self):
        """检查对冲状态"""
        if not self.hedge_config.enabled:
            return
        
        try:
            # 计算当前净敞口
            net_exposure = self._calculate_net_exposure()
            
            # 计算目标对冲敞口
            target_hedge = net_exposure * self.hedge_config.target_hedge_ratio
            
            # 计算当前对冲敞口
            current_hedge = sum(self.hedge_positions.values())
            
            # 检查是否需要再平衡
            deviation = abs(current_hedge - target_hedge)
            threshold = abs(target_hedge) * self.hedge_config.rebalance_threshold
            
            if deviation > threshold:
                logger.info(f"对冲偏差: {deviation:.4f}, 阈值: {threshold:.4f}")
                await self._rebalance_hedge(target_hedge)
                
        except Exception as e:
            logger.error(f"检查对冲失败: {e}")
    
    def _calculate_net_exposure(self) -> Decimal:
        """计算净敞口"""
        exposure = Decimal("0")
        
        for pos in self.positions.values():
            value = pos.amount * pos.mark_price
            if pos.side == PositionSide.LONG:
                exposure += value
            else:
                exposure -= value
        
        return exposure
    
    async def _rebalance_hedge(self, target_hedge: Decimal):
        """再平衡对冲"""
        if not self.order_manager:
            logger.warning("没有订单管理器，无法执行对冲")
            return
        
        try:
            # 使用对冲标的进行对冲
            for hedge_symbol in self.hedge_config.hedge_symbols:
                if hedge_symbol not in self.prices:
                    continue
                
                price = self.prices[hedge_symbol]
                hedge_amount = abs(target_hedge) / price
                
                # 确定方向
                if target_hedge > 0:
                    side = OrderSide.BUY
                else:
                    side = OrderSide.SELL
                
                # 下对冲单
                await self.order_manager.place_market_order(
                    symbol=hedge_symbol,
                    side=side,
                    amount=hedge_amount
                )
                
                self.hedge_positions[hedge_symbol] = target_hedge
                logger.info(f"对冲再平衡: {hedge_symbol} {side.value} {hedge_amount}")
                break
                
        except Exception as e:
            logger.error(f"对冲再平衡失败: {e}")
    
    def update_spot_position(
        self,
        symbol: str,
        amount: Decimal,
        avg_price: Decimal,
        current_price: Decimal
    ):
        """更新现货持仓"""
        # 解析交易对
        base, quote = symbol.split('/')
        
        # 获取当前持仓
        current_pos = self.spot_positions.get(symbol)
        
        if current_pos:
            # 更新现有持仓
            total_amount = current_pos.amount + amount
            if total_amount > 0:
                total_cost = (current_pos.amount * current_pos.entry_price + 
                            amount * avg_price)
                new_entry = total_cost / total_amount
            else:
                new_entry = avg_price
            
            unrealized_pnl = (current_price - new_entry) * total_amount
            
            detail = PositionDetail(
                symbol=symbol,
                position_type=PositionType.SPOT,
                side=PositionSide.LONG if total_amount > 0 else PositionSide.SHORT,
                amount=abs(total_amount),
                entry_price=new_entry,
                mark_price=current_price,
                liquidation_price=None,
                margin=Decimal("0"),
                leverage=Decimal("1"),
                unrealized_pnl=unrealized_pnl,
                realized_pnl=current_pos.realized_pnl,
                pnl_percent=(current_price - new_entry) / new_entry * 100 if new_entry > 0 else Decimal("0"),
                value=abs(total_amount) * current_price,
                weight=Decimal("0")
            )
        else:
            # 新建持仓
            detail = PositionDetail(
                symbol=symbol,
                position_type=PositionType.SPOT,
                side=PositionSide.LONG if amount > 0 else PositionSide.SHORT,
                amount=abs(amount),
                entry_price=avg_price,
                mark_price=current_price,
                liquidation_price=None,
                margin=Decimal("0"),
                leverage=Decimal("1"),
                unrealized_pnl=(current_price - avg_price) * amount,
                realized_pnl=Decimal("0"),
                pnl_percent=(current_price - avg_price) / avg_price * 100 if avg_price > 0 else Decimal("0"),
                value=abs(amount) * current_price,
                weight=Decimal("0")
            )
        
        self.spot_positions[symbol] = detail
        
        # 通知回调
        update = PositionUpdate(
            symbol=symbol,
            position_type=PositionType.SPOT,
            side=detail.side,
            amount=detail.amount,
            entry_price=detail.entry_price,
            mark_price=detail.mark_price,
            unrealized_pnl=detail.unrealized_pnl,
            realized_pnl=detail.realized_pnl,
            timestamp=int(time.time() * 1000)
        )
        
        for callback in self.position_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"持仓回调执行失败: {e}")
    
    def on_order_filled(self, order: Order):
        """订单成交回调"""
        try:
            # 更新现货持仓
            if '/' in order.symbol:
                base, quote = order.symbol.split('/')
                
                # 确定持仓变化
                if order.side == OrderSide.BUY:
                    amount_change = order.filled
                else:
                    amount_change = -order.filled
                
                avg_price = order.cost / order.filled if order.filled > 0 else Decimal("0")
                current_price = self.prices.get(order.symbol, avg_price)
                
                self.update_spot_position(
                    symbol=order.symbol,
                    amount=amount_change,
                    avg_price=avg_price,
                    current_price=current_price
                )
            
            # 更新统计
            self.total_trades += 1
            
        except Exception as e:
            logger.error(f"处理成交回调失败: {e}")
    
    def get_position(self, symbol: str) -> Optional[PositionDetail]:
        """获取持仓"""
        return self.positions.get(symbol) or self.spot_positions.get(symbol)
    
    def get_all_positions(self) -> List[PositionDetail]:
        """获取所有持仓"""
        all_positions = list(self.positions.values()) + list(self.spot_positions.values())
        return [p for p in all_positions if p.amount > 0]
    
    def get_positions_by_type(self, position_type: PositionType) -> List[PositionDetail]:
        """按类型获取持仓"""
        if position_type == PositionType.SPOT:
            return list(self.spot_positions.values())
        else:
            return list(self.futures_positions.values())
    
    def get_portfolio(self) -> Portfolio:
        """获取投资组合"""
        total_value = Decimal("0")
        position_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        realized_pnl = Decimal("0")
        margin_used = Decimal("0")
        
        all_positions = self.get_all_positions()
        
        for pos in all_positions:
            value = pos.value
            position_value += value
            unrealized_pnl += pos.unrealized_pnl
            realized_pnl += pos.realized_pnl
            margin_used += pos.margin
        
        # 计算现金价值
        cash_value = Decimal("0")
        for balance in self.balances.values():
            if balance.currency in ['USDT', 'USD', 'BUSD']:
                cash_value += balance.total
            elif f"{balance.currency}/USDT" in self.prices:
                cash_value += balance.total * self.prices[f"{balance.currency}/USDT"]
        
        total_value = cash_value + position_value
        
        # 计算权重
        positions_with_weight = {}
        for symbol, pos in self.positions.items():
            weight = pos.value / total_value if total_value > 0 else Decimal("0")
            pos_dict = {
                'symbol': pos.symbol,
                'position_type': pos.position_type,
                'side': pos.side,
                'amount': pos.amount,
                'entry_price': pos.entry_price,
                'mark_price': pos.mark_price,
                'value': pos.value,
                'weight': weight,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            }
            positions_with_weight[symbol] = pos_dict
        
        portfolio = Portfolio(
            total_value=total_value,
            cash_value=cash_value,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            margin_used=margin_used,
            margin_available=total_value - margin_used,
            leverage=position_value / total_value if total_value > 0 else Decimal("0"),
            positions=positions_with_weight,
            timestamp=int(time.time() * 1000)
        )
        
        # 保存历史
        self.position_history.append(portfolio)
        if len(self.position_history) > self.max_history_size:
            self.position_history = self.position_history[-self.max_history_size:]
        
        # 通知回调
        for callback in self.portfolio_callbacks:
            try:
                callback(portfolio)
            except Exception as e:
                logger.error(f"投资组合回调执行失败: {e}")
        
        return portfolio
    
    def set_hedge_config(self, config: HedgeConfig):
        """设置对冲配置"""
        self.hedge_config = config
        logger.info(f"对冲配置已更新: {config.mode.value}")
    
    def enable_hedge(
        self,
        mode: HedgeMode,
        target_ratio: Decimal = Decimal("1.0"),
        hedge_symbols: Optional[List[str]] = None,
        rebalance_threshold: Decimal = Decimal("0.05")
    ):
        """启用对冲"""
        self.hedge_config = HedgeConfig(
            enabled=True,
            mode=mode,
            target_hedge_ratio=target_ratio,
            hedge_symbols=hedge_symbols or [],
            rebalance_threshold=rebalance_threshold
        )
        logger.info(f"对冲已启用: {mode.value}, 目标比例: {target_ratio}")
    
    def disable_hedge(self):
        """禁用对冲"""
        self.hedge_config.enabled = False
        logger.info("对冲已禁用")
    
    def get_position_summary(self) -> Dict:
        """获取持仓摘要"""
        portfolio = self.get_portfolio()
        
        return {
            'total_value': float(portfolio.total_value),
            'cash_value': float(portfolio.cash_value),
            'position_value': float(portfolio.position_value),
            'unrealized_pnl': float(portfolio.unrealized_pnl),
            'realized_pnl': float(portfolio.realized_pnl),
            'margin_used': float(portfolio.margin_used),
            'margin_available': float(portfolio.margin_available),
            'leverage': float(portfolio.leverage),
            'position_count': len(portfolio.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
    
    def get_risk_metrics(self) -> Dict:
        """获取风险指标"""
        portfolio = self.get_portfolio()
        
        # 计算集中度
        max_weight = Decimal("0")
        for pos in portfolio.positions.values():
            max_weight = max(max_weight, pos['weight'])
        
        # 计算多空比例
        long_value = Decimal("0")
        short_value = Decimal("0")
        
        for pos in self.positions.values():
            if pos.side == PositionSide.LONG:
                long_value += pos.value
            else:
                short_value += pos.value
        
        return {
            'max_position_weight': float(max_weight),
            'long_value': float(long_value),
            'short_value': float(short_value),
            'net_exposure': float(self._calculate_net_exposure()),
            'gross_exposure': float(long_value + short_value),
            'margin_ratio': float(portfolio.margin_used / portfolio.total_value) if portfolio.total_value > 0 else 0,
            'hedge_enabled': self.hedge_config.enabled,
            'hedge_mode': self.hedge_config.mode.value if self.hedge_config.enabled else None
        }


class PositionSizer:
    """仓位管理器"""
    
    @staticmethod
    def fixed_size(
        capital: Decimal,
        fixed_amount: Decimal,
        price: Decimal
    ) -> Decimal:
        """固定金额仓位"""
        return fixed_amount / price
    
    @staticmethod
    def fixed_percentage(
        capital: Decimal,
        percentage: Decimal,
        price: Decimal
    ) -> Decimal:
        """固定百分比仓位"""
        amount = capital * percentage / price
        return amount
    
    @staticmethod
    def kelly_criterion(
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        capital: Decimal,
        price: Decimal
    ) -> Decimal:
        """凯利公式仓位"""
        if avg_loss == 0:
            return Decimal("0")
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(Decimal("0"), min(kelly, Decimal("0.25")))  # 限制最大25%
        
        return capital * kelly / price
    
    @staticmethod
    def risk_based(
        capital: Decimal,
        risk_per_trade: Decimal,
        stop_loss_pct: Decimal,
        price: Decimal
    ) -> Decimal:
        """基于风险的仓位"""
        if stop_loss_pct <= 0:
            return Decimal("0")
        
        risk_amount = capital * risk_per_trade
        position_value = risk_amount / stop_loss_pct
        
        return position_value / price


# 便捷函数
def create_position_tracker(
    exchange: BaseExchange,
    order_manager: Optional[OrderManager] = None
) -> PositionTracker:
    """创建持仓跟踪器"""
    return PositionTracker(exchange, order_manager)


# 示例用法
if __name__ == "__main__":
    async def test():
        from ..exchange.exchange_client import SimulatedExchange
        
        # 创建模拟交易所
        sim = SimulatedExchange({
            'USDT': Decimal('10000'),
            'BTC': Decimal('0.5')
        })
        await sim.connect()
        sim.set_price('BTC/USDT', Decimal('50000'))
        
        # 创建持仓跟踪器
        tracker = PositionTracker(sim)
        await tracker.start()
        
        # 更新现货持仓
        tracker.update_spot_position(
            symbol='BTC/USDT',
            amount=Decimal('0.1'),
            avg_price=Decimal('48000'),
            current_price=Decimal('50000')
        )
        
        # 获取投资组合
        portfolio = tracker.get_portfolio()
        print(f"投资组合总价值: {portfolio.total_value}")
        print(f"未实现盈亏: {portfolio.unrealized_pnl}")
        
        # 获取风险指标
        risk = tracker.get_risk_metrics()
        print(f"风险指标: {risk}")
        
        await tracker.stop()
        await sim.disconnect()
    
    asyncio.run(test())
