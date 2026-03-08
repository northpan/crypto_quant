"""
事件驱动回测引擎
支持多币种、多策略、分钟级精度的回测
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


class TradeType(Enum):
    """交易类型"""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"


@dataclass
class Order:
    """订单对象"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status: str = "pending"  # pending, filled, partial, cancelled, rejected
    trade_type: TradeType = TradeType.SPOT
    leverage: float = 1.0
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return abs(self.remaining_quantity) < 1e-10


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fee: float = 0.0
    slippage: float = 0.0
    trade_type: TradeType = TradeType.SPOT
    pnl: Optional[float] = None  # 已实现盈亏


@dataclass
class Position:
    """持仓对象"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_type: TradeType = TradeType.SPOT
    leverage: float = 1.0
    margin: float = 0.0
    
    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class MarketEvent:
    """市场事件"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def vwap(self) -> float:
        """成交量加权平均价"""
        return (self.high + self.low + self.close) / 3


@dataclass
class SignalEvent:
    """信号事件"""
    timestamp: datetime
    symbol: str
    signal_type: str  # entry, exit, adjust
    side: OrderSide
    strength: float  # 信号强度 0-1
    metadata: Dict = field(default_factory=dict)


class EventQueue:
    """事件队列 - 优先队列实现"""
    
    def __init__(self):
        self._queue = []
        self._counter = 0
    
    def put(self, event: Any, priority: int = 0):
        """添加事件到队列"""
        heapq.heappush(self._queue, (priority, self._counter, event))
        self._counter += 1
    
    def get(self) -> Optional[Any]:
        """获取下一个事件"""
        if self._queue:
            return heapq.heappop(self._queue)[2]
        return None
    
    def peek(self) -> Optional[Any]:
        """查看下一个事件"""
        if self._queue:
            return self._queue[0][2]
        return None
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0


class CostModel:
    """交易成本模型"""
    
    def __init__(
        self,
        maker_fee: float = 0.001,      # 挂单手续费
        taker_fee: float = 0.001,      # 吃单手续费
        slippage_model: str = "fixed",  # fixed, percentage, volatility
        slippage_value: float = 0.0005, # 滑点值
        impact_model: str = "none",     # 冲击成本模型
        funding_rate: float = 0.0       # 资金费率
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.impact_model = impact_model
        self.funding_rate = funding_rate
    
    def calculate_slippage(
        self, 
        price: float, 
        quantity: float, 
        volume: float,
        volatility: float = 0.0
    ) -> float:
        """计算滑点"""
        if self.slippage_model == "fixed":
            return self.slippage_value
        elif self.slippage_model == "percentage":
            return price * self.slippage_value
        elif self.slippage_model == "volatility":
            return price * volatility * self.slippage_value
        elif self.slippage_model == "volume_based":
            volume_ratio = min(quantity / max(volume, 1e-10), 1.0)
            return price * self.slippage_value * volume_ratio
        return 0.0
    
    def calculate_fee(self, notional: float, is_maker: bool = False) -> float:
        """计算手续费"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return notional * fee_rate
    
    def calculate_impact_cost(
        self, 
        price: float, 
        quantity: float, 
        market_depth: float
    ) -> float:
        """计算冲击成本"""
        if self.impact_model == "linear":
            impact = quantity / max(market_depth, 1e-10)
            return price * min(impact * 0.001, 0.01)  # 最大1%
        elif self.impact_model == "square_root":
            impact = np.sqrt(quantity / max(market_depth, 1e-10))
            return price * min(impact * 0.001, 0.01)
        return 0.0


class Portfolio:
    """投资组合管理"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        trade_type: TradeType = TradeType.SPOT
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.trade_type = trade_type
        
        # 持仓
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # 交易记录
        self.trades: List[Trade] = []
        
        # 权益曲线
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # 当前时间
        self.current_time: Optional[datetime] = None
    
    def update_time(self, timestamp: datetime):
        """更新时间"""
        self.current_time = timestamp
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        fee: float,
        trade_type: TradeType = TradeType.SPOT,
        leverage: float = 1.0
    ) -> Tuple[Optional[Position], float]:
        """更新持仓，返回(新持仓, 已实现盈亏)"""
        realized_pnl = 0.0
        position = self.positions.get(symbol)
        
        if position is None:
            # 新开仓
            if side == OrderSide.BUY:
                pos_side = PositionSide.LONG
            else:
                pos_side = PositionSide.SHORT
            
            margin = (quantity * price) / leverage if trade_type == TradeType.FUTURES else 0.0
            
            position = Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                entry_price=price,
                timestamp=self.current_time,
                trade_type=trade_type,
                leverage=leverage,
                margin=margin
            )
            self.positions[symbol] = position
            
            if trade_type == TradeType.SPOT:
                if side == OrderSide.BUY:
                    self.cash -= (quantity * price + fee)
                else:
                    self.cash += (quantity * price - fee)
            else:
                self.cash -= (margin + fee)
        else:
            # 现有持仓
            if (side == OrderSide.BUY and position.side == PositionSide.LONG) or \
               (side == OrderSide.SELL and position.side == PositionSide.SHORT):
                # 加仓
                total_quantity = position.quantity + quantity
                position.entry_price = (
                    position.entry_price * position.quantity + price * quantity
                ) / total_quantity
                position.quantity = total_quantity
                
                if trade_type == TradeType.SPOT:
                    if side == OrderSide.BUY:
                        self.cash -= (quantity * price + fee)
                    else:
                        self.cash += (quantity * price - fee)
                else:
                    additional_margin = (quantity * price) / leverage
                    position.margin += additional_margin
                    self.cash -= (additional_margin + fee)
            else:
                # 减仓或平仓
                if quantity >= position.quantity:
                    # 完全平仓
                    close_quantity = position.quantity
                    
                    # 计算已实现盈亏
                    if position.side == PositionSide.LONG:
                        realized_pnl = (price - position.entry_price) * close_quantity - fee
                    else:
                        realized_pnl = (position.entry_price - price) * close_quantity - fee
                    
                    position.realized_pnl = realized_pnl
                    self.closed_positions.append(position)
                    del self.positions[symbol]
                    
                    if trade_type == TradeType.SPOT:
                        if side == OrderSide.SELL:
                            self.cash += (close_quantity * price - fee)
                        else:
                            self.cash -= (close_quantity * price + fee)
                    else:
                        self.cash += (position.margin + realized_pnl)
                else:
                    # 部分平仓
                    close_quantity = quantity
                    
                    if position.side == PositionSide.LONG:
                        realized_pnl = (price - position.entry_price) * close_quantity - fee
                    else:
                        realized_pnl = (position.entry_price - price) * close_quantity - fee
                    
                    position.quantity -= close_quantity
                    position.realized_pnl += realized_pnl
                    
                    if trade_type == TradeType.SPOT:
                        if side == OrderSide.SELL:
                            self.cash += (close_quantity * price - fee)
                        else:
                            self.cash -= (close_quantity * price + fee)
                    else:
                        released_margin = (close_quantity * price) / leverage
                        position.margin -= released_margin
                        self.cash += (released_margin + realized_pnl)
        
        return position, realized_pnl
    
    def update_unrealized_pnl(self, market_data: Dict[str, MarketEvent]):
        """更新未实现盈亏"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].close
                
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (
                        current_price - position.entry_price
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        position.entry_price - current_price
                    ) * position.quantity
    
    @property
    def total_value(self) -> float:
        """总资产价值"""
        position_value = sum(
            pos.quantity * pos.entry_price + pos.unrealized_pnl
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    @property
    def total_pnl(self) -> float:
        """总盈亏"""
        return self.total_value - self.initial_capital
    
    @property
    def total_return(self) -> float:
        """总收益率"""
        return self.total_pnl / self.initial_capital
    
    def record_equity(self):
        """记录权益"""
        if self.current_time:
            self.equity_curve.append((self.current_time, self.total_value))


class ExecutionEngine:
    """执行引擎 - 处理订单执行"""
    
    def __init__(
        self,
        cost_model: CostModel,
        fill_model: str = "vwap"  # vwap, twap, close
    ):
        self.cost_model = cost_model
        self.fill_model = fill_model
        self.order_counter = 0
        self.trade_counter = 0
    
    def generate_order_id(self) -> str:
        """生成订单ID"""
        self.order_counter += 1
        return f"ORD_{self.order_counter:08d}"
    
    def generate_trade_id(self) -> str:
        """生成成交ID"""
        self.trade_counter += 1
        return f"TRD_{self.trade_counter:08d}"
    
    def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trade_type: TradeType = TradeType.SPOT,
        leverage: float = 1.0
    ) -> Order:
        """创建订单"""
        return Order(
            order_id=self.generate_order_id(),
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trade_type=trade_type,
            leverage=leverage
        )
    
    def execute_order(
        self,
        order: Order,
        market_event: MarketEvent
    ) -> Optional[Trade]:
        """执行订单"""
        if order.status in ["filled", "cancelled", "rejected"]:
            return None
        
        # 确定成交价格
        if self.fill_model == "vwap":
            fill_price = market_event.vwap
        elif self.fill_model == "close":
            fill_price = market_event.close
        elif self.fill_model == "twap":
            fill_price = (market_event.open + market_event.close) / 2
        else:
            fill_price = market_event.close
        
        # 市价单处理
        if order.order_type == OrderType.MARKET:
            # 添加滑点
            slippage = self.cost_model.calculate_slippage(
                fill_price, order.quantity, market_event.volume
            )
            
            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage
            
            # 确保价格在合理范围内
            fill_price = max(market_event.low, min(market_event.high, fill_price))
            
            # 计算手续费
            notional = order.quantity * fill_price
            fee = self.cost_model.calculate_fee(notional, is_maker=False)
            
            # 创建成交记录
            trade = Trade(
                trade_id=self.generate_trade_id(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=market_event.timestamp,
                fee=fee,
                slippage=slippage,
                trade_type=order.trade_type
            )
            
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.status = "filled"
            
            return trade
        
        # 限价单处理
        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                order.status = "rejected"
                return None
            
            # 检查是否能成交
            if order.side == OrderSide.BUY and order.price >= market_event.low:
                # 买单能成交
                fill_price = min(order.price, market_event.close)
            elif order.side == OrderSide.SELL and order.price <= market_event.high:
                # 卖单能成交
                fill_price = max(order.price, market_event.close)
            else:
                # 不能成交
                return None
            
            # 计算手续费（限价单通常作为maker）
            notional = order.quantity * fill_price
            fee = self.cost_model.calculate_fee(notional, is_maker=True)
            
            trade = Trade(
                trade_id=self.generate_trade_id(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=market_event.timestamp,
                fee=fee,
                slippage=0.0,
                trade_type=order.trade_type
            )
            
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.status = "filled"
            
            return trade
        
        # 止损单处理
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                order.status = "rejected"
                return None
            
            # 检查触发条件
            triggered = False
            if order.side == OrderSide.BUY and market_event.high >= order.stop_price:
                triggered = True
                fill_price = max(order.stop_price, market_event.close)
            elif order.side == OrderSide.SELL and market_event.low <= order.stop_price:
                triggered = True
                fill_price = min(order.stop_price, market_event.close)
            
            if triggered:
                slippage = self.cost_model.calculate_slippage(
                    fill_price, order.quantity, market_event.volume
                )
                
                if order.side == OrderSide.BUY:
                    fill_price += slippage
                else:
                    fill_price -= slippage
                
                notional = order.quantity * fill_price
                fee = self.cost_model.calculate_fee(notional, is_maker=False)
                
                trade = Trade(
                    trade_id=self.generate_trade_id(),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=fill_price,
                    timestamp=market_event.timestamp,
                    fee=fee,
                    slippage=slippage,
                    trade_type=order.trade_type
                )
                
                order.filled_quantity = order.quantity
                order.filled_price = fill_price
                order.status = "filled"
                
                return trade
        
        return None


class BacktestEngine:
    """
    事件驱动回测引擎
    
    特性:
    - 分钟级精度
    - 多币种支持
    - 现货和合约模拟
    - 完整的交易成本模型
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        trade_type: TradeType = TradeType.SPOT,
        cost_model: Optional[CostModel] = None,
        fill_model: str = "vwap"
    ):
        self.initial_capital = initial_capital
        self.trade_type = trade_type
        self.cost_model = cost_model or CostModel()
        self.fill_model = fill_model
        
        # 组件
        self.portfolio = Portfolio(initial_capital, trade_type)
        self.execution_engine = ExecutionEngine(self.cost_model, fill_model)
        self.event_queue = EventQueue()
        
        # 数据
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_data: Dict[str, MarketEvent] = {}
        
        # 策略
        self.strategy: Optional[Callable] = None
        
        # 状态
        self.is_running = False
        self.current_time: Optional[datetime] = None
        self.pending_orders: List[Order] = []
        
        # 回测结果
        self.results: Dict = {}
        
        logger.info(f"回测引擎初始化完成 | 初始资金: {initial_capital} | 交易类型: {trade_type.value}")
    
    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume"
    ):
        """加载市场数据"""
        # 标准化列名
        data = data.copy()
        data.rename(columns={
            timestamp_col: "timestamp",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            volume_col: "volume"
        }, inplace=True)
        
        # 确保时间戳是 datetime 类型并统一为 UTC，避免混合 naive/aware 导致排序报错
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        if data["timestamp"].dt.tz is None:
            data["timestamp"] = data["timestamp"].dt.tz_localize("UTC", ambiguous="infer")
        else:
            data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")
        
        # 排序
        data = data.sort_values("timestamp").reset_index(drop=True)
        
        self.market_data[symbol] = data
        logger.info(f"加载数据: {symbol} | {len(data)} 条记录 | 时间范围: {data['timestamp'].min()} ~ {data['timestamp'].max()}")
    
    def set_strategy(self, strategy: Callable):
        """设置策略"""
        self.strategy = strategy
        logger.info("策略已设置")
    
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        self.pending_orders.append(order)
        logger.debug(f"提交订单: {order.order_id} | {order.symbol} | {order.side.value} | {order.quantity}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = "cancelled"
                self.pending_orders.remove(order)
                return True
        return False
    
    def _process_orders(self):
        """处理待执行订单"""
        executed_orders = []
        
        for order in self.pending_orders:
            if order.symbol in self.current_data:
                market_event = self.current_data[order.symbol]
                trade = self.execution_engine.execute_order(order, market_event)
                
                if trade:
                    # 更新持仓
                    position, realized_pnl = self.portfolio.update_position(
                        symbol=trade.symbol,
                        side=trade.side,
                        quantity=trade.quantity,
                        price=trade.price,
                        fee=trade.fee,
                        trade_type=trade.trade_type
                    )
                    
                    trade.pnl = realized_pnl
                    self.portfolio.trades.append(trade)
                    executed_orders.append(order)
                    
                    logger.debug(
                        f"成交: {trade.trade_id} | {trade.symbol} | "
                        f"{trade.side.value} | {trade.quantity:.6f} @ {trade.price:.4f} | "
                        f"PnL: {realized_pnl:.2f}"
                    )
        
        # 移除已执行订单
        for order in executed_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
    
    def _generate_signals(self):
        """生成交易信号"""
        if self.strategy is None:
            return
        
        # 调用策略
        try:
            self.strategy(
                engine=self,
                portfolio=self.portfolio,
                market_data=self.current_data,
                timestamp=self.current_time
            )
        except Exception as e:
            logger.error(f"策略执行错误: {e}")
    
    def run(self) -> Dict:
        """运行回测"""
        if not self.market_data:
            raise ValueError("请先加载市场数据")
        
        if self.strategy is None:
            raise ValueError("请先设置策略")
        
        self.is_running = True
        logger.info("=" * 50)
        logger.info("开始回测")
        logger.info("=" * 50)
        
        # 获取所有时间戳
        all_timestamps = set()
        for data in self.market_data.values():
            all_timestamps.update(data["timestamp"].tolist())
        
        sorted_timestamps = sorted(all_timestamps)
        total_bars = len(sorted_timestamps)
        
        logger.info(f"总时间步数: {total_bars}")
        
        # 主回测循环
        for i, timestamp in enumerate(sorted_timestamps):
            self.current_time = timestamp
            self.portfolio.update_time(timestamp)
            
            # 更新当前市场数据
            self.current_data = {}
            for symbol, data in self.market_data.items():
                mask = data["timestamp"] == timestamp
                if mask.any():
                    row = data[mask].iloc[0]
                    self.current_data[symbol] = MarketEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"]
                    )
            
            # 更新未实现盈亏
            self.portfolio.update_unrealized_pnl(self.current_data)
            
            # 处理订单
            self._process_orders()
            
            # 生成信号
            self._generate_signals()
            
            # 记录权益
            self.portfolio.record_equity()
            
            # 进度报告
            if (i + 1) % max(1, total_bars // 10) == 0:
                progress = (i + 1) / total_bars * 100
                logger.info(f"回测进度: {progress:.1f}% | 当前权益: {self.portfolio.total_value:.2f}")
        
        self.is_running = False
        logger.info("=" * 50)
        logger.info("回测完成")
        logger.info("=" * 50)
        
        # 生成结果
        self._generate_results()
        
        return self.results

    def _build_round_trips(self) -> pd.DataFrame:
        """从逐笔成交构造回合（开仓-平仓配对，FIFO）。"""
        open_lots: Dict[str, List[Tuple]] = defaultdict(list)
        round_trips = []

        for t in self.portfolio.trades:
            key = t.symbol
            if t.side == OrderSide.BUY:
                open_lots[key].append((t.timestamp, t.price, t.quantity, t.fee or 0.0))
                continue
            if t.side != OrderSide.SELL:
                continue
            remaining = t.quantity
            exit_fee_total = t.fee or 0.0
            exit_price = t.price
            exit_time = t.timestamp
            closed_qty_sum = 0.0
            entry_fees_used = []
            entry_times = []
            entry_prices = []
            quantities_used = []

            while remaining > 1e-12 and open_lots[key]:
                et, ep, q, ef = open_lots[key].pop(0)
                if q <= remaining:
                    closed_qty_sum += q
                    remaining -= q
                    entry_fees_used.append(ef)
                    entry_times.append(et)
                    entry_prices.append(ep)
                    quantities_used.append(q)
                else:
                    part = remaining
                    closed_qty_sum += part
                    remaining = 0
                    entry_fees_used.append(ef * (part / q))
                    entry_times.append(et)
                    entry_prices.append(ep)
                    quantities_used.append(part)
                    open_lots[key].insert(0, (et, ep, q - part, ef * (1 - part / q)))

            if not closed_qty_sum:
                continue
            total_entry_fee = sum(entry_fees_used)
            exit_fee_portion = exit_fee_total * (closed_qty_sum / t.quantity) if t.quantity else 0
            vwap_entry = sum(p * q for p, q in zip(entry_prices, quantities_used)) / closed_qty_sum
            entry_time_first = entry_times[0]
            pnl = (exit_price - vwap_entry) * closed_qty_sum - total_entry_fee - exit_fee_portion
            round_trips.append({
                "trade_id": f"RT_{t.symbol}_{len(round_trips)}",
                "symbol": t.symbol,
                "entry_time": entry_time_first,
                "exit_time": exit_time,
                "entry_price": vwap_entry,
                "exit_price": exit_price,
                "quantity": closed_qty_sum,
                "side": "long",
                "pnl": pnl,
                "return_pct": (exit_price / vwap_entry - 1) * 100 if vwap_entry else 0,
                "fees": total_entry_fee + exit_fee_portion,
            })

        if not round_trips:
            return pd.DataFrame(
                columns=["trade_id", "symbol", "entry_time", "exit_time", "entry_price", "exit_price", "quantity", "side", "pnl", "return_pct", "fees"]
            )
        return pd.DataFrame(round_trips)
    
    def _generate_results(self) -> Dict:
        """生成回测结果"""
        equity_df = pd.DataFrame(
            self.portfolio.equity_curve,
            columns=["timestamp", "equity"]
        )
        
        trades_df = pd.DataFrame([
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "timestamp": t.timestamp,
                "fee": t.fee,
                "pnl": t.pnl
            }
            for t in self.portfolio.trades
        ])

        round_trips_df = self._build_round_trips()

        self.results = {
            "equity_curve": equity_df,
            "trades": trades_df,
            "round_trips": round_trips_df,
            "positions": self.portfolio.positions,
            "closed_positions": self.portfolio.closed_positions,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio.total_value,
            "total_return": self.portfolio.total_return,
            "total_pnl": self.portfolio.total_pnl,
            "num_trades": len(self.portfolio.trades)
        }

        return self.results
    
    def get_summary(self) -> str:
        """获取回测摘要"""
        if not self.results:
            return "请先运行回测"
        
        summary = f"""
{'='*50}
回测结果摘要
{'='*50}
初始资金: {self.initial_capital:,.2f}
最终权益: {self.portfolio.total_value:,.2f}
总盈亏: {self.portfolio.total_pnl:,.2f}
总收益率: {self.portfolio.total_return*100:.2f}%
交易次数: {len(self.portfolio.trades)}
持仓数量: {len(self.portfolio.positions)}
{'='*50}
"""
        return summary


# 辅助函数
def create_simple_strategy(
    entry_condition: Callable,
    exit_condition: Callable,
    position_size: float = 0.1
) -> Callable:
    """
    创建简单策略
    
    Args:
        entry_condition: 入场条件函数 (market_data) -> bool
        exit_condition: 出场条件函数 (market_data, position) -> bool
        position_size: 仓位大小 (占总资金比例)
    """
    def strategy(engine, portfolio, market_data, timestamp):
        for symbol, data in market_data.items():
            position = portfolio.get_position(symbol)
            
            # 检查出场条件
            if position:
                if exit_condition(data, position):
                    # 平仓
                    order = engine.execution_engine.create_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                        quantity=position.quantity,
                        trade_type=engine.trade_type
                    )
                    engine.submit_order(order)
                    continue
            
            # 检查入场条件
            if entry_condition(data):
                # 开新仓
                if not position:
                    current_price = data.close
                    cash_available = portfolio.cash * position_size
                    quantity = cash_available / current_price
                    
                    if quantity > 0:
                        order = engine.execution_engine.create_order(
                            symbol=symbol,
                            order_type=OrderType.MARKET,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            trade_type=engine.trade_type
                        )
                        engine.submit_order(order)
    
    return strategy


if __name__ == "__main__":
    # 示例用法
    print("回测引擎模块")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    
    # 生成随机价格数据
    price = 50000
    prices = []
    for _ in range(1000):
        price *= (1 + np.random.normal(0, 0.001))
        prices.append(price)
    
    data = pd.DataFrame({
        "timestamp": dates,
        "open": [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
        "high": [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        "close": prices,
        "volume": np.random.uniform(1, 100, 1000)
    })
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000.0,
        trade_type=TradeType.SPOT,
        cost_model=CostModel(
            maker_fee=0.001,
            taker_fee=0.001,
            slippage_model="fixed",
            slippage_value=0.0005
        )
    )
    
    # 加载数据
    engine.load_data("BTCUSDT", data)
    
    # 创建简单策略 (均线交叉)
    def entry_condition(data):
        return data.close > data.open * 1.001
    
    def exit_condition(data, position):
        return data.close < data.open * 0.999
    
    strategy = create_simple_strategy(entry_condition, exit_condition, position_size=0.2)
    engine.set_strategy(strategy)
    
    # 运行回测
    results = engine.run()
    
    # 打印结果
    print(engine.get_summary())
