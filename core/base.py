"""
核心基类模块
定义所有模块的通用基类和接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class TradingMode(Enum):
    """交易模式"""
    BACKTEST = 'backtest'
    PAPER = 'paper'
    LIVE = 'live'


class TradeType(Enum):
    """交易类型"""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARGIN = 'margin'


class OrderSide(Enum):
    """订单方向"""
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    """订单类型"""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    STOP_LIMIT = 'stop_limit'


class OrderStatus(Enum):
    """订单状态"""
    PENDING = 'pending'
    OPEN = 'open'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


@dataclass
class Tick:
    """行情数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame([{
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'trades': self.trades
        }], index=[self.timestamp])


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    remaining_amount: float = 0.0
    average_price: float = 0.0
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """成交记录"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    fee: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """持仓"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float = 0.0
    leverage: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def value(self) -> float:
        """持仓价值"""
        return abs(self.amount) * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * (1 if self.amount > 0 else -1)


@dataclass
class Account:
    """账户信息"""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    unrealized_pnl: float
    realized_pnl: float
    positions: Dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_value(self) -> float:
        """总资产"""
        return self.equity + self.unrealized_pnl


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    sqn: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0


class BaseModule(ABC):
    """模块基类"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self._logger = None
    
    @abstractmethod
    def initialize(self):
        """初始化模块"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置模块状态"""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)


class BaseDataProvider(ABC):
    """数据提供者基类"""
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                   start: datetime, end: datetime) -> pd.DataFrame:
        """获取历史数据"""
        pass
    
    @abstractmethod
    async def get_latest_tick(self, symbol: str) -> Tick:
        """获取最新行情"""
        pass
    
    @abstractmethod
    async def subscribe_ticks(self, symbols: List[str], 
                              callback: Callable[[Tick], None]):
        """订阅实时行情"""
        pass


class BaseExchange(ABC):
    """交易所基类"""
    
    @abstractmethod
    async def connect(self):
        """连接交易所"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, side: OrderSide, 
                          order_type: OrderType, amount: float,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None) -> Order:
        """创建订单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """获取订单信息"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        pass
    
    @abstractmethod
    async def get_account(self) -> Account:
        """获取账户信息"""
        pass


class BaseRiskManager(ABC):
    """风控管理器基类"""
    
    @abstractmethod
    def check_order(self, order: Order, account: Account) -> Dict[str, Any]:
        """检查订单是否通过风控"""
        pass
    
    @abstractmethod
    def check_position(self, position: Position, account: Account) -> Dict[str, Any]:
        """检查持仓风险"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal_strength: float,
                                account: Account) -> float:
        """计算仓位大小"""
        pass


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.positions: Dict[str, Position] = {}
        
    @abstractmethod
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """初始化策略"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Any, account_value: float) -> float:
        """计算仓位大小"""
        pass


# 工具函数
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                           periods_per_year: int = 365 * 24 * 60) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年周期数（分钟级）
        
    Returns:
        夏普比率
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    # 调整无风险利率到分钟级
    rf_per_period = risk_free_rate / periods_per_year
    
    # 计算超额收益
    excess_returns = returns - rf_per_period
    
    # 年化夏普比率
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple:
    """
    计算最大回撤
    
    Args:
        equity_curve: 权益曲线
        
    Returns:
        (最大回撤比例, 最大回撤持续时间)
    """
    # 计算累计最大值
    running_max = equity_curve.expanding().max()
    
    # 计算回撤
    drawdown = (equity_curve - running_max) / running_max
    
    # 最大回撤
    max_drawdown = drawdown.min()
    
    # 最大回撤持续时间
    is_drawdown = drawdown < 0
    max_duration = 0
    current_duration = 0
    
    for is_dd in is_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_drawdown, max_duration


def annualize_return(total_return: float, periods: int, 
                     periods_per_year: int = 365 * 24 * 60) -> float:
    """
    年化收益率
    
    Args:
        total_return: 总收益率
        periods: 总周期数
        periods_per_year: 每年周期数
        
    Returns:
        年化收益率
    """
    years = periods / periods_per_year
    if years == 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1
