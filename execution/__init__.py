"""
数字货币交易执行模块

提供完整的交易执行功能，包括:
- 交易所接口 (基于CCXT)
- 订单管理
- 持仓跟踪
- 执行策略 (TWAP, VWAP, 冰山订单, 智能路由)
- 滑点控制
- 交易记录

支持模拟和实盘两种模式
"""

# 交易所客户端
from .exchange.exchange_client import (
    BaseExchange,
    CCXTExchange,
    SimulatedExchange,
    ExchangeManager,
    ExchangeConfig,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
    PositionSide,
    Balance,
    create_binance_client,
    create_binance_futures_client,
    create_okx_client,
    create_simulated_exchange
)

# 订单管理
from .order.order_manager import (
    OrderManager,
    OrderRequest,
    ConditionalOrder,
    OrderUpdate,
    OrderCallback,
    TimeInForce,
    TriggerCondition,
    create_order_request
)

# 持仓跟踪
from .position.position_tracker import (
    PositionTracker,
    PositionDetail,
    PositionUpdate,
    Portfolio,
    PositionType,
    HedgeMode,
    HedgeConfig,
    PositionSizer,
    create_position_tracker
)

# 执行引擎
from .strategy.execution_engine import (
    ExecutionEngine,
    ExecutionConfig,
    ExecutionReport,
    ExecutionSlice,
    ExecutionStrategy,
    ExecutionStatus,
    BaseExecutionStrategy,
    TWAPStrategy,
    VWAPStrategy,
    IcebergStrategy,
    SmartRouter,
    create_execution_engine
)

# 滑点模型
from .slippage.slippage_model import (
    SlippageModel,
    MarketImpactModel,
    OrderSplitter,
    SlippageModelType,
    MarketImpactType,
    SlippageEstimate,
    MarketImpact,
    OrderSplitResult,
    estimate_slippage,
    split_large_order
)

# 交易记录
from .recorder.trade_recorder import (
    TradeRecorder,
    TradeRecord,
    DailySummary,
    PerformanceMetrics,
    RecordType,
    create_trade_recorder
)

__version__ = "1.0.0"
__all__ = [
    # 交易所
    'BaseExchange',
    'CCXTExchange',
    'SimulatedExchange',
    'ExchangeManager',
    'ExchangeConfig',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'Position',
    'PositionSide',
    'Balance',
    'create_binance_client',
    'create_binance_futures_client',
    'create_okx_client',
    'create_simulated_exchange',
    
    # 订单管理
    'OrderManager',
    'OrderRequest',
    'ConditionalOrder',
    'OrderUpdate',
    'OrderCallback',
    'TimeInForce',
    'TriggerCondition',
    'create_order_request',
    
    # 持仓跟踪
    'PositionTracker',
    'PositionDetail',
    'PositionUpdate',
    'Portfolio',
    'PositionType',
    'HedgeMode',
    'HedgeConfig',
    'PositionSizer',
    'create_position_tracker',
    
    # 执行引擎
    'ExecutionEngine',
    'ExecutionConfig',
    'ExecutionReport',
    'ExecutionSlice',
    'ExecutionStrategy',
    'ExecutionStatus',
    'BaseExecutionStrategy',
    'TWAPStrategy',
    'VWAPStrategy',
    'IcebergStrategy',
    'SmartRouter',
    'create_execution_engine',
    
    # 滑点模型
    'SlippageModel',
    'MarketImpactModel',
    'OrderSplitter',
    'SlippageModelType',
    'MarketImpactType',
    'SlippageEstimate',
    'MarketImpact',
    'OrderSplitResult',
    'estimate_slippage',
    'split_large_order',
    
    # 交易记录
    'TradeRecorder',
    'TradeRecord',
    'DailySummary',
    'PerformanceMetrics',
    'RecordType',
    'create_trade_recorder'
]
