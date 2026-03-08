"""
CryptoQuant 回测系统

提供完整的量化回测框架，包括:
- 事件驱动回测引擎
- 交易成本模型
- 绩效分析
- 交易分析
- 可视化
- 过拟合检测
- 报告生成
"""

__version__ = "1.0.0"
__author__ = "CryptoQuant Team"

# 回测引擎
from .engine.backtest_engine import (
    BacktestEngine,
    Order,
    Trade,
    Position,
    OrderType,
    OrderSide,
    PositionSide,
    TradeType,
    MarketEvent,
    SignalEvent,
    EventQueue,
    CostModel,
    Portfolio,
    ExecutionEngine,
    create_simple_strategy
)

# 交易成本
from .costs.transaction_costs import (
    TransactionCostModel,
    FeeStructure,
    SlippageParams,
    ImpactParams,
    FeeType,
    SlippageModel,
    ImpactModel,
    ExchangeCostProfile,
    estimate_realistic_costs
)

# 绩效分析
from .analytics.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    calculate_returns_from_equity,
    calculate_equity_from_returns,
    compare_strategies
)

# 交易分析
from .analytics.trade_analyzer import (
    TradeAnalyzer,
    TradeMetrics,
    Trade,
    generate_sample_trades
)

# 报告生成
from .analytics.report_generator import (
    ReportGenerator
)

# 可视化
from .visualization.visualizer import (
    BacktestVisualizer
)

# 过拟合检测
from .overfitting.overfitting_tests import (
    OverfittingDetector,
    OutOfSampleTest,
    ParameterSensitivityTest,
    MonteCarloSimulation,
    CSCVTest,
    OverfittingTestResult,
    generate_sample_strategy_returns
)

__all__ = [
    # 回测引擎
    'BacktestEngine',
    'Order',
    'Trade',
    'Position',
    'OrderType',
    'OrderSide',
    'PositionSide',
    'TradeType',
    'MarketEvent',
    'SignalEvent',
    'EventQueue',
    'CostModel',
    'Portfolio',
    'ExecutionEngine',
    'create_simple_strategy',
    
    # 交易成本
    'TransactionCostModel',
    'FeeStructure',
    'SlippageParams',
    'ImpactParams',
    'FeeType',
    'SlippageModel',
    'ImpactModel',
    'ExchangeCostProfile',
    'estimate_realistic_costs',
    
    # 绩效分析
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'calculate_returns_from_equity',
    'calculate_equity_from_returns',
    'compare_strategies',
    
    # 交易分析
    'TradeAnalyzer',
    'TradeMetrics',
    'Trade',
    'generate_sample_trades',
    
    # 报告生成
    'ReportGenerator',
    
    # 可视化
    'BacktestVisualizer',
    
    # 过拟合检测
    'OverfittingDetector',
    'OutOfSampleTest',
    'ParameterSensitivityTest',
    'MonteCarloSimulation',
    'CSCVTest',
    'OverfittingTestResult',
    'generate_sample_strategy_returns'
]
