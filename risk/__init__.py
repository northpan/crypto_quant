"""
量化交易风控模块

提供完整的风控体系，包括：
- 仓位管理（凯利公式、固定比例、波动率目标等）
- 止损止盈（固定、追踪、ATR自适应、时间止损等）
- 回撤控制（最大回撤、日内回撤、动态降仓）
- 组合风险（多币种相关性、行业分散、杠杆控制）
- 风险指标（VaR、CVaR、夏普比率、Calmar比率等）
"""

from .risk_manager import (
    RiskManager,
    MultiAccountRiskManager,
    RiskCheckResult,
    TradeApproval,
    RiskLevel,
    create_risk_manager,
    quick_risk_check
)

from .metrics.risk_metrics import (
    RiskMetricsCalculator,
    RiskMetricsResult,
    VaRMethod,
    calculate_rolling_metrics,
    quick_metrics
)

from .position.position_sizer import (
    PositionSizer,
    DynamicPositionSizer,
    PositionSize,
    PositionSizingMethod,
    quick_position_size
)

from .stop_loss.stop_manager import (
    StopManager,
    AdvancedStopManager,
    StopLevel,
    TakeProfitLevel,
    PositionStops,
    StopType,
    TakeProfitType,
    quick_stop_loss,
    quick_take_profit
)

from .drawdown.drawdown_controller import (
    DrawdownController,
    DynamicDrawdownController,
    MultiTierDrawdownController,
    DrawdownAction,
    DrawdownLevel,
    DrawdownState,
    calculate_drawdown,
    calculate_max_drawdown_info
)

from .portfolio.portfolio_risk import (
    PortfolioRiskManager,
    CrossExchangePortfolioRiskManager,
    Position,
    MarginInfo,
    PortfolioRiskMetrics,
    ContractType,
    calculate_portfolio_weights,
    check_correlation_clustering
)

__version__ = "1.0.0"
__author__ = "CryptoQuant"

__all__ = [
    # 主类
    'RiskManager',
    'MultiAccountRiskManager',
    
    # 风险指标
    'RiskMetricsCalculator',
    'RiskMetricsResult',
    'VaRMethod',
    
    # 仓位管理
    'PositionSizer',
    'DynamicPositionSizer',
    'PositionSize',
    'PositionSizingMethod',
    
    # 止损止盈
    'StopManager',
    'AdvancedStopManager',
    'StopLevel',
    'TakeProfitLevel',
    'PositionStops',
    'StopType',
    'TakeProfitType',
    
    # 回撤控制
    'DrawdownController',
    'DynamicDrawdownController',
    'MultiTierDrawdownController',
    'DrawdownAction',
    'DrawdownLevel',
    'DrawdownState',
    
    # 组合风险
    'PortfolioRiskManager',
    'CrossExchangePortfolioRiskManager',
    'Position',
    'MarginInfo',
    'PortfolioRiskMetrics',
    'ContractType',
    
    # 便捷函数
    'create_risk_manager',
    'quick_risk_check',
    'quick_metrics',
    'quick_position_size',
    'quick_stop_loss',
    'quick_take_profit',
    'calculate_drawdown',
    'calculate_max_drawdown_info',
    'calculate_rolling_metrics',
    'calculate_portfolio_weights',
    'check_correlation_clustering',
]
