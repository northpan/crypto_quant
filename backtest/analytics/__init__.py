"""
分析模块
"""

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    calculate_returns_from_equity,
    calculate_equity_from_returns,
    compare_strategies
)

from .trade_analyzer import (
    TradeAnalyzer,
    TradeMetrics,
    Trade,
    generate_sample_trades
)

from .report_generator import (
    ReportGenerator
)

__all__ = [
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'calculate_returns_from_equity',
    'calculate_equity_from_returns',
    'compare_strategies',
    'TradeAnalyzer',
    'TradeMetrics',
    'Trade',
    'generate_sample_trades',
    'ReportGenerator'
]
