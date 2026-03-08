"""
Core module for crypto quantitative trading system.

This module contains base classes and interfaces for all components.
"""

from .base import (
    BaseDataSource,
    BaseDataProcessor,
    BaseFactor,
    BaseModel,
    BaseEnsemble,
    BaseRiskManager,
    BaseExecutionEngine,
    BaseBacktestEngine,
    BaseStrategy,
    DataPacket,
    Signal,
    Order,
    Position,
    Portfolio,
    Trade,
    MarketData,
)

__all__ = [
    "BaseDataSource",
    "BaseDataProcessor",
    "BaseFactor",
    "BaseModel",
    "BaseEnsemble",
    "BaseRiskManager",
    "BaseExecutionEngine",
    "BaseBacktestEngine",
    "BaseStrategy",
    "DataPacket",
    "Signal",
    "Order",
    "Position",
    "Portfolio",
    "Trade",
    "MarketData",
]
