"""
策略基类
定义所有策略的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    predicted_return: float
    current_price: float
    timestamp: pd.Timestamp
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: pd.Timestamp


class BaseStrategy(ABC):
    """
    策略基类
    
    所有交易策略都需要继承此类并实现抽象方法
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            config: 策略配置
        """
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[Signal] = []
        
    @abstractmethod
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """
        初始化策略
        
        Args:
            data: 历史数据字典 {symbol: DataFrame}
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            data: 最新数据字典 {symbol: DataFrame}
            
        Returns:
            交易信号列表
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, 
                                account_value: float) -> float:
        """
        计算仓位大小
        
        Args:
            signal: 交易信号
            account_value: 账户总价值
            
        Returns:
            仓位大小（正数表示做多，负数表示做空）
        """
        pass
    
    def update_positions(self, positions: Dict[str, Position]):
        """
        更新持仓信息
        
        Args:
            positions: 当前持仓字典
        """
        self.positions = positions
    
    def on_trade(self, trade: Dict[str, Any]):
        """
        交易回调
        
        Args:
            trade: 交易信息
        """
        pass
    
    def on_bar(self, bar: Dict[str, pd.DataFrame]):
        """
        新K线回调
        
        Args:
            bar: 最新K线数据
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取策略绩效指标
        
        Returns:
            绩效指标字典
        """
        return {
            'total_signals': len(self.signals_history),
            'buy_signals': len([s for s in self.signals_history 
                               if s.signal_type == SignalType.BUY]),
            'sell_signals': len([s for s in self.signals_history 
                                if s.signal_type == SignalType.SELL]),
        }
    
    def reset(self):
        """重置策略状态"""
        self.positions = {}
        self.signals_history = []
        self.is_initialized = False
