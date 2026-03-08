"""
回撤控制模块

提供最大回撤限制、日内回撤监控、动态降仓机制等功能
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time


class DrawdownAction(Enum):
    """回撤触发的行动"""
    NONE = "none"  # 无行动
    WARNING = "warning"  # 警告
    REDUCE_POSITION = "reduce_position"  # 降仓
    HALT_TRADING = "halt_trading"  # 暂停交易
    CLOSE_ALL = "close_all"  # 平仓所有


@dataclass
class DrawdownLevel:
    """回撤水平设置"""
    threshold: float  # 回撤阈值
    action: DrawdownAction  # 触发行动
    position_reduction: float = 0.0  # 降仓比例
    cooldown_minutes: int = 0  # 冷却时间（分钟）
    message: str = ""  # 提示信息


@dataclass
class DrawdownState:
    """回撤状态"""
    current_drawdown: float = 0.0  # 当前回撤
    max_drawdown: float = 0.0  # 历史最大回撤
    peak_equity: float = 0.0  # 权益峰值
    drawdown_start: Optional[datetime] = None  # 回撤开始时间
    duration_minutes: int = 0  # 回撤持续时间
    is_in_drawdown: bool = False  # 是否在回撤中
    last_action: Optional[DrawdownAction] = None  # 上次行动
    last_action_time: Optional[datetime] = None  # 上次行动时间


class DrawdownController:
    """回撤控制器"""
    
    def __init__(
        self,
        max_drawdown_limit: float = 0.20,
        daily_drawdown_limit: float = 0.10,
        intraday_drawdown_limit: float = 0.05
    ):
        """
        初始化回撤控制器
        
        Args:
            max_drawdown_limit: 最大回撤限制
            daily_drawdown_limit: 日回撤限制
            intraday_drawdown_limit: 日内回撤限制
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_drawdown_limit = daily_drawdown_limit
        self.intraday_drawdown_limit = intraday_drawdown_limit
        
        # 回撤状态
        self.state = DrawdownState()
        
        # 回撤水平设置
        self.drawdown_levels: List[DrawdownLevel] = []
        self._setup_default_levels()
        
        # 历史数据
        self.equity_history: List[Tuple[datetime, float]] = []
        self.daily_equity: Dict[str, float] = {}  # 日初权益
        
        # 回调函数
        self.action_callbacks: Dict[DrawdownAction, List[Callable]] = {
            action: [] for action in DrawdownAction
        }
        
        # 交易控制
        self.trading_halted = False
        self.halt_until: Optional[datetime] = None
        self.position_scale = 1.0  # 仓位缩放系数
    
    def _setup_default_levels(self):
        """设置默认回撤水平"""
        self.drawdown_levels = [
            DrawdownLevel(
                threshold=0.05,
                action=DrawdownAction.WARNING,
                message="回撤达到5%，请注意风险"
            ),
            DrawdownLevel(
                threshold=0.10,
                action=DrawdownAction.REDUCE_POSITION,
                position_reduction=0.3,
                message="回撤达到10%，降低30%仓位"
            ),
            DrawdownLevel(
                threshold=0.15,
                action=DrawdownAction.REDUCE_POSITION,
                position_reduction=0.5,
                cooldown_minutes=30,
                message="回撤达到15%，降低50%仓位，暂停30分钟"
            ),
            DrawdownLevel(
                threshold=0.20,
                action=DrawdownAction.HALT_TRADING,
                cooldown_minutes=120,
                message="回撤达到20%，暂停交易2小时"
            ),
            DrawdownLevel(
                threshold=0.30,
                action=DrawdownAction.CLOSE_ALL,
                message="回撤达到30%，全部平仓"
            )
        ]
    
    def add_drawdown_level(self, level: DrawdownLevel):
        """添加回撤水平"""
        self.drawdown_levels.append(level)
        self.drawdown_levels.sort(key=lambda x: x.threshold)
    
    def register_callback(self, action: DrawdownAction, callback: Callable):
        """注册回撤行动回调"""
        self.action_callbacks[action].append(callback)
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """
        更新权益
        
        Args:
            equity: 当前权益
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 记录权益历史
        self.equity_history.append((timestamp, equity))
        
        # 更新日初权益
        date_key = timestamp.strftime('%Y-%m-%d')
        if date_key not in self.daily_equity:
            self.daily_equity[date_key] = equity
        
        # 更新峰值
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
            self.state.is_in_drawdown = False
            self.state.drawdown_start = None
        
        # 计算回撤
        if self.state.peak_equity > 0:
            self.state.current_drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
        
        # 更新最大回撤
        if self.state.current_drawdown > self.state.max_drawdown:
            self.state.max_drawdown = self.state.current_drawdown
        
        # 更新回撤状态
        if self.state.current_drawdown > 0:
            if not self.state.is_in_drawdown:
                self.state.is_in_drawdown = True
                self.state.drawdown_start = timestamp
            
            # 计算回撤持续时间
            if self.state.drawdown_start:
                duration = (timestamp - self.state.drawdown_start).total_seconds() / 60
                self.state.duration_minutes = int(duration)
        
        # 检查回撤水平
        self._check_drawdown_levels(timestamp)
    
    def _check_drawdown_levels(self, timestamp: datetime):
        """检查回撤水平"""
        # 检查冷却时间
        if self.halt_until and timestamp < self.halt_until:
            return
        
        if self.halt_until and timestamp >= self.halt_until:
            self.trading_halted = False
            self.halt_until = None
        
        # 检查各回撤水平
        for level in reversed(self.drawdown_levels):  # 从高到低检查
            if self.state.current_drawdown >= level.threshold:
                self._execute_action(level, timestamp)
                break
    
    def _execute_action(self, level: DrawdownLevel, timestamp: datetime):
        """执行回撤行动"""
        # 检查是否重复触发
        if (self.state.last_action == level.action and 
            self.state.last_action_time and
            (timestamp - self.state.last_action_time).total_seconds() < 300):  # 5分钟内不重复
            return
        
        self.state.last_action = level.action
        self.state.last_action_time = timestamp
        
        # 执行回调
        for callback in self.action_callbacks[level.action]:
            try:
                callback(level, self.state)
            except Exception as e:
                print(f"Callback error: {e}")
        
        # 执行行动
        if level.action == DrawdownAction.REDUCE_POSITION:
            self.position_scale = 1.0 - level.position_reduction
        
        elif level.action == DrawdownAction.HALT_TRADING:
            self.trading_halted = True
            if level.cooldown_minutes > 0:
                self.halt_until = timestamp + pd.Timedelta(minutes=level.cooldown_minutes)
            self.position_scale = 0.5
        
        elif level.action == DrawdownAction.CLOSE_ALL:
            self.trading_halted = True
            self.position_scale = 0.0
    
    def get_daily_drawdown(self, timestamp: Optional[datetime] = None) -> float:
        """
        获取当日回撤
        
        Args:
            timestamp: 时间戳
            
        Returns:
            当日回撤
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        date_key = timestamp.strftime('%Y-%m-%d')
        
        if date_key not in self.daily_equity:
            return 0.0
        
        daily_start = self.daily_equity[date_key]
        
        if len(self.equity_history) == 0:
            return 0.0
        
        current_equity = self.equity_history[-1][1]
        
        if daily_start > 0:
            return (daily_start - current_equity) / daily_start
        
        return 0.0
    
    def get_intraday_drawdown(self, minutes: int = 60) -> float:
        """
        获取日内回撤
        
        Args:
            minutes: 时间窗口（分钟）
            
        Returns:
            日内回撤
        """
        if len(self.equity_history) == 0:
            return 0.0
        
        now = datetime.now()
        cutoff = now - pd.Timedelta(minutes=minutes)
        
        # 获取窗口内的权益
        recent_equity = [e for t, e in self.equity_history if t >= cutoff]
        
        if len(recent_equity) == 0:
            return 0.0
        
        peak = max(recent_equity)
        current = recent_equity[-1]
        
        if peak > 0:
            return (peak - current) / peak
        
        return 0.0
    
    def check_trading_allowed(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        检查是否允许交易
        
        Args:
            timestamp: 时间戳
            
        Returns:
            (是否允许, 原因)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查暂停状态
        if self.trading_halted:
            if self.halt_until and timestamp < self.halt_until:
                remaining = (self.halt_until - timestamp).total_seconds() / 60
                return False, f"交易暂停中，剩余{remaining:.0f}分钟"
            else:
                self.trading_halted = False
                self.halt_until = None
        
        # 检查最大回撤
        if self.state.current_drawdown >= self.max_drawdown_limit:
            return False, f"超过最大回撤限制({self.max_drawdown_limit*100:.1f}%)"
        
        # 检查日回撤
        daily_dd = self.get_daily_drawdown(timestamp)
        if daily_dd >= self.daily_drawdown_limit:
            return False, f"超过日回撤限制({self.daily_drawdown_limit*100:.1f}%)"
        
        # 检查日内回撤
        intraday_dd = self.get_intraday_drawdown()
        if intraday_dd >= self.intraday_drawdown_limit:
            return False, f"超过日内回撤限制({self.intraday_drawdown_limit*100:.1f}%)"
        
        return True, "交易允许"
    
    def get_position_scale(self) -> float:
        """获取当前仓位缩放系数"""
        return self.position_scale
    
    def get_drawdown_report(self) -> Dict:
        """
        获取回撤报告
        
        Returns:
            回撤报告字典
        """
        return {
            'current_drawdown': self.state.current_drawdown,
            'max_drawdown': self.state.max_drawdown,
            'peak_equity': self.state.peak_equity,
            'daily_drawdown': self.get_daily_drawdown(),
            'intraday_drawdown_1h': self.get_intraday_drawdown(60),
            'intraday_drawdown_4h': self.get_intraday_drawdown(240),
            'drawdown_duration_minutes': self.state.duration_minutes,
            'is_in_drawdown': self.state.is_in_drawdown,
            'trading_halted': self.trading_halted,
            'position_scale': self.position_scale,
            'last_action': self.state.last_action.value if self.state.last_action else None
        }
    
    def reset(self):
        """重置回撤控制器"""
        self.state = DrawdownState()
        self.equity_history = []
        self.daily_equity = {}
        self.trading_halted = False
        self.halt_until = None
        self.position_scale = 1.0


class DynamicDrawdownController(DrawdownController):
    """动态回撤控制器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 动态参数
        self.recovery_threshold = 0.5  # 回撤恢复50%后解除限制
        self.volatility_adjustment = True  # 根据波动率调整
        self.volatility_lookback = 20
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """更新权益（带动态调整）"""
        super().update_equity(equity, timestamp)
        
        # 检查回撤恢复
        if self.state.is_in_drawdown:
            self._check_recovery()
    
    def _check_recovery(self):
        """检查回撤恢复"""
        if self.state.max_drawdown == 0:
            return
        
        # 计算回撤恢复比例
        recovery_pct = (self.state.max_drawdown - self.state.current_drawdown) / self.state.max_drawdown
        
        # 如果恢复超过阈值，逐步解除限制
        if recovery_pct >= self.recovery_threshold:
            # 逐步恢复仓位
            self.position_scale = min(1.0, self.position_scale + 0.1)
            
            # 如果完全恢复，重置交易限制
            if self.state.current_drawdown < 0.02:  # 2%以内认为恢复
                self.trading_halted = False
                self.halt_until = None
    
    def adjust_for_volatility(self, returns: pd.Series):
        """
        根据波动率调整回撤限制
        
        Args:
            returns: 收益率序列
        """
        if not self.volatility_adjustment or len(returns) < self.volatility_lookback:
            return
        
        # 计算当前波动率
        current_vol = returns.tail(self.volatility_lookback).std()
        
        # 计算历史平均波动率
        historical_vol = returns.std()
        
        if historical_vol == 0:
            return
        
        # 波动率比率
        vol_ratio = current_vol / historical_vol
        
        # 根据波动率调整回撤限制
        if vol_ratio > 1.5:  # 高波动环境
            self.max_drawdown_limit *= 1.2  # 放宽限制
            self.daily_drawdown_limit *= 1.2
        elif vol_ratio < 0.5:  # 低波动环境
            self.max_drawdown_limit *= 0.9  # 收紧限制
            self.daily_drawdown_limit *= 0.9


class MultiTierDrawdownController:
    """多层回撤控制器"""
    
    def __init__(
        self,
        tiers: List[Tuple[str, float, float, float]] = None
    ):
        """
        初始化多层回撤控制器
        
        Args:
            tiers: [(名称, 最大回撤, 日回撤, 日内回撤), ...]
        """
        if tiers is None:
            tiers = [
                ("保守", 0.10, 0.05, 0.03),
                ("稳健", 0.20, 0.10, 0.05),
                ("积极", 0.30, 0.15, 0.08),
                ("激进", 0.50, 0.25, 0.15)
            ]
        
        self.tiers = tiers
        self.current_tier = 1  # 默认稳健
        self.controllers: Dict[int, DrawdownController] = {}
        
        for i, (name, max_dd, daily_dd, intra_dd) in enumerate(tiers):
            self.controllers[i] = DrawdownController(max_dd, daily_dd, intra_dd)
    
    def set_tier(self, tier: int):
        """设置回撤层级"""
        if 0 <= tier < len(self.tiers):
            self.current_tier = tier
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """更新权益"""
        for controller in self.controllers.values():
            controller.update_equity(equity, timestamp)
    
    def check_trading_allowed(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """检查是否允许交易"""
        return self.controllers[self.current_tier].check_trading_allowed(timestamp)
    
    def get_position_scale(self) -> float:
        """获取仓位缩放系数"""
        return self.controllers[self.current_tier].get_position_scale()
    
    def get_current_tier_name(self) -> str:
        """获取当前层级名称"""
        return self.tiers[self.current_tier][0]


# 便捷函数
def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """
    计算回撤序列
    
    Args:
        equity_series: 权益序列
        
    Returns:
        回撤序列
    """
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return drawdown


def calculate_max_drawdown_info(equity_series: pd.Series) -> Dict:
    """
    计算最大回撤信息
    
    Args:
        equity_series: 权益序列
        
    Returns:
        最大回撤信息
    """
    drawdown = calculate_drawdown(equity_series)
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # 找到回撤开始点
    peak_idx = equity_series.loc[:max_dd_idx].idxmax()
    
    # 计算持续时间
    duration = (max_dd_idx - peak_idx).total_seconds() / 86400  # 天数
    
    return {
        'max_drawdown': max_dd,
        'peak_date': peak_idx,
        'trough_date': max_dd_idx,
        'duration_days': duration,
        'peak_value': equity_series.loc[peak_idx],
        'trough_value': equity_series.loc[max_dd_idx]
    }
