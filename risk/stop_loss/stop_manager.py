"""
止损止盈管理模块

提供多种止损止盈策略，包括固定、追踪、ATR自适应、时间止损等
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class StopType(Enum):
    """止损类型"""
    FIXED = "fixed"  # 固定止损
    TRAILING = "trailing"  # 追踪止损
    ATR = "atr"  # ATR自适应止损
    TIME = "time"  # 时间止损
    PERCENTAGE = "percentage"  # 百分比止损
    VOLATILITY = "volatility"  # 波动率止损
    SUPPORT_RESISTANCE = "support_resistance"  # 支撑阻力止损


class TakeProfitType(Enum):
    """止盈类型"""
    FIXED = "fixed"  # 固定止盈
    TRAILING = "trailing"  # 追踪止盈
    RISK_REWARD = "risk_reward"  # 风险回报比止盈
    PERCENTAGE = "percentage"  # 百分比止盈
    PARTIAL = "partial"  # 分批止盈


@dataclass
class StopLevel:
    """止损/止盈水平"""
    price: float  # 触发价格
    stop_type: StopType  # 类型
    size: float = 1.0  # 仓位比例
    created_at: datetime = field(default_factory=datetime.now)
    triggered: bool = False
    trigger_price: Optional[float] = None  # 实际触发价格
    
    def check_trigger(self, current_price: float, is_long: bool = True) -> bool:
        """
        检查是否触发
        
        Args:
            current_price: 当前价格
            is_long: 是否多头
            
        Returns:
            是否触发
        """
        if self.triggered:
            return False
        
        if is_long:
            triggered = current_price <= self.price
        else:
            triggered = current_price >= self.price
        
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
        
        return triggered


@dataclass
class TakeProfitLevel:
    """止盈水平"""
    price: float  # 目标价格
    tp_type: TakeProfitType  # 类型
    size: float = 1.0  # 止盈仓位比例
    triggered: bool = False
    trigger_price: Optional[float] = None
    
    def check_trigger(self, current_price: float, is_long: bool = True) -> bool:
        """检查是否触发"""
        if self.triggered:
            return False
        
        if is_long:
            triggered = current_price >= self.price
        else:
            triggered = current_price <= self.price
        
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
        
        return triggered


@dataclass
class PositionStops:
    """持仓的止损止盈设置"""
    entry_price: float
    is_long: bool = True
    position_size: float = 0.0
    stop_losses: List[StopLevel] = field(default_factory=list)
    take_profits: List[TakeProfitLevel] = field(default_factory=list)
    entry_time: datetime = field(default_factory=datetime.now)
    
    def add_stop_loss(self, stop: StopLevel):
        """添加止损"""
        self.stop_losses.append(stop)
    
    def add_take_profit(self, tp: TakeProfitLevel):
        """添加止盈"""
        self.take_profits.append(tp)
    
    def check_stops(self, current_price: float) -> Tuple[Optional[StopLevel], Optional[TakeProfitLevel]]:
        """
        检查所有止损止盈
        
        Returns:
            (触发的止损, 触发的止盈)
        """
        triggered_stop = None
        triggered_tp = None
        
        # 检查止损
        for stop in self.stop_losses:
            if stop.check_trigger(current_price, self.is_long):
                triggered_stop = stop
                break
        
        # 检查止盈
        for tp in self.take_profits:
            if tp.check_trigger(current_price, self.is_long):
                triggered_tp = tp
                break
        
        return triggered_stop, triggered_tp
    
    def get_current_risk(self, current_price: float) -> float:
        """
        计算当前风险金额
        
        Args:
            current_price: 当前价格
            
        Returns:
            风险金额
        """
        if not self.stop_losses:
            return 0.0
        
        # 使用最近的止损
        active_stops = [s for s in self.stop_losses if not s.triggered]
        if not active_stops:
            return 0.0
        
        stop = min(active_stops, key=lambda x: abs(x.price - current_price))
        
        if self.is_long:
            risk_per_unit = current_price - stop.price
        else:
            risk_per_unit = stop.price - current_price
        
        return risk_per_unit * self.position_size


class StopManager:
    """止损止盈管理器"""
    
    def __init__(
        self,
        default_stop_pct: float = 0.05,
        default_tp_pct: float = 0.10,
        atr_period: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        初始化止损止盈管理器
        
        Args:
            default_stop_pct: 默认止损百分比
            default_tp_pct: 默认止盈百分比
            atr_period: ATR计算周期
            atr_multiplier: ATR乘数
        """
        self.default_stop_pct = default_stop_pct
        self.default_tp_pct = default_tp_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # 存储各持仓的止损止盈设置
        self.position_stops: Dict[str, PositionStops] = {}
    
    def register_position(
        self,
        position_id: str,
        entry_price: float,
        position_size: float,
        is_long: bool = True
    ) -> PositionStops:
        """
        注册新持仓
        
        Args:
            position_id: 持仓ID
            entry_price: 入场价格
            position_size: 仓位大小
            is_long: 是否多头
            
        Returns:
            持仓止损止盈设置
        """
        pos_stops = PositionStops(
            entry_price=entry_price,
            is_long=is_long,
            position_size=position_size
        )
        self.position_stops[position_id] = pos_stops
        return pos_stops
    
    def remove_position(self, position_id: str):
        """移除持仓"""
        if position_id in self.position_stops:
            del self.position_stops[position_id]
    
    def set_fixed_stop(
        self,
        position_id: str,
        stop_price: Optional[float] = None,
        stop_pct: Optional[float] = None
    ) -> StopLevel:
        """
        设置固定止损
        
        Args:
            position_id: 持仓ID
            stop_price: 止损价格
            stop_pct: 止损百分比（从入场价计算）
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        if stop_price is None:
            if stop_pct is None:
                stop_pct = self.default_stop_pct
            
            if pos.is_long:
                stop_price = pos.entry_price * (1 - stop_pct)
            else:
                stop_price = pos.entry_price * (1 + stop_pct)
        
        stop = StopLevel(price=stop_price, stop_type=StopType.FIXED)
        pos.add_stop_loss(stop)
        
        return stop
    
    def set_trailing_stop(
        self,
        position_id: str,
        trail_pct: float = 0.05,
        activation_pct: Optional[float] = None
    ) -> StopLevel:
        """
        设置追踪止损
        
        Args:
            position_id: 持仓ID
            trail_pct: 追踪百分比
            activation_pct: 激活百分比（盈利多少后激活）
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        # 初始止损价
        if pos.is_long:
            stop_price = pos.entry_price * (1 - trail_pct)
        else:
            stop_price = pos.entry_price * (1 + trail_pct)
        
        stop = StopLevel(
            price=stop_price,
            stop_type=StopType.TRAILING
        )
        
        # 存储追踪参数
        stop.trail_pct = trail_pct
        stop.activation_pct = activation_pct
        stop.highest_price = pos.entry_price
        stop.lowest_price = pos.entry_price
        stop.activated = activation_pct is None
        
        pos.add_stop_loss(stop)
        
        return stop
    
    def update_trailing_stop(
        self,
        position_id: str,
        current_price: float
    ):
        """
        更新追踪止损
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
        """
        if position_id not in self.position_stops:
            return
        
        pos = self.position_stops[position_id]
        
        for stop in pos.stop_losses:
            if stop.stop_type != StopType.TRAILING or stop.triggered:
                continue
            
            # 更新最高/最低价
            if pos.is_long:
                stop.highest_price = max(stop.highest_price, current_price)
                
                # 检查激活
                if not stop.activated and stop.activation_pct:
                    profit_pct = (stop.highest_price - pos.entry_price) / pos.entry_price
                    if profit_pct >= stop.activation_pct:
                        stop.activated = True
                
                if stop.activated:
                    new_stop = stop.highest_price * (1 - stop.trail_pct)
                    stop.price = max(stop.price, new_stop)
            
            else:  # 空头
                stop.lowest_price = min(stop.lowest_price, current_price)
                
                # 检查激活
                if not stop.activated and stop.activation_pct:
                    profit_pct = (pos.entry_price - stop.lowest_price) / pos.entry_price
                    if profit_pct >= stop.activation_pct:
                        stop.activated = True
                
                if stop.activated:
                    new_stop = stop.lowest_price * (1 + stop.trail_pct)
                    stop.price = min(stop.price, new_stop)
    
    def set_atr_stop(
        self,
        position_id: str,
        atr: float,
        multiplier: Optional[float] = None
    ) -> StopLevel:
        """
        设置ATR自适应止损
        
        Args:
            position_id: 持仓ID
            atr: ATR值
            multiplier: ATR乘数
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        if multiplier is None:
            multiplier = self.atr_multiplier
        
        if pos.is_long:
            stop_price = pos.entry_price - atr * multiplier
        else:
            stop_price = pos.entry_price + atr * multiplier
        
        stop = StopLevel(price=stop_price, stop_type=StopType.ATR)
        stop.atr = atr
        stop.multiplier = multiplier
        
        pos.add_stop_loss(stop)
        
        return stop
    
    def set_time_stop(
        self,
        position_id: str,
        max_hours: float,
        exit_price: Optional[float] = None
    ) -> StopLevel:
        """
        设置时间止损
        
        Args:
            position_id: 持仓ID
            max_hours: 最大持仓时间（小时）
            exit_price: 退出价格（None表示市价）
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        if exit_price is None:
            exit_price = pos.entry_price
        
        stop = StopLevel(price=exit_price, stop_type=StopType.TIME)
        stop.max_hours = max_hours
        stop.exit_at_market = exit_price is None
        
        pos.add_stop_loss(stop)
        
        return stop
    
    def check_time_stops(self, position_id: str, current_time: datetime) -> List[StopLevel]:
        """
        检查时间止损
        
        Args:
            position_id: 持仓ID
            current_time: 当前时间
            
        Returns:
            触发的时间止损列表
        """
        if position_id not in self.position_stops:
            return []
        
        pos = self.position_stops[position_id]
        triggered = []
        
        for stop in pos.stop_losses:
            if stop.stop_type != StopType.TIME or stop.triggered:
                continue
            
            elapsed = (current_time - pos.entry_time).total_seconds() / 3600
            
            if elapsed >= stop.max_hours:
                stop.triggered = True
                triggered.append(stop)
        
        return triggered
    
    def set_fixed_take_profit(
        self,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_pct: Optional[float] = None
    ) -> TakeProfitLevel:
        """
        设置固定止盈
        
        Args:
            position_id: 持仓ID
            tp_price: 止盈价格
            tp_pct: 止盈百分比
            
        Returns:
            止盈水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        if tp_price is None:
            if tp_pct is None:
                tp_pct = self.default_tp_pct
            
            if pos.is_long:
                tp_price = pos.entry_price * (1 + tp_pct)
            else:
                tp_price = pos.entry_price * (1 - tp_pct)
        
        tp = TakeProfitLevel(price=tp_price, tp_type=TakeProfitType.FIXED)
        pos.add_take_profit(tp)
        
        return tp
    
    def set_risk_reward_tp(
        self,
        position_id: str,
        stop_price: float,
        risk_reward_ratio: float = 2.0
    ) -> TakeProfitLevel:
        """
        设置风险回报比止盈
        
        Args:
            position_id: 持仓ID
            stop_price: 止损价格
            risk_reward_ratio: 风险回报比
            
        Returns:
            止盈水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        # 计算风险距离
        if pos.is_long:
            risk = pos.entry_price - stop_price
            tp_price = pos.entry_price + risk * risk_reward_ratio
        else:
            risk = stop_price - pos.entry_price
            tp_price = pos.entry_price - risk * risk_reward_ratio
        
        tp = TakeProfitLevel(
            price=tp_price,
            tp_type=TakeProfitType.RISK_REWARD
        )
        tp.risk_reward_ratio = risk_reward_ratio
        
        pos.add_take_profit(tp)
        
        return tp
    
    def set_partial_take_profits(
        self,
        position_id: str,
        levels: List[Tuple[float, float]]
    ) -> List[TakeProfitLevel]:
        """
        设置分批止盈
        
        Args:
            position_id: 持仓ID
            levels: [(价格/百分比, 仓位比例), ...]
            
        Returns:
            止盈水平列表
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        tps = []
        
        for level, size in levels:
            if level < 1:  # 百分比
                if pos.is_long:
                    tp_price = pos.entry_price * (1 + level)
                else:
                    tp_price = pos.entry_price * (1 - level)
            else:  # 绝对价格
                tp_price = level
            
            tp = TakeProfitLevel(
                price=tp_price,
                tp_type=TakeProfitType.PARTIAL,
                size=size
            )
            pos.add_take_profit(tp)
            tps.append(tp)
        
        return tps
    
    def check_all_stops(
        self,
        position_id: str,
        current_price: float,
        current_time: Optional[datetime] = None
    ) -> Tuple[Optional[StopLevel], Optional[TakeProfitLevel]]:
        """
        检查所有止损止盈
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
            current_time: 当前时间（用于时间止损）
            
        Returns:
            (触发的止损, 触发的止盈)
        """
        if position_id not in self.position_stops:
            return None, None
        
        # 更新追踪止损
        self.update_trailing_stop(position_id, current_price)
        
        # 检查时间止损
        if current_time:
            self.check_time_stops(position_id, current_time)
        
        # 检查所有止损止盈
        pos = self.position_stops[position_id]
        return pos.check_stops(current_price)
    
    def get_stop_summary(self, position_id: str) -> Dict:
        """
        获取止损止盈摘要
        
        Args:
            position_id: 持仓ID
            
        Returns:
            摘要字典
        """
        if position_id not in self.position_stops:
            return {}
        
        pos = self.position_stops[position_id]
        
        return {
            'entry_price': pos.entry_price,
            'is_long': pos.is_long,
            'position_size': pos.position_size,
            'stop_losses': [
                {
                    'price': s.price,
                    'type': s.stop_type.value,
                    'triggered': s.triggered
                }
                for s in pos.stop_losses
            ],
            'take_profits': [
                {
                    'price': t.price,
                    'type': t.tp_type.value,
                    'size': t.size,
                    'triggered': t.triggered
                }
                for t in pos.take_profits
            ]
        }
    
    def calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """
        计算ATR
        
        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            period: 周期
            
        Returns:
            ATR值
        """
        if len(closes) < period:
            return 0.0
        
        # 计算真实波幅
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift(1))
        tr3 = abs(lows - closes.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr


class AdvancedStopManager(StopManager):
    """高级止损止盈管理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 高级参数
        self.volatility_lookback = 20
        self.breakeven_activation = 0.02  # 2%盈利后移到成本价
        self.scale_out_levels = [0.5, 0.3, 0.2]  # 分批出场比例
    
    def set_breakeven_stop(
        self,
        position_id: str,
        activation_pct: float = 0.02,
        buffer_pct: float = 0.005
    ) -> StopLevel:
        """
        设置保本止损
        
        Args:
            position_id: 持仓ID
            activation_pct: 激活百分比
            buffer_pct: 缓冲百分比
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        if pos.is_long:
            activation_price = pos.entry_price * (1 + activation_pct)
            stop_price = pos.entry_price * (1 + buffer_pct)
        else:
            activation_price = pos.entry_price * (1 - activation_pct)
            stop_price = pos.entry_price * (1 - buffer_pct)
        
        stop = StopLevel(price=stop_price, stop_type=StopType.FIXED)
        stop.activation_price = activation_price
        stop.activated = False
        
        pos.add_stop_loss(stop)
        
        return stop
    
    def update_breakeven_stops(self, position_id: str, current_price: float):
        """更新保本止损"""
        if position_id not in self.position_stops:
            return
        
        pos = self.position_stops[position_id]
        
        for stop in pos.stop_losses:
            if hasattr(stop, 'activation_price') and not stop.activated:
                if pos.is_long and current_price >= stop.activation_price:
                    stop.activated = True
                elif not pos.is_long and current_price <= stop.activation_price:
                    stop.activated = True
    
    def set_volatility_stop(
        self,
        position_id: str,
        prices: pd.Series,
        vol_period: int = 20,
        multiplier: float = 1.5
    ) -> StopLevel:
        """
        设置波动率止损
        
        Args:
            position_id: 持仓ID
            prices: 价格序列
            vol_period: 波动率周期
            multiplier: 波动率乘数
            
        Returns:
            止损水平
        """
        if position_id not in self.position_stops:
            raise ValueError(f"Position {position_id} not found")
        
        pos = self.position_stops[position_id]
        
        # 计算波动率
        returns = prices.pct_change().dropna()
        volatility = returns.tail(vol_period).std()
        
        # 设置止损
        if pos.is_long:
            stop_price = pos.entry_price * (1 - volatility * multiplier)
        else:
            stop_price = pos.entry_price * (1 + volatility * multiplier)
        
        stop = StopLevel(price=stop_price, stop_type=StopType.VOLATILITY)
        stop.volatility = volatility
        stop.vol_period = vol_period
        
        pos.add_stop_loss(stop)
        
        return stop


# 便捷函数
def quick_stop_loss(
    entry_price: float,
    stop_pct: float = 0.05,
    is_long: bool = True
) -> float:
    """
    快速计算止损价格
    
    Args:
        entry_price: 入场价格
        stop_pct: 止损百分比
        is_long: 是否多头
        
    Returns:
        止损价格
    """
    if is_long:
        return entry_price * (1 - stop_pct)
    else:
        return entry_price * (1 + stop_pct)


def quick_take_profit(
    entry_price: float,
    stop_price: float,
    risk_reward: float = 2.0,
    is_long: bool = True
) -> float:
    """
    快速计算止盈价格
    
    Args:
        entry_price: 入场价格
        stop_price: 止损价格
        risk_reward: 风险回报比
        is_long: 是否多头
        
    Returns:
        止盈价格
    """
    if is_long:
        risk = entry_price - stop_price
        return entry_price + risk * risk_reward
    else:
        risk = stop_price - entry_price
        return entry_price - risk * risk_reward
