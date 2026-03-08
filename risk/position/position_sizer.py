"""
仓位管理模块

提供多种仓位计算方法，包括凯利公式、固定比例、波动率目标等
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import warnings


class PositionSizingMethod(Enum):
    """仓位计算方法"""
    KELLY = "kelly"  # 凯利公式
    FIXED_FRACTION = "fixed_fraction"  # 固定比例
    FIXED_RATIO = "fixed_ratio"  # 固定比率
    VOLATILITY_TARGET = "volatility_target"  # 波动率目标
    ATR_BASED = "atr_based"  # ATR基础
    OPTIMAL_F = "optimal_f"  # Optimal f
    PERCENT_RISK = "percent_risk"  # 百分比风险


@dataclass
class PositionSize:
    """仓位大小结果"""
    size: float  # 仓位大小（基础货币单位）
    size_in_units: float  # 仓位大小（交易单位）
    leverage: float  # 建议杠杆
    risk_amount: float  # 风险金额
    risk_percent: float  # 风险百分比
    margin_required: float  # 所需保证金
    notional_value: float  # 名义价值


class PositionSizer:
    """仓位管理器"""
    
    def __init__(
        self,
        account_balance: float = 10000.0,
        max_position_pct: float = 0.5,
        max_leverage: float = 10.0,
        contract_type: str = "spot"  # spot, perpetual, futures
    ):
        """
        初始化仓位管理器
        
        Args:
            account_balance: 账户余额
            max_position_pct: 最大仓位比例（相对于账户）
            max_leverage: 最大杠杆倍数
            contract_type: 合约类型
        """
        self.account_balance = account_balance
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.contract_type = contract_type
        
        # 默认参数
        self.kelly_fraction = 0.5  # 半凯利
        self.volatility_target = 0.15  # 年化波动率目标
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.risk_per_trade = 0.02  # 每笔交易风险2%
    
    def set_account_balance(self, balance: float):
        """设置账户余额"""
        self.account_balance = balance
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5
    ) -> float:
        """
        凯利公式计算最优仓位比例
        
        f* = (p * b - q) / b
        其中:
        p = 胜率
        q = 败率 = 1 - p
        b = 平均盈利/平均亏损
        
        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利
            avg_loss: 平均亏损（正值）
            fraction: 凯利分数（0.5为半凯利）
            
        Returns:
            最优仓位比例
        """
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss  # 盈亏比
        q = 1 - win_rate
        
        kelly_f = (win_rate * b - q) / b
        
        # 应用凯利分数
        position_pct = kelly_f * fraction
        
        # 限制在合理范围内
        position_pct = max(0.0, min(position_pct, self.max_position_pct))
        
        return position_pct
    
    def kelly_from_returns(self, returns: pd.Series, fraction: float = 0.5) -> float:
        """
        从历史收益率计算凯利公式仓位
        
        Args:
            returns: 收益率序列
            fraction: 凯利分数
            
        Returns:
            最优仓位比例
        """
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        return self.kelly_criterion(win_rate, avg_win, avg_loss, fraction)
    
    def fixed_fraction(self, fraction: float) -> float:
        """
        固定比例仓位
        
        Args:
            fraction: 仓位比例
            
        Returns:
            仓位大小
        """
        fraction = min(fraction, self.max_position_pct)
        return self.account_balance * fraction
    
    def fixed_ratio(
        self,
        delta: float,
        current_position: int = 0
    ) -> float:
        """
        固定比率仓位（Ryan Jones方法）
        
        Args:
            delta: 每增加一个单位需要的盈利
            current_position: 当前持仓单位数
            
        Returns:
            仓位大小
        """
        # 计算可以持有的单位数
        units = int(np.sqrt(1 + 2 * self.account_balance / delta) - 1) / 2
        units = max(0, int(units))
        
        return units * delta
    
    def volatility_target_sizing(
        self,
        returns: pd.Series,
        target_volatility: Optional[float] = None,
        periods_per_year: int = 365
    ) -> float:
        """
        波动率目标仓位
        
        Args:
            returns: 收益率序列
            target_volatility: 目标年化波动率
            periods_per_year: 每年周期数
            
        Returns:
            仓位比例
        """
        if target_volatility is None:
            target_volatility = self.volatility_target
        
        if len(returns) < 2:
            return 0.0
        
        current_volatility = returns.std() * np.sqrt(periods_per_year)
        
        if current_volatility == 0:
            return self.max_position_pct
        
        position_pct = target_volatility / current_volatility
        position_pct = min(position_pct, self.max_position_pct)
        
        return position_pct
    
    def atr_based_sizing(
        self,
        current_price: float,
        atr: float,
        risk_pct: Optional[float] = None,
        atr_multiplier: float = 2.0
    ) -> PositionSize:
        """
        ATR基础仓位计算
        
        Args:
            current_price: 当前价格
            atr: ATR值
            risk_pct: 风险百分比
            atr_multiplier: ATR乘数
            
        Returns:
            仓位大小结果
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade
        
        if atr == 0:
            return PositionSize(0, 0, 0, 0, 0, 0, 0)
        
        # 计算止损距离
        stop_distance = atr * atr_multiplier
        
        # 计算风险金额
        risk_amount = self.account_balance * risk_pct
        
        # 计算仓位大小
        position_value = risk_amount / (stop_distance / current_price)
        
        # 限制最大仓位
        max_position = self.account_balance * self.max_position_pct
        position_value = min(position_value, max_position)
        
        # 计算交易单位
        size_in_units = position_value / current_price
        
        # 计算杠杆
        leverage = position_value / self.account_balance if self.account_balance > 0 else 0
        leverage = min(leverage, self.max_leverage)
        
        # 计算保证金
        if self.contract_type == "spot":
            margin_required = position_value
        else:
            margin_required = position_value / max(leverage, 1)
        
        return PositionSize(
            size=position_value,
            size_in_units=size_in_units,
            leverage=leverage,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            margin_required=margin_required,
            notional_value=position_value
        )
    
    def percent_risk_sizing(
        self,
        entry_price: float,
        stop_price: float,
        risk_pct: Optional[float] = None
    ) -> PositionSize:
        """
        百分比风险仓位计算
        
        Args:
            entry_price: 入场价格
            stop_price: 止损价格
            risk_pct: 风险百分比
            
        Returns:
            仓位大小结果
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade
        
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance == 0 or entry_price == 0:
            return PositionSize(0, 0, 0, 0, 0, 0, 0)
        
        # 风险金额
        risk_amount = self.account_balance * risk_pct
        
        # 计算仓位大小
        position_value = risk_amount * entry_price / stop_distance
        
        # 限制最大仓位
        max_position = self.account_balance * self.max_position_pct
        position_value = min(position_value, max_position)
        
        # 计算交易单位
        size_in_units = position_value / entry_price
        
        # 计算杠杆
        leverage = position_value / self.account_balance if self.account_balance > 0 else 0
        leverage = min(leverage, self.max_leverage)
        
        # 计算保证金
        if self.contract_type == "spot":
            margin_required = position_value
        else:
            margin_required = position_value / max(leverage, 1)
        
        return PositionSize(
            size=position_value,
            size_in_units=size_in_units,
            leverage=leverage,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            margin_required=margin_required,
            notional_value=position_value
        )
    
    def optimal_f(
        self,
        returns: pd.Series,
        num_points: int = 100
    ) -> float:
        """
        计算Optimal f（Ralph Vince）
        
        Args:
            returns: 收益率序列
            num_points: 搜索点数
            
        Returns:
            Optimal f值
        """
        if len(returns) == 0:
            return 0.0
        
        # 搜索最优f
        f_values = np.linspace(0.01, 1.0, num_points)
        best_f = 0.0
        best_growth = -np.inf
        
        for f in f_values:
            # 计算几何平均增长
            growths = 1 + f * returns
            
            # 如果有任何负值，跳过
            if np.any(growths <= 0):
                continue
            
            # 计算几何平均
            geo_mean = np.exp(np.mean(np.log(growths)))
            
            if geo_mean > best_growth:
                best_growth = geo_mean
                best_f = f
        
        # 应用限制
        best_f = min(best_f, self.max_position_pct)
        
        return best_f
    
    def calculate_position_size(
        self,
        method: PositionSizingMethod,
        current_price: float,
        returns: Optional[pd.Series] = None,
        atr: Optional[float] = None,
        stop_price: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        **kwargs
    ) -> PositionSize:
        """
        通用仓位计算接口
        
        Args:
            method: 计算方法
            current_price: 当前价格
            returns: 收益率序列
            atr: ATR值
            stop_price: 止损价格
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            **kwargs: 其他参数
            
        Returns:
            仓位大小结果
        """
        if method == PositionSizingMethod.KELLY:
            if returns is not None:
                position_pct = self.kelly_from_returns(returns, kwargs.get('fraction', self.kelly_fraction))
            elif all(x is not None for x in [win_rate, avg_win, avg_loss]):
                position_pct = self.kelly_criterion(win_rate, avg_win, avg_loss, kwargs.get('fraction', self.kelly_fraction))
            else:
                position_pct = 0.1
            
            position_value = self.account_balance * position_pct
            
        elif method == PositionSizingMethod.FIXED_FRACTION:
            fraction = kwargs.get('fraction', 0.1)
            position_value = self.fixed_fraction(fraction)
            position_pct = fraction
            
        elif method == PositionSizingMethod.FIXED_RATIO:
            delta = kwargs.get('delta', 1000)
            position_value = self.fixed_ratio(delta)
            position_pct = position_value / self.account_balance if self.account_balance > 0 else 0
            
        elif method == PositionSizingMethod.VOLATILITY_TARGET:
            if returns is None:
                position_pct = 0.1
            else:
                target_vol = kwargs.get('target_volatility', self.volatility_target)
                position_pct = self.volatility_target_sizing(returns, target_vol)
            position_value = self.account_balance * position_pct
            
        elif method == PositionSizingMethod.ATR_BASED:
            if atr is None:
                return PositionSize(0, 0, 0, 0, 0, 0, 0)
            return self.atr_based_sizing(
                current_price, atr,
                kwargs.get('risk_pct', self.risk_per_trade),
                kwargs.get('atr_multiplier', self.atr_multiplier)
            )
            
        elif method == PositionSizingMethod.OPTIMAL_F:
            if returns is None:
                position_pct = 0.1
            else:
                position_pct = self.optimal_f(returns)
            position_value = self.account_balance * position_pct
            
        elif method == PositionSizingMethod.PERCENT_RISK:
            if stop_price is None:
                return PositionSize(0, 0, 0, 0, 0, 0, 0)
            return self.percent_risk_sizing(
                current_price, stop_price,
                kwargs.get('risk_pct', self.risk_per_trade)
            )
        
        else:
            position_pct = 0.1
            position_value = self.account_balance * position_pct
        
        # 构建结果
        size_in_units = position_value / current_price if current_price > 0 else 0
        leverage = position_value / self.account_balance if self.account_balance > 0 else 0
        leverage = min(leverage, self.max_leverage)
        risk_amount = position_value * 0.02  # 默认2%风险
        
        if self.contract_type == "spot":
            margin_required = position_value
        else:
            margin_required = position_value / max(leverage, 1)
        
        return PositionSize(
            size=position_value,
            size_in_units=size_in_units,
            leverage=leverage,
            risk_amount=risk_amount,
            risk_percent=position_pct * 0.02,
            margin_required=margin_required,
            notional_value=position_value
        )
    
    def adjust_for_correlation(
        self,
        position_sizes: Dict[str, PositionSize],
        correlation_matrix: pd.DataFrame,
        max_correlation_exposure: float = 0.5
    ) -> Dict[str, PositionSize]:
        """
        根据相关性调整仓位
        
        Args:
            position_sizes: 各资产仓位
            correlation_matrix: 相关性矩阵
            max_correlation_exposure: 最大相关性敞口
            
        Returns:
            调整后的仓位
        """
        # 计算相关性调整因子
        adjusted_sizes = {}
        
        for symbol, pos_size in position_sizes.items():
            if symbol not in correlation_matrix.columns:
                adjusted_sizes[symbol] = pos_size
                continue
            
            # 计算平均相关性
            avg_corr = correlation_matrix[symbol].mean()
            
            # 高相关性时降低仓位
            if avg_corr > 0.7:
                adjustment = max_correlation_exposure / avg_corr
            else:
                adjustment = 1.0
            
            # 调整仓位
            adjusted_size = PositionSize(
                size=pos_size.size * adjustment,
                size_in_units=pos_size.size_in_units * adjustment,
                leverage=pos_size.leverage,
                risk_amount=pos_size.risk_amount * adjustment,
                risk_percent=pos_size.risk_percent,
                margin_required=pos_size.margin_required * adjustment,
                notional_value=pos_size.notional_value * adjustment
            )
            
            adjusted_sizes[symbol] = adjusted_size
        
        return adjusted_sizes


class DynamicPositionSizer(PositionSizer):
    """动态仓位管理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 动态调整参数
        self.equity_curve: List[float] = []
        self.drawdown_threshold = 0.1  # 10%回撤开始降仓
        self.drawdown_max = 0.2  # 20%回撤最大降仓
        self.recovery_factor = 0.5  # 恢复速度
    
    def update_equity(self, equity: float):
        """更新权益曲线"""
        self.equity_curve.append(equity)
        self.set_account_balance(equity)
    
    def get_drawdown_adjustment(self) -> float:
        """
        根据回撤调整仓位系数
        
        Returns:
            调整系数 (0-1)
        """
        if len(self.equity_curve) < 2:
            return 1.0
        
        # 计算当前回撤
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        if drawdown < self.drawdown_threshold:
            return 1.0
        elif drawdown >= self.drawdown_max:
            return 0.2  # 最大降仓到20%
        else:
            # 线性降仓
            adjustment = 1.0 - (drawdown - self.drawdown_threshold) / \
                        (self.drawdown_max - self.drawdown_threshold) * 0.8
            return max(0.2, adjustment)
    
    def calculate_dynamic_size(
        self,
        method: PositionSizingMethod,
        current_price: float,
        **kwargs
    ) -> PositionSize:
        """
        计算动态调整的仓位
        
        Args:
            method: 计算方法
            current_price: 当前价格
            **kwargs: 其他参数
            
        Returns:
            调整后的仓位
        """
        # 计算基础仓位
        base_size = self.calculate_position_size(method, current_price, **kwargs)
        
        # 应用回撤调整
        adjustment = self.get_drawdown_adjustment()
        
        return PositionSize(
            size=base_size.size * adjustment,
            size_in_units=base_size.size_in_units * adjustment,
            leverage=base_size.leverage,
            risk_amount=base_size.risk_amount * adjustment,
            risk_percent=base_size.risk_percent,
            margin_required=base_size.margin_required * adjustment,
            notional_value=base_size.notional_value * adjustment
        )


# 便捷函数
def quick_position_size(
    account_balance: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.02
) -> float:
    """
    快速计算仓位大小
    
    Args:
        account_balance: 账户余额
        entry_price: 入场价格
        stop_price: 止损价格
        risk_pct: 风险百分比
        
    Returns:
        仓位大小（交易单位）
    """
    sizer = PositionSizer(account_balance)
    result = sizer.percent_risk_sizing(entry_price, stop_price, risk_pct)
    return result.size_in_units
