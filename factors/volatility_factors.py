"""
波动率因子模块
Volatility Factors Module

包含历史波动率、Parkinson波动率、GARCH等波动率相关因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize

from .base_factor import (
    VolatilityFactor, FactorDirection,
    ema, sma, rolling_std, rolling_max, rolling_min,
    true_range, atr
)


# ==================== 历史波动率类因子 ====================

class HistoricalVolatilityFactor(VolatilityFactor):
    """
    历史波动率因子
    
    基于收盘价的标准差计算
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"HV_{window}",
            description="历史波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.annualize = annualize
        self.metadata.parameters = {'window': window, 'annualize': annualize}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 对数收益率
        log_returns = np.log(close / close.shift(1))
        
        # 波动率
        hv = log_returns.rolling(window=self.window, min_periods=1).std()
        
        # 年化
        if self.annualize:
            # 假设252个交易日
            hv = hv * np.sqrt(252)
        
        return hv


class RealizedVolatilityFactor(VolatilityFactor):
    """
    实现波动率因子
    
    基于高频数据的实现波动率
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"RV_{window}",
            description="实现波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 对数收益率
        log_returns = np.log(close / close.shift(1))
        
        # 实现波动率（平方收益率的和）
        rv = np.sqrt((log_returns ** 2).rolling(window=self.window, min_periods=1).sum())
        
        return rv


class CloseToCloseVolatilityFactor(VolatilityFactor):
    """
    收盘-收盘波动率因子
    
    基于收盘价变化的波动率
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"CCV_{window}",
            description="收盘-收盘波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 简单收益率
        returns = close.pct_change()
        
        # 波动率
        volatility = returns.rolling(window=self.window, min_periods=1).std()
        
        return volatility * np.sqrt(252)  # 年化


# ==================== Parkinson波动率类因子 ====================

class ParkinsonVolatilityFactor(VolatilityFactor):
    """
    Parkinson波动率因子
    
    基于高低价的波动率估计
    比收盘-收盘波动率更高效
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"Parkinson_{window}",
            description="Parkinson波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.annualize = annualize
        self.metadata.parameters = {'window': window, 'annualize': annualize}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        
        # Parkinson波动率
        log_hl = np.log(high / low)
        parkinson_var = (log_hl ** 2).rolling(window=self.window, min_periods=1).mean() / (4 * np.log(2))
        parkinson_vol = np.sqrt(parkinson_var)
        
        # 年化
        if self.annualize:
            parkinson_vol = parkinson_vol * np.sqrt(252)
        
        return parkinson_vol


class GarmanKlassVolatilityFactor(VolatilityFactor):
    """
    Garman-Klass波动率因子
    
    基于OHLC的波动率估计
    比Parkinson更高效
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"GK_{window}",
            description="Garman-Klass波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.annualize = annualize
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        open_price = data.get('open', close.shift(1))
        
        # Garman-Klass波动率
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_var = gk_var.rolling(window=self.window, min_periods=1).mean()
        gk_vol = np.sqrt(gk_var)
        
        # 年化
        if self.annualize:
            gk_vol = gk_vol * np.sqrt(252)
        
        return gk_vol


class RogersSatchellVolatilityFactor(VolatilityFactor):
    """
    Rogers-Satchell波动率因子
    
    允许有漂移的波动率估计
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"RS_{window}",
            description="Rogers-Satchell波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.annualize = annualize
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        open_price = data.get('open', close.shift(1))
        
        # Rogers-Satchell波动率
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)
        
        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs_var = rs_var.rolling(window=self.window, min_periods=1).mean()
        rs_vol = np.sqrt(rs_var)
        
        # 年化
        if self.annualize:
            rs_vol = rs_vol * np.sqrt(252)
        
        return rs_vol


class YangZhangVolatilityFactor(VolatilityFactor):
    """
    Yang-Zhang波动率因子
    
    综合隔夜和日内信息的波动率估计
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"YZ_{window}",
            description="Yang-Zhang波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.annualize = annualize
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        open_price = data.get('open', close.shift(1))
        
        # 隔夜波动率
        log_oc = np.log(open_price / close.shift(1))
        overnight_var = (log_oc ** 2).rolling(window=self.window, min_periods=1).mean()
        
        # Rogers-Satchell部分
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)
        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs_var = rs_var.rolling(window=self.window, min_periods=1).mean()
        
        # Yang-Zhang波动率
        k = 0.34 / (1.34 + (self.window + 1) / (self.window - 1))
        yz_var = overnight_var + k * log_oc.rolling(window=self.window, min_periods=1).mean() ** 2 + rs_var
        yz_vol = np.sqrt(yz_var)
        
        # 年化
        if self.annualize:
            yz_vol = yz_vol * np.sqrt(252)
        
        return yz_vol


# ==================== GARCH类因子 ====================

class GARCHVolatilityFactor(VolatilityFactor):
    """
    GARCH(1,1)波动率因子
    
    使用GARCH模型估计条件波动率
    """
    
    def __init__(self, window: int = 252, update_freq: int = 20):
        super().__init__(
            name=f"GARCH_{window}_{update_freq}",
            description="GARCH(1,1)波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.update_freq = update_freq
        self._garch_params = None
    
    def _garch_likelihood(self, params, returns):
        """GARCH对数似然函数"""
        omega, alpha, beta = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        return -log_likelihood
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 对数收益率
        returns = np.log(close / close.shift(1)).dropna()
        
        # 初始化波动率序列
        garch_vol = pd.Series(index=close.index, dtype=float)
        
        # 滚动估计GARCH参数
        for i in range(self.window, len(returns) + 1):
            if i == self.window or (i - self.window) % self.update_freq == 0:
                # 重新估计参数
                window_returns = returns.iloc[i-self.window:i].values
                
                # 初始参数
                x0 = [0.000001, 0.1, 0.85]
                
                # 约束优化
                bounds = [(1e-8, None), (0, 1), (0, 1)]
                
                try:
                    result = minimize(
                        self._garch_likelihood, x0,
                        args=(window_returns,),
                        bounds=bounds,
                        method='L-BFGS-B'
                    )
                    omega, alpha, beta = result.x
                    self._garch_params = (omega, alpha, beta)
                except:
                    omega, alpha, beta = x0
            else:
                if self._garch_params is not None:
                    omega, alpha, beta = self._garch_params
                else:
                    omega, alpha, beta = 0.000001, 0.1, 0.85
            
            # 计算条件波动率
            if i > self.window:
                prev_vol = garch_vol.iloc[i-2] ** 2 if not pd.isna(garch_vol.iloc[i-2]) else np.var(returns.iloc[i-self.window:i])
                prev_return = returns.iloc[i-1]
                current_vol2 = omega + alpha * prev_return**2 + beta * prev_vol
                garch_vol.iloc[i-1] = np.sqrt(current_vol2)
            else:
                garch_vol.iloc[i-1] = np.std(returns.iloc[:i])
        
        # 年化
        garch_vol = garch_vol * np.sqrt(252)
        
        return garch_vol


class EWMAVolatilityFactor(VolatilityFactor):
    """
    EWMA波动率因子
    
    指数加权移动平均波动率
    RiskMetrics方法
    """
    
    def __init__(self, lambda_param: float = 0.94, annualize: bool = True):
        super().__init__(
            name=f"EWMA_{lambda_param}",
            description="EWMA波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.lambda_param = lambda_param
        self.annualize = annualize
        self.metadata.parameters = {'lambda': lambda_param, 'annualize': annualize}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 对数收益率
        returns = np.log(close / close.shift(1))
        
        # EWMA方差
        ewma_var = returns.ewm(alpha=1-self.lambda_param, adjust=False).var()
        
        # 波动率
        ewma_vol = np.sqrt(ewma_var)
        
        # 年化
        if self.annualize:
            ewma_vol = ewma_vol * np.sqrt(252)
        
        return ewma_vol


# ==================== 波动率锥类因子 ====================

class VolatilityConeFactor(VolatilityFactor):
    """
    波动率锥因子
    
    当前波动率相对于历史分布的位置
    """
    
    def __init__(self, window: int = 20, lookback: int = 252):
        super().__init__(
            name=f"Vol_Cone_{window}_{lookback}",
            description="波动率锥因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算滚动波动率
        returns = close.pct_change()
        rolling_vol = returns.rolling(window=self.window, min_periods=1).std() * np.sqrt(252)
        
        # 计算历史分位数
        vol_rank = rolling_vol.rolling(window=self.lookback, min_periods=1).apply(
            lambda x: stats.percentileofscore(x[:-1], x[-1]) / 100 if len(x) > 1 else 0.5,
            raw=True
        )
        
        # 中心化
        vol_signal = vol_rank - 0.5
        
        return vol_signal


class VolatilityPercentileFactor(VolatilityFactor):
    """
    波动率百分位因子
    
    当前波动率的历史百分位
    """
    
    def __init__(self, window: int = 20, lookback: int = 252):
        super().__init__(
            name=f"Vol_Percentile_{window}_{lookback}",
            description="波动率百分位因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算滚动波动率
        returns = close.pct_change()
        rolling_vol = returns.rolling(window=self.window, min_periods=1).std()
        
        # 历史百分位
        percentile = rolling_vol.rolling(window=self.lookback, min_periods=1).apply(
            lambda x: np.mean(x[:-1] <= x[-1]) if len(x) > 1 else 0.5,
            raw=True
        )
        
        return percentile - 0.5


# ==================== 波动率状态类因子 ====================

class VolatilityRegimeFactor(VolatilityFactor):
    """
    波动率状态因子
    
    识别高波动/低波动状态
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20, 
                 threshold: float = 1.5):
        super().__init__(
            name=f"Vol_Regime_{short_window}_{long_window}_{threshold}",
            description="波动率状态因子",
            direction=FactorDirection.NEUTRAL
        )
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算收益率
        returns = close.pct_change()
        
        # 短期和长期波动率
        vol_short = returns.rolling(window=self.short_window, min_periods=1).std()
        vol_long = returns.rolling(window=self.long_window, min_periods=1).std()
        
        # 波动率比率
        vol_ratio = vol_short / (vol_long + 1e-8)
        
        # 状态判断
        regime = pd.Series(0, index=close.index)
        regime[vol_ratio > self.threshold] = 1   # 高波动
        regime[vol_ratio < 1/self.threshold] = -1  # 低波动
        
        return regime


class VolatilityTrendFactor(VolatilityFactor):
    """
    波动率趋势因子
    
    波动率的趋势方向
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Vol_Trend_{window}",
            description="波动率趋势因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算波动率
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window, min_periods=1).std()
        
        # 波动率的趋势
        vol_sma = sma(volatility, self.window)
        vol_trend = (volatility - vol_sma) / (vol_sma + 1e-8)
        
        return vol_trend


class VolatilitySkewnessFactor(VolatilityFactor):
    """
    波动率偏度因子
    
    收益率分布的偏度
    """
    
    def __init__(self, window: int = 60):
        super().__init__(
            name=f"Vol_Skewness_{window}",
            description="波动率偏度因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 收益率
        returns = close.pct_change()
        
        # 偏度
        skewness = returns.rolling(window=self.window, min_periods=1).skew()
        
        return skewness


class VolatilityKurtosisFactor(VolatilityFactor):
    """
    波动率峰度因子
    
    收益率分布的峰度
    """
    
    def __init__(self, window: int = 60):
        super().__init__(
            name=f"Vol_Kurtosis_{window}",
            description="波动率峰度因子",
            direction=FactorDirection.NEGATIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 收益率
        returns = close.pct_change()
        
        # 峰度
        kurtosis = returns.rolling(window=self.window, min_periods=1).kurt()
        
        return kurtosis


# ==================== 跳跃波动率类因子 ====================

class JumpVolatilityFactor(VolatilityFactor):
    """
    跳跃波动率因子
    
    检测价格跳跃
    """
    
    def __init__(self, window: int = 20, threshold: float = 3.0):
        super().__init__(
            name=f"Jump_Vol_{window}_{threshold}",
            description="跳跃波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 收益率
        returns = close.pct_change()
        
        # 滚动标准差
        rolling_std = returns.rolling(window=self.window, min_periods=1).std()
        
        # 跳跃检测
        jump = np.abs(returns) > self.threshold * rolling_std
        
        # 跳跃计数（滚动窗口）
        jump_count = jump.rolling(window=self.window, min_periods=1).sum()
        
        return jump_count


class IntradayRangeFactor(VolatilityFactor):
    """
    日内波幅因子
    
    基于日内高低点的波动率
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Intraday_Range_{window}",
            description="日内波幅因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 日内波幅
        intraday_range = (high - low) / (close + 1e-8)
        
        # 滚动平均
        range_ma = intraday_range.rolling(window=self.window, min_periods=1).mean()
        
        return range_ma


class OvernightGapFactor(VolatilityFactor):
    """
    隔夜跳空因子
    
    开盘相对于前收盘的跳空幅度
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Overnight_Gap_{window}",
            description="隔夜跳空因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        open_price = data.get('open', close)
        
        # 隔夜跳空
        overnight_gap = (open_price - close.shift(1)) / (close.shift(1) + 1e-8)
        
        # 跳空波动率
        gap_vol = overnight_gap.rolling(window=self.window, min_periods=1).std()
        
        return gap_vol * np.sqrt(252)


# ==================== 因子工厂函数 ====================

def create_all_volatility_factors() -> List[VolatilityFactor]:
    """创建所有波动率因子实例"""
    factors = []
    
    # 历史波动率类因子
    factors.extend([
        HistoricalVolatilityFactor(20),
        HistoricalVolatilityFactor(60),
        RealizedVolatilityFactor(20),
        CloseToCloseVolatilityFactor(20),
    ])
    
    # Parkinson波动率类因子
    factors.extend([
        ParkinsonVolatilityFactor(20),
        ParkinsonVolatilityFactor(60),
        GarmanKlassVolatilityFactor(20),
        RogersSatchellVolatilityFactor(20),
        YangZhangVolatilityFactor(20),
    ])
    
    # GARCH类因子
    factors.extend([
        GARCHVolatilityFactor(252, 20),
        EWMAVolatilityFactor(0.94),
        EWMAVolatilityFactor(0.97),
    ])
    
    # 波动率锥类因子
    factors.extend([
        VolatilityConeFactor(20, 252),
        VolatilityPercentileFactor(20, 252),
    ])
    
    # 波动率状态类因子
    factors.extend([
        VolatilityRegimeFactor(5, 20, 1.5),
        VolatilityTrendFactor(20),
        VolatilitySkewnessFactor(60),
        VolatilityKurtosisFactor(60),
    ])
    
    # 跳跃波动率类因子
    factors.extend([
        JumpVolatilityFactor(20, 3.0),
        IntradayRangeFactor(20),
        OvernightGapFactor(20),
    ])
    
    return factors


# 导出所有因子类
__all__ = [
    'HistoricalVolatilityFactor', 'RealizedVolatilityFactor', 'CloseToCloseVolatilityFactor',
    'ParkinsonVolatilityFactor', 'GarmanKlassVolatilityFactor',
    'RogersSatchellVolatilityFactor', 'YangZhangVolatilityFactor',
    'GARCHVolatilityFactor', 'EWMAVolatilityFactor',
    'VolatilityConeFactor', 'VolatilityPercentileFactor',
    'VolatilityRegimeFactor', 'VolatilityTrendFactor',
    'VolatilitySkewnessFactor', 'VolatilityKurtosisFactor',
    'JumpVolatilityFactor', 'IntradayRangeFactor', 'OvernightGapFactor',
    'create_all_volatility_factors'
]
