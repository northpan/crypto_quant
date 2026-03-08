"""
技术指标因子模块
Technical Indicator Factors Module

包含趋势类、动量类、波动类技术指标因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .base_factor import (
    TechnicalFactor, FactorDirection,
    ema, sma, rolling_std, rolling_max, rolling_min, rolling_sum,
    safe_divide, true_range, atr, rsi, macd, bollinger_bands, stochastic
)


# ==================== 趋势类因子 ====================

class MACDFactor(TechnicalFactor):
    """
    MACD因子
    
    基于MACD指标的趋势动量因子
    - MACD > 0 且上升：多头趋势
    - MACD < 0 且下降：空头趋势
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(
            name=f"MACD_{fast}_{slow}_{signal}",
            description="MACD趋势动量因子",
            direction=FactorDirection.POSITIVE
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.metadata.parameters = {'fast': fast, 'slow': slow, 'signal': signal}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        macd_result = macd(close, self.fast, self.slow, self.signal)
        
        # 返回MACD柱状图作为因子值
        return macd_result['histogram']


class MACDSignalFactor(TechnicalFactor):
    """
    MACD信号因子
    
    MACD与信号线的交叉信号
    - 金叉：买入信号
    - 死叉：卖出信号
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(
            name=f"MACD_Signal_{fast}_{slow}_{signal}",
            description="MACD交叉信号因子",
            direction=FactorDirection.POSITIVE
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        macd_result = macd(close, self.fast, self.slow, self.signal)
        
        # MACD线减去信号线
        diff = macd_result['macd'] - macd_result['signal']
        
        # 返回差值的变化方向
        return np.sign(diff)


class EMAcrossFactor(TechnicalFactor):
    """
    EMA交叉因子
    
    短期EMA与长期EMA的交叉
    - 短期上穿长期：买入信号
    - 短期下穿长期：卖出信号
    """
    
    def __init__(self, short: int = 12, long: int = 26):
        super().__init__(
            name=f"EMA_Cross_{short}_{long}",
            description="EMA交叉因子",
            direction=FactorDirection.POSITIVE
        )
        self.short = short
        self.long = long
        self.metadata.parameters = {'short': short, 'long': long}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        ema_short = ema(close, self.short)
        ema_long = ema(close, self.long)
        
        # 计算交叉信号
        diff = ema_short - ema_long
        diff_prev = diff.shift(1)
        
        # 金叉=1, 死叉=-1, 无交叉=0
        signal = pd.Series(0, index=close.index)
        signal[(diff > 0) & (diff_prev <= 0)] = 1   # 金叉
        signal[(diff < 0) & (diff_prev >= 0)] = -1  # 死叉
        
        return signal


class SMAcrossFactor(TechnicalFactor):
    """
    SMA交叉因子
    
    简单移动平均交叉因子
    """
    
    def __init__(self, short: int = 10, long: int = 30):
        super().__init__(
            name=f"SMA_Cross_{short}_{long}",
            description="SMA交叉因子",
            direction=FactorDirection.POSITIVE
        )
        self.short = short
        self.long = long
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        sma_short = sma(close, self.short)
        sma_long = sma(close, self.long)
        
        # 价格相对于双均线的位置
        price_position = (close - sma_short) / (sma_short - sma_long + 1e-8)
        
        return price_position


class ADXFactor(TechnicalFactor):
    """
    ADX趋势强度因子
    
    Average Directional Index
    - ADX > 25：强趋势
    - ADX < 20：弱趋势/震荡
    """
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"ADX_{window}",
            description="ADX趋势强度因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # +DM和-DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        
        # 真实波幅
        tr = true_range(high, low, close)
        
        # 平滑处理
        atr_val = tr.rolling(window=self.window, min_periods=1).mean()
        plus_di = 100 * plus_dm.rolling(window=self.window, min_periods=1).mean() / (atr_val + 1e-8)
        minus_di = 100 * minus_dm.rolling(window=self.window, min_periods=1).mean() / (atr_val + 1e-8)
        
        # DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=self.window, min_periods=1).mean()
        
        # 返回趋势强度与方向的组合
        return adx * np.sign(plus_di - minus_di)


class TrendStrengthFactor(TechnicalFactor):
    """
    趋势强度因子
    
    基于价格与多周期均线的偏离度
    """
    
    def __init__(self, windows: List[int] = [5, 10, 20, 60]):
        super().__init__(
            name=f"Trend_Strength_{'_'.join(map(str, windows))}",
            description="多周期趋势强度因子",
            direction=FactorDirection.POSITIVE
        )
        self.windows = windows
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算各周期EMA
        emas = [ema(close, w) for w in self.windows]
        
        # 计算价格与各EMA的偏离度
        deviations = [(close - e) / (e + 1e-8) for e in emas]
        
        # 加权平均（短期权重更高）
        weights = np.array(self.windows[::-1])  # 反转权重
        weights = weights / weights.sum()
        
        trend_strength = sum(w * d for w, d in zip(weights, deviations))
        
        return trend_strength


# ==================== 动量类因子 ====================

class RSIFactor(TechnicalFactor):
    """
    RSI动量因子
    
    相对强弱指数
    - RSI > 70：超买
    - RSI < 30：超卖
    """
    
    def __init__(self, window: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(
            name=f"RSI_{window}",
            description="RSI动量因子",
            direction=FactorDirection.NEGATIVE  # 反向因子（均值回归）
        )
        self.window = window
        self.overbought = overbought
        self.oversold = oversold
        self.metadata.parameters = {'window': window, 'overbought': overbought, 'oversold': oversold}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        rsi_val = rsi(close, self.window)
        
        # 转换为超买超卖信号（中心化处理）
        # 正值表示超卖（看涨），负值表示超买（看跌）
        signal = 50 - rsi_val
        
        return signal


class RSIDivergenceFactor(TechnicalFactor):
    """
    RSI背离因子
    
    价格与RSI的背离信号
    - 顶背离：看跌
    - 底背离：看涨
    """
    
    def __init__(self, window: int = 14, lookback: int = 5):
        super().__init__(
            name=f"RSI_Divergence_{window}_{lookback}",
            description="RSI背离因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        rsi_val = rsi(close, self.window)
        
        # 计算价格变化
        price_change = close.diff(self.lookback)
        rsi_change = rsi_val.diff(self.lookback)
        
        # 背离检测
        # 顶背离：价格新高，RSI未新高
        # 底背离：价格新低，RSI未新低
        divergence = np.where(
            (price_change > 0) & (rsi_change < 0), -1,  # 顶背离
            np.where(
                (price_change < 0) & (rsi_change > 0), 1,  # 底背离
                0
            )
        )
        
        return pd.Series(divergence, index=close.index)


class CCIFactor(TechnicalFactor):
    """
    CCI商品通道指数因子
    
    Commodity Channel Index
    - CCI > 100：超买
    - CCI < -100：超卖
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"CCI_{window}",
            description="CCI商品通道指数因子",
            direction=FactorDirection.NEGATIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # SMA of TP
        tp_sma = sma(tp, self.window)
        
        # 平均绝对偏差
        mean_dev = tp.rolling(window=self.window, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        # CCI计算
        cci = (tp - tp_sma) / (0.015 * mean_dev + 1e-8)
        
        return -cci  # 反向因子


class WilliamsRFactor(TechnicalFactor):
    """
    Williams %R因子
    
    威廉指标，与RSI类似但尺度相反
    - %R > -20：超买
    - %R < -80：超卖
    """
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"WilliamsR_{window}",
            description="Williams %R因子",
            direction=FactorDirection.NEGATIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        highest_high = rolling_max(high, self.window)
        lowest_low = rolling_min(low, self.window)
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
        
        # 转换为与价格同向的信号
        return williams_r + 50


class StochasticFactor(TechnicalFactor):
    """
    随机指标因子
    
    Stochastic Oscillator
    """
    
    def __init__(self, k_window: int = 14, d_window: int = 3, 
                 smooth_k: int = 3):
        super().__init__(
            name=f"Stochastic_{k_window}_{d_window}_{smooth_k}",
            description="随机指标因子",
            direction=FactorDirection.NEGATIVE
        )
        self.k_window = k_window
        self.d_window = d_window
        self.smooth_k = smooth_k
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        stoch = stochastic(high, low, close, self.k_window, self.d_window)
        
        # 使用K线与D线的差值
        k = stoch['k']
        d = stoch['d']
        
        # 超买超卖信号（中心化处理）
        signal = 50 - k
        
        return signal


class StochasticCrossFactor(TechnicalFactor):
    """
    随机指标交叉因子
    
    K线与D线的交叉信号
    """
    
    def __init__(self, k_window: int = 14, d_window: int = 3):
        super().__init__(
            name=f"Stochastic_Cross_{k_window}_{d_window}",
            description="随机指标交叉因子",
            direction=FactorDirection.POSITIVE
        )
        self.k_window = k_window
        self.d_window = d_window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        stoch = stochastic(high, low, close, self.k_window, self.d_window)
        k = stoch['k']
        d = stoch['d']
        
        # K-D差值
        diff = k - d
        
        return diff


class MomentumFactor(TechnicalFactor):
    """
    价格动量因子
    
    简单价格动量（N期收益率）
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"Momentum_{window}",
            description="价格动量因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        momentum = close.pct_change(self.window)
        return momentum


class RateOfChangeFactor(TechnicalFactor):
    """
    变化率因子
    
    Rate of Change (ROC)
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"ROC_{window}",
            description="变化率因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        roc = (close - close.shift(self.window)) / (close.shift(self.window) + 1e-8)
        return roc


class PriceAccelerationFactor(TechnicalFactor):
    """
    价格加速度因子
    
    动量的变化率（二阶导数）
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"Price_Acceleration_{window}",
            description="价格加速度因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        momentum = close.pct_change(self.window // 2)
        acceleration = momentum.diff(self.window // 2)
        return acceleration


# ==================== 波动类因子 ====================

class BollingerPositionFactor(TechnicalFactor):
    """
    布林带位置因子
    
    价格在布林带中的相对位置
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(
            name=f"BB_Position_{window}_{num_std}",
            description="布林带位置因子",
            direction=FactorDirection.NEGATIVE  # 均值回归
        )
        self.window = window
        self.num_std = num_std
        self.metadata.parameters = {'window': window, 'num_std': num_std}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        bb = bollinger_bands(close, self.window, self.num_std)
        
        # %B指标，中心化处理
        percent_b = bb['percent_b']
        signal = 0.5 - percent_b  # 中心化为0
        
        return signal


class BollingerWidthFactor(TechnicalFactor):
    """
    布林带宽度因子
    
    布林带宽度变化，反映波动率变化
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(
            name=f"BB_Width_{window}_{num_std}",
            description="布林带宽度因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.num_std = num_std
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        bb = bollinger_bands(close, self.window, self.num_std)
        
        bandwidth = bb['bandwidth']
        
        # 带宽变化率
        width_change = bandwidth.diff(5)
        
        return width_change


class BollingerSqueezeFactor(TechnicalFactor):
    """
    布林带挤压因子
    
    检测布林带挤压（低波动率）状态
    挤压后往往伴随大行情
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, 
                 lookback: int = 120):
        super().__init__(
            name=f"BB_Squeeze_{window}_{num_std}_{lookback}",
            description="布林带挤压因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.num_std = num_std
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        bb = bollinger_bands(close, self.window, self.num_std)
        
        bandwidth = bb['bandwidth']
        
        # 当前带宽相对于历史的位置
        bandwidth_rank = bandwidth.rolling(window=self.lookback, min_periods=1).apply(
            lambda x: stats.percentileofscore(x, x[-1]) / 100, raw=True
        )
        
        # 挤压程度（越低越挤压）
        squeeze = 0.5 - bandwidth_rank
        
        return squeeze


class ATRFactor(TechnicalFactor):
    """
    ATR波动率因子
    
    平均真实波幅
    """
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"ATR_{window}",
            description="ATR波动率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        atr_val = atr(high, low, close, self.window)
        
        # 标准化处理
        atr_normalized = atr_val / (close + 1e-8)
        
        return atr_normalized


class ATRRatioFactor(TechnicalFactor):
    """
    ATR比率因子
    
    当前ATR与历史ATR的比率
    """
    
    def __init__(self, window: int = 14, lookback: int = 100):
        super().__init__(
            name=f"ATR_Ratio_{window}_{lookback}",
            description="ATR比率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        atr_val = atr(high, low, close, self.window)
        atr_normalized = atr_val / (close + 1e-8)
        
        # 当前ATR与历史平均的比率
        atr_mean = atr_normalized.rolling(window=self.lookback, min_periods=1).mean()
        atr_ratio = atr_normalized / (atr_mean + 1e-8)
        
        return atr_ratio - 1  # 中心化


class KeltnerPositionFactor(TechnicalFactor):
    """
    Keltner通道位置因子
    
    基于ATR的通道指标
    """
    
    def __init__(self, ema_window: int = 20, atr_window: int = 14, 
                 atr_multiplier: float = 2.0):
        super().__init__(
            name=f"Keltner_{ema_window}_{atr_window}_{atr_multiplier}",
            description="Keltner通道位置因子",
            direction=FactorDirection.NEGATIVE
        )
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Keltner通道
        middle = ema(close, self.ema_window)
        atr_val = atr(high, low, close, self.atr_window)
        
        upper = middle + self.atr_multiplier * atr_val
        lower = middle - self.atr_multiplier * atr_val
        
        # 价格在通道中的位置
        position = (close - lower) / (upper - lower + 1e-8)
        signal = 0.5 - position
        
        return signal


class VolatilityRegimeFactor(TechnicalFactor):
    """
    波动率状态因子
    
    识别高波动/低波动状态
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__(
            name=f"Vol_Regime_{short_window}_{long_window}",
            description="波动率状态因子",
            direction=FactorDirection.NEUTRAL
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 计算收益率
        returns = close.pct_change()
        
        # 短期和长期波动率
        vol_short = returns.rolling(window=self.short_window, min_periods=1).std()
        vol_long = returns.rolling(window=self.long_window, min_periods=1).std()
        
        # 波动率比率
        vol_ratio = vol_short / (vol_long + 1e-8)
        
        return vol_ratio - 1


class DonchianChannelFactor(TechnicalFactor):
    """
    Donchian通道因子
    
    基于N期高低点的通道突破系统
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Donchian_{window}",
            description="Donchian通道因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Donchian通道
        upper = rolling_max(high, self.window)
        lower = rolling_min(low, self.window)
        middle = (upper + lower) / 2
        
        # 价格相对于通道的位置
        position = (close - middle) / (upper - lower + 1e-8)
        
        return position


class IchimokuFactor(TechnicalFactor):
    """
    一目均衡表因子
    
    Ichimoku Cloud简化版
    """
    
    def __init__(self, tenkan: int = 9, kijun: int = 26):
        super().__init__(
            name=f"Ichimoku_{tenkan}_{kijun}",
            description="一目均衡表因子",
            direction=FactorDirection.POSITIVE
        )
        self.tenkan = tenkan
        self.kijun = kijun
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (转换线)
        tenkan_sen = (rolling_max(high, self.tenkan) + rolling_min(low, self.tenkan)) / 2
        
        # Kijun-sen (基准线)
        kijun_sen = (rolling_max(high, self.kijun) + rolling_min(low, self.kijun)) / 2
        
        # 价格相对于两条线的位置
        price_vs_tenkan = (close - tenkan_sen) / (close + 1e-8)
        price_vs_kijun = (close - kijun_sen) / (close + 1e-8)
        
        # 综合信号
        signal = price_vs_tenkan + price_vs_kijun
        
        return signal


class ParabolicSARFactor(TechnicalFactor):
    """
    抛物线SAR因子
    
    Parabolic Stop and Reverse
    """
    
    def __init__(self, af: float = 0.02, max_af: float = 0.2):
        super().__init__(
            name=f"PSAR_{af}_{max_af}",
            description="抛物线SAR因子",
            direction=FactorDirection.POSITIVE
        )
        self.af = af
        self.max_af = max_af
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high'].values
        low = data['low'].values
        close = data['close']
        n = len(close)
        
        # 简化版PSAR实现
        psar = np.zeros(n)
        psar[0] = close.iloc[0]
        
        uptrend = True
        ep = high[0]  # 极值点
        af = self.af
        
        for i in range(1, n):
            if uptrend:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if low[i] < psar[i]:
                    uptrend = False
                    psar[i] = ep
                    ep = low[i]
                    af = self.af
                elif high[i] > ep:
                    ep = high[i]
                    af = min(af + self.af, self.max_af)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if high[i] > psar[i]:
                    uptrend = True
                    psar[i] = ep
                    ep = high[i]
                    af = self.af
                elif low[i] < ep:
                    ep = low[i]
                    af = min(af + self.af, self.max_af)
        
        psar_series = pd.Series(psar, index=close.index)
        
        # 价格相对于PSAR的位置
        signal = (close - psar_series) / (close + 1e-8)
        
        return signal


# ==================== 因子工厂函数 ====================

def create_all_technical_factors() -> List[TechnicalFactor]:
    """创建所有技术指标因子实例"""
    factors = []
    
    # 趋势类因子
    factors.extend([
        MACDFactor(12, 26, 9),
        MACDSignalFactor(12, 26, 9),
        EMAcrossFactor(12, 26),
        EMAcrossFactor(5, 20),
        SMAcrossFactor(10, 30),
        SMAcrossFactor(5, 60),
        ADXFactor(14),
        ADXFactor(20),
        TrendStrengthFactor([5, 10, 20, 60]),
    ])
    
    # 动量类因子
    factors.extend([
        RSIFactor(14),
        RSIFactor(7),
        RSIDivergenceFactor(14, 5),
        CCIFactor(20),
        CCIFactor(14),
        WilliamsRFactor(14),
        WilliamsRFactor(7),
        StochasticFactor(14, 3, 3),
        StochasticCrossFactor(14, 3),
        MomentumFactor(10),
        MomentumFactor(20),
        RateOfChangeFactor(10),
        RateOfChangeFactor(20),
        PriceAccelerationFactor(10),
    ])
    
    # 波动类因子
    factors.extend([
        BollingerPositionFactor(20, 2.0),
        BollingerPositionFactor(20, 2.5),
        BollingerWidthFactor(20, 2.0),
        BollingerSqueezeFactor(20, 2.0, 120),
        ATRFactor(14),
        ATRFactor(7),
        ATRRatioFactor(14, 100),
        KeltnerPositionFactor(20, 14, 2.0),
        VolatilityRegimeFactor(10, 30),
        DonchianChannelFactor(20),
        DonchianChannelFactor(10),
        IchimokuFactor(9, 26),
        ParabolicSARFactor(0.02, 0.2),
    ])
    
    return factors


# 导出所有因子类
__all__ = [
    'MACDFactor', 'MACDSignalFactor', 'EMAcrossFactor', 'SMAcrossFactor',
    'ADXFactor', 'TrendStrengthFactor',
    'RSIFactor', 'RSIDivergenceFactor', 'CCIFactor', 'WilliamsRFactor',
    'StochasticFactor', 'StochasticCrossFactor', 'MomentumFactor',
    'RateOfChangeFactor', 'PriceAccelerationFactor',
    'BollingerPositionFactor', 'BollingerWidthFactor', 'BollingerSqueezeFactor',
    'ATRFactor', 'ATRRatioFactor', 'KeltnerPositionFactor',
    'VolatilityRegimeFactor', 'DonchianChannelFactor', 'IchimokuFactor',
    'ParabolicSARFactor', 'create_all_technical_factors'
]
