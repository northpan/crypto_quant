"""
量价因子模块
Volume-Price Factors Module

包含OBV、MFI、VWAP等量价关系因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .base_factor import (
    VolumeFactor, FactorDirection,
    ema, sma, rolling_std, rolling_max, rolling_min, rolling_sum,
    safe_divide
)


# ==================== OBV类因子 ====================

class OBVFactor(VolumeFactor):
    """
    OBV能量潮因子
    
    On-Balance Volume
    累积成交量指标，反映资金流向
    - OBV上升：资金流入
    - OBV下降：资金流出
    """
    
    def __init__(self, smooth: int = 1):
        super().__init__(
            name=f"OBV_{smooth}",
            description="OBV能量潮因子",
            direction=FactorDirection.POSITIVE
        )
        self.smooth = smooth
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 计算价格变化方向
        price_change = close.diff()
        
        # OBV计算
        obv = pd.Series(0, index=close.index)
        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]
        obv = obv.cumsum()
        
        # 平滑处理
        if self.smooth > 1:
            obv = ema(obv, self.smooth)
        
        # 标准化
        obv_normalized = obv / (volume.rolling(window=20, min_periods=1).mean() + 1e-8)
        
        return obv_normalized


class OBVVelocityFactor(VolumeFactor):
    """
    OBV速度因子
    
    OBV的变化速度，反映资金流入流出加速度
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"OBV_Velocity_{window}",
            description="OBV速度因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 计算OBV
        price_change = close.diff()
        obv = pd.Series(0, index=close.index)
        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]
        obv = obv.cumsum()
        
        # OBV变化速度
        obv_velocity = obv.diff(self.window)
        
        # 标准化
        obv_velocity_norm = obv_velocity / (volume.rolling(window=self.window, min_periods=1).mean() + 1e-8)
        
        return obv_velocity_norm


class OBVDivergenceFactor(VolumeFactor):
    """
    OBV背离因子
    
    价格与OBV的背离信号
    - 顶背离：看跌
    - 底背离：看涨
    """
    
    def __init__(self, lookback: int = 10):
        super().__init__(
            name=f"OBV_Divergence_{lookback}",
            description="OBV背离因子",
            direction=FactorDirection.POSITIVE
        )
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 计算OBV
        price_change = close.diff()
        obv = pd.Series(0, index=close.index)
        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]
        obv = obv.cumsum()
        
        # 计算变化
        price_change_period = close.diff(self.lookback)
        obv_change = obv.diff(self.lookback)
        
        # 背离检测
        divergence = np.where(
            (price_change_period > 0) & (obv_change < 0), -1,  # 顶背离
            np.where(
                (price_change_period < 0) & (obv_change > 0), 1,  # 底背离
                0
            )
        )
        
        return pd.Series(divergence, index=close.index)


# ==================== MFI类因子 ====================

class MFIFactor(VolumeFactor):
    """
    MFI资金流量指数因子
    
    Money Flow Index
    类似RSI但考虑成交量加权
    - MFI > 80：超买
    - MFI < 20：超卖
    """
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"MFI_{window}",
            description="MFI资金流量指数因子",
            direction=FactorDirection.NEGATIVE  # 反向因子
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # 原始资金流量
        raw_mf = tp * volume
        
        # 资金流量方向
        tp_diff = tp.diff()
        positive_mf = pd.Series(0, index=close.index)
        negative_mf = pd.Series(0, index=close.index)
        
        positive_mf[tp_diff > 0] = raw_mf[tp_diff > 0]
        negative_mf[tp_diff < 0] = raw_mf[tp_diff < 0]
        
        # 滚动求和
        positive_sum = positive_mf.rolling(window=self.window, min_periods=1).sum()
        negative_sum = negative_mf.rolling(window=self.window, min_periods=1).sum()
        
        # MFI计算
        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-8)))
        
        # 转换为超买超卖信号
        signal = 50 - mfi
        
        return signal


class MFIVelocityFactor(VolumeFactor):
    """
    MFI速度因子
    
    MFI的变化速度
    """
    
    def __init__(self, window: int = 14, velocity_window: int = 5):
        super().__init__(
            name=f"MFI_Velocity_{window}_{velocity_window}",
            description="MFI速度因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.velocity_window = velocity_window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 计算MFI
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        tp_diff = tp.diff()
        
        positive_mf = pd.Series(0, index=close.index)
        negative_mf = pd.Series(0, index=close.index)
        positive_mf[tp_diff > 0] = raw_mf[tp_diff > 0]
        negative_mf[tp_diff < 0] = raw_mf[tp_diff < 0]
        
        positive_sum = positive_mf.rolling(window=self.window, min_periods=1).sum()
        negative_sum = negative_mf.rolling(window=self.window, min_periods=1).sum()
        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-8)))
        
        # MFI变化速度
        mfi_velocity = mfi.diff(self.velocity_window)
        
        return mfi_velocity


# ==================== VWAP类因子 ====================

class VWAPFactor(VolumeFactor):
    """
    VWAP成交量加权平均价因子
    
    Volume Weighted Average Price
    价格相对于VWAP的位置
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"VWAP_{window}",
            description="VWAP成交量加权平均价因子",
            direction=FactorDirection.NEGATIVE  # 均值回归
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # VWAP
        vwap = (tp * volume).rolling(window=self.window, min_periods=1).sum() / \
               volume.rolling(window=self.window, min_periods=1).sum()
        
        # 价格相对于VWAP的偏离
        deviation = (close - vwap) / (vwap + 1e-8)
        
        return deviation


class VWAPDeviationFactor(VolumeFactor):
    """
    VWAP偏离度因子
    
    价格偏离VWAP的标准差倍数
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"VWAP_Deviation_{window}",
            description="VWAP偏离度因子",
            direction=FactorDirection.NEGATIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # VWAP
        vwap = (tp * volume).rolling(window=self.window, min_periods=1).sum() / \
               volume.rolling(window=self.window, min_periods=1).sum()
        
        # 计算偏离度的标准差
        deviation = close - vwap
        deviation_std = deviation.rolling(window=self.window, min_periods=1).std()
        
        # Z-score
        zscore = deviation / (deviation_std + 1e-8)
        
        return -zscore  # 反向因子


class VWAPTrendFactor(VolumeFactor):
    """
    VWAP趋势因子
    
    VWAP的斜率方向
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"VWAP_Trend_{window}",
            description="VWAP趋势因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # VWAP
        vwap = (tp * volume).rolling(window=self.window, min_periods=1).sum() / \
               volume.rolling(window=self.window, min_periods=1).sum()
        
        # VWAP趋势
        vwap_sma = sma(vwap, self.window)
        vwap_slope = (vwap - vwap_sma) / (vwap_sma + 1e-8)
        
        return vwap_slope


# ==================== 成交量比率类因子 ====================

class VolumeRatioFactor(VolumeFactor):
    """
    成交量比率因子
    
    当前成交量与历史平均的比率
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Volume_Ratio_{window}",
            description="成交量比率因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        
        # 当前成交量
        current_vol = volume
        
        # 历史平均成交量
        avg_vol = volume.rolling(window=self.window, min_periods=1).mean()
        
        # 比率
        ratio = current_vol / (avg_vol + 1e-8)
        
        return ratio - 1  # 中心化


class VolumeTrendFactor(VolumeFactor):
    """
    成交量趋势因子
    
    成交量的趋势方向
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(
            name=f"Volume_Trend_{short_window}_{long_window}",
            description="成交量趋势因子",
            direction=FactorDirection.NEUTRAL
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        
        # 短期和长期成交量均线
        vol_short = sma(volume, self.short_window)
        vol_long = sma(volume, self.long_window)
        
        # 成交量趋势
        trend = (vol_short - vol_long) / (vol_long + 1e-8)
        
        return trend


class VolumeOscillatorFactor(VolumeFactor):
    """
    成交量振荡器因子
    
    短期与长期成交量均线的差值
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(
            name=f"Volume_Oscillator_{short_window}_{long_window}",
            description="成交量振荡器因子",
            direction=FactorDirection.NEUTRAL
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        
        # 成交量均线
        vol_short = sma(volume, self.short_window)
        vol_long = sma(volume, self.long_window)
        
        # 振荡器
        oscillator = vol_short - vol_long
        
        # 标准化
        oscillator_norm = oscillator / (vol_long + 1e-8)
        
        return oscillator_norm


# ==================== 量价背离类因子 ====================

class VolumePriceDivergenceFactor(VolumeFactor):
    """
    量价背离因子
    
    价格与成交量的背离信号
    - 价涨量缩：看跌
    - 价跌量缩：看涨
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"VP_Divergence_{window}",
            description="量价背离因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化
        price_change = close.pct_change(self.window)
        
        # 成交量变化
        volume_change = volume.pct_change(self.window)
        
        # 量价背离检测
        divergence = np.where(
            (price_change > 0) & (volume_change < 0), -1,  # 价涨量缩（看跌）
            np.where(
                (price_change < 0) & (volume_change < 0), 1,  # 价跌量缩（看涨）
                np.where(
                    (price_change > 0) & (volume_change > 0), 0.5,  # 价涨量增（看涨）
                    np.where(
                        (price_change < 0) & (volume_change > 0), -0.5,  # 价跌量增（看跌）
                        0
                    )
                )
            )
        )
        
        return pd.Series(divergence, index=close.index)


class VolumePriceTrendFactor(VolumeFactor):
    """
    量价趋势因子
    
    Volume-Price Trend (VPT)
    """
    
    def __init__(self, window: int = 1):
        super().__init__(
            name=f"VPT_{window}",
            description="量价趋势因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化率
        price_change = close.pct_change()
        
        # VPT
        vpt = (volume * price_change).cumsum()
        
        # 平滑
        if self.window > 1:
            vpt = ema(vpt, self.window)
        
        # 标准化
        vpt_normalized = vpt / (volume.rolling(window=20, min_periods=1).mean() + 1e-8)
        
        return vpt_normalized


class VolumePriceConfirmationFactor(VolumeFactor):
    """
    量价确认因子
    
    价格突破是否伴随成交量确认
    """
    
    def __init__(self, window: int = 20, volume_threshold: float = 1.5):
        super().__init__(
            name=f"VP_Confirmation_{window}_{volume_threshold}",
            description="量价确认因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.volume_threshold = volume_threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # 计算高低点
        highest = rolling_max(high, self.window)
        lowest = rolling_min(low, self.window)
        
        # 成交量均值
        avg_volume = volume.rolling(window=self.window, min_periods=1).mean()
        
        # 突破信号
        breakout_up = (close > highest.shift(1)) & (volume > self.volume_threshold * avg_volume)
        breakout_down = (close < lowest.shift(1)) & (volume > self.volume_threshold * avg_volume)
        
        # 确认信号
        confirmation = pd.Series(0, index=close.index)
        confirmation[breakout_up] = 1
        confirmation[breakout_down] = -1
        
        return confirmation


# ==================== 资金流入流出类因子 ====================

class MoneyFlowFactor(VolumeFactor):
    """
    资金流向因子
    
    基于典型价格和成交量的资金流向
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Money_Flow_{window}",
            description="资金流向因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # 资金流向
        money_flow = tp * volume
        
        # 价格相对于高低点的位置
        position = (close - low) / (high - low + 1e-8)
        
        # 加权资金流向
        weighted_mf = money_flow * (2 * position - 1)  # 映射到[-1, 1]
        
        # 累积
        mf_cumsum = weighted_mf.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        mf_normalized = mf_cumsum / (volume.rolling(window=self.window, min_periods=1).sum() + 1e-8)
        
        return mf_normalized


class BuyingPressureFactor(VolumeFactor):
    """
    买盘压力因子
    
    基于收盘价位置的买盘压力估计
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Buying_Pressure_{window}",
            description="买盘压力因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 买盘压力估计
        buying_pressure = (close - low) / (high - low + 1e-8) * volume
        selling_pressure = (high - close) / (high - low + 1e-8) * volume
        
        # 净压力
        net_pressure = buying_pressure - selling_pressure
        
        # 累积
        pressure_cumsum = net_pressure.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        total_volume = volume.rolling(window=self.window, min_periods=1).sum()
        pressure_normalized = pressure_cumsum / (total_volume + 1e-8)
        
        return pressure_normalized


class AccumulationDistributionFactor(VolumeFactor):
    """
    累积/派发因子
    
    A/D Line
    """
    
    def __init__(self, window: int = 1):
        super().__init__(
            name=f"AD_Line_{window}",
            description="累积/派发因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 资金流乘数
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
        
        # 资金流量
        mf_volume = mf_multiplier * volume
        
        # A/D线
        ad_line = mf_volume.cumsum()
        
        # 平滑
        if self.window > 1:
            ad_line = ema(ad_line, self.window)
        
        # 标准化
        ad_normalized = ad_line / (volume.rolling(window=20, min_periods=1).mean() + 1e-8)
        
        return ad_normalized


class ChaikinOscillatorFactor(VolumeFactor):
    """
    Chaikin振荡器因子
    
    A/D线的MACD
    """
    
    def __init__(self, fast: int = 3, slow: int = 10):
        super().__init__(
            name=f"Chaikin_Osc_{fast}_{slow}",
            description="Chaikin振荡器因子",
            direction=FactorDirection.POSITIVE
        )
        self.fast = fast
        self.slow = slow
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # A/D线
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
        mf_volume = mf_multiplier * volume
        ad_line = mf_volume.cumsum()
        
        # Chaikin振荡器
        ad_ema_fast = ema(ad_line, self.fast)
        ad_ema_slow = ema(ad_line, self.slow)
        chaikin_osc = ad_ema_fast - ad_ema_slow
        
        # 标准化
        chaikin_normalized = chaikin_osc / (volume.rolling(window=20, min_periods=1).mean() + 1e-8)
        
        return chaikin_normalized


class ForceIndexFactor(VolumeFactor):
    """
    力量指数因子
    
    Force Index
    """
    
    def __init__(self, window: int = 13):
        super().__init__(
            name=f"Force_Index_{window}",
            description="力量指数因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 力量指数
        force_index = (close - close.shift(1)) * volume
        
        # 平滑
        force_index_ema = ema(force_index, self.window)
        
        # 标准化
        force_normalized = force_index_ema / (volume.rolling(window=self.window, min_periods=1).mean() + 1e-8)
        
        return force_normalized


class EaseOfMovementFactor(VolumeFactor):
    """
    简易波动指标因子
    
    Ease of Movement (EOM)
    """
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"EOM_{window}",
            description="简易波动指标因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 中点移动
        mid_point_move = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # 高低点距离
        box_ratio = volume / (high - low + 1e-8)
        
        # EOM
        eom = mid_point_move / (box_ratio + 1e-8)
        
        # 平滑
        eom_sma = sma(eom, self.window)
        
        return eom_sma


class NegativeVolumeIndexFactor(VolumeFactor):
    """
    负成交量指数因子
    
    Negative Volume Index (NVI)
    """
    
    def __init__(self):
        super().__init__(
            name="NVI",
            description="负成交量指数因子",
            direction=FactorDirection.POSITIVE
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化率
        price_change = close.pct_change()
        
        # 成交量下降的日子
        volume_decrease = volume < volume.shift(1)
        
        # NVI
        nvi = pd.Series(1000, index=close.index)
        for i in range(1, len(close)):
            if volume_decrease.iloc[i]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        # 标准化（变化率）
        nvi_change = nvi.pct_change(20)
        
        return nvi_change


class PositiveVolumeIndexFactor(VolumeFactor):
    """
    正成交量指数因子
    
    Positive Volume Index (PVI)
    """
    
    def __init__(self):
        super().__init__(
            name="PVI",
            description="正成交量指数因子",
            direction=FactorDirection.POSITIVE
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化率
        price_change = close.pct_change()
        
        # 成交量上升的日子
        volume_increase = volume > volume.shift(1)
        
        # PVI
        pvi = pd.Series(1000, index=close.index)
        for i in range(1, len(close)):
            if volume_increase.iloc[i]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change.iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        # 标准化（变化率）
        pvi_change = pvi.pct_change(20)
        
        return pvi_change


# ==================== 因子工厂函数 ====================

def create_all_volume_factors() -> List[VolumeFactor]:
    """创建所有量价因子实例"""
    factors = []
    
    # OBV类因子
    factors.extend([
        OBVFactor(1),
        OBVFactor(5),
        OBVVelocityFactor(10),
        OBVDivergenceFactor(10),
    ])
    
    # MFI类因子
    factors.extend([
        MFIFactor(14),
        MFIFactor(7),
        MFIVelocityFactor(14, 5),
    ])
    
    # VWAP类因子
    factors.extend([
        VWAPFactor(20),
        VWAPFactor(10),
        VWAPDeviationFactor(20),
        VWAPTrendFactor(20),
    ])
    
    # 成交量比率类因子
    factors.extend([
        VolumeRatioFactor(20),
        VolumeRatioFactor(10),
        VolumeTrendFactor(5, 20),
        VolumeOscillatorFactor(5, 20),
    ])
    
    # 量价背离类因子
    factors.extend([
        VolumePriceDivergenceFactor(10),
        VolumePriceTrendFactor(1),
        VolumePriceConfirmationFactor(20, 1.5),
    ])
    
    # 资金流入流出类因子
    factors.extend([
        MoneyFlowFactor(20),
        BuyingPressureFactor(20),
        AccumulationDistributionFactor(1),
        ChaikinOscillatorFactor(3, 10),
        ForceIndexFactor(13),
        EaseOfMovementFactor(14),
        NegativeVolumeIndexFactor(),
        PositiveVolumeIndexFactor(),
    ])
    
    return factors


# 导出所有因子类
__all__ = [
    'OBVFactor', 'OBVVelocityFactor', 'OBVDivergenceFactor',
    'MFIFactor', 'MFIVelocityFactor',
    'VWAPFactor', 'VWAPDeviationFactor', 'VWAPTrendFactor',
    'VolumeRatioFactor', 'VolumeTrendFactor', 'VolumeOscillatorFactor',
    'VolumePriceDivergenceFactor', 'VolumePriceTrendFactor', 'VolumePriceConfirmationFactor',
    'MoneyFlowFactor', 'BuyingPressureFactor', 'AccumulationDistributionFactor',
    'ChaikinOscillatorFactor', 'ForceIndexFactor', 'EaseOfMovementFactor',
    'NegativeVolumeIndexFactor', 'PositiveVolumeIndexFactor',
    'create_all_volume_factors'
]
