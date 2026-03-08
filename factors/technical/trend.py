"""
Trend-based technical factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

from ...core.base import BaseFactor


class MovingAverageFactor(BaseFactor):
    """
    Moving Average factor with multiple periods.
    
    Generates trend signals based on price relative to moving averages.
    """

    def __init__(
        self,
        name: str = "MA",
        periods: List[int] = [5, 10, 20, 50, 200],
        ma_type: str = "ema",  # 'sma', 'ema', 'wma'
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, max(periods), config)
        self.periods = periods
        self.ma_type = ma_type

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute moving average factor."""
        df = data.copy()
        close = df['close']
        
        ma_values = {}
        
        for period in self.periods:
            if self.ma_type == "sma":
                ma = ta.sma(close, length=period)
            elif self.ma_type == "ema":
                ma = ta.ema(close, length=period)
            elif self.ma_type == "wma":
                ma = ta.wma(close, length=period)
            else:
                ma = ta.sma(close, length=period)
            
            ma_values[f"ma_{period}"] = ma
            
            # Price relative to MA
            df[f"price_to_ma_{period}"] = close / ma - 1
        
        # Composite factor: weighted average of price-to-MA ratios
        weights = np.array([1 / p for p in self.periods])
        weights = weights / weights.sum()
        
        factor = pd.Series(0.0, index=df.index)
        for i, period in enumerate(self.periods):
            factor += df[f"price_to_ma_{period}"] * weights[i]
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class MACDFactor(BaseFactor):
    """
    MACD (Moving Average Convergence Divergence) factor.
    """

    def __init__(
        self,
        name: str = "MACD",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, slow + signal, config)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute MACD factor."""
        close = data['close']
        
        # Calculate MACD
        macd_result = ta.macd(
            close,
            fast=self.fast,
            slow=self.slow,
            signal=self.signal
        )
        
        if macd_result is None:
            return pd.Series(0.0, index=data.index)
        
        # Use MACD histogram as factor
        histogram = macd_result[f"MACDh_{self.fast}_{self.slow}_{self.signal}"]
        
        # Normalize by price
        normalized = histogram / close
        
        return normalized

    def get_required_columns(self) -> List[str]:
        return ['close']


class RSIFactor(BaseFactor):
    """
    Relative Strength Index factor.
    """

    def __init__(
        self,
        name: str = "RSI",
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute RSI factor."""
        close = data['close']
        
        # Calculate RSI
        rsi = ta.rsi(close, length=self.period)
        
        # Convert to factor (-1 to 1 range)
        # Center at 50, scale to -1 to 1
        factor = (rsi - 50) / 50
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class BollingerBandsFactor(BaseFactor):
    """
    Bollinger Bands factor.
    """

    def __init__(
        self,
        name: str = "BB",
        period: int = 20,
        std_dev: float = 2.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.std_dev = std_dev

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Bollinger Bands factor."""
        close = data['close']
        
        # Calculate Bollinger Bands
        bb = ta.bbands(close, length=self.period, std=self.std_dev)
        
        if bb is None:
            return pd.Series(0.0, index=data.index)
        
        # Get band values
        lower = bb[f"BBL_{self.period}_{self.std_dev}.0"]
        middle = bb[f"BBM_{self.period}_{self.std_dev}.0"]
        upper = bb[f"BBU_{self.period}_{self.std_dev}.0"]
        
        # Calculate %B (position within bands)
        percent_b = (close - lower) / (upper - lower)
        
        # Convert to factor (-1 to 1 range)
        factor = 2 * (percent_b - 0.5)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class ADXFactor(BaseFactor):
    """
    Average Directional Index factor.
    
    Measures trend strength.
    """

    def __init__(
        self,
        name: str = "ADX",
        period: int = 14,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute ADX factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ADX
        adx_result = ta.adx(high, low, close, length=self.period)
        
        if adx_result is None:
            return pd.Series(0.0, index=data.index)
        
        adx = adx_result[f"ADX_{self.period}"]
        
        # Normalize to 0-1 range (ADX typically 0-100)
        factor = adx / 100
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class IchimokuFactor(BaseFactor):
    """
    Ichimoku Cloud factor.
    """

    def __init__(
        self,
        name: str = "Ichimoku",
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, max(tenkan, kijun, senkou), config)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Ichimoku factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Ichimoku
        ichi = ta.ichimoku(high, low, close)
        
        if ichi is None or len(ichi) < 2:
            return pd.Series(0.0, index=data.index)
        
        # Use the first DataFrame (ISA, ISB, ITS, IKS, ICS)
        ichi_df = ichi[0]
        
        # Calculate cloud position factor
        tenkan_sen = ichi_df[f"ITS_{self.tenkan}"]
        kijun_sen = ichi_df[f"IKS_{self.kijun}"]
        
        # Price relative to cloud
        factor = (close - tenkan_sen) / close
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
