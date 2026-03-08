"""
Volume-based technical factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

from ...core.base import BaseFactor


class VolumeFactor(BaseFactor):
    """
    Volume-based factor with multiple metrics.
    """

    def __init__(
        self,
        name: str = "Volume",
        period: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute volume factor."""
        volume = data['volume']
        close = data['close']
        
        # Volume moving average
        volume_ma = volume.rolling(window=self.period).mean()
        
        # Relative volume
        relative_volume = volume / volume_ma
        
        # Price change
        price_change = close.pct_change()
        
        # Volume-price relationship
        volume_price = np.sign(price_change) * np.log1p(relative_volume)
        
        return volume_price

    def get_required_columns(self) -> List[str]:
        return ['volume', 'close']


class OBVFactor(BaseFactor):
    """
    On-Balance Volume factor.
    """

    def __init__(
        self,
        name: str = "OBV",
        period: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute OBV factor."""
        close = data['close']
        volume = data['volume']
        
        # Calculate OBV
        obv = ta.obv(close, volume)
        
        # Calculate OBV moving average
        obv_ma = obv.rolling(window=self.period).mean()
        
        # OBV relative to its MA
        factor = (obv - obv_ma) / obv_ma.abs()
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']


class VWAPFactor(BaseFactor):
    """
    Volume Weighted Average Price factor.
    """

    def __init__(
        self,
        name: str = "VWAP",
        period: int = 20,
        anchor: str = "D",  # 'D' for daily, 'W' for weekly
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.anchor = anchor

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute VWAP factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate VWAP
        vwap = ta.vwap(high, low, close, volume, anchor=self.anchor)
        
        # Price relative to VWAP
        factor = (close - vwap) / close
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']


class MFI_Factor(BaseFactor):
    """
    Money Flow Index factor.
    """

    def __init__(
        self,
        name: str = "MFI",
        period: int = 14,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute MFI factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate MFI
        mfi = ta.mfi(high, low, close, volume, length=self.period)
        
        # Convert to factor (-1 to 1 range)
        factor = (mfi - 50) / 50
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']


class VolumeProfileFactor(BaseFactor):
    """
    Volume Profile factor.
    
    Analyzes volume distribution across price levels.
    """

    def __init__(
        self,
        name: str = "VolumeProfile",
        period: int = 50,
        bins: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.bins = bins

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Volume Profile factor."""
        close = data['close']
        volume = data['volume']
        
        # Calculate rolling volume-weighted price position
        def calc_position(window):
            if len(window) < self.period:
                return 0
            
            prices = window['close'].values
            volumes = window['volume'].values
            
            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            if price_min == price_max:
                return 0
            
            bins = np.linspace(price_min, price_max, self.bins)
            
            # Calculate volume in each bin
            volume_profile = np.zeros(self.bins - 1)
            for i in range(self.bins - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                volume_profile[i] = volumes[mask].sum()
            
            # Find Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Current price relative to POC
            current_price = prices[-1]
            factor = (current_price - poc_price) / (price_max - price_min)
            
            return factor
        
        # Apply rolling calculation
        factor = pd.Series(0.0, index=data.index)
        for i in range(self.period, len(data)):
            window = data.iloc[i - self.period:i]
            factor.iloc[i] = calc_position(window)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']


class VolumeDeltaFactor(BaseFactor):
    """
    Volume Delta factor.
    
    Estimates buying vs selling pressure.
    """

    def __init__(
        self,
        name: str = "VolumeDelta",
        period: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Volume Delta factor."""
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Estimate buying/selling volume using candlestick analysis
        # Higher close relative to range = more buying pressure
        candle_range = high - low
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, np.nan)
        
        # Buying pressure (0 to 1)
        buying_pressure = (close - low) / candle_range
        buying_pressure = buying_pressure.fillna(0.5)
        
        # Selling pressure
        selling_pressure = (high - close) / candle_range
        selling_pressure = selling_pressure.fillna(0.5)
        
        # Volume delta
        volume_delta = volume * (buying_pressure - selling_pressure)
        
        # Normalize by average volume
        avg_volume = volume.rolling(window=self.period).mean()
        factor = volume_delta / avg_volume
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
