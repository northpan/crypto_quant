"""
Momentum-based technical factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

from ...core.base import BaseFactor


class MomentumFactor(BaseFactor):
    """
    Price momentum factor.
    """

    def __init__(
        self,
        name: str = "Momentum",
        periods: List[int] = [5, 10, 20],
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, max(periods), config)
        self.periods = periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute momentum factor."""
        close = data['close']
        
        momentum_values = []
        weights = []
        
        for period in self.periods:
            # Calculate momentum
            mom = close.pct_change(period)
            momentum_values.append(mom)
            weights.append(1 / period)
        
        # Weighted average of momentum values
        weights = np.array(weights) / sum(weights)
        
        factor = pd.Series(0.0, index=data.index)
        for i, mom in enumerate(momentum_values):
            factor += mom * weights[i]
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class StochasticFactor(BaseFactor):
    """
    Stochastic Oscillator factor.
    """

    def __init__(
        self,
        name: str = "Stochastic",
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, k_period + d_period + smooth_k, config)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Stochastic factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Stochastic
        stoch = ta.stoch(
            high, low, close,
            k=self.k_period,
            d=self.d_period,
            smooth_k=self.smooth_k
        )
        
        if stoch is None:
            return pd.Series(0.0, index=data.index)
        
        # Use %K line
        k_line = stoch[f"STOCHk_{self.k_period}_{self.d_period}_{self.smooth_k}"]
        
        # Convert to factor (-1 to 1 range)
        factor = (k_line - 50) / 50
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class CCIFactor(BaseFactor):
    """
    Commodity Channel Index factor.
    """

    def __init__(
        self,
        name: str = "CCI",
        period: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute CCI factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate CCI
        cci = ta.cci(high, low, close, length=self.period)
        
        # Normalize to -1 to 1 range (CCI typically -200 to 200)
        factor = cci / 200
        factor = factor.clip(-1, 1)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class WilliamsRFactor(BaseFactor):
    """
    Williams %R factor.
    """

    def __init__(
        self,
        name: str = "WilliamsR",
        period: int = 14,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Williams %R factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Williams %R
        willr = ta.willr(high, low, close, length=self.period)
        
        # Williams %R is already -100 to 0, convert to -1 to 1
        factor = willr / 50
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class ROCFactor(BaseFactor):
    """
    Rate of Change factor.
    """

    def __init__(
        self,
        name: str = "ROC",
        period: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute ROC factor."""
        close = data['close']
        
        # Calculate ROC
        roc = ta.roc(close, length=self.period)
        
        # Normalize (clip extreme values)
        factor = roc / 100
        factor = factor.clip(-0.5, 0.5) * 2  # Scale to -1 to 1
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class TSIFactor(BaseFactor):
    """
    True Strength Index factor.
    """

    def __init__(
        self,
        name: str = "TSI",
        long_period: int = 25,
        short_period: int = 13,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, long_period + short_period, config)
        self.long_period = long_period
        self.short_period = short_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute TSI factor."""
        close = data['close']
        
        # Calculate TSI
        tsi = ta.tsi(close, long=self.long_period, short=self.short_period)
        
        if tsi is None:
            return pd.Series(0.0, index=data.index)
        
        # TSI is already normalized (-100 to 100)
        factor = tsi / 100
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']


class AwesomeOscillatorFactor(BaseFactor):
    """
    Awesome Oscillator factor.
    """

    def __init__(
        self,
        name: str = "AO",
        fast_period: int = 5,
        slow_period: int = 34,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, slow_period, config)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Awesome Oscillator factor."""
        high = data['high']
        low = data['low']
        
        # Calculate Awesome Oscillator
        ao = ta.ao(high, low, fast=self.fast_period, slow=self.slow_period)
        
        # Normalize by price
        median_price = (high + low) / 2
        factor = ao / median_price
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low']


class KDJFactor(BaseFactor):
    """
    KDJ (Stochastic RSI) factor.
    
    Popular in Asian markets.
    """

    def __init__(
        self,
        name: str = "KDJ",
        period: int = 9,
        k_smooth: int = 3,
        d_smooth: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period + k_smooth + d_smooth, config)
        self.period = period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute KDJ factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate RSV (Raw Stochastic Value)
        lowest_low = low.rolling(window=self.period).min()
        highest_high = high.rolling(window=self.period).max()
        
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)
        rsv = rsv.fillna(50)
        
        # Calculate K, D, J
        k = rsv.ewm(com=self.k_smooth - 1, adjust=False).mean()
        d = k.ewm(com=self.d_smooth - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        
        # Use J line as factor (more sensitive)
        factor = (j - 50) / 50
        factor = factor.clip(-1, 1)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
