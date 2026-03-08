"""
Volatility-based technical factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

from ...core.base import BaseFactor


class ATRFactor(BaseFactor):
    """
    Average True Range factor.
    
    Measures market volatility.
    """

    def __init__(
        self,
        name: str = "ATR",
        period: int = 14,
        normalize: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.normalize = normalize

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute ATR factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ATR
        atr = ta.atr(high, low, close, length=self.period)
        
        if self.normalize:
            # Normalize by close price
            factor = atr / close
        else:
            factor = atr
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class VolatilityFactor(BaseFactor):
    """
    Historical volatility factor.
    """

    def __init__(
        self,
        name: str = "Volatility",
        period: int = 20,
        annualize: bool = True,
        trading_periods: int = 365 * 24 * 60,  # For minute data
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.annualize = annualize
        self.trading_periods = trading_periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute volatility factor."""
        close = data['close']
        
        # Calculate log returns
        log_returns = np.log(close / close.shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=self.period).std()
        
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_periods)
        
        return volatility

    def get_required_columns(self) -> List[str]:
        return ['close']


class KeltnerChannelsFactor(BaseFactor):
    """
    Keltner Channels factor.
    """

    def __init__(
        self,
        name: str = "KC",
        period: int = 20,
        multiplier: float = 2.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period
        self.multiplier = multiplier

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Keltner Channels factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Keltner Channels
        kc = ta.kc(high, low, close, length=self.period, scalar=self.multiplier)
        
        if kc is None:
            return pd.Series(0.0, index=data.index)
        
        # Get channel values
        lower = kc[f"KCL_{self.period}_{self.multiplier}"]
        upper = kc[f"KCU_{self.period}_{self.multiplier}"]
        
        # Calculate position within channels
        position = (close - lower) / (upper - lower)
        
        # Convert to factor (-1 to 1 range)
        factor = 2 * (position - 0.5)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class DonchianChannelsFactor(BaseFactor):
    """
    Donchian Channels factor.
    """

    def __init__(
        self,
        name: str = "DC",
        period: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Donchian Channels factor."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Donchian Channels
        dc = ta.donchian(high, low, lower_length=self.period, upper_length=self.period)
        
        if dc is None:
            return pd.Series(0.0, index=data.index)
        
        # Get channel values
        lower = dc[f"DCL_{self.period}_{self.period}"]
        upper = dc[f"DCU_{self.period}_{self.period}"]
        
        # Calculate position within channels
        position = (close - lower) / (upper - lower)
        
        # Convert to factor (-1 to 1 range)
        factor = 2 * (position - 0.5)
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']


class UlcerIndexFactor(BaseFactor):
    """
    Ulcer Index factor.
    
    Measures downside risk/volatility.
    """

    def __init__(
        self,
        name: str = "UlcerIndex",
        period: int = 14,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, period, config)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Ulcer Index factor."""
        close = data['close']
        
        # Calculate Ulcer Index
        ui = ta.ui(close, length=self.period)
        
        # Normalize (UI typically 0-100)
        factor = ui / 100
        
        return factor

    def get_required_columns(self) -> List[str]:
        return ['close']
