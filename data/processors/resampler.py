"""
Data resampling processor for time series.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from ...core.base import BaseDataProcessor, DataFrequency


class DataResampler(BaseDataProcessor):
    """
    Resample time series data to different frequencies.
    """

    def __init__(
        self,
        name: str = "DataResampler",
        target_frequency: DataFrequency = DataFrequency.MINUTE_5,
        config: Dict[str, Any] = None,
    ):
        super().__init__(name, config)
        self.target_frequency = target_frequency

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to target frequency.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Resampled data
        """
        if data.empty:
            return data
        
        # Map DataFrequency to pandas frequency string
        freq_map = {
            DataFrequency.MINUTE_1: '1min',
            DataFrequency.MINUTE_5: '5min',
            DataFrequency.MINUTE_15: '15min',
            DataFrequency.MINUTE_30: '30min',
            DataFrequency.HOUR_1: '1H',
            DataFrequency.HOUR_4: '4H',
            DataFrequency.DAY_1: '1D',
        }
        
        target_freq = freq_map.get(self.target_frequency, '5min')
        
        # Resample using OHLCV aggregation
        resampled = data.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        return resampled

    def validate(self, data: pd.DataFrame) -> tuple:
        """Validate resampled data."""
        errors = []
        
        if data.empty:
            errors.append("Resampled data is empty")
        
        # Check for NaN values
        if data.isnull().any().any():
            errors.append("Resampled data contains NaN values")
        
        return len(errors) == 0, errors

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data (pass-through for resampler)."""
        return data

    def upsample(
        self,
        data: pd.DataFrame,
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        Upsample data to higher frequency.
        
        Args:
            data: Input data
            method: Interpolation method
            
        Returns:
            Upsampled data
        """
        if data.empty:
            return data
        
        freq_map = {
            DataFrequency.MINUTE_1: '1min',
            DataFrequency.MINUTE_5: '5min',
            DataFrequency.MINUTE_15: '15min',
            DataFrequency.MINUTE_30: '30min',
            DataFrequency.HOUR_1: '1H',
            DataFrequency.HOUR_4: '4H',
            DataFrequency.DAY_1: '1D',
        }
        
        target_freq = freq_map.get(self.target_frequency, '1min')
        
        # Create new index with target frequency
        new_index = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq=target_freq
        )
        
        # Reindex and interpolate
        upsampled = data.reindex(new_index)
        
        # Interpolate price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in upsampled.columns:
                upsampled[col] = upsampled[col].interpolate(method=method)
        
        # Forward fill volume
        if 'volume' in upsampled.columns:
            upsampled['volume'] = upsampled['volume'].fillna(0)
        
        return upsampled
