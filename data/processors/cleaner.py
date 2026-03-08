"""
Data cleaning and validation processor.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from ...core.base import BaseDataProcessor


class DataCleaner(BaseDataProcessor):
    """
    Data cleaning processor.
    
    Handles missing values, outliers, and data quality issues.
    """

    def __init__(
        self,
        name: str = "DataCleaner",
        fill_missing: bool = True,
        remove_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        config: Dict[str, Any] = None,
    ):
        super().__init__(name, config)
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Cleaned data
        """
        df = data.copy()
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Handle missing values
        if self.fill_missing:
            df = self._fill_missing_values(df)
        
        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Validate OHLCV relationships
        df = self._validate_ohlcv(df)
        
        return df

    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using forward fill and interpolation."""
        df = data.copy()
        
        # Forward fill for price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Fill volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Interpolate remaining missing values
        df = df.interpolate(method='linear', limit_direction='both')
        
        return df

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using specified method."""
        df = data.copy()
        
        if self.outlier_method == "iqr":
            df = self._remove_outliers_iqr(df)
        elif self.outlier_method == "zscore":
            df = self._remove_outliers_zscore(df)
        elif self.outlier_method == "mad":
            df = self._remove_outliers_mad(df)
        
        return df

    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        df = data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                # Clip outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df

    def _remove_outliers_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        df = data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df.loc[z_scores > self.outlier_threshold, col] = np.nan
                df[col] = df[col].ffill()
        
        return df

    def _remove_outliers_mad(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Median Absolute Deviation."""
        df = data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                modified_z = 0.6745 * (df[col] - median) / mad
                df.loc[np.abs(modified_z) > self.outlier_threshold, col] = np.nan
                df[col] = df[col].ffill()
        
        return df

    def _validate_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLCV relationships."""
        df = data.copy()
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Ensure high is the highest
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Ensure low is the lowest
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
            
            # Ensure all prices are positive
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].clip(lower=0.00000001)
        
        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        
        return df

    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check for empty data
        if data.empty:
            errors.append("Data is empty")
            return False, errors
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns and (data[col] < 0).any():
                errors.append(f"Negative values found in {col}")
        
        # Check OHLC relationships
        if all(col in data.columns for col in ['high', 'low']):
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                errors.append(f"Found {invalid_hl} bars where high < low")
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            errors.append("Found duplicate timestamps")
        
        return len(errors) == 0, errors

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data (min-max scaling).
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        df = data.copy()
        
        # Z-score normalization for price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f"{col}_norm"] = (df[col] - mean) / std
        
        # Log normalization for volume
        if 'volume' in df.columns:
            df['volume_norm'] = np.log1p(df['volume'])
        
        return df
