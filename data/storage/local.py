"""
Local file storage for market data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class LocalStorage:
    """
    Local file-based storage for OHLCV data.
    
    Supports Parquet, CSV, and HDF5 formats.
    """

    SUPPORTED_FORMATS = ['parquet', 'csv', 'hdf5', 'feather']

    def __init__(
        self,
        base_path: str,
        format: str = 'parquet',
        compression: str = 'zstd',
    ):
        self.base_path = Path(base_path)
        self.format = format.lower()
        self.compression = compression
        
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> Path:
        """Generate file path for symbol data."""
        # Clean symbol name for filename
        clean_symbol = symbol.replace('/', '_').replace('-', '_')
        
        # Create directory structure: base/market_type/frequency/
        dir_path = self.base_path / market_type / frequency
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # File name: symbol.format
        file_name = f"{clean_symbol}.{self.format}"
        
        return dir_path / file_name

    def save(
        self,
        data: pd.DataFrame,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> bool:
        """
        Save data to local storage.
        
        Args:
            data: OHLCV data to save
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type (spot/perpetual)
            
        Returns:
            True if successful
        """
        if data.empty:
            print(f"Warning: Empty data for {symbol}")
            return False
        
        file_path = self._get_file_path(symbol, frequency, market_type)
        
        try:
            if self.format == 'parquet':
                data.to_parquet(
                    file_path,
                    compression=self.compression,
                    engine='pyarrow'
                )
            elif self.format == 'csv':
                data.to_csv(file_path, index=True)
            elif self.format == 'hdf5':
                data.to_hdf(file_path, key='data', mode='w')
            elif self.format == 'feather':
                data.reset_index().to_feather(file_path)
            
            return True
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False

    def load(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load data from local storage.
        
        Args:
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Loaded data
        """
        file_path = self._get_file_path(symbol, frequency, market_type)
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            if self.format == 'parquet':
                data = pd.read_parquet(file_path)
            elif self.format == 'csv':
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif self.format == 'hdf5':
                data = pd.read_hdf(file_path, key='data')
            elif self.format == 'feather':
                data = pd.read_feather(file_path)
                data.set_index(data.columns[0], inplace=True)
            
            # Apply time filters
            if start_time:
                data = data[data.index >= start_time]
            if end_time:
                data = data[data.index <= end_time]
            
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def append(
        self,
        data: pd.DataFrame,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> bool:
        """
        Append data to existing file.
        
        Args:
            data: New data to append
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            
        Returns:
            True if successful
        """
        # Load existing data
        existing = self.load(symbol, frequency, market_type)
        
        if existing.empty:
            return self.save(data, symbol, frequency, market_type)
        
        # Combine and remove duplicates
        combined = pd.concat([existing, data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        return self.save(combined, symbol, frequency, market_type)

    def list_symbols(
        self,
        frequency: Optional[str] = None,
        market_type: str = 'spot',
    ) -> List[str]:
        """
        List all available symbols.
        
        Args:
            frequency: Filter by frequency
            market_type: Market type
            
        Returns:
            List of symbol names
        """
        if frequency:
            dir_path = self.base_path / market_type / frequency
        else:
            dir_path = self.base_path / market_type
        
        if not dir_path.exists():
            return []
        
        symbols = []
        for file_path in dir_path.rglob(f"*.{self.format}"):
            symbol = file_path.stem.replace('_', '-')
            symbols.append(symbol)
        
        return sorted(symbols)

    def delete(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> bool:
        """
        Delete data file.
        
        Args:
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            
        Returns:
            True if deleted
        """
        file_path = self._get_file_path(symbol, frequency, market_type)
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_metadata(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> Dict[str, Any]:
        """
        Get metadata for a data file.
        
        Args:
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            
        Returns:
            Metadata dictionary
        """
        data = self.load(symbol, frequency, market_type)
        
        if data.empty:
            return {}
        
        return {
            'symbol': symbol,
            'frequency': frequency,
            'market_type': market_type,
            'start_time': data.index[0],
            'end_time': data.index[-1],
            'rows': len(data),
            'columns': list(data.columns),
            'size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
        }
