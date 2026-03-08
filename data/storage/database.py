"""
Database storage for market data.

Supports PostgreSQL and ClickHouse for high-performance storage.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class DatabaseStorage:
    """
    Database storage for OHLCV data.
    """

    def __init__(
        self,
        connection_string: str,
        db_type: str = 'postgresql',
        batch_size: int = 10000,
    ):
        self.connection_string = connection_string
        self.db_type = db_type.lower()
        self.batch_size = batch_size
        self._engine = None

    def _get_engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.connection_string)
        return self._engine

    def _get_table_name(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> str:
        """Generate table name for symbol data."""
        clean_symbol = symbol.replace('/', '_').replace('-', '_').lower()
        return f"{market_type}_{frequency}_{clean_symbol}"

    def save(
        self,
        data: pd.DataFrame,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
        if_exists: str = 'append',
    ) -> bool:
        """
        Save data to database.
        
        Args:
            data: OHLCV data to save
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            if_exists: 'append', 'replace', or 'fail'
            
        Returns:
            True if successful
        """
        if data.empty:
            return False
        
        table_name = self._get_table_name(symbol, frequency, market_type)
        engine = self._get_engine()
        
        try:
            # Reset index to make timestamp a column
            df = data.reset_index()
            
            # Ensure column names are lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Save to database
            df.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=self.batch_size,
            )
            
            # Create index on timestamp
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp "
                    f"ON {table_name} (timestamp)"
                ))
                conn.commit()
            
            return True
        except Exception as e:
            print(f"Error saving data to database: {e}")
            return False

    def load(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load data from database.
        
        Args:
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum rows to return
            
        Returns:
            Loaded data
        """
        table_name = self._get_table_name(symbol, frequency, market_type)
        engine = self._get_engine()
        
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        if start_time:
            query += f" AND timestamp >= '{start_time}'"
        if end_time:
            query += f" AND timestamp <= '{end_time}'"
        
        query += " ORDER BY timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error loading data from database: {e}")
            return pd.DataFrame()

    def list_symbols(
        self,
        market_type: Optional[str] = None,
    ) -> List[str]:
        """
        List all available symbols in database.
        
        Args:
            market_type: Filter by market type
            
        Returns:
            List of symbol names
        """
        engine = self._get_engine()
        
        try:
            from sqlalchemy import text
            
            if self.db_type == 'postgresql':
                query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name LIKE '%_1m_%'
                """
            else:
                query = "SHOW TABLES"
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            
            # Parse symbol names from table names
            symbols = []
            for table in tables:
                parts = table.split('_')
                if len(parts) >= 3:
                    symbol = '_'.join(parts[2:]).upper()
                    symbols.append(symbol)
            
            return sorted(set(symbols))
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []

    def delete(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> bool:
        """
        Delete data table.
        
        Args:
            symbol: Trading pair symbol
            frequency: Data frequency
            market_type: Market type
            
        Returns:
            True if deleted
        """
        table_name = self._get_table_name(symbol, frequency, market_type)
        engine = self._get_engine()
        
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting table: {e}")
            return False

    def get_metadata(
        self,
        symbol: str,
        frequency: str,
        market_type: str = 'spot',
    ) -> Dict[str, Any]:
        """Get metadata for a data table."""
        table_name = self._get_table_name(symbol, frequency, market_type)
        engine = self._get_engine()
        
        try:
            from sqlalchemy import text
            
            with engine.connect() as conn:
                # Get row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.scalar()
                
                # Get time range
                time_result = conn.execute(text(
                    f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}"
                ))
                min_time, max_time = time_result.fetchone()
            
            return {
                'symbol': symbol,
                'frequency': frequency,
                'market_type': market_type,
                'table_name': table_name,
                'rows': row_count,
                'start_time': min_time,
                'end_time': max_time,
            }
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return {}
