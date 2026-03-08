"""
Base data source implementations.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import asyncio
import pandas as pd
import ccxt.async_support as ccxt

from ...core.base import BaseDataSource, MarketData, DataFrequency


class ExchangeDataSource(BaseDataSource):
    """
    Exchange data source using CCXT library.
    
    Supports multiple exchanges through unified interface.
    """

    SUPPORTED_EXCHANGES = [
        "binance", "binanceusdm", "okx", "bybit", 
        "kucoin", "gateio", "mexc", "bitget"
    ]

    def __init__(
        self,
        name: str,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange: Optional[ccxt.Exchange] = None
        self._subscribers: Dict[str, List[Callable]] = {}

    async def connect(self) -> bool:
        """Establish connection to exchange."""
        try:
            if self.exchange_id not in self.SUPPORTED_EXCHANGES:
                raise ValueError(f"Unsupported exchange: {self.exchange_id}")
            
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},  # Default to perpetual
            })
            await self._exchange.load_markets()
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to {self.exchange_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        """Close connection to exchange."""
        if self._exchange:
            await self._exchange.close()
            self._is_connected = False
        return True

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        if not self._is_connected:
            await self.connect()
        
        timeframe = frequency.value
        since = int(start_time.timestamp() * 1000) if start_time else None
        
        all_data = []
        while True:
            try:
                ohlcv = await self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Check if we've reached end_time
                last_timestamp = ohlcv[-1][0]
                if end_time and last_timestamp >= end_time.timestamp() * 1000:
                    break
                
                # Update since for next batch
                since = last_timestamp + 1
                
                # Rate limiting
                await asyncio.sleep(self._exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching OHLCV for {symbol}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter by end_time
        if end_time:
            df = df[df.index <= end_time]
        
        return df

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch current ticker data."""
        if not self._is_connected:
            await self.connect()
        
        try:
            if symbols:
                tickers = {}
                for symbol in symbols:
                    ticker = await self._exchange.fetch_ticker(symbol)
                    tickers[symbol] = ticker
            else:
                tickers = await self._exchange.fetch_tickers()
            
            return tickers
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return {}

    async def subscribe_ohlcv(
        self,
        symbols: List[str],
        frequency: DataFrequency,
        callback: Callable[[MarketData], None],
    ) -> bool:
        """Subscribe to real-time OHLCV updates via WebSocket."""
        # WebSocket implementation would go here
        # This is a placeholder for the interface
        raise NotImplementedError("WebSocket subscription not yet implemented")

    async def unsubscribe_ohlcv(self, symbols: List[str]) -> bool:
        """Unsubscribe from OHLCV updates."""
        raise NotImplementedError("WebSocket unsubscription not yet implemented")

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch funding rate for perpetual contracts."""
        if not self._is_connected:
            await self.connect()
        
        try:
            funding = await self._exchange.fetch_funding_rate(symbol)
            return funding
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return {}

    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch order book."""
        if not self._is_connected:
            await self.connect()
        
        try:
            orderbook = await self._exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return {}


class DatabaseDataSource(BaseDataSource):
    """
    Database data source for historical data.
    
    Supports multiple database backends (PostgreSQL, ClickHouse, etc.)
    """

    def __init__(
        self,
        name: str,
        connection_string: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.connection_string = connection_string
        self._engine = None

    async def connect(self) -> bool:
        """Establish database connection."""
        try:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.connection_string)
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False

    async def disconnect(self) -> bool:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._is_connected = False
        return True

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from database."""
        import pandas as pd
        
        table_name = f"ohlcv_{frequency.value}_{symbol.replace('-', '_').replace('/', '_')}"
        
        query = f"""
            SELECT * FROM {table_name}
            WHERE 1=1
        """
        
        if start_time:
            query += f" AND timestamp >= '{start_time}'"
        if end_time:
            query += f" AND timestamp <= '{end_time}'"
        
        query += f" ORDER BY timestamp LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self._engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data from database: {e}")
            return pd.DataFrame()

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch latest ticker data from database."""
        # Implementation depends on database schema
        raise NotImplementedError("Ticker fetch from database not implemented")

    async def subscribe_ohlcv(
        self,
        symbols: List[str],
        frequency: DataFrequency,
        callback: Callable[[MarketData], None],
    ) -> bool:
        """Subscribe not supported for database source."""
        raise NotImplementedError("Subscription not supported for database source")

    async def unsubscribe_ohlcv(self, symbols: List[str]) -> bool:
        """Unsubscribe not supported for database source."""
        raise NotImplementedError("Unsubscription not supported for database source")
