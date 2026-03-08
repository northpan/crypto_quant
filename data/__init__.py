"""
数字货币数据获取和预处理模块

提供完整的数字货币数据管理功能，包括:
- 数据获取：通过CCXT连接多个交易所
- 数据存储：高效的Parquet格式存储
- 数据处理：清洗、预处理、质量检查
- 数据管理：统一的数据管理接口

主要组件:
- DataCollector: 数据获取主类
- DataProcessor: 数据清洗和预处理
- DataManager: 数据管理器
- ParquetStorage: Parquet数据存储

示例:
    >>> from crypto_quant.data import DataManager
    >>> 
    >>> # 初始化数据管理器
    >>> manager = DataManager(data_path="./data", exchange_name="binance")
    >>> 
    >>> # 获取数据
    >>> df = manager.get_data("BTC/USDT", "1h", days=7)
    >>> 
    >>> # 批量获取
    >>> data = manager.batch_fetch(
    ...     symbols=["BTC/USDT", "ETH/USDT"],
    ...     timeframes=["1m", "5m", "15m"]
    ... )
"""

from .database import (
    ParquetStorage,
    DataCache,
    DataMetadata,
    get_storage,
    get_cache,
)

from .data_collector import (
    DataCollector,
    MultiExchangeCollector,
    ExchangeConfig,
    MarketType,
    TimeFrame,
)

from .data_processor import (
    DataProcessor,
    DataQualityReport,
    DataQualityLevel,
)

from .data_manager import (
    DataManager,
    MultiSymbolDataManager,
    DataUpdateConfig,
    DataTask,
)

__version__ = "1.0.0"
__all__ = [
    # Database
    "ParquetStorage",
    "DataCache",
    "DataMetadata",
    "get_storage",
    "get_cache",
    
    # Data Collector
    "DataCollector",
    "MultiExchangeCollector",
    "ExchangeConfig",
    "MarketType",
    "TimeFrame",
    
    # Data Processor
    "DataProcessor",
    "DataQualityReport",
    "DataQualityLevel",
    
    # Data Manager
    "DataManager",
    "MultiSymbolDataManager",
    "DataUpdateConfig",
    "DataTask",
]
