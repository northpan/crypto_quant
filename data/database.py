"""
数字货币数据存储接口模块
支持Parquet格式存储，提供高效的数据读写能力
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

import pandas as pd
import numpy as np

# 尝试导入pyarrow，如果失败则使用CSV备选
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None
    pq = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataMetadata:
    """数据元数据"""
    symbol: str
    timeframe: str
    market_type: str  # 'spot' or 'futures'
    exchange: str
    start_time: datetime
    end_time: datetime
    rows: int
    columns: List[str]
    file_path: str
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_type': self.market_type,
            'exchange': self.exchange,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'rows': self.rows,
            'columns': self.columns,
            'file_path': self.file_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetadata':
        """从字典创建"""
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            market_type=data['market_type'],
            exchange=data['exchange'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            rows=data['rows'],
            columns=data['columns'],
            file_path=data['file_path'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', '1.0')
        )


class ParquetStorage:
    """
    Parquet格式数据存储类
    
    特点:
    - 高效的列式存储
    - 支持数据压缩
    - 快速查询性能
    - 跨平台兼容
    """
    
    def __init__(self, base_path: str = "./data"):
        """
        初始化存储
        
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "metadata"
        self._ensure_directories()
        self._metadata_cache: Dict[str, DataMetadata] = {}
        self._load_metadata_cache()
    
    def _ensure_directories(self) -> None:
        """确保必要目录存在"""
        directories = [
            self.base_path,
            self.metadata_path,
            self.base_path / "spot",
            self.base_path / "futures",
            self.base_path / "cache"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata_cache(self) -> None:
        """加载元数据缓存"""
        if not self.metadata_path.exists():
            return
        
        for meta_file in self.metadata_path.glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    key = meta_file.stem
                    self._metadata_cache[key] = DataMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"加载元数据文件失败 {meta_file}: {e}")
    
    def _get_storage_key(self, symbol: str, timeframe: str, market_type: str) -> str:
        """生成存储键"""
        # 替换特殊字符，确保键可以作为文件名
        safe_symbol = symbol.lower().replace('/', '_').replace(':', '_')
        return f"{safe_symbol}_{timeframe}_{market_type}"
    
    def _get_file_path(self, symbol: str, timeframe: str, market_type: str) -> Path:
        """获取数据文件路径"""
        market_dir = self.base_path / market_type
        market_dir.mkdir(exist_ok=True)
        
        # 按交易所和币种组织目录
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        ext = "parquet" if HAS_PYARROW else "csv"
        filename = f"{safe_symbol}_{timeframe}.{ext}"
        return market_dir / filename
    
    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        return self.metadata_path / f"{key}.json"
    
    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        exchange: str = "binance",
        compression: str = "zstd"
    ) -> DataMetadata:
        """
        保存数据到Parquet文件
        
        Args:
            df: 要保存的数据框
            symbol: 交易对符号
            timeframe: 时间周期
            market_type: 市场类型 (spot/futures)
            exchange: 交易所名称
            compression: 压缩算法 (zstd, snappy, gzip, none)
        
        Returns:
            DataMetadata: 数据元数据
        """
        if df.empty:
            raise ValueError("数据框为空，无法保存")
        
        # 确保索引是datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')
        
        # 确保索引排序
        df = df.sort_index()
        
        file_path = self._get_file_path(symbol, timeframe, market_type)
        key = self._get_storage_key(symbol, timeframe, market_type)
        
        try:
            # 保存数据（Parquet或CSV）
            if HAS_PYARROW:
                # 使用Parquet格式
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    file_path,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True
                )
            else:
                # 使用CSV格式（备选）
                df.to_csv(file_path, index=True)
                logger.debug(f"使用CSV格式保存数据: {file_path}")
            
            # 创建元数据
            now = datetime.now()
            metadata = DataMetadata(
                symbol=symbol,
                timeframe=timeframe,
                market_type=market_type,
                exchange=exchange,
                start_time=df.index.min(),
                end_time=df.index.max(),
                rows=len(df),
                columns=list(df.columns),
                file_path=str(file_path),
                created_at=now,
                updated_at=now
            )
            
            # 保存元数据
            self._save_metadata(key, metadata)
            
            logger.info(f"数据已保存: {symbol} {timeframe} {market_type} - {len(df)} 行")
            return metadata
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise
    
    def _save_metadata(self, key: str, metadata: DataMetadata) -> None:
        """保存元数据到JSON文件"""
        meta_path = self._get_metadata_path(key)
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        self._metadata_cache[key] = metadata
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        加载Parquet数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            market_type: 市场类型
            start_time: 开始时间
            end_time: 结束时间
            columns: 要加载的列
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = self._get_file_path(symbol, timeframe, market_type)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        try:
            # 加载数据（Parquet或CSV）
            if HAS_PYARROW and file_path.suffix == '.parquet':
                # 使用Parquet格式
                if columns:
                    table = pq.read_table(file_path, columns=columns)
                else:
                    table = pq.read_table(file_path)
                df = table.to_pandas()
            else:
                # 使用CSV格式（备选）
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if columns:
                    df = df[columns]
            
            # 时间过滤
            if start_time:
                df = df[df.index >= start_time]
            if end_time:
                df = df[df.index <= end_time]
            
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def append_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        exchange: str = "binance"
    ) -> DataMetadata:
        """
        追加数据到现有文件
        
        Args:
            df: 要追加的新数据
            symbol: 交易对符号
            timeframe: 时间周期
            market_type: 市场类型
            exchange: 交易所名称
        
        Returns:
            DataMetadata: 更新后的元数据
        """
        file_path = self._get_file_path(symbol, timeframe, market_type)
        
        if file_path.exists():
            # 加载现有数据
            existing_df = self.load_data(symbol, timeframe, market_type)
            
            # 合并数据（去除重复）
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            return self.save_data(
                combined_df, symbol, timeframe, market_type, exchange
            )
        else:
            return self.save_data(df, symbol, timeframe, market_type, exchange)
    
    def get_metadata(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot"
    ) -> Optional[DataMetadata]:
        """获取数据元数据"""
        key = self._get_storage_key(symbol, timeframe, market_type)
        return self._metadata_cache.get(key)
    
    def list_available_data(
        self,
        market_type: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[DataMetadata]:
        """
        列出所有可用数据
        
        Args:
            market_type: 过滤市场类型
            timeframe: 过滤时间周期
        
        Returns:
            List[DataMetadata]: 元数据列表
        """
        results = []
        for metadata in self._metadata_cache.values():
            if market_type and metadata.market_type != market_type:
                continue
            if timeframe and metadata.timeframe != timeframe:
                continue
            results.append(metadata)
        return results
    
    def delete_data(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot"
    ) -> bool:
        """
        删除数据文件
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            market_type: 市场类型
        
        Returns:
            bool: 是否成功删除
        """
        key = self._get_storage_key(symbol, timeframe, market_type)
        file_path = self._get_file_path(symbol, timeframe, market_type)
        meta_path = self._get_metadata_path(key)
        
        success = True
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"删除数据文件: {file_path}")
        except Exception as e:
            logger.error(f"删除数据文件失败: {e}")
            success = False
        
        try:
            if meta_path.exists():
                meta_path.unlink()
        except Exception as e:
            logger.error(f"删除元数据文件失败: {e}")
        
        if key in self._metadata_cache:
            del self._metadata_cache[key]
        
        return success
    
    def get_data_info(self) -> pd.DataFrame:
        """获取所有数据的信息摘要"""
        data = []
        for meta in self._metadata_cache.values():
            data.append({
                'symbol': meta.symbol,
                'timeframe': meta.timeframe,
                'market_type': meta.market_type,
                'exchange': meta.exchange,
                'start_time': meta.start_time,
                'end_time': meta.end_time,
                'rows': meta.rows,
                'columns': len(meta.columns),
                'updated_at': meta.updated_at
            })
        
        return pd.DataFrame(data)
    
    def check_data_exists(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot"
    ) -> bool:
        """检查数据是否存在"""
        file_path = self._get_file_path(symbol, timeframe, market_type)
        return file_path.exists()
    
    def get_storage_size(self) -> Dict[str, Union[int, float]]:
        """获取存储使用情况"""
        total_size = 0
        file_count = 0
        
        # 支持Parquet和CSV文件
        extensions = ["*.parquet", "*.csv"]
        for ext in extensions:
            for file_path in self.base_path.rglob(ext):
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
        
        return {
            'file_count': file_count,
            'total_bytes': total_size,
            'total_mb': round(total_size / (1024 * 1024), 2),
            'total_gb': round(total_size / (1024 * 1024 * 1024), 3)
        }


class DataCache:
    """
    内存数据缓存类
    
    用于缓存频繁访问的数据，提高查询性能
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_time: Dict[str, datetime] = {}
    
    def _get_key(
        self,
        symbol: str,
        timeframe: str,
        market_type: str
    ) -> str:
        """生成缓存键"""
        return f"{symbol}_{timeframe}_{market_type}"
    
    def get(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot"
    ) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        key = self._get_key(symbol, timeframe, market_type)
        if key in self._cache:
            self._access_time[key] = datetime.now()
            return self._cache[key]
        return None
    
    def set(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_type: str = "spot"
    ) -> None:
        """设置缓存数据"""
        key = self._get_key(symbol, timeframe, market_type)
        
        # 如果缓存已满，移除最久未访问的
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        self._cache[key] = df.copy()
        self._access_time[key] = datetime.now()
    
    def _evict_oldest(self) -> None:
        """移除最久未访问的缓存"""
        if not self._access_time:
            return
        
        oldest_key = min(self._access_time, key=self._access_time.get)
        del self._cache[oldest_key]
        del self._access_time[oldest_key]
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_time.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size
        }


# 全局存储实例
_storage_instance: Optional[ParquetStorage] = None
_cache_instance: Optional[DataCache] = None


def get_storage(base_path: Optional[str] = None) -> ParquetStorage:
    """获取全局存储实例"""
    global _storage_instance
    if _storage_instance is None:
        path = base_path or os.environ.get('CRYPTO_DATA_PATH', './data')
        _storage_instance = ParquetStorage(path)
    return _storage_instance


def get_cache(max_size: int = 100) -> DataCache:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache(max_size)
    return _cache_instance


def reset_instances() -> None:
    """重置全局实例（用于测试）"""
    global _storage_instance, _cache_instance
    _storage_instance = None
    _cache_instance = None


if __name__ == "__main__":
    # 测试代码
    storage = ParquetStorage("./test_data")
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    test_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 保存数据
    metadata = storage.save_data(
        test_df, "BTC/USDT", "1m", "spot", "binance"
    )
    print(f"元数据: {metadata}")
    
    # 加载数据
    loaded_df = storage.load_data("BTC/USDT", "1m", "spot")
    print(f"加载数据: {len(loaded_df)} 行")
    print(loaded_df.head())
    
    # 获取存储信息
    info = storage.get_storage_size()
    print(f"存储信息: {info}")
    
    # 清理
    storage.delete_data("BTC/USDT", "1m", "spot")
