"""
数字货币数据管理器
整合数据获取、处理和存储功能，提供统一的数据管理接口
"""

import os
import time
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import pandas as pd
import numpy as np

from .database import ParquetStorage, DataCache, get_storage, get_cache
from .data_collector import DataCollector, ExchangeConfig, MarketType
from .data_processor import DataProcessor, DataQualityReport, DataQualityLevel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataUpdateConfig:
    """数据更新配置"""
    auto_update: bool = True
    update_interval: int = 300  # 秒
    check_quality: bool = True
    fill_missing: bool = True
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: int = 5  # 秒


@dataclass
class DataTask:
    """数据任务"""
    symbol: str
    timeframe: str
    market_type: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    priority: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class DataManager:
    """
    数据管理器
    
    功能:
    - 统一管理数据获取、处理和存储
    - 自动增量更新
    - 数据质量监控
    - 批量任务处理
    - 多时间frame数据对齐
    """
    
    def __init__(
        self,
        data_path: str = "./data",
        exchange_name: str = "binance",
        exchange_config: Optional[ExchangeConfig] = None,
        market_type: MarketType = MarketType.SPOT,
        update_config: Optional[DataUpdateConfig] = None
    ):
        """
        初始化数据管理器
        
        Args:
            data_path: 数据存储路径
            exchange_name: 交易所名称
            exchange_config: 交易所配置
            market_type: 市场类型
            update_config: 更新配置
        """
        self.data_path = data_path
        self.exchange_name = exchange_name
        self.market_type = market_type
        self.update_config = update_config or DataUpdateConfig()
        
        # 初始化组件
        self.storage = get_storage(data_path)
        self.cache = get_cache(max_size=100)
        self.collector = DataCollector(exchange_name, exchange_config, market_type)
        self.processor = DataProcessor()
        
        # 任务管理
        self._tasks: Dict[str, DataTask] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.update_config.max_workers)
        
        logger.info(f"数据管理器初始化完成: {exchange_name} {market_type.value}")
    
    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True,
        check_quality: bool = True
    ) -> pd.DataFrame:
        """
        获取并存储数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            use_cache: 是否使用缓存
            check_quality: 是否检查数据质量
        
        Returns:
            pd.DataFrame: 数据
        """
        # 1. 检查缓存
        if use_cache:
            cached_df = self.cache.get(symbol, timeframe, self.market_type.value)
            if cached_df is not None:
                logger.debug(f"从缓存获取数据: {symbol} {timeframe}")
                return cached_df
        
        # 2. 检查本地存储
        if self.storage.check_data_exists(symbol, timeframe, self.market_type.value):
            try:
                stored_df = self.storage.load_data(
                    symbol, timeframe, self.market_type.value,
                    start_time, end_time
                )
                
                if not stored_df.empty:
                    # 更新缓存
                    self.cache.set(stored_df, symbol, timeframe, self.market_type.value)
                    
                    # 检查是否需要增量更新
                    if self.update_config.auto_update:
                        stored_df = self._incremental_update(
                            symbol, timeframe, stored_df, end_time
                        )
                    
                    logger.info(f"从存储加载数据: {symbol} {timeframe} ({len(stored_df)} 行)")
                    return stored_df
                    
            except Exception as e:
                logger.warning(f"从存储加载数据失败: {e}")
        
        # 3. 从交易所获取
        logger.info(f"从交易所获取数据: {symbol} {timeframe}")
        
        df = self.collector.fetch_ohlcv_range(
            symbol, timeframe, start_time, end_time
        )
        
        if df.empty:
            logger.warning(f"未获取到数据: {symbol} {timeframe}")
            return df
        
        # 4. 数据清洗
        if self.update_config.fill_missing:
            df = self.processor.clean_data(df)
        
        # 5. 质量检查
        if check_quality:
            report = self.processor.check_quality(df, symbol, timeframe)
            if report.quality_level in [DataQualityLevel.POOR, DataQualityLevel.CRITICAL]:
                logger.warning(f"数据质量较差: {report}")
        
        # 6. 存储数据
        self.storage.save_data(
            df, symbol, timeframe, self.market_type.value, self.exchange_name
        )
        
        # 7. 更新缓存
        self.cache.set(df, symbol, timeframe, self.market_type.value)
        
        return df
    
    def _incremental_update(
        self,
        symbol: str,
        timeframe: str,
        existing_df: pd.DataFrame,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        增量更新数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            existing_df: 现有数据
            end_time: 结束时间
        
        Returns:
            pd.DataFrame: 更新后的数据
        """
        if existing_df.empty:
            return existing_df
        
        last_timestamp = existing_df.index.max()
        now = datetime.now()
        target_end = end_time or now
        
        # 检查是否需要更新
        if last_timestamp >= target_end - timedelta(minutes=5):
            return existing_df
        
        logger.info(f"增量更新: {symbol} {timeframe} 从 {last_timestamp}")
        
        try:
            # 获取新数据
            new_df = self.collector.fetch_ohlcv_range(
                symbol, timeframe, last_timestamp, target_end
            )
            
            if new_df.empty:
                return existing_df
            
            # 合并数据
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # 存储更新后的数据
            self.storage.save_data(
                combined_df, symbol, timeframe, self.market_type.value, self.exchange_name
            )
            
            # 更新缓存
            self.cache.set(combined_df, symbol, timeframe, self.market_type.value)
            
            logger.info(f"增量更新完成: {len(existing_df)} -> {len(combined_df)} 行")
            return combined_df
            
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            return existing_df
    
    def batch_fetch(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        批量获取多个币种和周期的数据
        
        Args:
            symbols: 交易对列表
            timeframes: 时间周期列表
            start_time: 开始时间
            end_time: 结束时间
            progress_callback: 进度回调函数 (symbol, timeframe, current, total)
        
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: 嵌套字典 {symbol: {timeframe: df}}
        """
        results = {}
        total_tasks = len(symbols) * len(timeframes)
        current = 0
        
        logger.info(f"开始批量获取: {len(symbols)} 个币种 x {len(timeframes)} 个周期 = {total_tasks} 个任务")
        
        for symbol in symbols:
            results[symbol] = {}
            
            for timeframe in timeframes:
                current += 1
                
                if progress_callback:
                    progress_callback(symbol, timeframe, current, total_tasks)
                
                try:
                    df = self.fetch_and_store(
                        symbol, timeframe, start_time, end_time
                    )
                    results[symbol][timeframe] = df
                    
                except Exception as e:
                    logger.error(f"获取 {symbol} {timeframe} 失败: {e}")
                    results[symbol][timeframe] = pd.DataFrame()
                
                # 添加延迟避免速率限制
                time.sleep(0.1)
        
        logger.info(f"批量获取完成: {current}/{total_tasks} 个任务")
        return results
    
    def parallel_batch_fetch(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        并行批量获取数据
        
        Args:
            symbols: 交易对列表
            timeframes: 时间周期列表
            start_time: 开始时间
            end_time: 结束时间
            max_workers: 最大并行工作数
        
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: 嵌套字典
        """
        max_workers = max_workers or self.update_config.max_workers
        results = {}
        
        # 创建所有任务
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe))
        
        logger.info(f"开始并行批量获取: {len(tasks)} 个任务, {max_workers} 个工作者")
        
        def fetch_task(symbol: str, timeframe: str) -> Tuple[str, str, pd.DataFrame]:
            """执行单个获取任务"""
            try:
                df = self.fetch_and_store(symbol, timeframe, start_time, end_time)
                return symbol, timeframe, df
            except Exception as e:
                logger.error(f"获取 {symbol} {timeframe} 失败: {e}")
                return symbol, timeframe, pd.DataFrame()
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_task, symbol, timeframe): (symbol, timeframe)
                for symbol, timeframe in tasks
            }
            
            for future in as_completed(futures):
                symbol, timeframe, df = future.result()
                
                if symbol not in results:
                    results[symbol] = {}
                results[symbol][timeframe] = df
        
        logger.info(f"并行批量获取完成")
        return results
    
    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        add_indicators: bool = False,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取数据（便捷方法）
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            add_indicators: 是否添加技术指标
            indicators: 指标列表
        
        Returns:
            pd.DataFrame: 数据
        """
        df = self.fetch_and_store(symbol, timeframe, start_time, end_time)
        
        if add_indicators and not df.empty:
            df = self.processor.add_technical_indicators(df, indicators)
        
        return df
    
    def align_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        获取并对齐多个时间周期的数据
        
        Args:
            symbol: 交易对符号
            timeframes: 时间周期列表
            start_time: 开始时间
            end_time: 结束时间
            method: 对齐方法
        
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据字典
        """
        # 获取所有时间周期的数据
        data_dict = {}
        for tf in timeframes:
            df = self.get_data(symbol, tf, start_time, end_time)
            if not df.empty:
                data_dict[tf] = df
        
        # 对齐数据
        aligned_data = self.processor.align_multiple_symbols(data_dict, method)
        
        return aligned_data
    
    def update_data(
        self,
        symbol: str,
        timeframe: str,
        force: bool = False
    ) -> bool:
        """
        更新数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            force: 是否强制更新
        
        Returns:
            bool: 是否成功更新
        """
        try:
            if force:
                # 强制重新获取
                self.storage.delete_data(symbol, timeframe, self.market_type.value)
                self.cache.get(symbol, timeframe, self.market_type.value)
            
            # 获取数据（会自动增量更新）
            df = self.fetch_and_store(symbol, timeframe)
            
            return not df.empty
            
        except Exception as e:
            logger.error(f"更新数据失败: {e}")
            return False
    
    def get_data_info(self) -> pd.DataFrame:
        """获取所有数据的信息"""
        return self.storage.get_data_info()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        return self.storage.get_storage_size()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")
    
    def delete_data(
        self,
        symbol: str,
        timeframe: str
    ) -> bool:
        """
        删除数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
        
        Returns:
            bool: 是否成功删除
        """
        # 从存储删除
        success = self.storage.delete_data(symbol, timeframe, self.market_type.value)
        
        # 从缓存删除（通过重新获取）
        # 实际实现中应该直接从缓存删除
        
        return success
    
    def check_data_quality(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[DataQualityReport]:
        """
        检查数据质量
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
        
        Returns:
            DataQualityReport: 质量报告
        """
        try:
            df = self.storage.load_data(symbol, timeframe, self.market_type.value)
            if df.empty:
                return None
            
            return self.processor.check_quality(df, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"检查数据质量失败: {e}")
            return None
    
    def get_available_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """获取可用交易对列表"""
        return self.collector.get_available_symbols(quote_currency)
    
    def get_popular_symbols(self, max_count: int = 50) -> List[str]:
        """获取热门币种列表"""
        all_symbols = self.collector.POPULAR_SYMBOLS
        available = self.get_available_symbols()
        
        # 过滤可用的热门币种
        popular = [s for s in all_symbols if s in available]
        return popular[:max_count]
    
    def close(self) -> None:
        """关闭管理器"""
        self.collector.close()
        self._executor.shutdown(wait=True)
        logger.info("数据管理器已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class MultiSymbolDataManager:
    """
    多币种数据管理器
    
    管理多个币种的数据，支持批量操作
    """
    
    def __init__(
        self,
        data_path: str = "./data",
        exchange_name: str = "binance",
        market_type: MarketType = MarketType.SPOT
    ):
        """初始化多币种数据管理器"""
        self.manager = DataManager(data_path, exchange_name, market_type=market_type)
        self.symbols: List[str] = []
        self.timeframes: List[str] = ["1m", "5m", "15m"]
    
    def set_symbols(self, symbols: List[str]) -> None:
        """设置关注的币种列表"""
        self.symbols = symbols
        logger.info(f"设置关注币种: {len(symbols)} 个")
    
    def set_timeframes(self, timeframes: List[str]) -> None:
        """设置关注的时间周期"""
        self.timeframes = timeframes
        logger.info(f"设置关注周期: {timeframes}")
    
    def fetch_all(
        self,
        days: int = 7,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        获取所有关注币种的数据
        
        Args:
            days: 获取天数
            progress_callback: 进度回调
        
        Returns:
            Dict: 所有数据
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        return self.manager.batch_fetch(
            self.symbols,
            self.timeframes,
            start_time,
            end_time,
            progress_callback
        )
    
    def get_correlation_matrix(
        self,
        timeframe: str = "1h",
        price_col: str = "close"
    ) -> pd.DataFrame:
        """
        获取币种相关性矩阵
        
        Args:
            timeframe: 时间周期
            price_col: 价格列名
        
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        # 获取所有币种数据
        price_data = {}
        
        for symbol in self.symbols:
            try:
                df = self.manager.get_data(symbol, timeframe)
                if not df.empty and price_col in df.columns:
                    # 使用对数收益率
                    returns = np.log(df[price_col] / df[price_col].shift(1))
                    price_data[symbol] = returns
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")
        
        if not price_data:
            return pd.DataFrame()
        
        # 构建DataFrame并计算相关性
        returns_df = pd.DataFrame(price_data)
        correlation = returns_df.corr()
        
        return correlation
    
    def close(self) -> None:
        """关闭管理器"""
        self.manager.close()


if __name__ == "__main__":
    # 测试代码
    with DataManager("./test_data") as manager:
        # 获取单个币种数据
        df = manager.get_data("BTC/USDT", "1h", add_indicators=True)
        print(f"获取到 {len(df)} 条BTC数据")
        print(df.tail())
        
        # 获取存储统计
        stats = manager.get_storage_stats()
        print(f"\n存储统计: {stats}")
        
        # 获取数据信息
        info = manager.get_data_info()
        print(f"\n数据信息:\n{info}")
