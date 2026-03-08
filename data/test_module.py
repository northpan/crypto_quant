"""
数字货币数据模块测试
验证模块的基本功能
"""

import unittest
import os
import tempfile
import shutil
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/mnt/okcomputer/output/crypto_quant/data')

from database import ParquetStorage, DataCache, DataMetadata
from data_processor import DataProcessor, DataQualityLevel


class TestParquetStorage(unittest.TestCase):
    """测试Parquet存储功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.test_dir = tempfile.mkdtemp()
        cls.storage = ParquetStorage(cls.test_dir)
        
        # 创建测试数据
        cls.test_df = cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def _create_test_data(rows: int = 100) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=rows, freq='1min')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.randn(rows).cumsum() + 50000,
            'high': np.random.randn(rows).cumsum() + 50100,
            'low': np.random.randn(rows).cumsum() + 49900,
            'close': np.random.randn(rows).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, rows)
        }, index=dates)
    
    def test_save_and_load(self):
        """测试保存和加载数据"""
        # 保存数据
        metadata = self.storage.save_data(
            self.test_df,
            symbol="BTC/USDT",
            timeframe="1m",
            market_type="spot"
        )
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.symbol, "BTC/USDT")
        self.assertEqual(metadata.timeframe, "1m")
        self.assertEqual(metadata.rows, len(self.test_df))
        
        # 加载数据
        loaded_df = self.storage.load_data("BTC/USDT", "1m", "spot")
        
        self.assertEqual(len(loaded_df), len(self.test_df))
        self.assertListEqual(
            list(loaded_df.columns),
            list(self.test_df.columns)
        )
    
    def test_append_data(self):
        """测试追加数据"""
        # 先保存初始数据
        self.storage.save_data(
            self.test_df,
            symbol="ETH/USDT",
            timeframe="1m",
            market_type="spot"
        )
        
        # 创建新数据
        new_dates = pd.date_range('2024-01-01 01:40', periods=20, freq='1min')
        new_df = pd.DataFrame({
            'open': np.random.randn(20).cumsum() + 50000,
            'high': np.random.randn(20).cumsum() + 50100,
            'low': np.random.randn(20).cumsum() + 49900,
            'close': np.random.randn(20).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 20)
        }, index=new_dates)
        
        # 追加数据
        metadata = self.storage.append_data(
            new_df,
            symbol="ETH/USDT",
            timeframe="1m",
            market_type="spot"
        )
        
        # 验证
        loaded_df = self.storage.load_data("ETH/USDT", "1m", "spot")
        self.assertGreater(len(loaded_df), len(self.test_df))
    
    def test_delete_data(self):
        """测试删除数据"""
        # 保存数据
        self.storage.save_data(
            self.test_df,
            symbol="XRP/USDT",
            timeframe="1m",
            market_type="spot"
        )
        
        # 验证存在
        self.assertTrue(
            self.storage.check_data_exists("XRP/USDT", "1m", "spot")
        )
        
        # 删除数据
        success = self.storage.delete_data("XRP/USDT", "1m", "spot")
        self.assertTrue(success)
        
        # 验证不存在
        self.assertFalse(
            self.storage.check_data_exists("XRP/USDT", "1m", "spot")
        )
    
    def test_get_storage_size(self):
        """测试获取存储大小"""
        stats = self.storage.get_storage_size()
        
        self.assertIn('file_count', stats)
        self.assertIn('total_bytes', stats)
        self.assertIn('total_mb', stats)
        self.assertIsInstance(stats['file_count'], int)


class TestDataCache(unittest.TestCase):
    """测试数据缓存功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.cache = DataCache(max_size=5)
        self.test_df = self._create_test_data()
    
    @staticmethod
    def _create_test_data(rows: int = 10) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=rows, freq='1min')
        return pd.DataFrame({
            'close': np.random.randn(rows).cumsum() + 50000
        }, index=dates)
    
    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        # 设置缓存
        self.cache.set(self.test_df, "BTC/USDT", "1m", "spot")
        
        # 获取缓存
        cached_df = self.cache.get("BTC/USDT", "1m", "spot")
        
        self.assertIsNotNone(cached_df)
        self.assertEqual(len(cached_df), len(self.test_df))
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        result = self.cache.get("ETH/USDT", "1m", "spot")
        self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """测试缓存淘汰"""
        # 添加超过最大容量的数据
        for i in range(7):
            self.cache.set(self.test_df, f"COIN{i}/USDT", "1m", "spot")
        
        # 验证缓存大小
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['size'], stats['max_size'])
    
    def test_cache_clear(self):
        """测试清空缓存"""
        self.cache.set(self.test_df, "BTC/USDT", "1m", "spot")
        self.cache.clear()
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['size'], 0)


class TestDataProcessor(unittest.TestCase):
    """测试数据处理器功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = DataProcessor()
        self.test_df = self._create_test_data()
    
    @staticmethod
    def _create_test_data(rows: int = 100) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=rows, freq='1min')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.randn(rows).cumsum() + 50000,
            'high': np.random.randn(rows).cumsum() + 50100,
            'low': np.random.randn(rows).cumsum() + 49900,
            'close': np.random.randn(rows).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, rows)
        }, index=dates)
    
    def test_check_quality(self):
        """测试质量检查"""
        report = self.processor.check_quality(
            self.test_df,
            symbol="BTC/USDT",
            timeframe="1m"
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.symbol, "BTC/USDT")
        self.assertEqual(report.timeframe, "1m")
        self.assertEqual(report.total_rows, len(self.test_df))
        self.assertIn(report.quality_level, DataQualityLevel)
    
    def test_check_quality_with_issues(self):
        """测试有问题数据的质量检查"""
        # 创建有问题数据
        bad_df = self.test_df.copy()
        bad_df.loc[bad_df.sample(10).index, 'close'] = np.nan
        bad_df = pd.concat([bad_df, bad_df.iloc[-5:]])  # 添加重复
        
        report = self.processor.check_quality(bad_df, "BTC/USDT", "1m")
        
        self.assertGreater(len(report.missing_values), 0)
        self.assertGreater(report.duplicate_rows, 0)
    
    def test_clean_data(self):
        """测试数据清洗"""
        # 创建脏数据
        dirty_df = self.test_df.copy()
        dirty_df.loc[dirty_df.sample(5).index, 'close'] = np.nan
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[-3:]])
        
        # 清洗
        clean_df = self.processor.clean_data(dirty_df)
        
        # 验证
        self.assertLess(len(clean_df), len(dirty_df))  # 去重
        self.assertEqual(clean_df['close'].isna().sum(), 0)  # 无缺失值
    
    def test_add_technical_indicators(self):
        """测试添加技术指标"""
        df_with_indicators = self.processor.add_technical_indicators(
            self.test_df,
            indicators=['sma', 'rsi', 'macd']
        )
        
        # 验证添加的指标
        self.assertIn('sma_7', df_with_indicators.columns)
        self.assertIn('sma_20', df_with_indicators.columns)
        self.assertIn('rsi', df_with_indicators.columns)
        self.assertIn('macd', df_with_indicators.columns)
    
    def test_resample(self):
        """测试重采样"""
        # 1分钟数据重采样为5分钟
        resampled_df = self.processor.resample(self.test_df, "5m")
        
        self.assertLess(len(resampled_df), len(self.test_df))
    
    def test_normalize_data(self):
        """测试数据标准化"""
        # Z-Score标准化
        normalized_df = self.processor.normalize_data(
            self.test_df,
            method="zscore",
            columns=['close']
        )
        
        # 验证均值接近0，标准差接近1
        self.assertAlmostEqual(normalized_df['close'].mean(), 0, delta=0.1)
        self.assertAlmostEqual(normalized_df['close'].std(), 1, delta=0.1)
    
    def test_calculate_returns(self):
        """测试收益率计算"""
        returns_df = self.processor.calculate_returns(self.test_df)
        
        self.assertIn('returns', returns_df.columns)
        self.assertIn('log_returns', returns_df.columns)
        self.assertIn('cumulative_returns', returns_df.columns)
    
    def test_detect_anomalies(self):
        """测试异常检测"""
        # 添加异常值
        anomaly_df = self.test_df.copy()
        anomaly_df.loc[anomaly_df.index[10], 'close'] = anomaly_df['close'].mean() * 3
        
        result_df = self.processor.detect_anomalies(anomaly_df, threshold=2.0)
        
        self.assertIn('is_anomaly', result_df.columns)
        self.assertIn('z_score', result_df.columns)
        self.assertTrue(result_df['is_anomaly'].any())


class TestDataMetadata(unittest.TestCase):
    """测试数据元数据"""
    
    def test_metadata_to_dict(self):
        """测试元数据转字典"""
        metadata = DataMetadata(
            symbol="BTC/USDT",
            timeframe="1m",
            market_type="spot",
            exchange="binance",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            rows=100,
            columns=['open', 'high', 'low', 'close', 'volume'],
            file_path="/test/data.parquet",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        data = metadata.to_dict()
        
        self.assertEqual(data['symbol'], "BTC/USDT")
        self.assertEqual(data['rows'], 100)
    
    def test_metadata_from_dict(self):
        """测试从字典创建元数据"""
        data = {
            'symbol': 'ETH/USDT',
            'timeframe': '1h',
            'market_type': 'futures',
            'exchange': 'binance',
            'start_time': '2024-01-01T00:00:00',
            'end_time': '2024-01-02T00:00:00',
            'rows': 24,
            'columns': ['open', 'high', 'low', 'close', 'volume'],
            'file_path': '/test/eth.parquet',
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
            'version': '1.0'
        }
        
        metadata = DataMetadata.from_dict(data)
        
        self.assertEqual(metadata.symbol, 'ETH/USDT')
        self.assertEqual(metadata.rows, 24)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestParquetStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCache))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataMetadata))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
