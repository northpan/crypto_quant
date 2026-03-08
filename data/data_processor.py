"""
数字货币数据清洗和预处理模块
提供数据质量检查、缺失值处理、异常值检测等功能
"""

from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np
from scipy import stats

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"            # 良好
    FAIR = "fair"            # 一般
    POOR = "poor"            # 较差
    CRITICAL = "critical"    # 严重问题


@dataclass
class DataQualityReport:
    """数据质量报告"""
    symbol: str
    timeframe: str
    total_rows: int
    missing_values: Dict[str, int]
    missing_percentage: float
    duplicate_rows: int
    outliers: Dict[str, int]
    gaps: List[Tuple[datetime, datetime]]
    quality_level: DataQualityLevel
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_rows': self.total_rows,
            'missing_values': self.missing_values,
            'missing_percentage': self.missing_percentage,
            'duplicate_rows': self.duplicate_rows,
            'outliers': self.outliers,
            'gaps': [(s.isoformat(), e.isoformat()) for s, e in self.gaps],
            'quality_level': self.quality_level.value,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"DataQualityReport({self.symbol}, {self.timeframe}): "
            f"{self.quality_level.value}, "
            f"missing={self.missing_percentage:.2f}%, "
            f"gaps={len(self.gaps)}"
        )


class DataProcessor:
    """
    数据处理器
    
    功能:
    - 数据质量检查
    - 缺失值处理
    - 异常值检测和处理
    - 数据对齐和重采样
    - 技术指标计算
    """
    
    def __init__(self):
        """初始化数据处理器"""
        self.quality_thresholds = {
            'missing_max': 0.05,      # 最大缺失率 5%
            'outlier_max': 0.01,      # 最大异常值率 1%
            'gap_max_minutes': 5,     # 最大允许间隔（分钟）
        }
    
    def check_quality(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
        timeframe: str = "1m"
    ) -> DataQualityReport:
        """
        检查数据质量
        
        Args:
            df: 输入数据
            symbol: 交易对符号
            timeframe: 时间周期
        
        Returns:
            DataQualityReport: 质量报告
        """
        if df.empty:
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_rows=0,
                missing_values={},
                missing_percentage=100.0,
                duplicate_rows=0,
                outliers={},
                gaps=[],
                quality_level=DataQualityLevel.CRITICAL,
                recommendations=["数据为空，需要重新获取"]
            )
        
        # 1. 检查缺失值
        missing_values = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
        
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(missing_values.values())
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # 2. 检查重复行
        duplicate_rows = int(df.index.duplicated().sum())
        
        # 3. 检测异常值
        outliers = self._detect_outliers(df)
        
        # 4. 检测时间间隔
        gaps = self._detect_time_gaps(df, timeframe)
        
        # 5. 评估质量等级
        quality_level = self._evaluate_quality(
            missing_percentage,
            outliers,
            gaps,
            duplicate_rows,
            len(df)
        )
        
        # 6. 生成建议
        recommendations = self._generate_recommendations(
            missing_values,
            outliers,
            gaps,
            duplicate_rows
        )
        
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_rows=len(df),
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            outliers=outliers,
            gaps=gaps,
            quality_level=quality_level,
            recommendations=recommendations
        )
    
    def _detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> Dict[str, int]:
        """
        检测异常值
        
        Args:
            df: 输入数据
            method: 检测方法 (iqr, zscore)
        
        Returns:
            Dict[str, int]: 各列异常值数量
        """
        outliers = {}
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            if method == "iqr":
                # IQR方法
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
            
            elif method == "zscore":
                # Z-Score方法
                z_scores = np.abs(stats.zscore(series))
                outlier_count = int((z_scores > 3).sum())
            
            else:
                continue
            
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        return outliers
    
    def _detect_time_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        检测时间间隔
        
        Args:
            df: 输入数据
            timeframe: 时间周期
        
        Returns:
            List[Tuple[datetime, datetime]]: 间隔列表
        """
        if len(df) < 2:
            return []
        
        # 获取时间间隔（分钟）
        timeframe_minutes = self._parse_timeframe(timeframe)
        expected_interval = timedelta(minutes=timeframe_minutes)
        max_gap = expected_interval * 2  # 允许2倍间隔
        
        gaps = []
        timestamps = df.index
        
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap > max_gap:
                gaps.append((timestamps[i-1], timestamps[i]))
        
        return gaps
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """解析时间周期为分钟数"""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return mapping.get(timeframe, 1)
    
    def _evaluate_quality(
        self,
        missing_percentage: float,
        outliers: Dict[str, int],
        gaps: List[Tuple],
        duplicate_rows: int,
        total_rows: int
    ) -> DataQualityLevel:
        """评估数据质量等级"""
        score = 100
        
        # 缺失值扣分
        score -= missing_percentage * 2
        
        # 异常值扣分
        total_outliers = sum(outliers.values())
        outlier_percentage = (total_outliers / total_rows) * 100 if total_rows > 0 else 0
        score -= outlier_percentage * 5
        
        # 间隔扣分
        score -= len(gaps) * 5
        
        # 重复值扣分
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        score -= duplicate_percentage * 2
        
        # 确定等级
        if score >= 95:
            return DataQualityLevel.EXCELLENT
        elif score >= 85:
            return DataQualityLevel.GOOD
        elif score >= 70:
            return DataQualityLevel.FAIR
        elif score >= 50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL
    
    def _generate_recommendations(
        self,
        missing_values: Dict[str, int],
        outliers: Dict[str, int],
        gaps: List[Tuple],
        duplicate_rows: int
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if missing_values:
            recommendations.append(f"处理缺失值: {missing_values}")
        
        if outliers:
            recommendations.append(f"检查异常值: {outliers}")
        
        if gaps:
            recommendations.append(f"填补时间间隔: 发现 {len(gaps)} 个间隔")
        
        if duplicate_rows > 0:
            recommendations.append(f"删除重复行: {duplicate_rows} 行")
        
        if not recommendations:
            recommendations.append("数据质量良好，无需处理")
        
        return recommendations
    
    def clean_data(
        self,
        df: pd.DataFrame,
        fill_missing: bool = True,
        remove_outliers: bool = False,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 输入数据
            fill_missing: 是否填充缺失值
            remove_outliers: 是否移除异常值
            remove_duplicates: 是否删除重复行
        
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 1. 删除重复行
        if remove_duplicates:
            before_count = len(df_clean)
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
            removed = before_count - len(df_clean)
            if removed > 0:
                logger.info(f"删除 {removed} 行重复数据")
        
        # 2. 排序
        df_clean = df_clean.sort_index()
        
        # 3. 处理缺失值
        if fill_missing:
            df_clean = self._fill_missing_values(df_clean)
        
        # 4. 处理异常值
        if remove_outliers:
            df_clean = self._remove_outliers(df_clean)
        
        return df_clean
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失值"""
        df_filled = df.copy()
        
        # OHLC价格数据使用线性插值
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].interpolate(method='linear', limit=5)
        
        # 成交量使用前向填充
        if 'volume' in df_filled.columns:
            df_filled['volume'] = df_filled['volume'].fillna(method='ffill', limit=3)
            df_filled['volume'] = df_filled['volume'].fillna(0)
        
        # 其他列使用前向填充
        df_filled = df_filled.fillna(method='ffill', limit=3)
        
        return df_filled
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """移除异常值"""
        df_clean = df.copy()
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            
            series = df_clean[col]
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # 使用中位数替换异常值
                median = series.median()
                df_clean[col] = series.where(
                    (series >= lower_bound) & (series <= upper_bound),
                    median
                )
        
        return df_clean
    
    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        agg_methods: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        重采样数据
        
        Args:
            df: 输入数据
            target_timeframe: 目标时间周期
            agg_methods: 聚合方法
        
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        if df.empty:
            return df
        
        # 默认聚合方法
        default_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        agg_methods = agg_methods or default_agg
        
        # 构建resample规则
        resample_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        rule = resample_map.get(target_timeframe, '1min')
        
        # 选择存在的列
        available_cols = [col for col in agg_methods.keys() if col in df.columns]
        agg_dict = {col: agg_methods[col] for col in available_cols}
        
        # 重采样
        resampled = df[available_cols].resample(rule).agg(agg_dict)
        
        # 删除全为NaN的行
        resampled = resampled.dropna(how='all')
        
        logger.info(f"重采样完成: {len(df)} -> {len(resampled)} 行")
        return resampled
    
    def align_multiple_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        对齐多个交易对的时间索引
        
        Args:
            data_dict: 交易对到数据的映射
            method: 对齐方法 (inner, outer)
        
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据
        """
        if not data_dict:
            return {}
        
        # 获取所有时间索引
        all_indices = [df.index for df in data_dict.values() if not df.empty]
        
        if not all_indices:
            return data_dict
        
        # 计算交集或并集
        if method == "inner":
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
        else:  # outer
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
        
        # 对齐所有数据
        aligned_data = {}
        for symbol, df in data_dict.items():
            if df.empty:
                aligned_data[symbol] = df
                continue
            
            if method == "inner":
                aligned_df = df.loc[df.index.isin(common_index)]
            else:
                aligned_df = df.reindex(common_index)
            
            aligned_data[symbol] = aligned_df
        
        logger.info(f"对齐完成: {len(common_index)} 个共同时间点")
        return aligned_data
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 输入数据
            indicators: 指标列表
        
        Returns:
            pd.DataFrame: 添加指标后的数据
        """
        if df.empty:
            return df
        
        df_with_indicators = df.copy()
        
        # 默认指标
        default_indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands']
        indicators = indicators or default_indicators
        
        # 简单移动平均线
        if 'sma' in indicators:
            for period in [7, 20, 50]:
                df_with_indicators[f'sma_{period}'] = df_with_indicators['close'].rolling(window=period).mean()
        
        # 指数移动平均线
        if 'ema' in indicators:
            for period in [12, 26]:
                df_with_indicators[f'ema_{period}'] = df_with_indicators['close'].ewm(span=period).mean()
        
        # RSI
        if 'rsi' in indicators:
            df_with_indicators['rsi'] = self._calculate_rsi(df_with_indicators['close'])
        
        # MACD
        if 'macd' in indicators:
            macd_line, signal_line, histogram = self._calculate_macd(df_with_indicators['close'])
            df_with_indicators['macd'] = macd_line
            df_with_indicators['macd_signal'] = signal_line
            df_with_indicators['macd_hist'] = histogram
        
        # 布林带
        if 'bbands' in indicators:
            upper, middle, lower = self._calculate_bollinger_bands(df_with_indicators['close'])
            df_with_indicators['bb_upper'] = upper
            df_with_indicators['bb_middle'] = middle
            df_with_indicators['bb_lower'] = lower
        
        # ATR
        if 'atr' in indicators:
            df_with_indicators['atr'] = self._calculate_atr(df_with_indicators)
        
        return df_with_indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        数据标准化
        
        Args:
            df: 输入数据
            method: 标准化方法 (zscore, minmax)
            columns: 要标准化的列
        
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if df.empty:
            return df
        
        df_normalized = df.copy()
        columns = columns or ['open', 'high', 'low', 'close', 'volume']
        columns = [col for col in columns if col in df.columns]
        
        for col in columns:
            series = df[col].dropna()
            
            if method == "zscore":
                mean = series.mean()
                std = series.std()
                if std > 0:
                    df_normalized[col] = (df[col] - mean) / std
            
            elif method == "minmax":
                min_val = series.min()
                max_val = series.max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close"
    ) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            df: 输入数据
            price_col: 价格列名
        
        Returns:
            pd.DataFrame: 添加收益率列的数据
        """
        if df.empty or price_col not in df.columns:
            return df
        
        df_returns = df.copy()
        
        # 简单收益率
        df_returns['returns'] = df[price_col].pct_change()
        
        # 对数收益率
        df_returns['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 累积收益率
        df_returns['cumulative_returns'] = (1 + df_returns['returns']).cumprod() - 1
        
        return df_returns
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        检测价格异常
        
        Args:
            df: 输入数据
            price_col: 价格列名
            window: 滚动窗口大小
            threshold: 异常阈值
        
        Returns:
            pd.DataFrame: 添加异常标记的数据
        """
        if df.empty or price_col not in df.columns:
            return df
        
        df_anomaly = df.copy()
        
        # 计算滚动均值和标准差
        rolling_mean = df[price_col].rolling(window=window).mean()
        rolling_std = df[price_col].rolling(window=window).std()
        
        # 计算Z-Score
        z_score = np.abs((df[price_col] - rolling_mean) / rolling_std)
        
        # 标记异常
        df_anomaly['is_anomaly'] = z_score > threshold
        df_anomaly['z_score'] = z_score
        
        anomaly_count = df_anomaly['is_anomaly'].sum()
        logger.info(f"检测到 {anomaly_count} 个价格异常点")
        
        return df_anomaly


if __name__ == "__main__":
    # 测试代码
    processor = DataProcessor()
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # 添加一些缺失值
    test_df.loc[test_df.sample(10).index, 'close'] = np.nan
    
    # 添加一些重复索引
    test_df = pd.concat([test_df, test_df.iloc[-5:]])
    
    print("原始数据:")
    print(test_df.head(10))
    print(f"\n数据形状: {test_df.shape}")
    
    # 质量检查
    report = processor.check_quality(test_df, "BTC/USDT", "1m")
    print(f"\n质量报告: {report}")
    print(f"质量等级: {report.quality_level.value}")
    print(f"建议: {report.recommendations}")
    
    # 清洗数据
    clean_df = processor.clean_data(test_df)
    print(f"\n清洗后数据形状: {clean_df.shape}")
    
    # 添加技术指标
    df_with_indicators = processor.add_technical_indicators(clean_df, ['sma', 'rsi', 'macd'])
    print("\n添加指标后的数据:")
    print(df_with_indicators[['close', 'sma_7', 'sma_20', 'rsi', 'macd']].tail())
    
    # 计算收益率
    df_returns = processor.calculate_returns(clean_df)
    print("\n收益率:")
    print(df_returns[['close', 'returns', 'log_returns']].head())
