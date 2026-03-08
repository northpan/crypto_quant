"""
数字货币量化因子基类模块
Base Factor Module for Cryptocurrency Quantitative Analysis

提供因子计算的基础框架和通用工具
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class FactorCategory(Enum):
    """因子分类枚举"""
    TECHNICAL = "technical"      # 技术指标因子
    VOLUME = "volume"            # 量价因子
    VOLATILITY = "volatility"    # 波动率因子
    ORDERFLOW = "orderflow"      # 订单流因子
    CROSS_MARKET = "cross_market" # 跨市场因子
    CUSTOM = "custom"            # 自定义因子


class FactorDirection(Enum):
    """因子方向枚举"""
    POSITIVE = 1      # 正向因子（值越大预期收益越高）
    NEGATIVE = -1     # 负向因子（值越小预期收益越高）
    NEUTRAL = 0       # 中性因子（需进一步处理）


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str
    category: FactorCategory
    description: str
    direction: FactorDirection
    parameters: Dict[str, Any] = field(default_factory=dict)
    min_periods: int = 1
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'direction': self.direction.name,
            'parameters': self.parameters,
            'min_periods': self.min_periods,
            'dependencies': self.dependencies
        }


class FactorValidator:
    """因子数据验证器"""
    
    @staticmethod
    def validate_ohlcv(data: pd.DataFrame) -> bool:
        """验证OHLCV数据完整性"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        return True
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> bool:
        """验证价格数据有效性"""
        if 'close' not in data.columns:
            raise ValueError("Missing 'close' column")
        if data['close'].isna().all():
            raise ValueError("All close prices are NaN")
        return True
    
    @staticmethod
    def check_lookback(data: pd.DataFrame, periods: int) -> bool:
        """检查数据是否满足回测期要求"""
        if len(data) < periods:
            raise ValueError(f"Insufficient data: need {periods}, got {len(data)}")
        return True


class BaseFactor(ABC):
    """
    因子基类
    
    所有因子的抽象基类，定义统一的接口和通用方法
    """
    
    def __init__(self, name: str, category: FactorCategory, 
                 description: str = "", direction: FactorDirection = FactorDirection.NEUTRAL):
        self.metadata = FactorMetadata(
            name=name,
            category=category,
            description=description,
            direction=direction
        )
        self.validator = FactorValidator()
        self._cache = {}
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 输入数据DataFrame
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 因子值序列
        """
        pass
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        带验证的因子计算入口
        
        Args:
            data: 输入数据
            **kwargs: 计算参数
            
        Returns:
            pd.Series: 因子值
        """
        # 数据验证
        self._validate_input(data)
        
        # 执行计算
        result = self.calculate(data, **kwargs)
        
        # 后处理
        result = self._post_process(result)
        
        return result
    
    def _validate_input(self, data: pd.DataFrame):
        """验证输入数据"""
        self.validator.validate_price_data(data)
    
    def _post_process(self, series: pd.Series) -> pd.Series:
        """因子后处理（去极值、标准化等）"""
        return series
    
    def winsorize(self, series: pd.Series, lower: float = 0.01, 
                  upper: float = 0.99) -> pd.Series:
        """
        缩尾处理（Winsorization）
        
        Args:
            series: 输入序列
            lower: 下分位数
            upper: 上分位数
            
        Returns:
            pd.Series: 处理后的序列
        """
        q_low = series.quantile(lower)
        q_high = series.quantile(upper)
        return series.clip(lower=q_low, upper=q_high)
    
    def standardize(self, series: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        标准化处理
        
        Args:
            series: 输入序列
            method: 标准化方法 ('zscore', 'minmax', 'rank')
            
        Returns:
            pd.Series: 标准化后的序列
        """
        if method == 'zscore':
            mean = series.mean()
            std = series.std()
            return (series - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            return (series - min_val) / (max_val - min_val + 1e-8)
        elif method == 'rank':
            return series.rank(pct=True)
        else:
            raise ValueError(f"Unknown standardization method: {method}")
    
    def neutralize(self, series: pd.Series, market_value: pd.Series) -> pd.Series:
        """
        市值中性化处理
        
        Args:
            series: 因子值
            market_value: 市值数据
            
        Returns:
            pd.Series: 中性化后的因子
        """
        # 对数市值
        log_mv = np.log(market_value + 1e-8)
        
        # 回归去市值影响
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(log_mv.dropna(), 
                                                      series.loc[log_mv.dropna().index])
        residual = series - (intercept + slope * log_mv)
        return residual
    
    def get_info(self) -> Dict:
        """获取因子信息"""
        return self.metadata.to_dict()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.metadata.name}')"


class TechnicalFactor(BaseFactor):
    """技术指标因子基类"""
    
    def __init__(self, name: str, description: str = "", 
                 direction: FactorDirection = FactorDirection.NEUTRAL):
        super().__init__(name, FactorCategory.TECHNICAL, description, direction)
    
    def _validate_input(self, data: pd.DataFrame):
        """技术指标需要OHLCV数据"""
        self.validator.validate_ohlcv(data)


class VolumeFactor(BaseFactor):
    """量价因子基类"""
    
    def __init__(self, name: str, description: str = "", 
                 direction: FactorDirection = FactorDirection.NEUTRAL):
        super().__init__(name, FactorCategory.VOLUME, description, direction)
    
    def _validate_input(self, data: pd.DataFrame):
        """量价因子需要OHLCV数据"""
        self.validator.validate_ohlcv(data)


class VolatilityFactor(BaseFactor):
    """波动率因子基类"""
    
    def __init__(self, name: str, description: str = "", 
                 direction: FactorDirection = FactorDirection.NEUTRAL):
        super().__init__(name, FactorCategory.VOLATILITY, description, direction)
    
    def _validate_input(self, data: pd.DataFrame):
        """波动率因子需要OHLC数据"""
        required = ['high', 'low', 'close']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")


class OrderFlowFactor(BaseFactor):
    """订单流因子基类"""
    
    def __init__(self, name: str, description: str = "", 
                 direction: FactorDirection = FactorDirection.NEUTRAL):
        super().__init__(name, FactorCategory.ORDERFLOW, description, direction)


class CrossMarketFactor(BaseFactor):
    """跨市场因子基类"""
    
    def __init__(self, name: str, description: str = "", 
                 direction: FactorDirection = FactorDirection.NEUTRAL):
        super().__init__(name, FactorCategory.CROSS_MARKET, description, direction)


# ==================== 通用工具函数 ====================

def safe_divide(a: pd.Series, b: pd.Series, fill_value: float = 0) -> pd.Series:
    """安全除法，避免除以零"""
    return a / (b + 1e-8)


def rolling_apply(series: pd.Series, window: int, func: Callable) -> pd.Series:
    """滚动窗口应用函数"""
    return series.rolling(window=window, min_periods=1).apply(func, raw=True)


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """滚动标准差"""
    return series.rolling(window=window, min_periods=1).std()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """滚动最大值"""
    return series.rolling(window=window, min_periods=1).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """滚动最小值"""
    return series.rolling(window=window, min_periods=1).min()


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """滚动求和"""
    return series.rolling(window=window, min_periods=1).sum()


def price_change(close: pd.Series, periods: int = 1) -> pd.Series:
    """价格变化率"""
    return close.pct_change(periods=periods)


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """对数收益率"""
    return np.log(close / close.shift(periods))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均真实波幅 ATR"""
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=1).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指数 RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, 
         signal: int = 9) -> Dict[str, pd.Series]:
    """MACD指标"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def bollinger_bands(close: pd.Series, window: int = 20, 
                    num_std: float = 2.0) -> Dict[str, pd.Series]:
    """布林带"""
    middle = sma(close, window)
    std = rolling_std(close, window)
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': (upper - lower) / (middle + 1e-8),
        'percent_b': (close - lower) / (upper - lower + 1e-8)
    }


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """随机指标 Stochastic Oscillator"""
    lowest_low = rolling_min(low, k_window)
    highest_high = rolling_max(high, k_window)
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
    d = sma(k, d_window)
    
    return {'k': k, 'd': d}


# ==================== 因子检验工具 ====================

class FactorTester:
    """因子有效性检验工具"""
    
    def __init__(self):
        self.results = {}
    
    def ic_analysis(self, factor: pd.Series, forward_return: pd.Series,
                    method: str = 'spearman') -> Dict:
        """
        IC分析（信息系数）
        
        Args:
            factor: 因子值
            forward_return: 未来收益
            method: 相关系数方法
            
        Returns:
            Dict: IC统计结果
        """
        from scipy import stats
        
        # 对齐数据
        common_idx = factor.dropna().index.intersection(forward_return.dropna().index)
        f = factor.loc[common_idx]
        r = forward_return.loc[common_idx]
        
        if method == 'spearman':
            ic, p_value = stats.spearmanr(f, r)
        else:
            ic, p_value = stats.pearsonr(f, r)
        
        return {
            'ic': ic,
            'p_value': p_value,
            'abs_ic': abs(ic),
            'significant': p_value < 0.05
        }
    
    def quantile_return(self, factor: pd.Series, forward_return: pd.Series,
                        n_quantiles: int = 5) -> pd.DataFrame:
        """
        分位数收益分析
        
        Args:
            factor: 因子值
            forward_return: 未来收益
            n_quantiles: 分位数数量
            
        Returns:
            pd.DataFrame: 各分位数收益统计
        """
        # 对齐数据
        common_idx = factor.dropna().index.intersection(forward_return.dropna().index)
        f = factor.loc[common_idx]
        r = forward_return.loc[common_idx]
        
        # 分位数分组
        labels = range(1, n_quantiles + 1)
        quantiles = pd.qcut(f, n_quantiles, labels=labels, duplicates='drop')
        
        # 计算各组收益
        result = r.groupby(quantiles).agg(['mean', 'std', 'count'])
        result.columns = ['mean_return', 'std_return', 'count']
        
        return result
    
    def turnover_analysis(self, factor: pd.Series, 
                          threshold: float = 0.1) -> Dict:
        """
        因子换手率分析
        
        Args:
            factor: 因子值序列
            threshold: 换手率计算阈值
            
        Returns:
            Dict: 换手率统计
        """
        # 计算因子变化
        factor_change = factor.diff().abs()
        
        # 计算换手率
        turnover = factor_change / (factor.abs() + 1e-8)
        
        return {
            'mean_turnover': turnover.mean(),
            'max_turnover': turnover.max(),
            'turnover_std': turnover.std(),
            'high_turnover_ratio': (turnover > threshold).mean()
        }
    
    def run_all_tests(self, factor: pd.Series, forward_return: pd.Series,
                      n_quantiles: int = 5) -> Dict:
        """运行全部检验"""
        return {
            'ic_analysis': self.ic_analysis(factor, forward_return),
            'quantile_return': self.quantile_return(factor, forward_return, n_quantiles),
            'turnover': self.turnover_analysis(factor)
        }


# 导出所有公共接口
__all__ = [
    'BaseFactor', 'TechnicalFactor', 'VolumeFactor', 
    'VolatilityFactor', 'OrderFlowFactor', 'CrossMarketFactor',
    'FactorCategory', 'FactorDirection', 'FactorMetadata',
    'FactorValidator', 'FactorTester',
    'safe_divide', 'rolling_apply', 'ema', 'sma', 
    'rolling_std', 'rolling_max', 'rolling_min', 'rolling_sum',
    'price_change', 'log_return', 'true_range', 'atr',
    'rsi', 'macd', 'bollinger_bands', 'stochastic'
]
