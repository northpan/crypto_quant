"""
跨市场因子模块
Cross-Market Factors Module

包含期现价差、基差、跨交易所价差等因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .base_factor import (
    CrossMarketFactor, FactorDirection,
    ema, sma, rolling_std, rolling_max, rolling_min
)


# ==================== 期现价差类因子 ====================

class BasisFactor(CrossMarketFactor):
    """
    基差因子
    
    期货价格与现货价格的差异
    - 正基差：期货溢价，可能看跌
    - 负基差：期货折价，可能看涨
    """
    
    def __init__(self, annualize: bool = True):
        super().__init__(
            name="Basis",
            description="基差因子",
            direction=FactorDirection.NEGATIVE  # 基差反向
        )
        self.annualize = annualize
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有期货和现货价格
        if 'spot_price' in data.columns and 'futures_price' in data.columns:
            spot = data['spot_price']
            futures = data['futures_price']
        elif 'close' in data.columns and 'futures_close' in data.columns:
            spot = data['close']
            futures = data['futures_close']
        else:
            # 使用代理变量（基于价格动量估计基差）
            close = data['close']
            # 估计基差（基于近期价格趋势）
            momentum = close.pct_change(30)
            basis = momentum * close * 0.01
            return -basis / (close + 1e-8)
        
        # 计算基差
        basis = futures - spot
        
        # 标准化为百分比
        basis_pct = basis / (spot + 1e-8)
        
        # 年化
        if self.annualize and 'days_to_expiry' in data.columns:
            dte = data['days_to_expiry']
            basis_pct = basis_pct * (365 / (dte + 1e-8))
        
        return -basis_pct  # 反向因子


class BasisMomentumFactor(CrossMarketFactor):
    """
    基差动量因子
    
    基差的变化趋势
    """
    
    def __init__(self, window: int = 5):
        super().__init__(
            name=f"Basis_Momentum_{window}",
            description="基差动量因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取基差
        if 'spot_price' in data.columns and 'futures_price' in data.columns:
            spot = data['spot_price']
            futures = data['futures_price']
            basis = (futures - spot) / (spot + 1e-8)
        elif 'basis' in data.columns:
            basis = data['basis']
        else:
            # 使用代理变量
            close = data['close']
            basis = close.pct_change(30)
        
        # 基差动量
        basis_momentum = basis.diff(self.window)
        
        return basis_momentum


class BasisZScoreFactor(CrossMarketFactor):
    """
    基差Z-Score因子
    
    基差相对于历史均值的位置
    """
    
    def __init__(self, window: int = 60):
        super().__init__(
            name=f"Basis_ZScore_{window}",
            description="基差Z-Score因子",
            direction=FactorDirection.NEGATIVE  # 极端值反向
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取基差
        if 'spot_price' in data.columns and 'futures_price' in data.columns:
            spot = data['spot_price']
            futures = data['futures_price']
            basis = (futures - spot) / (spot + 1e-8)
        elif 'basis' in data.columns:
            basis = data['basis']
        else:
            # 使用代理变量
            close = data['close']
            basis = close.pct_change(30)
        
        # 计算Z-Score
        basis_mean = basis.rolling(window=self.window, min_periods=1).mean()
        basis_std = basis.rolling(window=self.window, min_periods=1).std()
        zscore = (basis - basis_mean) / (basis_std + 1e-8)
        
        return -zscore  # 反向因子


class ContangoBackwardationFactor(CrossMarketFactor):
    """
    升贴水因子
    
    期货市场的升水/贴水状态
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Contango_Backwardation_{window}",
            description="升贴水因子",
            direction=FactorDirection.NEGATIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取基差
        if 'spot_price' in data.columns and 'futures_price' in data.columns:
            spot = data['spot_price']
            futures = data['futures_price']
            basis = (futures - spot) / (spot + 1e-8)
        elif 'basis' in data.columns:
            basis = data['basis']
        else:
            # 使用代理变量
            close = data['close']
            basis = close.pct_change(30)
        
        # 升贴水状态
        # 正值：升水（Contango），负值：贴水（Backwardation）
        basis_ma = basis.rolling(window=self.window, min_periods=1).mean()
        
        return -basis_ma  # 反向因子


# ==================== 跨期价差类因子 ====================

class CalendarSpreadFactor(CrossMarketFactor):
    """
    跨期价差因子
    
    不同到期日期货合约的价差
    """
    
    def __init__(self):
        super().__init__(
            name="Calendar_Spread",
            description="跨期价差因子",
            direction=FactorDirection.POSITIVE
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有不同到期日合约
        if 'futures_near' in data.columns and 'futures_far' in data.columns:
            near = data['futures_near']
            far = data['futures_far']
        elif 'close' in data.columns and 'futures_next' in data.columns:
            near = data['close']
            far = data['futures_next']
        else:
            # 使用代理变量
            close = data['close']
            # 估计跨期价差（基于价格动量）
            momentum_short = close.pct_change(7)
            momentum_long = close.pct_change(30)
            spread = momentum_short - momentum_long
            return spread
        
        # 跨期价差
        spread = (far - near) / (near + 1e-8)
        
        return spread


class TermStructureFactor(CrossMarketFactor):
    """
    期限结构因子
    
    期货期限结构的斜率
    """
    
    def __init__(self):
        super().__init__(
            name="Term_Structure",
            description="期限结构因子",
            direction=FactorDirection.POSITIVE
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有多个到期日合约
        if all(f'futures_t{i}' in data.columns for i in range(1, 4)):
            t1 = data['futures_t1']
            t2 = data['futures_t2']
            t3 = data['futures_t3']
            
            # 期限结构斜率
            slope1 = (t2 - t1) / (t1 + 1e-8)
            slope2 = (t3 - t2) / (t2 + 1e-8)
            term_structure = (slope1 + slope2) / 2
        else:
            # 使用代理变量
            close = data['close']
            # 基于价格趋势估计期限结构
            short_trend = close.pct_change(7)
            long_trend = close.pct_change(30)
            term_structure = long_trend - short_trend
        
        return term_structure


# ==================== 跨交易所价差类因子 ====================

class ExchangeSpreadFactor(CrossMarketFactor):
    """
    跨交易所价差因子
    
    同一资产在不同交易所的价格差异
    """
    
    def __init__(self, exchange1: str = 'binance', exchange2: str = 'okx'):
        super().__init__(
            name=f"Exchange_Spread_{exchange1}_{exchange2}",
            description="跨交易所价差因子",
            direction=FactorDirection.NEUTRAL
        )
        self.exchange1 = exchange1
        self.exchange2 = exchange2
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有不同交易所价格
        price1_col = f'price_{self.exchange1}'
        price2_col = f'price_{self.exchange2}'
        
        if price1_col in data.columns and price2_col in data.columns:
            price1 = data[price1_col]
            price2 = data[price2_col]
        elif 'close' in data.columns and 'close_exchange2' in data.columns:
            price1 = data['close']
            price2 = data['close_exchange2']
        else:
            # 使用代理变量（基于波动率估计）
            close = data['close']
            volatility = close.pct_change().rolling(window=20, min_periods=1).std()
            spread = volatility * 0.001  # 估计价差
            return spread
        
        # 跨交易所价差
        spread = (price1 - price2) / ((price1 + price2) / 2 + 1e-8)
        
        return spread


class ExchangeVolumeImbalanceFactor(CrossMarketFactor):
    """
    跨交易所成交量不平衡因子
    
    不同交易所成交量的相对差异
    """
    
    def __init__(self, exchange1: str = 'binance', exchange2: str = 'okx'):
        super().__init__(
            name=f"Exchange_Vol_Imbalance_{exchange1}_{exchange2}",
            description="跨交易所成交量不平衡因子",
            direction=FactorDirection.POSITIVE
        )
        self.exchange1 = exchange1
        self.exchange2 = exchange2
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有不同交易所成交量
        vol1_col = f'volume_{self.exchange1}'
        vol2_col = f'volume_{self.exchange2}'
        
        if vol1_col in data.columns and vol2_col in data.columns:
            vol1 = data[vol1_col]
            vol2 = data[vol2_col]
        elif 'volume' in data.columns and 'volume_exchange2' in data.columns:
            vol1 = data['volume']
            vol2 = data['volume_exchange2']
        else:
            # 使用代理变量
            return pd.Series(0, index=data.index)
        
        # 成交量不平衡
        imbalance = (vol1 - vol2) / (vol1 + vol2 + 1e-8)
        
        return imbalance


class ArbitrageOpportunityFactor(CrossMarketFactor):
    """
    套利机会因子
    
    检测跨市场套利机会
    """
    
    def __init__(self, threshold: float = 0.001):
        super().__init__(
            name=f"Arbitrage_Opportunity_{threshold}",
            description="套利机会因子",
            direction=FactorDirection.POSITIVE
        )
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有不同市场价格
        if 'price_exchange1' in data.columns and 'price_exchange2' in data.columns:
            price1 = data['price_exchange1']
            price2 = data['price_exchange2']
        elif 'close' in data.columns and 'close_exchange2' in data.columns:
            price1 = data['close']
            price2 = data['close_exchange2']
        else:
            # 使用代理变量
            return pd.Series(0, index=data.index)
        
        # 价差
        spread = np.abs(price1 - price2) / ((price1 + price2) / 2 + 1e-8)
        
        # 套利机会
        opportunity = pd.Series(0, index=data.index)
        opportunity[spread > self.threshold] = spread[spread > self.threshold]
        
        return opportunity


# ==================== 跨品种价差类因子 ====================

class CryptoPairSpreadFactor(CrossMarketFactor):
    """
    币对价差因子
    
    两个相关加密货币的价格比率
    """
    
    def __init__(self, pair1: str = 'BTC', pair2: str = 'ETH'):
        super().__init__(
            name=f"Pair_Spread_{pair1}_{pair2}",
            description="币对价差因子",
            direction=FactorDirection.NEUTRAL
        )
        self.pair1 = pair1
        self.pair2 = pair2
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有两种资产价格
        if 'price_pair1' in data.columns and 'price_pair2' in data.columns:
            price1 = data['price_pair1']
            price2 = data['price_pair2']
        elif 'close' in data.columns and 'close_pair2' in data.columns:
            price1 = data['close']
            price2 = data['close_pair2']
        else:
            # 使用代理变量
            close = data['close']
            # 估计配对价差（基于自身波动）
            spread = close.pct_change().rolling(window=20, min_periods=1).std()
            return spread
        
        # 价格比率
        ratio = price1 / (price2 + 1e-8)
        
        # 比率的变化
        ratio_ma = ratio.rolling(window=20, min_periods=1).mean()
        ratio_deviation = (ratio - ratio_ma) / (ratio_ma + 1e-8)
        
        return ratio_deviation


class BetaFactor(CrossMarketFactor):
    """
    Beta因子
    
    相对于基准资产的Beta值
    """
    
    def __init__(self, window: int = 60):
        super().__init__(
            name=f"Beta_{window}",
            description="Beta因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 检查是否有基准资产
        if 'benchmark_close' in data.columns:
            benchmark = data['benchmark_close']
        elif 'btc_close' in data.columns:
            benchmark = data['btc_close']
        else:
            # 使用自身作为基准（动量因子）
            returns = close.pct_change()
            momentum = returns.rolling(window=self.window, min_periods=1).mean()
            return momentum
        
        # 计算收益率
        returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        # 滚动计算Beta
        beta = pd.Series(index=close.index, dtype=float)
        
        for i in range(self.window, len(returns) + 1):
            r = returns.iloc[i-self.window:i]
            b = benchmark_returns.iloc[i-self.window:i]
            
            # 计算协方差和方差
            cov = r.cov(b)
            var = b.var()
            
            beta.iloc[i-1] = cov / (var + 1e-8) if var > 0 else 1
        
        beta.iloc[:self.window-1] = 1  # 填充初始值
        
        return beta


class CorrelationFactor(CrossMarketFactor):
    """
    相关性因子
    
    与基准资产的相关性
    """
    
    def __init__(self, window: int = 60):
        super().__init__(
            name=f"Correlation_{window}",
            description="相关性因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 检查是否有基准资产
        if 'benchmark_close' in data.columns:
            benchmark = data['benchmark_close']
        elif 'btc_close' in data.columns:
            benchmark = data['btc_close']
        else:
            # 使用自身滞后作为基准
            benchmark = close.shift(1)
        
        # 计算收益率
        returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        # 滚动相关性
        correlation = returns.rolling(window=self.window, min_periods=1).corr(benchmark_returns)
        
        return correlation


# ==================== 市场微观结构类因子 ====================

class PriceImpactFactor(CrossMarketFactor):
    """
    价格冲击因子
    
    成交量对价格的影响程度
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Price_Impact_{window}",
            description="价格冲击因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化
        price_change = close.diff().abs()
        
        # 成交量
        vol = volume
        
        # 价格冲击（价格变化/成交量）
        impact = price_change / (vol + 1e-8)
        
        # 滚动平均
        impact_ma = impact.rolling(window=self.window, min_periods=1).mean()
        
        return impact_ma


class LiquidityFactor(CrossMarketFactor):
    """
    流动性因子
    
    市场流动性指标
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Liquidity_{window}",
            description="流动性因子",
            direction=FactorDirection.POSITIVE  # 流动性越好越好
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Amihud非流动性指标（反转）
        price_change = close.diff().abs()
        illiquidity = price_change / (volume * close + 1e-8)
        
        # 滚动平均
        illiquidity_ma = illiquidity.rolling(window=self.window, min_periods=1).mean()
        
        # 转换为流动性（取倒数）
        liquidity = 1 / (illiquidity_ma + 1e-8)
        
        return liquidity


class MarketDepthFactor(CrossMarketFactor):
    """
    市场深度因子
    
    订单簿深度指标
    """
    
    def __init__(self, levels: int = 5):
        super().__init__(
            name=f"Market_Depth_{levels}",
            description="市场深度因子",
            direction=FactorDirection.POSITIVE
        )
        self.levels = levels
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        bid_vols = [f'bid_volume_{i}' for i in range(1, self.levels + 1)]
        ask_vols = [f'ask_volume_{i}' for i in range(1, self.levels + 1)]
        
        if all(col in data.columns for col in bid_vols + ask_vols):
            total_depth = sum(data[col] for col in bid_vols + ask_vols)
        else:
            # 使用成交量作为代理
            volume = data['volume']
            total_depth = volume.rolling(window=20, min_periods=1).mean()
        
        return total_depth


# ==================== 跨市场动量类因子 ====================

class CrossMarketMomentumFactor(CrossMarketFactor):
    """
    跨市场动量因子
    
    基于多个市场信息的综合动量
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Cross_Market_Momentum_{window}",
            description="跨市场动量因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 收集所有可用市场的收益率
        returns_list = []
        
        # 主市场
        returns_list.append(close.pct_change(self.window))
        
        # 其他市场
        for col in data.columns:
            if 'close_' in col and col != 'close':
                returns_list.append(data[col].pct_change(self.window))
        
        # 综合动量（平均）
        if len(returns_list) > 1:
            cross_momentum = pd.concat(returns_list, axis=1).mean(axis=1)
        else:
            cross_momentum = returns_list[0]
        
        return cross_momentum


class LeadLagFactor(CrossMarketFactor):
    """
    领先滞后因子
    
    检测市场间的领先滞后关系
    """
    
    def __init__(self, window: int = 20, lag: int = 1):
        super().__init__(
            name=f"Lead_Lag_{window}_{lag}",
            description="领先滞后因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.lag = lag
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 检查是否有领先市场
        if 'lead_close' in data.columns:
            lead = data['lead_close']
        elif 'btc_close' in data.columns:
            lead = data['btc_close']
        else:
            # 使用自身滞后
            lead = close.shift(self.lag)
        
        # 计算收益率
        returns = close.pct_change()
        lead_returns = lead.pct_change()
        
        # 领先市场的滞后收益率
        lead_lag_returns = lead_returns.shift(self.lag)
        
        # 领先滞后因子（领先市场对未来收益的影响）
        lead_lag_factor = lead_lag_returns
        
        return lead_lag_factor


# ==================== 因子工厂函数 ====================

def create_all_cross_market_factors() -> List[CrossMarketFactor]:
    """创建所有跨市场因子实例"""
    factors = []
    
    # 期现价差类因子
    factors.extend([
        BasisFactor(True),
        BasisFactor(False),
        BasisMomentumFactor(5),
        BasisZScoreFactor(60),
        ContangoBackwardationFactor(20),
    ])
    
    # 跨期价差类因子
    factors.extend([
        CalendarSpreadFactor(),
        TermStructureFactor(),
    ])
    
    # 跨交易所价差类因子
    factors.extend([
        ExchangeSpreadFactor('binance', 'okx'),
        ExchangeVolumeImbalanceFactor('binance', 'okx'),
        ArbitrageOpportunityFactor(0.001),
    ])
    
    # 跨品种价差类因子
    factors.extend([
        CryptoPairSpreadFactor('BTC', 'ETH'),
        BetaFactor(60),
        CorrelationFactor(60),
    ])
    
    # 市场微观结构类因子
    factors.extend([
        PriceImpactFactor(20),
        LiquidityFactor(20),
        MarketDepthFactor(5),
    ])
    
    # 跨市场动量类因子
    factors.extend([
        CrossMarketMomentumFactor(20),
        LeadLagFactor(20, 1),
    ])
    
    return factors


# 导出所有因子类
__all__ = [
    'BasisFactor', 'BasisMomentumFactor', 'BasisZScoreFactor', 'ContangoBackwardationFactor',
    'CalendarSpreadFactor', 'TermStructureFactor',
    'ExchangeSpreadFactor', 'ExchangeVolumeImbalanceFactor', 'ArbitrageOpportunityFactor',
    'CryptoPairSpreadFactor', 'BetaFactor', 'CorrelationFactor',
    'PriceImpactFactor', 'LiquidityFactor', 'MarketDepthFactor',
    'CrossMarketMomentumFactor', 'LeadLagFactor',
    'create_all_cross_market_factors'
]
