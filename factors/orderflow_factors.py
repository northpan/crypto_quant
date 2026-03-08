"""
订单流因子模块
Order Flow Factors Module

包含买卖压力、订单簿不平衡、资金费率、持仓量变化等因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from .base_factor import (
    OrderFlowFactor, FactorDirection,
    ema, sma, rolling_std, rolling_max, rolling_min, rolling_sum
)


# ==================== 买卖压力类因子 ====================

class BuySellPressureFactor(OrderFlowFactor):
    """
    买卖压力因子
    
    基于价格位置和成交量的买卖压力估计
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"BuySell_Pressure_{window}",
            description="买卖压力因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 价格位置（0-1）
        price_position = (close - low) / (high - low + 1e-8)
        
        # 买卖压力估计
        buying_pressure = price_position * volume
        selling_pressure = (1 - price_position) * volume
        
        # 净压力
        net_pressure = buying_pressure - selling_pressure
        
        # 累积
        pressure_sum = net_pressure.rolling(window=self.window, min_periods=1).sum()
        volume_sum = volume.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        pressure_ratio = pressure_sum / (volume_sum + 1e-8)
        
        return pressure_ratio


class TradeIntensityFactor(OrderFlowFactor):
    """
    交易强度因子
    
    单位时间内的交易活跃度
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Trade_Intensity_{window}",
            description="交易强度因子",
            direction=FactorDirection.NEUTRAL
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        
        # 成交量变化
        volume_change = volume.diff()
        
        # 交易强度（成交量变化的标准化）
        volume_std = volume.rolling(window=self.window, min_periods=1).std()
        intensity = volume_change / (volume_std + 1e-8)
        
        return intensity


class TickRuleFactor(OrderFlowFactor):
    """
    Tick规则因子
    
    基于价格变化的买卖方向分类
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Tick_Rule_{window}",
            description="Tick规则因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化
        price_change = close.diff()
        
        # Tick规则分类
        tick_direction = pd.Series(0, index=close.index)
        tick_direction[price_change > 0] = 1   # 主动买入
        tick_direction[price_change < 0] = -1  # 主动卖出
        
        # 处理平盘情况（使用前一次的值）
        tick_direction = tick_direction.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # 加权成交量
        signed_volume = tick_direction * volume
        
        # 累积
        signed_volume_sum = signed_volume.rolling(window=self.window, min_periods=1).sum()
        volume_sum = volume.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        tick_factor = signed_volume_sum / (volume_sum + 1e-8)
        
        return tick_factor


class LeeReadyFactor(OrderFlowFactor):
    """
    Lee-Ready因子
    
    改进的买卖方向分类方法
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Lee_Ready_{window}",
            description="Lee-Ready因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 中点价格
        mid_price = (high + low) / 2
        
        # Lee-Ready分类
        trade_direction = pd.Series(0, index=close.index)
        trade_direction[close > mid_price] = 1   # 主动买入
        trade_direction[close < mid_price] = -1  # 主动卖出
        
        # 处理等于中点的情况（使用前一次的值）
        trade_direction = trade_direction.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # 加权成交量
        signed_volume = trade_direction * volume
        
        # 累积
        signed_volume_sum = signed_volume.rolling(window=self.window, min_periods=1).sum()
        volume_sum = volume.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        lee_ready_factor = signed_volume_sum / (volume_sum + 1e-8)
        
        return lee_ready_factor


# ==================== 订单簿不平衡类因子 ====================

class OrderBookImbalanceFactor(OrderFlowFactor):
    """
    订单簿不平衡因子
    
    买卖订单簿的不平衡程度
    需要订单簿数据
    """
    
    def __init__(self, level: int = 1):
        super().__init__(
            name=f"OB_Imbalance_L{level}",
            description="订单簿不平衡因子",
            direction=FactorDirection.POSITIVE
        )
        self.level = level
        self.metadata.parameters = {'level': level}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        bid_col = f'bid_volume_{self.level}'
        ask_col = f'ask_volume_{self.level}'
        
        if bid_col not in data.columns or ask_col not in data.columns:
            # 使用代理变量（基于价格位置估计）
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # 估计订单簿不平衡
            price_position = (close - low) / (high - low + 1e-8)
            imbalance = 2 * price_position - 1  # 映射到[-1, 1]
            
            return imbalance
        
        # 使用真实订单簿数据
        bid_volume = data[bid_col]
        ask_volume = data[ask_col]
        
        # 订单簿不平衡
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        
        return imbalance


class OrderBookSlopeFactor(OrderFlowFactor):
    """
    订单簿斜率因子
    
    订单簿深度变化的斜率
    """
    
    def __init__(self, levels: int = 5):
        super().__init__(
            name=f"OB_Slope_{levels}",
            description="订单簿斜率因子",
            direction=FactorDirection.POSITIVE
        )
        self.levels = levels
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        bid_vols = [f'bid_volume_{i}' for i in range(1, self.levels + 1)]
        
        if not all(col in data.columns for col in bid_vols):
            # 使用代理变量
            return pd.Series(0, index=data.index)
        
        # 计算订单簿斜率
        bid_slope = pd.Series(0, index=data.index)
        
        for i in range(len(bid_vols) - 1):
            vol_current = data[bid_vols[i]]
            vol_next = data[bid_vols[i + 1]]
            bid_slope += (vol_next - vol_current) / (vol_current + 1e-8)
        
        return bid_slope / (self.levels - 1)


class OrderBookPressureFactor(OrderFlowFactor):
    """
    订单簿压力因子
    
    基于订单簿深度的买卖压力
    """
    
    def __init__(self, levels: int = 5):
        super().__init__(
            name=f"OB_Pressure_{levels}",
            description="订单簿压力因子",
            direction=FactorDirection.POSITIVE
        )
        self.levels = levels
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        bid_vols = [f'bid_volume_{i}' for i in range(1, self.levels + 1)]
        ask_vols = [f'ask_volume_{i}' for i in range(1, self.levels + 1)]
        
        if not all(col in data.columns for col in bid_vols + ask_vols):
            # 使用代理变量
            return pd.Series(0, index=data.index)
        
        # 计算总深度
        total_bid = sum(data[col] for col in bid_vols)
        total_ask = sum(data[col] for col in ask_vols)
        
        # 压力因子
        pressure = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        
        return pressure


class SpreadFactor(OrderFlowFactor):
    """
    买卖价差因子
    
    反映流动性和交易成本
    """
    
    def __init__(self):
        super().__init__(
            name="Spread",
            description="买卖价差因子",
            direction=FactorDirection.NEGATIVE  # 价差越大，流动性越差
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
            bid = data['bid_price_1']
            ask = data['ask_price_1']
            mid = (bid + ask) / 2
            spread = (ask - bid) / (mid + 1e-8)
        else:
            # 使用代理变量（日内波幅估计）
            high = data['high']
            low = data['low']
            close = data['close']
            spread = (high - low) / (close + 1e-8) / 10  # 估计值
        
        return spread


class DepthImbalanceFactor(OrderFlowFactor):
    """
    深度不平衡因子
    
    订单簿深度的买卖不平衡
    """
    
    def __init__(self, levels: int = 5):
        super().__init__(
            name=f"Depth_Imbalance_{levels}",
            description="深度不平衡因子",
            direction=FactorDirection.POSITIVE
        )
        self.levels = levels
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有订单簿数据
        bid_vols = [f'bid_volume_{i}' for i in range(1, self.levels + 1)]
        ask_vols = [f'ask_volume_{i}' for i in range(1, self.levels + 1)]
        
        if not all(col in data.columns for col in bid_vols + ask_vols):
            # 使用代理变量
            close = data['close']
            volume = data['volume']
            
            # 基于成交量变化的估计
            vol_change = volume.diff()
            imbalance = vol_change / (volume + 1e-8)
            
            return imbalance
        
        # 计算深度不平衡
        bid_depth = sum(data[col] for col in bid_vols)
        ask_depth = sum(data[col] for col in ask_vols)
        
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
        
        return imbalance


# ==================== 资金费率类因子 ====================

class FundingRateFactor(OrderFlowFactor):
    """
    资金费率因子
    
    永续合约的资金费率
    - 正费率：多头支付空头，可能看跌
    - 负费率：空头支付多头，可能看涨
    """
    
    def __init__(self):
        super().__init__(
            name="Funding_Rate",
            description="资金费率因子",
            direction=FactorDirection.NEGATIVE  # 高费率反向
        )
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有资金费率数据
        if 'funding_rate' in data.columns:
            funding_rate = data['funding_rate']
        else:
            # 使用代理变量（基于溢价估计）
            close = data['close']
            
            # 基于价格动量估计资金费率
            momentum = close.pct_change(8)  # 8小时约等于一个资金周期
            funding_rate = momentum * 0.01  # 缩放
        
        return -funding_rate  # 反向


class FundingRateMomentumFactor(OrderFlowFactor):
    """
    资金费率动量因子
    
    资金费率的变化趋势
    """
    
    def __init__(self, window: int = 3):
        super().__init__(
            name=f"Funding_Momentum_{window}",
            description="资金费率动量因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取资金费率
        if 'funding_rate' in data.columns:
            funding_rate = data['funding_rate']
        else:
            close = data['close']
            funding_rate = close.pct_change(8) * 0.01
        
        # 资金费率动量
        funding_momentum = funding_rate.diff(self.window)
        
        return funding_momentum


class FundingRateExtremeFactor(OrderFlowFactor):
    """
    资金费率极值因子
    
    资金费率处于极端水平时的反向信号
    """
    
    def __init__(self, threshold: float = 0.001):
        super().__init__(
            name=f"Funding_Extreme_{threshold}",
            description="资金费率极值因子",
            direction=FactorDirection.POSITIVE
        )
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取资金费率
        if 'funding_rate' in data.columns:
            funding_rate = data['funding_rate']
        else:
            close = data['close']
            funding_rate = close.pct_change(8) * 0.01
        
        # 极端值检测
        extreme = pd.Series(0, index=data.index)
        extreme[funding_rate > self.threshold] = -1  # 过高，看跌
        extreme[funding_rate < -self.threshold] = 1  # 过低，看涨
        
        return extreme


# ==================== 持仓量类因子 ====================

class OpenInterestFactor(OrderFlowFactor):
    """
    持仓量因子
    
    持仓量的变化趋势
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"Open_Interest_{window}",
            description="持仓量因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.metadata.parameters = {'window': window}
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 检查是否有持仓量数据
        if 'open_interest' in data.columns:
            oi = data['open_interest']
        else:
            # 使用成交量作为代理
            oi = data['volume'].cumsum()
        
        # 持仓量变化率
        oi_change = oi.pct_change(self.window)
        
        return oi_change


class OpenInterestPriceFactor(OrderFlowFactor):
    """
    持仓量-价格关系因子
    
    持仓量与价格的关系
    - 价涨仓增：趋势确认
    - 价涨仓减：趋势可能反转
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"OI_Price_{window}",
            description="持仓量-价格关系因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # 获取持仓量
        if 'open_interest' in data.columns:
            oi = data['open_interest']
        else:
            oi = data['volume'].cumsum()
        
        # 价格变化
        price_change = close.pct_change(self.window)
        
        # 持仓量变化
        oi_change = oi.pct_change(self.window)
        
        # 关系因子
        # 同向变化为正，反向变化为负
        relation = np.sign(price_change) * np.sign(oi_change) * np.abs(oi_change)
        
        return relation


class OpenInterestVelocityFactor(OrderFlowFactor):
    """
    持仓量速度因子
    
    持仓量变化的速度
    """
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"OI_Velocity_{window}",
            description="持仓量速度因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 获取持仓量
        if 'open_interest' in data.columns:
            oi = data['open_interest']
        else:
            oi = data['volume'].cumsum()
        
        # 持仓量变化速度
        oi_velocity = oi.diff(self.window) / oi.shift(self.window)
        
        return oi_velocity


# ==================== 大单追踪类因子 ====================

class LargeTradeFactor(OrderFlowFactor):
    """
    大单交易因子
    
    检测大单交易的方向
    """
    
    def __init__(self, window: int = 20, threshold: float = 2.0):
        super().__init__(
            name=f"Large_Trade_{window}_{threshold}",
            description="大单交易因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        close = data['close']
        
        # 成交量均值和标准差
        vol_mean = volume.rolling(window=self.window, min_periods=1).mean()
        vol_std = volume.rolling(window=self.window, min_periods=1).std()
        
        # 大单检测
        large_trade = volume > (vol_mean + self.threshold * vol_std)
        
        # 大单方向（基于价格变化）
        price_change = close.diff()
        large_direction = np.sign(price_change) * large_trade.astype(int)
        
        # 累积
        large_sum = large_direction.rolling(window=self.window, min_periods=1).sum()
        large_count = large_trade.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        large_factor = large_sum / (large_count + 1e-8)
        
        return large_factor


class VolumeWeightedTradeFactor(OrderFlowFactor):
    """
    成交量加权交易因子
    
    按成交量加权的交易方向
    """
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"VW_Trade_{window}",
            description="成交量加权交易因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 价格变化
        price_change = close.diff()
        
        # 成交量加权的交易方向
        weighted_trade = np.sign(price_change) * volume
        
        # 累积
        trade_sum = weighted_trade.rolling(window=self.window, min_periods=1).sum()
        volume_sum = volume.rolling(window=self.window, min_periods=1).sum()
        
        # 标准化
        vw_factor = trade_sum / (volume_sum + 1e-8)
        
        return vw_factor


# ==================== 因子工厂函数 ====================

def create_all_orderflow_factors() -> List[OrderFlowFactor]:
    """创建所有订单流因子实例"""
    factors = []
    
    # 买卖压力类因子
    factors.extend([
        BuySellPressureFactor(20),
        BuySellPressureFactor(10),
        TradeIntensityFactor(20),
        TickRuleFactor(20),
        LeeReadyFactor(20),
    ])
    
    # 订单簿不平衡类因子
    factors.extend([
        OrderBookImbalanceFactor(1),
        OrderBookImbalanceFactor(5),
        OrderBookSlopeFactor(5),
        OrderBookPressureFactor(5),
        SpreadFactor(),
        DepthImbalanceFactor(5),
    ])
    
    # 资金费率类因子
    factors.extend([
        FundingRateFactor(),
        FundingRateMomentumFactor(3),
        FundingRateExtremeFactor(0.001),
    ])
    
    # 持仓量类因子
    factors.extend([
        OpenInterestFactor(20),
        OpenInterestFactor(10),
        OpenInterestPriceFactor(20),
        OpenInterestVelocityFactor(10),
    ])
    
    # 大单追踪类因子
    factors.extend([
        LargeTradeFactor(20, 2.0),
        LargeTradeFactor(20, 3.0),
        VolumeWeightedTradeFactor(20),
    ])
    
    return factors


# 导出所有因子类
__all__ = [
    'BuySellPressureFactor', 'TradeIntensityFactor', 'TickRuleFactor', 'LeeReadyFactor',
    'OrderBookImbalanceFactor', 'OrderBookSlopeFactor', 'OrderBookPressureFactor',
    'SpreadFactor', 'DepthImbalanceFactor',
    'FundingRateFactor', 'FundingRateMomentumFactor', 'FundingRateExtremeFactor',
    'OpenInterestFactor', 'OpenInterestPriceFactor', 'OpenInterestVelocityFactor',
    'LargeTradeFactor', 'VolumeWeightedTradeFactor',
    'create_all_orderflow_factors'
]
