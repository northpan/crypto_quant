"""
数字货币量化因子库
Cryptocurrency Quantitative Factor Library

提供50+个数字货币量化因子，涵盖：
- 技术指标因子
- 量价因子
- 波动率因子
- 订单流因子
- 跨市场因子

作者: AI Quant Researcher
版本: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'AI Quant Researcher'

# 基础模块
from .base_factor import (
    BaseFactor, TechnicalFactor, VolumeFactor, 
    VolatilityFactor, OrderFlowFactor, CrossMarketFactor,
    FactorCategory, FactorDirection, FactorMetadata,
    FactorValidator, FactorTester,
    safe_divide, rolling_apply, ema, sma, 
    rolling_std, rolling_max, rolling_min, rolling_sum,
    price_change, log_return, true_range, atr,
    rsi, macd, bollinger_bands, stochastic
)

# 技术指标因子
from .technical_factors import (
    MACDFactor, MACDSignalFactor, EMAcrossFactor, SMAcrossFactor,
    ADXFactor, TrendStrengthFactor,
    RSIFactor, RSIDivergenceFactor, CCIFactor, WilliamsRFactor,
    StochasticFactor, StochasticCrossFactor, MomentumFactor,
    RateOfChangeFactor, PriceAccelerationFactor,
    BollingerPositionFactor, BollingerWidthFactor, BollingerSqueezeFactor,
    ATRFactor, ATRRatioFactor, KeltnerPositionFactor,
    VolatilityRegimeFactor, DonchianChannelFactor, IchimokuFactor,
    ParabolicSARFactor, create_all_technical_factors
)

# 量价因子
from .volume_factors import (
    OBVFactor, OBVVelocityFactor, OBVDivergenceFactor,
    MFIFactor, MFIVelocityFactor,
    VWAPFactor, VWAPDeviationFactor, VWAPTrendFactor,
    VolumeRatioFactor, VolumeTrendFactor, VolumeOscillatorFactor,
    VolumePriceDivergenceFactor, VolumePriceTrendFactor, VolumePriceConfirmationFactor,
    MoneyFlowFactor, BuyingPressureFactor, AccumulationDistributionFactor,
    ChaikinOscillatorFactor, ForceIndexFactor, EaseOfMovementFactor,
    NegativeVolumeIndexFactor, PositiveVolumeIndexFactor,
    create_all_volume_factors
)

# 波动率因子
from .volatility_factors import (
    HistoricalVolatilityFactor, RealizedVolatilityFactor, CloseToCloseVolatilityFactor,
    ParkinsonVolatilityFactor, GarmanKlassVolatilityFactor,
    RogersSatchellVolatilityFactor, YangZhangVolatilityFactor,
    GARCHVolatilityFactor, EWMAVolatilityFactor,
    VolatilityConeFactor, VolatilityPercentileFactor,
    VolatilityRegimeFactor, VolatilityTrendFactor,
    VolatilitySkewnessFactor, VolatilityKurtosisFactor,
    JumpVolatilityFactor, IntradayRangeFactor, OvernightGapFactor,
    create_all_volatility_factors
)

# 订单流因子
from .orderflow_factors import (
    BuySellPressureFactor, TradeIntensityFactor, TickRuleFactor, LeeReadyFactor,
    OrderBookImbalanceFactor, OrderBookSlopeFactor, OrderBookPressureFactor,
    SpreadFactor, DepthImbalanceFactor,
    FundingRateFactor, FundingRateMomentumFactor, FundingRateExtremeFactor,
    OpenInterestFactor, OpenInterestPriceFactor, OpenInterestVelocityFactor,
    LargeTradeFactor, VolumeWeightedTradeFactor,
    create_all_orderflow_factors
)

# 跨市场因子
from .cross_market_factors import (
    BasisFactor, BasisMomentumFactor, BasisZScoreFactor, ContangoBackwardationFactor,
    CalendarSpreadFactor, TermStructureFactor,
    ExchangeSpreadFactor, ExchangeVolumeImbalanceFactor, ArbitrageOpportunityFactor,
    CryptoPairSpreadFactor, BetaFactor, CorrelationFactor,
    PriceImpactFactor, LiquidityFactor, MarketDepthFactor,
    CrossMarketMomentumFactor, LeadLagFactor,
    create_all_cross_market_factors
)

# 因子池管理
from .factor_pool import (
    FactorPool, FactorPipeline, FactorResult, FactorTestResult,
    create_factor_pool, compute_all_factors, test_factors,
    get_top_factors, quick_factor_analysis
)

# 版本信息
def get_version():
    """获取版本信息"""
    return __version__

# 获取因子数量统计
def get_factor_stats():
    """获取因子统计信息"""
    technical = len(create_all_technical_factors())
    volume = len(create_all_volume_factors())
    volatility = len(create_all_volatility_factors())
    orderflow = len(create_all_orderflow_factors())
    cross_market = len(create_all_cross_market_factors())
    
    return {
        'technical': technical,
        'volume': volume,
        'volatility': volatility,
        'orderflow': orderflow,
        'cross_market': cross_market,
        'total': technical + volume + volatility + orderflow + cross_market
    }

# 打印因子统计信息
def print_factor_stats():
    """打印因子统计信息"""
    stats = get_factor_stats()
    print("=" * 50)
    print("数字货币量化因子库统计")
    print("=" * 50)
    print(f"技术指标因子: {stats['technical']}")
    print(f"量价因子: {stats['volume']}")
    print(f"波动率因子: {stats['volatility']}")
    print(f"订单流因子: {stats['orderflow']}")
    print(f"跨市场因子: {stats['cross_market']}")
    print("-" * 50)
    print(f"总计: {stats['total']} 个因子")
    print("=" * 50)

# 导出所有公共接口
__all__ = [
    # 版本
    '__version__', 'get_version',
    
    # 基础
    'BaseFactor', 'TechnicalFactor', 'VolumeFactor', 
    'VolatilityFactor', 'OrderFlowFactor', 'CrossMarketFactor',
    'FactorCategory', 'FactorDirection', 'FactorMetadata',
    'FactorValidator', 'FactorTester',
    
    # 工具函数
    'safe_divide', 'rolling_apply', 'ema', 'sma', 
    'rolling_std', 'rolling_max', 'rolling_min', 'rolling_sum',
    'price_change', 'log_return', 'true_range', 'atr',
    'rsi', 'macd', 'bollinger_bands', 'stochastic',
    
    # 技术指标因子
    'MACDFactor', 'MACDSignalFactor', 'EMAcrossFactor', 'SMAcrossFactor',
    'ADXFactor', 'TrendStrengthFactor',
    'RSIFactor', 'RSIDivergenceFactor', 'CCIFactor', 'WilliamsRFactor',
    'StochasticFactor', 'StochasticCrossFactor', 'MomentumFactor',
    'RateOfChangeFactor', 'PriceAccelerationFactor',
    'BollingerPositionFactor', 'BollingerWidthFactor', 'BollingerSqueezeFactor',
    'ATRFactor', 'ATRRatioFactor', 'KeltnerPositionFactor',
    'VolatilityRegimeFactor', 'DonchianChannelFactor', 'IchimokuFactor',
    'ParabolicSARFactor', 'create_all_technical_factors',
    
    # 量价因子
    'OBVFactor', 'OBVVelocityFactor', 'OBVDivergenceFactor',
    'MFIFactor', 'MFIVelocityFactor',
    'VWAPFactor', 'VWAPDeviationFactor', 'VWAPTrendFactor',
    'VolumeRatioFactor', 'VolumeTrendFactor', 'VolumeOscillatorFactor',
    'VolumePriceDivergenceFactor', 'VolumePriceTrendFactor', 'VolumePriceConfirmationFactor',
    'MoneyFlowFactor', 'BuyingPressureFactor', 'AccumulationDistributionFactor',
    'ChaikinOscillatorFactor', 'ForceIndexFactor', 'EaseOfMovementFactor',
    'NegativeVolumeIndexFactor', 'PositiveVolumeIndexFactor',
    'create_all_volume_factors',
    
    # 波动率因子
    'HistoricalVolatilityFactor', 'RealizedVolatilityFactor', 'CloseToCloseVolatilityFactor',
    'ParkinsonVolatilityFactor', 'GarmanKlassVolatilityFactor',
    'RogersSatchellVolatilityFactor', 'YangZhangVolatilityFactor',
    'GARCHVolatilityFactor', 'EWMAVolatilityFactor',
    'VolatilityConeFactor', 'VolatilityPercentileFactor',
    'VolatilityRegimeFactor', 'VolatilityTrendFactor',
    'VolatilitySkewnessFactor', 'VolatilityKurtosisFactor',
    'JumpVolatilityFactor', 'IntradayRangeFactor', 'OvernightGapFactor',
    'create_all_volatility_factors',
    
    # 订单流因子
    'BuySellPressureFactor', 'TradeIntensityFactor', 'TickRuleFactor', 'LeeReadyFactor',
    'OrderBookImbalanceFactor', 'OrderBookSlopeFactor', 'OrderBookPressureFactor',
    'SpreadFactor', 'DepthImbalanceFactor',
    'FundingRateFactor', 'FundingRateMomentumFactor', 'FundingRateExtremeFactor',
    'OpenInterestFactor', 'OpenInterestPriceFactor', 'OpenInterestVelocityFactor',
    'LargeTradeFactor', 'VolumeWeightedTradeFactor',
    'create_all_orderflow_factors',
    
    # 跨市场因子
    'BasisFactor', 'BasisMomentumFactor', 'BasisZScoreFactor', 'ContangoBackwardationFactor',
    'CalendarSpreadFactor', 'TermStructureFactor',
    'ExchangeSpreadFactor', 'ExchangeVolumeImbalanceFactor', 'ArbitrageOpportunityFactor',
    'CryptoPairSpreadFactor', 'BetaFactor', 'CorrelationFactor',
    'PriceImpactFactor', 'LiquidityFactor', 'MarketDepthFactor',
    'CrossMarketMomentumFactor', 'LeadLagFactor',
    'create_all_cross_market_factors',
    
    # 因子池
    'FactorPool', 'FactorPipeline', 'FactorResult', 'FactorTestResult',
    'create_factor_pool', 'compute_all_factors', 'test_factors',
    'get_top_factors', 'quick_factor_analysis',
    
    # 统计
    'get_factor_stats', 'print_factor_stats'
]
