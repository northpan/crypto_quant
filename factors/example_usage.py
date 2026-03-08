"""
数字货币因子库使用示例
Example Usage of Cryptocurrency Factor Library

演示如何使用因子库进行因子计算和检验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入因子库
from base_factor import FactorTester
from factor_pool import FactorPool, quick_factor_analysis
from technical_factors import create_all_technical_factors, RSIFactor, MACDFactor
from volume_factors import create_all_volume_factors, OBVFactor
from volatility_factors import create_all_volatility_factors, HistoricalVolatilityFactor


def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    生成示例数据
    
    Args:
        n: 数据条数
        seed: 随机种子
        
    Returns:
        pd.DataFrame: 示例OHLCV数据
    """
    np.random.seed(seed)
    
    # 生成时间序列
    dates = pd.date_range(start='2023-01-01', periods=n, freq='H')
    
    # 生成价格序列（随机游走）
    returns = np.random.normal(0.0001, 0.02, n)
    price = 50000 * np.exp(np.cumsum(returns))
    
    # 生成OHLCV
    data = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.001, n)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': price,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    # 确保 high >= max(open, close) 和 low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data


def example_1_basic_usage():
    """示例1: 基础用法 - 计算单个因子"""
    print("\n" + "="*60)
    print("示例1: 基础用法 - 计算单个因子")
    print("="*60)
    
    # 生成示例数据
    data = generate_sample_data(500)
    print(f"\n数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    print(f"\n前5行数据:")
    print(data.head())
    
    # 创建RSI因子
    rsi_factor = RSIFactor(window=14)
    print(f"\n因子信息: {rsi_factor.get_info()}")
    
    # 计算因子值
    rsi_values = rsi_factor.compute(data)
    print(f"\nRSI因子值统计:")
    print(rsi_values.describe())
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(data.index, data['close'], label='Close Price')
    axes[0].set_title('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(data.index, rsi_values, label='RSI Factor', color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('RSI Factor')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/okcomputer/output/crypto_quant/factors/example_1_rsi.png', dpi=150)
    print("\n图表已保存至: example_1_rsi.png")


def example_2_factor_pool():
    """示例2: 使用因子池批量计算"""
    print("\n" + "="*60)
    print("示例2: 使用因子池批量计算")
    print("="*60)
    
    # 生成示例数据
    data = generate_sample_data(500)
    
    # 创建因子池
    pool = FactorPool()
    print(f"\n因子池中的因子数量: {len(pool)}")
    
    # 获取因子统计
    stats = pool.get_factor_stats()
    print(f"\n因子分类统计:")
    print(stats.groupby('category').size())
    
    # 批量计算因子（只计算前10个）
    factor_names = pool.list_factors()[:10]
    print(f"\n计算前10个因子: {factor_names}")
    
    factor_values = pool.compute(data, factor_names=factor_names, verbose=True)
    
    print(f"\n因子值形状: {factor_values.shape}")
    print(f"\n因子值统计:")
    print(factor_values.describe())


def example_3_factor_testing():
    """示例3: 因子有效性检验"""
    print("\n" + "="*60)
    print("示例3: 因子有效性检验")
    print("="*60)
    
    # 生成示例数据
    data = generate_sample_data(500)
    
    # 生成未来收益（用于检验）
    forward_return = data['close'].pct_change(5).shift(-5)  # 未来5期收益
    
    # 创建因子池
    pool = FactorPool()
    
    # 选择几个因子进行检验
    test_factors = ['RSI_14', 'MACD_12_26_9', 'OBV_1', 'HV_20']
    
    # 计算因子值
    factor_values = pool.compute(data, factor_names=test_factors, verbose=False)
    
    # 检验因子
    test_results = pool.test(factor_values, forward_return, verbose=True)
    
    print("\n检验结果:")
    print(test_results[['name', 'ic', 'ic_pvalue', 'abs_ic', 'significant']])
    
    # 筛选有效因子
    selected = pool.select_factors(test_results, min_ic=0.02, max_pvalue=0.1)
    print(f"\n筛选出的有效因子: {selected}")
    
    # 因子排名
    ranked = pool.rank_factors(test_results)
    print("\n因子排名:")
    print(ranked)


def example_4_quick_analysis():
    """示例4: 快速因子分析"""
    print("\n" + "="*60)
    print("示例4: 快速因子分析")
    print("="*60)
    
    # 生成示例数据
    data = generate_sample_data(500)
    
    # 生成未来收益
    forward_return = data['close'].pct_change(5).shift(-5)
    
    # 快速分析（只计算部分因子以加快速度）
    pool = FactorPool()
    
    # 选择部分因子
    sample_factors = [
        'RSI_14', 'RSI_7',
        'MACD_12_26_9',
        'Momentum_10', 'Momentum_20',
        'OBV_1', 'MFI_14',
        'HV_20', 'ATR_14',
        'BuySell_Pressure_20'
    ]
    
    print(f"\n分析 {len(sample_factors)} 个因子...")
    
    # 计算因子
    factor_values = pool.compute(data, factor_names=sample_factors, verbose=True)
    
    # 检验因子
    test_results = pool.test(factor_values, forward_return, verbose=True)
    
    print("\n" + "-"*60)
    print("检验结果汇总:")
    print("-"*60)
    print(test_results.to_string())
    
    # 排名
    ranked = pool.rank_factors(test_results)
    print("\n" + "-"*60)
    print("因子排名 (Top 5):")
    print("-"*60)
    print(ranked.head().to_string())


def example_5_custom_factor():
    """示例5: 自定义因子"""
    print("\n" + "="*60)
    print("示例5: 自定义因子")
    print("="*60)
    
    from base_factor import TechnicalFactor, FactorDirection
    
    # 定义自定义因子
    class CustomMomentumFactor(TechnicalFactor):
        """自定义动量因子 - 价格相对于多周期均线的偏离"""
        
        def __init__(self, windows=[5, 10, 20, 60]):
            super().__init__(
                name=f"Custom_Momentum_{'_'.join(map(str, windows))}",
                description="自定义多周期动量因子",
                direction=FactorDirection.POSITIVE
            )
            self.windows = windows
        
        def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            close = data['close']
            
            # 计算各周期均线
            mas = [close.rolling(w, min_periods=1).mean() for w in self.windows]
            
            # 计算偏离度
            deviations = [(close - ma) / (ma + 1e-8) for ma in mas]
            
            # 加权平均
            weights = np.array([0.4, 0.3, 0.2, 0.1])
            result = sum(w * d for w, d in zip(weights, deviations))
            
            return result
    
    # 生成数据
    data = generate_sample_data(500)
    
    # 创建自定义因子
    custom_factor = CustomMomentumFactor()
    print(f"\n自定义因子信息: {custom_factor.get_info()}")
    
    # 计算因子值
    values = custom_factor.compute(data)
    print(f"\n因子值统计:")
    print(values.describe())


def example_6_factor_correlation():
    """示例6: 因子相关性分析"""
    print("\n" + "="*60)
    print("示例6: 因子相关性分析")
    print("="*60)
    
    # 生成数据
    data = generate_sample_data(500)
    
    # 创建因子池
    pool = FactorPool()
    
    # 选择部分因子
    sample_factors = [
        'RSI_14', 'RSI_7',
        'WilliamsR_14', 'CCI_20',
        'Momentum_10', 'ROC_10',
        'HV_20', 'ATR_14'
    ]
    
    # 计算因子值
    factor_values = pool.compute(data, factor_names=sample_factors, verbose=False)
    
    # 计算相关系数矩阵
    corr_matrix = factor_values.corr()
    
    print("\n因子相关系数矩阵:")
    print(corr_matrix.round(3))
    
    # 找出高度相关的因子对
    print("\n高度相关的因子对 (|correlation| > 0.9):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    # 去除高度相关的因子
    kept_factors = pool.remove_correlated_factors(factor_values, threshold=0.9)
    print(f"\n去除高度相关因子后保留: {kept_factors}")


def example_7_all_factors_count():
    """示例7: 统计所有因子数量"""
    print("\n" + "="*60)
    print("示例7: 统计所有因子数量")
    print("="*60)
    
    from technical_factors import create_all_technical_factors
    from volume_factors import create_all_volume_factors
    from volatility_factors import create_all_volatility_factors
    from orderflow_factors import create_all_orderflow_factors
    from cross_market_factors import create_all_cross_market_factors
    
    # 统计各类因子数量
    technical = create_all_technical_factors()
    volume = create_all_volume_factors()
    volatility = create_all_volatility_factors()
    orderflow = create_all_orderflow_factors()
    cross_market = create_all_cross_market_factors()
    
    print("\n" + "-"*60)
    print("因子分类统计:")
    print("-"*60)
    print(f"技术指标因子: {len(technical)}")
    print(f"量价因子: {len(volume)}")
    print(f"波动率因子: {len(volatility)}")
    print(f"订单流因子: {len(orderflow)}")
    print(f"跨市场因子: {len(cross_market)}")
    print("-"*60)
    print(f"总计: {len(technical) + len(volume) + len(volatility) + len(orderflow) + len(cross_market)} 个因子")
    print("-"*60)
    
    # 打印部分因子名称
    print("\n技术指标因子示例:")
    for f in technical[:10]:
        print(f"  - {f.metadata.name}: {f.metadata.description}")
    
    print("\n量价因子示例:")
    for f in volume[:10]:
        print(f"  - {f.metadata.name}: {f.metadata.description}")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "="*60)
    print("数字货币因子库使用示例")
    print("="*60)
    
    # 运行示例
    example_1_basic_usage()
    example_2_factor_pool()
    example_3_factor_testing()
    example_4_quick_analysis()
    example_5_custom_factor()
    example_6_factor_correlation()
    example_7_all_factors_count()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)


if __name__ == "__main__":
    main()
