"""
数字货币数据模块使用示例
展示如何使用各个组件进行数据获取、处理和存储
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入模块组件
from crypto_quant.data import (
    DataManager,
    DataCollector,
    DataProcessor,
    ParquetStorage,
    ExchangeConfig,
    MarketType,
    MultiSymbolDataManager,
)


def example_1_basic_data_fetch():
    """
    示例1: 基础数据获取
    展示如何获取单个币种的K线数据
    """
    print("=" * 60)
    print("示例1: 基础数据获取")
    print("=" * 60)
    
    # 创建数据管理器
    manager = DataManager(
        data_path="./example_data",
        exchange_name="binance",
        market_type=MarketType.SPOT
    )
    
    try:
        # 获取BTC 1小时数据（最近7天）
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        df = manager.get_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.head())
        print("\n数据统计:")
        print(df.describe())
        
    finally:
        manager.close()


def example_2_batch_fetch():
    """
    示例2: 批量数据获取
    展示如何同时获取多个币种和多个时间周期的数据
    """
    print("\n" + "=" * 60)
    print("示例2: 批量数据获取")
    print("=" * 60)
    
    manager = DataManager(
        data_path="./example_data",
        exchange_name="binance"
    )
    
    try:
        # 定义要获取的币种和周期
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        timeframes = ["1m", "5m", "15m"]
        
        # 定义进度回调
        def progress_callback(symbol, timeframe, current, total):
            print(f"进度: {current}/{total} - {symbol} {timeframe}")
        
        # 批量获取数据
        data = manager.batch_fetch(
            symbols=symbols,
            timeframes=timeframes,
            start_time=datetime.now() - timedelta(days=1),
            progress_callback=progress_callback
        )
        
        # 查看获取结果
        for symbol in symbols:
            print(f"\n{symbol}:")
            for timeframe in timeframes:
                df = data[symbol][timeframe]
                print(f"  {timeframe}: {len(df)} 条数据")
                
    finally:
        manager.close()


def example_3_data_quality_check():
    """
    示例3: 数据质量检查
    展示如何检查和处理数据质量问题
    """
    print("\n" + "=" * 60)
    print("示例3: 数据质量检查")
    print("=" * 60)
    
    processor = DataProcessor()
    
    # 创建测试数据（包含一些问题）
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # 添加一些问题
    # 1. 缺失值
    test_df.loc[test_df.sample(20).index, 'close'] = np.nan
    
    # 2. 重复行
    test_df = pd.concat([test_df, test_df.iloc[-10:]])
    
    # 3. 异常值
    test_df.loc[test_df.sample(5).index, 'close'] = test_df['close'].mean() * 2
    
    print(f"原始数据: {len(test_df)} 行")
    
    # 质量检查
    report = processor.check_quality(test_df, "BTC/USDT", "1m")
    print(f"\n质量报告:")
    print(f"  质量等级: {report.quality_level.value}")
    print(f"  缺失值: {report.missing_values}")
    print(f"  缺失率: {report.missing_percentage:.2f}%")
    print(f"  重复行: {report.duplicate_rows}")
    print(f"  异常值: {report.outliers}")
    print(f"  时间间隔: {len(report.gaps)} 个")
    print(f"  建议: {report.recommendations}")
    
    # 清洗数据
    clean_df = processor.clean_data(test_df)
    print(f"\n清洗后数据: {len(clean_df)} 行")
    
    # 再次检查质量
    new_report = processor.check_quality(clean_df, "BTC/USDT", "1m")
    print(f"清洗后质量等级: {new_report.quality_level.value}")


def example_4_technical_indicators():
    """
    示例4: 技术指标计算
    展示如何添加技术指标到数据中
    """
    print("\n" + "=" * 60)
    print("示例4: 技术指标计算")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 50000,
        'high': np.random.randn(200).cumsum() + 50100,
        'low': np.random.randn(200).cumsum() + 49900,
        'close': np.random.randn(200).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    processor = DataProcessor()
    
    # 添加技术指标
    df_with_indicators = processor.add_technical_indicators(
        test_df,
        indicators=['sma', 'ema', 'rsi', 'macd', 'bbands']
    )
    
    print("添加的指标列:")
    indicator_cols = [col for col in df_with_indicators.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in indicator_cols:
        print(f"  - {col}")
    
    print("\n数据预览:")
    print(df_with_indicators[indicator_cols].tail())


def example_5_multi_timeframe_analysis():
    """
    示例5: 多时间周期分析
    展示如何获取并对齐多个时间周期的数据
    """
    print("\n" + "=" * 60)
    print("示例5: 多时间周期分析")
    print("=" * 60)
    
    manager = DataManager(
        data_path="./example_data",
        exchange_name="binance"
    )
    
    try:
        # 获取多个时间周期的数据
        symbol = "BTC/USDT"
        timeframes = ["1m", "5m", "15m"]
        
        aligned_data = manager.align_timeframes(
            symbol=symbol,
            timeframes=timeframes,
            start_time=datetime.now() - timedelta(days=1),
            method="inner"
        )
        
        print(f"对齐后的数据:")
        for tf, df in aligned_data.items():
            print(f"  {tf}: {len(df)} 行")
        
        # 计算不同周期的收益率相关性
        returns = {}
        for tf, df in aligned_data.items():
            if not df.empty and 'close' in df.columns:
                returns[tf] = df['close'].pct_change()
        
        if returns:
            returns_df = pd.DataFrame(returns)
            correlation = returns_df.corr()
            print(f"\n收益率相关性:")
            print(correlation)
            
    finally:
        manager.close()


def example_6_data_storage():
    """
    示例6: 数据存储和加载
    展示如何使用Parquet存储数据
    """
    print("\n" + "=" * 60)
    print("示例6: 数据存储和加载")
    print("=" * 60)
    
    storage = ParquetStorage("./example_data")
    
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
    
    # 保存数据
    metadata = storage.save_data(
        df=test_df,
        symbol="BTC/USDT",
        timeframe="1m",
        market_type="spot",
        exchange="binance"
    )
    
    print(f"数据已保存:")
    print(f"  文件: {metadata.file_path}")
    print(f"  行数: {metadata.rows}")
    print(f"  时间范围: {metadata.start_time} 到 {metadata.end_time}")
    
    # 加载数据
    loaded_df = storage.load_data("BTC/USDT", "1m", "spot")
    print(f"\n加载数据: {len(loaded_df)} 行")
    
    # 获取存储统计
    stats = storage.get_storage_size()
    print(f"\n存储统计:")
    print(f"  文件数: {stats['file_count']}")
    print(f"  总大小: {stats['total_mb']} MB")
    
    # 清理
    storage.delete_data("BTC/USDT", "1m", "spot")
    print("\n数据已删除")


def example_7_futures_data():
    """
    示例7: 期货数据获取
    展示如何获取永续合约数据
    """
    print("\n" + "=" * 60)
    print("示例7: 期货数据获取")
    print("=" * 60)
    
    # 创建期货数据管理器
    manager = DataManager(
        data_path="./example_data_futures",
        exchange_name="binance",
        market_type=MarketType.PERPETUAL
    )
    
    try:
        # 获取BTC永续合约数据
        df = manager.get_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_time=datetime.now() - timedelta(days=3)
        )
        
        print(f"获取到 {len(df)} 条BTC永续合约数据")
        print("\n数据预览:")
        print(df.head())
        
    finally:
        manager.close()


def example_8_correlation_analysis():
    """
    示例8: 币种相关性分析
    展示如何分析多个币种之间的相关性
    """
    print("\n" + "=" * 60)
    print("示例8: 币种相关性分析")
    print("=" * 60)
    
    multi_manager = MultiSymbolDataManager(
        data_path="./example_data",
        exchange_name="binance"
    )
    
    try:
        # 设置关注的币种
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        multi_manager.set_symbols(symbols)
        multi_manager.set_timeframes(["1h"])
        
        # 获取数据
        print("正在获取数据...")
        multi_manager.fetch_all(days=7)
        
        # 计算相关性矩阵
        correlation = multi_manager.get_correlation_matrix(timeframe="1h")
        
        print("\n币种收益率相关性矩阵:")
        print(correlation.round(3))
        
        # 找出相关性最高的币种对
        corr_pairs = []
        for i in range(len(correlation.columns)):
            for j in range(i+1, len(correlation.columns)):
                symbol1 = correlation.columns[i]
                symbol2 = correlation.columns[j]
                corr_value = correlation.iloc[i, j]
                corr_pairs.append((symbol1, symbol2, corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("\n相关性最高的币种对:")
        for symbol1, symbol2, corr in corr_pairs[:3]:
            print(f"  {symbol1} - {symbol2}: {corr:.3f}")
            
    finally:
        multi_manager.close()


def example_9_incremental_update():
    """
    示例9: 增量更新
    展示如何增量更新已有数据
    """
    print("\n" + "=" * 60)
    print("示例9: 增量更新")
    print("=" * 60)
    
    manager = DataManager(
        data_path="./example_data",
        exchange_name="binance"
    )
    
    try:
        # 首次获取数据（3天前）
        print("首次获取数据...")
        df1 = manager.get_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_time=datetime.now() - timedelta(days=3)
        )
        print(f"首次获取: {len(df1)} 行")
        
        # 获取存储的元数据
        info = manager.get_data_info()
        print(f"\n存储信息:")
        print(info)
        
        # 再次获取（会自动增量更新）
        print("\n再次获取（增量更新）...")
        df2 = manager.get_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_time=datetime.now() - timedelta(days=3)
        )
        print(f"更新后: {len(df2)} 行")
        
    finally:
        manager.close()


def example_10_advanced_processing():
    """
    示例10: 高级数据处理
    展示数据标准化、异常检测等高级功能
    """
    print("\n" + "=" * 60)
    print("示例10: 高级数据处理")
    print("=" * 60)
    
    processor = DataProcessor()
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 50000,
        'high': np.random.randn(500).cumsum() + 50100,
        'low': np.random.randn(500).cumsum() + 49900,
        'close': np.random.randn(500).cumsum() + 50000,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    print("原始数据统计:")
    print(test_df.describe())
    
    # 数据标准化 (Z-Score)
    normalized_df = processor.normalize_data(test_df, method="zscore")
    print("\n标准化后 (Z-Score):")
    print(normalized_df.describe())
    
    # 数据标准化 (Min-Max)
    minmax_df = processor.normalize_data(test_df, method="minmax")
    print("\n标准化后 (Min-Max):")
    print(f"  Close范围: [{minmax_df['close'].min():.3f}, {minmax_df['close'].max():.3f}]")
    
    # 计算收益率
    returns_df = processor.calculate_returns(test_df)
    print("\n收益率统计:")
    print(returns_df[['returns', 'log_returns']].describe())
    
    # 异常检测
    anomaly_df = processor.detect_anomalies(test_df, threshold=2.5)
    anomaly_count = anomaly_df['is_anomaly'].sum()
    print(f"\n检测到 {anomaly_count} 个异常点")
    
    if anomaly_count > 0:
        print("异常点:")
        print(anomaly_df[anomaly_df['is_anomaly']][['close', 'z_score']])


def run_all_examples():
    """运行所有示例"""
    examples = [
        ("基础数据获取", example_1_basic_data_fetch),
        ("批量数据获取", example_2_batch_fetch),
        ("数据质量检查", example_3_data_quality_check),
        ("技术指标计算", example_4_technical_indicators),
        ("多时间周期分析", example_5_multi_timeframe_analysis),
        ("数据存储和加载", example_6_data_storage),
        ("期货数据获取", example_7_futures_data),
        ("币种相关性分析", example_8_correlation_analysis),
        ("增量更新", example_9_incremental_update),
        ("高级数据处理", example_10_advanced_processing),
    ]
    
    print("\n" + "=" * 60)
    print("数字货币数据模块使用示例")
    print("=" * 60)
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n示例 '{name}' 出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
