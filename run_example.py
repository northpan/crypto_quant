#!/usr/bin/env python3
"""
数字货币量化交易系统 - 示例运行脚本
展示如何使用各个模块
"""

import asyncio
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("数字货币量化交易系统 - 示例演示")
print("=" * 70)


async def demo_data_module():
    """演示数据模块"""
    print("\n【1. 数据模块演示】")
    print("-" * 50)
    
    try:
        from data import DataProcessor
        
        # 创建示例数据
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        # 生成模拟价格数据
        returns = np.random.normal(0.0001, 0.001, 1000)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0001, 1000)),
            'high': prices * (1 + abs(np.random.normal(0, 0.001, 1000))),
            'low': prices * (1 - abs(np.random.normal(0, 0.001, 1000))),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 1000)
        }, index=dates)
        
        print(f"✓ 生成示例数据: {len(df)} 条")
        print(f"  时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # 数据质量检查
        processor = DataProcessor()
        quality = processor.check_data_quality(df)
        print(f"\n✓ 数据质量检查:")
        print(f"  缺失值: {quality['missing_values']}")
        print(f"  重复值: {quality['duplicate_rows']}")
        print(f"  异常值: {quality['outliers']}")
        
        # 添加技术指标
        df_with_indicators = processor.add_technical_indicators(
            df, ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']
        )
        print(f"\n✓ 添加技术指标:")
        print(f"  新增列: {list(df_with_indicators.columns[-10:])}")
        
        return df_with_indicators
        
    except Exception as e:
        print(f"✗ 数据模块演示失败: {e}")
        return None


async def demo_factors_module(df):
    """演示因子模块"""
    print("\n【2. 因子模块演示】")
    print("-" * 50)
    
    try:
        from factors import FactorPool
        
        # 创建因子池
        pool = FactorPool()
        print(f"✓ 因子池初始化完成")
        
        # 计算所有因子
        df_factors = pool.calculate_all_factors(df)
        
        # 统计因子数量
        factor_cols = [c for c in df_factors.columns 
                      if c not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"✓ 计算完成: {len(factor_cols)} 个因子")
        
        # 显示部分因子
        print(f"\n  因子示例:")
        for col in factor_cols[:10]:
            print(f"    - {col}")
        
        return df_factors
        
    except Exception as e:
        print(f"✗ 因子模块演示失败: {e}")
        return None


async def demo_models_module(df):
    """演示模型模块"""
    print("\n【3. 模型模块演示】")
    print("-" * 50)
    
    try:
        from models import ModelConfig, LightGBMModel
        from sklearn.model_selection import train_test_split
        
        # 准备数据
        feature_cols = [c for c in df.columns 
                       if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        # 创建目标变量（未来5期收益率）
        df['target'] = df['close'].pct_change(5).shift(-5)
        df_clean = df[feature_cols + ['target']].dropna()
        
        if len(df_clean) < 100:
            print("✗ 数据量不足")
            return None
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"✓ 数据准备完成:")
        print(f"  训练集: {len(X_train)} 条")
        print(f"  测试集: {len(X_test)} 条")
        print(f"  特征数: {len(feature_cols)}")
        
        # 训练模型
        config = ModelConfig(
            model_name='demo_model',
            model_type='lightgbm',
            prediction_horizon=5
        )
        
        model = LightGBMModel(config)
        model.fit(X_train, y_train)
        
        print(f"✓ 模型训练完成")
        
        # 预测
        predictions = model.predict(X_test)
        
        # 计算性能
        mse = np.mean((predictions - y_test.values) ** 2)
        mae = np.mean(np.abs(predictions - y_test.values))
        
        print(f"\n✓ 模型性能:")
        print(f"  MSE: {mse:.8f}")
        print(f"  MAE: {mae:.8f}")
        
        # 特征重要性
        importance = model.get_feature_importance()
        if importance:
            print(f"\n✓ 特征重要性 (Top 5):")
            for feat, imp in list(importance.items())[:5]:
                print(f"  {feat}: {imp:.4f}")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型模块演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def demo_risk_module():
    """演示风控模块"""
    print("\n【4. 风控模块演示】")
    print("-" * 50)
    
    try:
        from risk import create_risk_manager
        
        # 创建风控管理器
        risk_mgr = create_risk_manager(
            account_balance=10000.0,
            risk_mode="moderate"
        )
        
        print(f"✓ 风控管理器初始化完成")
        print(f"  模式: Moderate")
        print(f"  初始资金: $10,000")
        
        # 测试交易审批
        test_cases = [
            ("BTCUSDT", "long", 50000.0, 47500.0),
            ("ETHUSDT", "short", 3000.0, 3150.0),
            ("SOLUSDT", "long", 100.0, 95.0),
        ]
        
        print(f"\n✓ 交易审批测试:")
        for symbol, direction, price, stop in test_cases:
            approval = risk_mgr.approve_trade(symbol, direction, 0, price, stop)
            status = "✓ 通过" if approval['approved'] else "✗ 拒绝"
            print(f"  {symbol} {direction}: {status}")
            if approval['approved']:
                print(f"    建议仓位: ${approval['position_value']:.2f}")
        
        # 风险指标
        metrics = risk_mgr.calculate_risk_metrics()
        print(f"\n✓ 风险指标:")
        print(f"  VaR (95%): {metrics.get('var_95', 0)*100:.2f}%")
        print(f"  CVaR (95%): {metrics.get('cvar_95', 0)*100:.2f}%")
        
        return risk_mgr
        
    except Exception as e:
        print(f"✗ 风控模块演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def demo_backtest_module(df):
    """演示回测模块"""
    print("\n【5. 回测模块演示】")
    print("-" * 50)
    
    try:
        from backtest import PerformanceAnalyzer, TradeAnalyzer
        
        # 生成模拟收益曲线
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, len(df))
        equity_curve = 100000 * np.exp(np.cumsum(returns))
        
        print(f"✓ 生成模拟收益曲线")
        
        # 绩效分析
        perf_analyzer = PerformanceAnalyzer()
        returns_series = pd.Series(returns, index=df.index)
        metrics = perf_analyzer.analyze(returns_series)
        
        print(f"\n✓ 绩效指标:")
        print(f"  总收益率: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"  年化收益率: {metrics.get('annual_return', 0)*100:.2f}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Calmar比率: {metrics.get('calmar_ratio', 0):.4f}")
        
        # 模拟交易记录
        trades = []
        for i in range(50):
            trade_return = np.random.normal(0.001, 0.02)
            trades.append({
                'symbol': np.random.choice(['BTC', 'ETH', 'SOL']),
                'return': trade_return,
                'pnl': trade_return * 1000
            })
        
        trade_analyzer = TradeAnalyzer()
        trade_metrics = trade_analyzer.analyze(trades)
        
        print(f"\n✓ 交易统计:")
        print(f"  总交易数: {trade_metrics.get('total_trades', 0)}")
        print(f"  胜率: {trade_metrics.get('win_rate', 0)*100:.2f}%")
        print(f"  盈亏比: {trade_metrics.get('profit_factor', 0):.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ 回测模块演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def demo_strategy_module():
    """演示策略模块"""
    print("\n【6. 策略模块演示】")
    print("-" * 50)
    
    try:
        from strategies import MomentumStrategy, MeanReversionStrategy
        
        # 生成示例数据
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        np.random.seed(42)
        
        returns = np.random.normal(0.0001, 0.005, 200)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = {
            'BTC/USDT': pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.0001, 200)),
                'high': prices * (1 + abs(np.random.normal(0, 0.003, 200))),
                'low': prices * (1 - abs(np.random.normal(0, 0.003, 200))),
                'close': prices,
                'volume': np.random.uniform(100, 1000, 200)
            }, index=dates)
        }
        
        # 动量策略
        momentum = MomentumStrategy({
            'short_window': 5,
            'long_window': 15,
            'rsi_period': 14
        })
        momentum.initialize(data)
        
        signals = momentum.generate_signals(data)
        print(f"✓ 动量策略生成 {len(signals)} 个信号")
        
        # 均值回归策略
        mean_rev = MeanReversionStrategy({
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14
        })
        mean_rev.initialize(data)
        
        signals = mean_rev.generate_signals(data)
        print(f"✓ 均值回归策略生成 {len(signals)} 个信号")
        
        return True
        
    except Exception as e:
        print(f"✗ 策略模块演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """主函数"""
    print("\n开始演示各模块功能...\n")
    
    # 1. 数据模块
    df = await demo_data_module()
    
    if df is not None:
        # 2. 因子模块
        df_factors = await demo_factors_module(df)
        
        # 3. 模型模块
        if df_factors is not None:
            model = await demo_models_module(df_factors)
    
    # 4. 风控模块
    risk_mgr = await demo_risk_module()
    
    # 5. 回测模块
    if df is not None:
        metrics = await demo_backtest_module(df)
    
    # 6. 策略模块
    strategy_result = await demo_strategy_module()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n提示:")
    print("  - 运行 'python main.py' 进行完整回测")
    print("  - 编辑 config/config.yaml 修改配置")
    print("  - 查看 README.md 获取更多信息")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
