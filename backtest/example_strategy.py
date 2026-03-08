"""
回测系统使用示例
演示如何构建一个完整的回测流程
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.backtest_engine import (
    BacktestEngine, OrderType, OrderSide, TradeType, CostModel
)
from costs.transaction_costs import (
    TransactionCostModel, ExchangeCostProfile, SlippageModel, ImpactModel
)
from analytics.performance_analyzer import PerformanceAnalyzer
from analytics.trade_analyzer import TradeAnalyzer
from analytics.report_generator import ReportGenerator
from visualization.visualizer import BacktestVisualizer
from overfitting.overfitting_tests import OverfittingDetector


def generate_sample_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2023-01-01",
    periods: int = 10000,
    freq: str = "1H"
) -> pd.DataFrame:
    """
    生成示例K线数据
    
    Args:
        symbol: 交易对
        start_date: 开始日期
        periods: 数据点数
        freq: 时间频率
    
    Returns:
        DataFrame包含OHLCV数据
    """
    np.random.seed(42)
    
    # 生成时间序列
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # 生成价格数据 (带趋势和均值回归)
    price = 50000
    prices = []
    trend = 0.0001
    
    for i in range(periods):
        # 添加趋势和噪声
        change = trend + np.random.normal(0, 0.005)
        price *= (1 + change)
        prices.append(price)
    
    # 生成OHLCV
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, periods)
    })
    
    return data


def moving_average_crossover_strategy(
    engine,
    portfolio,
    market_data,
    timestamp,
    short_window: int = 10,
    long_window: int = 30,
    position_size: float = 0.2
):
    """
    双均线交叉策略
    
    Args:
        engine: 回测引擎
        portfolio: 投资组合
        market_data: 当前市场数据
        timestamp: 当前时间
        short_window: 短期均线窗口
        long_window: 长期均线窗口
        position_size: 仓位大小
    """
    # 需要维护历史价格来计算均线
    if not hasattr(engine, '_price_history'):
        engine._price_history = {}
    
    for symbol, data in market_data.items():
        # 记录价格历史
        if symbol not in engine._price_history:
            engine._price_history[symbol] = []
        engine._price_history[symbol].append(data.close)
        
        # 检查是否有足够的数据
        if len(engine._price_history[symbol]) < long_window:
            continue
        
        # 计算均线
        prices = engine._price_history[symbol][-long_window:]
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices)
        
        # 获取当前持仓
        position = portfolio.get_position(symbol)
        
        # 交易逻辑
        if short_ma > long_ma * 1.001:  # 金叉，买入
            if position is None or position.side.value == 'short':
                # 如果有空仓，先平仓
                if position:
                    order = engine.execution_engine.create_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY if position.side.value == 'short' else OrderSide.SELL,
                        quantity=position.quantity,
                        trade_type=engine.trade_type
                    )
                    engine.submit_order(order)
                
                # 开多仓
                cash_available = portfolio.cash * position_size
                quantity = cash_available / data.close
                
                if quantity > 0:
                    order = engine.execution_engine.create_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        trade_type=engine.trade_type
                    )
                    engine.submit_order(order)
        
        elif short_ma < long_ma * 0.999:  # 死叉，卖出
            if position and position.side.value == 'long':
                # 平多仓
                order = engine.execution_engine.create_order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    trade_type=engine.trade_type
                )
                engine.submit_order(order)


def run_backtest_example():
    """运行回测示例"""
    print("=" * 80)
    print("CryptoQuant 回测系统示例")
    print("=" * 80)
    
    # 1. 生成示例数据
    print("\n[1/6] 生成示例数据...")
    data = generate_sample_data(
        symbol="BTCUSDT",
        start_date="2023-01-01",
        periods=5000,
        freq="1H"
    )
    print(f"  数据点数: {len(data)}")
    print(f"  时间范围: {data['timestamp'].min()} ~ {data['timestamp'].max()}")
    
    # 2. 创建回测引擎
    print("\n[2/6] 创建回测引擎...")
    
    # 创建交易成本模型
    cost_model = CostModel(
        maker_fee=0.0002,  # 币安合约挂单费率
        taker_fee=0.0005,  # 币安合约吃单费率
        slippage_model="fixed",
        slippage_value=0.0005
    )
    
    engine = BacktestEngine(
        initial_capital=100000.0,
        trade_type=TradeType.FUTURES,
        cost_model=cost_model,
        fill_model="vwap"
    )
    
    # 加载数据
    engine.load_data("BTCUSDT", data)
    
    # 3. 设置策略
    print("\n[3/6] 设置策略...")
    
    def strategy(engine, portfolio, market_data, timestamp):
        moving_average_crossover_strategy(
            engine, portfolio, market_data, timestamp,
            short_window=10,
            long_window=30,
            position_size=0.2
        )
    
    engine.set_strategy(strategy)
    
    # 4. 运行回测
    print("\n[4/6] 运行回测...")
    results = engine.run()
    
    print(f"\n  回测完成!")
    print(f"  初始资金: ${engine.initial_capital:,.2f}")
    print(f"  最终权益: ${engine.portfolio.total_value:,.2f}")
    print(f"  总收益率: {engine.portfolio.total_return*100:.2f}%")
    print(f"  交易次数: {len(engine.portfolio.trades)}")
    
    # 5. 绩效分析
    print("\n[5/6] 绩效分析...")
    
    # 计算收益率
    equity_curve = pd.Series(
        [e[1] for e in engine.portfolio.equity_curve],
        index=[e[0] for e in engine.portfolio.equity_curve]
    )
    returns = equity_curve.pct_change().dropna()
    
    # 绩效分析
    perf_analyzer = PerformanceAnalyzer(
        risk_free_rate=0.02,
        trading_days_per_year=365
    )
    
    # 生成基准收益率 (买入持有)
    benchmark_returns = data.set_index('timestamp')['close'].pct_change().dropna()
    benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0)
    
    perf_metrics = perf_analyzer.analyze(
        returns=returns,
        benchmark_returns=benchmark_returns,
        equity_curve=equity_curve
    )
    
    print(perf_analyzer.get_summary(perf_metrics))
    
    # 交易分析
    trade_analyzer = TradeAnalyzer(risk_per_trade=0.02)
    
    # 转换交易记录
    for trade in engine.portfolio.trades:
        from analytics.trade_analyzer import Trade as TradeRecord
        trade_record = TradeRecord(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            entry_time=trade.timestamp,
            exit_time=trade.timestamp,
            entry_price=trade.price,
            exit_price=trade.price,
            quantity=trade.quantity,
            side='long',
            pnl=trade.pnl if trade.pnl else 0,
            return_pct=(trade.pnl / (trade.price * trade.quantity)) if trade.pnl else 0,
            fees=trade.fee
        )
        trade_analyzer.add_trade(trade_record)
    
    trade_metrics = trade_analyzer.analyze()
    print(trade_analyzer.get_summary(trade_metrics))
    
    # 6. 生成报告和可视化
    print("\n[6/6] 生成报告和可视化...")
    
    # 创建输出目录
    output_dir = "/mnt/okcomputer/output/crypto_quant/backtest/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化
    visualizer = BacktestVisualizer(figsize=(14, 10), dpi=100)
    
    # 生成图表
    print("  生成权益曲线...")
    fig = visualizer.plot_equity_curve(
        equity_curve,
        benchmark=equity_curve.iloc[0] * (1 + benchmark_returns).cumprod(),
        title="双均线交叉策略 - 权益曲线",
        save_path=f"{output_dir}/equity_curve.png"
    )
    
    print("  生成回撤分析...")
    fig = visualizer.plot_drawdown(
        equity_curve,
        title="回撤分析",
        save_path=f"{output_dir}/drawdown.png"
    )
    
    print("  生成月度收益热力图...")
    fig = visualizer.plot_monthly_returns_heatmap(
        returns,
        title="月度收益热力图",
        save_path=f"{output_dir}/monthly_returns.png"
    )
    
    print("  生成综合报告...")
    trades_df = trade_analyzer.get_trade_list()
    fig = visualizer.plot_comprehensive_report(
        equity_curve=equity_curve,
        returns=returns,
        trades_df=trades_df if not trades_df.empty else None,
        benchmark=equity_curve.iloc[0] * (1 + benchmark_returns).cumprod(),
        title="回测综合报告",
        save_path=f"{output_dir}/comprehensive_report.png"
    )
    
    # 生成报告
    print("  生成文本报告...")
    
    # 准备报告数据
    performance_dict = {
        'initial_capital': engine.initial_capital,
        'final_value': engine.portfolio.total_value,
        'total_pnl': engine.portfolio.total_pnl,
        'total_return': engine.portfolio.total_return,
        'annualized_return': perf_metrics.annualized_return,
        'annualized_volatility': perf_metrics.annualized_volatility,
        'max_drawdown': perf_metrics.max_drawdown,
        'sharpe_ratio': perf_metrics.sharpe_ratio,
        'sortino_ratio': perf_metrics.sortino_ratio,
        'calmar_ratio': perf_metrics.calmar_ratio,
        'alpha': perf_metrics.alpha,
        'beta': perf_metrics.beta
    }
    
    trade_dict = {
        'total_trades': trade_metrics.total_trades,
        'winning_trades': trade_metrics.winning_trades,
        'losing_trades': trade_metrics.losing_trades,
        'win_rate': trade_metrics.win_rate,
        'profit_factor': trade_metrics.profit_factor,
        'avg_profit': trade_metrics.avg_profit,
        'avg_loss': trade_metrics.avg_loss,
        'max_profit': trade_metrics.max_profit,
        'max_loss': trade_metrics.max_loss,
        'expectancy': trade_metrics.expectancy,
        'sqn': trade_metrics.sqn
    }
    
    strategy_info = {
        'name': 'Dual MA Crossover',
        'description': '双均线交叉策略 (10日/30日)',
        'symbols': 'BTCUSDT',
        'timeframe': '1H',
        'start_date': str(data['timestamp'].min()),
        'end_date': str(data['timestamp'].max())
    }
    
    report_generator = ReportGenerator(output_dir=output_dir)
    saved_files = report_generator.save_all_reports(
        performance_metrics=performance_dict,
        trade_metrics=trade_dict,
        strategy_info=strategy_info,
        chart_paths=[
            f"{output_dir}/equity_curve.png",
            f"{output_dir}/drawdown.png",
            f"{output_dir}/monthly_returns.png",
            f"{output_dir}/comprehensive_report.png"
        ],
        prefix="backtest_report"
    )
    
    print("\n" + "=" * 80)
    print("回测完成!")
    print("=" * 80)
    print("\n生成的文件:")
    for format_type, path in saved_files.items():
        print(f"  {format_type}: {path}")
    print(f"\n图表保存在: {output_dir}")
    
    return {
        'engine': engine,
        'results': results,
        'perf_metrics': perf_metrics,
        'trade_metrics': trade_metrics,
        'saved_files': saved_files
    }


def run_overfitting_test_example():
    """运行过拟合检测示例"""
    print("\n" + "=" * 80)
    print("过拟合检测示例")
    print("=" * 80)
    
    # 生成收益率数据
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    
    # 创建检测器
    detector = OverfittingDetector()
    
    # 定义策略函数
    def dummy_strategy(returns, window=20):
        return {'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0}
    
    # 参数范围
    param_ranges = {
        'window': [10, 20, 30, 50]
    }
    
    # 运行测试
    print("\n运行过拟合检测...")
    results = detector.run_all_tests(
        returns=returns,
        strategy_func=dummy_strategy,
        param_ranges=param_ranges
    )
    
    # 打印结果
    print(detector.get_summary())
    
    return results


if __name__ == "__main__":
    # 运行回测示例
    backtest_results = run_backtest_example()
    
    # 运行过拟合检测示例
    overfitting_results = run_overfitting_test_example()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成!")
    print("=" * 80)
