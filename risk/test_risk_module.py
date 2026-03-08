"""
风控模块测试

测试所有风控功能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk_manager import RiskManager, create_risk_manager, quick_risk_check
from metrics.risk_metrics import RiskMetricsCalculator, quick_metrics
from position.position_sizer import PositionSizer, PositionSizingMethod, quick_position_size
from stop_loss.stop_manager import StopManager, quick_stop_loss, quick_take_profit
from drawdown.drawdown_controller import DrawdownController, calculate_drawdown
from portfolio.portfolio_risk import PortfolioRiskManager, Position, ContractType


def test_risk_metrics():
    """测试风险指标计算"""
    print("\n" + "=" * 50)
    print("测试风险指标计算")
    print("=" * 50)
    
    # 生成模拟价格数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 一年数据
    prices = 100 * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=pd.date_range('2023-01-01', periods=252, freq='D'))
    
    # 创建计算器
    calculator = RiskMetricsCalculator(risk_free_rate=0.02)
    
    # 计算所有指标
    metrics = calculator.calculate_all_metrics(price_series, periods_per_year=365)
    
    print(f"VaR 95%: {metrics.var_95*100:.2f}%")
    print(f"VaR 99%: {metrics.var_99*100:.2f}%")
    print(f"CVaR 95%: {metrics.cvar_95*100:.2f}%")
    print(f"波动率: {metrics.volatility*100:.2f}%")
    print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"索提诺比率: {metrics.sortino_ratio:.2f}")
    print(f"Calmar比率: {metrics.calmar_ratio:.2f}")
    print(f"最大回撤: {metrics.max_drawdown*100:.2f}%")
    
    # 快速指标
    quick = quick_metrics(price_series)
    print(f"\n快速指标: {quick}")
    
    print("✓ 风险指标测试通过")
    return True


def test_position_sizing():
    """测试仓位管理"""
    print("\n" + "=" * 50)
    print("测试仓位管理")
    print("=" * 50)
    
    # 创建仓位管理器
    sizer = PositionSizer(
        account_balance=10000.0,
        max_position_pct=0.5,
        max_leverage=10.0
    )
    
    current_price = 50000.0
    stop_price = 47500.0
    
    # 测试百分比风险法
    result = sizer.percent_risk_sizing(current_price, stop_price, risk_pct=0.02)
    print(f"\n百分比风险法:")
    print(f"  仓位价值: ${result.size:,.2f}")
    print(f"  交易单位: {result.size_in_units:.4f}")
    print(f"  杠杆: {result.leverage:.2f}x")
    print(f"  风险金额: ${result.risk_amount:,.2f}")
    
    # 测试ATR法
    atr = 1500.0
    result = sizer.atr_based_sizing(current_price, atr, risk_pct=0.02)
    print(f"\nATR基础法:")
    print(f"  仓位价值: ${result.size:,.2f}")
    print(f"  交易单位: {result.size_in_units:.4f}")
    
    # 测试凯利公式
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    kelly_pct = sizer.kelly_from_returns(returns, fraction=0.5)
    print(f"\n凯利公式仓位比例: {kelly_pct*100:.2f}%")
    
    # 快速仓位计算
    size = quick_position_size(10000, 50000, 47500, 0.02)
    print(f"快速仓位计算: {size:.4f} BTC")
    
    print("✓ 仓位管理测试通过")
    return True


def test_stop_management():
    """测试止损止盈管理"""
    print("\n" + "=" * 50)
    print("测试止损止盈管理")
    print("=" * 50)
    
    # 创建止损管理器
    manager = StopManager()
    
    # 注册持仓
    entry_price = 50000.0
    position_size = 0.1
    
    pos_stops = manager.register_position(
        "BTC_001", entry_price, position_size, is_long=True
    )
    
    # 设置固定止损
    stop = manager.set_fixed_stop("BTC_001", stop_pct=0.05)
    print(f"固定止损价格: ${stop.price:,.2f}")
    
    # 设置追踪止损
    trail_stop = manager.set_trailing_stop("BTC_001", trail_pct=0.03)
    print(f"追踪止损初始价格: ${trail_stop.price:,.2f}")
    
    # 更新追踪止损
    manager.update_trailing_stop("BTC_001", 52000.0)
    print(f"价格涨到$52,000后追踪止损: ${trail_stop.price:,.2f}")
    
    # 设置止盈
    tp = manager.set_fixed_take_profit("BTC_001", tp_pct=0.10)
    print(f"止盈价格: ${tp.price:,.2f}")
    
    # 检查止损触发
    triggered_stop, triggered_tp = manager.check_all_stops("BTC_001", 47000.0)
    if triggered_stop:
        print(f"止损触发! 触发价格: ${triggered_stop.trigger_price:,.2f}")
    
    # 快速计算
    stop_price = quick_stop_loss(50000, 0.05, True)
    tp_price = quick_take_profit(50000, stop_price, 2.0, True)
    print(f"\n快速计算 - 止损: ${stop_price:,.2f}, 止盈: ${tp_price:,.2f}")
    
    print("✓ 止损止盈管理测试通过")
    return True


def test_drawdown_control():
    """测试回撤控制"""
    print("\n" + "=" * 50)
    print("测试回撤控制")
    print("=" * 50)
    
    # 创建回撤控制器
    controller = DrawdownController(
        max_drawdown_limit=0.20,
        daily_drawdown_limit=0.10,
        intraday_drawdown_limit=0.05
    )
    
    # 模拟权益曲线
    initial_equity = 10000.0
    equity_curve = [initial_equity]
    
    # 正常增长
    for i in range(10):
        equity = equity_curve[-1] * 1.01
        equity_curve.append(equity)
        controller.update_equity(equity)
    
    print(f"权益峰值: ${controller.state.peak_equity:,.2f}")
    
    # 回撤
    for i in range(5):
        equity = equity_curve[-1] * 0.97
        equity_curve.append(equity)
        controller.update_equity(equity)
    
    print(f"当前权益: ${equity_curve[-1]:,.2f}")
    print(f"当前回撤: {controller.state.current_drawdown*100:.2f}%")
    
    # 检查交易许可
    allowed, reason = controller.check_trading_allowed()
    print(f"交易许可: {allowed}, 原因: {reason}")
    
    # 获取回撤报告
    report = controller.get_drawdown_report()
    print(f"\n回撤报告: {report}")
    
    # 计算回撤序列
    equity_series = pd.Series(equity_curve)
    drawdown_series = calculate_drawdown(equity_series)
    print(f"回撤序列: {drawdown_series.tail()}")
    
    print("✓ 回撤控制测试通过")
    return True


def test_portfolio_risk():
    """测试组合风险"""
    print("\n" + "=" * 50)
    print("测试组合风险")
    print("=" * 50)
    
    # 创建组合风险管理器
    manager = PortfolioRiskManager(
        account_balance=10000.0,
        max_total_leverage=3.0,
        max_single_position_pct=0.3
    )
    
    # 添加持仓
    positions = [
        Position("BTC", 0.1, 50000, 52000, "long", ContractType.SPOT),
        Position("ETH", 1.0, 3000, 3100, "long", ContractType.SPOT),
        Position("SOL", 10.0, 100, 105, "long", ContractType.SPOT),
    ]
    
    for pos in positions:
        manager.add_position(pos)
    
    # 设置行业分类
    manager.set_sector_map({
        "BTC": "crypto",
        "ETH": "crypto",
        "SOL": "crypto"
    })
    
    # 计算敞口
    total_exp = manager.get_total_exposure()
    net_exp = manager.get_net_exposure()
    gross_exp = manager.get_gross_exposure()
    
    print(f"总敞口: ${total_exp:,.2f}")
    print(f"净敞口: ${net_exp:,.2f}")
    print(f"总敞口(绝对值): ${gross_exp:,.2f}")
    
    # 集中度风险
    concentration = manager.get_concentration_risk()
    print(f"集中度风险: {concentration:.3f}")
    
    # 行业敞口
    sector_exp = manager.get_sector_exposure()
    print(f"行业敞口: {sector_exp}")
    
    # 检查限制
    limit_checks = manager.check_all_limits()
    print(f"\n限制检查结果:")
    for check, (passed, reason) in limit_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {reason}")
    
    # 保证金报告
    margin_report = manager.get_margin_report()
    print(f"\n保证金报告: {margin_report}")
    
    # 组合报告
    portfolio_report = manager.get_portfolio_report()
    print(f"\n组合报告 - 持仓数量: {portfolio_report['total_positions']}")
    
    print("✓ 组合风险测试通过")
    return True


def test_risk_manager():
    """测试风控管理器主类"""
    print("\n" + "=" * 50)
    print("测试风控管理器主类")
    print("=" * 50)
    
    # 创建风控管理器
    risk_mgr = create_risk_manager(
        account_balance=10000.0,
        risk_mode="moderate"
    )
    
    print(f"账户余额: ${risk_mgr.account_balance:,.2f}")
    print(f"风险模式: {risk_mgr.risk_mode}")
    print(f"最大回撤: {risk_mgr.max_drawdown*100:.1f}%")
    print(f"最大杠杆: {risk_mgr.max_leverage:.1f}x")
    
    # 审批交易
    approval = risk_mgr.approve_trade(
        symbol="BTC",
        side="long",
        entry_price=50000.0,
        stop_loss=47500.0,
        risk_reward=2.0
    )
    
    print(f"\n交易审批:")
    print(f"  批准: {approval.approved}")
    print(f"  仓位大小: {approval.position_size:.4f}")
    print(f"  止损: ${approval.stop_loss:,.2f}" if approval.stop_loss else "  止损: None")
    print(f"  止盈: ${approval.take_profit:,.2f}" if approval.take_profit else "  止盈: None")
    print(f"  风险金额: ${approval.risk_amount:,.2f}")
    
    # 注册持仓
    pos_stops = risk_mgr.register_position(
        position_id="BTC_001",
        symbol="BTC",
        entry_price=50000.0,
        position_size=approval.position_size,
        side="long",
        stop_loss=approval.stop_loss,
        take_profit=approval.take_profit
    )
    
    # 更新价格
    risk_mgr.update_position("BTC_001", 51000.0)
    
    # 生成风险报告
    report = risk_mgr.get_risk_report()
    print(f"\n风险报告生成成功")
    
    # 生成日报
    daily_report = risk_mgr.generate_daily_report()
    print(f"\n日报预览:")
    print(daily_report[:500] + "...")
    
    print("✓ 风控管理器测试通过")
    return True


def test_quick_functions():
    """测试便捷函数"""
    print("\n" + "=" * 50)
    print("测试便捷函数")
    print("=" * 50)
    
    # 快速风险检查
    result = quick_risk_check(10000, 50000, 47500, 0.1)
    print(f"快速风险检查: {result}")
    
    # 快速仓位计算
    size = quick_position_size(10000, 50000, 47500, 0.02)
    print(f"快速仓位计算: {size:.4f}")
    
    # 快速止损止盈
    stop = quick_stop_loss(50000, 0.05, True)
    tp = quick_take_profit(50000, stop, 2.0, True)
    print(f"快速止损: ${stop:,.2f}, 止盈: ${tp:,.2f}")
    
    print("✓ 便捷函数测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("风控模块综合测试")
    print("=" * 60)
    
    tests = [
        ("风险指标", test_risk_metrics),
        ("仓位管理", test_position_sizing),
        ("止损止盈", test_stop_management),
        ("回撤控制", test_drawdown_control),
        ("组合风险", test_portfolio_risk),
        ("风控管理器", test_risk_manager),
        ("便捷函数", test_quick_functions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name}测试失败: {e}")
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
