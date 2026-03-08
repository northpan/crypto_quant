"""
风控模块使用示例

展示如何使用风控模块进行完整的风险管理
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from risk_manager import RiskManager, create_risk_manager
from position.position_sizer import PositionSizingMethod


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)
    
    # 1. 创建风控管理器
    risk_mgr = create_risk_manager(
        account_balance=10000.0,
        risk_mode="moderate"  # 可选: conservative, moderate, aggressive
    )
    
    print(f"创建风控管理器 - 余额: ${risk_mgr.account_balance:,.2f}")
    print(f"风险模式: {risk_mgr.risk_mode}")
    
    # 2. 审批交易
    approval = risk_mgr.approve_trade(
        symbol="BTCUSDT",
        side="long",
        entry_price=50000.0,
        stop_loss=47500.0,  # 5%止损
        risk_reward=2.0  # 风险回报比 1:2
    )
    
    if approval.approved:
        print(f"\n✓ 交易已批准")
        print(f"  建议仓位: {approval.position_size:.4f} BTC")
        print(f"  止损价格: ${approval.stop_loss:,.2f}")
        print(f"  止盈价格: ${approval.take_profit:,.2f}")
        print(f"  风险金额: ${approval.risk_amount:,.2f}")
    else:
        print(f"\n✗ 交易被拒绝: {approval.message}")
        return
    
    # 3. 注册持仓
    position_id = "BTC_001"
    risk_mgr.register_position(
        position_id=position_id,
        symbol="BTCUSDT",
        entry_price=50000.0,
        position_size=approval.position_size,
        side="long",
        stop_loss=approval.stop_loss,
        take_profit=approval.take_profit
    )
    
    print(f"\n✓ 持仓已注册: {position_id}")
    
    # 4. 模拟价格更新
    prices = [51000, 52000, 51500, 53000, 52800]
    for price in prices:
        risk_mgr.update_position(position_id, price)
        print(f"  价格更新: ${price:,} - 检查止损止盈")
    
    # 5. 生成风险报告
    report = risk_mgr.get_risk_report()
    print(f"\n✓ 风险报告已生成")
    
    # 6. 平仓
    risk_mgr.close_position(position_id)
    print(f"✓ 持仓已平仓")


def example_different_risk_modes():
    """不同风险模式对比"""
    print("\n" + "=" * 60)
    print("不同风险模式对比")
    print("=" * 60)
    
    modes = ["conservative", "moderate", "aggressive"]
    
    for mode in modes:
        risk_mgr = create_risk_manager(
            account_balance=10000.0,
            risk_mode=mode
        )
        
        print(f"\n{mode.upper()} 模式:")
        print(f"  最大回撤: {risk_mgr.max_drawdown*100:.0f}%")
        print(f"  日回撤: {risk_mgr.daily_drawdown*100:.0f}%")
        print(f"  最大杠杆: {risk_mgr.max_leverage:.0f}x")
        print(f"  单笔风险: {risk_mgr.risk_per_trade*100:.1f}%")
        
        # 审批相同交易
        approval = risk_mgr.approve_trade(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            stop_loss=47500.0
        )
        
        if approval.approved:
            print(f"  批准仓位: {approval.position_size:.4f} BTC")
        else:
            print(f"  交易被拒绝: {approval.message}")


def example_position_sizing_methods():
    """不同仓位计算方法对比"""
    print("\n" + "=" * 60)
    print("不同仓位计算方法对比")
    print("=" * 60)
    
    risk_mgr = create_risk_manager(account_balance=10000.0)
    
    methods = [
        (PositionSizingMethod.PERCENT_RISK, "百分比风险法"),
        (PositionSizingMethod.ATR_BASED, "ATR基础法"),
        (PositionSizingMethod.FIXED_FRACTION, "固定比例法"),
    ]
    
    # 生成模拟收益率数据
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    for method, name in methods:
        print(f"\n{name}:")
        
        if method == PositionSizingMethod.ATR_BASED:
            # ATR法需要ATR值
            approval = risk_mgr.approve_trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                stop_loss=47500.0
            )
        else:
            approval = risk_mgr.approve_trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                stop_loss=47500.0,
                method=method
            )
        
        if approval.approved:
            print(f"  仓位: {approval.position_size:.4f} BTC")
            print(f"  杠杆: {approval.max_leverage:.2f}x")
            print(f"  风险: ${approval.risk_amount:.2f}")


def example_stop_loss_strategies():
    """止损策略示例"""
    print("\n" + "=" * 60)
    print("止损策略示例")
    print("=" * 60)
    
    risk_mgr = create_risk_manager(account_balance=10000.0)
    
    # 注册持仓
    position_id = "BTC_002"
    entry_price = 50000.0
    position_size = 0.1
    
    risk_mgr.register_position(
        position_id=position_id,
        symbol="BTCUSDT",
        entry_price=entry_price,
        position_size=position_size,
        side="long"
    )
    
    # 1. 固定止损
    print("\n1. 固定止损 (5%)")
    stop_loss = entry_price * 0.95
    print(f"   止损价格: ${stop_loss:,.2f}")
    
    # 2. 追踪止损
    print("\n2. 追踪止损 (3%)")
    risk_mgr.stop_manager.set_trailing_stop(
        position_id,
        trail_pct=0.03,
        activation_pct=0.02  # 2%盈利后激活
    )
    
    # 模拟价格上涨
    prices = [51000, 52000, 53000, 52500, 51500]
    for price in prices:
        risk_mgr.update_position(position_id, price)
        
        # 获取追踪止损状态
        summary = risk_mgr.stop_manager.get_stop_summary(position_id)
        if summary.get('stop_losses'):
            for stop in summary['stop_losses']:
                if stop['type'] == 'trailing':
                    print(f"   价格 ${price:,} -> 追踪止损: ${stop['price']:,.2f}")
    
    risk_mgr.close_position(position_id)


def example_drawdown_control():
    """回撤控制示例"""
    print("\n" + "=" * 60)
    print("回撤控制示例")
    print("=" * 60)
    
    risk_mgr = create_risk_manager(account_balance=10000.0)
    
    # 模拟权益曲线
    equity = 10000.0
    peak = equity
    
    print("\n模拟交易过程:")
    
    # 盈利阶段
    for i in range(5):
        equity *= 1.02
        risk_mgr.update_equity(equity)
        peak = max(peak, equity)
        print(f"  盈利后权益: ${equity:,.2f} (峰值: ${peak:,.2f})")
    
    # 回撤阶段
    for i in range(8):
        equity *= 0.97
        risk_mgr.update_equity(equity)
        
        drawdown = (peak - equity) / peak
        scale = risk_mgr.drawdown_controller.get_position_scale()
        allowed, reason = risk_mgr.drawdown_controller.check_trading_allowed()
        
        print(f"  回撤后权益: ${equity:,.2f} (回撤: {drawdown*100:.1f}%, 仓位缩放: {scale*100:.0f}%, 交易: {'允许' if allowed else '禁止'})")
    
    # 生成回撤报告
    report = risk_mgr.drawdown_controller.get_drawdown_report()
    print(f"\n回撤报告:")
    print(f"  当前回撤: {report['current_drawdown']*100:.2f}%")
    print(f"  最大回撤: {report['max_drawdown']*100:.2f}%")
    print(f"  仓位缩放: {report['position_scale']*100:.0f}%")


def example_portfolio_risk():
    """组合风险管理示例"""
    print("\n" + "=" * 60)
    print("组合风险管理示例")
    print("=" * 60)
    
    risk_mgr = create_risk_manager(account_balance=50000.0)
    
    # 添加多个持仓
    positions = [
        ("BTC_001", "BTCUSDT", 50000, 0.2, "long"),
        ("ETH_001", "ETHUSDT", 3000, 2.0, "long"),
        ("SOL_001", "SOLUSDT", 100, 20.0, "long"),
    ]
    
    print("\n添加持仓:")
    for pos_id, symbol, price, size, side in positions:
        # 审批交易
        approval = risk_mgr.approve_trade(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=price * 0.95
        )
        
        if approval.approved:
            risk_mgr.register_position(
                position_id=pos_id,
                symbol=symbol,
                entry_price=price,
                position_size=approval.position_size,
                side=side,
                stop_loss=approval.stop_loss,
                take_profit=approval.take_profit
            )
            
            print(f"  {symbol}: {approval.position_size:.4f} @ ${price:,}")
    
    # 获取组合报告
    report = risk_mgr.get_risk_report()
    portfolio = report['portfolio']
    
    print(f"\n组合风险报告:")
    print(f"  持仓数量: {portfolio['total_positions']}")
    print(f"  总敞口: ${portfolio['total_exposure']:,.2f}")
    print(f"  净敞口: ${portfolio['net_exposure']:,.2f}")
    print(f"  集中度风险: {portfolio['concentration_risk']:.3f}")
    
    # 行业敞口
    if portfolio['sector_exposure']:
        print(f"\n  行业敞口:")
        for sector, exposure in portfolio['sector_exposure'].items():
            pct = exposure / risk_mgr.account_balance * 100
            print(f"    {sector}: ${exposure:,.2f} ({pct:.1f}%)")
    
    # 限制检查
    print(f"\n  限制检查:")
    for check, (passed, reason) in portfolio['limit_checks'].items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}: {reason}")


def example_risk_report():
    """风险报告示例"""
    print("\n" + "=" * 60)
    print("风险报告示例")
    print("=" * 60)
    
    risk_mgr = create_risk_manager(account_balance=10000.0)
    
    # 添加一些持仓
    for i in range(3):
        approval = risk_mgr.approve_trade(
            symbol=f"COIN{i}",
            side="long",
            entry_price=100.0 + i * 10,
            stop_loss=95.0 + i * 10
        )
        
        if approval.approved:
            risk_mgr.register_position(
                position_id=f"POS_{i}",
                symbol=f"COIN{i}",
                entry_price=100.0 + i * 10,
                position_size=approval.position_size,
                side="long",
                stop_loss=approval.stop_loss,
                take_profit=approval.take_profit
            )
    
    # 生成日报
    daily_report = risk_mgr.generate_daily_report()
    print(daily_report)


def example_integration():
    """完整集成示例"""
    print("\n" + "=" * 60)
    print("完整集成示例 - 模拟交易流程")
    print("=" * 60)
    
    # 创建风控管理器
    risk_mgr = create_risk_manager(
        account_balance=50000.0,
        risk_mode="moderate"
    )
    
    print("\n=== 步骤1: 初始化 ===")
    print(f"账户余额: ${risk_mgr.account_balance:,.2f}")
    print(f"风险模式: {risk_mgr.risk_mode}")
    
    print("\n=== 步骤2: 交易审批 ===")
    
    # 模拟交易信号
    signals = [
        {"symbol": "BTCUSDT", "side": "long", "price": 50000, "confidence": 0.8},
        {"symbol": "ETHUSDT", "side": "long", "price": 3000, "confidence": 0.7},
        {"symbol": "SOLUSDT", "side": "short", "price": 100, "confidence": 0.6},
    ]
    
    approved_trades = []
    
    for signal in signals:
        print(f"\n  信号: {signal['side'].upper()} {signal['symbol']} @ ${signal['price']:,}")
        
        # 计算止损 (2%)
        stop_pct = 0.02
        if signal['side'] == "long":
            stop_loss = signal['price'] * (1 - stop_pct)
        else:
            stop_loss = signal['price'] * (1 + stop_pct)
        
        # 审批交易
        approval = risk_mgr.approve_trade(
            symbol=signal['symbol'],
            side=signal['side'],
            entry_price=signal['price'],
            stop_loss=stop_loss,
            risk_reward=2.5
        )
        
        if approval.approved:
            print(f"  ✓ 批准 - 仓位: {approval.position_size:.4f}, 风险: ${approval.risk_amount:.2f}")
            approved_trades.append({
                **signal,
                'position_size': approval.position_size,
                'stop_loss': approval.stop_loss,
                'take_profit': approval.take_profit
            })
        else:
            print(f"  ✗ 拒绝 - {approval.message}")
    
    print(f"\n=== 步骤3: 注册持仓 ({len(approved_trades)}个) ===")
    
    for i, trade in enumerate(approved_trades):
        position_id = f"POS_{i}"
        
        risk_mgr.register_position(
            position_id=position_id,
            symbol=trade['symbol'],
            entry_price=trade['price'],
            position_size=trade['position_size'],
            side=trade['side'],
            stop_loss=trade['stop_loss'],
            take_profit=trade['take_profit']
        )
        
        print(f"  {position_id}: {trade['symbol']} {trade['position_size']:.4f}")
    
    print("\n=== 步骤4: 模拟价格变动 ===")
    
    # 模拟价格变动
    np.random.seed(42)
    for day in range(5):
        print(f"\n  Day {day + 1}:")
        
        for i, trade in enumerate(approved_trades):
            position_id = f"POS_{i}"
            
            # 随机价格变动
            change = np.random.normal(0, 0.02)
            new_price = trade['price'] * (1 + change)
            trade['price'] = new_price
            
            # 更新持仓
            risk_mgr.update_position(position_id, new_price)
            
            # 检查止损止盈
            summary = risk_mgr.stop_manager.get_stop_summary(position_id)
            
            print(f"    {trade['symbol']}: ${new_price:,.2f} ({change*100:+.2f}%)")
    
    print("\n=== 步骤5: 生成风险报告 ===")
    
    report = risk_mgr.get_risk_report()
    
    print(f"  账户余额: ${report['account_balance']:,.2f}")
    print(f"  持仓数量: {report['portfolio']['total_positions']}")
    print(f"  总敞口: ${report['portfolio']['total_exposure']:,.2f}")
    print(f"  当前回撤: {report['drawdown']['current_drawdown']*100:.2f}%")
    
    print("\n=== 步骤6: 生成日报 ===")
    print(risk_mgr.generate_daily_report())


def run_all_examples():
    """运行所有示例"""
    examples = [
        ("基础使用", example_basic_usage),
        ("风险模式对比", example_different_risk_modes),
        ("仓位计算方法", example_position_sizing_methods),
        ("止损策略", example_stop_loss_strategies),
        ("回撤控制", example_drawdown_control),
        ("组合风险", example_portfolio_risk),
        ("风险报告", example_risk_report),
        ("完整集成", example_integration),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ {name}示例出错: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
