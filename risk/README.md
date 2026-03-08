# 量化交易风控模块

完整的多层风控体系，支持现货和合约交易的风险管理。

## 模块结构

```
risk/
├── __init__.py              # 模块入口
├── risk_manager.py          # 风控管理器主类
├── metrics/
│   └── risk_metrics.py      # 风险指标计算
├── position/
│   └── position_sizer.py    # 仓位管理
├── stop_loss/
│   └── stop_manager.py      # 止损止盈管理
├── drawdown/
│   └── drawdown_controller.py # 回撤控制
├── portfolio/
│   └── portfolio_risk.py    # 组合风险
├── test_risk_module.py      # 测试代码
└── examples.py              # 使用示例
```

## 功能特性

### 1. 仓位管理 (position/)

- **凯利公式仓位计算**: 基于胜率和盈亏比的最优仓位
- **固定比例仓位**: 固定账户百分比仓位
- **波动率目标仓位**: 根据目标波动率动态调整
- **ATR基础仓位**: 基于平均真实波幅的仓位计算
- **百分比风险仓位**: 基于单笔风险金额的仓位
- **Optimal f**: Ralph Vince的最优f仓位

### 2. 止损止盈 (stop_loss/)

- **固定止损/止盈**: 基于固定价格或百分比
- **追踪止损**: 随价格移动而调整的止损
- **ATR自适应止损**: 基于波动率的动态止损
- **时间止损**: 持仓时间限制
- **保本止损**: 盈利后移动到成本价
- **分批止盈**: 多目标分批出场

### 3. 回撤控制 (drawdown/)

- **最大回撤限制**: 账户级别回撤控制
- **日回撤监控**: 日内回撤限制
- **动态降仓机制**: 根据回撤自动调整仓位
- **多层回撤控制**: 不同回撤水平触发不同行动
- **交易暂停**: 回撤超限自动暂停交易

### 4. 组合风险 (portfolio/)

- **多币种相关性管理**: 相关性矩阵分析
- **行业/板块分散**: 行业敞口控制
- **杠杆控制**: 总杠杆和单币种杠杆限制
- **保证金监控**: 合约保证金水平监控
- **集中度风险**: 赫芬达尔指数计算

### 5. 风险指标 (metrics/)

- **VaR计算**: 历史法、参数法、蒙特卡洛法
- **CVaR (Expected Shortfall)**: 条件风险价值
- **夏普比率**: 风险调整收益
- **索提诺比率**: 下行风险调整收益
- **Calmar比率**: 最大回撤调整收益
- **Beta/Alpha**: 系统性风险指标

## 快速开始

### 基础使用

```python
from risk import create_risk_manager

# 创建风控管理器
risk_mgr = create_risk_manager(
    account_balance=10000.0,
    risk_mode="moderate"  # conservative, moderate, aggressive
)

# 审批交易
approval = risk_mgr.approve_trade(
    symbol="BTCUSDT",
    side="long",
    entry_price=50000.0,
    stop_loss=47500.0,
    risk_reward=2.0
)

if approval.approved:
    print(f"仓位: {approval.position_size}")
    print(f"止损: {approval.stop_loss}")
    print(f"止盈: {approval.take_profit}")
```

### 注册持仓并监控

```python
# 注册持仓
risk_mgr.register_position(
    position_id="BTC_001",
    symbol="BTCUSDT",
    entry_price=50000.0,
    position_size=0.1,
    side="long",
    stop_loss=47500.0,
    take_profit=55000.0
)

# 更新价格（自动检查止损止盈）
risk_mgr.update_position("BTC_001", current_price)

# 平仓
risk_mgr.close_position("BTC_001")
```

### 获取风险报告

```python
# 获取风险报告
report = risk_mgr.get_risk_report()

# 生成日报
daily_report = risk_mgr.generate_daily_report()
print(daily_report)
```

## 风险模式

### Conservative (保守)
- 最大回撤: 10%
- 日回撤: 5%
- 最大杠杆: 2x
- 单笔风险: 1%

### Moderate (稳健) - 默认
- 最大回撤: 20%
- 日回撤: 10%
- 最大杠杆: 5x
- 单笔风险: 2%

### Aggressive (激进)
- 最大回撤: 30%
- 日回撤: 15%
- 最大杠杆: 10x
- 单笔风险: 3%

## 仓位计算方法

```python
from position.position_sizer import PositionSizingMethod

# 百分比风险法
approval = risk_mgr.approve_trade(
    symbol="BTCUSDT",
    side="long",
    entry_price=50000.0,
    stop_loss=47500.0,
    method=PositionSizingMethod.PERCENT_RISK
)

# ATR基础法
approval = risk_mgr.approve_trade(
    symbol="BTCUSDT",
    side="long",
    entry_price=50000.0,
    method=PositionSizingMethod.ATR_BASED
)
```

## 止损策略

```python
# 固定止损
risk_mgr.stop_manager.set_fixed_stop("BTC_001", stop_pct=0.05)

# 追踪止损
risk_mgr.stop_manager.set_trailing_stop(
    "BTC_001",
    trail_pct=0.03,
    activation_pct=0.02
)

# ATR止损
risk_mgr.stop_manager.set_atr_stop("BTC_001", atr=1500, multiplier=2.0)
```

## 回撤控制

```python
# 更新权益（自动回撤监控）
risk_mgr.update_equity(current_equity)

# 检查交易许可
allowed, reason = risk_mgr.drawdown_controller.check_trading_allowed()

# 获取回撤报告
dd_report = risk_mgr.drawdown_controller.get_drawdown_report()
```

## 组合风险管理

```python
from portfolio.portfolio_risk import Position, ContractType

# 添加持仓
position = Position(
    symbol="BTCUSDT",
    size=0.1,
    entry_price=50000.0,
    current_price=52000.0,
    side="long",
    contract_type=ContractType.PERPETUAL,
    leverage=5.0
)

risk_mgr.portfolio_manager.add_position(position)

# 检查限制
limit_checks = risk_mgr.portfolio_manager.check_all_limits()

# 获取组合报告
portfolio_report = risk_mgr.portfolio_manager.get_portfolio_report()
```

## 测试

```bash
cd /mnt/okcomputer/output/crypto_quant/risk
python test_risk_module.py
```

## 示例

```bash
cd /mnt/okcomputer/output/crypto_quant/risk
python examples.py
```

## API文档

### RiskManager

主风控管理器类，整合所有风控功能。

#### 构造函数
```python
RiskManager(
    account_balance: float = 10000.0,
    risk_mode: str = "moderate",
    contract_type: str = "spot"
)
```

#### 主要方法
- `approve_trade()`: 审批交易
- `register_position()`: 注册持仓
- `update_position()`: 更新持仓价格
- `close_position()`: 平仓
- `get_risk_report()`: 获取风险报告
- `generate_daily_report()`: 生成日报

### PositionSizer

仓位管理器，提供多种仓位计算方法。

#### 主要方法
- `kelly_criterion()`: 凯利公式
- `percent_risk_sizing()`: 百分比风险法
- `atr_based_sizing()`: ATR基础法
- `volatility_target_sizing()`: 波动率目标法

### StopManager

止损止盈管理器。

#### 主要方法
- `set_fixed_stop()`: 固定止损
- `set_trailing_stop()`: 追踪止损
- `set_atr_stop()`: ATR止损
- `set_time_stop()`: 时间止损
- `check_all_stops()`: 检查所有止损止盈

### DrawdownController

回撤控制器。

#### 主要方法
- `update_equity()`: 更新权益
- `check_trading_allowed()`: 检查交易许可
- `get_drawdown_report()`: 获取回撤报告

### PortfolioRiskManager

组合风险管理器。

#### 主要方法
- `add_position()`: 添加持仓
- `check_all_limits()`: 检查所有限制
- `get_portfolio_metrics()`: 获取组合指标
- `get_portfolio_report()`: 获取组合报告

## 许可证

MIT License
