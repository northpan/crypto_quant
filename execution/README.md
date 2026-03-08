# 数字货币交易执行模块

## 概述

本模块提供完整的数字货币交易执行功能，支持模拟和实盘两种模式，包含订单管理、持仓跟踪、执行策略、滑点控制和交易记录等功能。

## 模块结构

```
execution/
├── __init__.py              # 模块入口
├── example_usage.py         # 使用示例
├── exchange/                # 交易所接口
│   ├── __init__.py
│   └── exchange_client.py   # 交易所客户端
├── order/                   # 订单管理
│   ├── __init__.py
│   └── order_manager.py     # 订单管理器
├── position/                # 持仓管理
│   ├── __init__.py
│   └── position_tracker.py  # 持仓跟踪器
├── strategy/                # 执行策略
│   ├── __init__.py
│   └── execution_engine.py  # 执行引擎
├── slippage/                # 滑点控制
│   ├── __init__.py
│   └── slippage_model.py    # 滑点模型
└── recorder/                # 交易记录
    ├── __init__.py
    └── trade_recorder.py    # 交易记录器
```

## 核心功能

### 1. 交易所接口 (exchange/)

基于CCXT的统一接口，支持：
- Binance (现货/合约)
- OKX
- 其他CCXT支持的交易所
- 模拟交易所 (用于回测)

**主要类：**
- `CCXTExchange` - 实盘交易所接口
- `SimulatedExchange` - 模拟交易所
- `ExchangeManager` - 多交易所管理

**使用示例：**
```python
from execution import create_binance_client, create_simulated_exchange

# 模拟交易所
sim = create_simulated_exchange({'USDT': Decimal('10000')})
await sim.connect()

# 实盘交易所
binance = create_binance_client(
    api_key='your_key',
    api_secret='your_secret',
    sandbox=True
)
await binance.connect()
```

### 2. 订单管理 (order/)

支持多种订单类型：
- 限价单 (Limit)
- 市价单 (Market)
- 止损单 (Stop Loss)
- 止盈单 (Take Profit)
- 条件单 (Conditional)
- 批量订单

**主要类：**
- `OrderManager` - 订单管理器
- `OrderRequest` - 订单请求
- `ConditionalOrder` - 条件单

**使用示例：**
```python
from execution import OrderManager, OrderSide, TimeInForce

order_manager = OrderManager(exchange)
await order_manager.start()

# 下限价单
order = await order_manager.place_limit_order(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    amount=Decimal('0.1'),
    price=Decimal('50000'),
    time_in_force=TimeInForce.GTC
)

# 下条件单
cond_order = await order_manager.place_conditional_order(
    trigger_condition=TriggerCondition.PRICE_BELOW,
    trigger_price=Decimal('48000'),
    order_request=OrderRequest(...)
)
```

### 3. 持仓管理 (position/)

- 现货持仓跟踪
- 合约持仓和保证金管理
- 多币种持仓管理
- 自动对冲 (Delta/Beta)

**主要类：**
- `PositionTracker` - 持仓跟踪器
- `Portfolio` - 投资组合
- `PositionSizer` - 仓位管理器

**使用示例：**
```python
from execution import PositionTracker, HedgeMode

position_tracker = PositionTracker(exchange, order_manager)
await position_tracker.start()

# 获取投资组合
portfolio = position_tracker.get_portfolio()
print(f"总资产: {portfolio.total_value}")

# 启用对冲
position_tracker.enable_hedge(
    mode=HedgeMode.DELTA,
    target_ratio=Decimal('0.9')
)
```

### 4. 执行策略 (strategy/)

- **TWAP** - 时间加权平均价格
- **VWAP** - 成交量加权平均价格
- **冰山订单** - 隐藏大单意图
- **智能路由** - 自动选择最优策略

**主要类：**
- `ExecutionEngine` - 执行引擎
- `TWAPStrategy` - TWAP策略
- `VWAPStrategy` - VWAP策略
- `IcebergStrategy` - 冰山订单
- `SmartRouter` - 智能路由

**使用示例：**
```python
from execution import ExecutionEngine, ExecutionConfig, ExecutionStrategy

engine = ExecutionEngine(order_manager)

# TWAP执行
config = ExecutionConfig(
    strategy=ExecutionStrategy.TWAP,
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    total_amount=Decimal('1.0'),
    time_limit=300.0
)

report = await engine.execute(config)
print(f"成交均价: {report.avg_price}")
print(f"滑点: {report.slippage}")
```

### 5. 滑点控制 (slippage/)

- 滑点估计 (固定/线性/平方根/对数/自适应)
- 冲击成本模型 (临时/永久)
- 大单拆分

**主要类：**
- `SlippageModel` - 滑点模型
- `MarketImpactModel` - 市场冲击模型
- `OrderSplitter` - 订单拆分器

**使用示例：**
```python
from execution import SlippageModel, OrderSplitter

# 滑点估计
slippage_model = SlippageModel()
estimate = slippage_model.estimate_slippage(
    symbol='BTC/USDT',
    order_size=Decimal('10'),
    side=OrderSide.BUY,
    reference_price=Decimal('50000')
)
print(f"估计滑点: {estimate.estimated_slippage}")

# 订单拆分
splitter = OrderSplitter(slippage_model)
result = splitter.split_order(
    symbol='BTC/USDT',
    total_amount=Decimal('100'),
    side=OrderSide.BUY,
    max_slice_size=Decimal('10')
)
print(f"切片: {result.slices}")
```

### 6. 交易记录 (recorder/)

- 订单历史
- 成交记录
- 持仓历史
- 性能分析
- 日度/月度报告

**主要类：**
- `TradeRecorder` - 交易记录器
- `TradeRecord` - 交易记录
- `PerformanceMetrics` - 性能指标

**使用示例：**
```python
from execution import TradeRecorder

recorder = TradeRecorder()
await recorder.start()

# 记录成交
recorder.record_trade(trade)

# 生成报告
report = recorder.generate_trade_report()
print(f"总收益: {report['performance']['total_return']}")
print(f"夏普比率: {report['performance']['sharpe_ratio']}")
```

## 完整示例

参见 `example_usage.py` 获取完整的使用示例。

```python
import asyncio
from execution import TradingSystem

async def main():
    system = TradingSystem(use_simulation=True)
    await system.initialize()
    await system.run_demo()
    await system.shutdown()

asyncio.run(main())
```

## 配置说明

### 交易所配置

```python
from execution import ExchangeConfig

config = ExchangeConfig(
    exchange_id='binance',
    api_key='your_key',
    api_secret='your_secret',
    sandbox=True,  # 使用测试网
    enable_rate_limit=True
)
```

### 执行策略配置

```python
from execution import ExecutionConfig, ExecutionStrategy

config = ExecutionConfig(
    strategy=ExecutionStrategy.TWAP,
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    total_amount=Decimal('1.0'),
    price_limit=Decimal('51000'),  # 价格限制
    time_limit=300.0,              # 时间限制(秒)
    max_slippage=Decimal('0.01'),  # 最大滑点1%
    urgency=0.5,                   # 紧急程度
    allow_partial=True             # 允许部分成交
)
```

## 错误处理

所有模块都包含完善的错误处理和重试机制：

```python
from execution import OrderManager

order_manager = OrderManager(exchange)
order_manager.max_retries = 3
order_manager.retry_delay = 0.5
```

## 日志记录

模块使用Python标准日志系统：

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('execution')
```

## 依赖

- ccxt >= 4.0.0
- asyncio
- decimal
- dataclasses

## 许可证

MIT License
