# 数字货币量化交易系统 (Crypto Quant)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **目标夏普比率 > 4** 的分钟级数字货币量化交易策略系统

## 系统架构

```
crypto_quant/
├── core/              # 核心基类和工具
├── data/              # 数据获取与预处理
├── factors/           # 因子挖掘（122个因子）
├── models/            # 模型组合（集成学习）
├── risk/              # 风控优化
├── execution/         # 交易执行
├── backtest/          # 回测系统
├── strategies/        # 策略实现
├── config/            # 配置文件
├── main.py            # 主程序入口
└── requirements.txt   # 依赖列表
```

## 核心特性

### 1. 数据模块
- 支持50+主流数字货币
- 分钟级数据获取（1m/5m/15m）
- 现货和永续合约数据
- 基于CCXT的多交易所支持
- 高效Parquet存储

### 2. 因子挖掘（122个因子）

| 类别 | 数量 | 示例 |
|------|------|------|
| 技术指标因子 | 36 | RSI, MACD, 布林带, KDJ |
| 量价因子 | 26 | OBV, MFI, VWAP, 成交量比率 |
| 波动率因子 | 21 | ATR, 历史波动率, GARCH |
| 订单流因子 | 21 | 买卖压力, 资金费率, 持仓量 |
| 跨市场因子 | 18 | 期现价差, 跨交易所价差 |

### 3. 模型组合
- **LightGBM**: 快速训练，特征重要性
- **XGBoost**: 强大正则化，GPU加速
- **LSTM**: 时序特征捕捉
- **Transformer**: 多头注意力机制
- **集成学习**: Stacking/Blending动态权重

### 4. 风控体系
- 多层风控架构
- 仓位管理（凯利公式、波动率目标）
- 止损止盈（固定、追踪、ATR自适应）
- 回撤控制（最大回撤、日内回撤）
- 组合风险（相关性、杠杆控制）

### 5. 交易执行
- TWAP/VWAP执行策略
- 冰山订单
- 智能滑点控制
- 订单状态管理
- 支持模拟和实盘

## 快速开始

### 安装依赖

```bash
# 克隆项目
cd /mnt/okcomputer/output/crypto_quant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

编辑 `config/config.yaml`:

```yaml
trading:
  mode: backtest  # backtest, paper, live
  trade_type: futures  # spot, futures
  symbols:
    - BTC/USDT
    - ETH/USDT
    # ... 更多币种
```

### 运行回测

```bash
python main.py
```

### 运行实盘

```bash
# 设置API密钥
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# 启动实盘
trading:
  mode: live
```

## 策略示例

### 多因子策略

```python
from strategies import MultiFactorStrategy

config = {
    'prediction_horizon': 60,
    'target_sharpe': 4.0,
    'max_positions': 10
}

strategy = MultiFactorStrategy(config)
```

### 动量策略

```python
from strategies import MomentumStrategy

config = {
    'short_window': 10,
    'long_window': 30,
    'rsi_period': 14
}

strategy = MomentumStrategy(config)
```

### 均值回归策略

```python
from strategies import MeanReversionStrategy

config = {
    'bb_period': 20,
    'bb_std': 2.0,
    'rsi_period': 14
}

strategy = MeanReversionStrategy(config)
```

## 回测结果

运行回测后将生成以下报告：

```
reports/
├── backtest_2024-01-01_2024-02-01.html
├── equity_curve.png
├── drawdown.png
├── monthly_returns.png
└── comprehensive_report.png
```

### 绩效指标

| 指标 | 说明 |
|------|------|
| 夏普比率 | 风险调整后收益 |
| 最大回撤 | 最大资金回落 |
| Calmar比率 | 年化收益/最大回撤 |
| 胜率 | 盈利交易比例 |
| 盈亏比 | 平均盈利/平均亏损 |

## 模块说明

### 数据模块

```python
from data import DataManager

manager = DataManager("./data", "binance")
df = manager.get_data("BTC/USDT", "1h", days=7)
```

### 因子模块

```python
from factors import FactorPool

pool = FactorPool()
df_with_factors = pool.calculate_all_factors(df)
```

### 模型模块

```python
from models import LightGBMModel, ModelConfig

config = ModelConfig(model_type='lightgbm')
model = LightGBMModel(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 风控模块

```python
from risk import create_risk_manager

risk_mgr = create_risk_manager(
    account_balance=10000.0,
    risk_mode="moderate"
)
approval = risk_mgr.approve_trade("BTCUSDT", "long", 50000.0, 47500.0)
```

### 执行模块

```python
from execution import create_simulated_exchange, ExecutionEngine

exchange = create_simulated_exchange({'USDT': 10000.0})
engine = ExecutionEngine(exchange)
order = await engine.place_order('BTC/USDT', 'buy', 0.1, 50000)
```

### 回测模块

```python
from backtest import BacktestEngine, PerformanceAnalyzer

engine = BacktestEngine(initial_capital=100000.0)
results = engine.run(strategy)

analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze(results['returns'])
print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
```

## 配置详解

### 交易配置

```yaml
trading:
  mode: backtest           # 运行模式
  trade_type: futures      # 交易类型
  prediction_horizon: 60   # 预测时间窗口（分钟）
  retrain_interval: 24     # 重训练间隔（小时）
```

### 风控配置

```yaml
risk:
  max_position_pct: 0.2    # 最大仓位比例
  max_drawdown_pct: 0.15   # 最大回撤限制
  daily_loss_limit_pct: 0.05  # 日亏损限制
  leverage: 3              # 杠杆倍数
```

### 模型配置

```yaml
model:
  type: ensemble           # 模型类型
  sharpe_target: 4.0       # 目标夏普比率
  lightgbm:
    learning_rate: 0.05
    n_estimators: 500
```

## 性能优化

### 并行计算

```yaml
performance:
  parallel:
    enabled: true
    max_workers: 8
```

### 数据缓存

```yaml
performance:
  cache:
    enabled: true
    type: memory
    ttl: 3600
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest data/test_module.py
pytest factors/

# 生成覆盖率报告
pytest --cov=crypto_quant --cov-report=html
```

## 常见问题

### Q: 如何添加新的因子？

在 `factors/` 目录下创建新的因子文件，继承 `BaseFactor` 类：

```python
from factors.base_factor import BaseFactor

class MyFactor(BaseFactor):
    def calculate(self, df):
        # 实现因子计算逻辑
        return df['close'] / df['close'].shift(1) - 1
```

### Q: 如何添加新的策略？

在 `strategies/` 目录下创建新的策略文件，继承 `BaseStrategy` 类：

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 实现信号生成逻辑
        pass
```

### Q: 如何优化夏普比率？

1. 增加更多有效因子
2. 调整模型超参数
3. 优化仓位管理
4. 改进止损策略
5. 增加品种分散

## 风险提示

⚠️ **量化交易存在重大风险，包括但不限于：**

- 历史回测不代表未来收益
- 过拟合风险
- 模型失效风险
- 市场极端行情风险
- 技术故障风险
- 流动性风险

**请在使用实盘交易前充分了解风险，并仅用可承受损失的资金进行交易。**

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至：cryptoquant@example.com

---

**免责声明**：本项目仅供学习和研究使用，不构成任何投资建议。使用本项目进行交易产生的任何损失，开发者不承担责任。
