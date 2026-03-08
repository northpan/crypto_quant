# 策略参数搜索（阶段 4）

用于对均线交叉策略进行调参与参数搜索：RSI/均线周期、止损止盈比例、仓位上限等。

## 脚本

- **`scripts/param_search_backtest.py`**：网格搜索 + 可选 Optuna 优化

## 使用方式

```bash
# 进入项目并激活虚拟环境
cd /path/to/crypto_quant
source .venv/bin/activate   # 或 .venv\Scripts\activate (Windows)

# 仅网格搜索（默认回测区间 2026-03-01 ~ 2026-03-05）
python scripts/param_search_backtest.py

# 小网格快速验证（约 2~8 组）
PARAM_GRID_SMALL=1 python scripts/param_search_backtest.py

# 仅 Optuna 搜索（需安装 optuna: pip install optuna）
PARAM_SEARCH_MODE=optuna OPTUNA_TRIALS=25 python scripts/param_search_backtest.py

# 先网格再 Optuna
PARAM_SEARCH_MODE=both OPTUNA_TRIALS=20 python scripts/param_search_backtest.py

# 自定义回测区间
BACKTEST_START=2026-03-01 BACKTEST_END=2026-03-07 python scripts/param_search_backtest.py
```

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `BACKTEST_START` | 回测开始日期 | 2026-03-01 |
| `BACKTEST_END` | 回测结束日期 | 2026-03-05 |
| `PARAM_SEARCH_MODE` | 搜索模式：grid / optuna / both | grid |
| `PARAM_GRID_SMALL` | 1/true/yes 时使用小网格 | - |
| `OPTUNA_TRIALS` | Optuna 试验次数 | 25 |

## 目标与约束

- **目标**：最大化总收益率（`total_return`，即 ret）。
- **约束**：最大回撤优于 -15%（`max_drawdown > -0.15`），总交易次数 ≥ 5。

结果会打印 Top 5 参数组合，并将前若干组写入：

- `reports/param_search_grid.json`（网格）
- `reports/param_search_optuna.json`（Optuna，若运行）

## 策略参数说明

可在 `run_backtest(..., strategy_params={...})` 或搜索脚本中配置：

| 参数 | 说明 | 典型范围 |
|------|------|----------|
| short_period | 短期均线周期 | 5, 10 |
| long_period | 长期均线周期 | 20, 40 |
| rsi_period | RSI 周期 | 14 |
| rsi_max_open | 开多时 RSI 上限 | 65~75 |
| stop_loss_pct | 止损比例 | 0.005~0.01 |
| take_profit_pct | 止盈比例 | 0.01~0.02 |
| max_positions | 最多同时持仓标的数 | 2~4 |
| max_exposure_pct | 总敞口占权益上限 | 0.5~0.7 |
| position_pct | 单笔开仓占用现金比例 | 0.15~0.25 |

将搜索得到的最佳参数填入 `strategy_params` 即可在 main 回测中使用。
