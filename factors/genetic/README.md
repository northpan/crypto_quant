# 遗传因子挖掘 (Genetic Factor Mining)

基于算子库 + 表达式树 + 遗传算法，自动生成并筛选因子，以 **IC** 与 **扣费后多空收益 ret** 为优化目标，挖掘 ret 正收益因子。

## 结构

| 文件 | 说明 |
|------|------|
| `operators.py` | 算子库：终端 `$close/$open/...`，滚动算子 `ts_mean/ts_std/ts_ema/ts_delta/...`，一元 `abs/log/sign/neg`，二元 `add/sub/mul/div`、`ts_corr` |
| `expression.py` | 表达式树：`eval_expr(expr, df)` 求值，`expr_to_str`/`get_subtrees` 供 GA 使用 |
| `evaluator.py` | 评估器：对单表达式计算 5/60 期 IC 与扣费后多空 ret（与 `factor_performance_report` 一致） |
| `genetic_factor.py` | 遗传算法：随机表达式、变异、交叉、适应度 `alpha*|IC| + beta*ret`，精英保留与进化 |

## 使用

```bash
# 默认 90 天数据、40 个体、25 代
python3 scripts/run_genetic_factor_mining.py

# 自定义
python3 scripts/run_genetic_factor_mining.py --days 90 --population 50 --generations 30 --max_depth 4 --out reports/genetic_factor_mining.csv --seed 42
```

输出：

- `reports/genetic_factor_mining.csv`：全体个体 expr、ic_5、ic_60、ret_5、ret_60、fitness
- 控制台：适应度 Top 10、以及 **ret_5 或 ret_60 > 0** 的因子列表

## 依赖

- 项目内 `factors.base_factor` 工具（如需要）
- 数据：`data/csv/BTC_USDT_1m.csv`（或 ETH/SOL），需含 `open, high, low, close, volume`
