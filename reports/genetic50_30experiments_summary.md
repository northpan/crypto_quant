# 50 遗传因子 · 30 次组合实验汇总

## 设置

- **因子池**：`reports/genetic_50_rounds/best_exprs.pkl`（50 个遗传因子）
- **数据**：BTC_USDT 1m，90 天；样本内 60 天，样本外 30 天
- **手续费**：10 bps/side，回测中已扣除
- **实验数**：30 次，参数网格：top_n ∈ {10,15,20,25,30,35,40,45}，n_quantiles ∈ {3,5}，weight_mode ∈ {equal, ic_weighted}
- **结果文件**：`results_genetic50.tsv`

## 最佳配置（样本外 oos_return_avg）

| 排名 | experiment_id | top_n | n_quantiles | weight_mode | oos_return_avg | oos_return_5 | oos_return_60 | oos_sharpe_5 | oos_sharpe_60 |
|------|----------------|-------|-------------|-------------|----------------|--------------|---------------|--------------|---------------|
| 1 | 26 | 40 | 3 | ic_weighted | **36.29%** | 33.43% | 39.15% | 5.85 | 6.93 |
| 2 | 28 | 45 | 3 | equal | 34.98% | 33.34% | 36.62% | 5.86 | 6.49 |
| 3 | 27 | 40 | 5 | ic_weighted | 25.80% | 26.22% | 25.37% | 6.09 | 5.95 |
| 4 | 22 | 35 | 3 | ic_weighted | 34.66% | 31.43% | 37.89% | 5.48 | 6.75 |
| 5 | 20 | 35 | 3 | equal | 33.75% | 29.26% | 38.23% | 5.18 | 6.79 |

## 结论

- **最佳单次**：top_n=40、n_quantiles=3、ic_weighted，oos_return_avg ≈ **36.29%**（扣费后）。
- **规律**：n_quantiles=3 普遍优于 5；ic_weighted 在 top_n 较大时略优；增加 top_n 至 40 左右收益最高，再增到 45 略降。
- **风险**：最佳配置 oos_max_drawdown_5/60 约 -5.7% / -5.1%，oos_sharpe 约 5.8 / 6.9。
