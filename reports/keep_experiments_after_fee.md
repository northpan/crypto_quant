# Keep/持平实验 扣费后重跑结果

**参数**: `TRAIN_LOOKBACK_DAYS=7`, `INFER_RETRAIN_DAYS=1`, `FEE_BPS=10`（crypto 风格突变，模型快速切换）

## 按 oos_return_avg 从高到低（扣费后）

| 排名 | 配置 | oos_return_avg | oos_5 | oos_60 | 说明 |
|------|------|----------------|-------|--------|------|
| 1 | **6_top18_60IC_q3_tree** | **-0.723** | -0.999 | -0.447 | Tree(GBDT) + 每日重训，相对最好 |
| 2 | 1_baseline_all_factors_q5 | -0.756 | -0.998 | -0.514 | 全因子等权 q5 |
| 3 | 2_top20_IC_q5 | -0.761 | -0.998 | -0.523 | top20 by \|IC5+IC60\| q5 |
| 4 | 4_top18_IC_q3 | -0.790 | -0.999 | -0.581 | top18 by \|IC5+IC60\| q3 |
| 5 | 3_top20_IC_q3 | -0.795 | -0.999 | -0.590 | top20 by \|IC5+IC60\| q3 |
| 6 | 5_top18_60IC_q3_linear | -0.798 | -0.999 | -0.597 | linear top18 by 60\|IC\| q3 |
| 7 | 8_top18_60IC_q3_OOS5 | -0.9996 | -0.9996 | -0.597 | 仅优化 5 期，5 期崩盘 |
| - | 7_top18_60IC_q3_lgb | (无有效输出) | - | - | LightGBM 本轮无有效结果，需排查 |

## 结论

- 在当前 90 天数据（样本外 30 天）下，**扣费后**所有配置样本外均为负收益；相对最好的是 **tree + 7 天回看 + 1 天重训**（-72.3%），其次为 baseline 全因子 q5（-75.6%）。
- **Tree 在 INFER_RETRAIN_DAYS=1 下**比 linear 单模型略好（-72.3% vs -79.8%），符合“风格突变时快速重训”的设定。
- **仅 5 期 (OOS5)** 在本段 OOS 表现极差（-99.96%），不宜单独作为目标。
- 详细数值见 `reports/keep_experiments_after_fee.csv`。
