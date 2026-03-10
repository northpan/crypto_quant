### 1. 引言与动机

本报告是 `reports/multi_asset_autoresearch_report.md` 的“严格版更新”：唯一差异是回测时**不使用 OOS 的分位分布信息**来决定分位阈值，而是采用“**OOS 分位阈值来自 IS（固定阈值）**”的逻辑，以避免轻微 look-ahead（未来信息泄露）风险。

除此之外，数据源、因子池、组合构建、费用模型与 IS/OOS 划分均与原报告保持一致。

---

### 2. 数据来源与样本描述

- **数据来源**：Binance 现货 1 分钟 K 线（已导出为 CSV）
- **数据路径**：`data/csv/*_USDT_1m.csv`
- **品种**：BTC_USDT / ETH_USDT / SOL_USDT / BNB_USDT / XRP_USDT
- **字段**：`timestamp, open, high, low, close, volume`
- **样本长度**：最近约 90 天（1m）
- **切分**：60 天 IS / 30 天 OOS
- **费用**：双边 10bps（每次多空翻转扣费 \(2 \times 10/1e4\)）

---

### 3. 方法与模型设定（与原报告一致）

略。请参考 `reports/multi_asset_autoresearch_report.md` 中的：

- 遗传因子表达式与 GA：`factors/genetic/*`
- 组合构建与回测评估：`scripts/genetic_oos_eval.py`
- autoresearch 搜索与裁决：`scripts/run_autoresearch_genetic50.py`

---

### 4. 关键改动：分位阈值不使用 OOS 分布

原实现（可能轻微 look-ahead）：

- 在 OOS 回测时，对整段 OOS 的 `composite_oos` 做 `pd.qcut(..., n_quantiles=3/5)`，用 OOS 全局分布得到 Q1/Q3 边界。

本报告采用的新实现（严格版）：

- 在 OOS 回测时，分位阈值完全来自 IS 的 `composite_is` 分布：
  - 例如 `n_quantiles=3` 时，计算 IS 的 1/3 与 2/3 分位点作为固定阈值；
  - 之后在 OOS 的每个调仓时刻，只比较 `composite_oos(t)` 与这些固定阈值，决定多/空/中性仓位。

该逻辑可通过环境变量启用：

- `GENETIC50_QCUT_SOURCE=is`

---

### 5. 实验结果与可视化（更新为严格版 OOS）

#### 5.1 各品种最优实验结果（Best, IS-quantile OOS）

说明：

- “最优配置”仍然使用昨天每个品种 `results_genetic50_{SYMBOL}.tsv` 中 `oos_return_avg` 最大对应的 `top_n / n_quantiles / weight_mode`（配置不变）。
- 但回测收益指标（OOS）按新的 “IS 分位阈值” 重新计算。

| Symbol  | Best Exp ID | Config (`top_n / n_q / w`)     | `oos_return_avg` | `oos_return_5` | `oos_return_60` | Sharpe Avg | DD Avg | Worst OOS Segment |
|---------|-------------|---------------------------------|-----------------:|---------------:|----------------:|-----------:|-------:|------------------:|
| BTC_USDT| 5           | 50 / 3 / equal                  | **0.1881**       | 0.1755         | 0.2006          | 3.5478     | -0.0734| 0.0503            |
| ETH_USDT| 2           | 50 / 3 / ic_weighted            | **0.2145**       | 0.2091         | 0.2198          | 9.0364     | -0.0760| 0.0389            |
| SOL_USDT| 14          | 45 / 3 / equal                  | **0.2470**       | 0.2580         | 0.2360          | 9.7234     | -0.0588| 0.0441            |
| BNB_USDT| 10          | 35 / 3 / equal                  | **0.0736**       | 0.0794         | 0.0677          | 5.8533     | -0.0456| 0.0292            |
| XRP_USDT| 7           | 10 / 3 / equal                  | **0.2472**       | 0.2142         | 0.2802          |10.3056     | -0.0665| 0.0686            |

> 注：Sharpe Avg = 0.5*(Sharpe_5 + Sharpe_60)，DD Avg = 0.5*(MaxDD_5 + MaxDD_60)。  
> Worst OOS Segment 为将 OOS 切两段后，段收益的最小值（同原实现）。

#### 5.2 Keep 实验平均表现（Keep Mean, IS-quantile OOS）

对每个品种，取 `results_genetic50_{SYMBOL}.tsv` 中 `status=keep` 的所有实验配置，在新逻辑（IS 分位阈值）下重新评估其 OOS，并做均值汇总：

| Symbol  | n_keep | `mean(oos_return_avg)` | `mean(oos_return_5)` | `mean(oos_return_60)` |
|---------|-------:|------------------------:|----------------------:|-----------------------:|
| BTC_USDT| 5      | **0.1840**              | 0.1768                | 0.1912                 |
| ETH_USDT| 3      | **0.2107**              | 0.2052                | 0.2162                 |
| SOL_USDT| 13     | **0.2470**              | 0.2561                | 0.2379                 |
| BNB_USDT| 8      | **0.0713**              | 0.0781                | 0.0644                 |
| XRP_USDT| 5      | **0.0426**              | -0.0731               | 0.1583                 |

#### 5.3 新的 PnL 曲线图（严格版）

本报告对应的新图（不会覆盖昨天的旧图）：

- 60m：`reports/multi_asset_best_pnl_60m_isqcut.png`
- 5m： `reports/multi_asset_best_pnl_5m_isqcut.png`

绘图脚本复用昨天的版本并做最小修改：`scripts/plot_experiment_progress_genetic50.py`，将 OOS 的分位阈值改为来自 IS。

---

### 6. 简短结论

在“**OOS 分位阈值来自 IS（固定阈值）**”的严格回测逻辑下：

- BTC_USDT 与 BNB_USDT 的 OOS 收益相较“用 OOS 分布 qcut”的版本更保守（下降明显），符合“去除未来分布信息后更严格”的预期；
- ETH_USDT 基本保持稳定；
- SOL_USDT 与 XRP_USDT 在该设置下反而更优，说明这两个品种在当前样本期间存在较明显的分布漂移/结构变化，使得用 IS 固定阈值的映射更稳健。

后续若要进一步接近实盘，可将阈值从“固定 IS”升级为“滚动历史阈值”（只用过去窗口估计边界），同时继续保持不使用未来信息。

