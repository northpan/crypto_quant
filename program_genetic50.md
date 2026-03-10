# crypto_quant 遗传 50 因子组合 autoresearch（program_genetic50）

本实验将 Karpathy/autoresearch 的“**持续进化研究循环**”迁移到 **已挖掘的 50 个遗传因子**（来自 `reports/genetic_50_rounds/`）的组合与评估中：在固定数据划分与评估框架下，AI/脚本反复提出新配置、运行评估、依据明确规则决定 keep/discard，并记录可复现审计信息，持续迭代直到人为停止或达到预算上限。

## 目标

- **因子池**：固定为 50 个遗传因子表达式（`reports/genetic_50_rounds/best_exprs.pkl`），由 `scripts/btc_oos_eval_genetic50.py` 加载并在 OHLCV 上求值。
- **数据**：`data/csv/BTC_USDT_1m.csv` 最近 90 天；固定划分 **样本内 60 天**、**样本外 30 天**（保持可比性）。
- **主优化目标**：最大化样本外多空收益 `oos_return_avg`（扣除手续费后）。
- **约束**：保持评估框架与输出格式固定，保证不同配置可直接比较；当收益接近时以风险指标与换手为 tie-break。

---

## Setup（开始前检查）

1. **因子库产物存在且版本可追溯**：确认 `reports/genetic_50_rounds/best_exprs.pkl` 与 `reports/genetic_50_rounds/best_per_round_summary.csv` 存在（若不存在先运行 `scripts/run_50_rounds_and_composite_backtest.py`）。建议记录 `best_exprs.pkl` 的文件哈希（用于审计）。
2. **数据存在**：确认 `data/csv/BTC_USDT_1m.csv` 覆盖近 90 天 1m 数据。
3. **结果文件**：本轮实验写入根目录 `results_genetic50.tsv`（制表符分隔），并生成进度图 `reports/experiment_progress_genetic50_annotated.png`。

---

## 可调参数（每轮实验，而非预设参数网格）

在 `scripts/btc_oos_eval_genetic50.py` / `scripts/run_autoresearch_genetic50.py` 中，通过参数与内部采样逻辑控制 **“当前这一轮实验”** 的配置。  
这些参数只是“本轮可改的旋钮”，**不会预先枚举成一个完整的参数网格**；下一轮的具体取值由上一轮的 best 结果与随机扰动共同决定。

- **TOP_N**：使用因子数量（如 10 / 20 / 30 / 50）。默认按样本内 |IC_60| 排序取前 N（注意：这是“二次筛选”，需配合稳健性验证避免过拟合）。
- **N_QUANTILES**：分档数（3 或 5），多空为最高档与最低档。
- **WEIGHT_MODE**：`equal`（等权）或 `ic_weighted`（按样本内 |IC_60| 加权）。
- **OOS_METRIC**：`avg`（(ret_5+ret_60)/2）、`5`、`60` 或 `weighted`（带 `OOS_WEIGHT_5`）。
- **SEED**：随机种子（用于任何可能的随机过程；即使当前实现无随机，也作为审计字段固定记录）。

每次实验必须保持：样本内 60 天、样本外 30 天、手续费模型与主 program 一致、输出格式固定，便于解析与对比。

---

## 评价与裁决规则（Keep / Discard）

### 主指标

- **主指标**：`oos_return_avg`（扣费后）

### 接近时的 tie-break（必须明确且可复现）

当两次实验 `oos_return_avg` 差异小于阈值 \(\\epsilon\\) 时，使用如下顺序裁决：

- **阈值**：建议 `EPS_RETURN = 0.005`（即 0.5%）
- **第一 tie-break**：更高的 `oos_sharpe_avg`（可用 (sharpe_5+sharpe_60)/2 近似）
- **第二 tie-break**：更小的 `oos_max_drawdown_avg`（更不负/回撤更小）
- **第三 tie-break**：更低的 `oos_turnover_avg`（换手更低，更可交易）

> 注：若 `OOS_METRIC` 不是 `avg`（比如只看 5 期），仍然必须同时输出 5/60 的风险与换手用于审计；但 “keep/discard” 裁决应基于同一口径，避免不可比。

### 状态定义

- **keep**：优于当前最佳，或在 EPS_RETURN 内持平但 tie-break 更优
- **discard**：劣于当前最佳，或在 EPS_RETURN 内持平但 tie-break 更差
- **crash**：运行失败、超时、输出缺失或指标为 NaN

---

## 输出格式

脚本需打印（便于解析）：

```
in_sample_return_5: <float>
in_sample_return_60: <float>
oos_return_5: <float>
oos_return_60: <float>
oos_return_avg: <float>
oos_sharpe_5: <float>
oos_sharpe_60: <float>
oos_max_drawdown_5: <float>
oos_max_drawdown_60: <float>
oos_turnover_5: <float>
oos_turnover_60: <float>
fee_bps: 10
```

## 结果记录

每次实验的结果追加写入项目根目录 **`results_genetic50.tsv`**（制表符分隔）。建议列包含（最少集 + 审计字段）：

- **配置字段**：`experiment_id`、`top_n`、`n_quantiles`、`weight_mode`、`oos_metric`、`oos_weight_5`、`seed`
- **核心指标**：`oos_return_avg`、`oos_return_5`、`oos_return_60`
- **风险指标**：`oos_sharpe_5`、`oos_sharpe_60`、`oos_max_drawdown_5`、`oos_max_drawdown_60`
- **交易属性**：`oos_turnover_5`、`oos_turnover_60`、`fee_bps`
- **审计字段**：`best_exprs_sha256`（或 md5）、`data_end_ts`（数据末尾时间）、`runner_version`（参数表版本/脚本版本标识）
- **决策字段**：`status`、`description`

- **status**：`keep`（优于或平齐当前最佳）、`discard`（更差）、`crash`（运行失败）。
- **description**：简短描述该次配置（一行）。

---

## 稳健性验证（建议纳入循环，而非事后补做）

为避免“二次筛选 + 短 OOS”导致过拟合，建议至少加入一种稳健性协议：

- **多段 OOS（推荐）**：在保持总 90 天不变的前提下，用不同 30 天片段做 OOS 复测（例如滚动 3 段），以平均表现或最差段表现作为辅助裁决。
- **分段一致性**：将 OOS 30 天拆成前/后两段，要求两段同方向或不过度分裂。
- **费率敏感性**：在 5/10/20 bps 下复测，确保收益不是纯靠高换手堆出来。

---

## The experiment loop（持续进化）

实验不使用“预设 30 组参数网格”，而是以 **预算/时间** 控制（例如 30 次为默认预算上限），并具备“围绕当前最佳持续探索”的能力：

1. **初始化**：运行一次默认配置作为 baseline（写入 `results_genetic50.tsv`）。
2. **自适应生成新实验**：每一轮由 `run_autoresearch_genetic50.py` 在当前 best 配置附近做 **局部变异 + 少量全局跳跃**，并参考历史已经尝试过的配置，自动产生一个新的 `(top_n, n_quantiles, weight_mode, oos_metric, …)`；这一步不会预先枚举所有组合。
3. **运行评估**：调用 `genetic_oos_eval.run_eval(...)`，解析输出。
4. **裁决**：按“评价与裁决规则”决定本轮是 keep 还是 discard；若 keep，则更新内部的 `best_metrics` 与 `current_best_cfg`，为下一轮提供新的出发点。
5. **记录与审计**：将本轮配置与结果追加写入 `results_genetic50.tsv`（包含 seed、best_exprs 哈希等）。
6. **可视化**：可随时更新进度图，观察 best-so-far 的提升路径与搜索轨迹。
7. **停止条件**：达到预算上限、长时间无提升、或人为停止；之后可以用 `--resume` 接着在已有结果基础上继续“学习”。

---

## 与主 program.md 的关系

- **program.md**：面向全因子池（技术/量价/波动率等），可修改 `scripts/btc_oos_eval.py`，在分支上做自主实验并写 `results.tsv`。
- **program_genetic50.md**：面向 50 个遗传因子，通过参数驱动 `scripts/btc_oos_eval_genetic50.py` 并按 autoresearch 循环持续迭代，写 `results_genetic50.tsv`，用于在固定评估框架下“持续进化”出更稳健的组合配置。
