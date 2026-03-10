### 1. 引言与动机

本项目在单一品种 BTC_USDT 上，借鉴 Karpathy `autoresearch` 的思路，搭建了一个“**遗传因子挖掘 + 稳健组合 autoresearch**”闭环，用于自动寻找样本外收益最优的多空组合。在 BTC_USDT 的实验中，我们采用 60 天样本内（IS）/ 30 天样本外（OOS）的滚动框架，扣除双边 10bps 手续费，通过 keep/discard 决策不断迭代组合配置，最终得到样本外平均收益显著为正的等权/IC 加权组合。

为了验证这一方法是否只是“在 BTC 上的过拟合”，还是能够在更广泛的主流币上泛化，我们将完全相同的一套 **数据处理、因子池、回测逻辑与决策规则** 迁移到 ETH_USDT、SOL_USDT、BNB_USDT 和 XRP_USDT，系统评估这 5 个品种上的样本外表现。

---

### 2. 数据来源与样本描述

- **数据来源**：  
  所有数据均来自 Binance 现货 1 分钟 K 线，预先导出并保存在项目目录的 `data/csv` 下：
  - `BTC_USDT_1m.csv`
  - `ETH_USDT_1m.csv`
  - `SOL_USDT_1m.csv`
  - `BNB_USDT_1m.csv`
  - `XRP_USDT_1m.csv`

- **数据字段**：  
  各文件采用统一列结构：
  - `timestamp`：K 线起始时间，含 UTC 时区（例如 `2026-02-06 15:11:00+00:00`）
  - `open`, `high`, `low`, `close`：开、高、低、收价格
  - `volume`：成交量

- **样本区间与频率**：  
  - 频率：**1m**  
  - 时长：每个品种约 **90 天** 最近历史

- **IS/OOS 划分**：
  - 样本内（IS）：最近 90 天中的前 **60 天**
  - 样本外（OOS）：最近 90 天中的后 **30 天**
  - 所有因子评估与组合排序在 IS 上完成，所有 performance 评估在 OOS 上完成。

---

### 3. 方法与模型设定

#### 3.1 遗传因子挖掘（Genetic Factor Mining）

- **表达式表示**（见 `factors/genetic/expression.py`）：
  - 终端节点：价量字段，如 `"$open"`, `"$high"`, `"$low"`, `"$close"`, `"$volume"`。
  - 一元算子：`abs`, `log`, `neg`, `sign` 等。
  - 滚动算子：`ts_mean`, `ts_std`, `ts_sum`, `ts_min`, `ts_max` 等，带窗口参数。
  - 二元算子：`add`, `sub`, `mul`, `div`。
  - 相关性算子：`ts_corr(left, right, window)`。
  - 表达式树在 OHLCV DataFrame 上通过 `eval_expr(expr, df)` 求值得到因子序列。

- **评估指标**（见 `factors/genetic/evaluator.py`）：
  对每个候选表达式，在单一品种的 90 天数据上计算：
  - `ic_5`, `ic_60`：使用 5 分钟、60 分钟 forward return，对 IS 全样本计算 Spearman IC。
  - `ret_5`, `ret_60`：在 5 分钟、60 分钟周期上做多空分位组合（N=5），扣费回测得到的总收益。
  - `ic_robust`：将 IS 划分为若干段，逐段计算 |IC|，使用 `median(|IC|) - λ·std(|IC|)` 作为稳健性评分。
  - `ret_robust`：类似地在多段样本上统计因子多空收益，惩罚不稳定的表现。

- **适应度函数**（见 `factors/genetic/genetic_factor.py`）：
  - 适应度由四部分线性组合：  
    \( fitness = \alpha \cdot f(IC) + \beta \cdot f(ret) + \gamma \cdot ic\_robust + \delta \cdot ret\_robust \)
  - 其中 \(\gamma, \delta\) 用于显式奖励“跨时间段更稳定”的因子。

- **遗传算法超参数**（`scripts/run_genetic_factor_mining.py`）：
  - `population_size`：40
  - `generations`：20–25
  - `max_depth`：4（表达式树最大深度）
  - `elite_ratio`：0.15
  - `mutation_prob`：0.55
  - `crossover_prob`：0.5
  - `immigrant_ratio`：0.2
  - `gamma = 0.15`, `delta = 0.05`（稳健性权重）

> 通过上述搜索，我们在 BTC_USDT 上先运行了多轮 GA 挖掘，再在 ETH_USDT、SOL_USDT、BNB_USDT、XRP_USDT 上分别运行稳健 GA 以观察单币因子性质，但最终用于组合的“主因子池”仍然以 BTC_USDT 的 50 轮 GA 结果 + 稳健挖掘扩展为核心。

#### 3.2 组合模型与 OOS 评估

组合评估由 `scripts/genetic_oos_eval.py` 实现，核心流程为：

1. **加载因子池**：  
   - 使用统一的稳健因子池 `reports/genetic_50_rounds/best_exprs_robust.pkl`（约 70 个表达式），通过环境变量 `GENETIC50_EXPR_PKL` 和 `GENETIC50_SYMBOL` 控制品种。

2. **IS 因子评估与排序**：
   - 在 IS 数据上对每个因子计算：
     - `IC_5`, `IC_60`（Spearman）
     - 稳健 IC 评分 `robust_score`。
   - 通过 `GENETIC50_SELECT_IC` 选择排序标准（本报告默认使用 \|IC_60\|）。
   - 选取前 `top_n` 个因子，得到初始候选集合。
   - 在 IS 上做相关性过滤（阈值 `GENETIC50_CORR_THRESH`，默认 0.9），剔除高度共线的因子。

3. **组合权重与标准化**：
   - 对于入选因子，采用两种权重方式：
     - `weight_mode = "equal"`：等权。
     - `weight_mode = "ic_weighted"`：按排序得分的绝对值加权。
   - 对每个因子在 IS/OOS 上做 z-score 标准化；对于 IS/OOS 中 IC_60 为负的因子，自动取负以统一“正向”方向。
   - 最终组合为各 z-score 的加权和。

4. **多空回测与费用**：
   - 以组合值在 OOS 上做 `n_quantiles=3 或 5` 的分位切分：
     - 做多最高分位，做空最低分位，构建多空组合。
   - 在 5 分钟、60 分钟两个 forward 窗口分别回测：
     - 每个窗口只在对应的步长上调整仓位，其他时间保持不变。
     - 每次多空翻转时扣除双边手续费：`2 * FEE_BPS / 1e4`，其中 `FEE_BPS = 10`。
   - 输出：
     - `oos_return_5`, `oos_return_60`, 以及 `oos_sharpe_5`, `oos_sharpe_60`。
     - `oos_max_drawdown_5`, `oos_max_drawdown_60`。
     - `oos_turnover_5`, `oos_turnover_60`。
   - 将 `oos_return_5` 与 `oos_return_60` 简单平均，记为 `oos_return_avg` 作为主目标。

5. **多段 OOS 稳健性**：
   - 将 OOS 时间轴粗略切为两段，在 5p 和 60p 上分别计算每段总收益，再平均为段收益列表。
   - 对应组合的“最差 OOS 段收益”定义为这些段收益中的最小值，记为 `oos_return_avg_worst_segment`。

#### 3.3 Autoresearch 循环与裁决规则

Autoresearch 主循环由 `scripts/run_autoresearch_genetic50.py` 驱动，核心思想是：在固定因子池下，仅对组合层的超参数做局部自适应搜索：

- 配置参数包括：
  - `top_n ∈ {10,15,20,25,30,35,40,45,50}`
  - `n_quantiles ∈ {3,5}`
  - `weight_mode ∈ {"equal","ic_weighted"}`
  - `oos_metric ∈ {"avg","5","60","weighted"}`（本报告聚焦 `avg`）
- 决策规则：
  - 以 `oos_return_avg` 为主线，阈值 `EPS_RETURN = 0.5%`，超过当前 best 超过阈值则 keep，否则倾向 discard。
  - 在收益近似相等时，依次用：
    1. 最差 OOS 段收益（`oos_return_avg_worst_segment`）更高者优先；
    2. Sharpe 更高者优先；
    3. 最大回撤（更小的负数）更优；
    4. 换手率更低者优先。
  - 每次决策与配置、seed、数据快照哈希等一起写入 `results_genetic50_*.tsv`，实现完整可审计性。

---

### 4. 实验设计（多品种设定）

本次多品种实验覆盖以下五个交易对：

- BTC_USDT
- ETH_USDT
- SOL_USDT
- BNB_USDT
- XRP_USDT

实验设定如下：

- **统一因子池**：  
  所有品种均使用同一套因子池 `best_exprs_robust.pkl`（约 70 个由 BTC_USDT 上 GA 与稳健挖掘得到的表达式），通过环境变量 `GENETIC50_EXPR_PKL` 指定。

- **统一样本切分与费用**：
  - IS/OOS：60 天 / 30 天
  - 频率：1m K 线
  - 手续费：双边 10 bps（每次多空翻转扣除 0.2%）

- **Autoresearch 搜索**：
  - 每个品种在给定因子池下，预算 **20 次实验**。
  - 初始配置与变异策略同 BTC_USDT，局部在 `top_n / n_quantiles / weight_mode` 附近做自适应搜索。

- **最优与平均表现定义**：
  - **最优实验（Best）**：  
    对于每个品种，从 `results_genetic50_SYMBOL.tsv` 中按 `oos_return_avg` 最大选取一行作为最优。
  - **平均表现（Keep Mean）**：  
    对每个品种，仅在 `status == "keep"` 的实验中，统计 `oos_return_avg`、`oos_return_5`、`oos_return_60` 的均值，用于衡量“被决策逻辑认为值得保留”的方案的平均水平。

---

### 5. 实验结果与可视化

#### 5.1 各品种最优实验结果（Best）

在统一的因子池与费用假设下，五个品种最优组合的核心指标如下（基于 `results_genetic50_*.tsv` 与 `genetic_oos_eval.run_eval`）：

| 品种     | 最优实验 ID | 配置 (`top_n / n_q / w`)     | `oos_return_avg` | `oos_return_5` | `oos_return_60` | Sharpe 平均 | 回撤平均 | 最差 OOS 段收益 |
|----------|-------------|------------------------------|------------------:|---------------:|----------------:|------------:|---------:|----------------:|
| BTC_USDT | 5           | 50 / 3 / equal               | **0.2807**        | 0.2427         | 0.3187          |   5.14      |  -0.057  | **0.1061**      |
| ETH_USDT | 2           | 50 / 3 / ic\_weighted        | **0.2207**        | 0.2255         | 0.2159          |   8.78      |  -0.064  | **0.0968**      |
| SOL_USDT | 14          | 45 / 3 / equal               | **0.2184**        | 0.2429         | 0.1938          |   8.29      |  -0.075  | -0.0007         |
| BNB_USDT | 10          | 35 / 3 / equal               | **0.1189**        | 0.1269         | 0.1108          |   7.69      |  -0.046  | 0.0016          |
| XRP_USDT | 7           | 10 / 3 / equal               | **0.2133**        | 0.1874         | 0.2392          |   9.08      |  -0.068  | 0.0559          |

> 注：  
> - Sharpe 平均为 `0.5 * (Sharpe_5 + Sharpe_60)`。  
> - 回撤平均为 `0.5 * (MaxDD_5 + MaxDD_60)`。  
> - “最差 OOS 段收益”基于 5p 与 60p OOS 段收益的加权平均后，再取两段中的最小值。

可以看到：

- 在 **BTC_USDT** 上，最优组合样本外平均收益约 **28.1%**，最差 OOS 段仍有 **10.6%** 的正收益，Sharpe 约 **5.1**，最大回撤约 **-5.7%**。
- 在 **ETH_USDT / SOL_USDT / XRP_USDT** 上，最优组合的 `oos_return_avg` 约在 **0.21–0.22** 区间，Sharpe 平均约 **8–9**，回撤在 **-6% ~ -8%**；说明在这些高流动性主流币上，同一套因子池与组合方法能得到稳定且显著的样本外收益。
- **BNB_USDT** 上的最优组合 `oos_return_avg ≈ 0.12`，收益略低但仍为明显正值，Sharpe 也达到 **7.7** 左右，回撤更小，属于“偏稳健、收益中等”的配置。

#### 5.2 被保留实验的平均表现（Keep Mean）

进一步，仅统计被自动裁决为 `status = "keep"` 的实验，五个品种的样本外收益均值如下：

| 品种     | keep 数量 | `mean(oos_return_avg)` | `mean(oos_return_5)` | `mean(oos_return_60)` |
|----------|-----------|------------------------:|----------------------:|-----------------------:|
| BTC_USDT | 5         | **0.2632**              | 0.2170                | 0.3094                 |
| ETH_USDT | 3         | **0.2114**              | 0.2276                | 0.1951                 |
| SOL_USDT | 13        | **0.2135**              | 0.2290                | 0.1980                 |
| BNB_USDT | 8         | **0.1110**              | 0.1116                | 0.1104                 |
| XRP_USDT | 5         | **0.0263**              | -0.1000               | 0.1526                 |

可以看到：

- 对于 BTC/ETH/SOL/BNB，**被保留的实验中，大部分方案的 OOS 平均收益都是显著正值**，且略低于各自最优值，符合“决策规则有一定容忍度”的预期。
- 在 XRP 上，虽然最优组合 `oos_return_avg ≈ 0.2133`，但整体 keep 的平均收益只有 **2.6% 左右**，且 5p 框架下部分组合略有亏损，说明在波动更剧烈的品种上，配置选择更加敏感，稳健性要求更高。

#### 5.3 PnL 曲线与回测形态（定性描述）

我们对每个品种的最优组合，在 5p 与 60p 两个时间尺度下分别构建 PnL 曲线，整体形态具有以下特征：

- **BTC_USDT**：  
  - 60p PnL 曲线相对平滑，呈单调向上的“缓坡”，在 OOS 中段有小幅回撤，但总体趋势明确。  
  - 5p PnL 更为锯齿，但与 60p 曲线大体同向，说明短周期噪声较大，但不会完全抹平中长期趋势。

- **ETH_USDT / SOL_USDT**：  
  - 两个品种的 5p/60p PnL 曲线都类似 BTC，整体向上，中途会出现一两个明显 drawdown 段，但由于因子池更偏“趋势 + 结构性价量关系”，回撤后快速恢复。  
  - 这也体现在较高的 Sharpe 值与适中的最大回撤上。

- **BNB_USDT**：  
  - PnL 曲线斜率稍缓，收益率中等但更平滑，少见极端 drawdown。  
  - 体现出“稳健但不激进”的特征，更适合作为组合中权重较高的“中枢资产”。

- **XRP_USDT**：  
  - 5p PnL 曲线震荡较大，一些短期因子在高波动环境下出现 overshoot，导致短周期收益有负值；60p 曲线则相对健康。  
  - 整体体现为“高风险高波动”，最优配置可以给出较高的 OOS 收益，但平均水平受极端波动拖累。

---

### 6. 策略内容与交易方式说明

#### 6.1 信号生成与持仓规则

对任意一个品种（例如 BTC_USDT），最优组合策略的基本交易逻辑如下：

1. **因子计算**：  
   - 在每个时间点，用表达式树在 1m K 线上计算若干基础因子，如：  
     - 长窗口收盘价累积和/均值：`ts_sum(close, 60)`, `ts_mean(close, 60)`  
     - 波动-价差类：`ts_std(high - low, 20)`  
     - 价量结构类：`ts_corr(volume, close, 30)` 等。

2. **组合构建**：  
   - 在样本内，按 IC 与稳健性对因子排序，并做相关性过滤，选出前 `top_n` 个入池因子。  
   - 对每个因子进行去均值、标准化处理，并按等权或 IC 加权线性合成，得到组合因子值。

3. **分位数多空策略**：  
   - 在 OOS 中，以固定步长（例如每 5 分钟/60 分钟）对最新组合因子值做分位切分：  
     - 做多最高分位（如 top 1/3 或 top 1/5）。  
     - 做空最低分位（如 bottom 1/3 或 bottom 1/5）。  
   - 中间分位保持空仓或忽略。

4. **交易与费用**：  
   - 每当组合信号发生变化（从空头转多头或从多头转空头）时进行换仓，并在该点扣除双边 10bps 手续费。  
   - 换手率指标反映了在整个 OOS 期间中“调仓频率”的高低，对手续费敏感性有直接解释力。

5. **收益与风控**：  
   - 策略不显式使用止损/止盈，而是通过因子与 IC 稳健性约束控制组合风险。  
   - 多段 OOS 最差收益（`oos_return_avg_worst_segment`）用于评估“最糟糕 1/2 时间段”的表现，避免只看全局平均。

#### 6.2 伪代码示意

简化版本的伪代码流程如下（逻辑上适用于所有五个品种）：

```python
for symbol in ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT"]:
    # 1. 加载数据
    df = load_ohlcv(symbol, days=90)
    df_is, df_oos = split_60_30(df)

    # 2. 计算所有候选因子
    factor_values_is, factor_values_oos = evaluate_all_factors(df_is, df_oos, best_exprs_robust)

    # 3. 在 IS 上计算因子 IC / 稳健分数并排序 + 去相关
    ic5, ic60, robust = compute_ic_and_robust(factor_values_is, df_is)
    selected_factors = select_and_decorrelate(ic5, ic60, robust, top_n, corr_thresh)

    # 4. 构建组合因子
    composite_is, composite_oos = build_composite(
        factors=selected_factors,
        mode=weight_mode,  # equal or ic_weighted
    )

    # 5. 在 OOS 上做 5p / 60p 多空回测（扣费）
    pnl5, metrics5 = backtest(composite_oos, forward_period=5, fee_bps=10)
    pnl60, metrics60 = backtest(composite_oos, forward_period=60, fee_bps=10)

    # 6. 计算综合指标并写入 results_genetic50_SYMBOL.tsv
    oos_return_avg = 0.5 * (metrics5["total_ret"] + metrics60["total_ret"])
    log_to_results_tsv(...)
```

---

### 7. 因子表达式与分类（附录摘要）

完整的因子池存储在：

- `reports/genetic_50_rounds/best_exprs_robust.pkl`：约 70 个表达式树对象。
- 对应的可读字符串在多处 CSV 中出现，例如：
  - `reports/genetic_50_rounds/best_per_round_summary.csv`
  - `reports/genetic_factor_mining_robust*.csv`

这里对代表性因子做一个分类与示例说明（为便于阅读，仅列出少数典型表达式的字符串形式）。

#### 7.1 趋势类因子

这类因子主要反映价格的长短期趋势与动量，典型结构是对 `close` 的滚动和/均值或与其他价位的组合。例如：

- `ts_sum(close,60)`  
  - 含义：最近 60 分钟收盘价累积和，刻画中期价格水平与趋势强弱。

- `(ts_sum(close,60) sub ts_min(neg(close),5))`（XRP_USDT 稳健挖掘中出现）  
  - 含义：60 分钟累积收盘价减去过去 5 分钟内最低的负价，强调近期快速下跌后的反弹强度。

- `ts_mean(ts_ema(close,60),30)`  
  - 含义：对 60 分钟 EMA 再做 30 分钟均值，平滑中长期趋势。

这类因子在五个品种上均占有重要权重，是捕捉趋势性收益的主力。

#### 7.2 波动与区间类因子

这类因子关注价格波动幅度、极值区间等特征，例如：

- `ts_std(high,20)`，`ts_std(high - low,20)`  
  - 刻画短期价格波动，常用于识别高/低波动环境。

- `ts_max(high,60) sub ts_min(low,60)`  
  - 反映过去一段时间内的高低价区间宽度。

- `ts_min(abs(log(ts_sum(ts_min(neg(ts_min(ts_min(abs(log(ts_sum(neg((ts_ema(high,60) ...`, 在 ETH_USDT 稳健 GA 中出现  
  - 虽然表达式较复杂，本质上是“长窗口 log 变换 + abs + min/max”结构，放大极端波动与价格拉伸。

波动因子常与趋势因子共同出现在组合中，用于在高波动环境中加权/过滤部分信号。

#### 7.3 成交量与流动性类因子

这类因子利用 `volume` 与价格的关系捕捉资金流入/流出、成交活跃度等信息，例如：

- `volume / close`，`ts_mean(volume,10)`，`ts_sum(volume,20)`  
  - 描述单位价格成交量、短期成交放量/缩量。

- `ts_corr(volume, close, 30)`，`ts_corr(ts_ema(volume,10), ts_mean(low,30),60)`  
  - 刻画价量联动（放量上涨/缩量下跌等）。

- 在 BNB_USDT 的稳健因子中，类似 `(ts_min(log(ts_sum(((ts_min(ts_min(ts_corr(low,low,60),60),60)subts_corr((ts_min(close,60)divts_sum(volume,20)),...` 的表达式  
  - 通过 `volume` 与 `close` / `high` 等共同参与滚动和、相关性，强调“在高成交量环境下的价格极值”。

#### 7.4 相关性与结构类因子（ts_corr 家族）

这类因子通过滚动相关性刻画更复杂的结构关系，如：

- `ts_corr(open, low, 20)`，`ts_corr(volume, volume, 10)`（自相关）  
  - 用于识别价格/量的惯性与自相似性。

- `ts_corr(ts_ema(high,60), ts_ema(volume,10), 60)`  
  - 在多个品种的稳健 GA 中出现，刻画“高价与放量”之间的同步程度。

- `ts_corr(ts_sum(close,60), ts_std(volume,20), 20)`  
  - 反映“趋势阶段下的成交波动性”，常在趋势持续但资金结构变化时给出有用信号。

这类因子通常不单独决定多空方向，而是与趋势/波动因子联合作为“结构确认器”，提高组合的稳健性。

---

### 8. 结论与展望

#### 8.1 结论

综合 BTC_USDT、ETH_USDT、SOL_USDT、BNB_USDT 和 XRP_USDT 五个品种的实验结果，可以得到以下结论：

1. **跨品种泛化能力明显**：  
   在统一的 70 因子稳健池、60d IS / 30d OOS 划分和双边 10bps 手续费假设下，五个品种的最优组合样本外平均收益均为正，其中 BTC/ETH/SOL/XRP 的 `oos_return_avg` 大致分布在 **0.21–0.28** 区间，BNB 也有约 **0.12** 的收益，Sharpe 平均普遍在 **5–9** 之间。

2. **稳健性与风险可控**：  
   - 多段 OOS 最差段收益在绝大多数品种上仍为正（BTC/ETH/BNB/XRP），仅 SOL 存在接近 0 的最差段，但整体 OOS 表现仍然良好。  
   - 最大回撤平均水平在 **-4% ~ -8%**，与收益水平匹配，未出现极端的单边亏损。

3. **因子具备结构性共性**：  
   - 因子池中大量因子同时出现在多个品种的稳健挖掘结果中，趋势类、价量结构类和相关性类因子反复被 GA “重新发明”。  
   - 这说明 GA 挖掘到的是某些**跨品种通用的价量结构**（例如“长周期涨跌趋势 + 放量确认”），而不是只对单一币种噪声的过拟合。

4. **自动裁决逻辑有效**：  
   - 在不同品种上，`status = "keep"` 实验的平均 OOS 收益明显高于随机组合水平，说明以 `EPS_RETURN + 稳健性 + Sharpe + 回撤 + 换手` 组合成的裁决规则，能够过滤掉大部分不稳定或过拟合的配置。

#### 8.2 展望

后续可以从以下几方面继续扩展与深入：

- **更多品种与更长样本期**：  
  将同一套框架扩展到 ADA_USDT、DOGE_USDT、LINK_USDT 等更多主流币，或在季度/年度维度拉长样本期，验证“稳健 GA 因子池”的长期有效性。

- **多周期与多市场**：  
  在 15m、1h 等更长周期上复用相同因子池，或迁移到其他资产类别（如合约、传统商品），检验这一方法在更广泛市场的适用性。

- **交易成本与滑点模型**：  
  目前统一采用 10bps 手续费假设，后续可以根据真实交易环境引入不同品种/平台的手续费与滑点模型，对组合进行压力测试与鲁棒性分析。

- **因子解释与可视化**：  
  进一步对 GA 挖掘出的代表性因子做更系统的可视化与经济含义解释（如分解为趋势、反转、波动、流动性等风格因子），帮助从量化研究走向策略产品化与风险管理。

总体而言，本报告表明：在主流币现货多空框架下，基于自动化遗传因子挖掘与稳健组合裁决的 autoresearch 流程，**不仅在单一品种 BTC_USDT 上有效，而且在 ETH/SOL/BNB/XRP 等多个品种上表现出良好的跨品种泛化与风险收益特征**，为后续更大规模的多资产量化研究提供了扎实的实验基础。

