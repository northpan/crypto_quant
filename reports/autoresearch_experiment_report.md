# Autoresearch 实验报告：借鉴 Karpathy 的 AI 自主实验循环

> 组会分享文档 · 量化因子实验

---

## 一、借鉴 Karpathy autoresearch 的核心思路

[Andrej Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) 是一个 **AI 自主实验循环**：代理在固定评估框架下反复修改代码、运行、根据指标决定 keep/discard，无需人工干预，持续探索直到人为打断。

我们将这一思路迁移到 **加密货币量化因子** 场景：

| 原项目 (nanochat 训练) | 本项目 (crypto 量化) |
|------------------------|----------------------|
| 修改训练脚本 | 修改 `scripts/btc_oos_eval.py` |
| 指标：validation loss | 指标：`oos_return_avg`（样本外多空收益） |
| results.tsv 记录 keep/discard | 同结构 results.tsv |
| git commit + reset 回滚 | 同流程 |

**关键设计原则**：

- **单一可量化指标**：`oos_return_avg` 作为唯一主优化目标
- **快速反馈**：单次运行约 1–2 分钟
- **只改一个目标文件**：`btc_oos_eval.py`，其余代码只读参考
- **评估框架不变**：数据划分、输出格式固定，保证可比性

---

## 二、实验设置

### 2.1 数据与划分

- **标的**：BTC_USDT 1 分钟 K 线，过去 3 个月（约 90 天）
- **样本内**：前 2 个月（约 60 天），用于因子 IC 估计、复合因子拟合
- **样本外**：最后 1 个月（约 30 天），用于评估，**主优化目标**

### 2.2 因子与策略

- **因子池**：83 个（技术/量价/波动率），来自项目 `FactorPool`
- **复合方式**：等权线性、或树模型（GBDT/LightGBM）/MLP 预测
- **多空规则**：按因子分位数做多 top、做空 bottom，非重叠区间累计收益
- **手续费**：10 bps 双边/次调仓，每次仓位变化时扣除

### 2.3 可调实验项

- 因子筛选：top N by |IC|，IC 周期（5 期/60 期/混合）
- 分位数档位：3 分位（tercile）、4 分位、5 分位
- 复合方式：linear / tree / mlp / lgb
- OOS 指标：`avg`（5+60）/2、仅 5 期、仅 60 期、加权
- 树/MLP：训练回看窗口、样本外滚动重训频率

---

## 三、实验结果

### 3.1 实验进度概览

共 **39 组实验**，7 次 keep（其中 6 次提升最优 OOS），实验进度见 `experiment_progress_33_annotated.png`。

### 3.2 最优 OOS 提升路径

| 实验 | OOS (avg) | 改进原因/条件 |
|------|-----------|----------------|
| 1 | 13.3% | baseline 全因子等权 |
| 2 | 19.1% | top 20 factors by \|IC\| |
| 8 | 20.4% | n_quantiles=3 三档多空 |
| 12 | 21.0% | top 18 factors n_quantiles=3 |
| 16 | 22.3% | top 18 by 60-period \|IC\| n_quantiles=3 |
| 29 | 38.8% | OOS_METRIC=5（仅 5 期，口径变化） |

### 3.3 重要发现

1. **线性优于非线性**：在相同因子与分位数下，linear 与 tree/LightGBM 持平，MLP 易过拟合
2. **60 期 IC 选因子更稳**：用 60-period |IC| 选 top 18 优于 5 期或混合
3. **3 分位多空**：比 5 分位更集中，信号更清晰
4. **指标口径影响大**：仅优化 5 期时 OOS 数值高，但与 avg 口径不可直接比较；扣费后重跑显示 5 期单独表现极差

### 3.4 扣费后重跑结论

对 keep/持平配置在 `TRAIN_LOOKBACK_DAYS=7`、`INFER_RETRAIN_DAYS=1`、`FEE_BPS=10` 下重跑（见 `keep_experiments_after_fee.md`）：

- **所有配置样本外均为负收益**
- 相对最好：tree + 7 天回看 + 1 天重训（-72.3%）
- 说明：当前 90 天数据下，需更稳健的因子或更长样本验证

---

## 四、迭代方式

### 4.1 实验循环

```
修改 btc_oos_eval.py → git commit → 运行评估 → 解析 oos_return_avg
    → 优于当前最佳：keep（保留 commit）
    → 否则：git reset --hard HEAD~1（回滚）
    → 写入 results.tsv
    → 重复
```

### 4.2 探索方向

- 因子数量（12 / 15 / 18 / 20 / 25 / 30）
- IC 周期与权重
- 分位数档位
- 线性 / 树 / MLP / LightGBM
- 训练窗口、样本外重训频率
- OOS 指标（avg / 5 / 60 / weighted）

### 4.3 后续计划

- **遗传算法因子挖掘**：`genetic_factor_mining_round*.csv` 已有多轮结果
- **扣费敏感性**：调整 FEE_BPS 做稳健性测试
- **多段 OOS**：不同 30 天片段复测，验证跨时段稳定性

---

## 五、参考资源

- `program.md` — 实验循环与规则
- `results.tsv` — 39 组实验记录
- `reports/experiment_progress_33_annotated.png` — 实验进度与改进点
- `reports/keep_experiments_after_fee.md` — 扣费后重跑结论
- `scripts/btc_oos_eval.py` — 唯一可修改的评估脚本
