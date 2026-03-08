# crypto_quant autoresearch

本实验让 AI 代理在固定数据划分下自主改进 **BTC_USDT** 最近三个月的收益率表现：**样本内 2 个月、样本外 1 个月**。代理仅能修改指定评估脚本，通过反复运行与结果记录，保留能提升样本外收益的改动。

## 目标

- **标的**：BTC_USDT 1 分钟 K 线（不变）。
- **总窗口**：过去 3 个月（约 90 天）数据，来自 `data/csv/BTC_USDT_1m.csv`。
- **样本内**：前 2 个月（约 60 天），用于因子 IC 估计、复合因子方向与 z-score 统计量。
- **样本外**：最后 1 个月（约 30 天），用于评估；**唯一优化目标**为样本外多空策略收益。
- **评价指标**：脚本输出的 `oos_return_avg`（5 期与 60 期样本外多空收益的简单平均）。**目标：最大化 oos_return_avg。**

## Setup

开始新实验前，与用户确认并完成：

1. **约定 run tag**：按日期取名（如 `btc3m_mar9`）。分支 `autoresearch/<tag>` 需不存在，即全新一轮。
2. **创建分支**：`git checkout -b autoresearch/<tag>`（从当前 main/master）。
3. **阅读范围内文件**：理解上下文后再改代码：
   - `README.md`、`docs/DATA_DOWNLOAD.md` — 项目与数据说明。
   - `scripts/factor_performance_report.py` — 因子计算、等权复合与多空回测逻辑（只读参考，不直接改）。
   - `scripts/btc_oos_eval.py` — **你唯一可修改的文件**。内含 2 个月样本内拟合、1 个月样本外评估与打印格式。
4. **确认数据存在**：`data/csv/BTC_USDT_1m.csv` 至少包含约 90 天 1m 数据。若不足，提示用户先执行：
   ```bash
   python scripts/download_okx_klines.py --days 90 --symbols BTC-USDT --interval 1m
   ```
5. **初始化 results.tsv**：在项目根目录创建 `results.tsv`，表头一行 + 一条基线记录。基线来自“第一次运行”得到的 `oos_return_avg`（见下方 Output format）。不要重复跑多次取平均，跑一次记一次即可。
6. **确认并开始**：确认上述无误后，进入实验循环。

## Experimentation

**你可以做的：**

- 只修改 `scripts/btc_oos_eval.py`。可改内容包括但不限于：
  - 因子池（技术/量价/波动率中选用或剔除哪些因子）、
  - 复合方式（等权、加权、筛选高 |IC| 因子）、
  - 预测周期（5/60 期以外的周期或组合）、
  - 分位数档位、非重叠区间规则等。
- 所有改动必须保持：**总数据 3 个月、样本内 2 个月、样本外 1 个月**，且脚本最后按下面 Output format 打印，便于解析。

**你不可以做的：**

- 修改 `scripts/factor_performance_report.py`（仅作参考）。不得修改数据准备脚本的固定逻辑（如 `download_okx_klines.py` 的调用方式）。
- 新增依赖或改 `requirements.txt` / `pyproject.toml`，仅能使用项目已有依赖。
- 修改样本内/样本外划分方式（总 90 天、前 60 天 in-sample、后 30 天 out-of-sample）。

**目标唯一：在样本外 1 个月上取得尽可能高的 oos_return_avg。** 样本内收益仅作参考，不作为保留/丢弃实验的依据。

**简洁性**：在提升相近时，优先保留更简单、更少改动的版本；若删除逻辑后 oos_return_avg 不降，视为改进。

**第一次运行**：必须先按当前脚本不做任何修改跑通一次，将得到的 `oos_return_avg` 作为基线写入 `results.tsv`。

## Output format

脚本运行结束后会打印若干行，其中必须包含（用于解析）：

```
in_sample_return_5: <float>
in_sample_return_60: <float>
oos_return_5: <float>
oos_return_60: <float>
oos_return_avg: <float>
```

从日志中提取主指标可用：

```bash
grep "^oos_return_avg:" run.log
```

若脚本崩溃或未打印上述行，本次实验记为失败（见 Logging results）。

## Logging results

每次实验结束后，将结果追加到项目根目录的 `results.tsv`（制表符分隔，不要用逗号）。

表头与列：

```
commit	oos_return_avg	status	description
```

1. `commit`：当前 git commit 短 hash（7 位）。
2. `oos_return_avg`：本次得到的样本外平均收益（小数形式，如 0.05 表示 5%）。
3. `status`：`keep`（优于或平齐当前最佳）、`discard`（变差）、`crash`（运行失败/无输出）。
4. `description`：简短描述本次尝试（一行内）。

示例：

```
commit	oos_return_avg	status	description
a1b2c3d	0.0234	keep	baseline equal-weight composite
b2c3d4e	0.0312	keep	top 20 factors by |IC| only
c3d4e5f	0.0189	discard	use 120-period forward only
d4e5f6g	0.0000	crash	typo in factor name
```

## The experiment loop

实验在独立分支上进行（如 `autoresearch/btc3m_mar9`）。

循环执行（直到人为停止）：

1. 查看当前 git 状态（分支与 commit）。
2. 在 **不改变 2m 样本内 / 1m 样本外划分与输出格式** 的前提下，修改 `scripts/btc_oos_eval.py`，实现一个新的实验想法。
3. `git add scripts/btc_oos_eval.py` 并 `git commit -m "简短描述"`。
4. 运行评估（建议重定向，避免刷屏）：
   ```bash
   python scripts/btc_oos_eval.py > run.log 2>&1
   ```
   或使用项目 venv：`./.venv/bin/python scripts/btc_oos_eval.py > run.log 2>&1`。
5. 从日志读取结果：`grep "^oos_return_avg:\|^in_sample_return" run.log`。若没有 `oos_return_avg`，视为崩溃，可看 `tail -n 50 run.log` 排查；若多次无法修复，记 status=crash 并跳过。
6. 将结果写入 `results.tsv`。
7. 若本次 **oos_return_avg 优于或等于当前最佳**，保留该 commit（advance 分支）。
8. 若本次 **oos_return_avg 更差**，执行 `git reset --hard HEAD~1` 回到上一 commit，丢弃本次改动。

注意：一旦进入循环，不要中途主动询问用户“是否继续”。用户可能离线或休息，期望代理持续运行直到被手动打断。若一时没有新想法，可重读 `scripts/factor_performance_report.py`、本 program、或尝试组合之前接近成功的思路。

**超时**：单次运行若超过 15 分钟，可终止并记为失败（discard/crash），再 revert。

**崩溃**：若为明显笔误或缺失 import，修掉后重跑；若思路本身不可行，记 status=crash，在 description 中简述原因，然后进行下一轮实验。
