# Autoresearch 100 字分享语

> 可直接复制到组会 PPT 或聊天

---

我们借鉴 Karpathy 的 autoresearch 思路，让 AI 在固定数据划分下自主改进 BTC 量化策略：只改一个评估脚本，每次跑完根据 oos_return_avg 决定 keep 或 discard，共 39 组实验、6 次有效提升。最优路径为：全因子等权 → top20 by |IC| → 3 分位多空 → top18 by 60-period IC。扣费后重跑显示样本外均为负，后续将结合遗传算法挖掘新因子并做稳健性验证。
