# 50 轮遗传因子等权组合回测报告

## 参数
- 训练轮数: 50
- 每轮最佳因子数: 1
- 样本内: 60 天
- 样本外: 30 天
- 有效因子数: 49
- 手续费: 10 bps/side
- 分位数: 5

## 等权组合 OOS 表现
| 指标 | 5期 | 60期 |
|------|-----|------|
| 累计收益 | 14.11% | 24.33% |
| Sharpe | 7.7941 | 44.0667 |
| 最大回撤 | -17.87% | -21.32% |
| 换手率 | 0.4072 | 0.4423 |

## 每轮最佳因子摘要
| 轮次 | 表达式 | IC_5 | IC_60 | ret_5 | ret_60 | fitness |
|------|--------|------|-------|-------|--------|--------|
| 0 | ts_corr(ts_corr(sign((opensubhigh)),(log(ts_corr(close,((... | -0.0120 | -0.0109 | 0.2590 | 0.2559 | 0.1837 |
| 1 | ts_corr(abs(volume),((ts_ema((ts_ema(ts_ema(ts_sum(ts_ema... | 0.0118 | -0.0058 | 0.2620 | 0.2559 | 0.1839 |
| 2 | ts_mean((log(abs(((ts_min(high,60)addhigh)addneg((ts_delt... | 0.0086 | 0.0294 | 0.1627 | 0.1883 | 0.1285 |
| 3 | ts_max(ts_max((ts_ema(ts_mean(ts_max(ts_max((neg(ts_sum(t... | -0.0078 | -0.0244 | 0.2698 | 0.3207 | 0.2115 |
| 4 | ts_ema(ts_max(ts_max(log(ts_ema(ts_max(log(ts_ema(ts_mean... | -0.0067 | -0.0196 | 0.3369 | 0.3029 | 0.2279 |
| 5 | (ts_sum(ts_min(ts_min(ts_min(ts_ema(ts_sum(ts_min(open,60... | -0.0165 | -0.0295 | 0.3498 | 0.1481 | 0.1812 |
| 6 | ts_ema((ts_corr(sign(ts_corr(ts_mean(high,20),ts_corr(ope... | -0.0085 | -0.0284 | 0.1670 | 0.2002 | 0.1340 |
| 7 | ts_min(ts_corr(high,ts_corr((neg((ts_corr(ts_corr((log(ts... | -0.0167 | -0.0356 | 0.3008 | 0.2917 | 0.2152 |
| 8 | ts_min(abs(ts_min(abs(ts_min((ts_ema(abs(ts_ema(ts_sum(((... | -0.0086 | -0.0287 | 0.2495 | 0.2659 | 0.1860 |
| 9 | ts_max(ts_max(ts_max(ts_max((ts_max(ts_max(ts_ema(ts_min(... | -0.0074 | -0.0229 | 0.3537 | 0.3502 | 0.2509 |
| 10 | ts_ema((ts_mean(ts_max(ts_mean(sign(((highmulopen)mults_e... | -0.0088 | -0.0247 | 0.3188 | 0.2763 | 0.2133 |
| 11 | ((abs((ts_min((ts_ema(abs((openmulhigh)),60)add((ts_delta... | -0.0077 | -0.0260 | 0.2623 | 0.2598 | 0.1878 |
| 12 | ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(t... | -0.0074 | -0.0230 | 0.2909 | 0.3113 | 0.2153 |
| 13 | log(ts_mean(ts_max(neg(ts_min(log(ts_ema(ts_ema(ts_sum(ts... | -0.0081 | -0.0229 | 0.3298 | 0.2991 | 0.2248 |
| 14 | ts_min(ts_min(log(((lowsubabs(ts_max(ts_mean(low,30),60))... | -0.0082 | -0.0276 | 0.2890 | 0.3188 | 0.2181 |
| 15 | ts_min(abs(ts_min(ts_min(abs(ts_ema(ts_min(abs(ts_min(abs... | -0.0068 | -0.0235 | 0.2799 | 0.2771 | 0.1995 |
| 16 | ts_ema((ts_max(neg(ts_max((log(ts_ema((ts_max(neg(sign(ts... | -0.0079 | -0.0273 | 0.1656 | 0.2168 | 0.1391 |
| 17 | neg(ts_min(ts_min(ts_min(ts_min(ts_min(ts_ema(ts_max(ts_e... | 0.0075 | 0.0254 | 0.2910 | 0.2748 | 0.2030 |
| 18 | ts_sum(ts_max(ts_max(ts_sum(ts_max(ts_max(ts_sum(neg(ts_s... | -0.0087 | -0.0285 | 0.2555 | 0.2501 | 0.1826 |
| 19 | ts_sum(log(neg(ts_sum(ts_sum(ts_sum(log(neg(ts_sum(abs(ts... | -0.0089 | -0.0258 | 0.3056 | 0.3048 | 0.2188 |
| 20 | ts_sum(ts_ema(ts_ema(ts_ema(ts_ema(ts_ema(ts_ema(ts_max(t... | -0.0078 | -0.0259 | 0.2758 | 0.3012 | 0.2070 |
| 21 | log(ts_min((ts_min((highaddts_max(abs((abs(log(high))mult... | -0.0090 | -0.0297 | 0.1672 | 0.1912 | 0.1313 |
| 22 | (ts_mean(ts_min(neg(ts_max(ts_mean((highsub((abs((lowdivl... | -0.0079 | -0.0271 | 0.2042 | 0.2083 | 0.1496 |
| 23 | ts_min(ts_min(ts_min(ts_ema(ts_min(ts_min(ts_min(ts_sum(t... | 0.0087 | 0.0260 | 0.2937 | 0.3009 | 0.2133 |
| 24 | abs(ts_max(ts_min(((ts_max(ts_mean(ts_max(ts_min(((ts_max... | -0.0072 | -0.0222 | 0.3337 | 0.3231 | 0.2343 |
| 25 | log(abs(ts_sum(ts_min(ts_sum((abs(ts_sum(ts_min(ts_ema((t... | -0.0085 | -0.0233 | 0.3054 | 0.2835 | 0.2109 |
| 26 | ts_min(ts_min(ts_sum(ts_min(ts_sum(ts_mean(ts_min(ts_sum(... | 0.0091 | 0.0272 | 0.3296 | 0.3399 | 0.2398 |
| 27 | ts_min((ts_min((log(abs((((volumeaddclose)divopen)mulclos... | -0.0089 | -0.0265 | 0.3108 | 0.2784 | 0.2115 |
| 28 | ts_sum(ts_mean((ts_ema((ts_sum(ts_ema(ts_mean((ts_ema((ts... | 0.0090 | 0.0265 | 0.3083 | 0.2815 | 0.2118 |
| 29 | ts_min(ts_ema(ts_min(log((neg(abs(abs(high)))addclose)),6... | 0.0035 | 0.0156 | 0.2137 | 0.3304 | 0.1933 |
| 30 | ts_max(neg(ts_sum(ts_min(ts_mean(ts_sum(ts_min(ts_mean(ts... | 0.0082 | 0.0280 | 0.1812 | 0.2054 | 0.1408 |
| 31 | (ts_min(ts_sum(ts_min(ts_sum((ts_min(ts_sum((ts_min((ts_m... | -0.0083 | -0.0240 | 0.3102 | 0.3222 | 0.2262 |
| 32 | neg(ts_ema((ts_sum(neg(ts_ema((ts_sum(low,60)add(((ts_ema... | -0.0082 | -0.0242 | 0.3285 | 0.3252 | 0.2336 |
| 33 | ts_mean((neg(ts_sum(ts_min(neg(ts_mean((ts_corr(ts_min(ts... | -0.0080 | -0.0262 | 0.2404 | 0.3183 | 0.2007 |
| 34 | ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(ts_max(t... | -0.0075 | -0.0240 | 0.3270 | 0.2995 | 0.2240 |
| 35 | ts_min(ts_sum(ts_min(ts_min(ts_min(ts_sum((ts_min(ts_sum(... | -0.0088 | -0.0287 | 0.3025 | 0.2984 | 0.2159 |
| 36 | ts_min(ts_min((volumeaddts_min((highaddts_min(ts_min((ts_... | -0.0077 | -0.0264 | 0.2843 | 0.2539 | 0.1935 |
| 37 | ts_ema(ts_sum((ts_max(abs(ts_std(low,5)),10)addts_sum((ts... | -0.0085 | -0.0288 | 0.2282 | 0.2497 | 0.1729 |
| 38 | ts_mean(ts_sum(neg((neg((ts_ema(low,20)add(ts_sum(ts_sum(... | 0.0088 | 0.0243 | 0.2670 | 0.2172 | 0.1744 |
| 39 | ((neg((ts_min(ts_min(ts_mean((ts_ema(abs(close),10)add(op... | 0.0081 | 0.0275 | 0.2034 | 0.2149 | 0.1517 |
| 40 | ts_min(ts_delta(ts_corr(ts_sum(sign(ts_min((neg(ts_mean(t... | -0.0056 | 0.0097 | 0.3477 | 0.3336 | 0.2408 |
| 41 | (ts_min(ts_mean(ts_min(low,60),10),60)subts_ema(ts_delta(... | -0.0083 | -0.0283 | 0.2196 | 0.2273 | 0.1619 |
| 42 | ((ts_ema(neg(low),30)div(abs((sign(open)sub(((lowdivopen)... | -0.0086 | -0.0287 | 0.1682 | 0.2142 | 0.1395 |
| 43 | ts_sum(ts_max((((neg((ts_std(log((ts_std((ts_min(volume,3... | 0.0087 | 0.0277 | 0.2315 | 0.1980 | 0.1558 |
| 44 | ts_max(ts_sum((ts_max(neg(((ts_sum(close,20)mults_mean((c... | 0.0090 | 0.0287 | 0.1840 | 0.1766 | 0.1318 |
| 45 | ts_sum(ts_ema((ts_corr((ts_corr(high,low,60)add((ts_max(t... | 0.0088 | 0.0291 | 0.2094 | 0.2210 | 0.1564 |
| 46 | ts_min((ts_corr(ts_corr(volume,volume,30),(openmulopen),3... | 0.0102 | -0.0078 | 0.2610 | 0.2559 | 0.1836 |
| 47 | ts_corr(ts_corr(sign(ts_corr(close,open,60)),ts_corr(ts_c... | -0.0129 | -0.0051 | 0.2600 | 0.2559 | 0.1833 |
| 48 | ts_sum(ts_ema(ts_sum(ts_max((ts_sum(ts_ema(((ts_mean(ts_s... | -0.0089 | -0.0259 | 0.3261 | 0.3186 | 0.2309 |
| 49 | ts_max(ts_max((ts_max(ts_max(((closemults_max((highsub((a... | 0.0078 | 0.0265 | 0.2624 | 0.2690 | 0.1911 |
