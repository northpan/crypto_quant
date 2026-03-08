# 数字货币量化因子库

Cryptocurrency Quantitative Factor Library

## 概述

本因子库提供 **60+** 个数字货币量化因子，涵盖技术指标、量价关系、波动率、订单流和跨市场五大类别。支持分钟级计算，适用于高频交易策略开发。

## 因子分类

### 1. 技术指标因子 (Technical Indicators) - 24个

#### 趋势类因子
| 因子名称 | 描述 | 方向 |
|---------|------|------|
| MACD | MACD趋势动量因子 | 正向 |
| MACD_Signal | MACD交叉信号因子 | 正向 |
| EMA_Cross | EMA交叉因子 | 正向 |
| SMA_Cross | SMA交叉因子 | 正向 |
| ADX | ADX趋势强度因子 | 中性 |
| Trend_Strength | 多周期趋势强度因子 | 正向 |

#### 动量类因子
| 因子名称 | 描述 | 方向 |
|---------|------|------|
| RSI | RSI动量因子 | 反向 |
| RSI_Divergence | RSI背离因子 | 正向 |
| CCI | CCI商品通道指数因子 | 反向 |
| WilliamsR | Williams %R因子 | 反向 |
| Stochastic | 随机指标因子 | 反向 |
| Stochastic_Cross | 随机指标交叉因子 | 正向 |
| Momentum | 价格动量因子 | 正向 |
| ROC | 变化率因子 | 正向 |
| Price_Acceleration | 价格加速度因子 | 正向 |

#### 波动类因子
| 因子名称 | 描述 | 方向 |
|---------|------|------|
| BB_Position | 布林带位置因子 | 反向 |
| BB_Width | 布林带宽度因子 | 中性 |
| BB_Squeeze | 布林带挤压因子 | 正向 |
| ATR | ATR波动率因子 | 中性 |
| ATR_Ratio | ATR比率因子 | 中性 |
| Keltner_Position | Keltner通道位置因子 | 反向 |
| Volatility_Regime | 波动率状态因子 | 中性 |
| Donchian_Channel | Donchian通道因子 | 正向 |
| Ichimoku | 一目均衡表因子 | 正向 |
| Parabolic_SAR | 抛物线SAR因子 | 正向 |

### 2. 量价因子 (Volume-Price) - 20个

| 因子名称 | 描述 | 方向 |
|---------|------|------|
| OBV | OBV能量潮因子 | 正向 |
| OBV_Velocity | OBV速度因子 | 正向 |
| OBV_Divergence | OBV背离因子 | 正向 |
| MFI | MFI资金流量指数因子 | 反向 |
| MFI_Velocity | MFI速度因子 | 正向 |
| VWAP | VWAP成交量加权平均价因子 | 反向 |
| VWAP_Deviation | VWAP偏离度因子 | 反向 |
| VWAP_Trend | VWAP趋势因子 | 正向 |
| Volume_Ratio | 成交量比率因子 | 中性 |
| Volume_Trend | 成交量趋势因子 | 中性 |
| Volume_Oscillator | 成交量振荡器因子 | 中性 |
| VP_Divergence | 量价背离因子 | 正向 |
| VPT | 量价趋势因子 | 正向 |
| VP_Confirmation | 量价确认因子 | 正向 |
| Money_Flow | 资金流向因子 | 正向 |
| Buying_Pressure | 买盘压力因子 | 正向 |
| AD_Line | 累积/派发因子 | 正向 |
| Chaikin_Oscillator | Chaikin振荡器因子 | 正向 |
| Force_Index | 力量指数因子 | 正向 |
| EOM | 简易波动指标因子 | 正向 |
| NVI | 负成交量指数因子 | 正向 |
| PVI | 正成交量指数因子 | 正向 |

### 3. 波动率因子 (Volatility) - 19个

| 因子名称 | 描述 | 方向 |
|---------|------|------|
| HV | 历史波动率因子 | 中性 |
| RV | 实现波动率因子 | 中性 |
| CCV | 收盘-收盘波动率因子 | 中性 |
| Parkinson | Parkinson波动率因子 | 中性 |
| GK | Garman-Klass波动率因子 | 中性 |
| RS | Rogers-Satchell波动率因子 | 中性 |
| YZ | Yang-Zhang波动率因子 | 中性 |
| GARCH | GARCH(1,1)波动率因子 | 中性 |
| EWMA | EWMA波动率因子 | 中性 |
| Vol_Cone | 波动率锥因子 | 中性 |
| Vol_Percentile | 波动率百分位因子 | 中性 |
| Vol_Regime | 波动率状态因子 | 中性 |
| Vol_Trend | 波动率趋势因子 | 中性 |
| Vol_Skewness | 波动率偏度因子 | 正向 |
| Vol_Kurtosis | 波动率峰度因子 | 反向 |
| Jump_Vol | 跳跃波动率因子 | 中性 |
| Intraday_Range | 日内波幅因子 | 中性 |
| Overnight_Gap | 隔夜跳空因子 | 中性 |

### 4. 订单流因子 (Order Flow) - 18个

| 因子名称 | 描述 | 方向 |
|---------|------|------|
| BuySell_Pressure | 买卖压力因子 | 正向 |
| Trade_Intensity | 交易强度因子 | 中性 |
| Tick_Rule | Tick规则因子 | 正向 |
| Lee_Ready | Lee-Ready因子 | 正向 |
| OB_Imbalance | 订单簿不平衡因子 | 正向 |
| OB_Slope | 订单簿斜率因子 | 正向 |
| OB_Pressure | 订单簿压力因子 | 正向 |
| Spread | 买卖价差因子 | 反向 |
| Depth_Imbalance | 深度不平衡因子 | 正向 |
| Funding_Rate | 资金费率因子 | 反向 |
| Funding_Momentum | 资金费率动量因子 | 正向 |
| Funding_Extreme | 资金费率极值因子 | 正向 |
| Open_Interest | 持仓量因子 | 正向 |
| OI_Price | 持仓量-价格关系因子 | 正向 |
| OI_Velocity | 持仓量速度因子 | 正向 |
| Large_Trade | 大单交易因子 | 正向 |
| VW_Trade | 成交量加权交易因子 | 正向 |

### 5. 跨市场因子 (Cross-Market) - 17个

| 因子名称 | 描述 | 方向 |
|---------|------|------|
| Basis | 基差因子 | 反向 |
| Basis_Momentum | 基差动量因子 | 正向 |
| Basis_ZScore | 基差Z-Score因子 | 反向 |
| Contango_Backwardation | 升贴水因子 | 反向 |
| Calendar_Spread | 跨期价差因子 | 正向 |
| Term_Structure | 期限结构因子 | 正向 |
| Exchange_Spread | 跨交易所价差因子 | 中性 |
| Exchange_Vol_Imbalance | 跨交易所成交量不平衡因子 | 正向 |
| Arbitrage_Opportunity | 套利机会因子 | 正向 |
| Pair_Spread | 币对价差因子 | 中性 |
| Beta | Beta因子 | 正向 |
| Correlation | 相关性因子 | 中性 |
| Price_Impact | 价格冲击因子 | 中性 |
| Liquidity | 流动性因子 | 正向 |
| Market_Depth | 市场深度因子 | 正向 |
| Cross_Market_Momentum | 跨市场动量因子 | 正向 |
| Lead_Lag | 领先滞后因子 | 正向 |

## 快速开始

### 安装依赖

```bash
pip install numpy pandas scipy matplotlib
```

### 基础用法

```python
import pandas as pd
from factors import FactorPool, RSIFactor

# 加载数据
data = pd.read_csv('btc_data.csv', index_col='timestamp', parse_dates=True)

# 创建因子池
pool = FactorPool()

# 计算单个因子
rsi_factor = RSIFactor(window=14)
rsi_values = rsi_factor.compute(data)

# 批量计算所有因子
factor_values = pool.compute(data, verbose=True)

# 因子有效性检验
forward_return = data['close'].pct_change(5).shift(-5)
test_results = pool.test(factor_values, forward_return, verbose=True)

# 获取表现最好的因子
top_factors = pool.rank_factors(test_results).head(10)
print(top_factors)
```

### 快速分析

```python
from factors import quick_factor_analysis

# 一键完成因子计算和检验
results = quick_factor_analysis(data, forward_return)

print(f"总因子数: {results['total_factors']}")
print(f"有效因子数: {results['significant_factors']}")
print(f"筛选出的因子: {results['selected_factors']}")
```

## 因子检验方法

### 1. IC分析 (Information Coefficient)
- 计算因子值与未来收益的相关系数
- IC > 0.03 且 p-value < 0.05 认为因子有效

### 2. 分位数收益分析
- 将因子分成5组，计算每组平均收益
- 观察单调性和多空收益

### 3. 换手率分析
- 评估因子稳定性
- 换手率过高会增加交易成本

### 4. 综合评分
- IC得分（40%）
- Sharpe比率（40%）
- 换手率（20%）

## 数据格式要求

### 基础OHLCV数据
```python
data = pd.DataFrame({
    'open': [...],    # 开盘价
    'high': [...],    # 最高价
    'low': [...],     # 最低价
    'close': [...],   # 收盘价
    'volume': [...]   # 成交量
}, index=pd.DatetimeIndex([...]))
```

### 扩展数据（可选）
- `open_interest`: 持仓量
- `funding_rate`: 资金费率
- `bid_price_1`, `ask_price_1`: 订单簿数据
- `spot_price`, `futures_price`: 期现价格

## 性能优化

### 并行计算
```python
# 使用多线程加速
factor_values = pool.compute(data, n_jobs=4, verbose=True)
```

### 增量计算
```python
# 只计算新数据
new_factor_values = pool.compute(new_data, factor_names=['RSI_14', 'MACD_12_26_9'])
```

## 自定义因子

```python
from factors import TechnicalFactor, FactorDirection

class MyCustomFactor(TechnicalFactor):
    def __init__(self, window=20):
        super().__init__(
            name=f"MyFactor_{window}",
            description="自定义因子",
            direction=FactorDirection.POSITIVE
        )
        self.window = window
    
    def calculate(self, data, **kwargs):
        close = data['close']
        # 自定义计算逻辑
        return close.pct_change(self.window)

# 注册并使用
pool = FactorPool()
pool.register(MyCustomFactor(window=20))
factor_values = pool.compute(data)
```

## 项目结构

```
factors/
├── __init__.py              # 包入口
├── base_factor.py           # 因子基类和工具函数
├── technical_factors.py     # 技术指标因子 (24个)
├── volume_factors.py        # 量价因子 (20个)
├── volatility_factors.py    # 波动率因子 (19个)
├── orderflow_factors.py     # 订单流因子 (18个)
├── cross_market_factors.py  # 跨市场因子 (17个)
├── factor_pool.py           # 因子池管理
├── example_usage.py         # 使用示例
└── README.md                # 本文档
```

## 因子统计

```python
from factors import print_factor_stats
print_factor_stats()
```

输出:
```
==================================================
数字货币量化因子库统计
==================================================
技术指标因子: 24
量价因子: 20
波动率因子: 19
订单流因子: 18
跨市场因子: 17
--------------------------------------------------
总计: 98 个因子
==================================================
```

## 注意事项

1. **数据质量**: 确保数据无缺失值和异常值
2. **参数调优**: 不同市场和周期可能需要调整参数
3. **过拟合风险**: 避免在过多因子上过度优化
4. **交易成本**: 考虑换手率对实际收益的影响

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。
