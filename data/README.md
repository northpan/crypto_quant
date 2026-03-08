# 数字货币数据获取和预处理模块

一个完整的数字货币数据管理解决方案，支持多交易所数据获取、数据清洗、质量检查和高效存储。

## 功能特性

- **多交易所支持**: Binance、OKX、Bybit等主流交易所
- **多币种支持**: 50+热门币种（BTC、ETH、SOL等）
- **多时间周期**: 1m、5m、15m、1h、4h、1d等
- **市场类型**: 现货、永续合约、期货
- **高效存储**: Parquet格式，支持压缩
- **数据质量**: 自动检查缺失值、异常值、时间间隔
- **技术指标**: SMA、EMA、RSI、MACD、布林带等
- **增量更新**: 智能增量下载，避免重复获取

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础数据获取

```python
from crypto_quant.data import DataManager
from datetime import datetime, timedelta

# 初始化数据管理器
manager = DataManager(
    data_path="./data",
    exchange_name="binance"
)

# 获取BTC 1小时数据
df = manager.get_data(
    symbol="BTC/USDT",
    timeframe="1h",
    start_time=datetime.now() - timedelta(days=7)
)

print(f"获取到 {len(df)} 条数据")
print(df.head())

manager.close()
```

### 2. 批量数据获取

```python
# 批量获取多个币种数据
data = manager.batch_fetch(
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframes=["1m", "5m", "15m"],
    start_time=datetime.now() - timedelta(days=1)
)

# 访问数据
btc_1m = data["BTC/USDT"]["1m"]
eth_5m = data["ETH/USDT"]["5m"]
```

### 3. 数据质量检查

```python
from crypto_quant.data import DataProcessor

processor = DataProcessor()

# 检查数据质量
report = processor.check_quality(df, symbol="BTC/USDT", timeframe="1h")
print(f"质量等级: {report.quality_level.value}")
print(f"缺失率: {report.missing_percentage:.2f}%")
print(f"建议: {report.recommendations}")

# 清洗数据
clean_df = processor.clean_data(df)
```

### 4. 添加技术指标

```python
# 添加技术指标
df_with_indicators = processor.add_technical_indicators(
    df,
    indicators=['sma', 'ema', 'rsi', 'macd', 'bbands']
)

print(df_with_indicators[['close', 'sma_20', 'rsi', 'macd']].tail())
```

### 5. 期货数据获取

```python
from crypto_quant.data import MarketType

# 创建期货数据管理器
futures_manager = DataManager(
    data_path="./futures_data",
    exchange_name="binance",
    market_type=MarketType.PERPETUAL
)

# 获取BTC永续合约数据
df = futures_manager.get_data("BTC/USDT", "1h")
```

### 6. 使用上下文管理器

```python
# 推荐：使用上下文管理器自动关闭连接
with DataManager("./data", "binance") as manager:
    df = manager.get_data("BTC/USDT", "1h")
    # 处理数据...
# 自动关闭连接
```

## 模块结构

```
crypto_quant/data/
├── __init__.py          # 模块导出
├── database.py          # 数据存储接口 (Parquet)
├── data_collector.py    # 数据获取 (CCXT)
├── data_processor.py    # 数据清洗和预处理
├── data_manager.py      # 数据管理器
├── examples.py          # 使用示例
├── requirements.txt     # 依赖列表
└── README.md           # 说明文档
```

## 核心类说明

### DataManager
数据管理器，整合获取、处理和存储功能。

```python
DataManager(
    data_path: str,              # 数据存储路径
    exchange_name: str,          # 交易所名称
    exchange_config: Optional[ExchangeConfig],  # 交易所配置
    market_type: MarketType,     # 市场类型
    update_config: Optional[DataUpdateConfig]   # 更新配置
)
```

主要方法:
- `get_data()`: 获取数据
- `batch_fetch()`: 批量获取
- `align_timeframes()`: 多时间周期对齐
- `get_data_info()`: 获取数据信息

### DataCollector
数据收集器，通过CCXT连接交易所获取数据。

```python
DataCollector(
    exchange_name: str = "binance",
    config: Optional[ExchangeConfig] = None,
    market_type: MarketType = MarketType.SPOT
)
```

主要方法:
- `fetch_ohlcv()`: 获取K线数据
- `fetch_ohlcv_range()`: 获取指定时间范围数据
- `fetch_multiple_symbols()`: 批量获取
- `get_ticker()`: 获取最新行情

### DataProcessor
数据处理器，提供清洗和预处理功能。

```python
processor = DataProcessor()
```

主要方法:
- `check_quality()`: 质量检查
- `clean_data()`: 清洗数据
- `add_technical_indicators()`: 添加技术指标
- `resample()`: 重采样
- `normalize_data()`: 数据标准化
- `calculate_returns()`: 计算收益率

### ParquetStorage
Parquet格式数据存储。

```python
storage = ParquetStorage("./data")
```

主要方法:
- `save_data()`: 保存数据
- `load_data()`: 加载数据
- `append_data()`: 追加数据
- `delete_data()`: 删除数据
- `get_storage_size()`: 获取存储统计

## 支持的交易对

模块内置50+热门币种列表:

**主流币**: BTC、ETH、SOL、BNB、XRP、ADA、AVAX、DOT、MATIC、LINK

**Layer 1**: NEAR、APT、SUI、SEI、TON、INJ、TIA、STRK、OP、ARB

**DeFi**: UNI、AAVE、COMP、MKR、CRV、SNX、YFI、1INCH、SUSHI、LDO

**Meme**: DOGE、SHIB、PEPE、FLOKI、BONK、WIF、BOME

**其他**: RENDER、TAO、FET、AGIX、IMX、GRT、FLOW、EGLD、XTZ、ALGO、VET、FIL、TRX、ETC、BCH、LTC、ATOM、ICP、HBAR

## 数据质量检查

自动检查以下问题:
- 缺失值
- 重复行
- 异常值 (使用IQR或Z-Score方法)
- 时间间隔
- 数据完整性

质量等级:
- **Excellent**: 优秀 (>=95分)
- **Good**: 良好 (>=85分)
- **Fair**: 一般 (>=70分)
- **Poor**: 较差 (>=50分)
- **Critical**: 严重问题 (<50分)

## 技术指标

支持的技术指标:
- **SMA**: 简单移动平均线 (7, 20, 50周期)
- **EMA**: 指数移动平均线 (12, 26周期)
- **RSI**: 相对强弱指数
- **MACD**: 指数平滑异同平均线
- **Bollinger Bands**: 布林带
- **ATR**: 平均真实波幅

## 性能优化

- **数据缓存**: 内存缓存频繁访问的数据
- **增量更新**: 只获取新增数据
- **并行处理**: 支持多线程批量获取
- **数据压缩**: Parquet格式支持多种压缩算法

## 使用示例

运行示例代码:

```bash
python examples.py
```

查看所有功能示例。

## 注意事项

1. **API限制**: 注意交易所的API速率限制
2. **网络稳定**: 确保网络连接稳定
3. **存储空间**: 大量数据需要足够的存储空间
4. **内存使用**: 大数据集注意内存管理

## 许可证

MIT License
