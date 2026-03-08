# 真实数据下载与测试指南

本文档说明在 **crypto_quant** 中获取真实行情数据用于回测和训练的多种方式，并参考 `agent_hedge_fund-main` 的做法。

---

## 一、crypto_quant 自带方式

### 1. 使用 DataManager（CCXT，需网络）

通过 CCXT 连接交易所，**无需 API Key** 即可拉取公开 K 线（仅读）。

```python
from data import DataManager
from datetime import datetime, timedelta

manager = DataManager(
    data_path="./data/storage",
    exchange_name="binance"  # 或 okx, bybit
)

# 按时间范围获取
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=30)
df = manager.get_data(
    symbol="BTC/USDT",
    timeframe="1m",   # 1m, 5m, 15m, 1h, 4h, 1d
    start_time=start_time,
    end_time=end_time
)

# 数据会自动写入 data_path（Parquet），下次可走缓存
print(df.shape)
```

- **优点**：与现有回测/训练流程一致，支持多交易所、多周期、增量更新、Parquet 存储。
- **缺点**：需要能访问交易所 API（如 api.binance.com），无网环境会失败（此时会退化为模拟数据）。

### 2. 批量下载多币种 / 多周期

```python
# 批量获取
data = manager.batch_fetch(
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframes=["1m", "5m", "15m"],
    start_time=start_time,
    end_time=end_time
)

# 或并行加速
data = manager.parallel_batch_fetch(
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframes=["1m", "5m"],
    start_time=start_time,
    end_time=end_time,
    max_workers=4
)
```

### 3. 直接使用 DataCollector（不写存储）

若只想拉取不落盘：

```python
from data import DataCollector
from datetime import datetime, timedelta

collector = DataCollector(exchange_name="binance")
df = collector.fetch_ohlcv_range(
    symbol="BTC/USDT",
    timeframe="1m",
    start_time=datetime.utcnow() - timedelta(days=7),
    end_time=datetime.utcnow()
)
# 注意：需能连交易所，否则 _init_exchange 会抛错
```

---

## 二、参考 agent_hedge_fund-main 的做法

### 1. Binance 公开 REST API + httpx（无需 API Key）

`agent_hedge_fund-main/download_crypto.py` 的做法：

- 使用 **Binance 公开接口** `GET /api/v3/klines`，不需要 API Key。
- 用 `httpx` 请求，按 `startTime`/`endTime` 分页（每页最多 1000 根）。
- 输出 CSV 到本地目录（如 `~/.qlib/crypto_data/csv/`），列：date, open, close, high, low, volume, money。

**特点**：实现简单、无密钥、适合脚本/CI；仅支持 Binance 现货 K 线。

### 2. 当 Binance 返回 451 时的替代：`scripts/download_okx_klines.py`

若您所在地区访问 Binance 被限制（HTTP 451），可使用 **OKX 公开 API** 下载 K 线，无需 API Key：

```bash
# 下载最近 7 天 1 分钟 K 线（OKX 交易对格式：BTC-USDT）
python3 scripts/download_okx_klines.py --days 7 --interval 1m --symbols BTC-USDT ETH-USDT SOL-USDT
```

- 输出目录与 Binance 脚本一致（默认 `./data/csv`），列名兼容：`timestamp, open, high, low, close, volume`。
- 支持 `--format csv|parquet`、`--output_dir`。交易对为 OKX 格式（如 `BTC-USDT`）。

**回测推荐：五品种、过去约 30 天数据**（与 main 默认回测区间一致）：

```bash
# 五品种、30 天 1m K 线，用于回测与参数优化
python3 scripts/download_okx_klines.py --days 30 --interval 1m --symbols BTC-USDT ETH-USDT SOL-USDT BNB-USDT XRP-USDT
```

- 生成 `data/csv` 下 `BTC_USDT_1m.csv`、`ETH_USDT_1m.csv`、`SOL_USDT_1m.csv`、`BNB_USDT_1m.csv`、`XRP_USDT_1m.csv`。
- 回测默认使用上述五品种、最近 30 天区间；优化目标为总收益率（total_return）。

### 3. 本项目的等价脚本：`scripts/download_binance_klines.py`

在 **crypto_quant** 下已提供脚本 `scripts/download_binance_klines.py`（见下一节），实现思路与 agent_hedge_fund 一致，但：

- 输出目录改为 `config/data_dir`（或 `./data/csv`），并可选择保存为 **CSV** 或 **Parquet**。
- 列名与 `DataManager` / 回测引擎一致：`timestamp, open, high, low, close, volume`，便于直接用于训练和回测。

### 4. agent_hedge_fund 中的 Qlib 加密货币采集器

路径：`qlib/scripts/data_collector/crypto/collector.py`  

- 数据源：**CoinGecko**（`pycoingecko`），日频（1d）。
- 用途：更多是价格/市值序列，**不是标准 OHLCV K 线**，且 Coingecko 无 1m/5m，不适合做分钟级回测。
- 若你需要日频、且能接受非交易所 K 线，可参考其 `download_data` / `normalize_data` 流程；crypto_quant 做分钟级更推荐用 Binance/CCXT。

---

## 三、推荐：用脚本先落盘再训练/回测

在有网络的环境一次性下载，无网或内网环境用已下载数据：

1. **有网络时**：运行  
   `python scripts/download_binance_klines.py --days 30 --interval 1m`  
   将 Binance 公开 K 线下载到 `./data/csv`（或 Parquet）。
2. **训练/回测时**：让 `DataManager` 使用同一 `data_path`；若本地已有 Parquet/CSV，会优先读本地，避免再请求交易所（具体逻辑见 `data_manager.fetch_and_store` 的缓存与存储）。

这样即可用**真实数据**做测试和训练，又能在无网环境复现。

---

## 四、数据源对比小结

| 方式 | 数据源 | 是否需要 API Key | 周期 | 适用场景 |
|------|--------|------------------|------|----------|
| DataManager + CCXT | Binance/OKX/Bybit 等 | 否（公开 K 线） | 1m/5m/15m/1h/4h/1d | 项目主流程、多交易所、Parquet 缓存 |
| scripts/download_binance_klines.py | Binance 公开 API | 否 | 1m/5m/15m/1h/1d | 一次性批量下载、CI、无密钥 |
| agent_hedge_fund download_crypto.py | Binance 公开 API | 否 | 15m（可改） | 为 Qlib 准备 CSV |
| qlib crypto collector | CoinGecko | 否 | 1d | 日频、非标准 K 线 |

---

## 五、使用已下载数据做回测/训练

- **数据在 `./data/storage` 且为 Parquet**：直接使用 `DataManager(data_path="./data/storage", exchange_name="binance")`，`get_data()` 会先读本地再决定是否请求交易所。
- **数据由 `scripts/download_binance_klines.py` 下载到 `./data/csv`**：
  - 方式 A：训练/回测前用 pandas 读取 CSV，再通过 `DataManager` 的存储接口或回测引擎的 `load_data(symbol, df)` 喂入。
  - 方式 B：将脚本的 `--output_dir` 设为 `./data/storage`，并让脚本按 `data/database.py` 的目录结构写入（即 `{data_path}/spot/{symbol}_{timeframe}.parquet`，symbol 如 `btc_usdt`），则 DataManager 可直接读。

以上方法组合即可在 **有网/无网** 下都用真实数据做测试和训练。

---

## 六、快速命令示例

```bash
# 下载最近 30 天 1 分钟 K 线到默认目录（无需 API Key）
python scripts/download_binance_klines.py --days 30 --interval 1m

# 输出为 Parquet，并指定目录
python scripts/download_binance_klines.py --days 14 --interval 5m --format parquet --output_dir ./data/storage

# 只下载部分交易对
python scripts/download_binance_klines.py --days 7 --symbols BTCUSDT ETHUSDT SOLUSDT
```
