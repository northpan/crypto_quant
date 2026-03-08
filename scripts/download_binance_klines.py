#!/usr/bin/env python3
"""
从 Binance 公开 API 下载 K 线数据（无需 API Key），用于回测和训练。

参考 agent_hedge_fund-main/download_crypto.py 的思路：
- 使用 GET /api/v3/klines 公开接口
- 分页拉取 startTime/endTime，每页最多 1000 根

输出目录默认与 config 一致，列名与 DataManager 兼容：timestamp, open, high, low, close, volume。
支持 CSV 或 Parquet。
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BINANCE_BASE = "https://api.binance.com"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "csv")
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

# 与 config/config.yaml 中 symbols 对齐的 Top 50（Binance 格式无斜杠）
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    "UNIUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT",
    "TRXUSDT", "EOSUSDT", "FILUSDT", "AAVEUSDT", "SUSHIUSDT",
    "COMPUSDT", "MKRUSDT", "YFIUSDT", "SNXUSDT", "CRVUSDT",
    "1INCHUSDT", "GRTUSDT", "MANAUSDT", "SANDUSDT", "AXSUSDT",
    "HTUSDT", "OKBUSDT", "LEOUSDT", "CROUSDT", "VETUSDT",
    "ICPUSDT", "THETAUSDT", "XTZUSDT", "ALGOUSDT", "ATOMUSDT",
    "NEARUSDT", "FTMUSDT", "GALAUSDT", "APEUSDT", "FLOWUSDT",
    "EGLDUSDT", "HBARUSDT", "QNTUSDT", "CHZUSDT",
]


def get_klines(
    symbol: str,
    interval: str = "1m",
    start_time: datetime = None,
    end_time: datetime = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """单次请求 Binance K 线。"""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time is not None:
        params["endTime"] = int(end_time.timestamp() * 1000)

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for k in data:
        ts_ms = k[0]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rows.append({
            "timestamp": dt,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(rows)


def fetch_klines_range(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 1000,
    delay: float = 0.1,
) -> pd.DataFrame:
    """分页拉取整段时间的 K 线（与 agent_hedge_fund 思路一致）。"""
    all_dfs = []
    current_start = start_time

    while current_start < end_time:
        df = get_klines(symbol, interval, start_time=current_start, end_time=end_time, limit=limit)
        if df.empty:
            break
        all_dfs.append(df)
        last_ts = df["timestamp"].iloc[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        # 下一段从最后一根 K 线的下一根开始（按 interval 步进需根据 interval 算，这里用最后一根 + 1ms 简单处理）
        next_ms = int(last_ts.timestamp() * 1000) + 1
        current_start = datetime.fromtimestamp(next_ms / 1000, tz=timezone.utc)
        if len(df) < limit:
            break
        time.sleep(delay)

    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def download_all(
    symbols: list = None,
    days: int = 30,
    interval: str = "1m",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    format: str = "csv",
) -> tuple:
    """下载所有 symbol 的 K 线。返回 (成功数, 失败列表)。"""
    symbols = symbols or SYMBOLS
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    fail_list = []

    for idx, symbol in enumerate(symbols):
        print(f"[{idx + 1}/{len(symbols)}] {symbol} {interval}...", end=" ", flush=True)
        df = fetch_klines_range(symbol, interval, start_time, end_time)
        if df is not None and len(df) > 0:
            # 文件名：与 DataManager 存储风格接近，例如 BTCUSDT_1m.csv
            safe_symbol = symbol.replace("/", "_")
            base = os.path.join(output_dir, f"{safe_symbol}_{interval}")
            if format == "parquet":
                path = base + ".parquet"
                df.to_parquet(path, index=False)
            else:
                path = base + ".csv"
                df.to_csv(path, index=False)
            print(f"OK - {len(df)} bars -> {path}")
            success_count += 1
        else:
            print("SKIP - no data")
            fail_list.append(symbol)
        time.sleep(0.1)

    return success_count, fail_list


def main():
    parser = argparse.ArgumentParser(description="Download Binance klines (no API key)")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--interval", type=str, default="1m", choices=INTERVALS, help="Kline interval")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"], help="Output format")
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Symbols (default: top 50)")
    args = parser.parse_args()

    print(f"Downloading {args.interval} klines, last {args.days} days")
    print(f"Output: {args.output_dir} ({args.format})\n")

    success_count, fail_list = download_all(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        output_dir=args.output_dir,
        format=args.format,
    )

    total = len(args.symbols or SYMBOLS)
    print("\n" + "=" * 50)
    print(f"Done: {success_count}/{total} succeeded")
    if fail_list:
        print(f"Failed: {', '.join(fail_list)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
