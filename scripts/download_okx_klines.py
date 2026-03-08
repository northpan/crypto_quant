#!/usr/bin/env python3
"""
从 OKX 公开 API 下载 K 线数据（无需 API Key），用于回测和训练。

当 Binance 在您所在地区返回 451 时，可改用本脚本从 OKX 拉取数据。
- 使用 GET /api/v5/market/history-candles 公开接口
- 分页用 after 拉取更早数据，每页最多 300 根

输出列名与 DataManager 兼容：timestamp, open, high, low, close, volume。
支持 CSV 或 Parquet。交易对格式：BTC-USDT（OKX 用横杠）。
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OKX_BASE = "https://www.okx.com"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "csv")
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W", "1M"]

# OKX 格式：BTC-USDT
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "MATIC-USDT", "LINK-USDT",
    "UNI-USDT", "LTC-USDT", "BCH-USDT", "DOGE-USDT", "ATOM-USDT",
]


def get_klines(
    symbol: str,
    bar: str = "1m",
    before: int = None,
    after: int = None,
    limit: int = 300,
) -> pd.DataFrame:
    """单次请求 OKX 历史 K 线。bar 如 1m/1H/1D；before/after 为毫秒时间戳。"""
    url = f"{OKX_BASE}/api/v5/market/history-candles"
    params = {"instId": symbol, "bar": bar, "limit": str(min(limit, 300))}
    if before is not None:
        params["before"] = str(before)
    if after is not None:
        params["after"] = str(after)

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return pd.DataFrame()

    if data.get("code") != "0" or not data.get("data"):
        return pd.DataFrame()

    rows = []
    for k in data["data"]:
        ts_ms = int(k[0])
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
    bar: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 300,
    delay: float = 0.15,
) -> pd.DataFrame:
    """分页拉取整段时间的 K 线。首页不传 after；之后用 after=最老 ts 取更早数据。"""
    end_ms = int(end_time.timestamp() * 1000)
    start_ms = int(start_time.timestamp() * 1000)
    all_dfs = []
    after_ts = None  # 第一页不传 after，拿最新数据

    while True:
        df = get_klines(symbol, bar, after=after_ts, limit=limit)
        if df.empty:
            break
        all_dfs.append(df)
        # OKX 返回按时间倒序，第一根是最新的，最后一根是最老的
        oldest_ts = df["timestamp"].iloc[-1]
        if oldest_ts.tzinfo is None:
            oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)
        oldest_ms = int(oldest_ts.timestamp() * 1000)
        if oldest_ms <= start_ms:
            break
        after_ts = oldest_ms
        time.sleep(delay)

    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = out[(out["timestamp"] >= start_time) & (out["timestamp"] <= end_time)]
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
            safe_symbol = symbol.replace("/", "_").replace("-", "_")
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
        time.sleep(0.15)

    return success_count, fail_list


def main():
    parser = argparse.ArgumentParser(description="Download OKX klines (no API key)")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--interval", type=str, default="1m", choices=INTERVALS, help="Kline interval")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"], help="Output format")
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Symbols e.g. BTC-USDT ETH-USDT")
    args = parser.parse_args()

    print(f"Downloading OKX {args.interval} klines, last {args.days} days")
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
