#!/usr/bin/env python3
"""
从 Binance 公共 API 下载指定交易对的 1m K 线数据，保存为 data/csv/{symbol}_1m.csv。

特点：
- 默认下载最近 90 天数据（可通过 --days 修改）
- symbol 使用本项目风格，如 BTC_USDT / ETH_USDT / SOL_USDT
- 输出列与现有 BTC_USDT_1m.csv 一致：timestamp,open,high,low,close,volume
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import time
import urllib.parse
import urllib.request
import json

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


BASE_URL = "https://api.binance.com/api/v3/klines"


def _rest_get(url: str, params: dict) -> List:
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(full, timeout=10) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1.0 + attempt)
    return []


def _to_binance_symbol(symbol: str) -> str:
    # 简单转换：BTC_USDT -> BTCUSDT
    return symbol.replace("_", "")


def download_ohlcv(symbol: str, days: int, timeframe: str = "1m") -> Path:
    binance_symbol = _to_binance_symbol(symbol)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows = []
    cur = start_ms
    print(f"Downloading {symbol} {timeframe} from Binance, ~{days} days...")

    while cur < end_ms:
        data = _rest_get(
            BASE_URL,
            {
                "symbol": binance_symbol,
                "interval": timeframe,
                "limit": 1000,
                "startTime": cur,
            },
        )
        if not data:
            break
        for k in data:
            open_time = int(k[0])
            if open_time < start_ms:
                continue
            if open_time > end_ms:
                break
            open_price = float(k[1])
            high = float(k[2])
            low = float(k[3])
            close = float(k[4])
            volume = float(k[5])
            ts = datetime.fromtimestamp(open_time / 1000.0, tz=timezone.utc)
            ts_str = ts.isoformat(sep=" ", timespec="minutes")
            # 与现有 BTC_USDT_1m.csv 格式对齐，包含 +00:00 时区后缀
            all_rows.append(
                [
                    ts_str,
                    f"{open_price}",
                    f"{high}",
                    f"{low}",
                    f"{close}",
                    f"{volume}",
                ]
            )
        # Binance 返回的数据已按时间排序，取最后一条的 open_time 继续
        last_open = int(data[-1][0])
        next_open = last_open + 60_000  # 下一分钟
        if next_open <= cur:
            next_open = cur + 60_000
        cur = next_open
        # 控制请求频率，避免过快
        time.sleep(0.2)

        # 保护：最多拉取约 200k 根，防止无限循环
        if len(all_rows) > 200_000:
            break

    if not all_rows:
        raise SystemExit(f"没有从 Binance 获取到 {symbol} 的 {timeframe} 数据，请检查交易对是否存在。")

    out_dir = PROJECT_ROOT / "data" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{timeframe}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in all_rows:
            w.writerow(row)

    print(f"Saved {len(all_rows)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download 1m OHLCV from Binance to data/csv")
    parser.add_argument("--symbol", type=str, default="BTC_USDT", help="Symbol like BTC_USDT / ETH_USDT")
    parser.add_argument("--days", type=int, default=90, help="Number of days to download")
    args = parser.parse_args()

    try:
        download_ohlcv(args.symbol, args.days, timeframe="1m")
    except Exception as e:
        print(f"Error downloading {args.symbol}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

