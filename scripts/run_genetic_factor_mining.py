#!/usr/bin/env python3
"""
遗传因子挖掘入口：加载 OHLCV，运行遗传算法，输出 IC/ret 及 ret 正收益因子。
"""

import sys
import csv
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_ohlcv(symbol: str = "BTC_USDT", timeframe: str = "1m", days: int = 90) -> "pd.DataFrame":
    """从 data/csv 加载 OHLCV。"""
    import pandas as pd
    csv_path = PROJECT_ROOT / "data" / "csv" / f"{symbol}_{timeframe}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    end = df["timestamp"].max()
    start = end - timedelta(days=days)
    if df["timestamp"].dt.tz is not None:
        start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    import argparse
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Genetic factor mining: IC + ret, find positive ret factors")
    parser.add_argument("--symbol", type=str, default="BTC_USDT", help="Symbol like BTC_USDT / ETH_USDT")
    parser.add_argument("--days", type=int, default=90, help="Days of 1m OHLCV to use")
    parser.add_argument("--population", type=int, default=40, help="Population size")
    parser.add_argument("--generations", type=int, default=25, help="Generations")
    parser.add_argument("--max_depth", type=int, default=4, help="Max expression tree depth")
    parser.add_argument("--out", type=str, default="reports/genetic_factor_mining.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.15, help="Weight for robust IC in fitness")
    parser.add_argument("--delta", type=float, default=0.05, help="Weight for robust ret in fitness")
    args = parser.parse_args()

    import pandas as pd
    from factors.genetic.evaluator import evaluate_expression
    from factors.genetic.genetic_factor import run_evolution
    from factors.genetic.expression import expr_to_str

    data = load_ohlcv(args.symbol, "1m", days=args.days)
    if data.empty:
        print(f"未找到 data/csv/{args.symbol}_1m.csv，请先下载或准备该交易对的 K 线数据")
        return

    print(f"{args.symbol} 数据: {len(data)} 行, {args.days} 天")
    n_quantiles = 5
    fee_bps = 10

    def evaluate_fn(expr):
        return evaluate_expression(
            expr,
            data,
            forward_periods=[5, 60],
            n_quantiles=n_quantiles,
            fee_bps=fee_bps,
        )

    rng = __import__("random").Random(args.seed)
    results = run_evolution(
        evaluate_fn,
        population_size=args.population,
        generations=args.generations,
        max_depth=args.max_depth,
        elite_ratio=0.15,
        mutation_prob=0.55,
        crossover_prob=0.5,
        immigrant_ratio=0.2,
        alpha=0.3,
        beta=0.7,
        ret_weight_5=0.5,
        ret_weight_60=0.5,
        gamma=args.gamma,
        delta=args.delta,
        rng=rng,
    )

    # 输出全部结果（按 fitness 降序）
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    positive_ret = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["expr_str", "ic_5", "ic_60", "ret_5", "ret_60", "fitness"])
        for expr, metrics, fitness in results:
            s = expr_to_str(expr)
            ic_5 = metrics.get("ic_5", float("nan"))
            ic_60 = metrics.get("ic_60", float("nan"))
            ret_5 = metrics.get("ret_5", float("nan"))
            ret_60 = metrics.get("ret_60", float("nan"))
            w.writerow([s, ic_5, ic_60, ret_5, ret_60, fitness])
            r5 = ret_5 if isinstance(ret_5, (int, float)) else float("nan")
            r60 = ret_60 if isinstance(ret_60, (int, float)) else float("nan")
            if (not pd.isna(r5) and r5 > 0) or (not pd.isna(r60) and r60 > 0):
                positive_ret.append((s, ic_5, ic_60, ret_5, ret_60, fitness))

    # 同步保存表达式对象列表，顺序与 CSV 完全一致，便于后续按行号选取
    expr_pkl_path = out_path.with_suffix(".pkl")
    with open(expr_pkl_path, "wb") as pf:
        pickle.dump([expr for expr, _m, _f in results], pf)

    print(f"结果已写入 {out_path}")
    print(f"表达式对象已写入 {expr_pkl_path}")
    print("\n--- 适应度 Top 10 ---")
    for i, (expr, metrics, fitness) in enumerate(results[:10], 1):
        print(f"{i}. {expr_to_str(expr)}  ic_5={metrics.get('ic_5', 0):.4f} ic_60={metrics.get('ic_60', 0):.4f}  ret_5={metrics.get('ret_5', 0):.4f}  ret_60={metrics.get('ret_60', 0):.4f}  fit={fitness:.4f}")

    if positive_ret:
        print("\n--- ret 正收益因子（ret_5 或 ret_60 > 0）---")
        for s, ic_5, ic_60, ret_5, ret_60, fit in positive_ret[:20]:
            print(f"  {s}  ic_5={ic_5:.4f} ic_60={ic_60:.4f} ret_5={ret_5:.4f} ret_60={ret_60:.4f} fitness={fit:.4f}")
    else:
        print("\n本轮未发现 ret_5 或 ret_60 为正的因子，可增加代数或种群再跑。")


if __name__ == "__main__":
    main()
