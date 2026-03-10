#!/usr/bin/env python3
"""
当前最佳实验（top18 by 60-period IC, n_quantiles=3, linear）扣除手续费后的样本外 PnL 曲线。
与 btc_oos_eval.py 当前配置一致，FEE_BPS=10。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from factors import FactorPool
from factors.base_factor import FactorCategory

import btc_oos_eval

load_ohlcv = btc_oos_eval.load_ohlcv
build_composite_with_insample_stats = btc_oos_eval.build_composite_with_insample_stats
FEE_BPS = btc_oos_eval.FEE_BPS
TOP_N_FACTORS = btc_oos_eval.TOP_N_FACTORS
N_QUANTILES = btc_oos_eval.N_QUANTILES


def get_pnl_curve_with_fee(
    factor_series,
    forward_return,
    forward_period,
    n_quantiles,
    fee_bps,
    timestamps=None,
):
    """Non-overlapping long-short equity curve (starts at 1), with fee at each rebalance. Returns (x, y)."""
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return np.array([]), np.array([])
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return np.array([]), np.array([])
    position = (q == q_max).astype(float) - (q == q_min).astype(float)
    indices = common[::forward_period]
    rets, index_used = [], []
    prev_pos = None
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos, fr = position.loc[i], r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        raw_ret = pos * fr
        if prev_pos is not None and pos != prev_pos:
            raw_ret -= 2 * (fee_bps / 1e4)
        prev_pos = pos
        rets.append(raw_ret)
        index_used.append(i)
    if not rets:
        return np.array([]), np.array([])
    curve = [1.0]
    cum = 1.0
    for r_t in rets:
        cum *= 1.0 + r_t
        curve.append(cum)
    y = np.array(curve)
    if timestamps is not None and len(index_used) > 0:
        try:
            first_idx = index_used[0] - forward_period
            if first_idx not in timestamps.index or first_idx < 0:
                first_idx = index_used[0]
            rest_ts = timestamps.reindex(index_used).dropna()
            if len(rest_ts) == len(index_used):
                first_ts = pd.to_datetime(timestamps.loc[first_idx])
                x = np.array([first_ts] + pd.to_datetime(rest_ts).tolist())
                if len(x) == len(y):
                    return x, y
        except Exception:
            pass
    return np.arange(len(y)), y


def main():
    total_days, in_days = 90, 60
    data = load_ohlcv("BTC_USDT", "1m", total_days)
    if len(data) < (in_days * 24 * 60) * 0.9:
        print("Not enough data")
        return
    n = len(data)
    n_in = int(n * in_days / total_days)
    in_df = data.iloc[:n_in].copy()
    oos_df = data.iloc[n_in:].copy()

    pool = FactorPool()
    categories = [FactorCategory.TECHNICAL, FactorCategory.VOLUME, FactorCategory.VOLATILITY]
    factor_names = []
    for cat in categories:
        factor_names.extend(pool.list_factors(cat))
    if not factor_names:
        factor_names = list(pool.factors.keys())[:40]

    factors_in = pool.compute(in_df, factor_names=factor_names, verbose=False)
    factors_in = factors_in.dropna(axis=1, how="all")
    valid_all = list(factors_in.columns)
    if len(valid_all) < 5:
        print("Too few factors")
        return

    forward_5_in = in_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_in = in_df["close"].astype(float).pct_change(60).shift(-60)
    test_5 = pool.test(factors_in[valid_all], forward_5_in, n_quantiles=N_QUANTILES, verbose=False)
    test_60 = pool.test(factors_in[valid_all], forward_60_in, n_quantiles=N_QUANTILES, verbose=False)
    ic_5 = test_5.set_index("name")["ic"]
    ic_60 = test_60.set_index("name")["ic"]

    # 当前最佳：top 18 by 60-period IC
    abs_ic = ic_60.abs().reindex(valid_all).fillna(0)
    top_factors = abs_ic.nlargest(TOP_N_FACTORS).index.tolist()
    valid = [c for c in valid_all if c in top_factors]
    in_sample_mean = factors_in[valid_all].mean().to_dict()
    in_sample_std = factors_in[valid_all].std().to_dict()

    factors_oos = pool.compute(oos_df, factor_names=valid_all, verbose=False)
    factors_oos = factors_oos.dropna(axis=1, how="all")
    common_cols = [c for c in valid if c in factors_oos.columns]
    if len(common_cols) < 5:
        print("Too few common cols")
        return

    comp_5_oos = build_composite_with_insample_stats(
        factors_oos[common_cols], ic_5, in_sample_mean, in_sample_std
    )
    comp_60_oos = build_composite_with_insample_stats(
        factors_oos[common_cols], ic_60, in_sample_mean, in_sample_std
    )
    forward_5_oos = oos_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_oos = oos_df["close"].astype(float).pct_change(60).shift(-60)
    ts_oos = oos_df["timestamp"] if "timestamp" in oos_df.columns else None

    x_5, y_5 = get_pnl_curve_with_fee(
        comp_5_oos, forward_5_oos, 5, N_QUANTILES, FEE_BPS, ts_oos
    )
    x_60, y_60 = get_pnl_curve_with_fee(
        comp_60_oos, forward_60_oos, 60, N_QUANTILES, FEE_BPS, ts_oos
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    if len(x_5) > 0 and len(y_5) > 0:
        if np.issubdtype(x_5.dtype, np.datetime64) or (hasattr(x_5, "dtype") and pd.api.types.is_datetime64_any_dtype(x_5)):
            axes[0].plot(x_5, y_5, color="C0", lw=1.5)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            axes[0].plot(x_5, y_5, color="C0", lw=1.5)
    axes[0].set_title(f"OOS 5-period long-short (after fee, {FEE_BPS} bps round-trip)")
    axes[0].set_ylabel("Cumulative equity (1 = initial)")
    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[0].grid(True, alpha=0.3)
    if len(y_5) > 0:
        total_ret_5 = y_5[-1] - 1.0
        axes[0].set_xlabel(f"Total return (after fee): {total_ret_5:.2%}")

    if len(x_60) > 0 and len(y_60) > 0:
        if np.issubdtype(x_60.dtype, np.datetime64) or (hasattr(x_60, "dtype") and pd.api.types.is_datetime64_any_dtype(x_60)):
            axes[1].plot(x_60, y_60, color="C1", lw=1.5)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            axes[1].plot(x_60, y_60, color="C1", lw=1.5)
    axes[1].set_title(f"OOS 60-period long-short (after fee, {FEE_BPS} bps round-trip)")
    axes[1].set_ylabel("Cumulative equity (1 = initial)")
    axes[1].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[1].grid(True, alpha=0.3)
    if len(y_60) > 0:
        total_ret_60 = y_60[-1] - 1.0
        axes[1].set_xlabel(f"Trading date · Total return (after fee): {total_ret_60:.2%}")
    else:
        axes[1].set_xlabel("Trading date")

    ret_5_str = f"{((y_5[-1]-1)*100):.2f}%" if len(y_5) > 0 else "N/A"
    ret_60_str = f"{((y_60[-1]-1)*100):.2f}%" if len(y_60) > 0 else "N/A"
    fig.suptitle(
        f"Best config: top{TOP_N_FACTORS} by 60|IC|, n_quantiles={N_QUANTILES}, linear · OOS PnL 扣费后 (fee={FEE_BPS} bps) · 5p: {ret_5_str}, 60p: {ret_60_str}",
        fontsize=10,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "oos_pnl_after_fee.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    if len(y_5) > 0 and len(y_60) > 0:
        print(f"OOS 5-period total return (after fee): {y_5[-1]-1:.4f}")
        print(f"OOS 60-period total return (after fee): {y_60[-1]-1:.4f}")
        print(f"OOS avg (5+60)/2: {((y_5[-1]-1)+(y_60[-1]-1))/2:.4f}")


if __name__ == "__main__":
    main()
