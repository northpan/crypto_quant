#!/usr/bin/env python3
"""
绘制 program 实验里「保留」的 5 次进步对应的样本内 / 样本外 PnL 曲线。
5 个配置：baseline, top20, top20_q3, top18_q3, top18_60ic_q3
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from factors import FactorPool
from factors.base_factor import FactorCategory


def load_ohlcv(symbol: str, timeframe: str, days: int):
    path = PROJECT_ROOT / "data" / "csv" / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    end = df["timestamp"].max()
    start = end - pd.Timedelta(days=days)
    if df["timestamp"].dt.tz is not None:
        start = start.tz_localize(None) if start.tzinfo is None else start
        end = end.tz_localize(None) if end.tzinfo is None else end
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)
    return df


def get_pnl_curve(factor_series, forward_return, forward_period, n_quantiles, timestamps=None):
    """Non-overlapping long-short equity curve (starts at 1). Returns (x, y)."""
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
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos, fr = position.loc[i], r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        rets.append(pos * fr)
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


def build_composite(factor_df, ic_series, in_sample_mean=None, in_sample_std=None):
    """Equal-weight composite; if in_sample_mean/std given, use for z-score (OOS)."""
    out = None
    n = 0
    for col in factor_df.columns:
        ic = ic_series.get(col, 0.0)
        if pd.isna(ic):
            ic = 0.0
        f = factor_df[col].astype(float)
        if ic < 0:
            f = -f
        if in_sample_mean is not None and in_sample_std is not None:
            mu, std = in_sample_mean.get(col), in_sample_std.get(col)
            if mu is None or std is None or pd.isna(std) or std < 1e-10:
                continue
            z = (f - mu) / std
        else:
            mu, std = f.mean(), f.std()
            if std is None or pd.isna(std) or std < 1e-10:
                continue
            z = (f - mu) / std
        out = z if out is None else out.add(z, fill_value=0)
        n += 1
    if out is None or n == 0:
        return pd.Series(dtype=float)
    return out / n


# 5 次保留的配置 (top_n, use_60_ic_only, n_quantiles)
CONFIGS = [
    {"name": "1_baseline", "top_n": None, "use_60_ic": False, "n_quantiles": 5},
    {"name": "2_top20", "top_n": 20, "use_60_ic": False, "n_quantiles": 5},
    {"name": "3_top20_q3", "top_n": 20, "use_60_ic": False, "n_quantiles": 3},
    {"name": "4_top18_q3", "top_n": 18, "use_60_ic": False, "n_quantiles": 3},
    {"name": "5_top18_60ic_q3", "top_n": 18, "use_60_ic": True, "n_quantiles": 3},
]


def run_one_config(cfg, in_df, oos_df, factors_in, factors_oos, valid_all, pool,
                   forward_5_in, forward_60_in, forward_5_oos, forward_60_oos,
                   in_sample_mean, in_sample_std):
    valid = list(valid_all)
    test_5 = pool.test(factors_in[valid], forward_5_in, n_quantiles=cfg["n_quantiles"], verbose=False)
    test_60 = pool.test(factors_in[valid], forward_60_in, n_quantiles=cfg["n_quantiles"], verbose=False)
    ic_5 = test_5.set_index("name")["ic"]
    ic_60 = test_60.set_index("name")["ic"]
    if cfg["top_n"] is not None and len(valid) > cfg["top_n"]:
        if cfg["use_60_ic"]:
            abs_ic = ic_60.abs().reindex(valid).fillna(0)
        else:
            abs_ic = (ic_5.abs() + ic_60.abs()).reindex(valid).fillna(0)
        top_factors = abs_ic.nlargest(cfg["top_n"]).index.tolist()
        valid = [c for c in valid if c in top_factors]
    common_cols = [c for c in valid if c in factors_oos.columns]
    if len(common_cols) < 5:
        return None
    # In-sample composites
    comp_5_in = build_composite(factors_in[valid], ic_5, None, None)
    comp_60_in = build_composite(factors_in[valid], ic_60, None, None)
    # OOS composites (use in-sample stats)
    comp_5_oos = build_composite(factors_oos[common_cols], ic_5, in_sample_mean, in_sample_std)
    comp_60_oos = build_composite(factors_oos[common_cols], ic_60, in_sample_mean, in_sample_std)
    q = cfg["n_quantiles"]
    ts_in = in_df["timestamp"] if "timestamp" in in_df.columns else None
    ts_oos = oos_df["timestamp"] if "timestamp" in oos_df.columns else None
    curves = {}
    curves["in_5"] = get_pnl_curve(comp_5_in, forward_5_in, 5, q, ts_in)
    curves["in_60"] = get_pnl_curve(comp_60_in, forward_60_in, 60, q, ts_in)
    curves["oos_5"] = get_pnl_curve(comp_5_oos, forward_5_oos, 5, q, ts_oos)
    curves["oos_60"] = get_pnl_curve(comp_60_oos, forward_60_oos, 60, q, ts_oos)
    return curves


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
    factors_oos = pool.compute(oos_df, factor_names=valid_all, verbose=False)
    factors_oos = factors_oos.dropna(axis=1, how="all")
    in_sample_mean = factors_in[valid_all].mean().to_dict()
    in_sample_std = factors_in[valid_all].std().to_dict()
    forward_5_in = in_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_in = in_df["close"].astype(float).pct_change(60).shift(-60)
    forward_5_oos = oos_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_oos = oos_df["close"].astype(float).pct_change(60).shift(-60)

    all_curves = {}
    for cfg in CONFIGS:
        curves = run_one_config(
            cfg, in_df, oos_df, factors_in, factors_oos, valid_all, pool,
            forward_5_in, forward_60_in, forward_5_oos, forward_60_oos,
            in_sample_mean, in_sample_std,
        )
        if curves:
            all_curves[cfg["name"]] = curves

    # Plot: 4 subplots [In-sample 5p] [OOS 5p] [In-sample 60p] [OOS 60p]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    labels = {"1_baseline": "1 baseline", "2_top20": "2 top20", "3_top20_q3": "3 top20 q3", "4_top18_q3": "4 top18 q3", "5_top18_60ic_q3": "5 top18 60ic q3"}
    for (name, curves), color in zip(all_curves.items(), colors):
        label = labels.get(name, name)
        for key, ax in [("in_5", axes[0, 0]), ("oos_5", axes[0, 1]), ("in_60", axes[1, 0]), ("oos_60", axes[1, 1])]:
            x, y = curves[key]
            if len(x) == 0 or len(y) == 0:
                continue
            if np.issubdtype(x.dtype, np.datetime64) or (hasattr(x, "dtype") and pd.api.types.is_datetime64_any_dtype(x)):
                ax.plot(x, y, label=label, color=color, lw=1.2)
            else:
                ax.plot(x, y, label=label, color=color, lw=1.2)
    axes[0, 0].set_title("In-sample · 5-period")
    axes[0, 0].set_ylabel("Cumulative equity (1 = initial)")
    axes[0, 0].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_title("Out-of-sample · 5-period")
    axes[0, 1].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[0, 1].legend(loc="best", fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].set_title("In-sample · 60-period")
    axes[1, 0].set_xlabel("Trading date")
    axes[1, 0].set_ylabel("Cumulative equity (1 = initial)")
    axes[1, 0].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[1, 0].legend(loc="best", fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_title("Out-of-sample · 60-period")
    axes[1, 1].set_xlabel("Trading date")
    axes[1, 1].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[1, 1].legend(loc="best", fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    for ax in axes.flat:
        if len(ax.get_lines()) > 0 and hasattr(ax.get_lines()[0].get_xdata(), "dtype") and pd.api.types.is_datetime64_any_dtype(ax.get_lines()[0].get_xdata()):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
    fig.suptitle("Best 5 improvements: In-sample vs Out-of-sample PnL (2m in / 1m out)", fontsize=11)
    fig.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "best5_pnl_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
