#!/usr/bin/env python3
"""
绘制 50 遗传因子最佳实验（top_n=40, n_quantiles=3, ic_weighted）的样本内、样本外 PnL 曲线（扣费）。
"""

import sys
import pickle
from pathlib import Path
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from factors.genetic.expression import eval_expr

FEE_BPS = 10
TOP_N = 40
N_QUANTILES = 3
WEIGHT_MODE = "ic_weighted"


def load_ohlcv(symbol: str = "BTC_USDT", timeframe: str = "1m", days: int = 90) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "csv" / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    end = df["timestamp"].max()
    start = end - timedelta(days=days)
    if df["timestamp"].dt.tz is not None:
        start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)
    return df


def _forward_return(close: pd.Series, period: int) -> pd.Series:
    return close.astype(float).pct_change(period).shift(-period)


def _compute_ic(factor_series: pd.Series, forward_return: pd.Series) -> float:
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < 20:
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    ic, _ = stats.spearmanr(f, r)
    return ic if not (np.isnan(ic) or np.isinf(ic)) else float("nan")


def pnl_curve(composite: pd.Series, forward_return: pd.Series, forward_period: int,
              n_quantiles: int, fee_bps: int, timestamps: pd.Series):
    """返回 (x_dates, equity) 用于绘图。equity 从 1 开始。"""
    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return np.array([]), np.array([])
    f = composite.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return np.array([]), np.array([])
    position = (q == q_max).astype(float) - (q == q_min).astype(float)
    indices = common[::forward_period]
    rets = []
    index_used = []
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
    equity = np.cumprod(1.0 + np.array(rets))
    equity = np.concatenate([[1.0], equity])
    if timestamps is not None and len(index_used) > 0:
        try:
            first_ts = timestamps.loc[index_used[0]]
            rest_ts = timestamps.reindex(index_used).dropna()
            if len(rest_ts) == len(index_used):
                dates_list = [first_ts] + rest_ts.tolist()
                x_dates = pd.to_datetime(dates_list)
                if len(x_dates) == len(equity):
                    return x_dates, equity
        except Exception:
            pass
    return np.arange(len(equity)), equity


def main():
    pkl_path = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs.pkl"
    if not pkl_path.exists():
        print("ERROR: best_exprs.pkl not found")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        best_exprs = pickle.load(f)

    data = load_ohlcv("BTC_USDT", "1m", days=90)
    if data.empty:
        print("ERROR: No OHLCV data")
        sys.exit(1)
    n_total = len(data)
    split = int(n_total * 60 / 90)
    data_is = data.iloc[:split].copy()
    data_oos = data.iloc[split:].copy()
    ts_is = data_is["timestamp"] if "timestamp" in data_is.columns else None
    ts_oos = data_oos["timestamp"] if "timestamp" in data_oos.columns else None

    factor_names = []
    factor_values_is = []
    factor_values_oos = []
    ic_60_list = []
    for i, expr in enumerate(best_exprs):
        try:
            f_is = eval_expr(expr, data_is)
            f_oos = eval_expr(expr, data_oos)
        except Exception:
            continue
        if f_oos is None or f_oos.dropna().empty or f_oos.std() < 1e-8:
            continue
        fwd_60_is = _forward_return(data_is["close"], 60)
        ic60 = _compute_ic(f_is, fwd_60_is)
        if np.isnan(ic60) or np.isinf(ic60):
            ic60 = 0.0
        factor_names.append(f"genetic_{i}")
        factor_values_is.append(f_is)
        factor_values_oos.append(f_oos)
        ic_60_list.append(ic60)

    ic_60_abs = np.abs(ic_60_list)
    order = np.argsort(ic_60_abs)[::-1]
    top_k = min(TOP_N, len(order))
    selected = order[:top_k]

    df_is = pd.DataFrame({factor_names[j]: factor_values_is[j] for j in selected}, index=data_is.index)
    df_oos = pd.DataFrame({factor_names[j]: factor_values_oos[j] for j in selected}, index=data_oos.index)
    ic_60_sel = [ic_60_list[j] for j in selected]

    if WEIGHT_MODE == "ic_weighted":
        weights = np.array([abs(ic_60_sel[i]) for i in range(len(ic_60_sel))], dtype=float)
        weights = weights / (weights.sum() + 1e-10)
    else:
        weights = np.ones(len(ic_60_sel)) / len(ic_60_sel)

    composite_is = None
    composite_oos = None
    for i, col in enumerate(df_oos.columns):
        f_oos = df_oos[col].astype(float)
        if ic_60_sel[i] < 0:
            f_oos = -f_oos
        mu, std = f_oos.mean(), f_oos.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f_oos - mu) / std
        w = weights[i]
        composite_oos = (z * w) if composite_oos is None else composite_oos.add(z * w, fill_value=0)
    composite_oos = composite_oos / weights.sum()

    for i, col in enumerate(df_is.columns):
        f_is = df_is[col].astype(float)
        if ic_60_sel[i] < 0:
            f_is = -f_is
        mu, std = f_is.mean(), f_is.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f_is - mu) / std
        w = weights[i]
        composite_is = (z * w) if composite_is is None else composite_is.add(z * w, fill_value=0)
    composite_is = composite_is / weights.sum()

    fwd_5_is = _forward_return(data_is["close"], 5)
    fwd_60_is = _forward_return(data_is["close"], 60)
    fwd_5_oos = _forward_return(data_oos["close"], 5)
    fwd_60_oos = _forward_return(data_oos["close"], 60)

    x_is_5, y_is_5 = pnl_curve(composite_is, fwd_5_is, 5, N_QUANTILES, FEE_BPS, ts_is)
    x_is_60, y_is_60 = pnl_curve(composite_is, fwd_60_is, 60, N_QUANTILES, FEE_BPS, ts_is)
    x_oos_5, y_oos_5 = pnl_curve(composite_oos, fwd_5_oos, 5, N_QUANTILES, FEE_BPS, ts_oos)
    x_oos_60, y_oos_60 = pnl_curve(composite_oos, fwd_60_oos, 60, N_QUANTILES, FEE_BPS, ts_oos)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle("50 Genetic Factors · Best Config (top_n=40, n_quantiles=3, ic_weighted) · Fee 10 bps", fontsize=12)

    # In-Sample
    ax = axes[0]
    if len(x_is_5) > 0 and len(y_is_5) > 0:
        ret_is_5 = (y_is_5[-1] - 1) * 100
        ax.plot(x_is_5, y_is_5, label=f"5-period (ret={ret_is_5:.1f}%)", color="C0", lw=1.5)
    if len(x_is_60) > 0 and len(y_is_60) > 0:
        ret_is_60 = (y_is_60[-1] - 1) * 100
        ax.plot(x_is_60, y_is_60, label=f"60-period (ret={ret_is_60:.1f}%)", color="C1", lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax.set_ylabel("Cumulative equity")
    ax.set_title("In-Sample (60 days)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    if len(x_is_5) > 0 and hasattr(x_is_5[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Out-of-Sample
    ax = axes[1]
    if len(x_oos_5) > 0 and len(y_oos_5) > 0:
        ret_oos_5 = (y_oos_5[-1] - 1) * 100
        ax.plot(x_oos_5, y_oos_5, label=f"5-period (ret={ret_oos_5:.1f}%)", color="C0", lw=1.5)
    if len(x_oos_60) > 0 and len(y_oos_60) > 0:
        ret_oos_60 = (y_oos_60[-1] - 1) * 100
        ax.plot(x_oos_60, y_oos_60, label=f"60-period (ret={ret_oos_60:.1f}%)", color="C1", lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative equity")
    ax.set_title("Out-of-Sample (30 days)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    if len(x_oos_5) > 0 and hasattr(x_oos_5[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "genetic50_best_pnl_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
