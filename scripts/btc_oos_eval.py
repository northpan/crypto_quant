#!/usr/bin/env python3
"""
BTC_USDT 样本外评估：过去 3 个月数据，样本内 2 个月、样本外 1 个月。
在样本内拟合等权复合因子（IC 方向 + z-score），在样本外计算多空收益。
供 program.md 中自动实验循环解析单一指标（oos_return_avg）。
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

# Experiment: use only top N factors by |IC| (None = all)
TOP_N_FACTORS = 18
N_QUANTILES = 3  # 3 = stronger long/short (top vs bottom tercile)
# Composite: "linear" (equal-weight) | "tree" (GBDT) | "mlp" (sklearn MLPRegressor)
COMPOSITE_MODE = "tree"  # exp 21: tree with stronger regularization


def load_ohlcv(symbol: str, timeframe: str, days: int):
    """Load OHLCV from data/csv, last `days` days."""
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


def compute_single_factor_backtest_ret(factor_series, forward_return, ic, forward_period, n_quantiles=None):
    """Non-overlapping long-short return; reverse direction if ic < 0."""
    if n_quantiles is None:
        n_quantiles = 5
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    if ic < 0:
        f = -f
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return float("nan")
    position = (q == q_max).astype(float) - (q == q_min).astype(float)
    indices = common[::forward_period]
    rets = []
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos, fr = position.loc[i], r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        rets.append(pos * fr)
    if not rets:
        return float("nan")
    cum = 1.0
    for r_t in rets:
        cum *= 1.0 + r_t
    return cum - 1.0


def build_composite_with_insample_stats(factor_df, ic_series, in_sample_mean, in_sample_std):
    """Build composite on OOS: z-score using in_sample mean/std, apply IC direction, then mean."""
    out = None
    n = 0
    for col in factor_df.columns:
        ic = ic_series.get(col, 0.0)
        if pd.isna(ic):
            ic = 0.0
        f = factor_df[col].astype(float)
        if ic < 0:
            f = -f
        mu = in_sample_mean.get(col)
        std = in_sample_std.get(col)
        if mu is None or std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f - mu) / std
        if out is None:
            out = z.copy()
        else:
            out = out.add(z, fill_value=0)
        n += 1
    if out is None or n == 0:
        return pd.Series(dtype=float)
    return out / n


def build_X_matrix(factor_df, ic_5, ic_60, in_sample_mean=None, in_sample_std=None):
    """Build feature matrix (z-scored, IC-directed) for model. Returns DataFrame index-aligned with factor_df."""
    cols_5 = [c for c in factor_df.columns if c in ic_5.index]
    cols_60 = [c for c in factor_df.columns if c in ic_60.index]
    cols = list(dict.fromkeys(cols_5 + cols_60))
    if not cols:
        return pd.DataFrame()
    out = {}
    for col in cols:
        ic5 = ic_5.get(col, 0.0)
        ic60 = ic_60.get(col, 0.0)
        if pd.isna(ic5):
            ic5 = 0.0
        if pd.isna(ic60):
            ic60 = 0.0
        f = factor_df[col].astype(float)
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
        if ic5 < 0:
            z = -z
        out[col] = z
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out, index=factor_df.index)


def _train_predict_composite(mode, X_in, y_in, X_oos, index_in, index_oos):
    """Train on (X_in, y_in), return (series_in, series_oos) of predictions."""
    mask = y_in.notna() & X_in.notna().all(axis=1)
    if mask.sum() < 50:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    X_tr = X_in.loc[mask].fillna(0)
    y_tr = y_in.loc[mask]
    if mode == "tree":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            # stronger regularization to reduce overfit (exp 21)
            m = GradientBoostingRegressor(n_estimators=20, max_depth=2, min_samples_leaf=50, random_state=42)
            m.fit(X_tr, y_tr)
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)
    elif mode == "mlp":
        try:
            from sklearn.neural_network import MLPRegressor
            m = MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=42, early_stopping=True)
            m.fit(X_tr, y_tr)
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    pred_in = pd.Series(m.predict(X_in.fillna(0)), index=index_in)
    pred_oos = pd.Series(m.predict(X_oos.fillna(0)), index=index_oos)
    return pred_in, pred_oos


def main():
    # Fixed: 3 months total, 2 months in-sample, 1 month out-of-sample
    total_days = 90
    in_sample_days = 60
    oos_days = 30
    symbol = "BTC_USDT"
    timeframe = "1m"

    data = load_ohlcv(symbol, timeframe, total_days)
    if len(data) < (in_sample_days * 24 * 60) * 0.9:
        print("in_sample_return_5: 0.0")
        print("in_sample_return_60: 0.0")
        print("oos_return_5: 0.0")
        print("oos_return_60: 0.0")
        print("oos_return_avg: 0.0")
        print("status: no_data")
        return

    # Split
    n = len(data)
    n_in = int(n * in_sample_days / total_days)
    in_df = data.iloc[:n_in].copy()
    oos_df = data.iloc[n_in:].copy()
    if len(oos_df) < 1000:
        print("oos_return_avg: 0.0")
        print("status: oos_too_short")
        return

    pool = FactorPool()
    categories = [FactorCategory.TECHNICAL, FactorCategory.VOLUME, FactorCategory.VOLATILITY]
    factor_names = []
    for cat in categories:
        factor_names.extend(pool.list_factors(cat))
    if not factor_names:
        factor_names = list(pool.factors.keys())[:40]

    # In-sample: compute factors, ICs, composite, returns
    factors_in = pool.compute(in_df, factor_names=factor_names, verbose=False)
    factors_in = factors_in.dropna(axis=1, how="all")
    valid = list(factors_in.columns)
    if len(valid) < 5:
        print("oos_return_avg: 0.0")
        print("status: too_few_factors")
        return

    in_sample_mean = factors_in[valid].mean().to_dict()
    in_sample_std = factors_in[valid].std().to_dict()

    forward_5_in = in_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_in = in_df["close"].astype(float).pct_change(60).shift(-60)
    test_5 = pool.test(factors_in[valid], forward_5_in, n_quantiles=N_QUANTILES, verbose=False)
    test_60 = pool.test(factors_in[valid], forward_60_in, n_quantiles=N_QUANTILES, verbose=False)
    ic_5 = test_5.set_index("name")["ic"]
    ic_60 = test_60.set_index("name")["ic"]
    if TOP_N_FACTORS is not None and len(valid) > TOP_N_FACTORS:
        abs_ic = ic_60.abs().reindex(valid).fillna(0)  # select by 60-period IC
        top_factors = abs_ic.nlargest(TOP_N_FACTORS).index.tolist()
        valid = [c for c in valid if c in top_factors]

    factors_oos = pool.compute(oos_df, factor_names=valid, verbose=False)
    factors_oos = factors_oos.dropna(axis=1, how="all")
    common_cols = [c for c in valid if c in factors_oos.columns]
    forward_5_oos = oos_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_oos = oos_df["close"].astype(float).pct_change(60).shift(-60)

    if COMPOSITE_MODE in ("tree", "mlp") and len(common_cols) >= 5:
        X_in = build_X_matrix(factors_in[valid], ic_5, ic_60, None, None)
        X_oos = build_X_matrix(factors_oos[common_cols], ic_5, ic_60, in_sample_mean, in_sample_std)
        if not X_in.empty and not X_oos.empty:
            pred_5_in, pred_5_oos = _train_predict_composite(
                COMPOSITE_MODE, X_in, forward_5_in, X_oos, X_in.index, X_oos.index
            )
            pred_60_in, pred_60_oos = _train_predict_composite(
                COMPOSITE_MODE, X_in, forward_60_in, X_oos, X_in.index, X_oos.index
            )
            if not pred_5_in.empty and not pred_5_oos.empty:
                in_ret_5 = compute_single_factor_backtest_ret(pred_5_in, forward_5_in, 1.0, 5, N_QUANTILES)
                in_ret_60 = compute_single_factor_backtest_ret(pred_60_in, forward_60_in, 1.0, 60, N_QUANTILES)
                oos_ret_5 = compute_single_factor_backtest_ret(pred_5_oos, forward_5_oos, 1.0, 5, N_QUANTILES)
                oos_ret_60 = compute_single_factor_backtest_ret(pred_60_oos, forward_60_oos, 1.0, 60, N_QUANTILES)
            else:
                in_ret_5 = in_ret_60 = oos_ret_5 = oos_ret_60 = float("nan")
        else:
            in_ret_5 = in_ret_60 = oos_ret_5 = oos_ret_60 = float("nan")
    else:
        def _composite_ret(factor_df, fwd, ic_series, period):
            comp = None
            for col in factor_df.columns:
                ic = ic_series.get(col, 0.0)
                if pd.isna(ic):
                    ic = 0.0
                f = factor_df[col].astype(float)
                if ic < 0:
                    f = -f
                mu, std = f.mean(), f.std()
                if std is None or pd.isna(std) or std < 1e-10:
                    continue
                z = (f - mu) / std
                comp = z if comp is None else comp.add(z, fill_value=0)
            if comp is None:
                return float("nan")
            comp = comp / len([c for c in factor_df.columns if c in ic_series.index])
            return compute_single_factor_backtest_ret(comp, fwd, 1.0, period, N_QUANTILES)

        in_ret_5 = _composite_ret(factors_in[valid], forward_5_in, ic_5, 5)
        in_ret_60 = _composite_ret(factors_in[valid], forward_60_in, ic_60, 60)
        if len(common_cols) < 5:
            oos_ret_5 = oos_ret_60 = float("nan")
        else:
            comp_5 = build_composite_with_insample_stats(
                factors_oos[common_cols], ic_5, in_sample_mean, in_sample_std
            )
            comp_60 = build_composite_with_insample_stats(
                factors_oos[common_cols], ic_60, in_sample_mean, in_sample_std
            )
            oos_ret_5 = compute_single_factor_backtest_ret(comp_5, forward_5_oos, 1.0, 5, N_QUANTILES)
            oos_ret_60 = compute_single_factor_backtest_ret(comp_60, forward_60_oos, 1.0, 60, N_QUANTILES)

    if pd.isna(oos_ret_5):
        oos_ret_5 = 0.0
    if pd.isna(oos_ret_60):
        oos_ret_60 = 0.0
    oos_avg = (float(oos_ret_5) + float(oos_ret_60)) / 2.0

    print("in_sample_return_5:", in_ret_5 if not pd.isna(in_ret_5) else 0.0)
    print("in_sample_return_60:", in_ret_60 if not pd.isna(in_ret_60) else 0.0)
    print("oos_return_5:", oos_ret_5)
    print("oos_return_60:", oos_ret_60)
    print("oos_return_avg:", oos_avg)


if __name__ == "__main__":
    main()
