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

# Experiment: use only top N factors by |IC| (None = all). For tree/mlp/lgb, USE_ALL_FACTORS_FOR_NL overrides.
TOP_N_FACTORS = 18
# Factor selection: "60" = by 60-period |IC| only, "5_60" = by |IC5|+|IC60|
SELECT_IC = "60"
N_QUANTILES = 3  # 3 = stronger long/short (top vs bottom tercile)
# Composite: "linear" | "tree" (GBDT) | "mlp" | "lgb" (LightGBM)
COMPOSITE_MODE = "linear"
# For non-linear modes: use all factors (before top-N filter) when True
USE_ALL_FACTORS_FOR_NL = False
# For non-linear: demean only (no scale) when True; else z-score
DEMEAN_FACTORS_FOR_NL = False
# For non-linear: rank X and y (pct) before train/predict when True
RANK_TRANSFORM = False
# Tree/MLP: use only last N days of in-sample for training (None = use full in-sample). Crypto 风格突变，用短回看加快切换
TRAIN_LOOKBACK_DAYS = None
# Tree/MLP: on OOS, retrain every N days (expanding: in_sample + previous OOS chunks); None = single model. 1=每天重训
INFER_RETRAIN_DAYS = None
# Reported oos_return_avg: "avg" = (oos_5+oos_60)/2, "5" = oos_return_5 only, "60" = oos_return_60 only
# "weighted" = OOS_WEIGHT_5 * oos_5 + (1 - OOS_WEIGHT_5) * oos_60
OOS_METRIC = "avg"
OOS_WEIGHT_5 = 0.5

# Transaction cost: bps per side (round-trip = 2 * FEE_BPS). Applied at each rebalance when position changes.
FEE_BPS = 10  # 10 bps = 0.1% round-trip per rebalance


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


def compute_single_factor_backtest_ret(
    factor_series,
    forward_return,
    ic,
    forward_period,
    n_quantiles=None,
    fee_bps=None,
    return_metrics=False,
):
    """Non-overlapping long-short return; reverse direction if ic < 0.
    If fee_bps is set, subtract 2*(fee_bps/1e4) at each rebalance when position changes.
    If return_metrics=True, return (total_return, sharpe, max_drawdown, turnover) else total_return only.
    """
    if n_quantiles is None:
        n_quantiles = 5
    if fee_bps is None:
        fee_bps = FEE_BPS
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        if return_metrics:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    if ic < 0:
        f = -f
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        if return_metrics:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return float("nan")
    position = (q == q_max).astype(float) - (q == q_min).astype(float)
    indices = common[::forward_period]
    rets = []
    prev_pos = None
    turnover_count = 0
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos, fr = position.loc[i], r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        raw_ret = pos * fr
        if prev_pos is not None and pos != prev_pos:
            raw_ret -= 2 * (fee_bps / 1e4)
            turnover_count += 1
        prev_pos = pos
        rets.append(raw_ret)
    if not rets:
        if return_metrics:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return float("nan")
    cum = 1.0
    period_rets = []
    for r_t in rets:
        cum *= 1.0 + r_t
        period_rets.append(r_t)
    total_ret = cum - 1.0
    if not return_metrics:
        return total_ret
    # Sharpe (annualized): assume ~252*24*60/forward_period periods per year for 1m data
    n_periods = len(period_rets)
    if n_periods < 2:
        return total_ret, float("nan"), float("nan"), float("nan")
    period_rets_arr = np.array(period_rets)
    mean_ret = period_rets_arr.mean()
    std_ret = period_rets_arr.std()
    periods_per_year = (252 * 24 * 60) / forward_period
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(periods_per_year) if std_ret > 1e-12 else 0.0
    # Max drawdown from cumulative
    cum_arr = np.cumprod(1.0 + period_rets_arr) - 1.0
    run_max = np.maximum.accumulate(1.0 + cum_arr)
    dd = (1.0 + cum_arr) / run_max - 1.0
    max_dd = float(np.min(dd)) if len(dd) else 0.0
    turnover = turnover_count / max(1, n_periods)
    return total_ret, sharpe, max_dd, turnover


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


def build_X_matrix(factor_df, ic_5, ic_60, in_sample_mean=None, in_sample_std=None, demean_only=False):
    """Build feature matrix (z-scored or demeaned, IC-directed) for model. Returns DataFrame index-aligned with factor_df."""
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
            if mu is None:
                continue
            if demean_only:
                z = (f - mu)
            else:
                if std is None or pd.isna(std) or std < 1e-10:
                    continue
                z = (f - mu) / std
        else:
            mu, std = f.mean(), f.std()
            if demean_only:
                z = (f - mu)
            else:
                if std is None or pd.isna(std) or std < 1e-10:
                    continue
                z = (f - mu) / std
        if ic5 < 0:
            z = -z
        out[col] = z
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out, index=factor_df.index)


def _train_predict_composite(mode, X_in, y_in, X_oos, index_in, index_oos, rank_transform=False, train_lookback_bars=None):
    """Train on (X_in, y_in), return (series_in, series_oos). train_lookback_bars: use only last N bars for training (None = all)."""
    mask = y_in.notna() & X_in.notna().all(axis=1)
    if mask.sum() < 50:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    X_tr = X_in.loc[mask].copy()
    y_tr = y_in.loc[mask].copy()
    if train_lookback_bars is not None and len(X_tr) > train_lookback_bars:
        X_tr = X_tr.iloc[-train_lookback_bars:]
        y_tr = y_tr.iloc[-train_lookback_bars:]
    if len(X_tr) < 50:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if rank_transform:
        X_tr = X_tr.rank(pct=True, method="average").fillna(0.5)
        y_tr = y_tr.rank(pct=True, method="average")
        X_in_use = X_in.rank(pct=True, method="average").fillna(0.5)
        X_oos_use = X_oos.rank(pct=True, method="average").fillna(0.5)
    else:
        X_in_use = X_in.copy()
        X_oos_use = X_oos.copy()
    X_tr = X_tr.fillna(0)
    X_in_use = X_in_use.fillna(0)
    X_oos_use = X_oos_use.fillna(0)
    # LightGBM requires string column names and same columns at predict; use DataFrame throughout
    def _ensure_str_cols(df):
        d = df.copy()
        d.columns = [str(c) for c in d.columns]
        return d
    def _safe_fillinf(df):
        d = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return d
    X_tr = _safe_fillinf(_ensure_str_cols(X_tr))
    X_in_use = _safe_fillinf(_ensure_str_cols(X_in_use))
    X_oos_use = _safe_fillinf(_ensure_str_cols(X_oos_use))
    if mode == "linear":
        try:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0, random_state=42)
            m.fit(X_tr, y_tr)
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)
    elif mode == "tree":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=30, max_depth=3, min_samples_leaf=20, random_state=42)
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
    elif mode == "lgb":
        try:
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=30, max_depth=3, min_child_samples=20, random_state=42, verbosity=-1)
            m.fit(X_tr, y_tr)
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    train_cols = list(X_tr.columns)
    X_in_use = X_in_use.reindex(columns=train_cols, fill_value=0.0)
    X_oos_use = X_oos_use.reindex(columns=train_cols, fill_value=0.0)
    if X_in_use.isna().all().all() or X_oos_use.isna().all().all():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    pred_in = pd.Series(m.predict(X_in_use), index=index_in)
    pred_oos = pd.Series(m.predict(X_oos_use), index=index_oos)
    return pred_in, pred_oos


def _oos_expanding_predict(
    in_df, oos_df, factors_oos, pool, valid_nl, common_cols, ic_5, ic_60,
    in_sample_mean, in_sample_std, mode, rank_transform, train_lookback_bars,
    infer_retrain_days, in_sample_days, demean_only,
):
    """Expanding retrain on OOS chunks; returns (pred_5_oos, pred_60_oos) with index=oos_df.index."""
    bars_per_day = 24 * 60
    chunk_bars = int(infer_retrain_days * bars_per_day)
    if chunk_bars <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    n_oos = len(oos_df)
    pred_5_list, pred_60_list, index_list = [], [], []
    for start in range(0, n_oos, chunk_bars):
        end = min(start + chunk_bars, n_oos)
        train_df = pd.concat([in_df, oos_df.iloc[:start]], ignore_index=True) if start > 0 else in_df
        factors_train = pool.compute(train_df, factor_names=valid_nl, verbose=False)
        factors_train = factors_train.dropna(axis=1, how="all")
        cols = [c for c in valid_nl if c in factors_train.columns]
        if len(cols) < 5:
            break
        X_train = build_X_matrix(
            factors_train[cols], ic_5, ic_60, in_sample_mean, in_sample_std, demean_only=demean_only
        )
        forward_5_train = train_df["close"].astype(float).pct_change(5).shift(-5)
        forward_60_train = train_df["close"].astype(float).pct_change(60).shift(-60)
        if train_lookback_bars is not None and len(X_train) > train_lookback_bars:
            X_train = X_train.iloc[-train_lookback_bars:]
            forward_5_train = forward_5_train.iloc[-train_lookback_bars:]
            forward_60_train = forward_60_train.iloc[-train_lookback_bars:]
        factors_chunk = factors_oos.iloc[start:end]
        if len(factors_chunk) == 0:
            break
        chunk_cols = [c for c in common_cols if c in factors_chunk.columns]
        if len(chunk_cols) < 5:
            break
        X_oos_chunk = build_X_matrix(
            factors_chunk[chunk_cols], ic_5, ic_60, in_sample_mean, in_sample_std, demean_only=demean_only
        )
        if X_oos_chunk.empty or len(X_train) < 50:
            break
        _, pred_5_chunk = _train_predict_composite(
            mode, X_train, forward_5_train, X_oos_chunk, X_train.index, X_oos_chunk.index,
            rank_transform=rank_transform, train_lookback_bars=None
        )
        _, pred_60_chunk = _train_predict_composite(
            mode, X_train, forward_60_train, X_oos_chunk, X_train.index, X_oos_chunk.index,
            rank_transform=rank_transform, train_lookback_bars=None
        )
        if pred_5_chunk.empty or pred_60_chunk.empty:
            break
        pred_5_list.append(pred_5_chunk)
        pred_60_list.append(pred_60_chunk)
        index_list.append(oos_df.index[start:end])
    if not pred_5_list:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    pred_5_oos = pd.concat(pred_5_list)
    pred_60_oos = pd.concat(pred_60_list)
    pred_5_oos = pred_5_oos.reindex(oos_df.index).fillna(0)
    pred_60_oos = pred_60_oos.reindex(oos_df.index).fillna(0)
    return pred_5_oos, pred_60_oos


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

    valid_all = list(valid)
    in_sample_mean = factors_in[valid_all].mean().to_dict()
    in_sample_std = factors_in[valid_all].std().to_dict()

    forward_5_in = in_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_in = in_df["close"].astype(float).pct_change(60).shift(-60)
    test_5 = pool.test(factors_in[valid], forward_5_in, n_quantiles=N_QUANTILES, verbose=False)
    test_60 = pool.test(factors_in[valid], forward_60_in, n_quantiles=N_QUANTILES, verbose=False)
    ic_5 = test_5.set_index("name")["ic"]
    ic_60 = test_60.set_index("name")["ic"]
    if TOP_N_FACTORS is not None and len(valid) > TOP_N_FACTORS:
        if SELECT_IC == "60":
            abs_ic = ic_60.abs().reindex(valid).fillna(0)
        else:
            abs_ic = (ic_5.abs() + ic_60.abs()).reindex(valid).fillna(0)
        top_factors = abs_ic.nlargest(TOP_N_FACTORS).index.tolist()
        valid = [c for c in valid if c in top_factors]
    valid_nl = valid_all if (COMPOSITE_MODE in ("tree", "mlp", "lgb") and USE_ALL_FACTORS_FOR_NL) else valid

    factors_oos = pool.compute(oos_df, factor_names=valid_nl, verbose=False)
    factors_oos = factors_oos.dropna(axis=1, how="all")
    common_cols = [c for c in valid_nl if c in factors_oos.columns]
    forward_5_oos = oos_df["close"].astype(float).pct_change(5).shift(-5)
    forward_60_oos = oos_df["close"].astype(float).pct_change(60).shift(-60)

    train_lookback_bars = None
    if TRAIN_LOOKBACK_DAYS is not None and in_sample_days and in_sample_days > 0:
        train_lookback_bars = int(n_in * min(1.0, TRAIN_LOOKBACK_DAYS / in_sample_days))

    use_nl_branch = (COMPOSITE_MODE in ("tree", "mlp", "lgb") or RANK_TRANSFORM) and len(common_cols) >= 5
    if use_nl_branch:
        X_in = build_X_matrix(
            factors_in[valid_nl], ic_5, ic_60, None, None, demean_only=DEMEAN_FACTORS_FOR_NL
        )
        X_oos = build_X_matrix(
            factors_oos[common_cols], ic_5, ic_60, in_sample_mean, in_sample_std, demean_only=DEMEAN_FACTORS_FOR_NL
        )
        if not X_in.empty and not X_oos.empty:
            mode = "linear" if (RANK_TRANSFORM and COMPOSITE_MODE == "linear") else COMPOSITE_MODE
            if INFER_RETRAIN_DAYS and COMPOSITE_MODE in ("tree", "mlp", "lgb"):
                pred_5_in, _ = _train_predict_composite(
                    mode, X_in, forward_5_in, X_oos, X_in.index, X_oos.index,
                    rank_transform=RANK_TRANSFORM, train_lookback_bars=train_lookback_bars
                )
                pred_60_in, _ = _train_predict_composite(
                    mode, X_in, forward_60_in, X_oos, X_in.index, X_oos.index,
                    rank_transform=RANK_TRANSFORM, train_lookback_bars=train_lookback_bars
                )
                pred_5_oos, pred_60_oos = _oos_expanding_predict(
                    in_df, oos_df, factors_oos, pool, valid_nl, common_cols, ic_5, ic_60,
                    in_sample_mean, in_sample_std, mode, RANK_TRANSFORM, train_lookback_bars,
                    INFER_RETRAIN_DAYS, in_sample_days, DEMEAN_FACTORS_FOR_NL,
                )
            else:
                pred_5_in, pred_5_oos = _train_predict_composite(
                    mode, X_in, forward_5_in, X_oos, X_in.index, X_oos.index,
                    rank_transform=RANK_TRANSFORM, train_lookback_bars=train_lookback_bars
                )
                pred_60_in, pred_60_oos = _train_predict_composite(
                    mode, X_in, forward_60_in, X_oos, X_in.index, X_oos.index,
                    rank_transform=RANK_TRANSFORM, train_lookback_bars=train_lookback_bars
                )
            if not pred_5_in.empty and not pred_5_oos.empty:
                in_ret_5 = compute_single_factor_backtest_ret(
                    pred_5_in, forward_5_in, 1.0, 5, N_QUANTILES, fee_bps=FEE_BPS
                )
                in_ret_60 = compute_single_factor_backtest_ret(
                    pred_60_in, forward_60_in, 1.0, 60, N_QUANTILES, fee_bps=FEE_BPS
                )
                oos_ret_5, oos_sharpe_5, oos_max_dd_5, oos_turnover_5 = compute_single_factor_backtest_ret(
                    pred_5_oos, forward_5_oos, 1.0, 5, N_QUANTILES, fee_bps=FEE_BPS, return_metrics=True
                )
                oos_ret_60, oos_sharpe_60, oos_max_dd_60, oos_turnover_60 = compute_single_factor_backtest_ret(
                    pred_60_oos, forward_60_oos, 1.0, 60, N_QUANTILES, fee_bps=FEE_BPS, return_metrics=True
                )
            else:
                in_ret_5 = in_ret_60 = oos_ret_5 = oos_ret_60 = float("nan")
                oos_sharpe_5 = oos_sharpe_60 = oos_max_dd_5 = oos_max_dd_60 = float("nan")
                oos_turnover_5 = oos_turnover_60 = float("nan")
        else:
            in_ret_5 = in_ret_60 = oos_ret_5 = oos_ret_60 = float("nan")
            oos_sharpe_5 = oos_sharpe_60 = oos_max_dd_5 = oos_max_dd_60 = float("nan")
            oos_turnover_5 = oos_turnover_60 = float("nan")
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
            return compute_single_factor_backtest_ret(
                comp, fwd, 1.0, period, N_QUANTILES, fee_bps=FEE_BPS
            )

        in_ret_5 = _composite_ret(factors_in[valid], forward_5_in, ic_5, 5)
        in_ret_60 = _composite_ret(factors_in[valid], forward_60_in, ic_60, 60)
        if len(common_cols) < 5:
            oos_ret_5 = oos_ret_60 = float("nan")
            oos_sharpe_5 = oos_sharpe_60 = oos_max_dd_5 = oos_max_dd_60 = float("nan")
            oos_turnover_5 = oos_turnover_60 = float("nan")
        else:
            comp_5 = build_composite_with_insample_stats(
                factors_oos[common_cols], ic_5, in_sample_mean, in_sample_std
            )
            comp_60 = build_composite_with_insample_stats(
                factors_oos[common_cols], ic_60, in_sample_mean, in_sample_std
            )
            (oos_ret_5, oos_sharpe_5, oos_max_dd_5, oos_turnover_5) = compute_single_factor_backtest_ret(
                comp_5, forward_5_oos, 1.0, 5, N_QUANTILES, fee_bps=FEE_BPS, return_metrics=True
            )
            (oos_ret_60, oos_sharpe_60, oos_max_dd_60, oos_turnover_60) = compute_single_factor_backtest_ret(
                comp_60, forward_60_oos, 1.0, 60, N_QUANTILES, fee_bps=FEE_BPS, return_metrics=True
            )

    if pd.isna(oos_ret_5):
        oos_ret_5 = 0.0
    if pd.isna(oos_ret_60):
        oos_ret_60 = 0.0
    if OOS_METRIC == "5":
        oos_avg = float(oos_ret_5)
    elif OOS_METRIC == "60":
        oos_avg = float(oos_ret_60)
    elif OOS_METRIC == "weighted":
        oos_avg = OOS_WEIGHT_5 * float(oos_ret_5) + (1.0 - OOS_WEIGHT_5) * float(oos_ret_60)
    else:
        oos_avg = (float(oos_ret_5) + float(oos_ret_60)) / 2.0

    print("in_sample_return_5:", in_ret_5 if not pd.isna(in_ret_5) else 0.0)
    print("in_sample_return_60:", in_ret_60 if not pd.isna(in_ret_60) else 0.0)
    print("oos_return_5:", oos_ret_5)
    print("oos_return_60:", oos_ret_60)
    print("oos_return_avg:", oos_avg)
    print("oos_sharpe_5:", oos_sharpe_5 if not pd.isna(oos_sharpe_5) else 0.0)
    print("oos_sharpe_60:", oos_sharpe_60 if not pd.isna(oos_sharpe_60) else 0.0)
    print("oos_max_drawdown_5:", oos_max_dd_5 if not pd.isna(oos_max_dd_5) else 0.0)
    print("oos_max_drawdown_60:", oos_max_dd_60 if not pd.isna(oos_max_dd_60) else 0.0)
    print("oos_turnover_5:", oos_turnover_5 if not pd.isna(oos_turnover_5) else 0.0)
    print("oos_turnover_60:", oos_turnover_60 if not pd.isna(oos_turnover_60) else 0.0)
    print("fee_bps:", FEE_BPS)


if __name__ == "__main__":
    main()
