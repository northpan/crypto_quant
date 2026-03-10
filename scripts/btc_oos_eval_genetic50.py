#!/usr/bin/env python3
"""
50 遗传因子样本外评估：加载 genetic_50_rounds 的 50 个因子，60 天样本内 / 30 天样本外，组合后计算多空收益（扣费）。
供 program_genetic50.md 与 run_30_experiments_genetic50.py 调用；输出格式与 btc_oos_eval 一致。
"""

import sys
import argparse
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


def _backtest_composite(
    composite: pd.Series,
    forward_return: pd.Series,
    forward_period: int,
    n_quantiles: int,
    fee_bps: int,
) -> tuple:
    """(total_ret, sharpe, max_dd, turnover)"""
    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    f = composite.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return float("nan"), float("nan"), float("nan"), float("nan")
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
        return float("nan"), float("nan"), float("nan"), float("nan")
    rets_arr = np.array(rets)
    cum = np.cumprod(1.0 + rets_arr)
    total_ret = cum[-1] - 1.0
    n_periods = len(rets_arr)
    mean_ret = rets_arr.mean()
    std_ret = rets_arr.std()
    periods_per_year = (252 * 24 * 60) / forward_period
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(periods_per_year) if std_ret > 1e-12 else 0.0
    run_max = np.maximum.accumulate(1.0 + (cum - 1.0))
    dd = (1.0 + (cum - 1.0)) / run_max - 1.0
    max_dd = float(np.min(dd)) if len(dd) else 0.0
    turnover = turnover_count / max(1, n_periods)
    return total_ret, sharpe, max_dd, turnover


def main():
    parser = argparse.ArgumentParser(description="OOS eval for 50 genetic factors")
    parser.add_argument("--top_n", type=int, default=50, help="Use top N factors by |IC_60|")
    parser.add_argument("--n_quantiles", type=int, default=5, help="Number of quantiles for long/short")
    parser.add_argument("--weight_mode", type=str, default="equal", choices=["equal", "ic_weighted"])
    parser.add_argument("--oos_metric", type=str, default="avg", choices=["avg", "5", "60", "weighted"])
    parser.add_argument("--oos_weight_5", type=float, default=0.5)
    args = parser.parse_args()

    top_n = args.top_n
    n_quantiles = args.n_quantiles
    weight_mode = args.weight_mode
    oos_metric = args.oos_metric
    oos_weight_5 = args.oos_weight_5

    pkl_path = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs.pkl"
    if not pkl_path.exists():
        print("ERROR: reports/genetic_50_rounds/best_exprs.pkl not found. Run run_50_rounds_and_composite_backtest.py first.")
        sys.exit(1)

    import pickle
    with open(pkl_path, "rb") as f:
        best_exprs = pickle.load(f)

    data = load_ohlcv("BTC_USDT", "1m", days=90)
    if data.empty:
        print("ERROR: No OHLCV data.")
        sys.exit(1)

    n_total = len(data)
    split = int(n_total * 60 / 90)
    data_is = data.iloc[:split].copy()
    data_oos = data.iloc[split:].copy()

    factor_names = []
    factor_values_is = []
    factor_values_oos = []
    ic_5_list = []
    ic_60_list = []

    for i, expr in enumerate(best_exprs):
        try:
            f_is = eval_expr(expr, data_is)
            f_oos = eval_expr(expr, data_oos)
        except Exception:
            continue
        if f_oos is None or f_oos.dropna().empty or f_oos.std() < 1e-8:
            continue
        fwd_5_is = _forward_return(data_is["close"], 5)
        fwd_60_is = _forward_return(data_is["close"], 60)
        ic5 = _compute_ic(f_is, fwd_5_is)
        ic60 = _compute_ic(f_is, fwd_60_is)
        if np.isnan(ic60) or np.isinf(ic60):
            ic60 = 0.0
        if np.isnan(ic5) or np.isinf(ic5):
            ic5 = 0.0
        factor_names.append(f"genetic_{i}")
        factor_values_is.append(f_is)
        factor_values_oos.append(f_oos)
        ic_5_list.append(ic5)
        ic_60_list.append(ic60)

    if not factor_names:
        print("ERROR: No valid factors.")
        sys.exit(1)

    ic_60_abs = np.abs(ic_60_list)
    order = np.argsort(ic_60_abs)[::-1]
    top_k = min(top_n, len(order))
    selected = order[:top_k]

    df_is = pd.DataFrame({factor_names[j]: factor_values_is[j] for j in selected}, index=data_is.index)
    df_oos = pd.DataFrame({factor_names[j]: factor_values_oos[j] for j in selected}, index=data_oos.index)
    ic_5_sel = [ic_5_list[j] for j in selected]
    ic_60_sel = [ic_60_list[j] for j in selected]

    if weight_mode == "ic_weighted":
        weights = np.array([abs(ic_60_sel[i]) for i in range(len(ic_60_sel))], dtype=float)
        weights = weights / (weights.sum() + 1e-10)
    else:
        weights = np.ones(len(ic_60_sel)) / len(ic_60_sel)

    composite_is = None
    composite_oos = None
    for i, col in enumerate(df_oos.columns):
        f_oos = df_oos[col].astype(float)
        ic60 = ic_60_sel[i]
        if ic60 < 0:
            f_oos = -f_oos
        mu, std = f_oos.mean(), f_oos.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f_oos - mu) / std
        w = weights[i]
        if composite_oos is None:
            composite_oos = z * w
        else:
            composite_oos = composite_oos.add(z * w, fill_value=0)
    if composite_oos is None:
        print("ERROR: No composite.")
        sys.exit(1)
    composite_oos = composite_oos / weights.sum()

    for i, col in enumerate(df_is.columns):
        f_is = df_is[col].astype(float)
        ic60 = ic_60_sel[i]
        if ic60 < 0:
            f_is = -f_is
        mu, std = f_is.mean(), f_is.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f_is - mu) / std
        w = weights[i]
        if composite_is is None:
            composite_is = z * w
        else:
            composite_is = composite_is.add(z * w, fill_value=0)
    composite_is = composite_is / weights.sum() if composite_is is not None else pd.Series(dtype=float)

    fwd_5_is = _forward_return(data_is["close"], 5)
    fwd_60_is = _forward_return(data_is["close"], 60)
    fwd_5_oos = _forward_return(data_oos["close"], 5)
    fwd_60_oos = _forward_return(data_oos["close"], 60)

    in_ret_5, in_sh_5, _, _ = _backtest_composite(composite_is, fwd_5_is, 5, n_quantiles, FEE_BPS)
    in_ret_60, in_sh_60, _, _ = _backtest_composite(composite_is, fwd_60_is, 60, n_quantiles, FEE_BPS)
    oos_ret_5, oos_sh_5, oos_dd_5, oos_to_5 = _backtest_composite(composite_oos, fwd_5_oos, 5, n_quantiles, FEE_BPS)
    oos_ret_60, oos_sh_60, oos_dd_60, oos_to_60 = _backtest_composite(composite_oos, fwd_60_oos, 60, n_quantiles, FEE_BPS)

    if oos_metric == "5":
        oos_return_avg = oos_ret_5
    elif oos_metric == "60":
        oos_return_avg = oos_ret_60
    elif oos_metric == "weighted":
        oos_return_avg = oos_weight_5 * oos_ret_5 + (1 - oos_weight_5) * oos_ret_60
    else:
        oos_return_avg = (oos_ret_5 + oos_ret_60) / 2.0

    if np.isnan(in_ret_5):
        in_ret_5 = 0.0
    if np.isnan(in_ret_60):
        in_ret_60 = 0.0
    if np.isnan(oos_ret_5):
        oos_ret_5 = 0.0
    if np.isnan(oos_ret_60):
        oos_ret_60 = 0.0
    if np.isnan(oos_return_avg):
        oos_return_avg = 0.0
    if np.isnan(oos_sh_5):
        oos_sh_5 = 0.0
    if np.isnan(oos_sh_60):
        oos_sh_60 = 0.0
    if np.isnan(oos_dd_5):
        oos_dd_5 = 0.0
    if np.isnan(oos_dd_60):
        oos_dd_60 = 0.0

    print("in_sample_return_5:", in_ret_5)
    print("in_sample_return_60:", in_ret_60)
    print("oos_return_5:", oos_ret_5)
    print("oos_return_60:", oos_ret_60)
    print("oos_return_avg:", oos_return_avg)
    print("oos_sharpe_5:", oos_sh_5)
    print("oos_sharpe_60:", oos_sh_60)
    print("oos_max_drawdown_5:", oos_dd_5)
    print("oos_max_drawdown_60:", oos_dd_60)
    print("oos_turnover_5:", oos_to_5)
    print("oos_turnover_60:", oos_to_60)
    print("fee_bps:", FEE_BPS)


if __name__ == "__main__":
    main()
