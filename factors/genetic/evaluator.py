"""
遗传因子挖掘 - 评估器
对单个表达式计算 IC（5/60 期）与扣费后多空收益 ret（5/60 期），供遗传算法适应度使用。
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from scipy import stats

from .expression import eval_expr

# 与 btc_oos_eval / factor_performance_report 一致
FEE_BPS = 10


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


def _compute_segment_metrics(
    factor_series: pd.Series,
    df: pd.DataFrame,
    forward_periods: list,
    n_quantiles: int,
    fee_bps: int,
    n_segments: int = 3,
) -> Dict[str, Dict[int, list]]:
    """
    将样本内时间轴切成 n_segments 段，分别计算每段的 IC 和 ret。
    返回:
      {
        "ic": {5: [...], 60: [...]},
        "ret": {5: [...], 60: [...]},
      }
    """
    ic_dict: Dict[int, list] = {p: [] for p in forward_periods}
    ret_dict: Dict[int, list] = {p: [] for p in forward_periods}

    # 使用 df 的索引做时间切分，若样本过短则直接返回空
    idx_all = df.index
    if len(idx_all) < n_segments * max(forward_periods) * 2:
        return {"ic": ic_dict, "ret": ret_dict}

    segments = np.array_split(idx_all, n_segments)
    for seg in segments:
        if len(seg) < n_quantiles * 2:
            continue
        df_seg = df.loc[seg]
        f_seg = factor_series.loc[seg]
        for period in forward_periods:
            fwd_seg = _forward_return(df_seg["close"], period)
            ic_seg = _compute_ic(f_seg, fwd_seg)
            if np.isnan(ic_seg) or np.isinf(ic_seg):
                continue
            ret_seg = _compute_single_factor_backtest_ret(
                f_seg, fwd_seg, ic_seg, period, n_quantiles=n_quantiles, fee_bps=fee_bps
            )
            if np.isnan(ret_seg) or np.isinf(ret_seg):
                continue
            ic_dict[period].append(float(ic_seg))
            ret_dict[period].append(float(ret_seg))

    return {"ic": ic_dict, "ret": ret_dict}


def _compute_single_factor_backtest_ret(
    factor_series: pd.Series,
    forward_return: pd.Series,
    ic: float,
    forward_period: int,
    n_quantiles: int = 5,
    fee_bps: int = FEE_BPS,
) -> float:
    """单因子多空回测收益（扣费）。IC<0 时做多 Q1、做空 Q5。"""
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    if ic < 0:
        f = -f
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max = q.max()
    q_min = q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return float("nan")
    long_signal = (q == q_max).astype(float)
    short_signal = (q == q_min).astype(float)
    position = long_signal - short_signal
    indices = common[::forward_period]
    if len(indices) < 2:
        return float("nan")
    rets = []
    prev_pos = None
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos = position.loc[i]
        fr = r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        raw_ret = pos * fr
        if prev_pos is not None and pos != prev_pos:
            raw_ret -= 2 * (fee_bps / 1e4)
        prev_pos = pos
        rets.append(raw_ret)
    if not rets:
        return float("nan")
    cum = 1.0
    for r_t in rets:
        cum *= 1.0 + r_t
    return cum - 1.0


def evaluate_expression(
    expr: Any,
    df: pd.DataFrame,
    forward_periods: list = (5, 60),
    n_quantiles: int = 5,
    fee_bps: int = FEE_BPS,
) -> Dict[str, float]:
    """
    对表达式在 df 上求值，计算因子序列，再计算 IC 与扣费后多空收益。
    df 需含列: open, high, low, close, volume。
    返回: ic_5, ic_60, ret_5, ret_60；若某期无效则为 nan。
    """
    if "close" not in df.columns:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}

    try:
        factor_series = eval_expr(expr, df)
    except Exception:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}

    if factor_series is None or factor_series.dropna().empty:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}

    # 过滤退化因子：方差过小或唯一值过少
    valid = factor_series.dropna()
    if len(valid) < 100:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}
    std_val = valid.std()
    n_unique = valid.nunique()
    if std_val is None or pd.isna(std_val) or std_val < 1e-8:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}
    if n_unique < 10:
        return {"ic_5": np.nan, "ic_60": np.nan, "ret_5": np.nan, "ret_60": np.nan}

    out: Dict[str, float] = {}
    for period in forward_periods:
        fwd = _forward_return(df["close"], period)
        ic = _compute_ic(factor_series, fwd)
        ret = _compute_single_factor_backtest_ret(
            factor_series, fwd, ic, period, n_quantiles=n_quantiles, fee_bps=fee_bps
        )
        out[f"ic_{period}"] = ic
        out[f"ret_{period}"] = ret

    # 稳健性指标：多段 IC 与 ret
    seg = _compute_segment_metrics(
        factor_series,
        df,
        forward_periods=list(forward_periods),
        n_quantiles=n_quantiles,
        fee_bps=fee_bps,
        n_segments=3,
    )

    ic_vals_abs = []
    ret_vals = []
    for p in forward_periods:
        ic_vals_abs += [abs(x) for x in seg["ic"][p]]
        ret_vals += seg["ret"][p]

    if ic_vals_abs:
        ic_arr = np.array(ic_vals_abs, dtype=float)
        ic_robust = float(np.median(ic_arr) - 0.5 * np.std(ic_arr))
    else:
        ic_robust = 0.0

    if ret_vals:
        ret_arr = np.array(ret_vals, dtype=float)
        # 使用多段 ret 中的最差一段作为稳健性下界
        ret_robust = float(np.nanmin(ret_arr))
    else:
        ret_robust = 0.0

    out["ic_robust"] = ic_robust
    out["ret_robust"] = ret_robust
    return out
