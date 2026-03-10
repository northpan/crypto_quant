#!/usr/bin/env python3
"""
遗传因子组合样本外评估：与 btc_oos_eval.py 结构一致。
加载 50 个遗传因子（reports/genetic_50_rounds/best_exprs.pkl），
60 天样本内 / 30 天样本外，组合后计算多空收益（扣费）。
供实验迭代与 results_genetic50.tsv 记录；输出格式与 btc_oos_eval 一致便于解析。
"""

import sys
import os
import argparse
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

# ---------- 实验配置（与 btc_oos_eval.py 风格一致，可被命令行覆盖）----------
TOP_N_FACTORS = 40
N_QUANTILES = 3
WEIGHT_MODE = "ic_weighted"  # "equal" | "ic_weighted"
OOS_METRIC = "avg"  # "avg" | "5" | "60" | "weighted"
OOS_WEIGHT_5 = 0.5
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
    """单段 IC（全样本）"""
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < 20:
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    ic, _ = stats.spearmanr(f, r)
    return ic if not (np.isnan(ic) or np.isinf(ic)) else float("nan")


def _robust_ic_score(
    factor_series: pd.Series,
    fwd_5: pd.Series,
    fwd_60: pd.Series,
    n_segments: int = 3,
    penalty_lambda: float = 0.5,
) -> float:
    """
    稳健 IC 评分：在样本内切成若干段，计算每段 IC_5 / IC_60，
    使用 median(|IC|) 奖励稳定贡献，std(|IC|) 作为惩罚项。
    """
    if n_segments <= 1:
        return float("nan")

    common_5 = factor_series.dropna().index.intersection(fwd_5.dropna().index)
    if len(common_5) < n_segments * 20:
        return float("nan")
    segments = np.array_split(common_5, n_segments)

    ic5_list = []
    ic60_list = []
    for seg in segments:
        if len(seg) < 20:
            continue
        f_seg = factor_series.loc[seg].astype(float)
        r5 = fwd_5.loc[seg].astype(float)
        r60 = fwd_60.loc[seg].astype(float)
        ic5, _ = stats.spearmanr(f_seg, r5)
        ic60, _ = stats.spearmanr(f_seg, r60)
        if not (np.isnan(ic5) or np.isinf(ic5)):
            ic5_list.append(ic5)
        if not (np.isnan(ic60) or np.isinf(ic60)):
            ic60_list.append(ic60)

    if not ic5_list and not ic60_list:
        return float("nan")

    vals = [abs(x) for x in ic5_list] + [abs(x) for x in ic60_list]
    med = np.median(vals)
    std = np.std(vals) if len(vals) > 1 else 0.0
    score = med - penalty_lambda * std
    return float(score)


def _backtest_composite(
    composite: pd.Series,
    forward_return: pd.Series,
    forward_period: int,
    n_quantiles: int,
    fee_bps: int,
    *,
    quantile_source: str = "oos",
    ref_composite: "pd.Series" = None,
) -> tuple:
    """(total_ret, sharpe, max_dd, turnover)"""
    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    f = composite.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)

    # 量化分位边界来源：
    # - oos: 使用当前段（通常是 OOS 段）自身分布做 qcut（原实现，可能轻微 look-ahead）
    # - is:  使用 ref_composite（通常为 IS 段）分布估计固定分位边界，再应用到当前段（无 OOS 分布信息）
    qs = (quantile_source or "oos").lower()
    if qs in ("is", "insample", "in_sample"):
        if ref_composite is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        # 注意：ref_composite 通常来自 IS 段，其索引与当前段（OOS）的 forward_return 不重叠；
        # 这里必须只使用 ref_composite 自身分布来估计分位边界，避免错误地得到空交集。
        ref_f = ref_composite.dropna().astype(float)
        if len(ref_f) < n_quantiles * 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        # 固定阈值：按分位点切出 n_quantiles 组，只用内部边界
        probs = [i / n_quantiles for i in range(1, n_quantiles)]
        edges = np.quantile(ref_f.values, probs)
        # position: top quantile -> +1, bottom quantile -> -1, else 0
        low_edge = float(edges[0]) if len(edges) else float("nan")
        high_edge = float(edges[-1]) if len(edges) else float("nan")
        position = (f >= high_edge).astype(float) - (f <= low_edge).astype(float)
    else:
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


def _segment_total_returns(
    composite: pd.Series,
    forward_return: pd.Series,
    forward_period: int,
    n_quantiles: int,
    fee_bps: int,
    n_segments: int = 2,
    *,
    quantile_source: str = "oos",
    ref_composite: "pd.Series" = None,
) -> list:
    """
    粗略的多段 OOS 稳健性：将 OOS 时间轴按索引均匀切成 n 段，
    对每一段分别调用 _backtest_composite，返回各段 total_ret 列表。
    """
    if n_segments <= 1:
        ret, _, _, _ = _backtest_composite(
            composite,
            forward_return,
            forward_period,
            n_quantiles,
            fee_bps,
            quantile_source=quantile_source,
            ref_composite=ref_composite,
        )
        return [ret]

    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2 * n_segments:
        # 数据太少时直接退化为整体一段
        ret, _, _, _ = _backtest_composite(
            composite,
            forward_return,
            forward_period,
            n_quantiles,
            fee_bps,
            quantile_source=quantile_source,
            ref_composite=ref_composite,
        )
        return [ret]

    segments = np.array_split(common, n_segments)
    rets = []
    for seg in segments:
        if len(seg) == 0:
            rets.append(float("nan"))
            continue
        comp_seg = composite.loc[seg]
        fwd_seg = forward_return.loc[seg]
        r, _, _, _ = _backtest_composite(
            comp_seg,
            fwd_seg,
            forward_period,
            n_quantiles,
            fee_bps,
            quantile_source=quantile_source,
            ref_composite=ref_composite,
        )
        rets.append(r)
    return rets


def run_eval(
    top_n: int,
    n_quantiles: int,
    weight_mode: str,
    oos_metric: str,
    oos_weight_5: float,
    fee_bps: int,
) -> dict:
    """执行一次评估，返回指标字典（用于写 TSV 或打印）。"""
    # 支持通过环境变量覆盖因子池 pkl 路径，便于切换到 best_exprs_robust.pkl 等不同因子集合
    expr_pkl_env = os.environ.get("GENETIC50_EXPR_PKL")
    if expr_pkl_env:
        pkl_path = Path(expr_pkl_env)
    else:
        pkl_path = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs.pkl"
    if not pkl_path.exists():
        return {"error": "best_exprs.pkl not found"}

    with open(pkl_path, "rb") as f:
        best_exprs = pickle.load(f)

    # 支持通过环境变量选择不同交易对，默认仍为 BTC_USDT
    symbol = os.environ.get("GENETIC50_SYMBOL", "BTC_USDT")
    data = load_ohlcv(symbol, "1m", days=90)
    if data.empty:
        return {"error": "No OHLCV data"}

    n_total = len(data)
    split = int(n_total * 60 / 90)
    data_is = data.iloc[:split].copy()
    data_oos = data.iloc[split:].copy()

    factor_names = []
    factor_values_is = []
    factor_values_oos = []
    ic_5_list = []
    ic_60_list = []
    robust_score_list = []

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
        robust = _robust_ic_score(f_is, fwd_5_is, fwd_60_is, n_segments=3, penalty_lambda=0.5)
        if np.isnan(ic60) or np.isinf(ic60):
            ic60 = 0.0
        if np.isnan(ic5) or np.isinf(ic5):
            ic5 = 0.0
        factor_names.append(f"genetic_{i}")
        factor_values_is.append(f_is)
        factor_values_oos.append(f_oos)
        ic_5_list.append(ic5)
        ic_60_list.append(ic60)
        robust_score_list.append(0.0 if np.isnan(robust) or np.isinf(robust) else robust)

    if not factor_names:
        return {"error": "No valid factors"}

    ic_5_abs = np.abs(ic_5_list)
    ic_60_abs = np.abs(ic_60_list)
    robust_arr = np.array(robust_score_list, dtype=float)

    # 选择因子排序标准：|IC_60| / |IC_5| / 0.4|IC5|+0.6|IC60| / robust
    # 默认仍为 |IC_60|（与原实现保持兼容）
    select_mode = os.environ.get("GENETIC50_SELECT_IC", "60").lower()
    if select_mode == "5":
        score = ic_5_abs
    elif select_mode in ("mix", "5_60", "0.4_0.6"):
        score = 0.4 * ic_5_abs + 0.6 * ic_60_abs
    elif select_mode in ("robust", "stable"):
        # 稳健 IC：segment 评分为负时视为弱因子
        score = np.maximum(robust_arr, 0.0)
    else:
        score = ic_60_abs

    order = np.argsort(score)[::-1]
    top_k = min(top_n, len(order))
    selected = order[:top_k]

    df_is = pd.DataFrame({factor_names[j]: factor_values_is[j] for j in selected}, index=data_is.index)
    df_oos = pd.DataFrame({factor_names[j]: factor_values_oos[j] for j in selected}, index=data_oos.index)
    ic_5_sel = [ic_5_list[j] for j in selected]
    ic_60_sel = [ic_60_list[j] for j in selected]
    score_sel = [score[j] for j in selected]

    # 去相关/去冗余：在 in-sample 上做相关性过滤，避免高度共线的因子重复进入
    # 使用简单的贪心：按当前列顺序依次加入，只要与已有列的相关系数都小于阈值
    corr_thresh = float(os.environ.get("GENETIC50_CORR_THRESH", "0.9"))
    if df_is.shape[1] > 1:
        corr = df_is.corr().abs()
        kept_cols = []
        kept_idx = []
        for idx, col in enumerate(df_is.columns):
            if not kept_cols:
                kept_cols.append(col)
                kept_idx.append(idx)
                continue
            ok = True
            for kc in kept_cols:
                if corr.loc[col, kc] >= corr_thresh:
                    ok = False
                    break
            if ok:
                kept_cols.append(col)
                kept_idx.append(idx)
        df_is = df_is[kept_cols]
        df_oos = df_oos[kept_cols]
        ic_5_sel = [ic_5_sel[i] for i in kept_idx]
        ic_60_sel = [ic_60_sel[i] for i in kept_idx]
        score_sel = [score_sel[i] for i in kept_idx]

    if weight_mode == "ic_weighted":
        # 与选择标准保持一致：按 score_sel 绝对值加权
        weights = np.array([abs(score_sel[i]) for i in range(len(score_sel))], dtype=float)
        weights = weights / (weights.sum() + 1e-10)
    else:
        weights = np.ones(len(score_sel)) / max(1, len(score_sel))

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
    if composite_oos is None:
        return {"error": "No composite"}
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

    # 分位边界来源：默认 oos（原实现）。设置 GENETIC50_QCUT_SOURCE=is 可避免使用 OOS 分布信息。
    qcut_source = os.environ.get("GENETIC50_QCUT_SOURCE", "oos").lower()

    in_ret_5, in_sh_5, _, _ = _backtest_composite(
        composite_is,
        fwd_5_is,
        5,
        n_quantiles,
        fee_bps,
        quantile_source="oos",
        ref_composite=None,
    )
    in_ret_60, in_sh_60, _, _ = _backtest_composite(
        composite_is,
        fwd_60_is,
        60,
        n_quantiles,
        fee_bps,
        quantile_source="oos",
        ref_composite=None,
    )

    oos_ret_5, oos_sh_5, oos_dd_5, oos_to_5 = _backtest_composite(
        composite_oos,
        fwd_5_oos,
        5,
        n_quantiles,
        fee_bps,
        quantile_source=qcut_source,
        ref_composite=composite_is if qcut_source in ("is", "insample", "in_sample") else None,
    )
    oos_ret_60, oos_sh_60, oos_dd_60, oos_to_60 = _backtest_composite(
        composite_oos,
        fwd_60_oos,
        60,
        n_quantiles,
        fee_bps,
        quantile_source=qcut_source,
        ref_composite=composite_is if qcut_source in ("is", "insample", "in_sample") else None,
    )

    # 多段 OOS：将 5p / 60p 的 OOS 分别切成两段，计算各段 total_ret，用于稳健性裁决
    seg5 = _segment_total_returns(
        composite_oos,
        fwd_5_oos,
        5,
        n_quantiles,
        fee_bps,
        n_segments=2,
        quantile_source=qcut_source,
        ref_composite=composite_is if qcut_source in ("is", "insample", "in_sample") else None,
    )
    seg60 = _segment_total_returns(
        composite_oos,
        fwd_60_oos,
        60,
        n_quantiles,
        fee_bps,
        n_segments=2,
        quantile_source=qcut_source,
        ref_composite=composite_is if qcut_source in ("is", "insample", "in_sample") else None,
    )

    if oos_metric == "5":
        oos_return_avg = oos_ret_5
        seg_avgs = seg5
    elif oos_metric == "60":
        oos_return_avg = oos_ret_60
        seg_avgs = seg60
    elif oos_metric == "weighted":
        oos_return_avg = oos_weight_5 * oos_ret_5 + (1 - oos_weight_5) * oos_ret_60
        seg_avgs = [oos_weight_5 * s5 + (1 - oos_weight_5) * s60 for s5, s60 in zip(seg5, seg60)]
    else:
        oos_return_avg = (oos_ret_5 + oos_ret_60) / 2.0
        seg_avgs = [(s5 + s60) / 2.0 for s5, s60 in zip(seg5, seg60)]

    def _nan0(x):
        return 0.0 if (x != x or np.isnan(x)) else float(x)

    # 稳健性指标：多段中最差的一段收益
    worst_seg = float("nan")
    if seg_avgs:
        try:
            worst_seg = float(np.nanmin(seg_avgs))
        except Exception:
            worst_seg = float("nan")

    return {
        "in_sample_return_5": _nan0(in_ret_5),
        "in_sample_return_60": _nan0(in_ret_60),
        "oos_return_5": _nan0(oos_ret_5),
        "oos_return_60": _nan0(oos_ret_60),
        "oos_return_avg": _nan0(oos_return_avg),
        "oos_sharpe_5": _nan0(oos_sh_5),
        "oos_sharpe_60": _nan0(oos_sh_60),
        "oos_max_drawdown_5": _nan0(oos_dd_5),
        "oos_max_drawdown_60": _nan0(oos_dd_60),
        "oos_return_avg_worst_segment": _nan0(worst_seg),
        "oos_turnover_5": _nan0(oos_to_5),
        "oos_turnover_60": _nan0(oos_to_60),
    }


def main():
    parser = argparse.ArgumentParser(description="遗传因子组合 OOS 评估（同 btc_oos_eval 输出格式）")
    parser.add_argument("--top_n", type=int, default=None, help="Top N factors by |IC_60|")
    parser.add_argument("--n_quantiles", type=int, default=None)
    parser.add_argument("--weight_mode", type=str, default=None, choices=["equal", "ic_weighted"])
    parser.add_argument("--oos_metric", type=str, default=None, choices=["avg", "5", "60", "weighted"])
    parser.add_argument("--oos_weight_5", type=float, default=None)
    parser.add_argument("--fee_bps", type=int, default=None)
    args = parser.parse_args()

    top_n = args.top_n if args.top_n is not None else TOP_N_FACTORS
    n_quantiles = args.n_quantiles if args.n_quantiles is not None else N_QUANTILES
    weight_mode = args.weight_mode if args.weight_mode is not None else WEIGHT_MODE
    oos_metric = args.oos_metric if args.oos_metric is not None else OOS_METRIC
    oos_weight_5 = args.oos_weight_5 if args.oos_weight_5 is not None else OOS_WEIGHT_5
    fee_bps = args.fee_bps if args.fee_bps is not None else FEE_BPS

    res = run_eval(top_n, n_quantiles, weight_mode, oos_metric, oos_weight_5, fee_bps)
    if "error" in res:
        print("ERROR:", res["error"], file=sys.stderr)
        sys.exit(1)

    print("in_sample_return_5:", res["in_sample_return_5"])
    print("in_sample_return_60:", res["in_sample_return_60"])
    print("oos_return_5:", res["oos_return_5"])
    print("oos_return_60:", res["oos_return_60"])
    print("oos_return_avg:", res["oos_return_avg"])
    print("oos_sharpe_5:", res["oos_sharpe_5"])
    print("oos_sharpe_60:", res["oos_sharpe_60"])
    print("oos_max_drawdown_5:", res["oos_max_drawdown_5"])
    print("oos_max_drawdown_60:", res["oos_max_drawdown_60"])
    print("oos_turnover_5:", res["oos_turnover_5"])
    print("oos_turnover_60:", res["oos_turnover_60"])
    print("fee_bps:", fee_bps)


if __name__ == "__main__":
    main()
