#!/usr/bin/env python3
"""
绘制多品种 Genetic50 最优组合的 IS/OOS PnL 曲线（60 分钟周期），用于展示跨品种泛化能力。

读取：
- results_genetic50_BTC_USDT.tsv / ETH_USDT / SOL_USDT / BNB_USDT / XRP_USDT
- reports/genetic_50_rounds/best_exprs_robust.pkl

步骤：
- 对每个品种，从对应 TSV 中选出 oos_return_avg 最优实验，得到 top_n / n_quantiles / weight_mode
- 复用 genetic_oos_eval 中的因子选择与组合构建逻辑
- 在 IS / OOS 上用 forward_period=60 回测，得到逐期收益并累积为 PnL 曲线
- 将五个品种的 IS/OOS PnL 画在一张 PNG 中：每行一个品种，两条曲线（IS/OOS）
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import genetic_oos_eval as goe  # type: ignore


def _backtest_pnl_series(
    composite: pd.Series,
    forward_return: pd.Series,
    timestamps: pd.Series,
    forward_period: int,
    n_quantiles: int,
    fee_bps: int,
    start_value: float = 0.0,
    *,
    quantile_source: str = "oos",
    ref_composite: pd.Series = None,
) -> pd.Series:
    """
    类似 genetic_oos_eval._backtest_composite，但返回逐期累计 PnL 序列。
    索引为实际时间戳（每 forward_period 一次），并允许指定起始累计值。
    """
    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return pd.Series(dtype=float)
    f = composite.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    qs = (quantile_source or "oos").lower()
    if qs in ("is", "insample", "in_sample"):
        if ref_composite is None:
            return pd.Series(dtype=float)
        ref_f = ref_composite.dropna().astype(float)
        if len(ref_f) < n_quantiles * 2:
            return pd.Series(dtype=float)
        probs = [i / n_quantiles for i in range(1, n_quantiles)]
        edges = np.quantile(ref_f.values, probs)
        low_edge = float(edges[0]) if len(edges) else float("nan")
        high_edge = float(edges[-1]) if len(edges) else float("nan")
        position = (f >= high_edge).astype(float) - (f <= low_edge).astype(float)
    else:
        q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
        q_max, q_min = q.max(), q.min()
        if pd.isna(q_max) or pd.isna(q_min):
            return pd.Series(dtype=float)
        position = (q == q_max).astype(float) - (q == q_min).astype(float)
    indices = common[::forward_period]
    rets = []
    time_list = []
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
        # 使用对应的时间戳作为横轴
        if i in timestamps.index:
            time_list.append(timestamps.loc[i])
        else:
            time_list.append(i)
    if not rets:
        return pd.Series(dtype=float)
    rets_arr = np.array(rets)
    cum = np.cumprod(1.0 + rets_arr) - 1.0 + start_value
    return pd.Series(cum, index=time_list)


def _load_best_config(symbol: str) -> dict:
    """从 results_genetic50_{symbol}.tsv 中选出 oos_return_avg 最优实验配置。"""
    path = PROJECT_ROOT / f"results_genetic50_{symbol}.tsv"
    df = pd.read_csv(path, sep="\t")
    row = df.loc[df["oos_return_avg"].astype(float).idxmax()]
    return {
        "top_n": int(row["top_n"]),
        "n_quantiles": int(row["n_quantiles"]),
        "weight_mode": str(row["weight_mode"]),
    }


def build_composite_for_symbol(
    symbol: str,
    cfg: dict,
    exprs: list,
    forward_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    为指定品种构建 IS/OOS 组合因子，并返回：
    composite_is, composite_oos, fwd_60_is, fwd_60_oos
    """
    # 1. 加载数据并切分 IS/OOS
    data = goe.load_ohlcv(symbol, "1m", days=90)
    if data.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    n_total = len(data)
    split = int(n_total * 60 / 90)
    data_is = data.iloc[:split].copy()
    data_oos = data.iloc[split:].copy()

    # 2. 复用 genetic_oos_eval 的因子评估与选择逻辑
    factor_names = []
    factor_values_is = []
    factor_values_oos = []
    ic_5_list = []
    ic_60_list = []
    robust_score_list = []

    for i, expr in enumerate(exprs):
        try:
            f_is = goe.eval_expr(expr, data_is)  # type: ignore[attr-defined]
            f_oos = goe.eval_expr(expr, data_oos)  # type: ignore[attr-defined]
        except Exception:
            continue
        if f_oos is None or f_oos.dropna().empty or f_oos.std() < 1e-8:
            continue
        fwd_5_is = goe._forward_return(data_is["close"], 5)  # type: ignore[attr-defined]
        fwd_60_is = goe._forward_return(data_is["close"], 60)  # type: ignore[attr-defined]
        ic5 = goe._compute_ic(f_is, fwd_5_is)  # type: ignore[attr-defined]
        ic60 = goe._compute_ic(f_is, fwd_60_is)  # type: ignore[attr-defined]
        robust = goe._robust_ic_score(f_is, fwd_5_is, fwd_60_is, n_segments=3, penalty_lambda=0.5)  # type: ignore[attr-defined]
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
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    ic_5_abs = np.abs(ic_5_list)
    ic_60_abs = np.abs(ic_60_list)
    robust_arr = np.array(robust_score_list, dtype=float)

    # 默认使用 |IC_60| 排序（与主实验保持一致）
    score = ic_60_abs
    order = np.argsort(score)[::-1]
    top_k = min(cfg["top_n"], len(order))
    selected = order[:top_k]

    df_is = pd.DataFrame({factor_names[j]: factor_values_is[j] for j in selected}, index=data_is.index)
    df_oos = pd.DataFrame({factor_names[j]: factor_values_oos[j] for j in selected}, index=data_oos.index)
    ic_60_sel = [ic_60_list[j] for j in selected]
    score_sel = [score[j] for j in selected]

    # 去相关
    corr_thresh = float(0.9)
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
        ic_60_sel = [ic_60_sel[i] for i in kept_idx]
        score_sel = [score_sel[i] for i in kept_idx]

    # 权重
    if cfg["weight_mode"] == "ic_weighted":
        weights = np.array([abs(score_sel[i]) for i in range(len(score_sel))], dtype=float)
        weights = weights / (weights.sum() + 1e-10)
    else:
        weights = np.ones(len(score_sel)) / max(1, len(score_sel))

    # 构建组合（IS / OOS）
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
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
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

    fwd_is = goe._forward_return(data_is["close"], forward_period)  # type: ignore[attr-defined]
    fwd_oos = goe._forward_return(data_oos["close"], forward_period)  # type: ignore[attr-defined]

    ts_is = data_is["timestamp"]
    ts_oos = data_oos["timestamp"]

    return composite_is, composite_oos, fwd_is, fwd_oos, ts_is, ts_oos


def plot_multi_asset_pnl(
    *,
    forward_period: int,
    out_name: str,
    title_suffix: str,
    quantile_source: str,
):
    # 全局字体 & 风格（英文文本，高级感）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC", "Heiti SC", "STHeiti", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-darkgrid")

    symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT"]

    # 读取稳健因子池
    exprs_path = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs_robust.pkl"
    with open(exprs_path, "rb") as f:
        best_exprs = pickle.load(f)

    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 14), sharex=False)
    if len(symbols) == 1:
        axes = [axes]

    for ax, sym in zip(axes, symbols):
        cfg = _load_best_config(sym)
        comp_is, comp_oos, fwd_is, fwd_oos, ts_is, ts_oos = build_composite_for_symbol(
            sym, cfg, best_exprs, forward_period=forward_period
        )
        if comp_is.empty or comp_oos.empty:
            ax.text(0.5, 0.5, f"{sym} no valid composite", ha="center", va="center", transform=ax.transAxes)
            continue
        # IS 段 PnL
        pnl_is = _backtest_pnl_series(
            comp_is,
            fwd_is,
            ts_is,
            forward_period=forward_period,
            n_quantiles=cfg["n_quantiles"],
            fee_bps=10,
            start_value=0.0,
            quantile_source="oos",
            ref_composite=None,
        )
        # OOS 段 PnL，从 IS 末值开始连续衔接
        start_oos = float(pnl_is.iloc[-1]) if not pnl_is.empty else 0.0
        pnl_oos = _backtest_pnl_series(
            comp_oos,
            fwd_oos,
            ts_oos,
            forward_period=forward_period,
            n_quantiles=cfg["n_quantiles"],
            fee_bps=10,
            start_value=start_oos,
            quantile_source=quantile_source,
            ref_composite=comp_is if (quantile_source or "").lower() in ("is", "insample", "in_sample") else None,
        )

        label_suffix = f"{forward_period}m"
        ax.plot(pnl_is.index, pnl_is.values, label=f"IS PnL ({label_suffix})", color="#4C78A8", linewidth=1.6)
        ax.plot(pnl_oos.index, pnl_oos.values, label=f"OOS PnL ({label_suffix})", color="#F58518", linewidth=1.6)
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.8, linestyle="--")

        ax.set_ylabel("Cumulative Return")
        ax.set_title(
            f"{sym} Best IS/OOS PnL (forward={forward_period}m, top_n={cfg['top_n']}, "
            f"q={cfg['n_quantiles']}, w={cfg['weight_mode']})",
            fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Multi-Asset Genetic50 Best IS/OOS PnL ({title_suffix})", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = PROJECT_ROOT / "reports" / out_name
    fig.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


def main():
    # Generate NEW charts without overwriting yesterday's images
    # Old charts used OOS qcut (potential mild look-ahead): keep them as-is.
    # New charts use IS-based fixed quantile edges for OOS (no OOS distribution usage).
    plot_multi_asset_pnl(
        forward_period=60,
        out_name="multi_asset_best_pnl_60m_isqcut.png",
        title_suffix="60-minute forward (OOS quantiles from IS)",
        quantile_source="is",
    )
    plot_multi_asset_pnl(
        forward_period=5,
        out_name="multi_asset_best_pnl_5m_isqcut.png",
        title_suffix="5-minute forward (OOS quantiles from IS)",
        quantile_source="is",
    )


if __name__ == "__main__":
    main()

