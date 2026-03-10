#!/usr/bin/env python3
"""
运行 50 轮遗传因子挖掘，汇总每轮最佳因子，并计算 50 个最佳因子等权组合的样本外回测（60 天 IS / 30 天 OOS，扣费）。
"""

import sys
import csv
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from factors.genetic.evaluator import evaluate_expression
from factors.genetic.genetic_factor import run_evolution
from factors.genetic.expression import eval_expr, expr_to_str

FEE_BPS = 10
N_QUANTILES = 5


def load_ohlcv(symbol: str = "BTC_USDT", timeframe: str = "1m", days: int = 90) -> pd.DataFrame:
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
    n_quantiles: int = N_QUANTILES,
    fee_bps: int = FEE_BPS,
) -> tuple:
    """复合因子多空回测（扣费）。返回 (total_ret, sharpe, max_dd, turnover)。"""
    common = composite.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    f = composite.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max, q_min = q.max(), q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return float("nan"), float("nan"), float("nan"), float("nan")
    long_signal = (q == q_max).astype(float)
    short_signal = (q == q_min).astype(float)
    position = long_signal - short_signal
    indices = common[::forward_period]
    if len(indices) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
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
        return float("nan"), float("nan"), float("nan"), float("nan")
    rets_arr = np.array(rets)
    cum = np.cumprod(1.0 + rets_arr)
    total_ret = cum[-1] - 1.0
    sharpe = np.mean(rets_arr) / (np.std(rets_arr) + 1e-10) * np.sqrt(252 * 24 * 60) if len(rets_arr) > 1 else 0.0
    max_dd = np.min(np.minimum.accumulate(cum) / cum - 1.0) if len(cum) > 0 else 0.0
    pos_changes = sum(1 for i in range(1, len(rets)) if rets[i] != rets[i-1])
    turnover = pos_changes / max(1, len(rets)) if rets else 0.0
    return total_ret, sharpe, max_dd, turnover


def main():
    n_rounds = 50
    n_best = 50
    days_total = 90
    days_is = 60
    days_oos = 30
    pop_size = 20
    generations = 10

    print("=== 1. 加载数据 ===")
    data_full = load_ohlcv("BTC_USDT", "1m", days=days_total)
    if data_full.empty:
        for sym in ["ETH_USDT", "SOL_USDT"]:
            data_full = load_ohlcv(sym.replace("_", "_"), "1m", days=days_total)
            if not data_full.empty:
                break
    if data_full.empty:
        print("未找到 OHLCV 数据")
        return

    n_rows = len(data_full)
    split_idx = int(n_rows * days_is / days_total)
    data_is = data_full.iloc[:split_idx].copy()
    data_oos = data_full.iloc[split_idx:].copy()
    print(f"总数据: {n_rows} 行, IS: {split_idx} 行 ({days_is}天), OOS: {n_rows - split_idx} 行 ({days_oos}天)")

    print("\n=== 2. 运行 50 轮遗传算法 ===")
    best_per_round = []

    def evaluate_fn(expr):
        return evaluate_expression(
            expr, data_full,
            forward_periods=[5, 60],
            n_quantiles=N_QUANTILES,
            fee_bps=FEE_BPS,
        )

    for r in range(n_rounds):
        rng = __import__("random").Random(r)
        results = run_evolution(
            evaluate_fn,
            population_size=pop_size,
            generations=generations,
            max_depth=4,
            elite_ratio=0.15,
            mutation_prob=0.55,
            crossover_prob=0.5,
            immigrant_ratio=0.2,
            alpha=0.3,
            beta=0.7,
            ret_weight_5=0.5,
            ret_weight_60=0.5,
            rng=rng,
        )
        best_expr, best_m, best_f = results[0]
        best_per_round.append((best_expr, best_m, best_f))
        if (r + 1) % 10 == 0:
            print(f"  完成 {r+1}/{n_rounds} 轮")

    print("\n=== 3. 汇总每轮最佳因子 ===")
    summary_dir = PROJECT_ROOT / "reports" / "genetic_50_rounds"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "best_per_round_summary.csv"
    best_exprs = []

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "expr_str", "ic_5", "ic_60", "ret_5", "ret_60", "fitness"])
        for r, (expr, m, fit) in enumerate(best_per_round):
            s = expr_to_str(expr)
            ic5 = m.get("ic_5", float("nan"))
            ic60 = m.get("ic_60", float("nan"))
            ret5 = m.get("ret_5", float("nan"))
            ret60 = m.get("ret_60", float("nan"))
            w.writerow([r, s, ic5, ic60, ret5, ret60, fit])
            best_exprs.append(expr)

    with open(summary_dir / "best_exprs.pkl", "wb") as f:
        pickle.dump(best_exprs, f)

    print(f"汇总已保存至 {summary_path}")

    print("\n=== 4. 等权组合回测 (60d IS / 30d OOS, 扣费) ===")
    factor_values_oos = []
    ic_5_is = []
    ic_60_is = []

    for expr in best_exprs:
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
        ic_5_is.append(ic5 if not (np.isnan(ic5) or np.isinf(ic5)) else 0.0)
        ic_60_is.append(ic60 if not (np.isnan(ic60) or np.isinf(ic60)) else 0.0)
        f = f_oos.astype(float)
        if ic60 < 0:
            f = -f
        mu, std = f.mean(), f.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f - mu) / std
        factor_values_oos.append(z)

    if not factor_values_oos:
        print("无有效因子可用")
        return

    composite = pd.concat(factor_values_oos, axis=1).mean(axis=1)
    n_factors_used = len(factor_values_oos)
    print(f"有效因子数: {n_factors_used}")

    fwd_5_oos = _forward_return(data_oos["close"], 5)
    fwd_60_oos = _forward_return(data_oos["close"], 60)

    ret_5, sharpe_5, dd_5, to_5 = _backtest_composite(composite, fwd_5_oos, 5, N_QUANTILES, FEE_BPS)
    ret_60, sharpe_60, dd_60, to_60 = _backtest_composite(composite, fwd_60_oos, 60, N_QUANTILES, FEE_BPS)

    report_path = summary_dir / "composite_backtest_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 50 轮遗传因子等权组合回测报告\n\n")
        f.write("## 参数\n")
        f.write(f"- 训练轮数: {n_rounds}\n")
        f.write(f"- 每轮最佳因子数: 1\n")
        f.write(f"- 样本内: {days_is} 天\n")
        f.write(f"- 样本外: {days_oos} 天\n")
        f.write(f"- 有效因子数: {n_factors_used}\n")
        f.write(f"- 手续费: {FEE_BPS} bps/side\n")
        f.write(f"- 分位数: {N_QUANTILES}\n\n")
        f.write("## 等权组合 OOS 表现\n")
        f.write("| 指标 | 5期 | 60期 |\n")
        f.write("|------|-----|------|\n")
        f.write(f"| 累计收益 | {ret_5*100:.2f}% | {ret_60*100:.2f}% |\n")
        f.write(f"| Sharpe | {sharpe_5:.4f} | {sharpe_60:.4f} |\n")
        f.write(f"| 最大回撤 | {dd_5*100:.2f}% | {dd_60*100:.2f}% |\n")
        f.write(f"| 换手率 | {to_5:.4f} | {to_60:.4f} |\n")
        f.write("\n## 每轮最佳因子摘要\n")
        f.write("| 轮次 | 表达式 | IC_5 | IC_60 | ret_5 | ret_60 | fitness |\n")
        f.write("|------|--------|------|-------|-------|--------|--------|\n")
        for r, (expr, m, fit) in enumerate(best_per_round):
            s = expr_to_str(expr)
            if len(s) > 60:
                s = s[:57] + "..."
            ic5 = m.get("ic_5", float("nan"))
            ic60 = m.get("ic_60", float("nan"))
            ret5 = m.get("ret_5", float("nan"))
            ret60 = m.get("ret_60", float("nan"))
            f.write(f"| {r} | {s} | {ic5:.4f} | {ic60:.4f} | {ret5:.4f} | {ret60:.4f} | {fit:.4f} |\n")

    print(f"\n报告已保存至 {report_path}")
    print("\n--- 等权组合 OOS 表现 (扣费) ---")
    print(f"  5期:   ret={ret_5*100:.2f}%  sharpe={sharpe_5:.4f}  max_dd={dd_5*100:.2f}%")
    print(f"  60期:  ret={ret_60*100:.2f}%  sharpe={sharpe_60:.4f}  max_dd={dd_60*100:.2f}%")


if __name__ == "__main__":
    main()
