#!/usr/bin/env python3
"""
单因子表现分析：在本地 OHLCV 数据上计算因子、检验 IC/分位数收益/换手率，并生成报告。
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 仅使用 OHLCV 即可计算的因子类别，避免订单簿/跨市场数据缺失
from factors import FactorPool
from factors.base_factor import FactorCategory


def compute_single_factor_backtest_ret(
    factor_series: pd.Series,
    forward_return: pd.Series,
    ic: float,
    forward_period: int,
    n_quantiles: int = 5,
) -> float:
    """
    单因子多空回测收益：按分位数做多 Q5、做空 Q1（IC>=0），或反向（IC<0 时做多 Q1、做空 Q5）。
    使用非重叠区间计算累计收益。
    """
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return float("nan")
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    # IC 为负则反向：用 -f 做分位数，等价于原因子低分组做多、高分组做空
    if ic < 0:
        f = -f
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    # 1 = 最高分位(Q5) 做多, 0 = 最低分位(Q1) 做空
    q_max = q.max()
    q_min = q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return float("nan")
    long_signal = (q == q_max).astype(float)
    short_signal = (q == q_min).astype(float)
    position = long_signal - short_signal  # 1, -1, 0
    # 非重叠区间：每 forward_period 取一个点
    indices = common[::forward_period]
    if len(indices) < 2:
        return float("nan")
    rets = []
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos = position.loc[i]
        fr = r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        rets.append(pos * fr)
    if not rets:
        return float("nan")
    # 累计收益 (1+r1)(1+r2)... - 1
    cum = 1.0
    for r_t in rets:
        cum *= 1.0 + r_t
    return cum - 1.0


def get_pnl_curve(
    factor_series: pd.Series,
    forward_return: pd.Series,
    ic: float,
    forward_period: int,
    n_quantiles: int = 5,
    timestamps: Optional[pd.Series] = None,
):
    """
    返回非重叠区间的多空累计权益曲线（从 1 开始），用于绘图。
    若提供 timestamps（与 factor_series 同索引），返回 (dates, equity)；否则 (period_indices, equity)。
    """
    common = factor_series.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < n_quantiles * 2:
        return np.array([]), np.array([])
    f = factor_series.loc[common].astype(float)
    r = forward_return.loc[common].astype(float)
    if ic < 0:
        f = -f
    q = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    q_max = q.max()
    q_min = q.min()
    if pd.isna(q_max) or pd.isna(q_min):
        return np.array([]), np.array([])
    long_signal = (q == q_max).astype(float)
    short_signal = (q == q_min).astype(float)
    position = long_signal - short_signal
    indices = common[::forward_period]
    if len(indices) < 2:
        return np.array([]), np.array([])
    rets = []
    index_used = []
    for i in indices:
        if i not in position.index or i not in r.index:
            continue
        pos = position.loc[i]
        fr = r.loc[i]
        if pd.isna(fr) or pd.isna(pos):
            continue
        rets.append(pos * fr)
        index_used.append(i)
    if not rets:
        return np.array([]), np.array([])
    cum = 1.0
    curve = [1.0]
    for r_t in rets:
        cum *= 1.0 + r_t
        curve.append(cum)
    y_out = np.array(curve)
    if timestamps is not None and len(index_used) > 0:
        try:
            # len(curve) = len(index_used) + 1; need one date per curve point
            first_idx = index_used[0] - forward_period
            if first_idx not in timestamps.index or first_idx < 0:
                first_idx = index_used[0]
            rest_ts = timestamps.reindex(index_used).dropna()
            if len(rest_ts) == len(index_used):
                first_ts = pd.to_datetime(timestamps.loc[first_idx])
                x_dates = [first_ts] + pd.to_datetime(rest_ts).tolist()
                x_out = np.array(x_dates)
                if len(x_out) == len(y_out):
                    return x_out, y_out
        except Exception:
            pass
    return np.arange(len(curve)), y_out


def build_equal_weight_composite(
    factor_values: pd.DataFrame,
    ic_by_name: pd.Series,
) -> pd.Series:
    """
    将所有因子线性等权合成：IC<0 的因子取反后，逐因子截面 z-score 标准化再等权平均。
    """
    out = None
    for col in factor_values.columns:
        ic = ic_by_name.get(col, 0.0)
        if pd.isna(ic):
            ic = 0.0
        f = factor_values[col].astype(float)
        if ic < 0:
            f = -f
        # 截面 z-score（按时间序列标准化）
        mu = f.mean()
        std = f.std()
        if std is None or pd.isna(std) or std < 1e-10:
            continue
        z = (f - mu) / std
        if out is None:
            out = z.copy()
        else:
            out = out.add(z, fill_value=0)
    if out is None:
        return pd.Series(dtype=float)
    n = len(factor_values.columns)
    return out / n


def load_ohlcv_from_csv(symbol: str = "BTC_USDT", timeframe: str = "1m", days: int = 30) -> pd.DataFrame:
    """从 data/csv 加载 OHLCV，列名统一为 open, high, low, close, volume。"""
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


def main():
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Factor performance report and linear composite PnL curves")
    parser.add_argument("--days", type=int, default=30, help="Days of 1m data to load from data/csv")
    parser.add_argument("--suffix", type=str, default="", help="Output suffix for files, e.g. _3m to avoid overwriting")
    args = parser.parse_args()
    days = args.days
    suffix = ("_" + args.suffix.strip("_")) if args.suffix.strip() else ""

    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)

    # 1) 数据：优先 BTC
    data = load_ohlcv_from_csv("BTC_USDT", "1m", days=days)
    if data.empty:
        for sym in ["ETH_USDT", "SOL_USDT"]:
            data = load_ohlcv_from_csv(sym.replace("_", "_"), "1m", days=days)
            if not data.empty:
                break
    if data.empty:
        print("未找到 data/csv 下 OHLCV，请先运行 download_okx_klines.py")
        return

    # 2) 未来收益：与预测周期一致，使用 5 与 60 两种
    forward_periods = [5, 60]
    results_by_period = {}
    pool = FactorPool()

    # 仅技术、量价、波动率（不需订单簿/跨市场）
    categories = [
        FactorCategory.TECHNICAL,
        FactorCategory.VOLUME,
        FactorCategory.VOLATILITY,
    ]
    factor_names = []
    for cat in categories:
        factor_names.extend(pool.list_factors(cat))

    if not factor_names:
        factor_names = list(pool.factors.keys())[:40]

    print(f"数据: {len(data)} 根 K 线, 因子数: {len(factor_names)}")
    print("计算因子...")
    factor_values = pool.compute(data, factor_names=factor_names, verbose=False)
    # 丢弃全 NaN 列
    factor_values = factor_values.dropna(axis=1, how="all")
    valid_factors = list(factor_values.columns)
    print(f"有效因子: {len(valid_factors)}")

    for fp in forward_periods:
        forward_return = data["close"].astype(float).pct_change(fp).shift(-fp)
        print(f"检验未来{fp}期收益...")
        test_df = pool.test(
            factor_values[valid_factors],
            forward_return,
            n_quantiles=5,
            verbose=False,
        )
        # 单因子回测 ret：IC<0 的因子按反向（做多 Q1、做空 Q5）参与回测
        backtest_rets = []
        for _, row in test_df.iterrows():
            name = row["name"]
            ic_val = row["ic"]
            if name not in factor_values.columns:
                backtest_rets.append(float("nan"))
                continue
            ret = compute_single_factor_backtest_ret(
                factor_values[name],
                forward_return,
                ic_val,
                fp,
                n_quantiles=5,
            )
            backtest_rets.append(ret)
        test_df["backtest_ret"] = backtest_rets
        results_by_period[fp] = {
            "test_df": test_df,
            "test_results": dict(pool.test_results),
        }

    # 2.5) 等权复合因子回测：所有因子线性等权相加（IC<0 取反后 z-score 等权平均）
    ic_by_name_5 = results_by_period[5]["test_df"].set_index("name")["ic"]
    ic_by_name_60 = results_by_period[60]["test_df"].set_index("name")["ic"]
    composite_5 = build_equal_weight_composite(factor_values[valid_factors], ic_by_name_5)
    composite_60 = build_equal_weight_composite(factor_values[valid_factors], ic_by_name_60)
    forward_ret_5 = data["close"].astype(float).pct_change(5).shift(-5)
    forward_ret_60 = data["close"].astype(float).pct_change(60).shift(-60)
    composite_backtest_ret_5 = compute_single_factor_backtest_ret(
        composite_5, forward_ret_5, ic=1.0, forward_period=5, n_quantiles=5
    )
    composite_backtest_ret_60 = compute_single_factor_backtest_ret(
        composite_60, forward_ret_60, ic=1.0, forward_period=60, n_quantiles=5
    )
    print(f"等权复合因子 5期回测ret: {composite_backtest_ret_5*100:.2f}%")
    print(f"等权复合因子 60期回测ret: {composite_backtest_ret_60*100:.2f}%")

    # 2.6) 绘制 5 期 / 60 期多空累计 PnL 曲线图（横轴交易日期，英文标注）
    ts_series = data["timestamp"] if "timestamp" in data.columns else None
    x5, y5 = get_pnl_curve(
        composite_5, forward_ret_5, ic=1.0, forward_period=5, n_quantiles=5, timestamps=ts_series
    )
    x60, y60 = get_pnl_curve(
        composite_60, forward_ret_60, ic=1.0, forward_period=60, n_quantiles=5, timestamps=ts_series
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        for period, x, y, label, fname in [
            (5, x5, y5, "5min prediction", f"linear_model_pnl_curve_5{suffix}.png"),
            (60, x60, y60, "60min prediction", f"linear_model_pnl_curve_60{suffix}.png"),
        ]:
            if len(x) == 0 or len(y) == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            total_ret_pct = (y[-1] - 1) * 100
            ax.plot(x, y, label=f"Total return: {total_ret_pct:.2f}%", color="C0", lw=1.5)
            ax.axhline(1.0, color="gray", ls="--", alpha=0.7)
            ax.set_xlabel("Trading date")
            ax.set_ylabel("Cumulative equity (1 = initial)")
            ax.set_title(f"Equal-weight linear composite · {label}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            if np.issubdtype(x.dtype, np.datetime64) or (hasattr(x, "dtype") and pd.api.types.is_datetime64_any_dtype(x)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()
            fig.tight_layout()
            pnl_img = PROJECT_ROOT / "reports" / fname
            fig.savefig(pnl_img, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"PnL curve saved: {pnl_img}")
    except Exception as e:
        print(f"Plot PnL curves failed: {e}")

    # 3) 生成报告
    report_path = PROJECT_ROOT / "reports" / f"factor_performance_report{suffix}.md"
    lines = [
        "# 单因子表现报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 一、数据与参数",
        "",
        f"- **数据源**: `data/csv` 本地 CSV",
        f"- **样本量**: {len(data)} 根 1 分钟 K 线",
        f"- **因子数量**: 参与检验 {len(valid_factors)} 个（技术/量价/波动率）",
        f"- **未来收益**: 使用 5 期与 60 期 forward return",
        f"- **分位数**: 5 分位（Q1 最低因子值 ~ Q5 最高因子值）",
        "- **方向**: IC<0 的因子在回测中按**反向**处理（做多 Q1、做空 Q5），再计算单因子回测 ret",
        "- **单因子回测 ret**: 非重叠区间多空收益，(1+r1)(1+r2)...-1",
        "",
        "### 等权复合因子回测结果",
        "",
        "将所有因子线性等权合成（IC<0 的因子取反后截面 z-score 标准化再等权平均），按同一多空规则回测：",
        "",
        f"- **未来 5 期** 多空累计收益: **{composite_backtest_ret_5*100:.2f}%**",
        f"- **未来 60 期** 多空累计收益: **{composite_backtest_ret_60*100:.2f}%**",
        "",
        "---",
        "",
    ]

    section_labels = ["二", "三", "四", "五"]
    for idx, fp in enumerate(forward_periods):
        res = results_by_period[fp]
        test_df = res["test_df"]
        if test_df.empty:
            continue
        test_df = test_df.sort_values("abs_ic", ascending=False).reset_index(drop=True)
        sec = section_labels[idx]
        lines.append(f"## {sec}、未来 {fp} 期收益下的因子表现")
        lines.append("")
        lines.append(f"### {sec}.1 IC 与显著性（按 |IC| 降序，含单因子回测 ret）")
        lines.append("")
        lines.append("| 因子名称 | IC | IC_pvalue | 显著(5%) | 多空收益(约) | 换手率(均值) | 单因子回测ret |")
        lines.append("|----------|-----|------------|----------|----------------|----------------|----------------|")
        test_results = res["test_results"]
        for _, row in test_df.head(30).iterrows():
            name = row["name"]
            ic = row["ic"]
            pv = row["ic_pvalue"]
            sig = "是" if row.get("significant", pv < 0.05) else "否"
            tr = row.get("turnover_mean", 0)
            ls_ret = ""
            if name in test_results and hasattr(test_results[name], "quantile_returns"):
                qr = test_results[name].quantile_returns
                if not qr.empty and "mean_return" in qr.columns:
                    q1 = qr.iloc[0]["mean_return"]
                    q5 = qr.iloc[-1]["mean_return"]
                    ls_ret = f"{(q5 - q1) * 100:.3f}%"
            bt_ret = row.get("backtest_ret", float("nan"))
            bt_str = f"{bt_ret * 100:.2f}%" if pd.notna(bt_ret) else "-"
            lines.append(f"| {name} | {ic:.4f} | {pv:.4f} | {sig} | {ls_ret} | {tr:.4f} | {bt_str} |")
        lines.append("")
        lines.append(f"### {sec}.2 综合排名（IC×0.4 + Sharpe×0.4 + 低换手×0.2）")
        lines.append("")
        try:
            ranked = pool.rank_factors(test_df, ic_weight=0.4, sharpe_weight=0.4, turnover_weight=0.2)
            for i, (_, row) in enumerate(ranked.head(15).iterrows(), 1):
                lines.append(f"{i}. **{row['name']}**  score={row['score']:.4f}  IC={row['ic']:.4f}  sharpe={row['sharpe']:.4f}  turnover={row['turnover_mean']:.4f}")
            lines.append("")
        except Exception:
            for i, (_, row) in enumerate(test_df.head(15).iterrows(), 1):
                lines.append(f"{i}. **{row['name']}**  IC={row['ic']:.4f}  p_value={row['ic_pvalue']:.4f}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # 所有单因子回测 ret 汇总（两周期）
    lines.append("## 四、所有单因子回测 ret 汇总")
    lines.append("")
    lines.append("（IC<0 的因子已按反向参与回测；ret 为非重叠区间多空累计收益。）")
    lines.append("")
    all_ret_rows = []
    for fp in forward_periods:
        res = results_by_period.get(fp)
        if res is None or res["test_df"].empty:
            continue
        df = res["test_df"].copy()
        df["forward_period"] = fp
        all_ret_rows.append(df[["name", "ic", "forward_period", "backtest_ret"]])
    if all_ret_rows:
        import io
        combined = pd.concat(all_ret_rows, ignore_index=True)
        # 透视：每行一个因子，列为 ret_5, ret_60
        wide = combined.pivot_table(index="name", columns="forward_period", values="backtest_ret")
        wide.columns = [f"ret_{int(c)}" for c in wide.columns]
        wide = wide.reset_index()
        wide = wide.sort_values(by=wide.columns[-1], ascending=False)
        lines.append("| 因子名称 | " + " | ".join(wide.columns[1:]) + " |")
        lines.append("|" + "----------|" * len(wide.columns) + "")
        for _, row in wide.iterrows():
            vals = [row["name"]]
            for c in wide.columns[1:]:
                v = row[c]
                vals.append(f"{v * 100:.2f}%" if pd.notna(v) else "-")
            lines.append("| " + " | ".join(str(x) for x in vals) + " |")
        lines.append("")
        out_csv = PROJECT_ROOT / "reports" / f"factor_backtest_ret{suffix}.csv"
        wide.to_csv(out_csv, index=False, encoding="utf-8-sig")
        lines.append(f"完整列表已导出: `reports/factor_backtest_ret{suffix}.csv`")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend([
        "## 五、分类汇总",
        "",
    ])
    for fp in forward_periods:
        res = results_by_period[fp]
        test_df = res["test_df"]
        if test_df.empty:
            continue
        sig = test_df[test_df["ic_pvalue"] < 0.05]
        lines.append(f"- **未来{fp}期**: 显著因子数( p<0.05 ) = {len(sig)} / {len(test_df)}")
        if not sig.empty:
            best_ic = sig.loc[sig["abs_ic"].idxmax()]
            lines.append(f"  - |IC| 最大: {best_ic['name']} (IC={best_ic['ic']:.4f})")
        lines.append("")
    lines.extend([
        "## 六、结论与建议",
        "",
        "1. **IC 与方向**: |IC| 越大且 p 值越小，因子对未来收益的预测能力越强；多空收益列为 Q5−Q1 分组收益差。",
        "2. **换手率**: 换手率过高会增加交易成本，实盘需结合成本评估。",
        "3. **周期**: 5 期与 60 期分别对应短期与中期预测，可按策略持仓周期选用。",
        f"4. **单因子回测 ret**: IC 为负的因子已按反向（做多 Q1、做空 Q5）计算回测 ret，所有单因子 ret 见第四节及 `reports/factor_backtest_ret{suffix}.csv`。",
        "5. **后续**: 可对 Top 因子做组合、中性化或纳入模型特征。",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    main()
