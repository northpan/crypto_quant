#!/usr/bin/env python3
"""
Per-symbol backtest: run the same strategy on each crypto separately,
then output per-coin PnL curves (images) and a metrics report (markdown + table).
All chart labels in English.
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reduce log noise during multiple backtest runs
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("main").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_symbols_from_csv(data_dir: Path, timeframe: str = "1m") -> list:
    """Discover symbols from data/csv/*_1m.csv (e.g. BTC_USDT, ETH_USDT)."""
    csv_dir = data_dir / "csv"
    if not csv_dir.exists():
        return []
    pattern = f"*_{timeframe}.csv"
    symbols = []
    for p in csv_dir.glob(pattern):
        stem = p.stem
        sym = stem.replace(f"_{timeframe}", "")
        if sym and "_" in sym:
            symbols.append(sym)
    return sorted(symbols)


async def run_per_symbol_backtest(
    start_date: str,
    end_date: str,
    strategy_params: dict = None,
    use_local_data_only: bool = True,
):
    """
    Run backtest once per symbol; return list of
    { symbol, equity_curve_df, performance_dict, trade_stats_dict }.
    """
    from main import CryptoQuantSystem

    system = CryptoQuantSystem()

    symbols = get_symbols_from_csv(PROJECT_ROOT / "data", "1m")
    if not symbols:
        log.warning("No *_1m.csv in data/csv; using default symbols.")
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

    results = []
    for sym in symbols:
        log.info("Backtesting symbol: %s ...", sym)
        try:
            out = await system.run_backtest(
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy_params or {},
                save_report=False,
                use_local_data_only=use_local_data_only,
                symbols_to_use=[sym],
            )
        except Exception as e:
            log.warning("Backtest failed for %s: %s", sym, e)
            continue
        equity_df = None
        if out.get("results") and out["results"].get("equity_curve") is not None:
            equity_df = out["results"]["equity_curve"]
        perf = out.get("performance") or {}
        trade_stats = out.get("trade_stats") or {}
        results.append({
            "symbol": sym,
            "equity_curve": equity_df,
            "performance": perf,
            "trade_stats": trade_stats,
        })
    return results


def plot_per_symbol_pnl_curves(results: list, report_dir: Path) -> list:
    """
    Plot PnL (equity) curves: one combined figure (all symbols) and one figure per symbol.
    Returns list of saved image paths. All labels in English.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    saved = []

    # 1) Combined: all symbols on one chart (normalized to start at 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        eq = r.get("equity_curve")
        if eq is None or eq.empty or "equity" not in eq.columns:
            continue
        ts = eq.get("timestamp")
        if ts is None or ts.empty:
            continue
        equity = eq["equity"].astype(float)
        if equity.iloc[0] and equity.iloc[0] != 0:
            equity = equity / float(equity.iloc[0])
        total_ret = (equity.iloc[-1] - 1) * 100 if len(equity) else 0
        ax.plot(
            pd.to_datetime(ts),
            equity,
            label=f"{r['symbol']} (total return: {total_ret:.2f}%)",
            lw=1.2,
        )
    ax.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative equity (1 = initial)")
    ax.set_title("Per-symbol backtest · PnL curves (normalized)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.tight_layout()
    combined_path = report_dir / "per_symbol_pnl_curves_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(combined_path)

    # 2) One figure per symbol
    for r in results:
        eq = r.get("equity_curve")
        if eq is None or eq.empty or "equity" not in eq.columns:
            continue
        sym = r["symbol"]
        ts = eq.get("timestamp")
        if ts is None or ts.empty:
            continue
        equity = eq["equity"].astype(float)
        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if equity.iloc[0] else 0
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(ts), equity, color="C0", lw=1.5)
        ax.axhline(equity.iloc[0], color="gray", ls="--", alpha=0.7)
        ax.set_xlabel("Trading date")
        ax.set_ylabel("Cumulative equity")
        ax.set_title(f"{sym} · Backtest PnL (total return: {total_ret:.2f}%)")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.tight_layout()
        single_path = report_dir / f"per_symbol_pnl_{sym}.png"
        fig.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(single_path)

    return saved


def write_metrics_report(results: list, report_dir: Path, start_date: str, end_date: str) -> Path:
    """Write markdown report with per-symbol metrics table (English labels)."""
    rows = []
    for r in results:
        sym = r["symbol"]
        perf = r.get("performance") or {}
        ts = r.get("trade_stats") or {}
        rows.append({
            "Symbol": sym,
            "Total Return (%)": round((perf.get("total_return") or 0) * 100, 2),
            "Annualized Return (%)": round((perf.get("annualized_return") or 0) * 100, 2),
            "Sharpe Ratio": round(perf.get("sharpe_ratio") or 0, 4),
            "Max Drawdown (%)": round((perf.get("max_drawdown") or 0) * 100, 2),
            "Calmar Ratio": round(perf.get("calmar_ratio") or 0, 4),
            "Total Trades": ts.get("total_trades", 0),
            "Win Rate (%)": round((ts.get("win_rate") or 0) * 100, 2),
            "Profit Factor": round(ts.get("profit_factor") or 0, 4),
            "Avg Holding (min)": round(ts.get("avg_holding_period") or 0, 1),
        })
    df = pd.DataFrame(rows)
    # Build markdown table without requiring tabulate
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    table = "\n".join([header, sep] + body)

    lines = [
        "# Per-Symbol Backtest Report",
        "",
        f"**Period:** {start_date} to {end_date}",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Metrics Summary",
        "",
        table,
        "",
        "## PnL Curves",
        "",
        "- **Combined (all symbols, normalized):** `reports/per_symbol_pnl_curves_combined.png`",
        "",
    ]
    for r in results:
        sym = r["symbol"]
        lines.append(f"- **{sym}:** `reports/per_symbol_pnl_{sym}.png`")
    lines.append("")

    report_path = report_dir / "per_symbol_backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


async def main():
    from datetime import datetime, timedelta, timezone

    # Default: last 30 days
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running per-symbol backtest (%s to %s) ...", start_date, end_date)
    results = await run_per_symbol_backtest(
        start_date=start_date,
        end_date=end_date,
        use_local_data_only=True,
    )
    if not results:
        log.error("No backtest results; check data/csv for *_1m.csv files.")
        return

    log.info("Plotting per-symbol PnL curves ...")
    saved_images = plot_per_symbol_pnl_curves(results, report_dir)
    for p in saved_images:
        log.info("Saved: %s", p)

    log.info("Writing metrics report ...")
    report_path = write_metrics_report(results, report_dir, start_date, end_date)
    log.info("Report: %s", report_path)

    # Also export CSV for metrics
    rows = []
    for r in results:
        row = {"symbol": r["symbol"]}
        row.update(r.get("performance") or {})
        for k, v in (r.get("trade_stats") or {}).items():
            row[f"trade_{k}"] = v
        rows.append(row)
    csv_path = report_dir / "per_symbol_backtest_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("Metrics CSV: %s", csv_path)


if __name__ == "__main__":
    asyncio.run(main())
