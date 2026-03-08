#!/usr/bin/env python3
"""
运行「三币共用一个组合」回测（每开一仓用当时现金的 20%，最多 3 仓，总敞口≤60%），
生成权益曲线图与 Markdown 回测报告。
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.getLogger("backtest").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def run_combined_backtest_and_report():
    from datetime import datetime, timezone, timedelta
    from main import CryptoQuantSystem

    # 默认回测区间：过去约 30 天
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    system = CryptoQuantSystem()
    # 不传 symbols_to_use => 使用 trading_config.symbols[:5]，即多币共用同一组合
    log.info("Running combined portfolio backtest (%s to %s) ...", start_date, end_date)
    result = await system.run_backtest(
        start_date=start_date,
        end_date=end_date,
        strategy_params={
            "position_pct": 0.2,
            "max_positions": 3,
            "max_exposure_pct": 0.6,
        },
        save_report=False,
        use_local_data_only=True,
    )

    if not result.get("results"):
        log.error("No backtest results (no data?).")
        return

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    equity_df = result["results"].get("equity_curve")
    perf = result.get("performance") or {}
    trade_stats = result.get("trade_stats") or {}

    # 1) 权益曲线图
    if equity_df is not None and not equity_df.empty and "equity" in equity_df.columns:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ts = equity_df["timestamp"] if "timestamp" in equity_df.columns else equity_df.index
        equity = equity_df["equity"].astype(float)
        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if equity.iloc[0] else 0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(ts), equity, color="C0", lw=1.5)
        ax.axhline(equity.iloc[0], color="gray", ls="--", alpha=0.7)
        ax.set_xlabel("Trading date")
        ax.set_ylabel("Cumulative equity")
        ax.set_title(
            "Combined portfolio (3 coins, 20% per position, max 3 positions, 60% cap) · "
            f"Total return: {total_ret:.2f}%"
        )
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.tight_layout()
        curve_path = report_dir / "combined_portfolio_pnl_curve.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", curve_path)

    # 2) Markdown 报告
    lines = [
        "# Combined Portfolio Backtest Report",
        "",
        "**Strategy:** Multi-symbol shared portfolio.",
        "- 20% of current cash per new position.",
        "- Max 3 open positions (across symbols).",
        "- Total exposure ≤ 60% of equity.",
        "",
        f"**Period:** {start_date} to {end_date}",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Return (%) | {(perf.get('total_return') or 0) * 100:.2f} |",
        f"| Annualized Return (%) | {(perf.get('annualized_return') or 0) * 100:.2f} |",
        f"| Sharpe Ratio | {perf.get('sharpe_ratio') or 0:.4f} |",
        f"| Max Drawdown (%) | {(perf.get('max_drawdown') or 0) * 100:.2f} |",
        f"| Calmar Ratio | {perf.get('calmar_ratio') or 0:.4f} |",
        f"| Volatility (ann.) | {(perf.get('annualized_volatility') or 0) * 100:.2f}% |",
        "",
        "## Trade Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Trades | {trade_stats.get('total_trades', 0)} |",
        f"| Winning Trades | {trade_stats.get('winning_trades', 0)} |",
        f"| Losing Trades | {trade_stats.get('losing_trades', 0)} |",
        f"| Win Rate (%) | {(trade_stats.get('win_rate') or 0) * 100:.2f} |",
        f"| Profit Factor | {trade_stats.get('profit_factor') or 0:.4f} |",
        f"| Total PnL | {trade_stats.get('total_pnl') or 0:.2f} |",
        f"| Avg Holding Period | {trade_stats.get('avg_holding_period') or 0:.2f} |",
        f"| Max Consecutive Losses | {trade_stats.get('max_consecutive_losses', 0)} |",
        "",
        "## Equity Curve",
        "",
        "![Combined portfolio PnL](combined_portfolio_pnl_curve.png)",
        "",
    ]
    report_path = report_dir / "combined_backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    asyncio.run(run_combined_backtest_and_report())
