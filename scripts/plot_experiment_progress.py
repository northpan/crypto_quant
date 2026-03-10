#!/usr/bin/env python3
"""
Plot experiment progress chart similar to karpathy/autoresearch progress.png.
- X-axis: experiment number (1..N)
- Y-axis: best OOS return so far (step function, updates only when we keep)
- Green markers: keep, Red markers: discard
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_results(tsv_path: Path, max_experiments: Optional[int] = None) -> pd.DataFrame:
    """Load results.tsv, deduplicate identical rows, optionally limit to first N experiments."""
    df = pd.read_csv(tsv_path, sep="\t")
    # Deduplicate fully identical rows (e.g. rows 20-21 in results.tsv)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df["exp_idx"] = np.arange(1, len(df) + 1)
    if max_experiments is not None:
        df = df.head(max_experiments)
    return df


def compute_best_so_far(df: pd.DataFrame) -> np.ndarray:
    """Best OOS return so far at each experiment (only updates when status=keep)."""
    best = np.full(len(df), np.nan)
    running_best = -np.inf
    for i in range(len(df)):
        if df.iloc[i]["status"] == "keep":
            running_best = max(running_best, df.iloc[i]["oos_return_avg"])
        best[i] = running_best
    # Fill initial NaNs with first valid best
    first_valid = np.where(~np.isnan(best))[0]
    if len(first_valid) > 0:
        for i in range(first_valid[0]):
            best[i] = best[first_valid[0]]
    return best


def get_improvement_points(df: pd.DataFrame) -> List[Tuple[int, float, str]]:
    """Return list of (exp_idx, oos, description) for each keep that improved best OOS."""
    best = compute_best_so_far(df)
    running_best = -np.inf
    improvements = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row["status"] != "keep":
            continue
        oos = row["oos_return_avg"]
        if oos > running_best:
            running_best = oos
            exp_idx = int(row["exp_idx"])
            desc = str(row.get("description", "")).strip()
            improvements.append((exp_idx, oos, desc))
    return improvements


def shorten_description(desc: str, max_len: int = 45) -> str:
    """Shorten description for annotation, keep key info."""
    if len(desc) <= max_len:
        return desc
    # Try to cut at word boundary
    truncated = desc[: max_len - 2].rsplit(maxsplit=1)[0] if " " in desc[: max_len - 2] else desc[: max_len - 3]
    return truncated + ".."


def plot_progress_annotated(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Experiment Progress (annotated)",
) -> None:
    """Plot progress chart with annotations for each keep that improved best OOS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use Chinese font on macOS if available
    try:
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    best = compute_best_so_far(df)
    x = df["exp_idx"].values
    oos = df["oos_return_avg"].values
    keep_mask = (df["status"] == "keep").values
    improvements = get_improvement_points(df)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.step(x, best, where="post", color="C0", linewidth=2, label="Best OOS (kept)")
    ax.scatter(
        x[keep_mask], oos[keep_mask],
        color="green", s=50, zorder=5, label="Keep", alpha=0.8
    )
    ax.scatter(
        x[~keep_mask], oos[~keep_mask],
        color="red", s=28, zorder=4, label="Discard", alpha=0.6
    )

    # Annotate each improvement point: all above data point, stagger y to avoid overlap
    # exp positions: 1, 2, 8, 12, 16, 29 — use tiered y-offsets for nearby points
    y_offsets = [22, 38, 22, 38, 54, 22]  # alternate tiers above
    for k, (exp_idx, oos_val, desc) in enumerate(improvements):
        short = shorten_description(desc)
        ax.annotate(
            short,
            xy=(exp_idx, oos_val),
            xytext=(0, y_offsets[k % len(y_offsets)]),
            textcoords="offset points",
            fontsize=8,
            alpha=0.95,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.8, edgecolor="gray"),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("OOS Return (avg)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, x[-1] + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Annotated progress chart saved: {out_path}")


def plot_progress(df: pd.DataFrame, out_path: Path, title: str = "Experiment Progress") -> None:
    """Plot experiment progress chart (autoresearch style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = compute_best_so_far(df)
    x = df["exp_idx"].values
    oos = df["oos_return_avg"].values
    keep_mask = (df["status"] == "keep").values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Step line: best OOS so far (use step="post" for right-continuous)
    ax.step(x, best, where="post", color="C0", linewidth=2, label="Best OOS (kept)")

    # Scatter: each experiment
    ax.scatter(
        x[keep_mask], oos[keep_mask],
        color="green", s=50, zorder=5, label="Keep", alpha=0.8
    )
    ax.scatter(
        x[~keep_mask], oos[~keep_mask],
        color="red", s=30, zorder=4, label="Discard", alpha=0.6
    )

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("OOS Return (avg)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, x[-1] + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Progress chart saved: {out_path}")


def main():
    tsv_path = PROJECT_ROOT / "results.tsv"
    if not tsv_path.exists():
        print(f"Not found: {tsv_path}")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # All 39 experiments (after dedup we have 41 rows -> 39 unique exps by description pattern)
    df_all = load_results(tsv_path)
    plot_progress(
        df_all,
        REPORTS_DIR / "experiment_progress_39.png",
        title="Experiment Progress (39 experiments)"
    )

    # First 33 experiments (before the negative OOS batch)
    df_33 = load_results(tsv_path, max_experiments=33)
    plot_progress(
        df_33,
        REPORTS_DIR / "experiment_progress_33.png",
        title="Experiment Progress (first 33 experiments)"
    )

    # Annotated chart: each keep that improved best OOS, with reason/condition
    plot_progress_annotated(
        df_33,
        REPORTS_DIR / "experiment_progress_33_annotated.png",
        title="Experiment Progress (前33组)",
    )


if __name__ == "__main__":
    main()
