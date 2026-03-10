#!/usr/bin/env python3
"""
Autoresearch loop for Genetic50 composite configs.

- Keeps a best-so-far configuration and explores local mutations.
- Applies explicit keep/discard rules (with tie-breaks).
- Writes an auditable TSV (seed, exprs hash, data_end_ts, runner_version, etc.).

Output: results_genetic50.tsv (tab-separated)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import genetic_oos_eval


RESULTS_PATH = PROJECT_ROOT / "results_genetic50.tsv"

# 允许通过环境变量覆盖因子池 pkl 路径和交易对，便于多币种 / 不同因子池实验
_expr_pkl_env = os.environ.get("GENETIC50_EXPR_PKL")
if _expr_pkl_env:
    BEST_EXPRS_PATH = Path(_expr_pkl_env)
    if not BEST_EXPRS_PATH.is_absolute():
        BEST_EXPRS_PATH = PROJECT_ROOT / BEST_EXPRS_PATH
else:
    BEST_EXPRS_PATH = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs.pkl"

SYMBOL = os.environ.get("GENETIC50_SYMBOL", "BTC_USDT")
DATA_PATH = PROJECT_ROOT / "data" / "csv" / f"{SYMBOL}_1m.csv"


RUNNER_VERSION = "autoresearch_genetic50_v2"

# Decision thresholds
EPS_RETURN = 0.005  # 0.5%


@dataclass(frozen=True)
class Config:
    top_n: int
    n_quantiles: int
    weight_mode: str  # equal | ic_weighted
    oos_metric: str  # avg | 5 | 60 | weighted
    oos_weight_5: float


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_data_end_ts(path: Path) -> str:
    # Read only timestamp column to avoid heavy load
    df = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"]).max()
    if pd.isna(ts):
        return ""
    # ISO string without timezone assumptions
    return ts.to_pydatetime().replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def _score_pack(res: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Return (oos_return_avg, oos_sharpe_avg, oos_max_dd_avg, oos_turnover_avg)."""
    r = float(res["oos_return_avg"])
    sh = (float(res.get("oos_sharpe_5", 0.0)) + float(res.get("oos_sharpe_60", 0.0))) / 2.0
    dd = (float(res.get("oos_max_drawdown_5", 0.0)) + float(res.get("oos_max_drawdown_60", 0.0))) / 2.0
    to = (float(res.get("oos_turnover_5", 0.0)) + float(res.get("oos_turnover_60", 0.0))) / 2.0
    return r, sh, dd, to


def is_better(candidate: Dict[str, float], best: Dict[str, float]) -> bool:
    """Keep/discard decision with tie-breaks."""
    cr, csh, cdd, cto = _score_pack(candidate)
    br, bsh, bdd, bto = _score_pack(best)

    if np.isnan(cr):
        return False
    if np.isnan(br):
        return True

    if cr > br + EPS_RETURN:
        return True
    if cr < br - EPS_RETURN:
        return False

    # Within epsilon: tie-breaks
    # 1) 稳健性：多段 OOS 中最差一段的收益更高
    cw = float(candidate.get("oos_return_avg_worst_segment", cr))
    bw = float(best.get("oos_return_avg_worst_segment", br))
    if not np.isnan(cw) and not np.isnan(bw) and cw != bw:
        return cw > bw

    # 2) Sharpe / 回撤 / 换手
    if csh != bsh:
        return csh > bsh
    if cdd != bdd:
        # max drawdown is negative; higher is better (less negative)
        return cdd > bdd
    if cto != bto:
        return cto < bto
    return True  # stable: allow keep on exact tie


def mutate(
    cfg: Config,
    rng: random.Random,
    *,
    allow_metric_mutation: bool,
    global_jump_p: float = 0.2,
) -> Config:
    """Propose a local mutation around current best config.

    - Mostly local steps around current best (autoresearch style)
    - Sometimes global jump to improve exploration
    - Optionally allow metric mutation (disabled by default to keep comparability)
    """
    top_candidates = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    idx = top_candidates.index(cfg.top_n) if cfg.top_n in top_candidates else 0

    choices = ["top_n", "n_quantiles", "weight_mode"]
    if allow_metric_mutation:
        choices.append("oos_metric")
    choice = rng.choice(choices)

    if choice == "top_n":
        if rng.random() < global_jump_p:
            idx2 = rng.randrange(0, len(top_candidates))
        else:
            step = rng.choice([-1, 1])
            idx2 = max(0, min(len(top_candidates) - 1, idx + step))
        return Config(
            top_n=top_candidates[idx2],
            n_quantiles=cfg.n_quantiles,
            weight_mode=cfg.weight_mode,
            oos_metric=cfg.oos_metric,
            oos_weight_5=cfg.oos_weight_5,
        )
    if choice == "n_quantiles":
        nq = 5 if cfg.n_quantiles == 3 else 3
        return Config(
            top_n=cfg.top_n,
            n_quantiles=nq,
            weight_mode=cfg.weight_mode,
            oos_metric=cfg.oos_metric,
            oos_weight_5=cfg.oos_weight_5,
        )
    if choice == "weight_mode":
        wm = "ic_weighted" if cfg.weight_mode == "equal" else "equal"
        return Config(
            top_n=cfg.top_n,
            n_quantiles=cfg.n_quantiles,
            weight_mode=wm,
            oos_metric=cfg.oos_metric,
            oos_weight_5=cfg.oos_weight_5,
        )
    # oos_metric mutation (optional)
    metric = rng.choice(["avg", "weighted", "60", "5"])
    w5 = cfg.oos_weight_5
    if metric == "weighted":
        # explore a few typical weights
        w5 = rng.choice([0.3, 0.5, 0.7])
    return Config(
        top_n=cfg.top_n,
        n_quantiles=cfg.n_quantiles,
        weight_mode=cfg.weight_mode,
        oos_metric=metric,
        oos_weight_5=w5,
    )


def ensure_header(path: Path) -> None:
    if path.exists():
        return
    header = [
        "experiment_id",
        "top_n",
        "n_quantiles",
        "weight_mode",
        "oos_metric",
        "oos_weight_5",
        "seed",
        "oos_return_avg",
        "oos_return_5",
        "oos_return_60",
        "oos_sharpe_5",
        "oos_sharpe_60",
        "oos_max_drawdown_5",
        "oos_max_drawdown_60",
        "oos_turnover_5",
        "oos_turnover_60",
        "fee_bps",
        "best_exprs_sha256",
        "data_end_ts",
        "runner_version",
        "status",
        "description",
        "timestamp",
    ]
    path.write_text("\t".join(header) + "\n", encoding="utf-8")


def _cfg_key(cfg: Config) -> Tuple:
    return (cfg.top_n, cfg.n_quantiles, cfg.weight_mode, cfg.oos_metric, float(cfg.oos_weight_5))


def load_existing_best(path: Path) -> Tuple[Optional[Dict[str, float]], Optional[Config], int, Set[Tuple]]:
    if not path.exists():
        return None, None, 0, set()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None, None, 0, set()
    # Determine next experiment id
    next_id = int(df["experiment_id"].max()) + 1 if "experiment_id" in df.columns else len(df)
    # Best-so-far by the same decision logic (replay sequentially)
    best_row: Optional[Dict[str, float]] = None
    best_cfg: Optional[Config] = None
    tried: Set[Tuple] = set()
    for _, row in df.iterrows():
        try:
            cfg = Config(
                top_n=int(row.get("top_n", 40)),
                n_quantiles=int(row.get("n_quantiles", 3)),
                weight_mode=str(row.get("weight_mode", "ic_weighted")),
                oos_metric=str(row.get("oos_metric", "avg")),
                oos_weight_5=float(row.get("oos_weight_5", 0.5)),
            )
            tried.add(_cfg_key(cfg))
        except Exception:
            pass
        if str(row.get("status", "")) == "crash":
            continue
        res = {
            "oos_return_avg": float(row.get("oos_return_avg", np.nan)),
            "oos_sharpe_5": float(row.get("oos_sharpe_5", 0.0)),
            "oos_sharpe_60": float(row.get("oos_sharpe_60", 0.0)),
            "oos_max_drawdown_5": float(row.get("oos_max_drawdown_5", 0.0)),
            "oos_max_drawdown_60": float(row.get("oos_max_drawdown_60", 0.0)),
            "oos_turnover_5": float(row.get("oos_turnover_5", 0.0)),
            "oos_turnover_60": float(row.get("oos_turnover_60", 0.0)),
        }
        if best_row is None or is_better(res, best_row):
            best_row = res
            try:
                best_cfg = Config(
                    top_n=int(row.get("top_n", 40)),
                    n_quantiles=int(row.get("n_quantiles", 3)),
                    weight_mode=str(row.get("weight_mode", "ic_weighted")),
                    oos_metric=str(row.get("oos_metric", "avg")),
                    oos_weight_5=float(row.get("oos_weight_5", 0.5)),
                )
            except Exception:
                best_cfg = None
    return best_row, best_cfg, next_id, tried


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30, help="Max number of experiments to run")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results_genetic50.tsv")
    parser.add_argument(
        "--allow_metric_mutation",
        action="store_true",
        help="Allow changing OOS metric during search (not recommended for comparability)",
    )
    args = parser.parse_args()

    if not BEST_EXPRS_PATH.exists():
        raise SystemExit(f"Missing: {BEST_EXPRS_PATH}")
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing: {DATA_PATH}")

    best_exprs_sha = _sha256_file(BEST_EXPRS_PATH)
    data_end_ts = _get_data_end_ts(DATA_PATH)

    if not args.resume and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    ensure_header(RESULTS_PATH)

    if args.resume:
        best_metrics, best_cfg, next_id, tried = load_existing_best(RESULTS_PATH)
    else:
        best_metrics, best_cfg, next_id, tried = None, None, 0, set()

    # Baseline config (close to current defaults in genetic_oos_eval)
    current_best_cfg = best_cfg or Config(top_n=40, n_quantiles=3, weight_mode="ic_weighted", oos_metric="avg", oos_weight_5=0.5)
    rng = random.Random(args.seed)

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for k in range(args.budget):
            exp_id = next_id + k
            run_seed = rng.randrange(1, 2**31 - 1)
            random.seed(run_seed)
            np.random.seed(run_seed)

            # propose: baseline for first run if no best yet, else mutate around current best
            if best_metrics is None and exp_id == 0:
                cfg = current_best_cfg
            else:
                # avoid repeats when possible
                cfg = None
                for _ in range(30):
                    candidate = mutate(
                        current_best_cfg,
                        rng,
                        allow_metric_mutation=args.allow_metric_mutation,
                    )
                    if _cfg_key(candidate) not in tried:
                        cfg = candidate
                        break
                if cfg is None:
                    cfg = mutate(
                        current_best_cfg,
                        rng,
                        allow_metric_mutation=args.allow_metric_mutation,
                    )
            tried.add(_cfg_key(cfg))

            desc = (
                f"top_n={cfg.top_n} n_quantiles={cfg.n_quantiles} "
                f"weight={cfg.weight_mode} metric={cfg.oos_metric}"
                + (f" w5={cfg.oos_weight_5}" if cfg.oos_metric == "weighted" else "")
            )
            print(f"[{k+1}/{args.budget}] exp_id={exp_id} {desc} ... ", end="", flush=True)

            res = genetic_oos_eval.run_eval(
                top_n=cfg.top_n,
                n_quantiles=cfg.n_quantiles,
                weight_mode=cfg.weight_mode,
                oos_metric=cfg.oos_metric,
                oos_weight_5=cfg.oos_weight_5,
                fee_bps=10,
            )

            ts = datetime.now().replace(microsecond=0).isoformat(sep=" ")

            if "error" in res:
                status = "crash"
                print("crash")
                row = {
                    "experiment_id": exp_id,
                    **asdict(cfg),
                    "seed": run_seed,
                    "oos_return_avg": "",
                    "oos_return_5": "",
                    "oos_return_60": "",
                    "oos_sharpe_5": "",
                    "oos_sharpe_60": "",
                    "oos_max_drawdown_5": "",
                    "oos_max_drawdown_60": "",
                    "oos_turnover_5": "",
                    "oos_turnover_60": "",
                    "fee_bps": 10,
                    "best_exprs_sha256": best_exprs_sha,
                    "data_end_ts": data_end_ts,
                    "runner_version": RUNNER_VERSION,
                    "status": status,
                    "description": f"{desc} crash: {res['error']}",
                    "timestamp": ts,
                }
            else:
                status = "keep" if (best_metrics is None or is_better(res, best_metrics)) else "discard"
                if status == "keep":
                    best_metrics = res
                    current_best_cfg = cfg
                print(f"oos_return_avg={res['oos_return_avg']:.4f} {status}")
                row = {
                    "experiment_id": exp_id,
                    **asdict(cfg),
                    "seed": run_seed,
                    "oos_return_avg": res["oos_return_avg"],
                    "oos_return_5": res["oos_return_5"],
                    "oos_return_60": res["oos_return_60"],
                    "oos_sharpe_5": res["oos_sharpe_5"],
                    "oos_sharpe_60": res["oos_sharpe_60"],
                    "oos_max_drawdown_5": res["oos_max_drawdown_5"],
                    "oos_max_drawdown_60": res["oos_max_drawdown_60"],
                    "oos_turnover_5": res["oos_turnover_5"],
                    "oos_turnover_60": res["oos_turnover_60"],
                    "fee_bps": 10,
                    "best_exprs_sha256": best_exprs_sha,
                    "data_end_ts": data_end_ts,
                    "runner_version": RUNNER_VERSION,
                    "status": status,
                    "description": desc,
                    "timestamp": ts,
                }

            # write row in header order
            header = RESULTS_PATH.read_text(encoding="utf-8").splitlines()[0].split("\t")
            f.write("\t".join(str(row.get(h, "")) for h in header) + "\n")
            f.flush()

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

