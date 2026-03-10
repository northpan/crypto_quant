#!/usr/bin/env python3
"""
重跑 results.tsv 中 keep/持平的实验配置，在 TRAIN_LOOKBACK_DAYS=7、INFER_RETRAIN_DAYS=1、FEE_BPS=10 下比较扣费后结果。
"""

import io
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# 要跑的 keep/持平 对应配置: (description, TOP_N_FACTORS, SELECT_IC, N_QUANTILES, COMPOSITE_MODE, OOS_METRIC)
CONFIGS = [
    ("1_baseline_all_factors_q5", None, "5_60", 5, "linear", "avg"),
    ("2_top20_IC_q5", 20, "5_60", 5, "linear", "avg"),
    ("3_top20_IC_q3", 20, "5_60", 3, "linear", "avg"),
    ("4_top18_IC_q3", 18, "5_60", 3, "linear", "avg"),
    ("5_top18_60IC_q3_linear", 18, "60", 3, "linear", "avg"),
    ("6_top18_60IC_q3_tree", 18, "60", 3, "tree", "avg"),
    ("7_top18_60IC_q3_lgb", 18, "60", 3, "lgb", "avg"),
    ("8_top18_60IC_q3_OOS5", 18, "60", 3, "linear", "5"),
]


def run_one(cfg):
    import btc_oos_eval
    name, top_n, select_ic, n_q, mode, oos_metric = cfg
    btc_oos_eval.TOP_N_FACTORS = top_n
    btc_oos_eval.SELECT_IC = select_ic
    btc_oos_eval.N_QUANTILES = n_q
    btc_oos_eval.COMPOSITE_MODE = mode
    btc_oos_eval.OOS_METRIC = oos_metric
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        btc_oos_eval.main()
    finally:
        sys.stdout = old
    out = buf.getvalue()
    # parse
    oos_avg = oos_5 = oos_60 = sharpe_5 = sharpe_60 = dd_5 = dd_60 = None
    for line in out.splitlines():
        m = re.match(r"^oos_return_avg:\s*(.+)$", line)
        if m:
            oos_avg = float(m.group(1).strip())
        m = re.match(r"^oos_return_5:\s*(.+)$", line)
        if m:
            oos_5 = float(m.group(1).strip())
        m = re.match(r"^oos_return_60:\s*(.+)$", line)
        if m:
            oos_60 = float(m.group(1).strip())
        m = re.match(r"^oos_sharpe_5:\s*(.+)$", line)
        if m:
            sharpe_5 = float(m.group(1).strip())
        m = re.match(r"^oos_sharpe_60:\s*(.+)$", line)
        if m:
            sharpe_60 = float(m.group(1).strip())
        m = re.match(r"^oos_max_drawdown_5:\s*(.+)$", line)
        if m:
            dd_5 = float(m.group(1).strip())
        m = re.match(r"^oos_max_drawdown_60:\s*(.+)$", line)
        if m:
            dd_60 = float(m.group(1).strip())
    return {
        "name": name,
        "oos_return_avg": oos_avg,
        "oos_return_5": oos_5,
        "oos_return_60": oos_60,
        "oos_sharpe_5": sharpe_5,
        "oos_sharpe_60": sharpe_60,
        "oos_max_dd_5": dd_5,
        "oos_max_dd_60": dd_60,
    }


def main():
    print("TRAIN_LOOKBACK_DAYS=7, INFER_RETRAIN_DAYS=1, FEE_BPS=10. Running keep/持平 configs...\n")
    results = []
    for i, cfg in enumerate(CONFIGS):
        print(f"[{i+1}/{len(CONFIGS)}] {cfg[0]} ...")
        try:
            r = run_one(cfg)
            results.append(r)
            print(f"  oos_return_avg={r['oos_return_avg']:.4f}  oos_5={r['oos_return_5']:.4f}  oos_60={r['oos_return_60']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": cfg[0], "oos_return_avg": None, "oos_return_5": None, "oos_return_60": None, "oos_sharpe_5": None, "oos_sharpe_60": None, "oos_max_dd_5": None, "oos_max_dd_60": None})

    # sort by oos_return_avg desc (best first), None last
    valid = [r for r in results if r.get("oos_return_avg") is not None]
    valid.sort(key=lambda x: x["oos_return_avg"], reverse=True)
    print("\n" + "=" * 80)
    print("扣费后 按 oos_return_avg 从高到低 (TRAIN_LOOKBACK=7, INFER_RETRAIN=1, FEE_BPS=10)")
    print("=" * 80)
    for i, r in enumerate(valid, 1):
        print(f"{i}. {r['name']}")
        print(f"   oos_return_avg: {r['oos_return_avg']:.4f}  oos_5: {r['oos_return_5']:.4f}  oos_60: {r['oos_return_60']:.4f}")
        if r.get("oos_sharpe_5") is not None:
            print(f"   sharpe_5: {r['oos_sharpe_5']:.2f}  sharpe_60: {r['oos_sharpe_60']:.2f}  max_dd_5: {r['oos_max_dd_5']:.4f}  max_dd_60: {r['oos_max_dd_60']:.4f}")
        print()
    if any(r.get("oos_return_avg") is None for r in results):
        print("Failed configs:")
        for r in results:
            if r.get("oos_return_avg") is None:
                print(f"  - {r['name']}")
    out_csv = PROJECT_ROOT / "reports" / "keep_experiments_after_fee.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("name,oos_return_avg,oos_return_5,oos_return_60,oos_sharpe_5,oos_sharpe_60,oos_max_dd_5,oos_max_dd_60\n")
        for r in results:
            f.write(f"{r['name']},{r.get('oos_return_avg') or ''},{r.get('oos_return_5') or ''},{r.get('oos_return_60') or ''},{r.get('oos_sharpe_5') or ''},{r.get('oos_sharpe_60') or ''},{r.get('oos_max_dd_5') or ''},{r.get('oos_max_dd_60') or ''}\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
