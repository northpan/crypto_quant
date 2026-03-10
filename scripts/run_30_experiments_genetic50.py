#!/usr/bin/env python3
"""
按 program_genetic50.md 运行 30 次实验：对 50 个遗传因子做不同组合配置，记录样本外收益到 results_genetic50.tsv。
"""

import sys
import subprocess
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_configs():
    """生成 30 组 (top_n, n_quantiles, weight_mode)。"""
    configs = []
    top_n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    modes = [(3, "equal"), (5, "equal"), (3, "ic_weighted"), (5, "ic_weighted")]
    for top_n in top_n_list:
        for n_q, wm in modes:
            configs.append({"top_n": top_n, "n_quantiles": n_q, "weight_mode": wm})
            if len(configs) >= 30:
                return configs
    # 若不足 30 则重复部分
    while len(configs) < 30:
        configs.append(configs[len(configs) % len(configs)])
    return configs[:30]


def run_one(experiment_id: int, top_n: int, n_quantiles: int, weight_mode: str) -> dict:
    """运行一次 btc_oos_eval_genetic50.py，解析输出，返回指标字典。"""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "btc_oos_eval_genetic50.py"),
        "--top_n", str(top_n),
        "--n_quantiles", str(n_quantiles),
        "--weight_mode", weight_mode,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT))
        text = out.stdout or ""
        if out.returncode != 0:
            return {"crash": True, "stderr": (out.stderr or "")[:500]}
    except subprocess.TimeoutExpired:
        return {"crash": True, "stderr": "timeout"}
    except Exception as e:
        return {"crash": True, "stderr": str(e)[:500]}

    def parse_float(key: str) -> float:
        m = re.search(rf"^{re.escape(key)}\s*:\s*([-\d.eE]+)", text, re.MULTILINE)
        return float(m.group(1)) if m else float("nan")

    return {
        "crash": False,
        "in_sample_return_5": parse_float("in_sample_return_5"),
        "in_sample_return_60": parse_float("in_sample_return_60"),
        "oos_return_5": parse_float("oos_return_5"),
        "oos_return_60": parse_float("oos_return_60"),
        "oos_return_avg": parse_float("oos_return_avg"),
        "oos_sharpe_5": parse_float("oos_sharpe_5"),
        "oos_sharpe_60": parse_float("oos_sharpe_60"),
        "oos_max_drawdown_5": parse_float("oos_max_drawdown_5"),
        "oos_max_drawdown_60": parse_float("oos_max_drawdown_60"),
    }


def main():
    configs = build_configs()
    results_file = PROJECT_ROOT / "results_genetic50.tsv"
    best_avg = None
    header = "experiment_id\ttop_n\tn_quantiles\tweight_mode\toos_return_avg\toos_return_5\toos_return_60\toos_sharpe_5\toos_sharpe_60\toos_max_drawdown_5\toos_max_drawdown_60\tstatus\tdescription\n"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(header)
        for i, c in enumerate(configs):
            top_n = c["top_n"]
            n_q = c["n_quantiles"]
            wm = c["weight_mode"]
            desc = f"top_n={top_n} n_quantiles={n_q} weight={wm}"
            print(f"Experiment {i+1}/30: {desc} ... ", end="", flush=True)
            res = run_one(i, top_n, n_q, wm)
            if res.get("crash"):
                status = "crash"
                oos_avg = float("nan")
                line = f"{i}\t{top_n}\t{n_q}\t{wm}\t\t\t\t\t\t\t\t{status}\t{desc} crash\n"
                print("crash")
            else:
                oos_avg = res["oos_return_avg"]
                if best_avg is None or (not (oos_avg != oos_avg) and oos_avg >= best_avg):
                    best_avg = oos_avg
                    status = "keep"
                else:
                    status = "discard"
                line = (
                    f"{i}\t{top_n}\t{n_q}\t{wm}\t"
                    f"{res['oos_return_avg']}\t{res['oos_return_5']}\t{res['oos_return_60']}\t"
                    f"{res['oos_sharpe_5']}\t{res['oos_sharpe_60']}\t"
                    f"{res['oos_max_drawdown_5']}\t{res['oos_max_drawdown_60']}\t"
                    f"{status}\t{desc}\n"
                )
                print(f"oos_return_avg={oos_avg:.4f} {status}")
            f.write(line)
            f.flush()

    print(f"\nResults written to {results_file}")
    print(f"Best oos_return_avg: {best_avg}")


if __name__ == "__main__":
    main()
