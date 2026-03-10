#!/usr/bin/env python3
"""
对遗传因子组合进行多组实验迭代，写入 results_genetic50.tsv（格式与 results.tsv 一致：
commit, oos_return_avg, status, description），便于与 plot_experiment_progress 同逻辑绘图。
使用 scripts/genetic_oos_eval.py 的 run_eval 执行单次评估。
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import genetic_oos_eval
run_eval = genetic_oos_eval.run_eval

# 与 results.tsv 一致的列
RESULTS_HEADER = "commit\toos_return_avg\tstatus\tdescription\n"
COMMIT_PREFIX = "genetic50"


def build_configs(n_experiments: int = 30):
    """生成 n_experiments 组 (top_n, n_quantiles, weight_mode)。"""
    configs = []
    top_n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    modes = [(3, "equal"), (5, "equal"), (3, "ic_weighted"), (5, "ic_weighted")]
    for top_n in top_n_list:
        for n_q, wm in modes:
            configs.append({"top_n": top_n, "n_quantiles": n_q, "weight_mode": wm})
            if len(configs) >= n_experiments:
                return configs
    while len(configs) < n_experiments:
        configs.append(configs[len(configs) % len(configs)])
    return configs[:n_experiments]


def main():
    n_experiments = 30
    if len(sys.argv) > 1:
        try:
            n_experiments = int(sys.argv[1])
        except ValueError:
            pass

    configs = build_configs(n_experiments)
    results_file = PROJECT_ROOT / "results_genetic50.tsv"
    best_avg = None

    with open(results_file, "w", encoding="utf-8") as f:
        f.write(RESULTS_HEADER)
        for i, c in enumerate(configs):
            top_n = c["top_n"]
            n_q = c["n_quantiles"]
            wm = c["weight_mode"]
            desc = f"top_n={top_n} n_quantiles={n_q} weight={wm}"
            print(f"Experiment {i+1}/{len(configs)}: {desc} ... ", end="", flush=True)
            res = run_eval(
                top_n=top_n,
                n_quantiles=n_q,
                weight_mode=wm,
                oos_metric="avg",
                oos_weight_5=0.5,
                fee_bps=10,
            )
            if "error" in res:
                status = "crash"
                oos_avg = float("nan")
                line = f"{COMMIT_PREFIX}\t\t{status}\t{desc} {res['error']}\n"
                print("crash:", res["error"])
            else:
                oos_avg = res["oos_return_avg"]
                if best_avg is None or oos_avg >= best_avg:
                    best_avg = oos_avg
                    status = "keep"
                else:
                    status = "discard"
                line = f"{COMMIT_PREFIX}\t{oos_avg}\t{status}\t{desc}\n"
                print(f"oos_return_avg={oos_avg:.4f} {status}")
            f.write(line)
            f.flush()

    print(f"\nResults written to {results_file}")
    print(f"Best oos_return_avg: {best_avg}")


if __name__ == "__main__":
    main()
