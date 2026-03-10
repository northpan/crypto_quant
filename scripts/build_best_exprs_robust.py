#!/usr/bin/env python3
"""
从稳健遗传挖掘结果中选出一批新因子，与现有 best_exprs.pkl 合并，生成新的 best_exprs_robust.pkl。

规则：
- 读取 reports/genetic_factor_mining_robust.csv（按 fitness 降序）
- 依次扫描，选出前 20 个“非平凡 + ret_5 或 ret_60 为正”的表达式
- 与 reports/genetic_50_rounds/best_exprs.pkl 中的 50 个表达式取并集（按 expr_to_str 去重）
- 合并后的表达式列表写入 reports/genetic_50_rounds/best_exprs_robust.pkl
"""

import csv
import pickle
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from factors.genetic.expression import expr_to_str  # type: ignore


def is_trivial_expr_str(s: str) -> bool:
    """过滤掉过于简单或明显退化的表达式字符串。"""
    s = s.strip()
    if not s:
        return True
    # 直接是基础价量字段
    if s in {"open", "high", "low", "close", "volume"}:
        return True
    # 长度极短且不包含任何算子符号
    ops_tokens = ["ts_", "add", "sub", "mul", "div", "corr", "ema", "std", "mean", "delta", "min", "max", "log", "sign", "abs"]
    if len(s) < 8 and not any(tok in s for tok in ops_tokens):
        return True
    return False


def main():
    robust_csv = PROJECT_ROOT / "reports" / "genetic_factor_mining_robust.csv"
    if not robust_csv.exists():
        print(f"Not found: {robust_csv}")
        print("请先运行: python scripts/run_genetic_factor_mining.py --out reports/genetic_factor_mining_robust.csv ...")
        return

    robust_exprs_pkl = PROJECT_ROOT / "reports" / "genetic_factor_mining_robust.pkl"
    if not robust_exprs_pkl.exists():
        print(f"Not found: {robust_exprs_pkl}")
        print("请先用带稳健适应度的参数重新运行 run_genetic_factor_mining.py，生成表达式对象 pkl。")
        return

    base_pkl = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs.pkl"
    if not base_pkl.exists():
        print(f"Not found: {base_pkl}")
        print("请先运行: python scripts/run_50_rounds_and_composite_backtest.py")
        return

    # 读取基础 expr 列表
    with open(base_pkl, "rb") as f:
        base_exprs = pickle.load(f)

    expr_map = {}
    for e in base_exprs:
        expr_map[expr_to_str(e)] = e

    # 读取稳健 GA CSV，选出前 20 个合格表达式的“行号”
    selected_indices = []
    selected_strs = []
    with open(robust_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if len(selected_indices) >= 20:
                break
            expr_s = row.get("expr_str", "").strip()
            if is_trivial_expr_str(expr_s):
                continue
            try:
                ret5 = float(row.get("ret_5", "nan"))
                ret60 = float(row.get("ret_60", "nan"))
            except ValueError:
                ret5 = ret60 = float("nan")
            # 至少一个周期的 ret 为正，避免明显回测亏损
            if not ((not np.isnan(ret5) and ret5 > 0) or (not np.isnan(ret60) and ret60 > 0)):
                continue
            if expr_s in expr_map or expr_s in selected_strs:
                continue
            selected_indices.append(idx)
            selected_strs.append(expr_s)

    print(f"从稳健 GA 结果中选出 {len(selected_indices)} 个新表达式（目标 20 个）。")

    # 读取与 CSV 对齐的表达式对象列表
    with open(robust_exprs_pkl, "rb") as f:
        robust_exprs = pickle.load(f)

    if len(robust_exprs) < max(selected_indices, default=-1) + 1:
        print("表达式 pkl 行数与 CSV 不一致，请检查数据生成流程。")
        return

    for idx in selected_indices:
        expr = robust_exprs[idx]
        s = expr_to_str(expr)
        expr_map[s] = expr

    merged_exprs = list(expr_map.values())

    out_pkl = PROJECT_ROOT / "reports" / "genetic_50_rounds" / "best_exprs_robust.pkl"
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(merged_exprs, f)

    print(f"合并后的表达式数量: {len(merged_exprs)}")
    print(f"已保存: {out_pkl}")


if __name__ == "__main__":
    main()

