"""
遗传因子挖掘 - 表达式树
表达式为嵌套结构，在 OHLCV DataFrame 上求值得到 pd.Series。
"""

import pandas as pd
from typing import Union, Tuple, Any, List

from . import operators as ops


def is_terminal(expr: Any) -> bool:
    """是否为终端（列名）"""
    return isinstance(expr, str) and expr.startswith("$")


def eval_expr(expr: Any, df: pd.DataFrame) -> pd.Series:
    """
    在 df 上求值表达式，返回因子序列。
    expr 形式:
      - 终端: "$close", "$volume" 等
      - 滚动: ("ts_mean", sub_expr, window), ("ts_std", sub_expr, window) 等
      - 一元: ("abs", sub_expr), ("log", sub_expr), ("neg", sub_expr), ("sign", sub_expr)
      - 二元: ("add", left, right), ("sub", left, right), ("mul", left, right), ("div", left, right)
      - 滚动相关: ("ts_corr", left, right, window)
    """
    if is_terminal(expr):
        return ops.get_terminal(df, expr)

    if not isinstance(expr, (list, tuple)) or len(expr) < 2:
        raise ValueError(f"Invalid expression: {expr}")

    head = expr[0]

    # 一元
    if head in ops.list_unary_ops():
        if len(expr) != 2:
            raise ValueError(f"Unary op needs 1 arg: {expr}")
        x = eval_expr(expr[1], df)
        return ops.eval_unary(head, x)

    # 滚动一元 (op, sub_expr, window)
    if head in ops.list_rolling_ops():
        if len(expr) != 3:
            raise ValueError(f"Rolling op needs (expr, window): {expr}")
        x = eval_expr(expr[1], df)
        w = int(expr[2])
        return ops.eval_rolling(head, x, w)

    # 二元 (add/sub/mul/div)
    if head in ["add", "sub", "mul", "div"]:
        if len(expr) != 3:
            raise ValueError(f"Binary op needs 2 args: {expr}")
        left = eval_expr(expr[1], df)
        right = eval_expr(expr[2], df)
        return ops.eval_binary(head, left, right)

    # ts_corr(left, right, window)
    if head == "ts_corr":
        if len(expr) != 4:
            raise ValueError(f"ts_corr needs (left, right, window): {expr}")
        left = eval_expr(expr[1], df)
        right = eval_expr(expr[2], df)
        w = int(expr[3])
        return ops.eval_binary("ts_corr", left, right, window=w)

    raise ValueError(f"Unknown op: {head}")


def expr_to_str(expr: Any) -> str:
    """将表达式转为可读字符串，便于命名与日志"""
    if is_terminal(expr):
        return expr.replace("$", "")
    if not isinstance(expr, (list, tuple)):
        return str(expr)
    head = expr[0]
    if head in ops.list_unary_ops():
        return f"{head}({expr_to_str(expr[1])})"
    if head in ops.list_rolling_ops():
        return f"{head}({expr_to_str(expr[1])},{expr[2]})"
    if head in ["add", "sub", "mul", "div"]:
        return f"({expr_to_str(expr[1])}{head}{expr_to_str(expr[2])})"
    if head == "ts_corr":
        return f"ts_corr({expr_to_str(expr[1])},{expr_to_str(expr[2])},{expr[3]})"
    return str(expr)


def expr_depth(expr: Any) -> int:
    """表达式深度（终端为 0）"""
    if is_terminal(expr):
        return 0
    if not isinstance(expr, (list, tuple)) or len(expr) < 2:
        return 0
    return 1 + max(expr_depth(c) for c in expr[1:])


def _expression_children(expr: Any) -> List[Any]:
    """返回直接子表达式（不含字面量如 window 整数），用于安全地交叉/变异。"""
    if is_terminal(expr) or not isinstance(expr, (list, tuple)) or len(expr) < 2:
        return []
    head = expr[0]
    if head in ops.list_rolling_ops():
        return [expr[1]]
    if head in ops.list_unary_ops():
        return [expr[1]]
    if head in ["add", "sub", "mul", "div"]:
        return [expr[1], expr[2]]
    if head == "ts_corr":
        return [expr[1], expr[2]]
    return []


def get_subtrees(expr: Any) -> List[Any]:
    """收集所有表达式子树（含自身），仅包含可替换为另一表达式的节点。"""
    out = [expr]
    for child in _expression_children(expr):
        out.extend(get_subtrees(child))
    return out
