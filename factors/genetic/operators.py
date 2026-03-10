"""
遗传因子挖掘 - 算子库
提供可在 OHLCV 上计算的原子算子，供表达式树组合使用。
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any

# 终端（叶子）：列名
TERMINALS = ["$close", "$open", "$high", "$low", "$volume"]

# 一元滚动算子：(series, window) -> series
ROLLING_OPS = ["ts_mean", "ts_std", "ts_ema", "ts_delta", "ts_sum", "ts_max", "ts_min"]

# 一元无参算子：(series) -> series
UNARY_OPS = ["abs", "log", "sign", "neg"]

# 二元算子：(left, right) 或 (left, right, window)
BINARY_OPS = ["add", "sub", "mul", "div", "ts_corr"]

# 允许的滚动窗口
WINDOWS = [5, 10, 20, 30, 60]


def _ts_mean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).mean()


def _ts_std(x: pd.Series, w: int) -> pd.Series:
    s = x.rolling(window=w, min_periods=1).std()
    return s.replace(0, np.nan).bfill().fillna(1e-8)


def _ts_ema(x: pd.Series, w: int) -> pd.Series:
    return x.ewm(span=w, adjust=False, min_periods=1).mean()


def _ts_delta(x: pd.Series, w: int) -> pd.Series:
    return x - x.shift(w)


def _ts_sum(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).sum()


def _ts_max(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).max()


def _ts_min(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).min()


def _ts_corr(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=2).corr(y).fillna(0)


def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.abs() + 1e-8)


def _safe_sign(x: pd.Series) -> pd.Series:
    return np.sign(x).replace(0, np.nan).ffill().fillna(0)


def get_terminal(df: pd.DataFrame, name: str) -> pd.Series:
    """从 DataFrame 取终端列。name 如 '$close'."""
    col = name.replace("$", "")
    if col not in df.columns:
        raise KeyError(f"Column {col} not in DataFrame")
    return df[col].astype(float)


def eval_rolling(op: str, x: pd.Series, w: int) -> pd.Series:
    """执行滚动算子"""
    f = {
        "ts_mean": _ts_mean,
        "ts_std": _ts_std,
        "ts_ema": _ts_ema,
        "ts_delta": _ts_delta,
        "ts_sum": _ts_sum,
        "ts_max": _ts_max,
        "ts_min": _ts_min,
    }
    return f[op](x, w)


def eval_unary(op: str, x: pd.Series) -> pd.Series:
    """执行一元算子"""
    if op == "abs":
        return x.abs()
    if op == "log":
        return _safe_log(x)
    if op == "sign":
        return _safe_sign(x)
    if op == "neg":
        return -x
    raise ValueError(f"Unknown unary op: {op}")


def eval_binary(op: str, left: pd.Series, right: pd.Series, window: int = None) -> pd.Series:
    """执行二元算子。ts_corr 需要 window."""
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / (right.abs() + 1e-8)
    if op == "ts_corr" and window is not None:
        return _ts_corr(left, right, window)
    raise ValueError(f"Unknown binary op or missing window: {op}")


def list_rolling_ops() -> list:
    return list(ROLLING_OPS)


def list_unary_ops() -> list:
    return list(UNARY_OPS)


def list_binary_ops() -> list:
    return list(BINARY_OPS)


def list_terminals() -> list:
    return list(TERMINALS)


def list_windows() -> list:
    return list(WINDOWS)
