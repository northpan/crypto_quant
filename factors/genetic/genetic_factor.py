"""
遗传因子挖掘 - 遗传算法
随机表达式生成、变异、交叉、适应度（IC + ret），迭代进化筛选 ret 正收益因子。
"""

import copy
import random
from typing import Any, List, Tuple, Callable, Optional
import numpy as np

from . import operators as ops
from .expression import expr_to_str, expr_depth, get_subtrees, eval_expr


def random_terminal() -> str:
    return random.choice(ops.list_terminals())


def random_window() -> int:
    return random.choice(ops.list_windows())


def random_expr(max_depth: int, depth: int = 0) -> Any:
    """
    随机生成一棵表达式树。depth 为当前深度，max_depth 为最大深度。
    根节点为算子，叶子为终端。
    """
    if depth >= max_depth:
        return random_terminal()

    roll = random.random()
    # 一定概率直接返回终端
    if roll < 0.25:
        return random_terminal()

    # 一元滚动: (ts_mean, expr, window)
    if roll < 0.5:
        op = random.choice(ops.list_rolling_ops())
        return (op, random_expr(max_depth, depth + 1), random_window())

    # 一元无参: (abs, expr)
    if roll < 0.65:
        op = random.choice(ops.list_unary_ops())
        return (op, random_expr(max_depth, depth + 1))

    # 二元 add/sub/mul/div
    if roll < 0.9:
        op = random.choice(["add", "sub", "mul", "div"])
        return (op, random_expr(max_depth, depth + 1), random_expr(max_depth, depth + 1))

    # ts_corr(left, right, window)
    return (
        "ts_corr",
        random_expr(max_depth, depth + 1),
        random_expr(max_depth, depth + 1),
        random_window(),
    )


def mutate(expr: Any, max_depth: int = 4) -> Any:
    """随机替换一棵子树（含根）为新的随机子树。"""
    trees = get_subtrees(expr)
    if not trees:
        return random_expr(max_depth)
    sub = random.choice(trees)
    new_sub = random_expr(max_depth)
    return _replace_subtree(copy.deepcopy(expr), sub, new_sub)


def _replace_subtree(expr: Any, target: Any, replacement: Any, done: Optional[List[bool]] = None) -> Any:
    """在 expr 的深拷贝中将第一处与 target 结构相同的子树替换为 replacement。"""
    if done is None:
        done = [False]
    if not done[0] and _expr_eq(expr, target):
        done[0] = True
        return copy.deepcopy(replacement)
    if isinstance(expr, (list, tuple)):
        out = [expr[0]] + [_replace_subtree(c, target, replacement, done) for c in expr[1:]]
        return tuple(out) if isinstance(expr, tuple) else out
    return expr


def _expr_eq(a: Any, b: Any) -> bool:
    """结构相等（用于子树匹配）。"""
    if a == b:
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return all(_expr_eq(x, y) for x, y in zip(a, b))
    return False


def crossover(expr_a: Any, expr_b: Any) -> Any:
    """从 expr_b 中随机取一棵子树，替换 expr_a 中的随机一棵子树。"""
    trees_a = get_subtrees(expr_a)
    trees_b = get_subtrees(expr_b)
    if not trees_a or not trees_b:
        return copy.deepcopy(expr_a)
    sub_a = random.choice(trees_a)
    sub_b = random.choice(trees_b)
    return _replace_subtree(copy.deepcopy(expr_a), sub_a, copy.deepcopy(sub_b))


def fitness_from_metrics(
    metrics: dict,
    alpha: float = 0.3,
    beta: float = 0.7,
    ret_weight_5: float = 0.5,
    ret_weight_60: float = 0.5,
    gamma: float = 0.15,
    delta: float = 0.05,
) -> float:
    """
    根据评估结果计算适应度。
    metrics 含 ic_5, ic_60, ret_5, ret_60, 以及可选的 ic_robust、ret_robust。

    适应度 = alpha * (|ic_5| + |ic_60|)/2
           + beta  * (ret_weight_5*ret_5 + ret_weight_60*ret_60)
           + gamma * ic_robust
           + delta * ret_robust
    若值为 nan 则视为 0。
    """
    ic_5 = metrics.get("ic_5", float("nan"))
    ic_60 = metrics.get("ic_60", float("nan"))
    ret_5 = metrics.get("ret_5", float("nan"))
    ret_60 = metrics.get("ret_60", float("nan"))
    ic_robust = metrics.get("ic_robust", float("nan"))
    ret_robust = metrics.get("ret_robust", float("nan"))
    try:
        if np.isnan(ic_5) or np.isinf(ic_5):
            ic_5 = 0.0
        if np.isnan(ic_60) or np.isinf(ic_60):
            ic_60 = 0.0
        if np.isnan(ret_5) or np.isinf(ret_5):
            ret_5 = 0.0
        if np.isnan(ret_60) or np.isinf(ret_60):
            ret_60 = 0.0
        if np.isnan(ic_robust) or np.isinf(ic_robust):
            ic_robust = 0.0
        if np.isnan(ret_robust) or np.isinf(ret_robust):
            ret_robust = 0.0
    except (TypeError, ValueError):
        ic_5 = ic_60 = ret_5 = ret_60 = ic_robust = ret_robust = 0.0
    ic_part = (abs(ic_5) + abs(ic_60)) / 2.0
    ret_part = ret_weight_5 * ret_5 + ret_weight_60 * ret_60
    return alpha * ic_part + beta * ret_part + gamma * ic_robust + delta * ret_robust


def run_evolution(
    evaluate_fn: Callable[[Any], dict],
    population_size: int = 50,
    generations: int = 30,
    max_depth: int = 4,
    elite_ratio: float = 0.1,
    mutation_prob: float = 0.55,
    crossover_prob: float = 0.5,
    immigrant_ratio: float = 0.2,
    alpha: float = 0.3,
    beta: float = 0.7,
    ret_weight_5: float = 0.5,
    ret_weight_60: float = 0.5,
    gamma: float = 0.15,
    delta: float = 0.05,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Any, dict, float]]:
    """
    运行遗传算法。
    evaluate_fn(expr) -> metrics dict (ic_5, ic_60, ret_5, ret_60)。
    返回列表 [(expr, metrics, fitness), ...]，按 fitness 降序；同时筛选 ret_5 或 ret_60 为正的个体。
    """
    rng = rng or random.Random()

    # 适应度计算中需要 pandas 检查 nan
    def fit(metrics: dict) -> float:
        return fitness_from_metrics(
            metrics,
            alpha=alpha,
            beta=beta,
            ret_weight_5=ret_weight_5,
            ret_weight_60=ret_weight_60,
            gamma=gamma,
            delta=delta,
        )

    population: List[Any] = []
    for _ in range(population_size):
        population.append(random_expr(max_depth))

    # 缓存 (expr_str -> (metrics, fitness))
    cache: dict = {}

    def eval_cached(expr: Any):
        key = expr_to_str(expr)
        if key not in cache:
            try:
                m = evaluate_fn(expr)
                cache[key] = (m, fit(m))
            except Exception:
                cache[key] = ({"ic_5": float("nan"), "ic_60": float("nan"), "ret_5": float("nan"), "ret_60": float("nan")}, float("-inf"))
        return cache[key]

    for gen in range(generations):
        scored = [(e, *eval_cached(e)) for e in population]
        scored.sort(key=lambda x: x[2], reverse=True)
        elite_n = max(1, int(population_size * elite_ratio))
        # 多样性：精英去重（同 expr_str 只保留第一个）
        seen_str = set()
        elite = []
        for i in range(len(scored)):
            if len(elite) >= elite_n:
                break
            s = expr_to_str(scored[i][0])
            if s not in seen_str:
                seen_str.add(s)
                elite.append(scored[i][0])
        if not elite:
            elite = [scored[0][0]]
        new_pop = list(elite)
        n_immigrant = max(0, int(population_size * immigrant_ratio))
        for _ in range(n_immigrant):
            new_pop.append(random_expr(max_depth))
        while len(new_pop) < population_size:
            if rng.random() < crossover_prob and len(elite) >= 2:
                a, b = rng.sample(elite, 2)
                child = crossover(a, b)
            else:
                child = copy.deepcopy(rng.choice(elite))
            if rng.random() < mutation_prob:
                child = mutate(child, max_depth)
            new_pop.append(child)
        population = new_pop[:population_size]

    # 最终排序并返回所有个体（含 ret 信息）
    scored = [(e, *eval_cached(e)) for e in population]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(e, m, f) for e, m, f in scored]
