#!/usr/bin/env python3
"""
策略参数搜索：网格搜索 + 可选 Optuna 优化
用于调参 RSI/均线周期、止损止盈、仓位上限等，目标最大化总收益率（total_return），并满足约束。
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 减少回测过程中的日志输出
logging.getLogger("__main__").setLevel(logging.WARNING)
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("data").setLevel(logging.WARNING)


def _default_backtest_range():
    """默认回测区间：过去约 30 天，与五品种一个月本地数据一致。"""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# 默认回测区间（可用 BACKTEST_START / BACKTEST_END 覆盖）
_default_range = _default_backtest_range()
DEFAULT_START = os.environ.get("BACKTEST_START", _default_range[0])
DEFAULT_END = os.environ.get("BACKTEST_END", _default_range[1])

# 约束：最大回撤不差于 -15%，交易次数至少 5 笔
MIN_TRADES = 5
MAX_DRAWDOWN_THRESHOLD = -0.15  # 即 max_drawdown > -0.15 才纳入有效结果


def _to_json_serializable(obj: Any) -> Any:
    """将 numpy 等类型转为 JSON 可序列化。"""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    return obj


def get_param_grid(small: bool = False) -> List[Dict[str, Any]]:
    """网格搜索参数组合。small=True 时约 8 组便于快速验证。"""
    shorts = [5, 10] if not small else [5]
    longs = [20, 40] if not small else [20, 40]
    rsis = [65, 70] if not small else [70]
    stops = [0.005, 0.01] if not small else [0.005]
    tps = [0.01, 0.02] if not small else [0.015]
    maxpos = [2, 3] if not small else [3]
    grid = []
    for short_period in shorts:
        for long_period in longs:
            if long_period <= short_period:
                continue
            for rsi_max_open in rsis:
                for stop_loss_pct in stops:
                    for take_profit_pct in tps:
                        for max_positions in maxpos:
                            grid.append({
                                "short_period": short_period,
                                "long_period": long_period,
                                "rsi_max_open": rsi_max_open,
                                "stop_loss_pct": stop_loss_pct,
                                "take_profit_pct": take_profit_pct,
                                "max_positions": max_positions,
                                "rsi_period": 14,
                                "volume_ma_period": 20,
                                "use_volume_filter": True,
                                "max_exposure_pct": 0.6,
                                "position_pct": 0.2,
                            })
    return grid


async def run_single_backtest(
    system: Any,
    start_date: str,
    end_date: str,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """跑一轮回测，返回 (performance_dict, trade_stats)。"""
    result = await system.run_backtest(
        start_date=start_date,
        end_date=end_date,
        strategy_params=params,
        save_report=False
    )
    perf = result.get("performance") or {}
    trade_stats = result.get("trade_stats") or {}
    return perf, trade_stats


def is_valid_result(perf: Dict, trade_stats: Dict) -> bool:
    """是否满足约束：交易次数、最大回撤。"""
    num_trades = trade_stats.get("total_trades", 0)
    max_dd = perf.get("max_drawdown", 0)
    if max_dd is None:
        max_dd = 0
    return num_trades >= MIN_TRADES and max_dd > MAX_DRAWDOWN_THRESHOLD


async def grid_search(
    system: Any,
    start_date: str,
    end_date: str,
    objective: str = "total_return"
) -> List[Dict[str, Any]]:
    """网格搜索，返回按目标指标降序排列的结果列表。"""
    grid = get_param_grid(small=os.environ.get("PARAM_GRID_SMALL", "").lower() in ("1", "true", "yes"))
    results = []
    for i, params in enumerate(grid):
        perf, trade_stats = await run_single_backtest(system, start_date, end_date, params)
        score = perf.get(objective)
        if score is None:
            score = -1e9
        valid = is_valid_result(perf, trade_stats)
        results.append({
            "params": params,
            "sharpe_ratio": perf.get("sharpe_ratio"),
            "calmar_ratio": perf.get("calmar_ratio"),
            "total_return": perf.get("total_return"),
            "max_drawdown": perf.get("max_drawdown"),
            "total_trades": trade_stats.get("total_trades"),
            "win_rate": trade_stats.get("win_rate"),
            "score": score,
            "valid": valid,
        })
    # 按 score 降序，有效结果优先
    results.sort(key=lambda x: (x["valid"], x["score"] if x["score"] is not None else -1e9), reverse=True)
    return results


def optuna_search(
    system: Any,
    start_date: str,
    end_date: str,
    n_trials: int = 30,
    objective: str = "total_return"
) -> List[Dict[str, Any]]:
    """使用 Optuna 做贝叶斯式参数优化（若已安装 optuna）。"""
    try:
        import optuna
    except ImportError:
        print("未安装 optuna，跳过 Optuna 搜索。可执行: pip install optuna")
        return []

    all_results: List[Dict[str, Any]] = []

    def objective_fn(trial: "optuna.Trial") -> float:
        params = {
            "short_period": trial.suggest_int("short_period", 5, 12),
            "long_period": trial.suggest_int("long_period", 18, 45),
            "rsi_max_open": trial.suggest_int("rsi_max_open", 60, 78),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.003, 0.015),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.008, 0.025),
            "max_positions": trial.suggest_int("max_positions", 2, 4),
            "rsi_period": 14,
            "volume_ma_period": 20,
            "use_volume_filter": True,
            "max_exposure_pct": 0.6,
            "position_pct": 0.2,
        }
        if params["long_period"] <= params["short_period"]:
            return -1e9
        # 每个 trial 使用独立事件循环，避免与外部 asyncio 冲突
        loop = asyncio.new_event_loop()
        try:
            perf, trade_stats = loop.run_until_complete(
                run_single_backtest(system, start_date, end_date, params)
            )
        finally:
            loop.close()
        score = perf.get(objective)
        if score is None:
            score = -1e9
        if not is_valid_result(perf, trade_stats):
            score = -1e9
        all_results.append({
            "params": params,
            "sharpe_ratio": perf.get("sharpe_ratio"),
            "calmar_ratio": perf.get("calmar_ratio"),
            "total_return": perf.get("total_return"),
            "max_drawdown": perf.get("max_drawdown"),
            "total_trades": trade_stats.get("total_trades"),
            "win_rate": trade_stats.get("win_rate"),
            "score": score,
        })
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    all_results.sort(key=lambda x: (x["score"] or -1e9), reverse=True)
    return all_results


async def main_async():
    import warnings
    warnings.filterwarnings("ignore")
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)

    from main import CryptoQuantSystem

    print("初始化系统（数据/因子/模型/风控/执行）...")
    system = CryptoQuantSystem()
    await system.initialize()

    start_date = os.environ.get("BACKTEST_START", DEFAULT_START)
    end_date = os.environ.get("BACKTEST_END", DEFAULT_END)
    mode = os.environ.get("PARAM_SEARCH_MODE", "grid")  # grid | optuna | both
    n_optuna_trials = int(os.environ.get("OPTUNA_TRIALS", "25"))

    if mode in ("grid", "both"):
        print(f"\n网格搜索 (start={start_date}, end={end_date})...")
        grid_results = await grid_search(system, start_date, end_date, objective="total_return")
        print(f"共 {len(grid_results)} 组参数，有效（满足约束）组数: {sum(1 for r in grid_results if r['valid'])}")
        top = grid_results[:5]
        print("\nTop 5 参数组合 (按总收益率):")
        for i, r in enumerate(top, 1):
            sr = r.get("sharpe_ratio")
            tr = r.get("total_return") or 0
            md = r.get("max_drawdown") or 0
            nt = r.get("total_trades") or 0
            print(f"  {i}. 夏普={(sr if sr is not None else 0):.4f} 收益={tr*100:.2f}% 回撤={md*100:.2f}% 交易数={nt} 有效={r['valid']}")
            print(f"      params: {r['params']}")
        best = grid_results[0]
        if best["valid"]:
            print(f"\n推荐参数（网格）: {json.dumps(best['params'], indent=2)}")
        out_path = PROJECT_ROOT / "reports" / "param_search_grid.json"
        payload = [_to_json_serializable({**r, "params": r["params"]}) for r in grid_results[:20]]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"前 20 组已写入: {out_path}")

    if mode in ("optuna", "both"):
        print(f"\nOptuna 搜索 (n_trials={n_optuna_trials})...")
        optuna_results = optuna_search(system, start_date, end_date, n_trials=n_optuna_trials, objective="total_return")
        if optuna_results:
            best_o = optuna_results[0]
            so = best_o.get("sharpe_ratio") or 0
            to = best_o.get("total_return") or 0
            mo = best_o.get("max_drawdown") or 0
            print(f"Optuna 最佳 夏普={so:.4f} 收益={to*100:.2f}% 回撤={mo*100:.2f}%")
            print(f"推荐参数（Optuna）: {json.dumps(best_o['params'], indent=2)}")
            out_path = PROJECT_ROOT / "reports" / "param_search_optuna.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(_to_json_serializable(optuna_results[:15]), f, indent=2, ensure_ascii=False)
            print(f"前 15 组已写入: {out_path}")

    print("\n参数搜索完成。")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
