"""
过拟合检测模块
包含多种过拟合检测方法:
- 样本外测试
- 参数敏感性分析
- 蒙特卡洛模拟
- 组合对称交叉验证 (CSCV)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')


@dataclass
class OverfittingTestResult:
    """过拟合测试结果"""
    test_name: str
    is_overfitted: bool
    probability: float
    statistic: float
    p_value: float
    details: Dict = field(default_factory=dict)
    recommendation: str = ""


class OutOfSampleTest:
    """
    样本外测试
    
    将数据分为训练集和测试集，评估策略在样本外的表现
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        n_splits: int = 5
    ):
        self.train_ratio = train_ratio
        self.n_splits = n_splits
    
    def test(
        self,
        returns: pd.Series,
        strategy_func: Callable,
        **strategy_params
    ) -> OverfittingTestResult:
        """
        执行样本外测试
        
        Args:
            returns: 收益率序列
            strategy_func: 策略函数
            **strategy_params: 策略参数
        
        Returns:
            测试结果
        """
        n = len(returns)
        train_size = int(n * self.train_ratio)
        
        # 训练集表现
        train_returns = returns.iloc[:train_size]
        train_sharpe = self._calculate_sharpe(train_returns)
        
        # 测试集表现
        test_returns = returns.iloc[train_size:]
        test_sharpe = self._calculate_sharpe(test_returns)
        
        # 计算表现衰减
        if train_sharpe != 0:
            decay_ratio = test_sharpe / train_sharpe
        else:
            decay_ratio = 0
        
        # 判断是否过拟合
        is_overfitted = decay_ratio < 0.5 or test_sharpe < 0
        
        # 计算概率
        probability = 1 - decay_ratio if decay_ratio > 0 else 1
        
        return OverfittingTestResult(
            test_name="Out-of-Sample Test",
            is_overfitted=is_overfitted,
            probability=min(probability, 1.0),
            statistic=decay_ratio,
            p_value=1 - decay_ratio if decay_ratio > 0 else 1,
            details={
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'decay_ratio': decay_ratio,
                'train_size': train_size,
                'test_size': n - train_size
            },
            recommendation="策略在样本外表现显著下降，可能存在过拟合" if is_overfitted else "样本外表现良好"
        )
    
    def cross_validate(
        self,
        returns: pd.Series,
        strategy_func: Callable,
        **strategy_params
    ) -> List[OverfittingTestResult]:
        """交叉验证"""
        n = len(returns)
        fold_size = n // self.n_splits
        results = []
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n
            
            # 训练集
            train_indices = list(range(0, test_start)) + list(range(test_end, n))
            train_returns = returns.iloc[train_indices]
            
            # 测试集
            test_returns = returns.iloc[test_start:test_end]
            
            train_sharpe = self._calculate_sharpe(train_returns)
            test_sharpe = self._calculate_sharpe(test_returns)
            
            decay_ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0
            
            results.append(OverfittingTestResult(
                test_name=f"CV Fold {i+1}",
                is_overfitted=decay_ratio < 0.5,
                probability=1 - decay_ratio if decay_ratio > 0 else 1,
                statistic=decay_ratio,
                p_value=0,
                details={
                    'train_sharpe': train_sharpe,
                    'test_sharpe': test_sharpe,
                    'decay_ratio': decay_ratio
                }
            ))
        
        return results
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        excess_return = returns.mean() * 365 - risk_free_rate
        volatility = returns.std() * np.sqrt(365)
        return excess_return / volatility


class ParameterSensitivityTest:
    """
    参数敏感性分析
    
    测试策略对参数变化的敏感程度
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, List],
        metric: str = "sharpe"
    ):
        self.param_ranges = param_ranges
        self.metric = metric
    
    def test(
        self,
        returns: pd.Series,
        strategy_func: Callable,
        **base_params
    ) -> OverfittingTestResult:
        """
        执行参数敏感性测试
        
        Args:
            returns: 收益率序列
            strategy_func: 策略函数
            **base_params: 基础参数
        
        Returns:
            测试结果
        """
        results = []
        
        # 对每个参数进行敏感性测试
        for param_name, param_values in self.param_ranges.items():
            param_results = []
            
            for value in param_values:
                test_params = base_params.copy()
                test_params[param_name] = value
                
                try:
                    result = strategy_func(returns, **test_params)
                    metric_value = self._extract_metric(result)
                    param_results.append(metric_value)
                except Exception as e:
                    param_results.append(np.nan)
            
            # 计算参数敏感性
            valid_results = [r for r in param_results if not np.isnan(r)]
            if len(valid_results) > 1:
                sensitivity = np.std(valid_results) / abs(np.mean(valid_results)) if np.mean(valid_results) != 0 else 0
                results.append({
                    'param': param_name,
                    'sensitivity': sensitivity,
                    'values': param_values,
                    'results': param_results
                })
        
        # 计算整体敏感性
        if results:
            avg_sensitivity = np.mean([r['sensitivity'] for r in results])
            max_sensitivity = np.max([r['sensitivity'] for r in results])
        else:
            avg_sensitivity = 0
            max_sensitivity = 0
        
        # 判断是否过拟合
        is_overfitted = max_sensitivity > 1.0 or avg_sensitivity > 0.5
        
        return OverfittingTestResult(
            test_name="Parameter Sensitivity Test",
            is_overfitted=is_overfitted,
            probability=min(max_sensitivity, 1.0),
            statistic=max_sensitivity,
            p_value=0,
            details={
                'avg_sensitivity': avg_sensitivity,
                'max_sensitivity': max_sensitivity,
                'param_results': results
            },
            recommendation="参数过于敏感，策略可能过拟合" if is_overfitted else "参数稳定性良好"
        )
    
    def grid_search(
        self,
        returns: pd.Series,
        strategy_func: Callable,
        **param_ranges
    ) -> pd.DataFrame:
        """
        网格搜索最佳参数
        
        Args:
            returns: 收益率序列
            strategy_func: 策略函数
            **param_ranges: 参数范围
        
        Returns:
            搜索结果DataFrame
        """
        from itertools import product
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        results = []
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                result = strategy_func(returns, **params)
                metric_value = self._extract_metric(result)
                
                row = params.copy()
                row[self.metric] = metric_value
                results.append(row)
            except Exception as e:
                row = params.copy()
                row[self.metric] = np.nan
                results.append(row)
        
        return pd.DataFrame(results)
    
    def _extract_metric(self, result: Any) -> float:
        """提取指标值"""
        if isinstance(result, dict):
            return result.get(self.metric, 0)
        elif isinstance(result, (int, float)):
            return float(result)
        else:
            return 0


class MonteCarloSimulation:
    """
    蒙特卡洛模拟
    
    通过随机打乱交易顺序来测试策略的稳健性
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def test(
        self,
        returns: pd.Series,
        strategy_func: Optional[Callable] = None,
        **strategy_params
    ) -> OverfittingTestResult:
        """
        执行蒙特卡洛模拟
        
        Args:
            returns: 收益率序列
            strategy_func: 策略函数 (可选)
            **strategy_params: 策略参数
        
        Returns:
            测试结果
        """
        original_sharpe = self._calculate_sharpe(returns)
        simulated_sharpes = []
        
        np.random.seed(42)
        
        for _ in range(self.n_simulations):
            # 随机打乱收益率
            shuffled_returns = returns.sample(frac=1).reset_index(drop=True)
            simulated_sharpe = self._calculate_sharpe(shuffled_returns)
            simulated_sharpes.append(simulated_sharpe)
        
        simulated_sharpes = np.array(simulated_sharpes)
        
        # 计算统计量
        mean_simulated = np.mean(simulated_sharpes)
        std_simulated = np.std(simulated_sharpes)
        
        # 计算p值
        if std_simulated > 0:
            z_score = (original_sharpe - mean_simulated) / std_simulated
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 0.5
        
        # 置信区间
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(simulated_sharpes, alpha/2 * 100)
        ci_upper = np.percentile(simulated_sharpes, (1 - alpha/2) * 100)
        
        # 判断是否过拟合
        is_overfitted = original_sharpe < ci_lower or p_value > 0.1
        
        return OverfittingTestResult(
            test_name="Monte Carlo Simulation",
            is_overfitted=is_overfitted,
            probability=1 - p_value,
            statistic=z_score,
            p_value=p_value,
            details={
                'original_sharpe': original_sharpe,
                'mean_simulated': mean_simulated,
                'std_simulated': std_simulated,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'simulated_sharpes': simulated_sharpes.tolist()
            },
            recommendation="策略表现与随机序列无显著差异，可能过拟合" if is_overfitted else "策略表现显著优于随机序列"
        )
    
    def simulate_equity_curves(
        self,
        returns: pd.Series,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        模拟多条权益曲线
        
        Args:
            returns: 收益率序列
            initial_capital: 初始资金
        
        Returns:
            模拟权益曲线DataFrame
        """
        equity_curves = []
        
        np.random.seed(42)
        
        for i in range(min(self.n_simulations, 100)):  # 限制显示数量
            shuffled_returns = returns.sample(frac=1).reset_index(drop=True)
            equity = initial_capital * (1 + shuffled_returns).cumprod()
            equity_curves.append(equity.values)
        
        return pd.DataFrame(equity_curves).T
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        excess_return = returns.mean() * 365 - risk_free_rate
        volatility = returns.std() * np.sqrt(365)
        return excess_return / volatility


class CSCVTest:
    """
    组合对称交叉验证 (Combinatorially Symmetric Cross-Validation)
    
    参考: Bailey et al. (2016) "The Probability of Backtest Overfitting"
    """
    
    def __init__(
        self,
        n_splits: int = 4,
        objective: str = "sharpe"
    ):
        self.n_splits = n_splits
        self.objective = objective
    
    def test(
        self,
        returns_matrix: pd.DataFrame,
        strategy_labels: Optional[List[str]] = None
    ) -> OverfittingTestResult:
        """
        执行CSCV测试
        
        Args:
            returns_matrix: 策略收益率矩阵 (每列是一个策略)
            strategy_labels: 策略标签
        
        Returns:
            测试结果
        """
        n_strategies = returns_matrix.shape[1]
        n_periods = returns_matrix.shape[0]
        
        if strategy_labels is None:
            strategy_labels = [f"Strategy_{i}" for i in range(n_strategies)]
        
        # 计算每个策略的IS和OOS表现
        is_scores = []
        oos_scores = []
        
        # 生成所有可能的组合
        period_indices = list(range(n_periods))
        half = n_periods // 2
        
        # 简化版本: 使用固定分割
        for i in range(self.n_splits):
            test_start = i * (n_periods // self.n_splits)
            test_end = test_start + (n_periods // self.n_splits)
            
            # IS: 训练集
            is_indices = list(range(0, test_start)) + list(range(test_end, n_periods))
            is_returns = returns_matrix.iloc[is_indices, :]
            
            # OOS: 测试集
            oos_returns = returns_matrix.iloc[test_start:test_end, :]
            
            # 计算表现
            is_score = self._calculate_objective(is_returns)
            oos_score = self._calculate_objective(oos_returns)
            
            is_scores.append(is_score)
            oos_scores.append(oos_score)
        
        is_scores = np.array(is_scores)
        oos_scores = np.array(oos_scores)
        
        # 计算PBO (Probability of Backtest Overfitting)
        # 简化的PBO计算
        rank_differences = []
        for i in range(len(is_scores)):
            is_rank = np.argsort(np.argsort(is_scores[i]))[0]
            oos_rank = np.argsort(np.argsort(oos_scores[i]))[0]
            rank_differences.append(abs(is_rank - oos_rank))
        
        avg_rank_diff = np.mean(rank_differences)
        max_rank_diff = len(is_scores[0]) - 1 if len(is_scores) > 0 else 0
        
        pbo = avg_rank_diff / max_rank_diff if max_rank_diff > 0 else 0
        
        # 判断是否过拟合
        is_overfitted = pbo > 0.5
        
        return OverfittingTestResult(
            test_name="CSCV Test",
            is_overfitted=is_overfitted,
            probability=pbo,
            statistic=avg_rank_diff,
            p_value=pbo,
            details={
                'pbo': pbo,
                'avg_rank_diff': avg_rank_diff,
                'n_strategies': n_strategies,
                'n_periods': n_periods
            },
            recommendation=f"PBO = {pbo:.2%}，存在过拟合风险" if is_overfitted else f"PBO = {pbo:.2%}，过拟合风险较低"
        )
    
    def _calculate_objective(self, returns: pd.DataFrame) -> np.ndarray:
        """计算目标函数值"""
        if self.objective == "sharpe":
            means = returns.mean() * 365
            stds = returns.std() * np.sqrt(365)
            return np.where(stds > 0, (means - 0.02) / stds, 0)
        elif self.objective == "return":
            return (1 + returns).prod() - 1
        else:
            return returns.mean()


class OverfittingDetector:
    """
    过拟合检测器
    
    综合多种过拟合检测方法
    """
    
    def __init__(self):
        self.tests: Dict[str, Any] = {
            'oos': OutOfSampleTest(),
            'sensitivity': ParameterSensitivityTest(param_ranges={}),
            'monte_carlo': MonteCarloSimulation(),
            'cscv': CSCVTest()
        }
        self.results: List[OverfittingTestResult] = []
    
    def run_all_tests(
        self,
        returns: pd.Series,
        strategy_func: Optional[Callable] = None,
        param_ranges: Optional[Dict] = None,
        **strategy_params
    ) -> Dict[str, OverfittingTestResult]:
        """
        运行所有过拟合测试
        
        Args:
            returns: 收益率序列
            strategy_func: 策略函数
            param_ranges: 参数范围
            **strategy_params: 策略参数
        
        Returns:
            测试结果字典
        """
        results = {}
        
        # 样本外测试
        try:
            oos_test = OutOfSampleTest()
            results['out_of_sample'] = oos_test.test(returns, strategy_func, **strategy_params)
        except Exception as e:
            print(f"OOS test failed: {e}")
        
        # 参数敏感性测试
        if param_ranges:
            try:
                sensitivity_test = ParameterSensitivityTest(param_ranges=param_ranges)
                results['parameter_sensitivity'] = sensitivity_test.test(
                    returns, strategy_func, **strategy_params
                )
            except Exception as e:
                print(f"Sensitivity test failed: {e}")
        
        # 蒙特卡洛模拟
        try:
            mc_test = MonteCarloSimulation()
            results['monte_carlo'] = mc_test.test(returns, strategy_func, **strategy_params)
        except Exception as e:
            print(f"Monte Carlo test failed: {e}")
        
        self.results = list(results.values())
        return results
    
    def get_overall_assessment(self) -> Dict:
        """获取整体评估"""
        if not self.results:
            return {
                'is_overfitted': False,
                'confidence': 0,
                'risk_level': 'unknown',
                'recommendation': '未进行测试'
            }
        
        overfitted_count = sum(1 for r in self.results if r.is_overfitted)
        total_tests = len(self.results)
        
        avg_probability = np.mean([r.probability for r in self.results])
        
        if overfitted_count >= total_tests * 0.5:
            risk_level = 'high'
            recommendation = '策略存在严重过拟合风险，建议重新设计'
        elif overfitted_count >= total_tests * 0.25:
            risk_level = 'medium'
            recommendation = '策略存在一定过拟合风险，建议优化参数'
        else:
            risk_level = 'low'
            recommendation = '策略过拟合风险较低，表现稳健'
        
        return {
            'is_overfitted': overfitted_count > 0,
            'confidence': avg_probability,
            'risk_level': risk_level,
            'overfitted_tests': overfitted_count,
            'total_tests': total_tests,
            'recommendation': recommendation
        }
    
    def get_summary(self) -> str:
        """获取测试摘要"""
        if not self.results:
            return "未进行过拟合测试"
        
        summary = ["=" * 60, "过拟合检测报告", "=" * 60]
        
        for result in self.results:
            summary.append(f"\n{result.test_name}:")
            summary.append(f"  是否过拟合: {'是' if result.is_overfitted else '否'}")
            summary.append(f"  概率: {result.probability:.2%}")
            summary.append(f"  统计量: {result.statistic:.4f}")
            summary.append(f"  P值: {result.p_value:.4f}")
            summary.append(f"  建议: {result.recommendation}")
        
        overall = self.get_overall_assessment()
        summary.extend([
            "\n" + "=" * 60,
            "整体评估:",
            "=" * 60,
            f"  过拟合风险: {overall['risk_level'].upper()}",
            f"  置信度: {overall['confidence']:.2%}",
            f"  过拟合测试数: {overall['overfitted_tests']}/{overall['total_tests']}",
            f"  建议: {overall['recommendation']}",
            "=" * 60
        ])
        
        return "\n".join(summary)


def generate_sample_strategy_returns(n: int = 365, seed: int = 42) -> pd.Series:
    """生成示例策略收益率"""
    np.random.seed(seed)
    
    # 生成带趋势和噪声的收益率
    trend = np.linspace(0.001, 0.002, n)
    noise = np.random.normal(0, 0.01, n)
    returns = trend + noise
    
    return pd.Series(returns)


if __name__ == "__main__":
    # 测试过拟合检测
    print("过拟合检测模块测试")
    print("=" * 60)
    
    # 生成测试数据
    returns = generate_sample_strategy_returns(500)
    
    # 创建检测器
    detector = OverfittingDetector()
    
    # 定义简单的策略函数
    def dummy_strategy(returns, window=20):
        return {'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0}
    
    # 参数范围
    param_ranges = {
        'window': [10, 20, 30, 50]
    }
    
    # 运行测试
    results = detector.run_all_tests(
        returns=returns,
        strategy_func=dummy_strategy,
        param_ranges=param_ranges
    )
    
    # 打印摘要
    print(detector.get_summary())
    
    # 蒙特卡洛模拟详细结果
    print("\n蒙特卡洛模拟结果:")
    mc_result = results.get('monte_carlo')
    if mc_result:
        print(f"  原始夏普: {mc_result.details['original_sharpe']:.4f}")
        print(f"  模拟平均夏普: {mc_result.details['mean_simulated']:.4f}")
        print(f"  95%置信区间: [{mc_result.details['ci_lower']:.4f}, {mc_result.details['ci_upper']:.4f}]")
    
    print("\n测试完成!")
