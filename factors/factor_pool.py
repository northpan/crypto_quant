"""
因子池管理模块
Factor Pool Management Module

提供因子的统一管理、计算、检验和筛选功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from .base_factor import (
    BaseFactor, FactorCategory, FactorDirection, FactorMetadata,
    FactorTester, FactorValidator
)

# 导入所有因子
from .technical_factors import create_all_technical_factors
from .volume_factors import create_all_volume_factors
from .volatility_factors import create_all_volatility_factors
from .orderflow_factors import create_all_orderflow_factors
from .cross_market_factors import create_all_cross_market_factors

warnings.filterwarnings('ignore')


@dataclass
class FactorResult:
    """因子计算结果"""
    name: str
    values: pd.Series
    metadata: Dict[str, Any]
    computation_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'values': self.values.to_dict(),
            'metadata': self.metadata,
            'computation_time': self.computation_time
        }


@dataclass
class FactorTestResult:
    """因子检验结果"""
    name: str
    ic: float
    ic_pvalue: float
    quantile_returns: pd.DataFrame
    turnover: Dict[str, float]
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'ic': self.ic,
            'ic_pvalue': self.ic_pvalue,
            'quantile_returns': self.quantile_returns.to_dict(),
            'turnover': self.turnover,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown
        }


class FactorPool:
    """
    因子池管理类
    
    统一管理所有因子，提供批量计算、检验、筛选功能
    """
    
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_results: Dict[str, FactorResult] = {}
        self.test_results: Dict[str, FactorTestResult] = {}
        self.tester = FactorTester()
        self._register_default_factors()
    
    def _register_default_factors(self):
        """注册默认因子"""
        # 技术指标因子
        technical_factors = create_all_technical_factors()
        for factor in technical_factors:
            self.register(factor)
        
        # 量价因子
        volume_factors = create_all_volume_factors()
        for factor in volume_factors:
            self.register(factor)
        
        # 波动率因子
        volatility_factors = create_all_volatility_factors()
        for factor in volatility_factors:
            self.register(factor)
        
        # 订单流因子
        orderflow_factors = create_all_orderflow_factors()
        for factor in orderflow_factors:
            self.register(factor)
        
        # 跨市场因子
        cross_market_factors = create_all_cross_market_factors()
        for factor in cross_market_factors:
            self.register(factor)
    
    def register(self, factor: BaseFactor) -> 'FactorPool':
        """
        注册因子
        
        Args:
            factor: 因子实例
            
        Returns:
            FactorPool: 支持链式调用
        """
        self.factors[factor.metadata.name] = factor
        return self
    
    def unregister(self, name: str) -> 'FactorPool':
        """
        注销因子
        
        Args:
            name: 因子名称
            
        Returns:
            FactorPool: 支持链式调用
        """
        if name in self.factors:
            del self.factors[name]
        return self
    
    def get(self, name: str) -> Optional[BaseFactor]:
        """
        获取因子
        
        Args:
            name: 因子名称
            
        Returns:
            Optional[BaseFactor]: 因子实例或None
        """
        return self.factors.get(name)
    
    def list_factors(self, category: Optional[FactorCategory] = None) -> List[str]:
        """
        列出因子名称
        
        Args:
            category: 因子分类过滤
            
        Returns:
            List[str]: 因子名称列表
        """
        if category is None:
            return list(self.factors.keys())
        else:
            return [
                name for name, factor in self.factors.items()
                if factor.metadata.category == category
            ]
    
    def get_factor_info(self, name: str) -> Optional[Dict]:
        """
        获取因子信息
        
        Args:
            name: 因子名称
            
        Returns:
            Optional[Dict]: 因子信息字典
        """
        factor = self.get(name)
        if factor is None:
            return None
        return factor.get_info()
    
    def compute(self, data: pd.DataFrame, factor_names: Optional[List[str]] = None,
                verbose: bool = False, n_jobs: int = 1) -> pd.DataFrame:
        """
        批量计算因子值
        
        Args:
            data: 输入数据
            factor_names: 指定因子列表，None表示全部
            verbose: 是否显示进度
            n_jobs: 并行计算线程数
            
        Returns:
            pd.DataFrame: 因子值DataFrame
        """
        import time
        
        # 确定要计算的因子
        if factor_names is None:
            factor_names = list(self.factors.keys())
        
        results = {}
        
        if n_jobs > 1:
            # 并行计算
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(self._compute_single, name, data): name 
                    for name in factor_names
                }
                
                for future in futures:
                    name = futures[future]
                    try:
                        result = future.result()
                        results[name] = result
                        if verbose:
                            print(f"Computed: {name}")
                    except Exception as e:
                        if verbose:
                            print(f"Error computing {name}: {e}")
        else:
            # 串行计算
            for name in factor_names:
                try:
                    start_time = time.time()
                    result = self._compute_single(name, data)
                    computation_time = time.time() - start_time
                    
                    results[name] = result
                    self.factor_results[name] = FactorResult(
                        name=name,
                        values=result,
                        metadata=self.factors[name].get_info(),
                        computation_time=computation_time
                    )
                    
                    if verbose:
                        print(f"Computed: {name} ({computation_time:.3f}s)")
                except Exception as e:
                    if verbose:
                        print(f"Error computing {name}: {e}")
        
        # 合并结果
        result_df = pd.DataFrame(results)
        result_df.index = data.index
        
        return result_df
    
    def _compute_single(self, name: str, data: pd.DataFrame) -> pd.Series:
        """计算单个因子"""
        factor = self.factors[name]
        return factor.compute(data)
    
    def compute_single(self, name: str, data: pd.DataFrame) -> pd.Series:
        """
        计算单个因子
        
        Args:
            name: 因子名称
            data: 输入数据
            
        Returns:
            pd.Series: 因子值
        """
        return self._compute_single(name, data)
    
    def test(self, factor_values: pd.DataFrame, forward_return: pd.Series,
             n_quantiles: int = 5, verbose: bool = False) -> pd.DataFrame:
        """
        批量检验因子有效性
        
        Args:
            factor_values: 因子值DataFrame
            forward_return: 未来收益序列
            n_quantiles: 分位数数量
            verbose: 是否显示进度
            
        Returns:
            pd.DataFrame: 检验结果汇总
        """
        results = []
        
        for name in factor_values.columns:
            try:
                result = self._test_single(
                    name, factor_values[name], forward_return, n_quantiles
                )
                self.test_results[name] = result
                results.append({
                    'name': name,
                    'ic': result.ic,
                    'ic_pvalue': result.ic_pvalue,
                    'abs_ic': abs(result.ic),
                    'sharpe': result.sharpe,
                    'max_drawdown': result.max_drawdown,
                    'turnover_mean': result.turnover.get('mean_turnover', 0),
                    'significant': result.ic_pvalue < 0.05
                })
                
                if verbose:
                    print(f"Tested: {name}, IC={result.ic:.4f}, p={result.ic_pvalue:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Error testing {name}: {e}")
        
        return pd.DataFrame(results)
    
    def _test_single(self, name: str, factor: pd.Series, 
                     forward_return: pd.Series, n_quantiles: int) -> FactorTestResult:
        """检验单个因子"""
        # IC分析
        ic_result = self.tester.ic_analysis(factor, forward_return)
        
        # 分位数收益
        quantile_returns = self.tester.quantile_return(factor, forward_return, n_quantiles)
        
        # 换手率
        turnover = self.tester.turnover_analysis(factor)
        
        # 计算多空组合收益
        long_short_return = quantile_returns.iloc[-1]['mean_return'] - quantile_returns.iloc[0]['mean_return']
        
        # 计算Sharpe（简化版）
        sharpe = long_short_return / (quantile_returns['std_return'].mean() + 1e-8)
        
        return FactorTestResult(
            name=name,
            ic=ic_result['ic'],
            ic_pvalue=ic_result['p_value'],
            quantile_returns=quantile_returns,
            turnover=turnover,
            sharpe=sharpe,
            max_drawdown=0.0  # 需要更详细数据计算
        )
    
    def select_factors(self, test_results: pd.DataFrame,
                       min_ic: float = 0.03,
                       max_pvalue: float = 0.05,
                       min_sharpe: float = 0.5) -> List[str]:
        """
        根据检验结果筛选有效因子
        
        Args:
            test_results: 检验结果DataFrame
            min_ic: 最小IC阈值
            max_pvalue: 最大p值阈值
            min_sharpe: 最小Sharpe阈值
            
        Returns:
            List[str]: 有效因子名称列表
        """
        selected = test_results[
            (test_results['abs_ic'] >= min_ic) &
            (test_results['ic_pvalue'] <= max_pvalue) &
            (test_results['sharpe'] >= min_sharpe)
        ]
        
        return selected['name'].tolist()
    
    def rank_factors(self, test_results: pd.DataFrame,
                     ic_weight: float = 0.4,
                     sharpe_weight: float = 0.4,
                     turnover_weight: float = 0.2) -> pd.DataFrame:
        """
        对因子进行排名
        
        Args:
            test_results: 检验结果DataFrame
            ic_weight: IC权重
            sharpe_weight: Sharpe权重
            turnover_weight: 换手率权重
            
        Returns:
            pd.DataFrame: 排名结果
        """
        # 标准化指标
        results = test_results.copy()
        results['ic_score'] = (results['abs_ic'] - results['abs_ic'].min()) / (results['abs_ic'].max() - results['abs_ic'].min() + 1e-8)
        results['sharpe_score'] = (results['sharpe'] - results['sharpe'].min()) / (results['sharpe'].max() - results['sharpe'].min() + 1e-8)
        results['turnover_score'] = 1 - (results['turnover_mean'] - results['turnover_mean'].min()) / (results['turnover_mean'].max() - results['turnover_mean'].min() + 1e-8)
        
        # 综合得分
        results['score'] = (
            ic_weight * results['ic_score'] +
            sharpe_weight * results['sharpe_score'] +
            turnover_weight * results['turnover_score']
        )
        
        # 排序
        results = results.sort_values('score', ascending=False)
        
        return results[['name', 'ic', 'sharpe', 'turnover_mean', 'score']]
    
    def remove_correlated_factors(self, factor_values: pd.DataFrame,
                                  threshold: float = 0.9) -> List[str]:
        """
        去除高度相关的因子
        
        Args:
            factor_values: 因子值DataFrame
            threshold: 相关系数阈值
            
        Returns:
            List[str]: 保留的因子名称列表
        """
        # 计算相关系数矩阵
        corr_matrix = factor_values.corr().abs()
        
        # 上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出高度相关的因子对
        to_drop = set()
        for column in upper.columns:
            if any(upper[column] > threshold):
                to_drop.add(column)
        
        # 保留的因子
        kept = [col for col in factor_values.columns if col not in to_drop]
        
        return kept
    
    def get_factor_stats(self) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Returns:
            pd.DataFrame: 因子统计信息
        """
        stats = []
        
        for name, factor in self.factors.items():
            info = factor.get_info()
            stats.append({
                'name': name,
                'category': info['category'],
                'direction': info['direction'],
                'description': info['description'],
                'parameters': str(info['parameters'])
            })
        
        return pd.DataFrame(stats)
    
    def export_results(self, filepath: str):
        """
        导出因子结果到JSON
        
        Args:
            filepath: 文件路径
        """
        results = {
            'factor_results': {
                name: {
                    'metadata': result.metadata,
                    'computation_time': result.computation_time
                }
                for name, result in self.factor_results.items()
            },
            'test_results': {
                name: {
                    'ic': result.ic,
                    'ic_pvalue': result.ic_pvalue,
                    'sharpe': result.sharpe,
                    'turnover': result.turnover
                }
                for name, result in self.test_results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def __len__(self) -> int:
        """返回因子数量"""
        return len(self.factors)
    
    def __contains__(self, name: str) -> bool:
        """检查因子是否存在"""
        return name in self.factors


class FactorPipeline:
    """
    因子处理流水线
    
    提供因子计算、处理、筛选的流水线功能
    """
    
    def __init__(self):
        self.pool = FactorPool()
        self.steps: List[Callable] = []
    
    def add_step(self, func: Callable, **kwargs) -> 'FactorPipeline':
        """
        添加处理步骤
        
        Args:
            func: 处理函数
            **kwargs: 函数参数
            
        Returns:
            FactorPipeline: 支持链式调用
        """
        self.steps.append((func, kwargs))
        return self
    
    def run(self, data: pd.DataFrame, forward_return: Optional[pd.Series] = None) -> Dict:
        """
        运行流水线
        
        Args:
            data: 输入数据
            forward_return: 未来收益（用于检验）
            
        Returns:
            Dict: 处理结果
        """
        results = {
            'factor_values': None,
            'test_results': None,
            'selected_factors': None
        }
        
        # 计算所有因子
        print("Computing factors...")
        factor_values = self.pool.compute(data, verbose=True)
        results['factor_values'] = factor_values
        
        # 执行流水线步骤
        for func, kwargs in self.steps:
            if func.__name__ == 'test':
                if forward_return is not None:
                    print("Testing factors...")
                    test_results = self.pool.test(factor_values, forward_return, **kwargs)
                    results['test_results'] = test_results
            elif func.__name__ == 'select':
                if results['test_results'] is not None:
                    print("Selecting factors...")
                    selected = self.pool.select_factors(results['test_results'], **kwargs)
                    results['selected_factors'] = selected
            elif func.__name__ == 'remove_correlated':
                print("Removing correlated factors...")
                kept = self.pool.remove_correlated_factors(factor_values, **kwargs)
                results['uncorrelated_factors'] = kept
        
        return results


# ==================== 便捷函数 ====================

def create_factor_pool() -> FactorPool:
    """创建默认因子池"""
    return FactorPool()


def compute_all_factors(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    计算所有因子
    
    Args:
        data: 输入数据
        verbose: 是否显示进度
        
    Returns:
        pd.DataFrame: 因子值
    """
    pool = create_factor_pool()
    return pool.compute(data, verbose=verbose)


def test_factors(factor_values: pd.DataFrame, forward_return: pd.Series,
                 verbose: bool = True) -> pd.DataFrame:
    """
    检验因子有效性
    
    Args:
        factor_values: 因子值
        forward_return: 未来收益
        verbose: 是否显示进度
        
    Returns:
        pd.DataFrame: 检验结果
    """
    pool = create_factor_pool()
    return pool.test(factor_values, forward_return, verbose=verbose)


def get_top_factors(data: pd.DataFrame, forward_return: pd.Series,
                    n_top: int = 10, verbose: bool = True) -> pd.DataFrame:
    """
    获取表现最好的因子
    
    Args:
        data: 输入数据
        forward_return: 未来收益
        n_top: 返回前N个因子
        verbose: 是否显示进度
        
    Returns:
        pd.DataFrame: 排名结果
    """
    pool = create_factor_pool()
    
    # 计算因子
    factor_values = pool.compute(data, verbose=verbose)
    
    # 检验因子
    test_results = pool.test(factor_values, forward_return, verbose=verbose)
    
    # 排名
    ranked = pool.rank_factors(test_results)
    
    return ranked.head(n_top)


def quick_factor_analysis(data: pd.DataFrame, forward_return: pd.Series) -> Dict:
    """
    快速因子分析
    
    Args:
        data: 输入数据
        forward_return: 未来收益
        
    Returns:
        Dict: 分析结果
    """
    pool = create_factor_pool()
    
    print(f"Total factors: {len(pool)}")
    
    # 计算因子
    print("\nComputing factors...")
    factor_values = pool.compute(data, verbose=True)
    
    # 检验因子
    print("\nTesting factors...")
    test_results = pool.test(factor_values, forward_return, verbose=True)
    
    # 筛选有效因子
    print("\nSelecting factors...")
    selected = pool.select_factors(test_results)
    print(f"Selected {len(selected)} factors")
    
    # 排名
    ranked = pool.rank_factors(test_results)
    
    return {
        'total_factors': len(pool),
        'computed_factors': len(factor_values.columns),
        'significant_factors': len(test_results[test_results['significant']]),
        'selected_factors': selected,
        'top_factors': ranked.head(10),
        'test_results': test_results,
        'factor_values': factor_values
    }


# 导出所有公共接口
__all__ = [
    'FactorPool', 'FactorPipeline', 'FactorResult', 'FactorTestResult',
    'create_factor_pool', 'compute_all_factors', 'test_factors',
    'get_top_factors', 'quick_factor_analysis'
]
