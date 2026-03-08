"""
风险指标计算模块

提供各种风险指标的计算，包括VaR、CVaR、夏普比率、Calmar比率等
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class VaRMethod(Enum):
    """VaR计算方法"""
    HISTORICAL = "historical"  # 历史模拟法
    PARAMETRIC = "parametric"  # 参数法（方差-协方差）
    MONTE_CARLO = "monte_carlo"  # 蒙特卡洛模拟


@dataclass
class RiskMetricsResult:
    """风险指标结果"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    volatility: float  # 波动率
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # Calmar比率
    max_drawdown: float  # 最大回撤
    max_drawdown_duration: int  # 最大回撤持续时间
    beta: Optional[float] = None  # Beta系数
    alpha: Optional[float] = None  # Alpha系数
    information_ratio: Optional[float] = None  # 信息比率
    skewness: float = 0.0  # 偏度
    kurtosis: float = 0.0  # 峰度


class RiskMetricsCalculator:
    """风险指标计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化风险指标计算器
        
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算收益率序列
        
        Args:
            prices: 价格序列
            
        Returns:
            收益率序列
        """
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算对数收益率序列
        
        Args:
            prices: 价格序列
            
        Returns:
            对数收益率序列
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
        time_horizon: int = 1,
        num_simulations: int = 10000
    ) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平 (0-1)
            method: 计算方法
            time_horizon: 时间 horizon (天数)
            num_simulations: 蒙特卡洛模拟次数
            
        Returns:
            VaR值 (负值表示损失)
        """
        if len(returns) == 0:
            return 0.0
        
        alpha = 1 - confidence_level
        
        if method == VaRMethod.HISTORICAL:
            # 历史模拟法
            var = np.percentile(returns, alpha * 100)
            
        elif method == VaRMethod.PARAMETRIC:
            # 参数法
            mean = returns.mean()
            std = returns.std()
            var = mean - std * np.percentile(np.random.standard_normal(10000), confidence_level * 100)
            
        elif method == VaRMethod.MONTE_CARLO:
            # 蒙特卡洛模拟
            mean = returns.mean()
            std = returns.std()
            simulated_returns = np.random.normal(mean, std, num_simulations)
            var = np.percentile(simulated_returns, alpha * 100)
        
        # 调整时间 horizon
        var = var * np.sqrt(time_horizon)
        
        return var
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        计算条件风险价值(CVaR) / Expected Shortfall
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            time_horizon: 时间 horizon
            
        Returns:
            CVaR值
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level, VaRMethod.HISTORICAL)
        cvar = returns[returns <= var].mean()
        
        if pd.isna(cvar):
            cvar = var
        
        return cvar * np.sqrt(time_horizon)
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
        periods_per_year: int = 365
    ) -> float:
        """
        计算波动率
        
        Args:
            returns: 收益率序列
            annualize: 是否年化
            periods_per_year: 每年周期数
            
        Returns:
            波动率
        """
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            vol = vol * np.sqrt(periods_per_year)
        
        return vol
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年周期数
            
        Returns:
            夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - self.risk_free_rate / periods_per_year
        sharpe = excess_returns / returns.std() * np.sqrt(periods_per_year)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 365,
        target_return: float = 0.0
    ) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年周期数
            target_return: 目标收益率
            
        Returns:
            索提诺比率
        """
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if downside_std == 0:
            return float('inf')
        
        excess_return = returns.mean() - self.risk_free_rate / periods_per_year
        sortino = excess_return / downside_std * np.sqrt(periods_per_year)
        
        return sortino
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, int, int]:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            (最大回撤比例, 回撤开始索引, 回撤结束索引)
        """
        if len(prices) == 0:
            return 0.0, 0, 0
        
        # 计算累计最大值
        cumulative_max = prices.cummax()
        
        # 计算回撤
        drawdown = (prices - cumulative_max) / cumulative_max
        
        # 找到最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.loc[max_dd_idx]
        
        # 找到回撤开始点（峰值）
        peak_idx = prices.loc[:max_dd_idx].idxmax()
        
        # 转换为位置索引
        start_idx = prices.index.get_loc(peak_idx)
        end_idx = prices.index.get_loc(max_dd_idx)
        
        return max_dd, start_idx, end_idx
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        prices: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        计算Calmar比率
        
        Args:
            returns: 收益率序列
            prices: 价格序列
            periods_per_year: 每年周期数
            
        Returns:
            Calmar比率
        """
        if len(returns) == 0 or len(prices) == 0:
            return 0.0
        
        max_dd, _, _ = self.calculate_max_drawdown(prices)
        
        if max_dd == 0:
            return float('inf')
        
        annual_return = returns.mean() * periods_per_year
        calmar = annual_return / abs(max_dd)
        
        return calmar
    
    def calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        计算Beta系数
        
        Args:
            returns: 资产收益率
            benchmark_returns: 基准收益率
            
        Returns:
            Beta系数
        """
        # 对齐数据
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        asset_rets = aligned_data.iloc[:, 0]
        bench_rets = aligned_data.iloc[:, 1]
        
        covariance = asset_rets.cov(bench_rets)
        benchmark_variance = bench_rets.var()
        
        if benchmark_variance == 0:
            return 1.0
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        计算Alpha系数
        
        Args:
            returns: 资产收益率
            benchmark_returns: 基准收益率
            periods_per_year: 每年周期数
            
        Returns:
            Alpha系数（年化）
        """
        beta = self.calculate_beta(returns, benchmark_returns)
        
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return 0.0
        
        asset_rets = aligned_data.iloc[:, 0]
        bench_rets = aligned_data.iloc[:, 1]
        
        alpha = (asset_rets.mean() - self.risk_free_rate / periods_per_year) - \
                beta * (bench_rets.mean() - self.risk_free_rate / periods_per_year)
        
        return alpha * periods_per_year
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        计算信息比率
        
        Args:
            returns: 资产收益率
            benchmark_returns: 基准收益率
            periods_per_year: 每年周期数
            
        Returns:
            信息比率
        """
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        asset_rets = aligned_data.iloc[:, 0]
        bench_rets = aligned_data.iloc[:, 1]
        
        tracking_error = (asset_rets - bench_rets).std()
        
        if tracking_error == 0:
            return 0.0
        
        excess_return = (asset_rets.mean() - bench_rets.mean()) * periods_per_year
        ir = excess_return / (tracking_error * np.sqrt(periods_per_year))
        
        return ir
    
    def calculate_all_metrics(
        self,
        prices: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        periods_per_year: int = 365
    ) -> RiskMetricsResult:
        """
        计算所有风险指标
        
        Args:
            prices: 资产价格序列
            benchmark_prices: 基准价格序列（可选）
            periods_per_year: 每年周期数
            
        Returns:
            风险指标结果
        """
        returns = self.calculate_returns(prices)
        
        if len(returns) == 0:
            return RiskMetricsResult(
                var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
                volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, max_drawdown=0.0, max_drawdown_duration=0
            )
        
        # 计算VaR和CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # 计算波动率
        volatility = self.calculate_volatility(returns, True, periods_per_year)
        
        # 计算比率
        sharpe = self.calculate_sharpe_ratio(returns, periods_per_year)
        sortino = self.calculate_sortino_ratio(returns, periods_per_year)
        calmar = self.calculate_calmar_ratio(returns, prices, periods_per_year)
        
        # 计算最大回撤
        max_dd, start_idx, end_idx = self.calculate_max_drawdown(prices)
        max_dd_duration = end_idx - start_idx
        
        # 计算统计量
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 计算相对于基准的指标
        beta, alpha, info_ratio = None, None, None
        if benchmark_prices is not None:
            benchmark_returns = self.calculate_returns(benchmark_prices)
            beta = self.calculate_beta(returns, benchmark_returns)
            alpha = self.calculate_alpha(returns, benchmark_returns, periods_per_year)
            info_ratio = self.calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        
        return RiskMetricsResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            beta=beta,
            alpha=alpha,
            information_ratio=info_ratio,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def calculate_portfolio_var(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        计算投资组合VaR
        
        Args:
            returns_df: 各资产收益率数据框
            weights: 资产权重
            confidence_level: 置信水平
            
        Returns:
            组合VaR
        """
        if len(returns_df) == 0:
            return 0.0
        
        # 计算组合收益率
        portfolio_returns = returns_df.dot(weights)
        
        return self.calculate_var(portfolio_returns, confidence_level)
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率相关性矩阵
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            相关性矩阵
        """
        return returns_df.corr()
    
    def calculate_covariance_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率协方差矩阵
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            协方差矩阵
        """
        return returns_df.cov()


def calculate_rolling_metrics(
    prices: pd.Series,
    window: int = 30,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365
) -> pd.DataFrame:
    """
    计算滚动风险指标
    
    Args:
        prices: 价格序列
        window: 滚动窗口
        risk_free_rate: 无风险利率
        periods_per_year: 每年周期数
        
    Returns:
        滚动风险指标数据框
    """
    calculator = RiskMetricsCalculator(risk_free_rate)
    returns = calculator.calculate_returns(prices)
    
    results = {
        'volatility': [],
        'sharpe_ratio': [],
        'var_95': [],
        'max_drawdown': []
    }
    
    for i in range(window, len(prices) + 1):
        window_prices = prices.iloc[i-window:i]
        window_returns = returns.iloc[i-window:i]
        
        results['volatility'].append(calculator.calculate_volatility(window_returns, True, periods_per_year))
        results['sharpe_ratio'].append(calculator.calculate_sharpe_ratio(window_returns, periods_per_year))
        results['var_95'].append(calculator.calculate_var(window_returns, 0.95))
        
        dd, _, _ = calculator.calculate_max_drawdown(window_prices)
        results['max_drawdown'].append(dd)
    
    return pd.DataFrame(results, index=prices.index[window-1:])


# 便捷函数
def quick_metrics(prices: pd.Series, periods_per_year: int = 365) -> Dict[str, float]:
    """
    快速计算主要风险指标
    
    Args:
        prices: 价格序列
        periods_per_year: 每年周期数
        
    Returns:
        风险指标字典
    """
    calculator = RiskMetricsCalculator()
    metrics = calculator.calculate_all_metrics(prices, periods_per_year=periods_per_year)
    
    return {
        'volatility': metrics.volatility,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown,
        'var_95': metrics.var_95,
        'calmar_ratio': metrics.calmar_ratio,
        'sortino_ratio': metrics.sortino_ratio
    }
