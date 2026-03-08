"""
绩效分析模块
计算各种风险调整收益指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # 回撤相关
    avg_drawdown: float = 0.0
    avg_drawdown_duration: float = 0.0
    recovery_time: int = 0
    
    # 收益分布
    skewness: float = 0.0
    kurtosis: float = 0.0
    positive_days: float = 0.0
    
    # 其他
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0


class PerformanceAnalyzer:
    """
    绩效分析器
    
    计算各种绩效指标:
    - 收益率指标
    - 风险指标
    - 风险调整收益指标
    - 回撤分析
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 无风险利率 (年化)
        trading_days_per_year: int = 365,  # 加密货币全年交易
        periods_per_day: int = 1  # 每天的数据点数
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.periods_per_day = periods_per_day
        self.periods_per_year = trading_days_per_year * periods_per_day
    
    def analyze(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        equity_curve: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
    ) -> PerformanceMetrics:
        """
        分析绩效
        
        Args:
            returns: 策略收益率序列 (小数形式)
            benchmark_returns: 基准收益率序列
            equity_curve: 权益曲线 (可选)
        
        Returns:
            绩效指标
        """
        returns = pd.Series(returns)
        
        metrics = PerformanceMetrics()
        
        # 基础收益指标
        metrics.total_return = self._calculate_total_return(returns)
        metrics.annualized_return = self._calculate_annualized_return(returns)
        metrics.cagr = self._calculate_cagr(returns)
        
        # 风险指标
        metrics.volatility = returns.std()
        metrics.annualized_volatility = self._calculate_annualized_volatility(returns)
        
        # 回撤分析
        if equity_curve is not None:
            dd_metrics = self._calculate_drawdown_metrics(equity_curve)
            metrics.max_drawdown = dd_metrics['max_drawdown']
            metrics.max_drawdown_duration = dd_metrics['max_duration']
            metrics.avg_drawdown = dd_metrics['avg_drawdown']
            metrics.avg_drawdown_duration = dd_metrics['avg_duration']
            metrics.recovery_time = dd_metrics['recovery_time']
        
        # VaR和CVaR
        metrics.var_95 = self._calculate_var(returns, 0.95)
        metrics.cvar_95 = self._calculate_cvar(returns, 0.95)
        
        # 风险调整收益
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(returns, metrics.max_drawdown)
        
        # 基准相关指标
        if benchmark_returns is not None:
            benchmark_returns = pd.Series(benchmark_returns)
            metrics.beta, metrics.alpha = self._calculate_alpha_beta(returns, benchmark_returns)
            metrics.information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            metrics.tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
            metrics.treynor_ratio = self._calculate_treynor_ratio(returns, metrics.beta)
        
        # 收益分布
        metrics.skewness = returns.skew()
        metrics.kurtosis = returns.kurtosis()
        metrics.positive_days = (returns > 0).sum() / len(returns)
        
        return metrics
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        if n_periods == 0:
            return 0.0
        return (1 + total_return) ** (self.periods_per_year / n_periods) - 1
    
    def _calculate_cagr(self, returns: pd.Series) -> float:
        """计算复合年化增长率"""
        total_return = self._calculate_total_return(returns)
        n_years = len(returns) / self.periods_per_year
        if n_years == 0:
            return 0.0
        return (1 + total_return) ** (1 / n_years) - 1
    
    def _calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        return returns.std() * np.sqrt(self.periods_per_year)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        volatility = self._calculate_annualized_volatility(returns)
        if volatility == 0:
            return 0.0
        return excess_return / volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        if downside_std == 0:
            return 0.0
        return excess_return / downside_std
    
    def _calculate_calmar_ratio(
        self, 
        returns: pd.Series, 
        max_drawdown: float
    ) -> float:
        """计算Calmar比率"""
        if max_drawdown == 0:
            return 0.0
        annualized_return = self._calculate_annualized_return(returns)
        return annualized_return / abs(max_drawdown)
    
    def _calculate_drawdown_metrics(
        self, 
        equity_curve: Union[pd.Series, np.ndarray, List[float]]
    ) -> Dict:
        """计算回撤指标"""
        equity = pd.Series(equity_curve)
        
        # 计算回撤
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        # 最大回撤
        max_drawdown = drawdown.min()
        
        # 回撤持续时间
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_start = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and current_start is None:
                current_start = i
            elif not in_dd and current_start is not None:
                drawdown_periods.append((current_start, i - 1))
                current_start = None
        
        if current_start is not None:
            drawdown_periods.append((current_start, len(is_drawdown) - 1))
        
        # 最大回撤持续时间
        max_duration = 0
        if drawdown_periods:
            durations = [end - start for start, end in drawdown_periods]
            max_duration = max(durations) if durations else 0
        
        # 平均回撤
        avg_drawdown = drawdown[drawdown < 0].mean() if drawdown.min() < 0 else 0
        
        # 平均回撤持续时间
        avg_duration = np.mean(durations) if drawdown_periods else 0
        
        # 恢复时间
        recovery_time = 0
        if max_drawdown < 0:
            max_dd_idx = drawdown.idxmin()
            recovery_mask = (drawdown.index > max_dd_idx) & (drawdown == 0)
            if recovery_mask.any():
                recovery_idx = drawdown[recovery_mask].index[0]
                recovery_time = recovery_idx - max_dd_idx
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'avg_drawdown': avg_drawdown,
            'avg_duration': avg_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """计算VaR (Value at Risk)"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """计算CVaR (Conditional Value at Risk)"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_alpha_beta(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        # 对齐数据
        aligned_returns = returns.align(benchmark_returns, join='inner')
        strategy = aligned_returns[0]
        benchmark = aligned_returns[1]
        
        if len(strategy) < 2:
            return 0.0, 0.0
        
        # 计算Beta
        covariance = np.cov(strategy, benchmark)[0, 1]
        benchmark_variance = benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # 计算Alpha
        strategy_annual = self._calculate_annualized_return(strategy)
        benchmark_annual = self._calculate_annualized_return(benchmark)
        alpha = strategy_annual - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
        
        return beta, alpha
    
    def _calculate_information_ratio(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """计算信息比率"""
        aligned_returns = returns.align(benchmark_returns, join='inner')
        active_return = aligned_returns[0] - aligned_returns[1]
        tracking_error = active_return.std() * np.sqrt(self.periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        annualized_active = self._calculate_annualized_return(active_return)
        return annualized_active / tracking_error
    
    def _calculate_tracking_error(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """计算跟踪误差"""
        aligned_returns = returns.align(benchmark_returns, join='inner')
        active_return = aligned_returns[0] - aligned_returns[1]
        return active_return.std() * np.sqrt(self.periods_per_year)
    
    def _calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> float:
        """计算特雷诺比率"""
        if beta == 0:
            return 0.0
        excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
        return excess_return / beta
    
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30,
        metric: str = "sharpe"
    ) -> pd.Series:
        """计算滚动指标"""
        if metric == "sharpe":
            rolling_return = returns.rolling(window).mean() * self.periods_per_year
            rolling_std = returns.rolling(window).std() * np.sqrt(self.periods_per_year)
            return (rolling_return - self.risk_free_rate) / rolling_std
        elif metric == "return":
            return returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        elif metric == "volatility":
            return returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def calculate_monthly_returns(
        self, 
        returns: pd.Series,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """计算月度收益"""
        if timestamps is not None:
            returns.index = timestamps
        
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建月度收益表
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return pivot
    
    def calculate_yearly_returns(
        self,
        returns: pd.Series,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """计算年度收益"""
        if timestamps is not None:
            returns.index = timestamps
        
        yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        return pd.DataFrame({
            'return': yearly,
            'volatility': returns.resample('Y').std() * np.sqrt(self.periods_per_year)
        })
    
    def get_summary(self, metrics: PerformanceMetrics) -> str:
        """获取绩效摘要"""
        summary = f"""
{'='*60}
绩效分析报告
{'='*60}

收益指标:
  总收益率:           {metrics.total_return*100:>10.2f}%
  年化收益率:         {metrics.annualized_return*100:>10.2f}%
  CAGR:               {metrics.cagr*100:>10.2f}%

风险指标:
  年化波动率:         {metrics.annualized_volatility*100:>10.2f}%
  最大回撤:           {metrics.max_drawdown*100:>10.2f}%
  最大回撤持续期:     {metrics.max_drawdown_duration:>10} 期
  平均回撤:           {metrics.avg_drawdown*100:>10.2f}%
  VaR (95%):          {metrics.var_95*100:>10.2f}%
  CVaR (95%):         {metrics.cvar_95*100:>10.2f}%

风险调整收益:
  夏普比率:           {metrics.sharpe_ratio:>10.2f}
  索提诺比率:         {metrics.sortino_ratio:>10.2f}
  Calmar比率:         {metrics.calmar_ratio:>10.2f}
  信息比率:           {metrics.information_ratio:>10.2f}
  特雷诺比率:         {metrics.treynor_ratio:>10.2f}

基准相关:
  Alpha:              {metrics.alpha*100:>10.2f}%
  Beta:               {metrics.beta:>10.2f}
  跟踪误差:           {metrics.tracking_error*100:>10.2f}%

收益分布:
  偏度:               {metrics.skewness:>10.2f}
  峰度:               {metrics.kurtosis:>10.2f}
  正收益比例:         {metrics.positive_days*100:>10.2f}%

{'='*60}
"""
        return summary


def calculate_returns_from_equity(equity: pd.Series) -> pd.Series:
    """从权益曲线计算收益率"""
    return equity.pct_change().dropna()


def calculate_equity_from_returns(
    returns: pd.Series, 
    initial_value: float = 1.0
) -> pd.Series:
    """从收益率计算权益曲线"""
    return initial_value * (1 + returns).cumprod()


def compare_strategies(
    strategies: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    比较多个策略
    
    Args:
        strategies: 策略收益率字典 {name: returns}
        benchmark: 基准收益率
        risk_free_rate: 无风险利率
    
    Returns:
        比较表格
    """
    analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    
    results = []
    for name, returns in strategies.items():
        metrics = analyzer.analyze(returns, benchmark)
        results.append({
            'Strategy': name,
            'Total Return (%)': metrics.total_return * 100,
            'Annual Return (%)': metrics.annualized_return * 100,
            'Volatility (%)': metrics.annualized_volatility * 100,
            'Max DD (%)': metrics.max_drawdown * 100,
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Calmar': metrics.calmar_ratio,
            'Alpha (%)': metrics.alpha * 100,
            'Beta': metrics.beta
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试绩效分析
    print("绩效分析模块测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    n_days = 365
    
    # 策略收益率 (带正偏)
    strategy_returns = np.random.normal(0.001, 0.02, n_days)
    strategy_returns = pd.Series(strategy_returns)
    
    # 基准收益率
    benchmark_returns = np.random.normal(0.0005, 0.015, n_days)
    benchmark_returns = pd.Series(benchmark_returns)
    
    # 创建分析器
    analyzer = PerformanceAnalyzer(
        risk_free_rate=0.02,
        trading_days_per_year=365
    )
    
    # 计算权益曲线
    equity = calculate_equity_from_returns(strategy_returns, 100000)
    
    # 分析绩效
    metrics = analyzer.analyze(
        returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        equity_curve=equity
    )
    
    # 打印摘要
    print(analyzer.get_summary(metrics))
    
    # 测试月度收益
    print("\n月度收益表:")
    monthly = analyzer.calculate_monthly_returns(strategy_returns)
    print(monthly.round(4))
    
    # 测试策略比较
    print("\n策略比较:")
    strategies = {
        'Strategy A': strategy_returns,
        'Strategy B': benchmark_returns,
        'Strategy C': np.random.normal(0.0008, 0.025, n_days)
    }
    comparison = compare_strategies(strategies, benchmark_returns)
    print(comparison.round(4))
