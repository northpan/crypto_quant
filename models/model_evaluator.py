"""
模型评估模块
提供全面的模型性能评估和可视化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging
from collections import defaultdict

from .base_model import BaseModel, ModelMetrics, PredictionResult, calculate_returns_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """评估报告"""
    model_name: str
    model_version: str
    evaluation_time: datetime
    metrics: ModelMetrics
    predictions: np.ndarray
    targets: np.ndarray
    timestamps: Optional[np.ndarray] = None
    additional_stats: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'evaluation_time': self.evaluation_time.isoformat(),
            'metrics': self.metrics.to_dict(),
            'additional_stats': self.additional_stats or {}
        }


class ModelEvaluator:
    """
    模型评估器
    
    提供：
    - 全面的性能指标计算
    - 回测模拟
    - 可视化报告
    - 统计检验
    """
    
    def __init__(self, risk_free_rate: float = 0.0, 
                 trading_cost: float = 0.001):
        """
        初始化评估器
        
        Args:
            risk_free_rate: 无风险利率（年化）
            trading_cost: 交易成本（单次）
        """
        self.risk_free_rate = risk_free_rate
        self.trading_cost = trading_cost
        self.evaluation_history = []
    
    def evaluate(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                 timestamps: Optional[np.ndarray] = None,
                 prices: Optional[np.ndarray] = None) -> EvaluationReport:
        """
        全面评估模型
        
        Args:
            model: 模型实例
            X: 特征
            y: 目标（真实收益率）
            timestamps: 时间戳
            prices: 价格序列（用于回测）
            
        Returns:
            评估报告
        """
        logger.info(f"评估模型: {model.config.model_name}")
        
        # 获取预测
        prediction_result = model.predict(X)
        predictions = prediction_result.predictions
        
        # 计算基础指标
        metrics = self._calculate_metrics(y, predictions)
        
        # 回测模拟
        backtest_results = self._backtest(y, predictions, prices)
        metrics.sharpe_ratio = backtest_results['sharpe_ratio']
        metrics.max_drawdown = backtest_results['max_drawdown']
        metrics.calmar_ratio = backtest_results['calmar_ratio']
        
        # 额外统计
        additional_stats = {
            'backtest': backtest_results,
            'prediction_distribution': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'target_distribution': {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y))
            }
        }
        
        # 创建报告
        report = EvaluationReport(
            model_name=model.config.model_name,
            model_version=model.model_version,
            evaluation_time=datetime.now(),
            metrics=metrics,
            predictions=predictions,
            targets=y,
            timestamps=timestamps,
            additional_stats=additional_stats
        )
        
        self.evaluation_history.append(report)
        
        return report
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> ModelMetrics:
        """计算评估指标"""
        metrics = ModelMetrics()
        
        # 回归指标
        metrics.mse = np.mean((y_true - y_pred) ** 2)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 分类指标（方向预测）
        y_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        metrics.accuracy = np.mean(y_direction == pred_direction)
        
        # 计算精确率和召回率
        tp = np.sum((y_direction == 1) & (pred_direction == 1))
        fp = np.sum((y_direction == 0) & (pred_direction == 1))
        fn = np.sum((y_direction == 1) & (pred_direction == 0))
        
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        
        return metrics
    
    def _backtest(self, returns: np.ndarray, predictions: np.ndarray,
                  prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        回测模拟
        
        Args:
            returns: 真实收益率
            predictions: 预测值
            prices: 价格序列
            
        Returns:
            回测结果
        """
        # 生成交易信号
        signals = np.sign(predictions)
        
        # 计算策略收益率
        strategy_returns = signals * returns
        
        # 扣除交易成本
        position_changes = np.diff(signals, prepend=signals[0])
        transaction_costs = np.abs(position_changes) * self.trading_cost
        strategy_returns = strategy_returns - transaction_costs
        
        # 计算指标
        result = calculate_returns_metrics(strategy_returns)
        
        # 额外指标
        result['n_trades'] = int(np.sum(np.abs(position_changes) > 0))
        result['win_rate'] = float(np.mean(strategy_returns > 0))
        result['profit_factor'] = self._calculate_profit_factor(strategy_returns)
        
        return result
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        gains = np.sum(returns[returns > 0])
        losses = np.abs(np.sum(returns[returns < 0]))
        return gains / losses if losses > 0 else np.inf
    
    def evaluate_direction_accuracy(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     threshold: float = 0.0) -> Dict[str, float]:
        """
        评估方向预测准确性
        
        Args:
            y_true: 真实收益率
            y_pred: 预测收益率
            threshold: 阈值（只预测超过阈值的信号）
            
        Returns:
            方向准确性指标
        """
        # 过滤强信号
        if threshold > 0:
            mask = np.abs(y_pred) >= threshold
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred
        
        # 方向
        true_direction = (y_true_filtered > 0).astype(int)
        pred_direction = (y_pred_filtered > 0).astype(int)
        
        # 计算混淆矩阵
        tp = np.sum((true_direction == 1) & (pred_direction == 1))
        tn = np.sum((true_direction == 0) & (pred_direction == 0))
        fp = np.sum((true_direction == 0) & (pred_direction == 1))
        fn = np.sum((true_direction == 1) & (pred_direction == 0))
        
        total = tp + tn + fp + fn
        
        return {
            'accuracy': (tp + tn) / total if total > 0 else 0,
            'precision_up': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall_up': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision_down': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'recall_down': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_up': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'f1_down': 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0,
            'n_signals': int(total)
        }
    
    def evaluate_by_regime(self, model: BaseModel, X: np.ndarray, 
                           y: np.ndarray, 
                           regime_labels: np.ndarray) -> Dict[int, Dict]:
        """
        按市场状态评估模型
        
        Args:
            model: 模型
            X: 特征
            y: 目标
            regime_labels: 市场状态标签
            
        Returns:
            各状态的评估结果
        """
        predictions = model.predict(X).predictions
        
        results = {}
        for regime in np.unique(regime_labels):
            mask = regime_labels == regime
            y_regime = y[mask]
            pred_regime = predictions[mask]
            
            # 计算该状态的指标
            metrics = self._calculate_metrics(y_regime, pred_regime)
            
            # 回测
            backtest = self._backtest(y_regime, pred_regime)
            
            results[int(regime)] = {
                'metrics': metrics.to_dict(),
                'backtest': backtest,
                'n_samples': int(np.sum(mask))
            }
        
        return results
    
    def statistical_tests(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, Any]:
        """
        统计检验
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            统计检验结果
        """
        from scipy import stats
        
        results = {}
        
        # 残差分析
        residuals = y_true - y_pred
        
        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
        results['normality_test'] = {
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_pvalue': float(shapiro_p),
            'is_normal': shapiro_p > 0.05
        }
        
        # 自相关检验（Ljung-Box）
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            results['autocorrelation_test'] = {
                'lb_statistic': float(lb_test['lb_stat'].iloc[-1]),
                'lb_pvalue': float(lb_test['lb_pvalue'].iloc[-1]),
                'no_autocorrelation': lb_test['lb_pvalue'].iloc[-1] > 0.05
            }
        except ImportError:
            results['autocorrelation_test'] = {'note': 'statsmodels not installed'}
        
        # 异方差检验
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.api import add_constant
            
            exog = add_constant(y_pred)
            bp_test = het_breuschpagan(residuals, exog)
            results['heteroscedasticity_test'] = {
                'bp_statistic': float(bp_test[0]),
                'bp_pvalue': float(bp_test[1]),
                'no_heteroscedasticity': bp_test[1] > 0.05
            }
        except ImportError:
            results['heteroscedasticity_test'] = {'note': 'statsmodels not installed'}
        
        return results
    
    def generate_report(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                        output_path: str = None) -> str:
        """
        生成完整评估报告
        
        Args:
            model: 模型
            X: 特征
            y: 目标
            output_path: 输出路径
            
        Returns:
            报告路径
        """
        # 执行评估
        evaluation = self.evaluate(model, X, y)
        
        # 统计检验
        stat_tests = self.statistical_tests(y, evaluation.predictions)
        
        # 方向准确性
        direction_acc = self.evaluate_direction_accuracy(y, evaluation.predictions)
        
        # 构建报告
        report = {
            'model_info': {
                'name': model.config.model_name,
                'type': model.config.model_type,
                'version': model.model_version,
                'prediction_horizon': model.config.prediction_horizon,
                'target_type': model.config.target_type
            },
            'evaluation_summary': {
                'evaluation_time': evaluation.evaluation_time.isoformat(),
                'n_samples': len(y),
                'metrics': evaluation.metrics.to_dict()
            },
            'backtest_results': evaluation.additional_stats.get('backtest', {}),
            'direction_accuracy': direction_acc,
            'statistical_tests': stat_tests,
            'feature_importance': model.feature_importance
        }
        
        # 保存报告
        if output_path is None:
            output_path = f"./reports/evaluation_{model.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"评估报告已保存: {output_path}")
        return output_path
    
    def plot_evaluation(self, evaluation: EvaluationReport, 
                        save_path: str = None) -> Any:
        """
        绘制评估图表
        
        Args:
            evaluation: 评估报告
            save_path: 保存路径
            
        Returns:
            图形对象
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 预测vs真实值
            ax = axes[0, 0]
            ax.scatter(evaluation.targets, evaluation.predictions, alpha=0.5, s=1)
            ax.plot([evaluation.targets.min(), evaluation.targets.max()], 
                   [evaluation.targets.min(), evaluation.targets.max()], 
                   'r--', lw=2)
            ax.set_xlabel('Actual Returns')
            ax.set_ylabel('Predicted Returns')
            ax.set_title(f'Predictions vs Actual (R²={evaluation.metrics.r2:.4f})')
            ax.grid(True, alpha=0.3)
            
            # 2. 残差分布
            ax = axes[0, 1]
            residuals = evaluation.targets - evaluation.predictions
            ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            ax.grid(True, alpha=0.3)
            
            # 3. 累计收益
            ax = axes[1, 0]
            signals = np.sign(evaluation.predictions)
            strategy_returns = signals * evaluation.targets
            cumulative = np.cumprod(1 + strategy_returns)
            ax.plot(cumulative, label='Strategy')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Return')
            ax.set_title(f'Cumulative Returns (Sharpe={evaluation.metrics.sharpe_ratio:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. 回撤
            ax = axes[1, 1]
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Drawdown')
            ax.set_title(f'Drawdown (Max DD={evaluation.metrics.max_drawdown:.4f})')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"图表已保存: {save_path}")
            
            return fig
        
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制图表")
            return None
    
    def compare_models(self, models: List[BaseModel], X: np.ndarray, 
                       y: np.ndarray) -> pd.DataFrame:
        """
        比较多个模型
        
        Args:
            models: 模型列表
            X: 特征
            y: 目标
            
        Returns:
            比较结果DataFrame
        """
        results = []
        
        for model in models:
            evaluation = self.evaluate(model, X, y)
            
            results.append({
                'Model': model.config.model_name,
                'Type': model.config.model_type,
                'RMSE': evaluation.metrics.rmse,
                'MAE': evaluation.metrics.mae,
                'R²': evaluation.metrics.r2,
                'Accuracy': evaluation.metrics.accuracy,
                'Sharpe Ratio': evaluation.metrics.sharpe_ratio,
                'Max Drawdown': evaluation.metrics.max_drawdown,
                'Calmar Ratio': evaluation.metrics.calmar_ratio,
                'Win Rate': evaluation.additional_stats['backtest']['win_rate']
            })
        
        df = pd.DataFrame(results)
        
        # 高亮最佳值
        def highlight_best(s):
            if s.name in ['RMSE', 'MAE', 'Max Drawdown']:
                return ['background-color: lightgreen' if v == s.min() else '' for v in s]
            else:
                return ['background-color: lightgreen' if v == s.max() else '' for v in s]
        
        return df.style.apply(highlight_best, subset=df.columns[2:])


class BacktestEngine:
    """
    回测引擎
    模拟真实交易环境
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def run(self, predictions: np.ndarray, prices: np.ndarray,
            position_size: str = 'fixed') -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            predictions: 预测信号
            prices: 价格序列
            position_size: 仓位管理策略
            
        Returns:
            回测结果
        """
        n = len(predictions)
        
        # 初始化
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, n):
            # 生成信号
            signal = np.sign(predictions[i])
            
            # 计算价格变动
            price_change = (prices[i] - prices[i-1]) / prices[i-1]
            
            # 仓位管理
            if position_size == 'fixed':
                trade_size = capital * 0.1  # 10%固定仓位
            elif position_size == 'kelly':
                # 简化Kelly公式
                win_rate = 0.55  # 假设胜率
                avg_win = 0.02
                avg_loss = 0.01
                kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                trade_size = capital * max(0, kelly_f)
            else:
                trade_size = capital
            
            # 执行交易
            if signal != 0:
                # 交易成本
                cost = trade_size * (self.commission + self.slippage)
                
                # 收益
                pnl = signal * trade_size * price_change - cost
                capital += pnl
                
                trades.append({
                    'time': i,
                    'signal': signal,
                    'pnl': pnl,
                    'capital': capital
                })
            
            equity_curve.append(capital)
        
        # 计算指标
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = calculate_returns_metrics(returns)
        
        return {
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'equity_curve': equity_curve.tolist(),
            'trades': trades,
            'n_trades': len(trades),
            **metrics
        }


# 工具函数
def quick_evaluate(model: BaseModel, X_test: np.ndarray, 
                   y_test: np.ndarray) -> Dict[str, float]:
    """
    快速评估模型
    
    Args:
        model: 模型
        X_test: 测试特征
        y_test: 测试目标
        
    Returns:
        关键指标
    """
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, X_test, y_test)
    
    return {
        'sharpe_ratio': report.metrics.sharpe_ratio,
        'max_drawdown': report.metrics.max_drawdown,
        'accuracy': report.metrics.accuracy,
        'rmse': report.metrics.rmse,
        'r2': report.metrics.r2
    }


def print_evaluation_report(report: EvaluationReport) -> None:
    """
    打印评估报告
    
    Args:
        report: 评估报告
    """
    print("\n" + "="*60)
    print(f"模型评估报告: {report.model_name}")
    print("="*60)
    
    print("\n【基础指标】")
    print(f"  RMSE: {report.metrics.rmse:.6f}")
    print(f"  MAE:  {report.metrics.mae:.6f}")
    print(f"  R²:   {report.metrics.r2:.6f}")
    
    print("\n【分类指标】")
    print(f"  准确率:    {report.metrics.accuracy:.4f}")
    print(f"  精确率:    {report.metrics.precision:.4f}")
    print(f"  召回率:    {report.metrics.recall:.4f}")
    print(f"  F1分数:    {report.metrics.f1_score:.4f}")
    
    print("\n【回测指标】")
    backtest = report.additional_stats.get('backtest', {})
    print(f"  夏普比率:  {backtest.get('sharpe_ratio', 0):.4f}")
    print(f"  最大回撤:  {backtest.get('max_drawdown', 0):.4f}")
    print(f"  Calmar:    {backtest.get('calmar_ratio', 0):.4f}")
    print(f"  胜率:      {backtest.get('win_rate', 0):.4f}")
    print(f"  交易次数:  {backtest.get('n_trades', 0)}")
    
    print("\n" + "="*60)
