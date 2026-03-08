"""
模型训练流程模块
提供完整的模型训练、验证和优化流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from collections import defaultdict
import warnings

from .base_model import BaseModel, ModelConfig, ModelMetrics, ModelRegistry, calculate_returns_metrics

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 时间序列划分
    use_time_split: bool = True  # 按时间划分而非随机
    
    # 训练参数
    random_seed: int = 42
    verbose: bool = True
    
    # 早停
    early_stopping_patience: int = 20
    min_delta: float = 1e-4
    
    # 超参数优化
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    hyperopt_timeout: int = 3600
    
    # 模型保存
    save_best_only: bool = True
    checkpoint_dir: str = './checkpoints/'
    
    # 评估
    evaluate_on_test: bool = True
    calculate_sharpe: bool = True


class ModelTrainer:
    """
    模型训练器
    
    提供：
    - 数据划分和预处理
    - 模型训练流程
    - 超参数优化
    - 交叉验证
    - 模型保存和加载
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model_registry = ModelRegistry()
        self.training_history = []
        self.best_model = None
        self.best_metrics = None
        
        # 创建检查点目录
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     timestamps: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        准备训练/验证/测试数据
        
        Args:
            X: 特征数据
            y: 目标数据
            timestamps: 时间戳（用于时间序列划分）
            
        Returns:
            划分后的数据字典
        """
        n_samples = len(X)
        
        if self.config.use_time_split:
            # 按时间顺序划分
            train_end = int(n_samples * self.config.train_ratio)
            val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
            
            splits = {
                'X_train': X[:train_end],
                'y_train': y[:train_end],
                'X_val': X[train_end:val_end],
                'y_val': y[train_end:val_end],
                'X_test': X[val_end:],
                'y_test': y[val_end:]
            }
            
            if timestamps is not None:
                splits['train_timestamps'] = timestamps[:train_end]
                splits['val_timestamps'] = timestamps[train_end:val_end]
                splits['test_timestamps'] = timestamps[val_end:]
        else:
            # 随机划分
            from sklearn.model_selection import train_test_split
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, 
                test_size=1 - self.config.train_ratio,
                random_state=self.config.random_seed
            )
            
            val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=1 - val_ratio_adjusted,
                random_state=self.config.random_seed
            )
            
            splits = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test
            }
        
        logger.info(f"数据划分: 训练集 {len(splits['X_train'])}, "
                   f"验证集 {len(splits['X_val'])}, "
                   f"测试集 {len(splits['X_test'])}")
        
        return splits
    
    def train(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model: 模型实例
            X: 训练特征
            y: 训练目标
            validation_data: 验证数据
            **kwargs: 额外参数
            
        Returns:
            训练结果
        """
        logger.info(f"开始训练模型: {model.config.model_name}")
        
        start_time = datetime.now()
        
        # 训练模型
        train_result = model.fit(X, y, validation_data=validation_data, **kwargs)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 评估
        if validation_data is not None:
            X_val, y_val = validation_data
            metrics = model.evaluate(X_val, y_val)
        else:
            metrics = ModelMetrics()
        
        result = {
            'model_name': model.config.model_name,
            'model_type': model.config.model_type,
            'training_time': training_time,
            'train_result': train_result,
            'metrics': metrics.to_dict(),
            'model_version': model.model_version
        }
        
        self.training_history.append(result)
        
        # 保存最佳模型
        if self.config.save_best_only:
            if self.best_metrics is None or metrics.sharpe_ratio > self.best_metrics.sharpe_ratio:
                self.best_model = model
                self.best_metrics = metrics
                model.save()
        
        logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
        
        return result
    
    def train_with_cv(self, model_class: type, model_config: ModelConfig,
                      X: np.ndarray, y: np.ndarray,
                      n_folds: int = 5) -> Dict[str, Any]:
        """
        交叉验证训练
        
        Args:
            model_class: 模型类
            model_config: 模型配置
            X: 特征
            y: 目标
            n_folds: 折数
            
        Returns:
            交叉验证结果
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        fold_results = []
        all_predictions = []
        all_targets = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"交叉验证 Fold {fold + 1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建并训练模型
            model = model_class(model_config)
            model.fit(X_train, y_train, verbose=False)
            
            # 评估
            metrics = model.evaluate(X_val, y_val)
            
            # 预测
            predictions = model.predict(X_val).predictions
            all_predictions.extend(predictions)
            all_targets.extend(y_val)
            
            fold_results.append({
                'fold': fold + 1,
                'metrics': metrics.to_dict()
            })
        
        # 计算整体指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        overall_metrics = ModelMetrics()
        overall_metrics.mse = np.mean((all_targets - all_predictions) ** 2)
        overall_metrics.rmse = np.sqrt(overall_metrics.mse)
        overall_metrics.mae = np.mean(np.abs(all_targets - all_predictions))
        
        return {
            'fold_results': fold_results,
            'overall_metrics': overall_metrics.to_dict(),
            'mean_sharpe': np.mean([f['metrics']['sharpe_ratio'] for f in fold_results]),
            'std_sharpe': np.std([f['metrics']['sharpe_ratio'] for f in fold_results])
        }
    
    def hyperparameter_optimize(self, model_class: type, 
                                 base_config: ModelConfig,
                                 param_space: Dict[str, List],
                                 X: np.ndarray, y: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 n_trials: int = None) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            model_class: 模型类
            base_config: 基础配置
            param_space: 参数搜索空间
            X: 训练特征
            y: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            n_trials: 试验次数
            
        Returns:
            优化结果
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna未安装，使用网格搜索")
            return self._grid_search_optimize(model_class, base_config, param_space,
                                              X, y, X_val, y_val)
        
        n_trials = n_trials or self.config.hyperopt_trials
        
        def objective(trial):
            # 建议参数
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    if all(isinstance(x, int) for x in param_range):
                        params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                    elif all(isinstance(x, float) for x in param_range):
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range), log=True)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # 创建配置
            config_dict = base_config.to_dict()
            config_dict.update(params)
            config = ModelConfig(**config_dict)
            
            # 训练模型
            model = model_class(config)
            model.fit(X, y, verbose=False)
            
            # 评估
            metrics = model.evaluate(X_val, y_val)
            
            # 返回负夏普比率（最小化）
            return -metrics.sharpe_ratio
        
        # 创建study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=self.config.hyperopt_timeout)
        
        best_params = study.best_params
        best_score = -study.best_value
        
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳夏普比率: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_sharpe': best_score,
            'n_trials': len(study.trials),
            'optimization_history': [
                {'trial': t.number, 'value': -t.value, 'params': t.params}
                for t in study.trials
            ]
        }
    
    def _grid_search_optimize(self, model_class: type, base_config: ModelConfig,
                              param_space: Dict[str, List],
                              X: np.ndarray, y: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """网格搜索优化"""
        from itertools import product
        
        best_score = -np.inf
        best_params = {}
        
        # 生成所有参数组合
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # 创建配置
            config_dict = base_config.to_dict()
            config_dict.update(params)
            config = ModelConfig(**config_dict)
            
            # 训练模型
            model = model_class(config)
            model.fit(X, y, verbose=False)
            
            # 评估
            metrics = model.evaluate(X_val, y_val)
            
            if metrics.sharpe_ratio > best_score:
                best_score = metrics.sharpe_ratio
                best_params = params
        
        return {
            'best_params': best_params,
            'best_sharpe': best_score
        }
    
    def walk_forward_train(self, model_class: type, model_config: ModelConfig,
                           X: np.ndarray, y: np.ndarray,
                           timestamps: np.ndarray,
                           train_window: int = 1000,
                           test_window: int = 100,
                           step_size: int = 50) -> Dict[str, Any]:
        """
        滚动窗口训练（Walk-forward）
        
        Args:
            model_class: 模型类
            model_config: 模型配置
            X: 特征
            y: 目标
            timestamps: 时间戳
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step_size: 步长
            
        Returns:
            滚动训练结果
        """
        n_samples = len(X)
        results = []
        all_predictions = []
        all_targets = []
        all_timestamps = []
        
        start_idx = train_window
        
        while start_idx + test_window <= n_samples:
            end_idx = start_idx + test_window
            
            # 划分数据
            X_train = X[start_idx - train_window:start_idx]
            y_train = y[start_idx - train_window:start_idx]
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            test_ts = timestamps[start_idx:end_idx]
            
            # 训练模型
            model = model_class(model_config)
            model.fit(X_train, y_train, verbose=False)
            
            # 预测
            predictions = model.predict(X_test).predictions
            
            # 计算收益率
            returns = np.sign(predictions) * y_test
            
            # 记录结果
            results.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'sharpe': self._calculate_sharpe(returns),
                'returns': returns.tolist()
            })
            
            all_predictions.extend(predictions)
            all_targets.extend(y_test)
            all_timestamps.extend(test_ts)
            
            start_idx += step_size
        
        # 计算整体指标
        all_returns = np.sign(np.array(all_predictions)) * np.array(all_targets)
        
        return {
            'window_results': results,
            'overall_sharpe': self._calculate_sharpe(all_returns),
            'total_return': np.prod(1 + all_returns) - 1,
            'predictions': all_predictions,
            'targets': all_targets,
            'timestamps': all_timestamps
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)
    
    def train_ensemble(self, ensemble_model, X: np.ndarray, y: np.ndarray,
                       validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        训练集成模型
        
        Args:
            ensemble_model: 集成模型实例
            X: 训练特征
            y: 训练目标
            validation_data: 验证数据
            
        Returns:
            训练结果
        """
        logger.info("开始训练集成模型")
        
        start_time = datetime.now()
        
        # 训练集成模型
        train_result = ensemble_model.fit(X, y, validation_data=validation_data)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 评估
        if validation_data is not None:
            X_val, y_val = validation_data
            metrics = ensemble_model.evaluate(X_val, y_val)
        else:
            metrics = ModelMetrics()
        
        result = {
            'model_name': ensemble_model.config.model_name,
            'model_type': 'ensemble',
            'training_time': training_time,
            'train_result': train_result,
            'metrics': metrics.to_dict(),
            'model_weights': ensemble_model.model_weights
        }
        
        self.training_history.append(result)
        
        logger.info(f"集成模型训练完成，耗时: {training_time:.2f}秒")
        
        return result
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        
        Returns:
            训练摘要
        """
        if not self.training_history:
            return {}
        
        summary = {
            'n_models_trained': len(self.training_history),
            'models': [],
            'best_model': None,
            'best_sharpe': -np.inf
        }
        
        for result in self.training_history:
            model_info = {
                'name': result['model_name'],
                'type': result['model_type'],
                'sharpe': result['metrics'].get('sharpe_ratio', 0),
                'training_time': result['training_time']
            }
            summary['models'].append(model_info)
            
            if model_info['sharpe'] > summary['best_sharpe']:
                summary['best_sharpe'] = model_info['sharpe']
                summary['best_model'] = model_info['name']
        
        return summary
    
    def save_training_report(self, filepath: str = None) -> str:
        """
        保存训练报告
        
        Args:
            filepath: 报告路径
            
        Returns:
            报告路径
        """
        if filepath is None:
            filepath = f"{self.config.checkpoint_dir}/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'training_config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'random_seed': self.config.random_seed
            },
            'training_history': self.training_history,
            'summary': self.get_training_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"训练报告已保存: {filepath}")
        return filepath


class OnlineTrainer:
    """
    在线训练器
    支持模型的增量学习和实时更新
    """
    
    def __init__(self, model: BaseModel, update_frequency: int = 100):
        self.model = model
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.update_count = 0
        self.performance_history = []
    
    def update(self, X: np.ndarray, y: np.ndarray, 
               current_return: float = None) -> bool:
        """
        在线更新模型
        
        Args:
            X: 新样本特征
            y: 新样本目标
            current_return: 当前收益率（用于性能跟踪）
            
        Returns:
            是否执行了更新
        """
        # 添加到缓冲区
        self.data_buffer.append((X, y))
        
        # 记录性能
        if current_return is not None:
            self.performance_history.append(current_return)
        
        # 检查是否需要更新
        if len(self.data_buffer) >= self.update_frequency:
            self._perform_update()
            return True
        
        return False
    
    def _perform_update(self) -> None:
        """执行模型更新"""
        # 合并缓冲区数据
        X_batch = np.vstack([x for x, _ in self.data_buffer])
        y_batch = np.concatenate([y for _, y in self.data_buffer])
        
        logger.info(f"在线更新: {len(X_batch)} 个样本")
        
        # 增量训练
        self.model.partial_fit(X_batch, y_batch)
        
        # 清空缓冲区
        self.data_buffer = []
        self.update_count += 1
    
    def get_performance_metrics(self, window: int = 100) -> Dict[str, float]:
        """
        获取近期性能指标
        
        Args:
            window: 回看窗口
            
        Returns:
            性能指标
        """
        if len(self.performance_history) < window:
            window = len(self.performance_history)
        
        recent_returns = np.array(self.performance_history[-window:])
        
        return calculate_returns_metrics(recent_returns)


# 工具函数
def create_training_pipeline(model_type: str, target_type: str = 'return') -> Tuple[ModelTrainer, ModelConfig]:
    """
    创建训练流水线
    
    Args:
        model_type: 模型类型
        target_type: 目标类型
        
    Returns:
        (训练器, 模型配置)
    """
    from lightgbm_model import LightGBMModel
    from xgboost_model import XGBoostModel
    from lstm_model import LSTMModel
    from transformer_model import TransformerModel
    
    # 创建训练配置
    training_config = TrainingConfig()
    trainer = ModelTrainer(training_config)
    
    # 创建模型配置
    model_config = ModelConfig(
        model_name=f'{model_type}_model',
        model_type=model_type,
        target_type=target_type,
        prediction_horizon=60
    )
    
    return trainer, model_config


def compare_models(models: List[BaseModel], X_test: np.ndarray, 
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    比较多个模型性能
    
    Args:
        models: 模型列表
        X_test: 测试特征
        y_test: 测试目标
        
    Returns:
        性能对比DataFrame
    """
    results = []
    
    for model in models:
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Model': model.config.model_name,
            'Type': model.config.model_type,
            'RMSE': metrics.rmse,
            'MAE': metrics.mae,
            'R²': metrics.r2,
            'Sharpe': metrics.sharpe_ratio,
            'Max DD': metrics.max_drawdown
        })
    
    return pd.DataFrame(results)
