"""
模型集成模块
实现Stacking、Blending和动态权重组合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict

from .base_model import BaseModel, ModelConfig, PredictionResult, ModelMetrics, calculate_returns_metrics

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成配置"""
    ensemble_method: str = 'stacking'  # 'stacking', 'blending', 'voting', 'weighted'
    meta_model_type: str = 'ridge'  # 元模型类型
    use_proba: bool = True  # 是否使用概率
    cv_folds: int = 5  # 交叉验证折数
    optimize_weights: bool = True  # 是否优化权重
    
    # 动态权重参数
    lookback_window: int = 100  # 回看窗口
    weight_update_freq: int = 10  # 权重更新频率


class EnsembleModel(BaseModel):
    """
    模型集成基类
    
    支持多种集成策略：
    - Stacking: 使用元模型组合基模型预测
    - Blending: 使用hold-out验证集组合
    - Voting: 投票集成
    - Weighted: 加权平均
    """
    
    def __init__(self, config: ModelConfig, ensemble_config: Optional[EnsembleConfig] = None):
        super().__init__(config)
        
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.base_models: Dict[str, BaseModel] = {}
        self.meta_model = None
        self.model_weights: Dict[str, float] = {}
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """
        添加基模型
        
        Args:
            name: 模型名称
            model: 模型实例
            weight: 初始权重
        """
        self.base_models[name] = model
        self.model_weights[name] = weight
        logger.info(f"添加模型: {name} (权重: {weight})")
    
    def remove_model(self, name: str) -> None:
        """移除基模型"""
        if name in self.base_models:
            del self.base_models[name]
            del self.model_weights[name]
            logger.info(f"移除模型: {name}")
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> Any:
        """构建元模型"""
        if self.ensemble_config.meta_model_type == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        elif self.ensemble_config.meta_model_type == 'lasso':
            from sklearn.linear_model import Lasso
            return Lasso(alpha=0.1)
        elif self.ensemble_config.meta_model_type == 'elasticnet':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.ensemble_config.meta_model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
        else:
            raise ValueError(f"Unknown meta model type: {self.ensemble_config.meta_model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练集成模型
        
        Args:
            X: 训练特征
            y: 训练目标
            validation_data: 验证数据
            **kwargs: 额外参数
        """
        logger.info(f"开始训练集成模型，方法: {self.ensemble_config.ensemble_method}")
        
        if self.ensemble_config.ensemble_method == 'stacking':
            return self._fit_stacking(X, y, validation_data, **kwargs)
        elif self.ensemble_config.ensemble_method == 'blending':
            return self._fit_blending(X, y, **kwargs)
        elif self.ensemble_config.ensemble_method == 'voting':
            return self._fit_voting(X, y, **kwargs)
        elif self.ensemble_config.ensemble_method == 'weighted':
            return self._fit_weighted(X, y, validation_data, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_config.ensemble_method}")
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray,
                      validation_data: Optional[Tuple] = None,
                      **kwargs) -> Dict:
        """Stacking训练"""
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # 准备交叉验证
        if self.config.target_type == 'direction':
            kf = StratifiedKFold(n_splits=self.ensemble_config.cv_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=self.ensemble_config.cv_folds, shuffle=True, random_state=42)
        
        # 存储基模型预测
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # 训练基模型并生成元特征
        for name, model in self.base_models.items():
            logger.info(f"训练基模型: {name}")
            
            fold_predictions = np.zeros(n_samples)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y if self.config.target_type == 'direction' else None)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # 训练基模型
                model.fit(X_train_fold, y_train_fold, verbose=False)
                
                # 预测验证集
                pred = model.predict(X_val_fold).predictions
                fold_predictions[val_idx] = pred
            
            # 存储该模型的预测作为元特征
            model_idx = list(self.base_models.keys()).index(name)
            meta_features[:, model_idx] = fold_predictions
            
            # 在所有数据上重新训练
            model.fit(X, y, verbose=False)
        
        # 训练元模型
        logger.info("训练元模型")
        self.meta_model = self.build_model()
        
        if self.config.target_type == 'direction':
            self.meta_model.fit(meta_features, y)
        else:
            self.meta_model.fit(meta_features, y)
        
        self.is_trained = True
        
        return {
            'method': 'stacking',
            'n_base_models': n_models,
            'cv_folds': self.ensemble_config.cv_folds
        }
    
    def _fit_blending(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Blending训练"""
        # 划分训练集和hold-out集
        split_idx = int(len(X) * 0.8)
        X_train, X_hold = X[:split_idx], X[split_idx:]
        y_train, y_hold = y[:split_idx], y[split_idx:]
        
        # 训练基模型
        meta_features = np.zeros((len(X_hold), len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"训练基模型: {name}")
            model.fit(X_train, y_train, verbose=False)
            
            # 在hold-out集上预测
            pred = model.predict(X_hold).predictions
            meta_features[:, idx] = pred
        
        # 训练元模型
        logger.info("训练元模型")
        self.meta_model = self.build_model()
        self.meta_model.fit(meta_features, y_hold)
        
        # 在所有数据上重新训练基模型
        for model in self.base_models.values():
            model.fit(X, y, verbose=False)
        
        self.is_trained = True
        
        return {
            'method': 'blending',
            'n_base_models': len(self.base_models),
            'holdout_ratio': 0.2
        }
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """投票训练（简单平均所有模型）"""
        for name, model in self.base_models.items():
            logger.info(f"训练模型: {name}")
            model.fit(X, y, verbose=False)
        
        # 等权重
        n_models = len(self.base_models)
        self.model_weights = {name: 1.0 / n_models for name in self.base_models}
        
        self.is_trained = True
        
        return {
            'method': 'voting',
            'n_models': n_models
        }
    
    def _fit_weighted(self, X: np.ndarray, y: np.ndarray,
                      validation_data: Optional[Tuple] = None,
                      **kwargs) -> Dict:
        """加权训练（基于验证集性能优化权重）"""
        # 训练所有基模型
        for name, model in self.base_models.items():
            logger.info(f"训练模型: {name}")
            model.fit(X, y, verbose=False)
        
        # 如果有验证数据，优化权重
        if validation_data is not None:
            X_val, y_val = validation_data
            self._optimize_weights(X_val, y_val)
        else:
            # 等权重
            n_models = len(self.base_models)
            self.model_weights = {name: 1.0 / n_models for name in self.base_models}
        
        self.is_trained = True
        
        return {
            'method': 'weighted',
            'n_models': len(self.base_models),
            'weights': self.model_weights
        }
    
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """优化模型权重"""
        from scipy.optimize import minimize
        
        # 获取各模型预测
        predictions = {}
        for name, model in self.base_models.items():
            pred = model.predict(X_val).predictions
            predictions[name] = pred
        
        # 定义目标函数（最小化MSE）
        def objective(weights):
            weighted_pred = np.zeros(len(y_val))
            for i, name in enumerate(self.base_models.keys()):
                weighted_pred += weights[i] * predictions[name]
            return np.mean((y_val - weighted_pred) ** 2)
        
        # 约束：权重和为1，权重非负
        n_models = len(self.base_models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # 优化
        result = minimize(objective, 
                         x0=np.ones(n_models) / n_models,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        # 更新权重
        for i, name in enumerate(self.base_models.keys()):
            self.model_weights[name] = result.x[i]
        
        logger.info(f"优化后的权重: {self.model_weights}")
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        集成预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise RuntimeError("集成模型尚未训练")
        
        # 获取各模型预测
        predictions = {}
        probabilities = {}
        
        for name, model in self.base_models.items():
            result = model.predict(X)
            predictions[name] = result.predictions
            if result.probabilities is not None:
                probabilities[name] = result.probabilities
        
        # 根据集成方法组合预测
        if self.ensemble_config.ensemble_method in ['stacking', 'blending']:
            # 使用元模型
            meta_features = np.column_stack([predictions[name] for name in self.base_models.keys()])
            ensemble_pred = self.meta_model.predict(meta_features)
        else:
            # 加权平均
            ensemble_pred = np.zeros(len(X))
            for name, weight in self.model_weights.items():
                ensemble_pred += weight * predictions[name]
        
        # 分类问题处理
        if self.config.target_type == 'direction':
            if self.ensemble_config.ensemble_method in ['stacking', 'blending']:
                binary_pred = (ensemble_pred > 0.5).astype(int)
                probs = np.column_stack([1 - ensemble_pred, ensemble_pred])
            else:
                binary_pred = (ensemble_pred > 0).astype(int)
                # 概率估计
                probs_list = [probabilities[name] for name in probabilities.keys()]
                if probs_list:
                    weighted_probs = np.zeros(probs_list[0].shape)
                    for name in probabilities.keys():
                        weighted_probs += self.model_weights[name] * probabilities[name]
                    probs = weighted_probs
                else:
                    probs = None
            
            return PredictionResult(
                predictions=binary_pred,
                probabilities=probs,
                model_version=self.model_version
            )
        else:
            return PredictionResult(
                predictions=ensemble_pred,
                model_version=self.model_version
            )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（元模型系数）"""
        if self.meta_model is not None and hasattr(self.meta_model, 'coef_'):
            importance = np.abs(self.meta_model.coef_)
            return dict(zip(self.base_models.keys(), importance))
        return self.model_weights
    
    def update_weights_online(self, recent_returns: Dict[str, np.ndarray]) -> None:
        """
        在线更新权重（基于近期表现）
        
        Args:
            recent_returns: 各模型近期收益率
        """
        # 计算夏普比率
        sharpe_ratios = {}
        for name, returns in recent_returns.items():
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)
                sharpe_ratios[name] = max(0, sharpe)  # 只考虑正夏普
            else:
                sharpe_ratios[name] = 0
        
        # 根据夏普比率更新权重
        total_sharpe = sum(sharpe_ratios.values())
        if total_sharpe > 0:
            for name in self.model_weights:
                self.model_weights[name] = sharpe_ratios[name] / total_sharpe
        
        logger.info(f"在线更新权重: {self.model_weights}")
    
    def get_model_contributions(self) -> Dict[str, Dict[str, float]]:
        """
        获取各模型贡献度
        
        Returns:
            模型贡献度字典
        """
        contributions = {}
        for name in self.base_models:
            contributions[name] = {
                'weight': self.model_weights.get(name, 0),
                'performance_history': self.model_performance.get(name, [])
            }
        return contributions


class DynamicEnsemble(EnsembleModel):
    """
    动态集成模型
    根据市场状态动态调整模型权重
    """
    
    def __init__(self, config: ModelConfig, 
                 ensemble_config: Optional[EnsembleConfig] = None):
        super().__init__(config, ensemble_config)
        
        self.regime_detector = None
        self.regime_models: Dict[int, Dict[str, float]] = {}
        self.current_regime = 0
    
    def set_regime_detector(self, detector: Callable[[np.ndarray], int]) -> None:
        """设置市场状态检测器"""
        self.regime_detector = detector
    
    def predict(self, X: np.ndarray, market_features: Optional[np.ndarray] = None) -> PredictionResult:
        """
        动态预测
        
        Args:
            X: 输入特征
            market_features: 市场状态特征
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        # 检测市场状态
        if self.regime_detector is not None and market_features is not None:
            self.current_regime = self.regime_detector(market_features)
            
            # 使用该状态的权重
            if self.current_regime in self.regime_models:
                weights = self.regime_models[self.current_regime]
            else:
                weights = self.model_weights
        else:
            weights = self.model_weights
        
        # 加权预测
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X).predictions
        
        ensemble_pred = np.zeros(len(X))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]
        
        if self.config.target_type == 'direction':
            binary_pred = (ensemble_pred > 0).astype(int)
        else:
            binary_pred = ensemble_pred
        
        return PredictionResult(
            predictions=binary_pred,
            model_version=self.model_version
        )
    
    def fit_regime_weights(self, X: np.ndarray, y: np.ndarray,
                          regime_labels: np.ndarray) -> None:
        """
        为不同市场状态训练权重
        
        Args:
            X: 特征
            y: 目标
            regime_labels: 市场状态标签
        """
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            X_regime = X[mask]
            y_regime = y[mask]
            
            # 获取各模型预测
            predictions = {}
            for name, model in self.base_models.items():
                pred = model.predict(X_regime).predictions
                predictions[name] = pred
            
            # 优化该状态的权重
            from scipy.optimize import minimize
            
            def objective(weights):
                weighted_pred = np.zeros(len(y_regime))
                for i, name in enumerate(self.base_models.keys()):
                    weighted_pred += weights[i] * predictions[name]
                return np.mean((y_regime - weighted_pred) ** 2)
            
            n_models = len(self.base_models)
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            
            result = minimize(objective,
                            x0=np.ones(n_models) / n_models,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            self.regime_models[int(regime)] = {
                name: result.x[i] 
                for i, name in enumerate(self.base_models.keys())
            }
        
        logger.info(f"市场状态权重: {self.regime_models}")


class StackingRegressor:
    """
    通用的Stacking回归器
    兼容sklearn接口
    """
    
    def __init__(self, base_models: List[Tuple[str, Any]], 
                 meta_model: Any = None,
                 cv: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.trained_models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingRegressor':
        """训练Stacking模型"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # 训练基模型
        for idx, (name, model) in enumerate(self.base_models):
            fold_preds = np.zeros(n_samples)
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)
                fold_preds[val_idx] = model_clone.predict(X_val)
            
            meta_features[:, idx] = fold_preds
            
            # 在全部数据上训练
            model.fit(X, y)
            self.trained_models[name] = model
        
        # 训练元模型
        if self.meta_model is None:
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
        
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        meta_features = np.column_stack([
            model.predict(X) for model in self.trained_models.values()
        ])
        return self.meta_model.predict(meta_features)
    
    def _clone_model(self, model: Any) -> Any:
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)


# 工具函数
def create_diverse_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> EnsembleModel:
    """
    创建多样化的集成模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        
    Returns:
        训练好的集成模型
    """
    from lightgbm_model import LightGBMModel
    from xgboost_model import XGBoostModel
    
    # 创建配置
    config = ModelConfig(
        model_name='diverse_ensemble',
        model_type='ensemble',
        target_type='return'
    )
    
    ensemble_config = EnsembleConfig(
        ensemble_method='stacking',
        meta_model_type='ridge',
        cv_folds=5
    )
    
    # 创建集成模型
    ensemble = EnsembleModel(config, ensemble_config)
    
    # 添加LightGBM模型
    lgb_config = ModelConfig(
        model_name='lgb_model',
        model_type='lightgbm',
        learning_rate=0.05,
        epochs=100
    )
    lgb_model = LightGBMModel(lgb_config)
    ensemble.add_model('lightgbm', lgb_model)
    
    # 添加XGBoost模型
    xgb_config = ModelConfig(
        model_name='xgb_model',
        model_type='xgboost',
        learning_rate=0.05,
        epochs=100
    )
    xgb_model = XGBoostModel(xgb_config)
    ensemble.add_model('xgboost', xgb_model)
    
    # 训练集成模型
    ensemble.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    return ensemble


def calculate_ensemble_diversity(predictions: Dict[str, np.ndarray]) -> float:
    """
    计算集成多样性（预测相关性）
    
    Args:
        predictions: 各模型预测字典
        
    Returns:
        多样性分数（越低越多样）
    """
    pred_matrix = np.column_stack(list(predictions.values()))
    corr_matrix = np.corrcoef(pred_matrix.T)
    
    # 计算平均相关性（排除对角线）
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    avg_corr = np.mean(np.abs(corr_matrix[mask]))
    
    return avg_corr
