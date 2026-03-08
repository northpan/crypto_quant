"""
LightGBM 量化预测模型
支持回归和分类任务，针对分钟级预测优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from .base_model import BaseModel, ModelConfig, PredictionResult, ModelMetrics

logger = logging.getLogger(__name__)

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM未安装，将使用模拟实现")


class LightGBMModel(BaseModel):
    """
    LightGBM预测模型
    
    特点：
    - 训练速度快，内存占用低
    - 支持类别特征
    - 内置特征重要性
    - 支持早停和自定义评估函数
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM未安装，请运行: pip install lightgbm")
            raise ImportError("LightGBM is required")
        
        # LightGBM特定参数
        self.lgb_params = self._get_default_params()
        
        # 早停回调
        self.callbacks = []
        if self.config.early_stopping_rounds > 0:
            self.callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=True
                )
            )
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        if self.config.target_type == 'direction':
            # 分类任务参数
            return {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': self.config.epochs,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            # 回归任务参数
            return {
                'objective': 'regression',
                'metric': ['rmse', 'mae'],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': self.config.epochs,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> lgb.LGBMModel:
        """
        构建LightGBM模型
        
        Args:
            input_shape: 输入数据形状（可选）
            
        Returns:
            LightGBM模型实例
        """
        if self.config.target_type == 'direction':
            model = lgb.LGBMClassifier(**self.lgb_params)
        else:
            model = lgb.LGBMRegressor(**self.lgb_params)
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练LightGBM模型
        
        Args:
            X: 训练特征 [n_samples, n_features]
            y: 训练目标 [n_samples]
            validation_data: 验证数据 (X_val, y_val)
            **kwargs: 额外参数
            
        Returns:
            训练历史记录
        """
        # 预处理特征
        X = self.preprocess_features(X)
        
        # 构建模型
        if self.model is None:
            self.model = self.build_model()
        
        # 准备验证数据
        eval_set = None
        if validation_data is not None:
            X_train, y_train = X, y
            X_val, y_val = validation_data
            X_val = self.preprocess_features(X_val)
            eval_set = [(X_val, y_val)]
        elif self.config.validation_split > 0:
            # 自动划分验证集
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
        
        # 训练模型
        logger.info(f"开始训练LightGBM模型，样本数: {len(X_train)}")
        
        fit_params = {
            'eval_set': eval_set,
            'callbacks': self.callbacks,
        }
        if eval_set:
            fit_params['eval_set'] = eval_set
        self.model.fit(X_train, y_train, **fit_params)
        
        # 记录训练历史
        if hasattr(self.model, 'evals_result_'):
            self.training_history.append(self.model.evals_result_)
        
        # 更新特征重要性
        self._update_feature_importance()
        
        self.is_trained = True
        logger.info("LightGBM模型训练完成")
        
        return {
            'best_iteration': getattr(self.model, 'best_iteration_', None),
            'best_score': getattr(self.model, 'best_score_', None),
            'n_features': X.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        使用LightGBM模型进行预测
        
        Args:
            X: 输入特征 [n_samples, n_features]
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        X = self.preprocess_features(X)
        
        if self.config.target_type == 'direction':
            # 分类预测
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            confidence = np.max(probabilities, axis=1)
        else:
            # 回归预测
            predictions = self.model.predict(X)
            probabilities = None
            confidence = None
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            feature_importance=self.feature_importance,
            model_version=self.model_version
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        # 如果有特征名称，使用特征名称
        if self.config.feature_columns:
            return dict(zip(self.config.feature_columns, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def _update_feature_importance(self) -> None:
        """更新特征重要性"""
        self.feature_importance = self.get_feature_importance()
    
    def predict_with_shap(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用SHAP值进行预测解释
        
        Args:
            X: 输入特征
            
        Returns:
            (预测值, SHAP值)
        """
        try:
            import shap
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            predictions = self.model.predict(X)
            
            return predictions, shap_values
        except ImportError:
            logger.warning("SHAP未安装，无法计算SHAP值")
            predictions = self.predict(X).predictions
            return predictions, None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = {
            'model_type': 'LightGBM',
            'objective': self.lgb_params.get('objective'),
            'num_trees': self.model.n_estimators_ if hasattr(self.model, 'n_estimators_') else None,
            'num_features': len(self.feature_importance),
            'feature_importance_top10': dict(
                sorted(self.feature_importance.items(), 
                       key=lambda x: x[1], reverse=True)[:10]
            ) if self.feature_importance else {}
        }
        
        if hasattr(self.model, 'best_iteration_'):
            summary['best_iteration'] = self.model.best_iteration_
        
        if hasattr(self.model, 'best_score_'):
            summary['best_score'] = self.model.best_score_
        
        return summary
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                  param_grid: Optional[Dict] = None,
                                  cv: int = 5) -> Dict[str, Any]:
        """
        使用网格搜索优化超参数
        
        Args:
            X: 训练特征
            y: 训练目标
            param_grid: 参数网格
            cv: 交叉验证折数
            
        Returns:
            最佳参数
        """
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            param_grid = {
                'num_leaves': [20, 31, 50],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200]
            }
        
        # 创建基础模型
        if self.config.target_type == 'direction':
            base_model = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                random_state=42
            )
            scoring = 'roc_auc'
        else:
            base_model = lgb.LGBMRegressor(
                objective='regression',
                boosting_type='gbdt',
                random_state=42
            )
            scoring = 'neg_mean_squared_error'
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳得分: {grid_search.best_score_}")
        
        # 更新模型参数
        self.lgb_params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }


class LightGBMQuantileModel(BaseModel):
    """
    LightGBM分位数回归模型
    用于预测收益率的置信区间
    """
    
    def __init__(self, config: ModelConfig, quantiles: List[float] = None):
        super().__init__(config)
        
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models = {}
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required")
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> Dict[float, lgb.LGBMRegressor]:
        """构建分位数回归模型"""
        models = {}
        for q in self.quantiles:
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': self.config.learning_rate,
                'n_estimators': self.config.epochs,
                'random_state': 42,
                'verbose': -1
            }
            models[q] = lgb.LGBMRegressor(**params)
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """训练分位数回归模型"""
        X = self.preprocess_features(X)
        
        if not self.models:
            self.models = self.build_model()
        
        for q, model in self.models.items():
            logger.info(f"训练分位数 {q} 模型")
            
            fit_params = {'verbose': False}
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = self.preprocess_features(X_val)
                fit_params['eval_set'] = [(X_val, y_val)]
            
            model.fit(X, y, **fit_params)
        
        self.is_trained = True
        self.model = self.models  # 兼容基类
        
        return {'quantiles': self.quantiles, 'n_samples': len(X)}
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """预测分位数"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        X = self.preprocess_features(X)
        
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
        
        # 使用中位数作为主要预测
        median_pred = predictions.get(0.5, list(predictions.values())[0])
        
        # 计算置信区间宽度作为置信度
        if 0.1 in predictions and 0.9 in predictions:
            confidence = predictions[0.9] - predictions[0.1]
        else:
            confidence = None
        
        return PredictionResult(
            predictions=median_pred,
            probabilities=None,
            confidence=confidence,
            feature_importance=self.get_feature_importance(),
            model_version=self.model_version
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（取平均）"""
        if not self.models:
            return {}
        
        importances = []
        for model in self.models.values():
            importances.append(model.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        if self.config.feature_columns:
            return dict(zip(self.config.feature_columns, avg_importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(avg_importance)}
    
    def predict_interval(self, X: np.ndarray, 
                         lower_q: float = 0.1, 
                         upper_q: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测置信区间
        
        Returns:
            (下界, 中位数, 上界)
        """
        X = self.preprocess_features(X)
        
        lower = self.models.get(lower_q, self.models[0.1]).predict(X)
        upper = self.models.get(upper_q, self.models[0.9]).predict(X)
        median = self.models.get(0.5, list(self.models.values())[0]).predict(X)
        
        return lower, median, upper


# 工具函数
def create_lightgbm_datasets(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None) -> Tuple:
    """
    创建LightGBM数据集
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        feature_names: 特征名称
        
    Returns:
        (train_dataset, val_dataset)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required")
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    
    val_data = None
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    return train_data, val_data


def train_with_cv(X: np.ndarray, y: np.ndarray,
                  params: Dict[str, Any],
                  num_boost_round: int = 100,
                  nfold: int = 5,
                  stratified: bool = False) -> Dict[str, Any]:
    """
    使用交叉验证训练LightGBM
    
    Args:
        X: 特征
        y: 目标
        params: 模型参数
        num_boost_round: 迭代次数
        nfold: 交叉验证折数
        stratified: 是否分层
        
    Returns:
        交叉验证结果
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required")
    
    cv_results = lgb.cv(
        params,
        lgb.Dataset(X, label=y),
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=stratified,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=10)]
    )
    
    return cv_results
