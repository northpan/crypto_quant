"""
XGBoost 量化预测模型
支持回归和分类任务，提供高级特征和正则化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from .base_model import BaseModel, ModelConfig, PredictionResult, ModelMetrics

logger = logging.getLogger(__name__)

# 尝试导入XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost未安装，将使用模拟实现")


class XGBoostModel(BaseModel):
    """
    XGBoost预测模型
    
    特点：
    - 强大的正则化防止过拟合
    - 支持自定义损失函数
    - 内置交叉验证
    - GPU加速支持
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost未安装，请运行: pip install xgboost")
            raise ImportError("XGBoost is required")
        
        # XGBoost特定参数
        self.xgb_params = self._get_default_params()
        
        # 评估结果
        self.evals_result = {}
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        base_params = {
            'max_depth': 6,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.epochs,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.config.target_type == 'direction':
            # 分类任务
            base_params.update({
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
            })
        elif self.config.target_type == 'volatility':
            # 波动率预测（使用Gamma回归）
            base_params.update({
                'objective': 'reg:gamma',
                'eval_metric': ['rmse', 'mae'],
            })
        else:
            # 回归任务
            base_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae'],
            })
        
        return base_params
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> xgb.XGBModel:
        """
        构建XGBoost模型
        
        Args:
            input_shape: 输入数据形状（可选）
            
        Returns:
            XGBoost模型实例
        """
        if self.config.target_type == 'direction':
            model = xgb.XGBClassifier(**self.xgb_params)
        else:
            model = xgb.XGBRegressor(**self.xgb_params)
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练XGBoost模型
        
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
            X_val, y_val = validation_data
            X_val = self.preprocess_features(X_val)
            eval_set = [(X_val, y_val)]
        elif self.config.validation_split > 0:
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
        
        # 训练参数
        fit_params = {
            'eval_set': eval_set,
            'verbose': kwargs.get('verbose', True)
        }
        
        if self.config.early_stopping_rounds > 0:
            fit_params['early_stopping_rounds'] = self.config.early_stopping_rounds
        
        # 训练模型
        logger.info(f"开始训练XGBoost模型，样本数: {len(X_train)}")
        
        self.model.fit(X_train, y_train, **fit_params)
        
        # 记录评估结果
        if hasattr(self.model, 'evals_result()'):
            self.evals_result = self.model.evals_result()
            self.training_history.append(self.evals_result)
        
        # 更新特征重要性
        self._update_feature_importance()
        
        self.is_trained = True
        logger.info("XGBoost模型训练完成")
        
        return {
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None),
            'n_features': X.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        使用XGBoost模型进行预测
        
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
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
            特征重要性字典
        """
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 转换为数组格式
        if self.config.feature_columns:
            result = {}
            for i, feat_name in enumerate(self.config.feature_columns):
                key = f'f{i}'
                result[feat_name] = importance.get(key, 0.0)
            return result
        else:
            return importance
    
    def _update_feature_importance(self) -> None:
        """更新特征重要性"""
        self.feature_importance = self.get_feature_importance()
    
    def get_feature_importance_plot(self, top_n: int = 20) -> Any:
        """
        获取特征重要性可视化
        
        Args:
            top_n: 显示前N个特征
            
        Returns:
            matplotlib图形对象
        """
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance()
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            features, scores = zip(*sorted_importance)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(features)), scores, align='center')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'XGBoost Feature Importance (Top {top_n})')
            
            return fig
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制特征重要性")
            return None
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        增量学习更新模型
        
        Args:
            X: 新样本特征
            y: 新样本目标
        """
        if not self.config.online_learning:
            return
        
        X = self.preprocess_features(X)
        
        # XGBoost支持增量学习
        if self.model is not None and self.is_trained:
            self.model.fit(X, y, xgb_model=self.model.get_booster())
            logger.info(f"XGBoost模型增量更新: {len(X)} 个样本")
        else:
            self.fit(X, y)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       nfold: int = 5,
                       num_boost_round: int = None) -> Dict[str, Any]:
        """
        使用XGBoost内置交叉验证
        
        Args:
            X: 特征
            y: 目标
            nfold: 交叉验证折数
            num_boost_round: 迭代次数
            
        Returns:
            交叉验证结果
        """
        # 创建DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.config.feature_columns)
        
        # 设置参数
        params = self.xgb_params.copy()
        params.pop('n_estimators', None)
        
        num_boost_round = num_boost_round or self.config.epochs
        
        # 执行交叉验证
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            early_stopping_rounds=self.config.early_stopping_rounds,
            metrics=params.get('eval_metric', ['rmse']),
            as_pandas=True,
            seed=42
        )
        
        return {
            'cv_results': cv_results,
            'best_iteration': cv_results.shape[0],
            'mean_test_score': cv_results.iloc[-1].to_dict()
        }
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                  param_grid: Optional[Dict] = None,
                                  cv: int = 5) -> Dict[str, Any]:
        """
        使用随机搜索优化超参数
        
        Args:
            X: 训练特征
            y: 训练目标
            param_grid: 参数网格
            cv: 交叉验证折数
            
        Returns:
            最佳参数
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 1.0, 5.0, 10.0]
            }
        
        # 创建基础模型
        if self.config.target_type == 'direction':
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1
            )
            scoring = 'roc_auc'
        else:
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            scoring = 'neg_mean_squared_error'
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        logger.info(f"最佳参数: {random_search.best_params_}")
        logger.info(f"最佳得分: {random_search.best_score_}")
        
        # 更新模型
        self.xgb_params.update(random_search.best_params_)
        self.model = random_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def get_model_dump(self) -> str:
        """
        获取模型文本表示（用于分析）
        
        Returns:
            模型文本
        """
        if not self.is_trained:
            return ""
        
        return self.model.get_booster().get_dump()[0]
    
    def plot_tree(self, num_trees: int = 0) -> Any:
        """
        绘制决策树
        
        Args:
            num_trees: 树的索引
            
        Returns:
            图形对象
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(20, 10))
            xgb.plot_tree(self.model, num_trees=num_trees, ax=ax)
            return fig
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制树")
            return None


class XGBoostEnsemble(BaseModel):
    """
    XGBoost集成模型
    使用多个XGBoost模型进行Bagging
    """
    
    def __init__(self, config: ModelConfig, n_models: int = 5):
        super().__init__(config)
        self.n_models = n_models
        self.models = []
        self.sample_indices = []
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> List[xgb.XGBModel]:
        """构建多个XGBoost模型"""
        models = []
        for i in range(self.n_models):
            params = self.xgb_params.copy()
            params['random_state'] = 42 + i  # 不同的随机种子
            
            if self.config.target_type == 'direction':
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)
            
            models.append(model)
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """训练Bagging集成模型"""
        X = self.preprocess_features(X)
        
        if not self.models:
            self.models = self.build_model()
        
        n_samples = len(X)
        
        for i, model in enumerate(self.models):
            logger.info(f"训练第 {i+1}/{self.n_models} 个模型")
            
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            self.sample_indices.append(indices)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            model.fit(X_bootstrap, y_bootstrap, verbose=False)
        
        self.is_trained = True
        self.model = self.models  # 兼容基类
        
        return {'n_models': self.n_models, 'n_samples': n_samples}
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """集成预测"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        X = self.preprocess_features(X)
        
        # 收集所有模型的预测
        predictions_list = []
        for model in self.models:
            if self.config.target_type == 'direction':
                pred = model.predict_proba(X)[:, 1]  # 使用正类概率
            else:
                pred = model.predict(X)
            predictions_list.append(pred)
        
        # 平均预测
        predictions_array = np.array(predictions_list)
        mean_predictions = np.mean(predictions_array, axis=0)
        std_predictions = np.std(predictions_array, axis=0)
        
        # 分类问题需要转换
        if self.config.target_type == 'direction':
            binary_predictions = (mean_predictions > 0.5).astype(int)
            probabilities = np.column_stack([1 - mean_predictions, mean_predictions])
        else:
            binary_predictions = mean_predictions
            probabilities = None
        
        return PredictionResult(
            predictions=binary_predictions,
            probabilities=probabilities,
            confidence=1 - std_predictions,  # 标准差越小，置信度越高
            feature_importance=self.get_feature_importance(),
            model_version=self.model_version
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取平均特征重要性"""
        if not self.models:
            return {}
        
        importances = []
        for model in self.models:
            imp = model.get_booster().get_score(importance_type='gain')
            importances.append(imp)
        
        # 合并重要性
        all_features = set()
        for imp in importances:
            all_features.update(imp.keys())
        
        avg_importance = {}
        for feat in all_features:
            values = [imp.get(feat, 0) for imp in importances]
            avg_importance[feat] = np.mean(values)
        
        return avg_importance


# 工具函数
def create_xgb_dmatrix(X: np.ndarray, y: Optional[np.ndarray] = None,
                       feature_names: Optional[List[str]] = None) -> xgb.DMatrix:
    """
    创建XGBoost DMatrix
    
    Args:
        X: 特征
        y: 目标（可选）
        feature_names: 特征名称
        
    Returns:
        DMatrix对象
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required")
    
    return xgb.DMatrix(X, label=y, feature_names=feature_names)


def xgb_shap_explainer(model: xgb.XGBModel, X: np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    使用SHAP解释XGBoost模型
    
    Args:
        model: XGBoost模型
        X: 特征数据
        
    Returns:
        (SHAP值, explainer对象)
    """
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        return shap_values, explainer
    except ImportError:
        logger.warning("SHAP未安装")
        return None, None
