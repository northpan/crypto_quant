"""
量化模型基类模块
提供所有预测模型的统一接口
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
import hashlib
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str
    model_type: str  # 'lightgbm', 'xgboost', 'lstm', 'transformer', etc.
    prediction_horizon: int = 60  # 预测未来N分钟
    target_type: str = 'return'  # 'return', 'direction', 'volatility'
    feature_columns: List[str] = None
    
    # 训练参数
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    early_stopping_rounds: int = 50
    validation_split: float = 0.2
    
    # 在线学习参数
    online_learning: bool = True
    update_frequency: int = 100  # 每N个样本更新一次
    
    # 模型保存路径
    model_path: str = './models/'
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'target_type': self.target_type,
            'feature_columns': self.feature_columns,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_split': self.validation_split,
            'online_learning': self.online_learning,
            'update_frequency': self.update_frequency,
            'model_path': self.model_path
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        return cls(**config_dict)


@dataclass
class PredictionResult:
    """预测结果类"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None  # 分类问题的概率
    confidence: Optional[np.ndarray] = None  # 预测置信度
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = None
    model_version: str = ''
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelMetrics:
    """模型评估指标"""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    accuracy: float = 0.0  # 分类准确率
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio
        }


class BaseModel(ABC):
    """
    量化模型基类
    所有预测模型都需要继承此类并实现抽象方法
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.model_version = self._generate_version()
        self.training_history = []
        self.feature_importance = {}
        
        # 在线学习相关
        self.online_buffer = []
        self.online_counter = 0
        
        # 创建模型保存目录
        Path(config.model_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化模型: {config.model_name} (版本: {self.model_version})")
    
    def _generate_version(self) -> str:
        """生成模型版本号"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_hash = hashlib.md5(
            f"{self.config.model_name}_{timestamp}".encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{model_hash}"
    
    @abstractmethod
    def build_model(self, input_shape: Optional[Tuple] = None) -> Any:
        """
        构建模型架构
        
        Args:
            input_shape: 输入数据形状
            
        Returns:
            构建好的模型对象
        """
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练目标
            validation_data: 验证数据 (X_val, y_val)
            **kwargs: 额外参数
            
        Returns:
            训练历史记录
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        进行预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        pass
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        在线学习更新模型
        
        Args:
            X: 新样本特征
            y: 新样本目标
        """
        if not self.config.online_learning:
            logger.warning("在线学习未启用")
            return
        
        # 添加到缓冲区
        self.online_buffer.append((X, y))
        self.online_counter += len(X)
        
        # 达到更新频率时进行更新
        if self.online_counter >= self.config.update_frequency:
            self._update_model()
            self.online_buffer = []
            self.online_counter = 0
    
    def _update_model(self) -> None:
        """执行模型更新（子类可重写）"""
        if not self.online_buffer:
            return
        
        # 合并缓冲区数据
        X_update = np.vstack([x for x, _ in self.online_buffer])
        y_update = np.concatenate([y for _, y in self.online_buffer])
        
        # 默认实现：使用新数据重新训练
        # 子类可以重写此方法实现增量学习
        logger.info(f"在线学习更新: {len(X_update)} 个样本")
        self.fit(X_update, y_update, verbose=False)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        评估模型性能
        
        Args:
            X: 测试特征
            y: 测试目标
            
        Returns:
            评估指标
        """
        predictions = self.predict(X).predictions
        
        metrics = ModelMetrics()
        
        # 回归指标
        metrics.mse = np.mean((y - predictions) ** 2)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.mae = np.mean(np.abs(y - predictions))
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 分类指标（如果是分类问题）
        if self.config.target_type == 'direction':
            y_binary = (y > 0).astype(int)
            pred_binary = (predictions > 0).astype(int)
            metrics.accuracy = np.mean(y_binary == pred_binary)
        
        return metrics
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, 
                                risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        std = np.std(excess_returns)
        
        if std == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / std * np.sqrt(252 * 24 * 60)  # 分钟级年化
        return sharpe
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        计算最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        保存模型
        
        Args:
            filepath: 保存路径，默认为None则使用配置中的路径
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            filepath = Path(self.config.model_path) / \
                      f"{self.config.model_name}_{self.model_version}.pkl"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'config': self.config.to_dict(),
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # 同时保存配置为JSON
        config_path = filepath.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        logger.info(f"模型已保存: {filepath}")
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        config = ModelConfig.from_dict(save_dict['config'])
        
        # 创建实例（需要子类实现）
        instance = cls(config)
        instance.model = save_dict['model']
        instance.model_version = save_dict['model_version']
        instance.is_trained = save_dict['is_trained']
        instance.training_history = save_dict['training_history']
        instance.feature_importance = save_dict['feature_importance']
        
        logger.info(f"模型已加载: {filepath} (版本: {instance.model_version})")
        return instance
    
    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        特征预处理
        
        Args:
            X: 原始特征
            
        Returns:
            预处理后的特征
        """
        # 基础预处理：处理缺失值和无穷值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时序序列（用于深度学习模型）
        
        Args:
            X: 特征数组
            y: 目标数组
            seq_length: 序列长度
            
        Returns:
            (X_seq, y_seq) 序列化后的数据
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:(i + seq_length)])
            y_seq.append(y[i + seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'prediction_horizon': self.config.prediction_horizon,
            'target_type': self.config.target_type,
            'feature_columns': self.config.feature_columns,
            'training_history_length': len(self.training_history),
            'feature_importance': self.feature_importance
        }


class ModelRegistry:
    """
    模型注册表
    管理多个模型的版本和元数据
    """
    
    def __init__(self, registry_path: str = './model_registry/'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """加载注册表"""
        registry_file = self.registry_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.models = json.load(f)
    
    def _save_registry(self) -> None:
        """保存注册表"""
        registry_file = self.registry_path / 'registry.json'
        with open(registry_file, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)
    
    def register(self, model: BaseModel, metrics: ModelMetrics) -> str:
        """
        注册模型
        
        Args:
            model: 模型实例
            metrics: 模型评估指标
            
        Returns:
            模型ID
        """
        model_id = f"{model.config.model_name}_{model.model_version}"
        
        self.models[model_id] = {
            'model_name': model.config.model_name,
            'model_type': model.config.model_type,
            'model_version': model.model_version,
            'registration_time': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'config': model.config.to_dict()
        }
        
        self._save_registry()
        logger.info(f"模型已注册: {model_id}")
        return model_id
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Optional[str]:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标名称
            
        Returns:
            最佳模型ID
        """
        if not self.models:
            return None
        
        best_model = max(
            self.models.items(),
            key=lambda x: x[1]['metrics'].get(metric, 0)
        )
        return best_model[0]
    
    def list_models(self) -> List[Dict]:
        """
        列出所有模型
        
        Returns:
            模型列表
        """
        return [
            {'model_id': k, **v} 
            for k, v in self.models.items()
        ]


# 工具函数
def calculate_returns_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    计算收益率相关指标
    
    Args:
        returns: 收益率序列
        
    Returns:
        指标字典
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0
        }
    
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (252 * 24 * 60 / n_periods) - 1
    volatility = np.std(returns) * np.sqrt(252 * 24 * 60)
    
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) \
             if np.std(returns) > 0 else 0.0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = np.mean(returns > 0)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate
    }
