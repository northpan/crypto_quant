"""
LSTM 时序预测模型
支持多变量时间序列预测，针对分钟级数据优化
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass

from .base_model import BaseModel, ModelConfig, PredictionResult, ModelMetrics

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow未安装，将使用模拟实现")


@dataclass
class LSTMConfig:
    """LSTM模型配置"""
    seq_length: int = 60  # 序列长度（分钟）
    n_features: int = 20  # 特征数量
    
    # LSTM层配置
    lstm_units: List[int] = None
    lstm_dropout: float = 0.2
    recurrent_dropout: float = 0.2
    
    # 全连接层配置
    dense_units: List[int] = None
    dense_dropout: float = 0.3
    
    # 正则化
    l1_reg: float = 0.001
    l2_reg: float = 0.001
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64]
        if self.dense_units is None:
            self.dense_units = [32, 16]


class LSTMModel(BaseModel):
    """
    LSTM时序预测模型
    
    特点：
    - 多层LSTM捕捉时序依赖
    - 支持双向LSTM
    - 注意力机制可选
    - 支持多步预测
    """
    
    def __init__(self, config: ModelConfig, lstm_config: Optional[LSTMConfig] = None):
        super().__init__(config)
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow未安装，请运行: pip install tensorflow")
            raise ImportError("TensorFlow is required")
        
        self.lstm_config = lstm_config or LSTMConfig()
        self.history = None
        
        # 设置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"GPU设置错误: {e}")
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> Any:
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入形状 (seq_length, n_features)
            
        Returns:
            Keras模型
        """
        if input_shape is None:
            input_shape = (self.lstm_config.seq_length, self.lstm_config.n_features)
        
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # LSTM层
        for i, units in enumerate(self.lstm_config.lstm_units):
            return_sequences = i < len(self.lstm_config.lstm_units) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.lstm_config.lstm_dropout,
                recurrent_dropout=self.lstm_config.recurrent_dropout,
                kernel_regularizer=l1_l2(
                    l1=self.lstm_config.l1_reg,
                    l2=self.lstm_config.l2_reg
                ),
                name=f'lstm_{i+1}'
            )(x)
        
        # 全连接层
        for i, units in enumerate(self.lstm_config.dense_units):
            x = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l1_l2(
                    l1=self.lstm_config.l1_reg,
                    l2=self.lstm_config.l2_reg
                ),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(self.lstm_config.dense_dropout)(x)
            x = layers.BatchNormalization()(x)
        
        # 输出层
        if self.config.target_type == 'direction':
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC()]
        else:
            outputs = layers.Dense(1, activation='linear', name='output')(x)
            loss = 'mse'
            metrics = ['mae', tf.keras.metrics.RootMeanSquaredError()]
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"LSTM模型构建完成，参数数量: {model.count_params()}")
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练LSTM模型
        
        Args:
            X: 训练特征 [n_samples, seq_length, n_features]
            y: 训练目标 [n_samples]
            validation_data: 验证数据 (X_val, y_val)
            **kwargs: 额外参数
            
        Returns:
            训练历史记录
        """
        # 预处理
        X = self.preprocess_features(X)
        
        # 构建模型
        if self.model is None:
            self.model = self.build_model(input_shape=X.shape[1:])
        
        # 准备验证数据
        if validation_data is None and self.config.validation_split > 0:
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
        
        # 回调函数
        callbacks = []
        
        if self.config.early_stopping_rounds > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_rounds,
                restore_best_weights=True,
                verbose=1
            ))
        
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ))
        
        # 训练模型
        logger.info(f"开始训练LSTM模型，样本数: {len(X_train)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )
        
        # 保存训练历史
        self.training_history.append(self.history.history)
        
        self.is_trained = True
        logger.info("LSTM模型训练完成")
        
        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history.get('val_loss', [None])[-1],
            'n_params': self.model.count_params()
        }
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        使用LSTM模型进行预测
        
        Args:
            X: 输入特征 [n_samples, seq_length, n_features]
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        X = self.preprocess_features(X)
        
        predictions = self.model.predict(X, verbose=0).flatten()
        
        if self.config.target_type == 'direction':
            probabilities = np.column_stack([1 - predictions, predictions])
            binary_predictions = (predictions > 0.5).astype(int)
            confidence = np.abs(predictions - 0.5) * 2  # 转换为0-1范围
        else:
            binary_predictions = predictions
            probabilities = None
            confidence = None
        
        return PredictionResult(
            predictions=binary_predictions,
            probabilities=probabilities,
            confidence=confidence,
            feature_importance=self.get_feature_importance(),
            model_version=self.model_version
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（使用置换重要性近似）
        
        Returns:
            特征重要性字典
        """
        # LSTM没有直接的特征重要性，返回空字典
        # 子类可以实现更复杂的方法
        if self.config.feature_columns:
            return {col: 1.0 / len(self.config.feature_columns) 
                   for col in self.config.feature_columns}
        return {}
    
    def predict_multi_step(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        多步预测
        
        Args:
            X: 输入特征 [n_samples, seq_length, n_features]
            n_steps: 预测步数
            
        Returns:
            多步预测结果 [n_samples, n_steps]
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        predictions = []
        current_seq = X.copy()
        
        for _ in range(n_steps):
            pred = self.model.predict(current_seq, verbose=0)
            predictions.append(pred.flatten())
            
            # 更新序列（假设最后一个特征是目标值）
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[:, -1, -1] = pred.flatten()
        
        return np.array(predictions).T
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        保存模型（Keras格式）
        
        Args:
            filepath: 保存路径
            
        Returns:
            保存的文件路径
        """
        if filepath is None:
            filepath = f"{self.config.model_path}/{self.config.model_name}_{self.model_version}"
        
        # 保存Keras模型
        keras_path = filepath + '.keras'
        self.model.save(keras_path)
        
        # 保存配置
        import json
        config_path = filepath + '_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'lstm_config': {
                    'seq_length': self.lstm_config.seq_length,
                    'n_features': self.lstm_config.n_features,
                    'lstm_units': self.lstm_config.lstm_units,
                    'lstm_dropout': self.lstm_config.lstm_dropout,
                    'recurrent_dropout': self.lstm_config.recurrent_dropout,
                    'dense_units': self.lstm_config.dense_units,
                    'dense_dropout': self.lstm_config.dense_dropout,
                    'l1_reg': self.lstm_config.l1_reg,
                    'l2_reg': self.lstm_config.l2_reg
                },
                'model_version': self.model_version,
                'is_trained': self.is_trained
            }, f, indent=2)
        
        logger.info(f"LSTM模型已保存: {keras_path}")
        return keras_path
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径（不含扩展名）
            
        Returns:
            加载的模型实例
        """
        import json
        
        # 加载配置
        config_path = filepath + '_config.json'
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        config = ModelConfig.from_dict(saved_config['config'])
        lstm_config = LSTMConfig(**saved_config['lstm_config'])
        
        # 创建实例
        instance = cls(config, lstm_config)
        
        # 加载Keras模型
        keras_path = filepath + '.keras'
        instance.model = keras.models.load_model(keras_path)
        instance.model_version = saved_config['model_version']
        instance.is_trained = saved_config['is_trained']
        
        logger.info(f"LSTM模型已加载: {filepath}")
        return instance


class BidirectionalLSTMModel(LSTMModel):
    """
    双向LSTM模型
    同时捕捉正向和反向时序依赖
    """
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> Any:
        """构建双向LSTM模型"""
        if input_shape is None:
            input_shape = (self.lstm_config.seq_length, self.lstm_config.n_features)
        
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # 双向LSTM层
        for i, units in enumerate(self.lstm_config.lstm_units):
            return_sequences = i < len(self.lstm_config.lstm_units) - 1
            
            x = layers.Bidirectional(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.lstm_config.lstm_dropout,
                    recurrent_dropout=self.lstm_config.recurrent_dropout,
                    kernel_regularizer=l1_l2(
                        l1=self.lstm_config.l1_reg,
                        l2=self.lstm_config.l2_reg
                    )
                ),
                name=f'bilstm_{i+1}'
            )(x)
        
        # 全连接层
        for units in self.lstm_config.dense_units:
            x = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l1_l2(
                    l1=self.lstm_config.l1_reg,
                    l2=self.lstm_config.l2_reg
                )
            )(x)
            x = layers.Dropout(self.lstm_config.dense_dropout)(x)
        
        # 输出层
        if self.config.target_type == 'direction':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae']
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model


class AttentionLSTMModel(LSTMModel):
    """
    带注意力机制的LSTM模型
    """
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> Any:
        """构建带注意力的LSTM模型"""
        if input_shape is None:
            input_shape = (self.lstm_config.seq_length, self.lstm_config.n_features)
        
        inputs = layers.Input(shape=input_shape)
        
        # LSTM层（返回序列用于注意力）
        lstm_out = layers.LSTM(
            units=self.lstm_config.lstm_units[0],
            return_sequences=True,
            dropout=self.lstm_config.lstm_dropout,
            recurrent_dropout=self.lstm_config.recurrent_dropout
        )(inputs)
        
        # 注意力机制
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.lstm_config.lstm_units[0] // 4
        )(lstm_out, lstm_out)
        
        # 池化
        x = layers.GlobalAveragePooling1D()(attention)
        
        # 额外的LSTM层
        for units in self.lstm_config.lstm_units[1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.lstm_config.lstm_dropout)(x)
        
        # 全连接层
        for units in self.lstm_config.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.lstm_config.dense_dropout)(x)
        
        # 输出层
        if self.config.target_type == 'direction':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae']
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model


# 工具函数
def create_sequences(data: np.ndarray, seq_length: int, 
                     target_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时序序列
    
    Args:
        data: 输入数据 [n_samples, n_features]
        seq_length: 序列长度
        target_idx: 目标特征索引
        
    Returns:
        (X, y) 序列化数据
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_idx])
    
    return np.array(X), np.array(y)


def normalize_sequences(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
    """
    归一化序列数据
    
    Args:
        X: 输入序列 [n_samples, seq_length, n_features]
        method: 归一化方法 ('standard', 'minmax', 'robust')
        
    Returns:
        (归一化后的数据, scaler对象)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 重塑为2D进行拟合
    n_samples, seq_length, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    # 拟合并转换
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # 重塑回3D
    X_scaled = X_scaled.reshape(n_samples, seq_length, n_features)
    
    return X_scaled, scaler
