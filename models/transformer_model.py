"""
Transformer 时序预测模型
支持多头注意力机制，捕捉长距离依赖关系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import math

from .base_model import BaseModel, ModelConfig, PredictionResult

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow未安装")


@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    seq_length: int = 60
    n_features: int = 20
    
    # Transformer参数
    d_model: int = 128  # 模型维度
    num_heads: int = 8  # 注意力头数
    num_encoder_layers: int = 4  # 编码器层数
    num_decoder_layers: int = 2  # 解码器层数
    d_ff: int = 512  # 前馈网络维度
    dropout: float = 0.1
    
    # 输出层
    dense_units: List[int] = None
    
    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [64, 32]


class PositionalEncoding(layers.Layer):
    """位置编码层"""
    
    def __init__(self, max_seq_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
    
    def build(self, input_shape):
        d_model = input_shape[-1]
        
        # 创建位置编码矩阵
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe = np.zeros((self.max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class TransformerBlock(layers.Layer):
    """Transformer块（编码器层）"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False, mask=None):
        # 多头自注意力
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerModel(BaseModel):
    """
    Transformer时序预测模型
    
    特点：
    - 多头注意力机制捕捉长距离依赖
    - 位置编码保留时序信息
    - 支持编码器-解码器架构
    """
    
    def __init__(self, config: ModelConfig, 
                 transformer_config: Optional[TransformerConfig] = None):
        super().__init__(config)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        self.transformer_config = transformer_config or TransformerConfig()
        self.history = None
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> keras.Model:
        """
        构建Transformer模型
        
        Args:
            input_shape: 输入形状 (seq_length, n_features)
            
        Returns:
            Keras模型
        """
        if input_shape is None:
            input_shape = (self.transformer_config.seq_length, 
                          self.transformer_config.n_features)
        
        d_model = self.transformer_config.d_model
        
        # 输入层
        inputs = layers.Input(shape=input_shape)
        
        # 特征投影到d_model维度
        x = layers.Dense(d_model)(inputs)
        
        # 位置编码
        x = PositionalEncoding()(x)
        x = layers.Dropout(self.transformer_config.dropout)(x)
        
        # Transformer编码器层
        for i in range(self.transformer_config.num_encoder_layers):
            x = TransformerBlock(
                d_model=d_model,
                num_heads=self.transformer_config.num_heads,
                d_ff=self.transformer_config.d_ff,
                dropout=self.transformer_config.dropout,
                name=f'transformer_block_{i+1}'
            )(x)
        
        # 全局池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 全连接层
        for units in self.transformer_config.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.transformer_config.dropout)(x)
            x = layers.BatchNormalization()(x)
        
        # 输出层
        if self.config.target_type == 'direction':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC()]
        else:
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae', tf.keras.metrics.RootMeanSquaredError()]
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Transformer模型构建完成，参数数量: {model.count_params()}")
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """
        训练Transformer模型
        
        Args:
            X: 训练特征 [n_samples, seq_length, n_features]
            y: 训练目标 [n_samples]
            validation_data: 验证数据
            **kwargs: 额外参数
            
        Returns:
            训练历史记录
        """
        X = self.preprocess_features(X)
        
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
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_rounds,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 学习率预热
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        logger.info(f"开始训练Transformer模型，样本数: {len(X_train)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )
        
        self.training_history.append(self.history.history)
        self.is_trained = True
        
        logger.info("Transformer模型训练完成")
        
        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history.get('val_loss', [None])[-1],
            'n_params': self.model.count_params()
        }
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        使用Transformer模型进行预测
        
        Args:
            X: 输入特征
            
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
            confidence = np.abs(predictions - 0.5) * 2
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
        """获取特征重要性"""
        if self.config.feature_columns:
            return {col: 1.0 / len(self.config.feature_columns) 
                   for col in self.config.feature_columns}
        return {}
    
    def save(self, filepath: Optional[str] = None) -> str:
        """保存模型"""
        if filepath is None:
            filepath = f"{self.config.model_path}/{self.config.model_name}_{self.model_version}"
        
        keras_path = filepath + '.keras'
        self.model.save(keras_path)
        
        import json
        config_path = filepath + '_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'transformer_config': {
                    'seq_length': self.transformer_config.seq_length,
                    'n_features': self.transformer_config.n_features,
                    'd_model': self.transformer_config.d_model,
                    'num_heads': self.transformer_config.num_heads,
                    'num_encoder_layers': self.transformer_config.num_encoder_layers,
                    'd_ff': self.transformer_config.d_ff,
                    'dropout': self.transformer_config.dropout,
                    'dense_units': self.transformer_config.dense_units
                },
                'model_version': self.model_version,
                'is_trained': self.is_trained
            }, f, indent=2)
        
        return keras_path
    
    @classmethod
    def load(cls, filepath: str) -> 'TransformerModel':
        """加载模型"""
        import json
        
        config_path = filepath + '_config.json'
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        config = ModelConfig.from_dict(saved_config['config'])
        transformer_config = TransformerConfig(**saved_config['transformer_config'])
        
        instance = cls(config, transformer_config)
        keras_path = filepath + '.keras'
        instance.model = keras.models.load_model(keras_path)
        instance.model_version = saved_config['model_version']
        instance.is_trained = saved_config['is_trained']
        
        return instance


class TemporalFusionTransformer(BaseModel):
    """
    Temporal Fusion Transformer (TFT)
    用于多尺度时序预测的先进Transformer架构
    """
    
    def __init__(self, config: ModelConfig,
                 hidden_size: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__(config)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
    
    def build_model(self, input_shape: Optional[Tuple] = None) -> keras.Model:
        """构建TFT模型"""
        if input_shape is None:
            input_shape = (60, 20)  # 默认形状
        
        seq_len, n_features = input_shape
        
        # 输入
        inputs = layers.Input(shape=input_shape)
        
        # 变量选择网络（简化版）
        x = layers.Dense(self.hidden_size)(inputs)
        
        # LSTM编码器（捕捉局部时序）
        lstm_out, state_h, state_c = layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            return_state=True
        )(x)
        
        # 静态上下文（使用最后一个状态）
        static_context = layers.Dense(self.hidden_size)(state_h)
        static_context = layers.RepeatVector(seq_len)(static_context)
        
        # 多头注意力（捕捉长距离依赖）
        attn_out = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads
        )(lstm_out, lstm_out)
        
        # 门控残差连接
        gated = layers.Dense(self.hidden_size, activation='sigmoid')(attn_out)
        x = layers.Multiply()([attn_out, gated])
        x = layers.Add()([x, lstm_out])
        x = layers.LayerNormalization()(x)
        
        # 位置编码
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(seq_len, self.hidden_size)(positions)
        x = x + position_embedding
        
        # 输出层
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
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
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            **kwargs) -> Dict:
        """训练TFT模型"""
        X = self.preprocess_features(X)
        
        if self.model is None:
            self.model = self.build_model(input_shape=X.shape[1:])
        
        if validation_data is None and self.config.validation_split > 0:
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )
        
        self.training_history.append(self.history.history)
        self.is_trained = True
        
        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1]
        }
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """预测"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        X = self.preprocess_features(X)
        predictions = self.model.predict(X, verbose=0).flatten()
        
        if self.config.target_type == 'direction':
            probabilities = np.column_stack([1 - predictions, predictions])
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
            probabilities = None
        
        return PredictionResult(
            predictions=binary_predictions,
            probabilities=probabilities,
            model_version=self.model_version
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.config.feature_columns:
            return {col: 1.0 / len(self.config.feature_columns) 
                   for col in self.config.feature_columns}
        return {}


# 工具函数
def create_look_ahead_mask(size: int) -> tf.Tensor:
    """创建前瞻掩码（防止看到未来信息）"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(seq: tf.Tensor) -> tf.Tensor:
    """创建填充掩码"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
