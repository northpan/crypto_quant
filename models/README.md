# 量化模型组合模块

高性能机器学习模型组合，目标夏普比率 > 4，支持分钟级预测。

## 模块结构

```
models/
├── __init__.py              # 模块入口
├── base_model.py            # 模型基类和通用工具
├── lightgbm_model.py        # LightGBM模型
├── xgboost_model.py         # XGBoost模型
├── lstm_model.py            # LSTM时序模型
├── transformer_model.py     # Transformer模型
├── ensemble_model.py        # 模型集成
├── model_trainer.py         # 训练流程
├── model_evaluator.py       # 模型评估
├── example_training.py      # 训练示例
└── README.md                # 本文档
```

## 功能特性

### 1. 基础模型
- **LightGBM**: 快速梯度提升，支持分类/回归/分位数预测
- **XGBoost**: 强大的正则化，支持GPU加速
- **随机森林**: 内置Bagging集成

### 2. 深度学习模型
- **LSTM**: 多层LSTM，支持双向和注意力机制
- **Transformer**: 多头注意力，捕捉长距离依赖
- **Temporal Fusion Transformer**: 多尺度时序预测

### 3. 模型集成
- **Stacking**: 使用元模型组合基模型
- **Blending**: Hold-out验证集组合
- **Voting**: 投票集成
- **动态权重**: 根据市场状态调整权重

### 4. 训练流程
- 超参数优化（Optuna/Grid Search）
- 时间序列交叉验证
- 滚动窗口训练（Walk-forward）
- 在线学习

### 5. 模型评估
- 回测模拟
- 夏普比率/最大回撤/Calmar比率
- 统计检验
- 可视化报告

## 快速开始

### 安装依赖

```bash
pip install numpy pandas scikit-learn
pip install lightgbm xgboost
pip install tensorflow
pip install optuna matplotlib
```

### 基本使用

```python
from models import (
    ModelConfig, LightGBMModel, 
    ModelTrainer, ModelEvaluator
)

# 1. 准备数据
X_train, y_train = ...  # 你的数据
X_test, y_test = ...

# 2. 创建模型配置
config = ModelConfig(
    model_name='my_model',
    model_type='lightgbm',
    target_type='return',
    prediction_horizon=60
)

# 3. 创建并训练模型
model = LightGBMModel(config)
model.fit(X_train, y_train)

# 4. 预测
predictions = model.predict(X_test)

# 5. 评估
evaluator = ModelEvaluator()
report = evaluator.evaluate(model, X_test, y_test)
print(f"夏普比率: {report.metrics.sharpe_ratio}")
```

### 集成模型

```python
from models import EnsembleModel, EnsembleConfig

# 创建集成模型
ensemble_config = EnsembleConfig(
    ensemble_method='stacking',
    meta_model_type='ridge'
)

ensemble = EnsembleModel(config, ensemble_config)

# 添加基模型
ensemble.add_model('lightgbm', lgb_model)
ensemble.add_model('xgboost', xgb_model)
ensemble.add_model('lstm', lstm_model)

# 训练
ensemble.fit(X_train, y_train)

# 预测
predictions = ensemble.predict(X_test)
```

### LSTM时序预测

```python
from models import LSTMModel, LSTMConfig

# 创建LSTM配置
lstm_config = LSTMConfig(
    seq_length=60,           # 60分钟序列
    n_features=20,           # 20个特征
    lstm_units=[128, 64],    # LSTM层大小
    lstm_dropout=0.2
)

# 创建模型
model = LSTMModel(config, lstm_config)

# 训练（X形状: [n_samples, seq_length, n_features]）
model.fit(X_train_seq, y_train)

# 预测
predictions = model.predict(X_test_seq)
```

## 配置参数

### ModelConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | - | 模型名称 |
| model_type | str | - | 模型类型 |
| prediction_horizon | int | 60 | 预测未来N分钟 |
| target_type | str | 'return' | 目标类型 |
| learning_rate | float | 0.01 | 学习率 |
| epochs | int | 100 | 训练轮数 |
| batch_size | int | 32 | 批次大小 |
| early_stopping_rounds | int | 50 | 早停轮数 |

### TrainingConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| train_ratio | float | 0.7 | 训练集比例 |
| val_ratio | float | 0.15 | 验证集比例 |
| use_time_split | bool | True | 按时间划分 |
| enable_hyperopt | bool | True | 启用超参优化 |

## 评估指标

- **MSE/RMSE/MAE**: 回归误差
- **R²**: 决定系数
- **Accuracy/Precision/Recall/F1**: 分类指标
- **Sharpe Ratio**: 夏普比率（年化）
- **Max Drawdown**: 最大回撤
- **Calmar Ratio**: Calmar比率

## 性能目标

- 夏普比率 > 4
- 支持分钟级预测
- 预测延迟 < 10ms
- 支持在线学习

## 示例运行

```bash
cd /mnt/okcomputer/output/crypto_quant/models
python example_training.py
```

## 注意事项

1. **数据预处理**: 确保输入数据已归一化/标准化
2. **特征工程**: 高质量特征比复杂模型更重要
3. **过拟合**: 使用正则化和交叉验证防止过拟合
4. **回测**: 前向验证比简单回测更可靠

## 许可证

MIT License
