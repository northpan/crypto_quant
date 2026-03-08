"""
量化模型组合模块

提供多种机器学习模型用于加密货币量化交易：
- 基础模型：LightGBM, XGBoost, RandomForest
- 深度学习：LSTM, Transformer, TFT
- 模型集成：Stacking, Blending, Voting
- 训练流程：超参数优化、交叉验证、在线学习
- 模型评估：回测、夏普比率、可视化
"""

from .base_model import (
    BaseModel,
    ModelConfig,
    PredictionResult,
    ModelMetrics,
    ModelRegistry,
    calculate_returns_metrics
)

from .lightgbm_model import (
    LightGBMModel,
    LightGBMQuantileModel,
    create_lightgbm_datasets,
    train_with_cv
)

from .xgboost_model import (
    XGBoostModel,
    XGBoostEnsemble,
    create_xgb_dmatrix,
    xgb_shap_explainer
)

try:
    from .lstm_model import (
        LSTMModel,
        BidirectionalLSTMModel,
        AttentionLSTMModel,
        LSTMConfig,
        create_sequences,
        normalize_sequences
    )
except (ImportError, NameError) as e:
    LSTMModel = None
    BidirectionalLSTMModel = None
    AttentionLSTMModel = None
    LSTMConfig = None
    create_sequences = None
    normalize_sequences = None

try:
    from .transformer_model import (
        TransformerModel,
        TemporalFusionTransformer,
        TransformerConfig,
        PositionalEncoding,
        TransformerBlock
    )
except (ImportError, NameError) as e:
    TransformerModel = None
    TemporalFusionTransformer = None
    TransformerConfig = None
    PositionalEncoding = None
    TransformerBlock = None

from .ensemble_model import (
    EnsembleModel,
    DynamicEnsemble,
    EnsembleConfig,
    StackingRegressor,
    create_diverse_ensemble,
    calculate_ensemble_diversity
)

from .model_trainer import (
    ModelTrainer,
    OnlineTrainer,
    TrainingConfig,
    create_training_pipeline,
    compare_models
)

from .model_evaluator import (
    ModelEvaluator,
    BacktestEngine,
    EvaluationReport,
    quick_evaluate,
    print_evaluation_report
)

__version__ = '1.0.0'
__author__ = 'CryptoQuant Team'

__all__ = [
    # 基础类
    'BaseModel',
    'ModelConfig',
    'PredictionResult',
    'ModelMetrics',
    'ModelRegistry',
    
    # LightGBM
    'LightGBMModel',
    'LightGBMQuantileModel',
    
    # XGBoost
    'XGBoostModel',
    'XGBoostEnsemble',
    
    # LSTM
    'LSTMModel',
    'BidirectionalLSTMModel',
    'AttentionLSTMModel',
    'LSTMConfig',
    
    # Transformer
    'TransformerModel',
    'TemporalFusionTransformer',
    'TransformerConfig',
    
    # 集成
    'EnsembleModel',
    'DynamicEnsemble',
    'EnsembleConfig',
    'StackingRegressor',
    
    # 训练
    'ModelTrainer',
    'OnlineTrainer',
    'TrainingConfig',
    
    # 评估
    'ModelEvaluator',
    'BacktestEngine',
    'EvaluationReport',
    
    # 工具函数
    'calculate_returns_metrics',
    'quick_evaluate',
    'print_evaluation_report',
    'create_training_pipeline',
    'compare_models'
]
