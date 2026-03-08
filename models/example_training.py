"""
模型训练示例
展示如何使用量化模型组合模块
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000, n_features: int = 20,
                            seq_length: int = 60, seed: int = 42) -> dict:
    """
    生成合成数据用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        seq_length: 序列长度（用于时序模型）
        seed: 随机种子
        
    Returns:
        数据字典
    """
    np.random.seed(seed)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些相关性
    for i in range(1, min(5, n_features)):
        X[:, i] = X[:, i-1] * 0.5 + np.random.randn(n_samples) * 0.5
    
    # 生成目标（收益率）- 添加一些可预测性
    weights = np.random.randn(n_features) * 0.1
    y = X @ weights + np.random.randn(n_samples) * 0.5
    
    # 添加趋势
    trend = np.sin(np.linspace(0, 10 * np.pi, n_samples)) * 0.1
    y += trend
    
    # 生成时间戳
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # 生成价格序列
    returns = y / 100  # 转换为收益率
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 创建时序数据（用于LSTM/Transformer）
    X_seq = []
    y_seq = []
    for i in range(n_samples - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return {
        'X': X,
        'y': y,
        'X_seq': X_seq,
        'y_seq': y_seq,
        'timestamps': timestamps,
        'prices': prices,
        'returns': returns
    }


def example_lightgbm_training():
    """LightGBM模型训练示例"""
    logger.info("\n" + "="*60)
    logger.info("LightGBM模型训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from lightgbm_model import LightGBMModel
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator, print_evaluation_report
    
    # 生成数据
    data = generate_synthetic_data(n_samples=5000)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    
    # 划分数据
    splits = trainer.prepare_data(data['X'], data['y'])
    
    # 创建模型配置
    config = ModelConfig(
        model_name='lightgbm_example',
        model_type='lightgbm',
        target_type='return',
        prediction_horizon=60,
        learning_rate=0.05,
        epochs=100,
        early_stopping_rounds=20
    )
    
    # 创建并训练模型
    model = LightGBMModel(config)
    
    train_result = trainer.train(
        model,
        splits['X_train'],
        splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val'])
    )
    
    logger.info(f"训练结果: {train_result}")
    
    # 评估
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, splits['X_test'], splits['y_test'])
    
    print_evaluation_report(report)
    
    # 特征重要性
    importance = model.get_feature_importance()
    logger.info(f"\nTop 10 特征重要性:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {feat}: {imp:.4f}")
    
    return model, report


def example_xgboost_training():
    """XGBoost模型训练示例"""
    logger.info("\n" + "="*60)
    logger.info("XGBoost模型训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from xgboost_model import XGBoostModel
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator, print_evaluation_report
    
    # 生成数据
    data = generate_synthetic_data(n_samples=5000)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    splits = trainer.prepare_data(data['X'], data['y'])
    
    # 创建模型
    config = ModelConfig(
        model_name='xgboost_example',
        model_type='xgboost',
        target_type='return',
        learning_rate=0.05,
        epochs=100
    )
    
    model = XGBoostModel(config)
    
    # 训练
    train_result = trainer.train(
        model,
        splits['X_train'],
        splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val'])
    )
    
    logger.info(f"训练结果: {train_result}")
    
    # 评估
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, splits['X_test'], splits['y_test'])
    
    print_evaluation_report(report)
    
    return model, report


def example_lstm_training():
    """LSTM模型训练示例"""
    logger.info("\n" + "="*60)
    logger.info("LSTM模型训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from lstm_model import LSTMModel, LSTMConfig
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator, print_evaluation_report
    
    # 生成时序数据
    data = generate_synthetic_data(n_samples=5000, seq_length=60)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    splits = trainer.prepare_data(data['X_seq'], data['y_seq'])
    
    # 创建LSTM配置
    lstm_config = LSTMConfig(
        seq_length=60,
        n_features=20,
        lstm_units=[64, 32],
        dense_units=[16],
        lstm_dropout=0.2,
        recurrent_dropout=0.2
    )
    
    config = ModelConfig(
        model_name='lstm_example',
        model_type='lstm',
        target_type='return',
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        early_stopping_rounds=10
    )
    
    model = LSTMModel(config, lstm_config)
    
    # 训练
    train_result = trainer.train(
        model,
        splits['X_train'],
        splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val']),
        verbose=1
    )
    
    logger.info(f"训练结果: {train_result}")
    
    # 评估
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, splits['X_test'], splits['y_test'])
    
    print_evaluation_report(report)
    
    return model, report


def example_transformer_training():
    """Transformer模型训练示例"""
    logger.info("\n" + "="*60)
    logger.info("Transformer模型训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from transformer_model import TransformerModel, TransformerConfig
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator, print_evaluation_report
    
    # 生成时序数据
    data = generate_synthetic_data(n_samples=5000, seq_length=60)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    splits = trainer.prepare_data(data['X_seq'], data['y_seq'])
    
    # 创建Transformer配置
    transformer_config = TransformerConfig(
        seq_length=60,
        n_features=20,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        d_ff=128,
        dropout=0.1,
        dense_units=[32]
    )
    
    config = ModelConfig(
        model_name='transformer_example',
        model_type='transformer',
        target_type='return',
        learning_rate=0.001,
        epochs=30,
        batch_size=32,
        early_stopping_rounds=10
    )
    
    model = TransformerModel(config, transformer_config)
    
    # 训练
    train_result = trainer.train(
        model,
        splits['X_train'],
        splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val']),
        verbose=1
    )
    
    logger.info(f"训练结果: {train_result}")
    
    # 评估
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, splits['X_test'], splits['y_test'])
    
    print_evaluation_report(report)
    
    return model, report


def example_ensemble_training():
    """集成模型训练示例"""
    logger.info("\n" + "="*60)
    logger.info("集成模型训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from lightgbm_model import LightGBMModel
    from xgboost_model import XGBoostModel
    from ensemble_model import EnsembleModel, EnsembleConfig
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator, print_evaluation_report
    
    # 生成数据
    data = generate_synthetic_data(n_samples=5000)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    splits = trainer.prepare_data(data['X'], data['y'])
    
    # 创建集成模型
    ensemble_config = EnsembleConfig(
        ensemble_method='stacking',
        meta_model_type='ridge',
        cv_folds=3
    )
    
    config = ModelConfig(
        model_name='ensemble_example',
        model_type='ensemble',
        target_type='return'
    )
    
    ensemble = EnsembleModel(config, ensemble_config)
    
    # 添加基模型
    lgb_config = ModelConfig(
        model_name='lgb_base',
        model_type='lightgbm',
        learning_rate=0.05,
        epochs=50
    )
    lgb_model = LightGBMModel(lgb_config)
    ensemble.add_model('lightgbm', lgb_model)
    
    xgb_config = ModelConfig(
        model_name='xgb_base',
        model_type='xgboost',
        learning_rate=0.05,
        epochs=50
    )
    xgb_model = XGBoostModel(xgb_config)
    ensemble.add_model('xgboost', xgb_model)
    
    # 训练集成模型
    train_result = trainer.train_ensemble(
        ensemble,
        splits['X_train'],
        splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val'])
    )
    
    logger.info(f"训练结果: {train_result}")
    
    # 评估
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(ensemble, splits['X_test'], splits['y_test'])
    
    print_evaluation_report(report)
    
    # 模型权重
    logger.info(f"\n模型权重:")
    for name, weight in ensemble.model_weights.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    return ensemble, report


def example_walk_forward_training():
    """滚动窗口训练示例"""
    logger.info("\n" + "="*60)
    logger.info("滚动窗口训练示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from lightgbm_model import LightGBMModel
    from model_trainer import ModelTrainer, TrainingConfig
    
    # 生成数据
    data = generate_synthetic_data(n_samples=3000)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    
    # 创建模型配置
    config = ModelConfig(
        model_name='walk_forward_example',
        model_type='lightgbm',
        target_type='return'
    )
    
    # 滚动窗口训练
    results = trainer.walk_forward_train(
        LightGBMModel,
        config,
        data['X'],
        data['y'],
        data['timestamps'].values,
        train_window=500,
        test_window=100,
        step_size=50
    )
    
    logger.info(f"滚动窗口结果:")
    logger.info(f"  总窗口数: {len(results['window_results'])}")
    logger.info(f"  整体夏普比率: {results['overall_sharpe']:.4f}")
    logger.info(f"  总收益率: {results['total_return']:.4f}")
    
    return results


def example_model_comparison():
    """模型对比示例"""
    logger.info("\n" + "="*60)
    logger.info("模型对比示例")
    logger.info("="*60)
    
    from base_model import ModelConfig
    from lightgbm_model import LightGBMModel
    from xgboost_model import XGBoostModel
    from model_trainer import ModelTrainer, TrainingConfig
    from model_evaluator import ModelEvaluator
    
    # 生成数据
    data = generate_synthetic_data(n_samples=5000)
    
    # 创建训练器
    trainer = ModelTrainer(TrainingConfig())
    splits = trainer.prepare_data(data['X'], data['y'])
    
    # 训练多个模型
    models = []
    
    # LightGBM
    lgb_config = ModelConfig(model_name='LightGBM', model_type='lightgbm')
    lgb_model = LightGBMModel(lgb_config)
    lgb_model.fit(splits['X_train'], splits['y_train'], verbose=False)
    models.append(lgb_model)
    
    # XGBoost
    xgb_config = ModelConfig(model_name='XGBoost', model_type='xgboost')
    xgb_model = XGBoostModel(xgb_config)
    xgb_model.fit(splits['X_train'], splits['y_train'], verbose=False)
    models.append(xgb_model)
    
    # 对比
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(models, splits['X_test'], splits['y_test'])
    
    logger.info("\n模型对比结果:")
    print(comparison)
    
    return comparison


def run_all_examples():
    """运行所有示例"""
    logger.info("\n" + "="*80)
    logger.info("量化模型组合 - 完整示例")
    logger.info("="*80)
    
    results = {}
    
    # 1. LightGBM
    try:
        results['lightgbm'] = example_lightgbm_training()
    except Exception as e:
        logger.error(f"LightGBM示例失败: {e}")
    
    # 2. XGBoost
    try:
        results['xgboost'] = example_xgboost_training()
    except Exception as e:
        logger.error(f"XGBoost示例失败: {e}")
    
    # 3. LSTM
    try:
        results['lstm'] = example_lstm_training()
    except Exception as e:
        logger.error(f"LSTM示例失败: {e}")
    
    # 4. Transformer
    try:
        results['transformer'] = example_transformer_training()
    except Exception as e:
        logger.error(f"Transformer示例失败: {e}")
    
    # 5. Ensemble
    try:
        results['ensemble'] = example_ensemble_training()
    except Exception as e:
        logger.error(f"集成示例失败: {e}")
    
    # 6. Walk-forward
    try:
        results['walk_forward'] = example_walk_forward_training()
    except Exception as e:
        logger.error(f"滚动窗口示例失败: {e}")
    
    # 7. 模型对比
    try:
        results['comparison'] = example_model_comparison()
    except Exception as e:
        logger.error(f"模型对比示例失败: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("所有示例运行完成")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    # 运行所有示例
    results = run_all_examples()
