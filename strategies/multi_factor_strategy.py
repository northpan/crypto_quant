"""
多因子策略
结合122个因子，使用集成模型生成信号
目标夏普比率 > 4
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.base_strategy import BaseStrategy, Signal, SignalType
from factors import FactorPool
from models import EnsembleModel, ModelConfig


class MultiFactorStrategy(BaseStrategy):
    """
    多因子量化策略
    
    特点：
    1. 使用122个因子（技术、量价、波动率、订单流、跨市场）
    2. 集成模型预测（LightGBM + XGBoost + LSTM）
    3. 动态仓位管理
    4. 多时间框架确认
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化多因子策略
        
        Args:
            config: 策略配置
        """
        super().__init__("MultiFactorStrategy", config)
        
        # 初始化因子池
        self.factor_pool = FactorPool()
        
        # 初始化预测模型
        model_config = ModelConfig(
            model_name='multi_factor_predictor',
            model_type='ensemble',
            prediction_horizon=config.get('prediction_horizon', 60),
            target_sharpe=config.get('target_sharpe', 4.0)
        )
        self.model = EnsembleModel(model_config)
        
        # 策略参数
        self.lookback = config.get('lookback', 120)
        self.min_confidence = config.get('min_confidence', 0.6)
        self.max_positions = config.get('max_positions', 10)
        
        # 信号缓存
        self.signal_cache = {}
        
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """
        初始化策略，训练模型
        
        Args:
            data: 历史数据
        """
        print(f"初始化多因子策略，训练数据包含 {len(data)} 个品种...")
        
        # 准备训练数据
        X_list = []
        y_list = []
        
        for symbol, df in data.items():
            # 计算所有因子
            df_factors = self.factor_pool.calculate_all_factors(df)
            
            # 选择特征列
            feature_cols = [c for c in df_factors.columns 
                          if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            # 计算目标（未来收益率）
            horizon = self.config.get('prediction_horizon', 60)
            df_factors['target'] = df_factors['close'].pct_change(horizon).shift(-horizon)
            
            # 删除NaN
            df_clean = df_factors.dropna()
            
            if len(df_clean) > self.lookback:
                X_list.append(df_clean[feature_cols])
                y_list.append(df_clean['target'])
        
        if X_list and y_list:
            X = pd.concat(X_list, ignore_index=True)
            y = pd.concat(y_list, ignore_index=True)
            
            # 训练模型
            print(f"训练模型，样本数: {len(X)}")
            self.model.fit(X, y)
            self.is_initialized = True
            print("模型训练完成")
        else:
            print("警告：没有足够的训练数据")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            data: 最新数据
            
        Returns:
            交易信号列表
        """
        signals = []
        
        for symbol, df in data.items():
            try:
                # 计算因子
                df_factors = self.factor_pool.calculate_all_factors(df)
                
                # 选择特征
                feature_cols = [c for c in df_factors.columns 
                              if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                
                # 获取最新特征
                latest = df_factors[feature_cols].iloc[-1:]
                current_price = df['close'].iloc[-1]
                timestamp = df.index[-1]
                
                # 预测
                prediction = self.model.predict(latest)[0]
                
                # 计算置信度（基于预测值和历史波动率）
                volatility = df['close'].pct_change().std() * np.sqrt(1440)
                confidence = min(abs(prediction) / (volatility * 0.5), 1.0)
                
                # 生成信号
                if confidence >= self.min_confidence:
                    if prediction > 0:
                        signal_type = SignalType.BUY
                    else:
                        signal_type = SignalType.SELL
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=confidence,
                        predicted_return=prediction,
                        current_price=current_price,
                        timestamp=timestamp,
                        metadata={
                            'volatility': volatility,
                            'lookback_data': len(df)
                        }
                    )
                    signals.append(signal)
                    self.signals_history.append(signal)
                    
            except Exception as e:
                print(f"生成{symbol}信号失败: {e}")
                continue
        
        # 按置信度排序，选择前N个
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, signal: Signal, 
                                account_value: float) -> float:
        """
        计算仓位大小
        
        使用波动率目标法：
        仓位 = (目标波动率 / 品种波动率) * 置信度
        
        Args:
            signal: 交易信号
            account_value: 账户价值
            
        Returns:
            仓位大小
        """
        # 目标波动率（日波动率2%）
        target_volatility = 0.02
        
        # 品种波动率
        symbol_volatility = signal.metadata.get('volatility', 0.05)
        
        # 基于波动率的仓位
        vol_based_size = target_volatility / symbol_volatility
        
        # 考虑置信度
        confidence_adjusted = vol_based_size * signal.confidence
        
        # 限制最大仓位
        max_position = self.config.get('max_position_pct', 0.1)
        position_size = min(confidence_adjusted, max_position)
        
        # 转换为金额
        position_value = account_value * position_size
        
        # 根据信号方向确定正负
        if signal.signal_type == SignalType.SELL:
            position_value = -position_value
        
        return position_value
    
    def get_factor_exposure(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        获取因子暴露
        
        Args:
            data: 最新数据
            
        Returns:
            各因子暴露度
        """
        exposures = {}
        
        for symbol, df in data.items():
            df_factors = self.factor_pool.calculate_all_factors(df)
            
            # 计算各因子类别的平均暴露
            tech_factors = [c for c in df_factors.columns if 'rsi' in c or 'macd' in c or 'bb' in c]
            vol_factors = [c for c in df_factors.columns if 'atr' in c or 'volatility' in c]
            
            if tech_factors:
                exposures[f'{symbol}_technical'] = df_factors[tech_factors].iloc[-1].mean()
            if vol_factors:
                exposures[f'{symbol}_volatility'] = df_factors[vol_factors].iloc[-1].mean()
        
        return exposures
