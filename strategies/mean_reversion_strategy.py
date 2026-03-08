"""
均值回归策略
基于布林带和RSI的均值回归交易
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from strategies.base_strategy import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    
    策略逻辑：
    1. 价格触及布林带上轨 + RSI超买 -> 卖出
    2. 价格触及布林带下轨 + RSI超卖 -> 买入
    3. 价格回归中轨时平仓
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("MeanReversionStrategy", config)
        
        # 策略参数
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 0.5)
        
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """初始化策略"""
        self.is_initialized = True
        print(f"均值回归策略初始化完成，布林带周期:{self.bb_period}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """生成均值回归信号"""
        signals = []
        
        for symbol, df in data.items():
            try:
                # 计算布林带
                df['BB_middle'] = df['close'].rolling(window=self.bb_period).mean()
                bb_std = df['close'].rolling(window=self.bb_period).std()
                df['BB_upper'] = df['BB_middle'] + self.bb_std * bb_std
                df['BB_lower'] = df['BB_middle'] - self.bb_std * bb_std
                df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
                
                # 计算RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 计算Z-score
                df['Z_score'] = (df['close'] - df['BB_middle']) / bb_std
                
                # 获取最新数据
                current = df.iloc[-1]
                current_price = current['close']
                timestamp = df.index[-1]
                
                # 判断信号
                signal_type = SignalType.HOLD
                confidence = 0.5
                
                # 超卖信号（买入）
                if current['BB_position'] < 0.1 and current['RSI'] < self.rsi_oversold:
                    signal_type = SignalType.BUY
                    confidence = min((self.rsi_oversold - current['RSI']) / self.rsi_oversold + 
                                   (0.1 - current['BB_position']) / 0.1, 1.0)
                
                # 超买信号（卖出）
                elif current['BB_position'] > 0.9 and current['RSI'] > self.rsi_overbought:
                    signal_type = SignalType.SELL
                    confidence = min((current['RSI'] - self.rsi_overbought) / (100 - self.rsi_overbought) +
                                   (current['BB_position'] - 0.9) / 0.1, 1.0)
                
                if signal_type != SignalType.HOLD:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=confidence,
                        predicted_return=abs(current['Z_score']) * 0.01,  # 基于Z-score预估收益
                        current_price=current_price,
                        timestamp=timestamp,
                        metadata={
                            'bb_position': current['BB_position'],
                            'bb_width': (current['BB_upper'] - current['BB_lower']) / current['BB_middle'],
                            'rsi': current['RSI'],
                            'z_score': current['Z_score']
                        }
                    )
                    signals.append(signal)
                    self.signals_history.append(signal)
                    
            except Exception as e:
                print(f"生成{symbol}均值回归信号失败: {e}")
                continue
        
        return signals
    
    def calculate_position_size(self, signal: Signal, 
                                account_value: float) -> float:
        """
        计算仓位大小
        
        基于布林带宽度和Z-score调整仓位
        """
        bb_width = signal.metadata.get('bb_width', 0.05)
        z_score = abs(signal.metadata.get('z_score', 2))
        
        # 布林带越窄、Z-score越大，仓位越大
        base_size = 0.03
        width_adjustment = 0.05 / max(bb_width, 0.01)  # 带宽调整
        z_adjustment = min(z_score / 3, 2)  # Z-score调整
        
        adjusted_size = base_size * width_adjustment * z_adjustment * signal.confidence
        adjusted_size = min(adjusted_size, 0.1)  # 限制最大10%
        
        position_value = account_value * adjusted_size
        
        if signal.signal_type == SignalType.SELL:
            position_value = -position_value
        
        return position_value
    
    def should_exit(self, position: Dict, current_data: pd.Series) -> bool:
        """
        判断是否应该平仓（均值回归完成）
        
        Args:
            position: 持仓信息
            current_data: 当前数据
            
        Returns:
            是否应该平仓
        """
        bb_position = current_data.get('BB_position', 0.5)
        
        # 价格回归中轨附近
        if position['direction'] == 'long' and bb_position > 0.5:
            return True
        if position['direction'] == 'short' and bb_position < 0.5:
            return True
        
        return False
