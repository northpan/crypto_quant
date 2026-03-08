"""
动量策略
基于价格动量和趋势跟踪
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from strategies.base_strategy import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """
    动量策略
    
    策略逻辑：
    1. 短期均线上穿长期均线 -> 买入
    2. 短期均线下穿长期均线 -> 卖出
    3. 结合RSI过滤超买超卖
    4. 结合成交量确认趋势
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("MomentumStrategy", config)
        
        # 策略参数
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 30)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.volume_confirm = config.get('volume_confirm', True)
        
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """初始化策略"""
        self.is_initialized = True
        print(f"动量策略初始化完成，短周期:{self.short_window}, 长周期:{self.long_window}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """生成动量信号"""
        signals = []
        
        for symbol, df in data.items():
            try:
                # 计算移动平均线
                df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
                df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
                
                # 计算RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 计算成交量均值
                if self.volume_confirm:
                    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
                
                # 获取最新数据
                current = df.iloc[-1]
                prev = df.iloc[-2]
                
                current_price = current['close']
                timestamp = df.index[-1]
                
                # 判断信号
                signal_type = SignalType.HOLD
                confidence = 0.5
                
                # 金叉买入
                if prev['SMA_short'] <= prev['SMA_long'] and current['SMA_short'] > current['SMA_long']:
                    if current['RSI'] < self.rsi_overbought:  # 非超买
                        if not self.volume_confirm or current['volume'] > current['Volume_MA']:
                            signal_type = SignalType.BUY
                            confidence = (self.rsi_overbought - current['RSI']) / self.rsi_overbought
                
                # 死叉卖出
                elif prev['SMA_short'] >= prev['SMA_long'] and current['SMA_short'] < current['SMA_long']:
                    if current['RSI'] > self.rsi_oversold:  # 非超卖
                        if not self.volume_confirm or current['volume'] > current['Volume_MA']:
                            signal_type = SignalType.SELL
                            confidence = (current['RSI'] - self.rsi_oversold) / (100 - self.rsi_oversold)
                
                if signal_type != SignalType.HOLD:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=confidence,
                        predicted_return=confidence * 0.02,  # 预估收益
                        current_price=current_price,
                        timestamp=timestamp,
                        metadata={
                            'sma_short': current['SMA_short'],
                            'sma_long': current['SMA_long'],
                            'rsi': current['RSI']
                        }
                    )
                    signals.append(signal)
                    self.signals_history.append(signal)
                    
            except Exception as e:
                print(f"生成{symbol}动量信号失败: {e}")
                continue
        
        return signals
    
    def calculate_position_size(self, signal: Signal, 
                                account_value: float) -> float:
        """计算仓位大小"""
        base_size = 0.05  # 基础仓位5%
        adjusted_size = base_size * signal.confidence
        position_value = account_value * adjusted_size
        
        if signal.signal_type == SignalType.SELL:
            position_value = -position_value
        
        return position_value
