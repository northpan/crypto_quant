"""
交易分析模块
分析交易记录，计算交易相关指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeMetrics:
    """交易指标数据类"""
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # 胜率
    win_rate: float = 0.0
    loss_rate: float = 0.0
    
    # 盈亏
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    
    # 盈亏比
    profit_factor: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_loss_ratio: float = 0.0
    
    # 单笔统计
    avg_trade_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # 连续统计
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_consecutive_wins: float = 0.0
    avg_consecutive_losses: float = 0.0
    
    # 持仓时间
    avg_holding_period: float = 0.0
    avg_win_holding_period: float = 0.0
    avg_loss_holding_period: float = 0.0
    
    # 其他
    expectancy: float = 0.0
    r_multiple_avg: float = 0.0
    sqn: float = 0.0  # System Quality Number


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    side: str = "long"  # long, short
    pnl: float = 0.0
    return_pct: float = 0.0
    fees: float = 0.0
    
    @property
    def is_win(self) -> bool:
        return self.pnl > 0
    
    @property
    def is_loss(self) -> bool:
        return self.pnl < 0
    
    @property
    def holding_period(self) -> Optional[timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None


class TradeAnalyzer:
    """
    交易分析器
    
    分析交易记录，计算:
    - 胜率、盈亏比
    - 连续盈亏统计
    - 持仓时间分析
    - 品种贡献分析
    """
    
    def __init__(self, risk_per_trade: float = 0.01):
        """
        Args:
            risk_per_trade: 每笔交易风险比例 (用于计算R-multiple)
        """
        self.risk_per_trade = risk_per_trade
        self.trades: List[Trade] = []
    
    def add_trade(self, trade: Trade):
        """添加交易记录"""
        self.trades.append(trade)
    
    def add_trades_from_dataframe(self, df: pd.DataFrame):
        """从DataFrame添加交易记录"""
        required_cols = ['trade_id', 'symbol', 'entry_time', 'exit_time', 
                        'entry_price', 'exit_price', 'quantity', 'pnl']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for _, row in df.iterrows():
            trade = Trade(
                trade_id=row['trade_id'],
                symbol=row['symbol'],
                entry_time=pd.to_datetime(row['entry_time']),
                exit_time=pd.to_datetime(row['exit_time']) if pd.notna(row['exit_time']) else None,
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                quantity=row['quantity'],
                side=row.get('side', 'long'),
                pnl=row['pnl'],
                return_pct=row.get('return_pct', 0),
                fees=row.get('fees', 0)
            )
            self.add_trade(trade)
    
    def analyze(self) -> TradeMetrics:
        """分析交易记录"""
        if not self.trades:
            return TradeMetrics()
        
        metrics = TradeMetrics()
        
        # 基本统计
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.is_win)
        metrics.losing_trades = sum(1 for t in self.trades if t.is_loss)
        metrics.break_even_trades = metrics.total_trades - metrics.winning_trades - metrics.losing_trades
        
        # 胜率
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        metrics.loss_rate = metrics.losing_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # 盈亏统计
        profits = [t.pnl for t in self.trades if t.is_win]
        losses = [t.pnl for t in self.trades if t.is_loss]
        
        metrics.gross_profit = sum(profits) if profits else 0
        metrics.gross_loss = sum(losses) if losses else 0
        metrics.net_profit = metrics.gross_profit + metrics.gross_loss
        metrics.total_pnl = metrics.net_profit
        
        # 盈亏比
        metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss) if metrics.gross_loss != 0 else float('inf')
        metrics.avg_profit = np.mean(profits) if profits else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.profit_loss_ratio = abs(metrics.avg_profit / metrics.avg_loss) if metrics.avg_loss != 0 else 0
        
        # 单笔统计
        all_pnls = [t.pnl for t in self.trades]
        metrics.avg_trade_pnl = np.mean(all_pnls)
        metrics.max_profit = max(all_pnls) if all_pnls else 0
        metrics.max_loss = min(all_pnls) if all_pnls else 0
        
        # 连续统计
        consecutive_stats = self._calculate_consecutive_stats()
        metrics.max_consecutive_wins = consecutive_stats['max_wins']
        metrics.max_consecutive_losses = consecutive_stats['max_losses']
        metrics.avg_consecutive_wins = consecutive_stats['avg_wins']
        metrics.avg_consecutive_losses = consecutive_stats['avg_losses']
        
        # 持仓时间
        holding_stats = self._calculate_holding_stats()
        metrics.avg_holding_period = holding_stats['avg']
        metrics.avg_win_holding_period = holding_stats['avg_win']
        metrics.avg_loss_holding_period = holding_stats['avg_loss']
        
        # 期望值
        metrics.expectancy = self._calculate_expectancy(metrics)
        
        # R-multiple
        metrics.r_multiple_avg = self._calculate_r_multiple_avg()
        
        # SQN
        metrics.sqn = self._calculate_sqn()
        
        return metrics
    
    def _calculate_consecutive_stats(self) -> Dict:
        """计算连续盈亏统计"""
        if not self.trades:
            return {'max_wins': 0, 'max_losses': 0, 'avg_wins': 0, 'avg_losses': 0}
        
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None
        
        for trade in self.trades:
            if trade.is_win:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_type == 'loss':
                        loss_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'win'
            elif trade.is_loss:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_type == 'win':
                        win_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'loss'
        
        # 添加最后一个连续序列
        if current_type == 'win':
            win_streaks.append(current_streak)
        elif current_type == 'loss':
            loss_streaks.append(current_streak)
        
        return {
            'max_wins': max(win_streaks) if win_streaks else 0,
            'max_losses': max(loss_streaks) if loss_streaks else 0,
            'avg_wins': np.mean(win_streaks) if win_streaks else 0,
            'avg_losses': np.mean(loss_streaks) if loss_streaks else 0
        }
    
    def _calculate_holding_stats(self) -> Dict:
        """计算持仓时间统计"""
        holding_periods = []
        win_periods = []
        loss_periods = []
        
        for trade in self.trades:
            if trade.holding_period:
                hours = trade.holding_period.total_seconds() / 3600
                holding_periods.append(hours)
                
                if trade.is_win:
                    win_periods.append(hours)
                elif trade.is_loss:
                    loss_periods.append(hours)
        
        return {
            'avg': np.mean(holding_periods) if holding_periods else 0,
            'avg_win': np.mean(win_periods) if win_periods else 0,
            'avg_loss': np.mean(loss_periods) if loss_periods else 0
        }
    
    def _calculate_expectancy(self, metrics: TradeMetrics) -> float:
        """计算期望值"""
        # 期望值 = (胜率 * 平均盈利) - (败率 * 平均亏损)
        win_expectation = metrics.win_rate * metrics.avg_profit
        loss_expectation = metrics.loss_rate * abs(metrics.avg_loss)
        return win_expectation - loss_expectation
    
    def _calculate_r_multiple_avg(self) -> float:
        """计算平均R-multiple"""
        # R-multiple = 盈亏 / 风险金额
        r_multiples = []
        for trade in self.trades:
            risk_amount = trade.entry_price * trade.quantity * self.risk_per_trade
            if risk_amount > 0:
                r_multiples.append(trade.pnl / risk_amount)
        
        return np.mean(r_multiples) if r_multiples else 0
    
    def _calculate_sqn(self) -> float:
        """计算系统质量数 (System Quality Number)"""
        # SQN = sqrt(N) * (平均R-multiple / R-multiple标准差)
        r_multiples = []
        for trade in self.trades:
            risk_amount = trade.entry_price * trade.quantity * self.risk_per_trade
            if risk_amount > 0:
                r_multiples.append(trade.pnl / risk_amount)
        
        if len(r_multiples) < 2:
            return 0
        
        avg_r = np.mean(r_multiples)
        std_r = np.std(r_multiples)
        
        if std_r == 0:
            return 0
        
        return np.sqrt(len(r_multiples)) * (avg_r / std_r)
    
    def analyze_by_symbol(self) -> pd.DataFrame:
        """按品种分析交易"""
        symbol_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0
        })
        
        for trade in self.trades:
            stats = symbol_stats[trade.symbol]
            stats['trades'] += 1
            stats['total_pnl'] += trade.pnl
            
            if trade.is_win:
                stats['wins'] += 1
                stats['gross_profit'] += trade.pnl
            elif trade.is_loss:
                stats['losses'] += 1
                stats['gross_loss'] += trade.pnl
        
        results = []
        for symbol, stats in symbol_stats.items():
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            profit_factor = abs(stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] != 0 else 0
            
            results.append({
                'symbol': symbol,
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'gross_profit': stats['gross_profit'],
                'gross_loss': stats['gross_loss'],
                'profit_factor': profit_factor
            })
        
        return pd.DataFrame(results).sort_values('total_pnl', ascending=False)
    
    def analyze_by_time(self, freq: str = 'D') -> pd.DataFrame:
        """按时间分析交易"""
        if not self.trades:
            return pd.DataFrame()
        
        time_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0
        })
        
        for trade in self.trades:
            if freq == 'D':
                key = trade.entry_time.date()
            elif freq == 'W':
                key = trade.entry_time.isocalendar()[1]
            elif freq == 'M':
                key = (trade.entry_time.year, trade.entry_time.month)
            elif freq == 'H':
                key = trade.entry_time.hour
            else:
                key = trade.entry_time.date()
            
            stats = time_stats[key]
            stats['trades'] += 1
            stats['total_pnl'] += trade.pnl
            if trade.is_win:
                stats['wins'] += 1
        
        results = []
        for time_key, stats in time_stats.items():
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            results.append({
                'time': time_key,
                'trades': stats['trades'],
                'wins': stats['wins'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl']
            })
        
        return pd.DataFrame(results)
    
    def get_pnl_distribution(self, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """获取盈亏分布"""
        pnls = [t.pnl for t in self.trades]
        hist, edges = np.histogram(pnls, bins=bins)
        return hist, edges
    
    def get_trade_list(self) -> pd.DataFrame:
        """获取交易列表"""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'side': trade.side,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'is_win': trade.is_win,
                'holding_hours': trade.holding_period.total_seconds() / 3600 if trade.holding_period else None
            })
        
        return pd.DataFrame(data)
    
    def get_summary(self, metrics: Optional[TradeMetrics] = None) -> str:
        """获取交易分析摘要"""
        if metrics is None:
            metrics = self.analyze()
        
        summary = f"""
{'='*60}
交易分析报告
{'='*60}

交易统计:
  总交易次数:         {metrics.total_trades:>10}
  盈利交易:           {metrics.winning_trades:>10}
  亏损交易:           {metrics.losing_trades:>10}
  持平交易:           {metrics.break_even_trades:>10}

胜率分析:
  胜率:               {metrics.win_rate*100:>10.2f}%
  败率:               {metrics.loss_rate*100:>10.2f}%

盈亏分析:
  总盈亏:             ${metrics.total_pnl:>10.2f}
  总盈利:             ${metrics.gross_profit:>10.2f}
  总亏损:             ${metrics.gross_loss:>10.2f}
  净利润:             ${metrics.net_profit:>10.2f}
  盈亏比:             {metrics.profit_factor:>10.2f}
  平均盈利:           ${metrics.avg_profit:>10.2f}
  平均亏损:           ${metrics.avg_loss:>10.2f}
  平均单笔盈亏:       ${metrics.avg_trade_pnl:>10.2f}
  最大盈利:           ${metrics.max_profit:>10.2f}
  最大亏损:           ${metrics.max_loss:>10.2f}

连续统计:
  最大连续盈利:       {metrics.max_consecutive_wins:>10}
  最大连续亏损:       {metrics.max_consecutive_losses:>10}
  平均连续盈利:       {metrics.avg_consecutive_wins:>10.1f}
  平均连续亏损:       {metrics.avg_consecutive_losses:>10.1f}

持仓时间:
  平均持仓时间:       {metrics.avg_holding_period:>10.1f} 小时
  盈利持仓时间:       {metrics.avg_win_holding_period:>10.1f} 小时
  亏损持仓时间:       {metrics.avg_loss_holding_period:>10.1f} 小时

其他指标:
  期望值:             ${metrics.expectancy:>10.2f}
  平均R-multiple:     {metrics.r_multiple_avg:>10.2f}
  SQN:                {metrics.sqn:>10.2f}

{'='*60}
"""
        return summary


def generate_sample_trades(n_trades: int = 100, seed: int = 42) -> List[Trade]:
    """生成示例交易数据"""
    np.random.seed(seed)
    trades = []
    
    base_time = datetime(2024, 1, 1)
    
    for i in range(n_trades):
        # 随机生成交易
        symbol = np.random.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        entry_price = np.random.uniform(40000, 60000) if 'BTC' in symbol else np.random.uniform(2000, 4000)
        quantity = np.random.uniform(0.01, 0.5)
        
        # 70%胜率
        is_win = np.random.random() < 0.7
        
        if is_win:
            pnl = np.random.uniform(10, 500)
            exit_price = entry_price * (1 + pnl / (entry_price * quantity))
        else:
            pnl = -np.random.uniform(10, 200)
            exit_price = entry_price * (1 + pnl / (entry_price * quantity))
        
        entry_time = base_time + timedelta(hours=i * 4)
        holding_hours = np.random.randint(1, 48)
        exit_time = entry_time + timedelta(hours=holding_hours)
        
        trade = Trade(
            trade_id=f'TRD_{i:04d}',
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side='long',
            pnl=pnl,
            return_pct=pnl / (entry_price * quantity),
            fees=entry_price * quantity * 0.001
        )
        
        trades.append(trade)
    
    return trades


if __name__ == "__main__":
    # 测试交易分析
    print("交易分析模块测试")
    print("=" * 60)
    
    # 生成示例交易
    trades = generate_sample_trades(100)
    
    # 创建分析器
    analyzer = TradeAnalyzer(risk_per_trade=0.02)
    
    # 添加交易
    for trade in trades:
        analyzer.add_trade(trade)
    
    # 分析
    metrics = analyzer.analyze()
    
    # 打印摘要
    print(analyzer.get_summary(metrics))
    
    # 按品种分析
    print("\n按品种分析:")
    symbol_analysis = analyzer.analyze_by_symbol()
    print(symbol_analysis.to_string())
    
    # 按时间分析
    print("\n按小时分析:")
    time_analysis = analyzer.analyze_by_time(freq='H')
    print(time_analysis.to_string())
    
    # 盈亏分布
    print("\n盈亏分布:")
    hist, edges = analyzer.get_pnl_distribution(bins=10)
    for i in range(len(hist)):
        print(f"  ${edges[i]:.2f} - ${edges[i+1]:.2f}: {hist[i]} 笔")
