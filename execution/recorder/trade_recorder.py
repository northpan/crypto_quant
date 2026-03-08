"""
交易记录器模块
- 交易日志记录
- 订单历史管理
- 执行分析
- 性能报告
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import csv
import os
from pathlib import Path

from ..exchange.exchange_client import Order, OrderSide, OrderStatus, OrderType
from ..order.order_manager import OrderUpdate
from ..position.position_tracker import Portfolio, PositionUpdate
from ..strategy.execution_engine import ExecutionReport, ExecutionStatus

logger = logging.getLogger(__name__)


class RecordType(Enum):
    """记录类型"""
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
    ERROR = "error"


@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    timestamp: int
    symbol: str
    side: OrderSide
    amount: Decimal
    price: Decimal
    cost: Decimal
    fee: Decimal
    order_id: str
    execution_id: Optional[str] = None
    strategy: Optional[str] = None
    pnl: Optional[Decimal] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DailySummary:
    """日度汇总"""
    date: str
    total_trades: int
    buy_volume: Decimal
    sell_volume: Decimal
    total_volume: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    fees: Decimal
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: float
    max_drawdown: Decimal
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_trade: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal


class TradeRecorder:
    """交易记录器"""
    
    def __init__(
        self,
        data_dir: str = "/mnt/okcomputer/output/crypto_quant/data/trades",
        auto_save: bool = True,
        save_interval: int = 60
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # 内存存储
        self.orders: Dict[str, Order] = {}
        self.trades: List[TradeRecord] = []
        self.positions: Dict[str, List[PositionUpdate]] = defaultdict(list)
        self.portfolios: List[Portfolio] = []
        self.executions: Dict[str, ExecutionReport] = {}
        self.errors: List[Dict] = []
        
        # 统计
        self.daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            'trades': [],
            'pnl': Decimal("0"),
            'fees': Decimal("0")
        })
        
        # 回调
        self.record_callbacks: List[Callable[[RecordType, Any], None]] = []
        
        # 保存任务
        self._save_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 加载历史数据
        self._load_history()
    
    def add_record_callback(self, callback: Callable[[RecordType, Any], None]):
        """添加记录回调"""
        self.record_callbacks.append(callback)
    
    async def start(self):
        """启动记录器"""
        self._running = True
        if self.auto_save:
            self._save_task = asyncio.create_task(self._auto_save())
        logger.info("交易记录器已启动")
    
    async def stop(self):
        """停止记录器"""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        
        # 最后保存
        await self.save_all()
        logger.info("交易记录器已停止")
    
    async def _auto_save(self):
        """自动保存"""
        while self._running:
            try:
                await asyncio.sleep(self.save_interval)
                await self.save_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动保存失败: {e}")
    
    def record_order(self, order: Order):
        """记录订单"""
        self.orders[order.id] = order
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.ORDER, order)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def record_trade(self, trade: TradeRecord):
        """记录成交"""
        self.trades.append(trade)
        
        # 更新日度统计
        date = datetime.fromtimestamp(trade.timestamp / 1000).strftime('%Y-%m-%d')
        self.daily_stats[date]['trades'].append(trade)
        self.daily_stats[date]['fees'] += trade.fee
        if trade.pnl:
            self.daily_stats[date]['pnl'] += trade.pnl
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.TRADE, trade)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def record_position(self, position: PositionUpdate):
        """记录持仓"""
        self.positions[position.symbol].append(position)
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.POSITION, position)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def record_portfolio(self, portfolio: Portfolio):
        """记录投资组合"""
        self.portfolios.append(portfolio)
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.PORTFOLIO, portfolio)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def record_execution(self, execution: ExecutionReport):
        """记录执行"""
        self.executions[execution.execution_id] = execution
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.EXECUTION, execution)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def record_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """记录错误"""
        error_record = {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'type': error_type,
            'message': message,
            'details': details or {}
        }
        self.errors.append(error_record)
        
        # 触发回调
        for callback in self.record_callbacks:
            try:
                callback(RecordType.ERROR, error_record)
            except Exception as e:
                logger.error(f"记录回调执行失败: {e}")
    
    def on_order_update(self, update: OrderUpdate):
        """订单更新回调"""
        if update.order_id in self.orders:
            order = self.orders[update.order_id]
            order.status = update.status
            order.filled = update.filled
            order.remaining = update.remaining
            order.cost = update.cost
            order.last_update = update.timestamp
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        status: Optional[OrderStatus] = None
    ) -> List[Order]:
        """获取订单历史"""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if start_time:
            orders = [o for o in orders if o.timestamp >= start_time]
        
        if end_time:
            orders = [o for o in orders if o.timestamp <= end_time]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return sorted(orders, key=lambda x: x.timestamp, reverse=True)
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[TradeRecord]:
        """获取成交历史"""
        trades = self.trades.copy()
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_time:
            trades = [t for t in trades if t.timestamp >= start_time]
        
        if end_time:
            trades = [t for t in trades if t.timestamp <= end_time]
        
        return sorted(trades, key=lambda x: x.timestamp, reverse=True)
    
    def get_daily_summary(self, date: Optional[str] = None) -> Optional[DailySummary]:
        """获取日度汇总"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if date not in self.daily_stats:
            return None
        
        stats = self.daily_stats[date]
        trades = stats['trades']
        
        if not trades:
            return DailySummary(
                date=date,
                total_trades=0,
                buy_volume=Decimal("0"),
                sell_volume=Decimal("0"),
                total_volume=Decimal("0"),
                gross_pnl=Decimal("0"),
                net_pnl=Decimal("0"),
                fees=Decimal("0"),
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0
            )
        
        buy_volume = sum(t.amount for t in trades if t.side == OrderSide.BUY)
        sell_volume = sum(t.amount for t in trades if t.side == OrderSide.SELL)
        
        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        gross_pnl = sum(pnls) if pnls else Decimal("0")
        fees = stats['fees']
        net_pnl = gross_pnl - fees
        
        return DailySummary(
            date=date,
            total_trades=len(trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=buy_volume + sell_volume,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees=fees,
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=len(wins) / len(pnls) * 100 if pnls else 0.0,
            avg_win=sum(wins) / len(wins) if wins else Decimal("0"),
            avg_loss=sum(losses) / len(losses) if losses else Decimal("0"),
            max_drawdown=Decimal("0"),  # 需要更多数据计算
            sharpe_ratio=0.0  # 需要更多数据计算
        )
    
    def calculate_performance_metrics(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> PerformanceMetrics:
        """计算性能指标"""
        trades = self.get_trade_history(start_time=start_time, end_time=end_time)
        
        if not trades:
            return PerformanceMetrics(
                total_return=Decimal("0"),
                annualized_return=Decimal("0"),
                volatility=Decimal("0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0"),
                max_drawdown_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade=Decimal("0"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0")
            )
        
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        if not pnls:
            return PerformanceMetrics(
                total_return=Decimal("0"),
                annualized_return=Decimal("0"),
                volatility=Decimal("0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0"),
                max_drawdown_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade=Decimal("0"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0")
            )
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else Decimal("0")
        gross_loss = sum(losses) if losses else Decimal("0")
        
        # 计算最大回撤
        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")
        
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        # 计算波动率
        if len(pnls) > 1:
            mean_pnl = total_pnl / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            volatility = variance.sqrt()
        else:
            volatility = Decimal("0")
        
        # 计算夏普比率 (假设无风险利率为0)
        if volatility > 0:
            sharpe = float((total_pnl / len(pnls)) / volatility) * (252 ** 0.5)
        else:
            sharpe = 0.0
        
        return PerformanceMetrics(
            total_return=total_pnl,
            annualized_return=total_pnl * Decimal("252.0") / len(pnls) if pnls else Decimal("0"),
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=float(max_dd / peak * 100) if peak > 0 else 0.0,
            win_rate=len(wins) / len(pnls) * 100 if pnls else 0.0,
            profit_factor=abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
            avg_trade=total_pnl / len(pnls),
            avg_win=sum(wins) / len(wins) if wins else Decimal("0"),
            avg_loss=sum(losses) / len(losses) if losses else Decimal("0"),
            largest_win=max(wins) if wins else Decimal("0"),
            largest_loss=min(losses) if losses else Decimal("0")
        )
    
    def generate_trade_report(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict:
        """生成交易报告"""
        trades = self.get_trade_history(start_time=start_time, end_time=end_time)
        orders = self.get_order_history(start_time=start_time, end_time=end_time)
        
        metrics = self.calculate_performance_metrics(start_time, end_time)
        
        # 按symbol统计
        symbol_stats = defaultdict(lambda: {
            'trades': 0,
            'volume': Decimal("0"),
            'pnl': Decimal("0")
        })
        
        for trade in trades:
            symbol_stats[trade.symbol]['trades'] += 1
            symbol_stats[trade.symbol]['volume'] += trade.amount
            if trade.pnl:
                symbol_stats[trade.symbol]['pnl'] += trade.pnl
        
        return {
            'period': {
                'start': start_time,
                'end': end_time
            },
            'summary': {
                'total_orders': len(orders),
                'total_trades': len(trades),
                'filled_orders': len([o for o in orders if o.status == OrderStatus.CLOSED]),
                'cancelled_orders': len([o for o in orders if o.status == OrderStatus.CANCELED])
            },
            'performance': {
                'total_return': float(metrics.total_return),
                'annualized_return': float(metrics.annualized_return),
                'volatility': float(metrics.volatility),
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': float(metrics.max_drawdown),
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_trade': float(metrics.avg_trade),
                'avg_win': float(metrics.avg_win),
                'avg_loss': float(metrics.avg_loss),
                'largest_win': float(metrics.largest_win),
                'largest_loss': float(metrics.largest_loss)
            },
            'symbol_breakdown': {
                symbol: {
                    'trades': stats['trades'],
                    'volume': float(stats['volume']),
                    'pnl': float(stats['pnl'])
                }
                for symbol, stats in symbol_stats.items()
            }
        }
    
    async def save_all(self):
        """保存所有数据"""
        try:
            await self._save_orders()
            await self._save_trades()
            await self._save_portfolios()
            await self._save_executions()
            logger.debug("所有数据已保存")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    async def _save_orders(self):
        """保存订单"""
        if not self.orders:
            return
        
        filepath = self.data_dir / "orders.json"
        orders_data = {
            order_id: {
                'id': o.id,
                'symbol': o.symbol,
                'side': o.side.value,
                'order_type': o.order_type.value,
                'amount': str(o.amount),
                'price': str(o.price) if o.price else None,
                'stop_price': str(o.stop_price) if o.stop_price else None,
                'status': o.status.value,
                'filled': str(o.filled),
                'remaining': str(o.remaining),
                'cost': str(o.cost),
                'fee': str(o.fee),
                'timestamp': o.timestamp,
                'last_update': o.last_update,
                'client_order_id': o.client_order_id
            }
            for order_id, o in self.orders.items()
        }
        
        await asyncio.to_thread(self._write_json, filepath, orders_data)
    
    async def _save_trades(self):
        """保存成交"""
        if not self.trades:
            return
        
        filepath = self.data_dir / "trades.csv"
        
        def write_csv():
            with open(filepath, 'w', newline='') as f:
                if self.trades:
                    writer = csv.DictWriter(f, fieldnames=vars(self.trades[0]).keys())
                    writer.writeheader()
                    for trade in self.trades:
                        row = {k: str(v) if isinstance(v, (Decimal, OrderSide)) else v 
                               for k, v in vars(trade).items()}
                        writer.writerow(row)
        
        await asyncio.to_thread(write_csv)
    
    async def _save_portfolios(self):
        """保存投资组合"""
        if not self.portfolios:
            return
        
        filepath = self.data_dir / "portfolios.json"
        portfolios_data = [
            {
                'total_value': str(p.total_value),
                'cash_value': str(p.cash_value),
                'position_value': str(p.position_value),
                'unrealized_pnl': str(p.unrealized_pnl),
                'realized_pnl': str(p.realized_pnl),
                'margin_used': str(p.margin_used),
                'margin_available': str(p.margin_available),
                'leverage': str(p.leverage),
                'timestamp': p.timestamp
            }
            for p in self.portfolios
        ]
        
        await asyncio.to_thread(self._write_json, filepath, portfolios_data)
    
    async def _save_executions(self):
        """保存执行记录"""
        if not self.executions:
            return
        
        filepath = self.data_dir / "executions.json"
        executions_data = {
            exec_id: {
                'execution_id': e.execution_id,
                'strategy': e.strategy.value,
                'symbol': e.symbol,
                'side': e.side.value,
                'status': e.status.value,
                'total_amount': str(e.total_amount),
                'filled_amount': str(e.filled_amount),
                'avg_price': str(e.avg_price),
                'total_cost': str(e.total_cost),
                'slippage': str(e.slippage),
                'start_time': e.start_time,
                'end_time': e.end_time,
                'orders': e.orders
            }
            for exec_id, e in self.executions.items()
        }
        
        await asyncio.to_thread(self._write_json, filepath, executions_data)
    
    def _write_json(self, filepath: Path, data: Any):
        """写入JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_history(self):
        """加载历史数据"""
        try:
            # 加载订单
            orders_file = self.data_dir / "orders.json"
            if orders_file.exists():
                with open(orders_file, 'r') as f:
                    orders_data = json.load(f)
                    for order_id, data in orders_data.items():
                        self.orders[order_id] = Order(
                            id=data['id'],
                            symbol=data['symbol'],
                            side=OrderSide(data['side']),
                            order_type=OrderType(data['order_type']),
                            amount=Decimal(data['amount']),
                            price=Decimal(data['price']) if data['price'] else None,
                            stop_price=Decimal(data['stop_price']) if data['stop_price'] else None,
                            status=OrderStatus(data['status']),
                            filled=Decimal(data['filled']),
                            remaining=Decimal(data['remaining']),
                            cost=Decimal(data['cost']),
                            fee=Decimal(data['fee']),
                            timestamp=data['timestamp'],
                            last_update=data['last_update'],
                            client_order_id=data.get('client_order_id')
                        )
            
            logger.info(f"已加载 {len(self.orders)} 个历史订单")
            
        except Exception as e:
            logger.warning(f"加载历史数据失败: {e}")


# 便捷函数
def create_trade_recorder(
    data_dir: str = "/mnt/okcomputer/output/crypto_quant/data/trades"
) -> TradeRecorder:
    """创建交易记录器"""
    return TradeRecorder(data_dir)


# 示例用法
if __name__ == "__main__":
    async def test():
        recorder = create_trade_recorder()
        await recorder.start()
        
        # 记录一些测试数据
        from decimal import Decimal
        
        test_trade = TradeRecord(
            id="test_1",
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            cost=Decimal("5000"),
            fee=Decimal("5"),
            order_id="order_1",
            pnl=Decimal("100")
        )
        
        recorder.record_trade(test_trade)
        
        # 生成报告
        report = recorder.generate_trade_report()
        print(json.dumps(report, indent=2))
        
        await recorder.stop()
    
    asyncio.run(test())
