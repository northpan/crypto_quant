"""
数字货币交易执行模块 - 使用示例

本示例展示如何使用交易执行模块进行:
1. 连接交易所 (模拟/实盘)
2. 管理订单
3. 跟踪持仓
4. 执行策略 (TWAP, VWAP, 冰山订单)
5. 滑点控制
6. 交易记录
"""

import asyncio
from decimal import Decimal
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入执行模块
from execution import (
    # 交易所
    create_simulated_exchange,
    create_binance_client,
    OrderSide,
    OrderType,
    OrderStatus,
    
    # 订单管理
    OrderManager,
    OrderRequest,
    TimeInForce,
    TriggerCondition,
    
    # 持仓跟踪
    PositionTracker,
    PositionType,
    HedgeMode,
    PositionSizer,
    
    # 执行引擎
    ExecutionEngine,
    ExecutionConfig,
    ExecutionStrategy,
    ExecutionStatus,
    
    # 滑点模型
    SlippageModel,
    OrderSplitter,
    
    # 交易记录
    TradeRecorder,
    TradeRecord
)


class TradingSystem:
    """交易系统示例"""
    
    def __init__(self, use_simulation: bool = True):
        self.use_simulation = use_simulation
        self.exchange = None
        self.order_manager = None
        self.position_tracker = None
        self.execution_engine = None
        self.trade_recorder = None
        
    async def initialize(self):
        """初始化交易系统"""
        logger.info("=" * 50)
        logger.info("初始化交易系统")
        logger.info("=" * 50)
        
        # 1. 创建交易所连接
        if self.use_simulation:
            logger.info("使用模拟交易所")
            self.exchange = create_simulated_exchange({
                'USDT': Decimal('100000'),
                'BTC': Decimal('1.0'),
                'ETH': Decimal('10.0')
            })
        else:
            logger.info("使用实盘交易所")
            # 使用实际的API密钥
            self.exchange = create_binance_client(
                api_key='your_api_key',
                api_secret='your_api_secret',
                sandbox=True  # 使用测试网
            )
        
        await self.exchange.connect()
        
        # 设置模拟价格
        if self.use_simulation:
            self.exchange.set_price('BTC/USDT', Decimal('50000'))
            self.exchange.set_price('ETH/USDT', Decimal('3000'))
        
        # 2. 创建订单管理器
        self.order_manager = OrderManager(self.exchange)
        await self.order_manager.start()
        logger.info("订单管理器已启动")
        
        # 3. 创建持仓跟踪器
        self.position_tracker = PositionTracker(
            self.exchange,
            self.order_manager
        )
        await self.position_tracker.start()
        logger.info("持仓跟踪器已启动")
        
        # 4. 创建执行引擎
        self.execution_engine = ExecutionEngine(self.order_manager)
        logger.info("执行引擎已创建")
        
        # 5. 创建交易记录器
        self.trade_recorder = TradeRecorder()
        await self.trade_recorder.start()
        logger.info("交易记录器已启动")
        
        logger.info("交易系统初始化完成")
        logger.info("=" * 50)
    
    async def run_demo(self):
        """运行演示"""
        try:
            # 演示1: 基础订单操作
            await self.demo_basic_orders()
            
            # 演示2: 条件单
            await self.demo_conditional_orders()
            
            # 演示3: TWAP执行
            await self.demo_twap_execution()
            
            # 演示4: 冰山订单
            await self.demo_iceberg_execution()
            
            # 演示5: 持仓管理
            await self.demo_position_management()
            
            # 演示6: 生成报告
            await self.demo_reports()
            
        except Exception as e:
            logger.error(f"演示运行失败: {e}", exc_info=True)
    
    async def demo_basic_orders(self):
        """演示基础订单操作"""
        logger.info("\n" + "=" * 50)
        logger.info("演示1: 基础订单操作")
        logger.info("=" * 50)
        
        # 下限价单
        logger.info("\n1. 下限价买单")
        limit_order = await self.order_manager.place_limit_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            amount=Decimal('0.1'),
            price=Decimal('49500'),
            time_in_force=TimeInForce.GTC
        )
        
        if limit_order:
            logger.info(f"限价单已创建: {limit_order.id}")
            self.trade_recorder.record_order(limit_order)
        
        # 下市价单
        logger.info("\n2. 下市价买单")
        market_order = await self.order_manager.place_market_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            amount=Decimal('0.05')
        )
        
        if market_order:
            logger.info(f"市价单已创建: {market_order.id}")
            logger.info(f"成交价格: {market_order.cost / market_order.filled if market_order.filled > 0 else 0}")
            self.trade_recorder.record_order(market_order)
            
            # 记录成交
            trade = TradeRecord(
                id=f"trade_{market_order.id}",
                timestamp=market_order.timestamp,
                symbol=market_order.symbol,
                side=market_order.side,
                amount=market_order.filled,
                price=market_order.cost / market_order.filled if market_order.filled > 0 else Decimal("0"),
                cost=market_order.cost,
                fee=market_order.fee,
                order_id=market_order.id
            )
            self.trade_recorder.record_trade(trade)
        
        # 查看未成交订单
        logger.info("\n3. 查看未成交订单")
        open_orders = self.order_manager.get_open_orders()
        for order in open_orders:
            logger.info(f"  - {order.id}: {order.symbol} {order.side.value} {order.amount} @ {order.price}")
        
        # 取消订单
        if limit_order:
            logger.info(f"\n4. 取消订单: {limit_order.id}")
            success = await self.order_manager.cancel_order(
                limit_order.id,
                limit_order.symbol
            )
            logger.info(f"取消{'成功' if success else '失败'}")
    
    async def demo_conditional_orders(self):
        """演示条件单"""
        logger.info("\n" + "=" * 50)
        logger.info("演示2: 条件单")
        logger.info("=" * 50)
        
        # 创建条件单
        logger.info("\n1. 创建条件单 (价格低于48000时买入)")
        cond_order = await self.order_manager.place_conditional_order(
            trigger_condition=TriggerCondition.PRICE_BELOW,
            trigger_price=Decimal('48000'),
            order_request=OrderRequest(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal('0.1')
            )
        )
        
        logger.info(f"条件单已创建: {cond_order.id}")
        logger.info(f"触发条件: 价格低于 {cond_order.trigger_price}")
        
        # 查看条件单
        logger.info("\n2. 查看所有条件单")
        cond_orders = self.order_manager.get_conditional_orders()
        for order in cond_orders:
            logger.info(f"  - {order.id}: {order.symbol} 触发价={order.trigger_price} 状态={order.status.value}")
    
    async def demo_twap_execution(self):
        """演示TWAP执行"""
        logger.info("\n" + "=" * 50)
        logger.info("演示3: TWAP执行")
        logger.info("=" * 50)
        
        # 配置TWAP执行
        config = ExecutionConfig(
            strategy=ExecutionStrategy.TWAP,
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            total_amount=Decimal('0.5'),
            time_limit=30.0,  # 30秒执行
            max_slippage=Decimal("0.02")
        )
        
        logger.info(f"\n1. 开始TWAP执行")
        logger.info(f"   交易对: {config.symbol}")
        logger.info(f"   方向: {config.side.value}")
        logger.info(f"   总量: {config.total_amount}")
        logger.info(f"   时间限制: {config.time_limit}秒")
        
        # 执行并监控进度
        def on_progress(report):
            progress = float(report.filled_amount / report.total_amount * 100)
            logger.info(f"   TWAP进度: {progress:.1f}% | 均价: {report.avg_price}")
        
        report = await self.execution_engine.execute(config, on_progress)
        
        logger.info(f"\n2. TWAP执行完成")
        logger.info(f"   状态: {report.status.value}")
        logger.info(f"   成交量: {report.filled_amount}/{report.total_amount}")
        logger.info(f"   成交均价: {report.avg_price}")
        logger.info(f"   滑点: {report.slippage}")
        logger.info(f"   执行时间: {report.end_time - report.start_time:.2f}秒")
        
        # 记录执行
        self.trade_recorder.record_execution(report)
    
    async def demo_iceberg_execution(self):
        """演示冰山订单执行"""
        logger.info("\n" + "=" * 50)
        logger.info("演示4: 冰山订单执行")
        logger.info("=" * 50)
        
        # 配置冰山订单
        config = ExecutionConfig(
            strategy=ExecutionStrategy.ICEBERG,
            symbol='ETH/USDT',
            side=OrderSide.SELL,
            total_amount=Decimal('2.0'),
            time_limit=30.0,
            max_slippage=Decimal("0.01")
        )
        
        logger.info(f"\n1. 开始冰山订单执行")
        logger.info(f"   交易对: {config.symbol}")
        logger.info(f"   方向: {config.side.value}")
        logger.info(f"   总量: {config.total_amount}")
        
        report = await self.execution_engine.execute(config)
        
        logger.info(f"\n2. 冰山订单执行完成")
        logger.info(f"   状态: {report.status.value}")
        logger.info(f"   切片数量: {len(report.slices)}")
        logger.info(f"   成交量: {report.filled_amount}/{report.total_amount}")
        logger.info(f"   成交均价: {report.avg_price}")
        
        # 记录执行
        self.trade_recorder.record_execution(report)
    
    async def demo_position_management(self):
        """演示持仓管理"""
        logger.info("\n" + "=" * 50)
        logger.info("演示5: 持仓管理")
        logger.info("=" * 50)
        
        # 获取投资组合
        logger.info("\n1. 查看投资组合")
        portfolio = self.position_tracker.get_portfolio()
        logger.info(f"   总资产: {portfolio.total_value} USDT")
        logger.info(f"   现金: {portfolio.cash_value} USDT")
        logger.info(f"   持仓价值: {portfolio.position_value} USDT")
        logger.info(f"   未实现盈亏: {portfolio.unrealized_pnl} USDT")
        
        # 获取持仓摘要
        logger.info("\n2. 持仓摘要")
        summary = self.position_tracker.get_position_summary()
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")
        
        # 获取风险指标
        logger.info("\n3. 风险指标")
        risk = self.position_tracker.get_risk_metrics()
        for key, value in risk.items():
            logger.info(f"   {key}: {value}")
        
        # 启用对冲
        logger.info("\n4. 启用Delta对冲")
        self.position_tracker.enable_hedge(
            mode=HedgeMode.DELTA,
            target_ratio=Decimal("0.9"),
            hedge_symbols=['BTC/USDT'],
            rebalance_threshold=Decimal("0.05")
        )
        logger.info("   Delta对冲已启用")
    
    async def demo_reports(self):
        """演示报告生成"""
        logger.info("\n" + "=" * 50)
        logger.info("演示6: 生成交易报告")
        logger.info("=" * 50)
        
        # 生成交易报告
        logger.info("\n1. 交易报告")
        report = self.trade_recorder.generate_trade_report()
        
        logger.info(f"   总订单数: {report['summary']['total_orders']}")
        logger.info(f"   总成交数: {report['summary']['total_trades']}")
        logger.info(f"   已成交订单: {report['summary']['filled_orders']}")
        logger.info(f"   已取消订单: {report['summary']['cancelled_orders']}")
        
        logger.info("\n2. 性能指标")
        perf = report['performance']
        logger.info(f"   总收益: {perf['total_return']}")
        logger.info(f"   胜率: {perf['win_rate']:.2f}%")
        logger.info(f"   夏普比率: {perf['sharpe_ratio']:.2f}")
        logger.info(f"   最大回撤: {perf['max_drawdown']}")
        logger.info(f"   盈亏比: {perf['profit_factor']:.2f}")
        
        logger.info("\n3. 交易品种分布")
        for symbol, stats in report['symbol_breakdown'].items():
            logger.info(f"   {symbol}: {stats['trades']}笔交易, 成交量={stats['volume']}, 盈亏={stats['pnl']}")
    
    async def shutdown(self):
        """关闭交易系统"""
        logger.info("\n" + "=" * 50)
        logger.info("关闭交易系统")
        logger.info("=" * 50)
        
        if self.trade_recorder:
            await self.trade_recorder.stop()
            logger.info("交易记录器已停止")
        
        if self.position_tracker:
            await self.position_tracker.stop()
            logger.info("持仓跟踪器已停止")
        
        if self.order_manager:
            await self.order_manager.stop()
            logger.info("订单管理器已停止")
        
        if self.exchange:
            await self.exchange.disconnect()
            logger.info("交易所已断开")
        
        logger.info("交易系统已关闭")


async def main():
    """主函数"""
    # 创建交易系统 (使用模拟模式)
    system = TradingSystem(use_simulation=True)
    
    try:
        # 初始化
        await system.initialize()
        
        # 运行演示
        await system.run_demo()
        
    finally:
        # 关闭
        await system.shutdown()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
