"""
订单管理器模块
- 订单创建、取消、修改
- 订单状态跟踪
- 批量订单管理
- 条件单支持
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import defaultdict
import time
from abc import ABC, abstractmethod

from ..exchange.exchange_client import (
    BaseExchange, Order, OrderType, OrderSide, OrderStatus,
    PositionSide, SimulatedExchange
)

logger = logging.getLogger(__name__)


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class TriggerCondition(Enum):
    """触发条件"""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_TOUCH = "price_touch"


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = False
    hidden: bool = False
    params: Dict = field(default_factory=dict)


@dataclass
class ConditionalOrder:
    """条件订单"""
    id: str
    symbol: str
    trigger_condition: TriggerCondition
    trigger_price: Decimal
    order_request: OrderRequest
    status: OrderStatus = OrderStatus.PENDING
    created_at: int = 0
    triggered_at: Optional[int] = None
    order_id: Optional[str] = None


@dataclass
class OrderUpdate:
    """订单更新"""
    order_id: str
    symbol: str
    status: OrderStatus
    filled: Decimal
    remaining: Decimal
    cost: Decimal
    timestamp: int


class OrderCallback(ABC):
    """订单回调接口"""
    
    @abstractmethod
    def on_order_created(self, order: Order):
        """订单创建回调"""
        pass
    
    @abstractmethod
    def on_order_filled(self, order: Order):
        """订单成交回调"""
        pass
    
    @abstractmethod
    def on_order_cancelled(self, order: Order):
        """订单取消回调"""
        pass
    
    @abstractmethod
    def on_order_failed(self, order_id: str, error: str):
        """订单失败回调"""
        pass


class OrderManager:
    """订单管理器"""
    
    def __init__(self, exchange: BaseExchange):
        self.exchange = exchange
        self.orders: Dict[str, Order] = {}  # 所有订单
        self.open_orders: Dict[str, Order] = {}  # 未成交订单
        self.conditional_orders: Dict[str, ConditionalOrder] = {}  # 条件单
        self.order_callbacks: List[OrderCallback] = []
        self.update_callbacks: List[Callable[[OrderUpdate], None]] = []
        
        # 批量订单管理
        self.batch_orders: Dict[str, List[str]] = defaultdict(list)
        
        # 订单历史
        self.order_history: List[Order] = []
        self.max_history_size = 10000
        
        # 同步任务
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._sync_interval = 1.0  # 同步间隔(秒)
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 0.5
        
    def add_callback(self, callback: OrderCallback):
        """添加订单回调"""
        self.order_callbacks.append(callback)
    
    def remove_callback(self, callback: OrderCallback):
        """移除订单回调"""
        if callback in self.order_callbacks:
            self.order_callbacks.remove(callback)
    
    def add_update_callback(self, callback: Callable[[OrderUpdate], None]):
        """添加更新回调"""
        self.update_callbacks.append(callback)
    
    async def start(self):
        """启动订单管理器"""
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_orders())
        logger.info("订单管理器已启动")
    
    async def stop(self):
        """停止订单管理器"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("订单管理器已停止")
    
    async def _sync_orders(self):
        """同步订单状态"""
        while self._running:
            try:
                await self._update_open_orders()
                await self._check_conditional_orders()
                await asyncio.sleep(self._sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"订单同步错误: {e}")
                await asyncio.sleep(self._sync_interval)
    
    async def _update_open_orders(self):
        """更新未成交订单状态"""
        if not self.open_orders:
            return
        
        try:
            open_orders = await self.exchange.get_open_orders()
            current_ids = {o.id for o in open_orders}
            
            # 检查已成交或取消的订单
            for order_id in list(self.open_orders.keys()):
                if order_id not in current_ids:
                    # 订单已成交或取消，获取最终状态
                    order = await self._get_order_with_retry(order_id, self.open_orders[order_id].symbol)
                    if order:
                        self._update_order(order)
            
            # 更新现有订单
            for order in open_orders:
                if order.id in self.open_orders:
                    self._update_order(order)
                    
        except Exception as e:
            logger.error(f"更新未成交订单失败: {e}")
    
    async def _get_order_with_retry(self, order_id: str, symbol: str) -> Optional[Order]:
        """带重试的获取订单"""
        for attempt in range(self.max_retries):
            try:
                return await self.exchange.get_order(order_id, symbol)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"获取订单失败(重试{self.max_retries}次): {e}")
        return None
    
    def _update_order(self, order: Order):
        """更新订单状态"""
        old_order = self.orders.get(order.id)
        
        # 更新订单
        self.orders[order.id] = order
        
        # 更新未成交订单列表
        if order.status == OrderStatus.OPEN:
            self.open_orders[order.id] = order
        elif order.id in self.open_orders:
            del self.open_orders[order.id]
        
        # 触发回调
        if old_order:
            if old_order.filled != order.filled:
                # 部分成交
                self._notify_fill(order)
            
            if old_order.status != order.status:
                # 状态变化
                if order.status == OrderStatus.CLOSED:
                    self._notify_fill(order)
                elif order.status == OrderStatus.CANCELED:
                    self._notify_cancel(order)
        
        # 添加到历史
        if order.status in [OrderStatus.CLOSED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
            self._add_to_history(order)
        
        # 发送更新通知
        update = OrderUpdate(
            order_id=order.id,
            symbol=order.symbol,
            status=order.status,
            filled=order.filled,
            remaining=order.remaining,
            cost=order.cost,
            timestamp=order.last_update
        )
        for callback in self.update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"更新回调执行失败: {e}")
    
    def _notify_fill(self, order: Order):
        """通知订单成交"""
        for callback in self.order_callbacks:
            try:
                callback.on_order_filled(order)
            except Exception as e:
                logger.error(f"成交回调执行失败: {e}")
    
    def _notify_cancel(self, order: Order):
        """通知订单取消"""
        for callback in self.order_callbacks:
            try:
                callback.on_order_cancelled(order)
            except Exception as e:
                logger.error(f"取消回调执行失败: {e}")
    
    def _notify_failed(self, order_id: str, error: str):
        """通知订单失败"""
        for callback in self.order_callbacks:
            try:
                callback.on_order_failed(order_id, error)
            except Exception as e:
                logger.error(f"失败回调执行失败: {e}")
    
    def _add_to_history(self, order: Order):
        """添加到历史记录"""
        self.order_history.append(order)
        if len(self.order_history) > self.max_history_size:
            self.order_history = self.order_history[-self.max_history_size:]
    
    async def place_order(
        self,
        request: OrderRequest,
        batch_id: Optional[str] = None
    ) -> Optional[Order]:
        """下单"""
        try:
            # 构建参数
            params = request.params.copy()
            
            if request.time_in_force != TimeInForce.GTC:
                params['timeInForce'] = request.time_in_force.value
            
            if request.reduce_only:
                params['reduceOnly'] = True
            
            if request.post_only:
                params['postOnly'] = True
            
            if request.hidden:
                params['hidden'] = True
            
            if request.client_order_id:
                params['clientOrderId'] = request.client_order_id
            
            if request.stop_price:
                params['stopPrice'] = float(request.stop_price)
            
            # 创建订单
            order = await self.exchange.create_order(
                symbol=request.symbol,
                order_type=request.order_type,
                side=request.side,
                amount=request.amount,
                price=request.price,
                params=params
            )
            
            # 记录订单
            self.orders[order.id] = order
            if order.status == OrderStatus.OPEN:
                self.open_orders[order.id] = order
            
            # 批量订单
            if batch_id:
                self.batch_orders[batch_id].append(order.id)
            
            # 通知回调
            for callback in self.order_callbacks:
                try:
                    callback.on_order_created(order)
                except Exception as e:
                    logger.error(f"创建回调执行失败: {e}")
            
            logger.info(f"下单成功: {order.id} - {request.symbol} {request.side.value} {request.amount}")
            return order
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            self._notify_failed(request.client_order_id or "unknown", str(e))
            return None
    
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
        **kwargs
    ) -> Optional[Order]:
        """下限价单"""
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
            post_only=post_only,
            **kwargs
        )
        return await self.place_order(request)
    
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        **kwargs
    ) -> Optional[Order]:
        """下市价单"""
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            **kwargs
        )
        return await self.place_order(request)
    
    async def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        stop_price: Decimal,
        limit_price: Optional[Decimal] = None,
        **kwargs
    ) -> Optional[Order]:
        """下止损单"""
        order_type = OrderType.STOP_LOSS_LIMIT if limit_price else OrderType.STOP_LOSS
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=limit_price,
            stop_price=stop_price,
            **kwargs
        )
        return await self.place_order(request)
    
    async def place_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: Decimal,
        stop_price: Decimal,
        limit_price: Optional[Decimal] = None,
        **kwargs
    ) -> Optional[Order]:
        """下止盈单"""
        order_type = OrderType.TAKE_PROFIT_LIMIT if limit_price else OrderType.TAKE_PROFIT
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=limit_price,
            stop_price=stop_price,
            **kwargs
        )
        return await self.place_order(request)
    
    async def place_conditional_order(
        self,
        trigger_condition: TriggerCondition,
        trigger_price: Decimal,
        order_request: OrderRequest
    ) -> ConditionalOrder:
        """下条件单"""
        cond_order_id = f"cond_{int(time.time() * 1000)}"
        
        cond_order = ConditionalOrder(
            id=cond_order_id,
            symbol=order_request.symbol,
            trigger_condition=trigger_condition,
            trigger_price=trigger_price,
            order_request=order_request,
            created_at=int(time.time() * 1000)
        )
        
        self.conditional_orders[cond_order_id] = cond_order
        logger.info(f"条件单创建: {cond_order_id} - 触发价 {trigger_price}")
        return cond_order
    
    async def _check_conditional_orders(self):
        """检查条件单触发"""
        if not self.conditional_orders:
            return
        
        for cond_id, cond_order in list(self.conditional_orders.items()):
            if cond_order.status != OrderStatus.PENDING:
                continue
            
            try:
                # 获取当前价格
                ticker = await self.exchange.get_ticker(cond_order.symbol)
                current_price = Decimal(str(ticker.get('last', 0)))
                
                triggered = False
                
                if cond_order.trigger_condition == TriggerCondition.PRICE_ABOVE:
                    triggered = current_price >= cond_order.trigger_price
                elif cond_order.trigger_condition == TriggerCondition.PRICE_BELOW:
                    triggered = current_price <= cond_order.trigger_price
                elif cond_order.trigger_condition == TriggerCondition.PRICE_TOUCH:
                    triggered = abs(current_price - cond_order.trigger_price) / cond_order.trigger_price < Decimal('0.001')
                
                if triggered:
                    # 触发条件单
                    logger.info(f"条件单触发: {cond_id} @ {current_price}")
                    cond_order.status = OrderStatus.OPEN
                    cond_order.triggered_at = int(time.time() * 1000)
                    
                    # 提交实际订单
                    order = await self.place_order(cond_order.order_request)
                    if order:
                        cond_order.order_id = order.id
                    else:
                        cond_order.status = OrderStatus.REJECTED
                        
            except Exception as e:
                logger.error(f"检查条件单失败 {cond_id}: {e}")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            success = await self.exchange.cancel_order(order_id, symbol)
            if success and order_id in self.open_orders:
                order = self.open_orders[order_id]
                order.status = OrderStatus.CANCELED
                self._update_order(order)
            return success
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, bool]:
        """取消所有订单"""
        results = {}
        orders_to_cancel = [
            order for order in self.open_orders.values()
            if symbol is None or order.symbol == symbol
        ]
        
        for order in orders_to_cancel:
            results[order.id] = await self.cancel_order(order.id, order.symbol)
        
        logger.info(f"批量取消订单: {len(orders_to_cancel)} 个订单")
        return results
    
    async def cancel_batch(self, batch_id: str) -> Dict[str, bool]:
        """取消批量订单"""
        if batch_id not in self.batch_orders:
            return {}
        
        results = {}
        for order_id in self.batch_orders[batch_id]:
            if order_id in self.orders:
                order = self.orders[order_id]
                results[order_id] = await self.cancel_order(order_id, order.symbol)
        
        return results
    
    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        new_price: Optional[Decimal] = None,
        new_amount: Optional[Decimal] = None
    ) -> Optional[Order]:
        """修改订单"""
        try:
            # 先取消原订单
            if not await self.cancel_order(order_id, symbol):
                return None
            
            # 获取原订单
            old_order = self.orders.get(order_id)
            if not old_order:
                return None
            
            # 创建新订单
            request = OrderRequest(
                symbol=symbol,
                side=old_order.side,
                order_type=old_order.order_type,
                amount=new_amount or old_order.amount,
                price=new_price or old_order.price
            )
            
            return await self.place_order(request)
            
        except Exception as e:
            logger.error(f"修改订单失败: {e}")
            return None
    
    async def place_batch_orders(
        self,
        requests: List[OrderRequest],
        batch_id: Optional[str] = None
    ) -> List[Optional[Order]]:
        """批量下单"""
        if batch_id is None:
            batch_id = f"batch_{int(time.time() * 1000)}"
        
        # 并发下单
        tasks = [self.place_order(req, batch_id) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        orders = []
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量订单失败: {result}")
                orders.append(None)
            else:
                orders.append(result)
                if result:
                    success_count += 1
        
        logger.info(f"批量下单完成: {success_count}/{len(requests)} 成功")
        return orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        orders = list(self.open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """获取指定交易对的订单"""
        return [o for o in self.orders.values() if o.symbol == symbol]
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """获取订单历史"""
        history = self.order_history
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        return history[-limit:]
    
    def get_conditional_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None
    ) -> List[ConditionalOrder]:
        """获取条件单"""
        orders = list(self.conditional_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
    
    async def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 60.0,
        poll_interval: float = 0.5
    ) -> Optional[Order]:
        """等待订单成交"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            order = self.orders.get(order_id)
            if order and order.status == OrderStatus.CLOSED:
                return order
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"等待订单成交超时: {order_id}")
        return self.orders.get(order_id)
    
    async def wait_for_batch_fill(
        self,
        batch_id: str,
        timeout: float = 60.0
    ) -> Dict[str, Optional[Order]]:
        """等待批量订单成交"""
        if batch_id not in self.batch_orders:
            return {}
        
        order_ids = self.batch_orders[batch_id]
        results = {}
        
        for order_id in order_ids:
            results[order_id] = await self.wait_for_fill(
                order_id,
                timeout=timeout / len(order_ids)
            )
        
        return results


# 便捷函数
def create_order_request(
    symbol: str,
    side: str,
    order_type: str,
    amount: float,
    price: Optional[float] = None,
    **kwargs
) -> OrderRequest:
    """创建订单请求"""
    return OrderRequest(
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType(order_type),
        amount=Decimal(str(amount)),
        price=Decimal(str(price)) if price else None,
        **kwargs
    )


# 示例用法
if __name__ == "__main__":
    async def test():
        # 创建模拟交易所
        sim = SimulatedExchange({'USDT': Decimal('10000')})
        await sim.connect()
        sim.set_price('BTC/USDT', Decimal('50000'))
        
        # 创建订单管理器
        manager = OrderManager(sim)
        await manager.start()
        
        # 下限价单
        order = await manager.place_limit_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            amount=Decimal('0.1'),
            price=Decimal('49000')
        )
        
        print(f"订单: {order}")
        print(f"未成交订单: {manager.get_open_orders()}")
        
        # 下条件单
        cond_order = await manager.place_conditional_order(
            trigger_condition=TriggerCondition.PRICE_BELOW,
            trigger_price=Decimal('48000'),
            order_request=OrderRequest(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=Decimal('0.05')
            )
        )
        
        print(f"条件单: {cond_order}")
        
        await manager.stop()
        await sim.disconnect()
    
    asyncio.run(test())
