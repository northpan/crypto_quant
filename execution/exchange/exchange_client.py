"""
交易所客户端模块 - 基于CCXT的统一接口
支持Binance、OKX等主流交易所
支持模拟和实盘两种模式
"""

import ccxt
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
from decimal import Decimal
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    LIMIT = "limit"
    MARKET = "market"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """订单数据结构"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled: Decimal = Decimal("0")
    remaining: Decimal = Decimal("0")
    cost: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    trades: List[Dict] = field(default_factory=list)
    timestamp: int = 0
    last_update: int = 0
    client_order_id: Optional[str] = None
    params: Dict = field(default_factory=dict)


@dataclass
class Balance:
    """账户余额"""
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    amount: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Optional[Decimal] = None
    margin: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


class ExchangeConfig:
    """交易所配置"""
    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        password: Optional[str] = None,
        sandbox: bool = True,
        enable_rate_limit: bool = True,
        options: Optional[Dict] = None
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.sandbox = sandbox
        self.enable_rate_limit = enable_rate_limit
        self.options = options or {}


class BaseExchange(ABC):
    """交易所基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接交易所"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """获取账户余额"""
        pass
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        params: Optional[Dict] = None
    ) -> Order:
        """创建订单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """获取订单信息"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓信息"""
        pass


class CCXTExchange(BaseExchange):
    """基于CCXT的交易所实现"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self._connected = False
        self._order_callbacks: List[Callable[[Order], None]] = []
        self._trade_callbacks: List[Callable[[Dict], None]] = []
        
    async def connect(self) -> bool:
        """连接交易所"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange_id)
            
            config = {
                'enableRateLimit': self.config.enable_rate_limit,
                'options': self.config.options
            }
            
            if self.config.api_key:
                config['apiKey'] = self.config.api_key
            if self.config.api_secret:
                config['secret'] = self.config.api_secret
            if self.config.password:
                config['password'] = self.config.password
                
            self.exchange = exchange_class(config)
            
            # 启用模拟模式
            if self.config.sandbox and hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(True)
                
            # 加载市场数据
            await asyncio.to_thread(self.exchange.load_markets)
            
            self._connected = True
            logger.info(f"成功连接到交易所: {self.config.exchange_id} (Sandbox: {self.config.sandbox})")
            return True
            
        except Exception as e:
            logger.error(f"连接交易所失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        self._connected = False
        if self.exchange:
            await asyncio.to_thread(self.exchange.close)
            logger.info(f"已断开与交易所的连接: {self.config.exchange_id}")
    
    def _convert_order(self, ccxt_order: Dict) -> Order:
        """将CCXT订单转换为内部订单格式"""
        return Order(
            id=ccxt_order.get('id', ''),
            symbol=ccxt_order.get('symbol', ''),
            side=OrderSide(ccxt_order.get('side', 'buy')),
            order_type=OrderType(ccxt_order.get('type', 'limit')),
            amount=Decimal(str(ccxt_order.get('amount', 0) or 0)),
            price=Decimal(str(ccxt_order.get('price', 0) or 0)) if ccxt_order.get('price') else None,
            stop_price=Decimal(str(ccxt_order.get('stopPrice', 0) or 0)) if ccxt_order.get('stopPrice') else None,
            status=self._convert_status(ccxt_order.get('status', 'open')),
            filled=Decimal(str(ccxt_order.get('filled', 0) or 0)),
            remaining=Decimal(str(ccxt_order.get('remaining', 0) or 0)),
            cost=Decimal(str(ccxt_order.get('cost', 0) or 0)),
            fee=Decimal(str(ccxt_order.get('fee', {}).get('cost', 0) or 0)),
            trades=ccxt_order.get('trades', []),
            timestamp=ccxt_order.get('timestamp', int(time.time() * 1000)),
            last_update=ccxt_order.get('lastUpdateTimestamp', int(time.time() * 1000)),
            client_order_id=ccxt_order.get('clientOrderId'),
            params=ccxt_order.get('info', {})
        )
    
    def _convert_status(self, status: str) -> OrderStatus:
        """转换订单状态"""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.CLOSED,
            'canceled': OrderStatus.CANCELED,
            'cancelled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
            'pending': OrderStatus.PENDING
        }
        return status_map.get(status, OrderStatus.PENDING)
    
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """获取账户余额"""
        try:
            response = await asyncio.to_thread(self.exchange.fetch_balance)
            balances = {}
            
            for curr, data in response.items():
                if isinstance(data, dict) and 'free' in data:
                    if currency and curr != currency:
                        continue
                    balances[curr] = Balance(
                        currency=curr,
                        free=Decimal(str(data.get('free', 0) or 0)),
                        used=Decimal(str(data.get('used', 0) or 0)),
                        total=Decimal(str(data.get('total', 0) or 0))
                    )
            
            return balances
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            raise
    
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        params: Optional[Dict] = None
    ) -> Order:
        """创建订单"""
        try:
            order_params = params or {}
            
            ccxt_order = await asyncio.to_thread(
                self.exchange.create_order,
                symbol,
                order_type.value,
                side.value,
                float(amount),
                float(price) if price else None,
                order_params
            )
            
            order = self._convert_order(ccxt_order)
            
            # 触发回调
            for callback in self._order_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"订单回调执行失败: {e}")
            
            logger.info(f"创建订单成功: {order.id} - {symbol} {side.value} {amount} @ {price}")
            return order
            
        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            await asyncio.to_thread(self.exchange.cancel_order, order_id, symbol)
            logger.info(f"取消订单成功: {order_id}")
            return True
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """获取订单信息"""
        try:
            ccxt_order = await asyncio.to_thread(
                self.exchange.fetch_order, order_id, symbol
            )
            return self._convert_order(ccxt_order)
        except Exception as e:
            logger.error(f"获取订单信息失败: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        try:
            ccxt_orders = await asyncio.to_thread(
                self.exchange.fetch_open_orders, symbol
            )
            return [self._convert_order(o) for o in ccxt_orders]
        except Exception as e:
            logger.error(f"获取未成交订单失败: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓信息"""
        try:
            if not hasattr(self.exchange, 'fetch_positions'):
                return []
                
            ccxt_positions = await asyncio.to_thread(
                self.exchange.fetch_positions, [symbol] if symbol else None
            )
            
            positions = []
            for pos in ccxt_positions:
                if pos.get('contracts', 0) == 0:
                    continue
                    
                positions.append(Position(
                    symbol=pos.get('symbol', ''),
                    side=PositionSide.LONG if pos.get('side') == 'long' else PositionSide.SHORT,
                    amount=Decimal(str(abs(pos.get('contracts', 0) or 0))),
                    entry_price=Decimal(str(pos.get('entryPrice', 0) or 0)),
                    mark_price=Decimal(str(pos.get('markPrice', 0) or 0)),
                    liquidation_price=Decimal(str(pos.get('liquidationPrice', 0) or 0)) if pos.get('liquidationPrice') else None,
                    margin=Decimal(str(pos.get('margin', 0) or 0)),
                    leverage=Decimal(str(pos.get('leverage', 1) or 1)),
                    unrealized_pnl=Decimal(str(pos.get('unrealizedPnl', 0) or 0)),
                    realized_pnl=Decimal(str(pos.get('realizedPnl', 0) or 0))
                ))
            
            return positions
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        try:
            return await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """获取订单簿"""
        try:
            return await asyncio.to_thread(self.exchange.fetch_order_book, symbol, limit)
        except Exception as e:
            logger.error(f"获取订单簿失败: {e}")
            raise
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取最近成交"""
        try:
            return await asyncio.to_thread(self.exchange.fetch_trades, symbol, None, limit)
        except Exception as e:
            logger.error(f"获取成交记录失败: {e}")
            raise
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """添加订单回调"""
        self._order_callbacks.append(callback)
    
    def remove_order_callback(self, callback: Callable[[Order], None]):
        """移除订单回调"""
        if callback in self._order_callbacks:
            self._order_callbacks.remove(callback)


class SimulatedExchange(BaseExchange):
    """模拟交易所 - 用于回测和模拟交易"""
    
    def __init__(self, initial_balances: Optional[Dict[str, Decimal]] = None):
        self.balances: Dict[str, Balance] = {}
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_counter = 0
        self.price_data: Dict[str, Decimal] = {}
        
        # 初始化余额
        if initial_balances:
            for currency, amount in initial_balances.items():
                self.balances[currency] = Balance(
                    currency=currency,
                    free=amount,
                    used=Decimal("0"),
                    total=amount
                )
    
    def set_price(self, symbol: str, price: Decimal):
        """设置模拟价格"""
        self.price_data[symbol] = price
    
    async def connect(self) -> bool:
        """连接模拟交易所"""
        logger.info("模拟交易所已连接")
        return True
    
    async def disconnect(self):
        """断开模拟交易所"""
        logger.info("模拟交易所已断开")
    
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """获取模拟余额"""
        if currency:
            return {currency: self.balances.get(currency)} if currency in self.balances else {}
        return self.balances.copy()
    
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        params: Optional[Dict] = None
    ) -> Order:
        """创建模拟订单"""
        self.order_counter += 1
        order_id = f"sim_{self.order_counter}"
        
        current_price = self.price_data.get(symbol, price or Decimal("0"))
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            status=OrderStatus.OPEN,
            timestamp=int(time.time() * 1000),
            last_update=int(time.time() * 1000)
        )
        
        # 市价单立即成交
        if order_type == OrderType.MARKET:
            order.status = OrderStatus.CLOSED
            order.filled = amount
            order.remaining = Decimal("0")
            order.cost = amount * current_price
            
            # 更新余额
            await self._update_balance(symbol, side, amount, current_price)
        
        self.orders[order_id] = order
        logger.info(f"模拟订单创建: {order_id} - {symbol} {side.value} {amount}")
        return order
    
    async def _update_balance(self, symbol: str, side: OrderSide, amount: Decimal, price: Decimal):
        """更新模拟余额"""
        base, quote = symbol.split('/')
        
        if side == OrderSide.BUY:
            # 买入: 减少quote, 增加base
            cost = amount * price
            if quote in self.balances:
                self.balances[quote].free -= cost
                self.balances[quote].total -= cost
            if base not in self.balances:
                self.balances[base] = Balance(base, Decimal("0"), Decimal("0"), Decimal("0"))
            self.balances[base].free += amount
            self.balances[base].total += amount
        else:
            # 卖出: 减少base, 增加quote
            revenue = amount * price
            if base in self.balances:
                self.balances[base].free -= amount
                self.balances[base].total -= amount
            if quote not in self.balances:
                self.balances[quote] = Balance(quote, Decimal("0"), Decimal("0"), Decimal("0"))
            self.balances[quote].free += revenue
            self.balances[quote].total += revenue
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消模拟订单"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELED
            logger.info(f"模拟订单取消: {order_id}")
            return True
        return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """获取模拟订单"""
        return self.orders.get(order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交模拟订单"""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.OPEN]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取模拟持仓"""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions


class ExchangeManager:
    """交易所管理器 - 管理多个交易所实例"""
    
    def __init__(self):
        self.exchanges: Dict[str, BaseExchange] = {}
        self._default_exchange: Optional[str] = None
    
    def add_exchange(self, name: str, exchange: BaseExchange, set_default: bool = False):
        """添加交易所"""
        self.exchanges[name] = exchange
        if set_default or self._default_exchange is None:
            self._default_exchange = name
    
    def get_exchange(self, name: Optional[str] = None) -> BaseExchange:
        """获取交易所实例"""
        if name is None:
            name = self._default_exchange
        if name not in self.exchanges:
            raise ValueError(f"交易所未找到: {name}")
        return self.exchanges[name]
    
    async def connect_all(self):
        """连接所有交易所"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.connect()
            except Exception as e:
                logger.error(f"连接交易所失败 {name}: {e}")
    
    async def disconnect_all(self):
        """断开所有交易所"""
        for exchange in self.exchanges.values():
            await exchange.disconnect()


# 便捷函数
def create_binance_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True
) -> CCXTExchange:
    """创建Binance客户端"""
    config = ExchangeConfig(
        exchange_id='binance',
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox,
        options={'defaultType': 'spot'}
    )
    return CCXTExchange(config)


def create_binance_futures_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True
) -> CCXTExchange:
    """创建Binance合约客户端"""
    config = ExchangeConfig(
        exchange_id='binance',
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox,
        options={'defaultType': 'future'}
    )
    return CCXTExchange(config)


def create_okx_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    password: Optional[str] = None,
    sandbox: bool = True
) -> CCXTExchange:
    """创建OKX客户端"""
    config = ExchangeConfig(
        exchange_id='okx',
        api_key=api_key,
        api_secret=api_secret,
        password=password,
        sandbox=sandbox
    )
    return CCXTExchange(config)


def create_simulated_exchange(
    initial_balances: Optional[Dict[str, Decimal]] = None
) -> SimulatedExchange:
    """创建模拟交易所"""
    return SimulatedExchange(initial_balances)


# 示例用法
if __name__ == "__main__":
    async def test():
        # 创建模拟交易所
        sim = create_simulated_exchange({
            'USDT': Decimal('10000'),
            'BTC': Decimal('0.5')
        })
        
        await sim.connect()
        
        # 设置价格
        sim.set_price('BTC/USDT', Decimal('50000'))
        
        # 创建市价买单
        order = await sim.create_order(
            symbol='BTC/USDT',
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Decimal('0.1')
        )
        
        print(f"订单: {order}")
        
        # 查看余额
        balances = await sim.get_balance()
        for curr, bal in balances.items():
            print(f"{curr}: {bal.free} (free) / {bal.used} (used)")
        
        await sim.disconnect()
    
    asyncio.run(test())
