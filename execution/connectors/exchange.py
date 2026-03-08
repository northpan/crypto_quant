"""
Exchange execution engine using CCXT.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import uuid

import ccxt.async_support as ccxt

from ...core.base import (
    BaseExecutionEngine,
    Order,
    Trade,
    Position,
    Portfolio,
    OrderSide,
    OrderType,
    PositionSide,
    MarketType,
)


class CCXTExecutionEngine(BaseExecutionEngine):
    """
    Execution engine using CCXT library.
    
    Supports multiple exchanges through unified interface.
    """

    def __init__(
        self,
        name: str,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self._exchange: Optional[ccxt.Exchange] = None
        self._order_callbacks: List[callable] = []

    async def connect(self) -> bool:
        """Establish connection to exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': self.sandbox,
                'options': {'defaultType': 'swap'},
            })
            await self._exchange.load_markets()
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to {self.exchange_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        """Close connection to exchange."""
        if self._exchange:
            await self._exchange.close()
            self._is_connected = False
        return True

    async def submit_order(self, order: Order) -> Optional[Trade]:
        """Submit order to exchange."""
        if not self._is_connected:
            await self.connect()
        
        try:
            # Map order type
            order_type_map = {
                OrderType.MARKET: 'market',
                OrderType.LIMIT: 'limit',
                OrderType.STOP: 'stop',
                OrderType.STOP_LIMIT: 'stop_limit',
            }
            
            # Map order side
            side_map = {
                OrderSide.BUY: 'buy',
                OrderSide.SELL: 'sell',
            }
            
            # Prepare parameters
            params = {
                'reduceOnly': order.reduce_only,
            }
            
            if order.post_only:
                params['postOnly'] = True
            
            # Create order
            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=order_type_map.get(order.order_type, 'limit'),
                side=side_map.get(order.side, 'buy'),
                amount=order.quantity,
                price=order.price,
                params=params,
            )
            
            # Check if filled immediately
            if result.get('status') == 'closed' and result.get('filled', 0) > 0:
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=result['filled'],
                    price=result['average'] or result['price'],
                    timestamp=datetime.now(),
                    fee=result.get('fee', {}).get('cost', 0),
                    fee_currency=result.get('fee', {}).get('currency', 'USDT'),
                )
                return trade
            
            return None
            
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        if not self._is_connected:
            await self.connect()
        
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status."""
        if not self._is_connected:
            await self.connect()
        
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return {
                'order_id': order['id'],
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'average_price': order['average'],
            }
        except Exception as e:
            print(f"Error fetching order status: {e}")
            return {}

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position."""
        if not self._is_connected:
            await self.connect()
        
        try:
            positions = await self._exchange.fetch_positions([symbol])
            
            for pos in positions:
                if pos['symbol'] == symbol and pos['contracts'] != 0:
                    return Position(
                        symbol=symbol,
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        quantity=abs(float(pos['contracts'])),
                        entry_price=float(pos['entryPrice']),
                        mark_price=float(pos['markPrice']),
                        unrealized_pnl=float(pos['unrealizedPnl']),
                        margin=float(pos.get('initialMargin', 0)),
                        leverage=float(pos['leverage']),
                        liquidation_price=float(pos.get('liquidationPrice', 0)) if pos.get('liquidationPrice') else None,
                        timestamp=datetime.now(),
                    )
            
            return None
            
        except Exception as e:
            print(f"Error fetching position: {e}")
            return None

    async def get_portfolio(self) -> Portfolio:
        """Get portfolio state."""
        if not self._is_connected:
            await self.connect()
        
        try:
            balance = await self._exchange.fetch_balance()
            positions = await self._exchange.fetch_positions()
            
            # Build position dictionary
            position_dict = {}
            for pos in positions:
                if pos['contracts'] != 0:
                    symbol = pos['symbol']
                    position_dict[symbol] = Position(
                        symbol=symbol,
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        quantity=abs(float(pos['contracts'])),
                        entry_price=float(pos['entryPrice']),
                        mark_price=float(pos['markPrice']),
                        unrealized_pnl=float(pos['unrealizedPnl']),
                        margin=float(pos.get('initialMargin', 0)),
                        leverage=float(pos['leverage']),
                        timestamp=datetime.now(),
                    )
            
            # Get cash balance
            cash = balance.get('USDT', {}).get('free', 0)
            total_value = balance.get('USDT', {}).get('total', 0)
            
            return Portfolio(
                positions=position_dict,
                cash=cash,
                total_value=total_value,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            print(f"Error fetching portfolio: {e}")
            return Portfolio()

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for symbol."""
        if not self._is_connected:
            await self.connect()
        
        try:
            await self._exchange.set_leverage(int(leverage), symbol)
            return True
        except Exception as e:
            print(f"Error setting leverage: {e}")
            return False

    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Set margin mode (isolated/cross)."""
        if not self._is_connected:
            await self.connect()
        
        try:
            await self._exchange.set_margin_mode(mode, symbol)
            return True
        except Exception as e:
            print(f"Error setting margin mode: {e}")
            return False
