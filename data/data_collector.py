"""
数字货币数据获取模块
使用CCXT库连接多个交易所获取K线数据
支持Binance、OKX等主流交易所
"""

import time
import asyncio
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

import ccxt
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketType(Enum):
    """市场类型枚举"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "swap"


class TimeFrame(Enum):
    """时间周期枚举"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    password: Optional[str] = None
    sandbox: bool = False
    enable_rate_limit: bool = True
    options: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为CCXT配置字典"""
        config = {
            'enableRateLimit': self.enable_rate_limit,
            'sandbox': self.sandbox,
        }
        if self.api_key:
            config['apiKey'] = self.api_key
        if self.api_secret:
            config['secret'] = self.api_secret
        if self.password:
            config['password'] = self.password
        if self.options:
            config['options'] = self.options
        return config


class DataCollector:
    """
    数字货币数据收集器
    
    功能:
    - 连接多个交易所
    - 获取历史K线数据
    - 批量下载多币种数据
    - 支持增量更新
    """
    
    # 常用币种列表 (50+)
    POPULAR_SYMBOLS = [
        # 主流币
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT",
        # Layer 1
        "NEAR/USDT", "APT/USDT", "SUI/USDT", "SEI/USDT", "TON/USDT",
        "INJ/USDT", "TIA/USDT", "STRK/USDT", "OP/USDT", "ARB/USDT",
        # DeFi
        "UNI/USDT", "AAVE/USDT", "COMP/USDT", "MKR/USDT", "CRV/USDT",
        "SNX/USDT", "YFI/USDT", "1INCH/USDT", "SUSHI/USDT", "LDO/USDT",
        # Meme & 其他
        "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", "BONK/USDT",
        "WIF/USDT", "BOME/USDT", "WLD/USDT", "ARKM/USDT", "PYTH/USDT",
        # 其他热门
        "RENDER/USDT", "TAO/USDT", "FET/USDT", "AGIX/USDT", "RNDR/USDT",
        "IMX/USDT", "GRT/USDT", "FLOW/USDT", "EGLD/USDT", "XTZ/USDT",
        "ALGO/USDT", "VET/USDT", "FIL/USDT", "TRX/USDT", "ETC/USDT",
        "BCH/USDT", "LTC/USDT", "ATOM/USDT", "ICP/USDT", "HBAR/USDT",
    ]
    
    # 时间周期映射 (毫秒)
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(
        self,
        exchange_name: str = "binance",
        config: Optional[ExchangeConfig] = None,
        market_type: MarketType = MarketType.SPOT
    ):
        """
        初始化数据收集器
        
        Args:
            exchange_name: 交易所名称 (binance, okx, bybit, etc.)
            config: 交易所配置
            market_type: 市场类型
        """
        self.exchange_name = exchange_name.lower()
        self.market_type = market_type
        self.config = config or ExchangeConfig(name=exchange_name)
        self.exchange: Optional[ccxt.Exchange] = None
        self._rate_limit_remaining = 1000
        self._rate_limit_reset = 0
        
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """初始化交易所连接"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class(self.config.to_dict())
            
            # 加载市场信息
            self.exchange.load_markets()
            
            logger.info(f"成功连接到 {self.exchange_name} 交易所")
            logger.info(f"支持的交易对数量: {len(self.exchange.markets)}")
            
        except Exception as e:
            logger.error(f"初始化交易所失败: {e}")
            raise
    
    def _get_symbol_for_market(self, symbol: str) -> str:
        """
        根据市场类型调整交易对格式
        
        Args:
            symbol: 基础交易对 (如 BTC/USDT)
        
        Returns:
            str: 适配当前市场的交易对
        """
        if self.market_type == MarketType.FUTURES:
            # 期货合约格式
            if self.exchange_name == "binance":
                return symbol.replace("/USDT", "/USDT:USDT")
            elif self.exchange_name == "okx":
                return symbol.replace("/USDT", "-USDT-SWAP")
            elif self.exchange_name == "bybit":
                return symbol.replace("/USDT", "USDT")
        elif self.market_type == MarketType.PERPETUAL:
            # 永续合约格式
            if self.exchange_name == "binance":
                return symbol.replace("/USDT", "/USDT:USDT")
            elif self.exchange_name == "okx":
                return symbol.replace("/USDT", "-USDT-SWAP")
        
        return symbol
    
    def _handle_rate_limit(self) -> None:
        """处理速率限制"""
        if self.exchange:
            # 使用CCXT内置的速率限制
            pass
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            since: 开始时间戳 (毫秒)
            limit: 获取条数
            params: 额外参数
        
        Returns:
            pd.DataFrame: OHLCV数据
        """
        if not self.exchange:
            raise RuntimeError("交易所未初始化")
        
        adjusted_symbol = self._get_symbol_for_market(symbol)
        params = params or {}
        
        try:
            # 获取原始数据
            ohlcv = self.exchange.fetch_ohlcv(
                adjusted_symbol,
                timeframe,
                since=since,
                limit=limit,
                params=params
            )
            
            if not ohlcv:
                logger.warning(f"未获取到数据: {symbol} {timeframe}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.debug(f"获取到 {len(df)} 条数据: {symbol} {timeframe}")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"网络错误: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"交易所错误: {e}")
            raise
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            raise
    
    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        获取指定时间范围的K线数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            max_retries: 最大重试次数
        
        Returns:
            pd.DataFrame: OHLCV数据
        """
        if not self.exchange:
            raise RuntimeError("交易所未初始化")
        
        # 默认获取最近7天数据
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=7)
        
        # 转换为毫秒时间戳
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # 计算每次请求的数据量
        timeframe_ms = self.TIMEFRAME_MS.get(timeframe, 60 * 1000)
        limit = 1000  # 大多数交易所的最大限制
        
        all_data = []
        current_start = start_ms
        
        logger.info(f"开始获取 {symbol} {timeframe} 数据，时间范围: {start_time} 到 {end_time}")
        
        while current_start < end_ms:
            retries = 0
            success = False
            
            while retries < max_retries and not success:
                try:
                    self._handle_rate_limit()
                    
                    df = self.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_start,
                        limit=limit
                    )
                    
                    if df.empty:
                        # 没有更多数据
                        current_start = end_ms
                        success = True
                        continue
                    
                    all_data.append(df)
                    
                    # 更新下一次请求的开始时间
                    last_timestamp = df.index[-1]
                    current_start = int(last_timestamp.timestamp() * 1000) + timeframe_ms
                    
                    # 如果已经到达结束时间，退出循环
                    if current_start >= end_ms:
                        success = True
                        break
                    
                    # 检查是否获取到新数据
                    if len(df) < 2:
                        success = True
                        break
                    
                    success = True
                    
                    # 添加小延迟避免触发速率限制
                    time.sleep(0.1)
                    
                except Exception as e:
                    retries += 1
                    logger.warning(f"请求失败 (重试 {retries}/{max_retries}): {e}")
                    time.sleep(1 * retries)  # 指数退避
            
            if not success:
                logger.error(f"获取数据失败，已达到最大重试次数")
                break
        
        if not all_data:
            logger.warning(f"未获取到任何数据: {symbol} {timeframe}")
            return pd.DataFrame()
        
        # 合并所有数据
        combined_df = pd.concat(all_data)
        
        # 去重并排序
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        # 过滤时间范围
        combined_df = combined_df[
            (combined_df.index >= start_time) & 
            (combined_df.index <= end_time)
        ]
        
        logger.info(f"成功获取 {len(combined_df)} 条数据: {symbol} {timeframe}")
        return combined_df
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个交易对数据
        
        Args:
            symbols: 交易对列表
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            progress_callback: 进度回调函数 (symbol, current, total)
        
        Returns:
            Dict[str, pd.DataFrame]: 交易对到数据的映射
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    progress_callback(symbol, i + 1, total)
                
                df = self.fetch_ohlcv_range(
                    symbol,
                    timeframe,
                    start_time,
                    end_time
                )
                
                if not df.empty:
                    results[symbol] = df
                
                # 添加延迟避免速率限制
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
                continue
        
        logger.info(f"批量获取完成: {len(results)}/{total} 个交易对")
        return results
    
    def fetch_popular_symbols(
        self,
        timeframe: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_symbols: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        获取热门币种数据
        
        Args:
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            max_symbols: 最大币种数量
        
        Returns:
            Dict[str, pd.DataFrame]: 交易对到数据的映射
        """
        symbols = self.POPULAR_SYMBOLS[:max_symbols] if max_symbols else self.POPULAR_SYMBOLS
        
        # 过滤当前交易所支持的币种
        available_symbols = [
            s for s in symbols 
            if self._get_symbol_for_market(s) in self.exchange.markets
        ]
        
        logger.info(f"可用币种数量: {len(available_symbols)}/{len(symbols)}")
        
        return self.fetch_multiple_symbols(
            available_symbols,
            timeframe,
            start_time,
            end_time
        )
    
    def get_available_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        获取交易所支持的交易对列表
        
        Args:
            quote_currency: 计价货币
        
        Returns:
            List[str]: 交易对列表
        """
        if not self.exchange:
            raise RuntimeError("交易所未初始化")
        
        symbols = []
        for symbol, market in self.exchange.markets.items():
            if market.get('quote') == quote_currency:
                if self.market_type == MarketType.SPOT and market.get('spot'):
                    symbols.append(symbol)
                elif self.market_type == MarketType.FUTURES and market.get('future'):
                    symbols.append(symbol)
                elif self.market_type == MarketType.PERPETUAL and market.get('swap'):
                    symbols.append(symbol)
        
        return sorted(symbols)
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        if not self.exchange:
            raise RuntimeError("交易所未初始化")
        
        return {
            'name': self.exchange_name,
            'id': self.exchange.id,
            'markets_count': len(self.exchange.markets),
            'timeframes': list(self.exchange.timeframes.keys()) if self.exchange.timeframes else [],
            'has': self.exchange.has,
            'urls': self.exchange.urls,
            'version': self.exchange.version,
        }
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取最新行情
        
        Args:
            symbol: 交易对符号
        
        Returns:
            Dict: 行情数据
        """
        if not self.exchange:
            raise RuntimeError("交易所未初始化")
        
        adjusted_symbol = self._get_symbol_for_market(symbol)
        
        try:
            ticker = self.exchange.fetch_ticker(adjusted_symbol)
            return {
                'symbol': symbol,
                'last': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'volume': ticker.get('quoteVolume'),
                'change': ticker.get('change'),
                'percentage': ticker.get('percentage'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'timestamp': ticker.get('timestamp'),
            }
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            raise
    
    def close(self) -> None:
        """关闭交易所连接"""
        if self.exchange:
            # CCXT没有显式的关闭方法
            self.exchange = None
            logger.info("交易所连接已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class MultiExchangeCollector:
    """
    多交易所数据收集器
    
    支持同时连接多个交易所，提供数据聚合功能
    """
    
    def __init__(self):
        """初始化多交易所收集器"""
        self.collectors: Dict[str, DataCollector] = {}
    
    def add_exchange(
        self,
        exchange_name: str,
        config: Optional[ExchangeConfig] = None,
        market_type: MarketType = MarketType.SPOT
    ) -> DataCollector:
        """
        添加交易所
        
        Args:
            exchange_name: 交易所名称
            config: 交易所配置
            market_type: 市场类型
        
        Returns:
            DataCollector: 数据收集器实例
        """
        collector = DataCollector(exchange_name, config, market_type)
        self.collectors[exchange_name] = collector
        return collector
    
    def remove_exchange(self, exchange_name: str) -> None:
        """移除交易所"""
        if exchange_name in self.collectors:
            self.collectors[exchange_name].close()
            del self.collectors[exchange_name]
    
    def fetch_best_price(
        self,
        symbol: str,
        side: str = "buy"
    ) -> Tuple[str, float]:
        """
        从多个交易所获取最优价格
        
        Args:
            symbol: 交易对符号
            side: 买卖方向 (buy/sell)
        
        Returns:
            Tuple[str, float]: (交易所名称, 价格)
        """
        prices = []
        
        for name, collector in self.collectors.items():
            try:
                ticker = collector.get_ticker(symbol)
                price = ticker['ask'] if side == "buy" else ticker['bid']
                prices.append((name, price))
            except Exception as e:
                logger.warning(f"从 {name} 获取价格失败: {e}")
        
        if not prices:
            raise RuntimeError("无法从任何交易所获取价格")
        
        # 买入取最低价，卖出取最高价
        if side == "buy":
            return min(prices, key=lambda x: x[1])
        else:
            return max(prices, key=lambda x: x[1])
    
    def close_all(self) -> None:
        """关闭所有交易所连接"""
        for collector in self.collectors.values():
            collector.close()
        self.collectors.clear()


if __name__ == "__main__":
    # 测试代码
    collector = DataCollector("binance")
    
    # 获取交易所信息
    info = collector.get_exchange_info()
    print(f"交易所信息: {info['name']}")
    print(f"支持的交易对: {info['markets_count']}")
    
    # 获取单个币种数据
    df = collector.fetch_ohlcv("BTC/USDT", "1m", limit=100)
    print(f"\n获取到 {len(df)} 条BTC数据")
    print(df.head())
    
    # 获取行情
    ticker = collector.get_ticker("BTC/USDT")
    print(f"\nBTC当前价格: {ticker['last']}")
    
    collector.close()
