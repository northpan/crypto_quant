#!/usr/bin/env python3
"""
数字货币量化交易系统 - 主程序入口
Crypto Quantitative Trading System - Main Entry Point

目标：实现夏普比率 > 4 的分钟级量化交易策略
支持：50+ 数字货币、现货和永续合约
模块：因子挖掘、模型组合、风控优化、交易执行
"""

import os
import sys
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 提前创建日志和报告目录，避免 FileHandler 报错
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
(PROJECT_ROOT / 'reports').mkdir(exist_ok=True)

# 导入各模块
from data import DataManager
from factors import FactorPool
from models import (
    ModelConfig, LightGBMModel, XGBoostModel,
    LSTMModel, TransformerModel, EnsembleModel, ModelTrainer, ModelEvaluator
)
from risk import create_risk_manager, RiskManager
from execution import (
    OrderManager, ExecutionEngine,
    create_simulated_exchange
)
from execution.exchange.exchange_client import (
    create_binance_client, create_okx_client, OrderSide
)
from backtest import (
    BacktestEngine, PerformanceAnalyzer, TradeAnalyzer,
    ReportGenerator, CostModel
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class _MockDataManager:
    """无网络时使用的模拟数据管理器，返回合成 K 线"""

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_time=None,
        end_time=None,
    ):
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=7)
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            delta = (end_time - start_time).total_seconds() / 60
        else:
            delta = 7 * 24 * 60
        n = max(100, int(delta))
        freq = '1min' if timeframe == '1m' else '5min' if timeframe == '5m' else '15min'
        dates = pd.date_range(start=start_time, periods=n, freq=freq)
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0001, 0.002, n)
        prices = 50000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.002)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.002)),
            'close': prices * (1 + np.random.randn(n) * 0.001),
            'volume': np.random.uniform(100, 1000, n).astype(int)
        })
        return df


def _load_local_csv(symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
    """从 data/csv 目录加载本地 CSV（如 scripts/download_okx_klines.py 下载的）。"""
    safe = symbol.replace("/", "_").replace("-", "_").upper()
    if not safe.endswith("_USDT"):
        safe = f"{safe}_USDT" if "USDT" not in safe else safe
    csv_dir = PROJECT_ROOT / "data" / "csv"
    path = csv_dir / f"{safe}_{timeframe}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 若 CSV 为 UTC 等带时区，与 naive 的 start/end 比较会丢数据，统一为 UTC 再筛
        s, e = start_time, end_time
        if df["timestamp"].dt.tz is not None:
            s = start_time.replace(tzinfo=timezone.utc) if start_time.tzinfo is None else start_time
            e = end_time.replace(tzinfo=timezone.utc) if end_time.tzinfo is None else end_time
        df = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
        return df if len(df) > 0 else None
    except Exception:
        return None


@dataclass
class TradingConfig:
    """交易配置"""
    mode: str = 'backtest'  # backtest, paper, live
    symbols: List[str] = None
    timeframes: List[str] = None
    trade_type: str = 'futures'  # spot, futures
    prediction_horizon: int = 60  # 预测未来60分钟
    retrain_interval: int = 24  # 每24小时重训练
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
                'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ETC/USDT', 'XLM/USDT',
                'TRX/USDT', 'EOS/USDT', 'FIL/USDT', 'AAVE/USDT', 'SUSHI/USDT',
                'COMP/USDT', 'MKR/USDT', 'YFI/USDT', 'SNX/USDT', 'CRV/USDT',
                '1INCH/USDT', 'GRT/USDT', 'MANA/USDT', 'SAND/USDT', 'AXS/USDT',
                'FTT/USDT', 'HT/USDT', 'OKB/USDT', 'LEO/USDT', 'CRO/USDT',
                'VET/USDT', 'ICP/USDT', 'THETA/USDT', 'XTZ/USDT', 'ALGO/USDT',
                'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT', 'GALA/USDT', 'APE/USDT',
                'FLOW/USDT', 'EGLD/USDT', 'HBAR/USDT', 'QNT/USDT', 'CHZ/USDT'
            ]
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m']


class CryptoQuantSystem:
    """
    数字货币量化交易系统主类
    
    整合数据、因子、模型、风控、执行五大模块
    实现分钟级预测和交易，目标夏普 > 4
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化交易系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.trading_config = TradingConfig(**self.config.get('trading', {}))
        
        # 初始化各模块
        self.data_manager: Optional[DataManager] = None
        self.factor_pool: Optional[FactorPool] = None
        self.model: Optional[Any] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.backtest_engine: Optional[BacktestEngine] = None
        
        # 运行状态
        self.is_running = False
        self.positions = {}
        self.signals = {}
        
        logger.info("=" * 60)
        logger.info("数字货币量化交易系统初始化")
        logger.info(f"模式: {self.trading_config.mode}")
        logger.info(f"交易品种: {len(self.trading_config.symbols)}个")
        logger.info(f"时间周期: {self.trading_config.timeframes}")
        logger.info("=" * 60)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = PROJECT_ROOT / 'config' / 'config.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'trading': {
                'mode': 'backtest',
                'trade_type': 'futures',
                'prediction_horizon': 60,
                'retrain_interval': 24
            },
            'data': {
                'exchange': 'binance',
                'data_dir': './data'
            },
            'risk': {
                'max_position_pct': 0.2,
                'max_drawdown_pct': 0.15,
                'daily_loss_limit_pct': 0.05,
                'leverage': 3
            },
            'model': {
                'type': 'ensemble',
                'sharpe_target': 4.0
            }
        }
    
    async def initialize(self):
        """初始化所有模块"""
        logger.info("开始初始化各模块...")
        
        # 1. 初始化数据管理器
        await self._init_data_manager()
        
        # 2. 初始化因子池
        await self._init_factor_pool()
        
        # 3. 初始化模型
        await self._init_model()
        
        # 4. 初始化风控管理器
        await self._init_risk_manager()
        
        # 5. 初始化执行引擎
        await self._init_execution_engine()
        
        logger.info("所有模块初始化完成")
    
    async def _init_data_manager(self):
        """初始化数据管理器（短超时连接交易所，失败则用模拟数据）"""
        logger.info("初始化数据管理器...")
        data_cfg = self.config.get('data', {})
        try:
            # 在线程中创建并设置短超时，避免 Binance 不可达时长时间阻塞
            def _create():
                return DataManager(
                    data_path=data_cfg.get('data_dir', './data'),
                    exchange_name=data_cfg.get('exchange', 'binance')
                )
            loop = asyncio.get_event_loop()
            self.data_manager = await asyncio.wait_for(
                loop.run_in_executor(None, _create),
                timeout=15.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"数据管理器连接交易所失败或超时 ({e})，使用模拟数据运行")
            self.data_manager = _MockDataManager()
    
    async def _init_factor_pool(self):
        """初始化因子池"""
        logger.info("初始化因子池...")
        self.factor_pool = FactorPool()
    
    async def _init_model(self):
        """初始化预测模型"""
        logger.info("初始化预测模型...")
        model_type = self.config.get('model', {}).get('type', 'ensemble')
        if model_type == 'ensemble' and (LSTMModel is None or TransformerModel is None):
            model_type = 'lightgbm'
            logger.info("LSTM/Transformer 未安装，使用 lightgbm 作为预测模型")
        model_config = ModelConfig(
            model_name='crypto_predictor',
            model_type=model_type,
            prediction_horizon=self.trading_config.prediction_horizon
        )
        
        if model_type == 'lightgbm':
            self.model = LightGBMModel(model_config)
        elif model_type == 'xgboost':
            self.model = XGBoostModel(model_config)
        elif model_type == 'lstm':
            self.model = LSTMModel(model_config)
        else:  # ensemble
            self.model = EnsembleModel(model_config)
    
    async def _init_risk_manager(self):
        """初始化风控管理器"""
        logger.info("初始化风控管理器...")
        risk_mode = self.config.get('risk', {}).get('mode', 'moderate')
        self.risk_manager = create_risk_manager(
            account_balance=100000.0,
            risk_mode=risk_mode
        )
    
    async def _init_execution_engine(self):
        """初始化执行引擎"""
        logger.info("初始化执行引擎...")
        from decimal import Decimal

        if self.trading_config.mode == 'backtest':
            exchange = create_simulated_exchange(
                initial_balances={'USDT': Decimal('100000.0')}
            )
        else:
            ex_name = self.config.get('data', {}).get('exchange', 'binance')
            api_key = os.getenv('EXCHANGE_API_KEY') or os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('EXCHANGE_API_SECRET') or os.getenv('BINANCE_API_SECRET')
            if ex_name == 'okx':
                exchange = create_okx_client(api_key or '', api_secret or '', sandbox=(self.trading_config.mode == 'paper'))
            else:
                exchange = create_binance_client(api_key or '', api_secret or '', sandbox=(self.trading_config.mode == 'paper'))
        await exchange.connect()
        self._order_manager = OrderManager(exchange)
        await self._order_manager.start()
        self.execution_engine = ExecutionEngine(self._order_manager)
    
    async def train_model(self, symbols: Optional[List[str]] = None, 
                         days: int = 90) -> Dict[str, Any]:
        """
        训练预测模型
        
        Args:
            symbols: 训练用的币种列表
            days: 历史数据天数
            
        Returns:
            训练结果报告
        """
        logger.info(f"开始训练模型，数据范围: {days}天")
        
        symbols = symbols or self.trading_config.symbols[:2]  # 演示用 2 个币种
        
        # 1. 获取历史数据（DataManager.get_data 为同步接口，按时间范围拉取）
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        all_data = {}
        for symbol in symbols:
            df = self.data_manager.get_data(
                symbol=symbol,
                timeframe='1m',
                start_time=start_time,
                end_time=end_time
            )
            all_data[symbol] = df
        
        # 2. 计算因子（仅用部分因子以加速）
        factor_names = list(self.factor_pool.factors.keys())[:8]
        factor_data = {}
        for symbol, df in all_data.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            if df.index.name is None and 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            factors_df = self.factor_pool.compute(df, factor_names=factor_names, verbose=False)
            if factors_df is not None and not factors_df.empty:
                df_with_factors = pd.concat([df, factors_df], axis=1)
            else:
                df_with_factors = df
            factor_data[symbol] = df_with_factors
        
        # 3. 准备训练数据
        X_train, y_train, X_val, y_val = self._prepare_training_data(factor_data)
        
        # 4. 训练模型
        trainer = ModelTrainer()
        training_result = trainer.train(
            self.model,
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train.values if hasattr(y_train, 'values') else y_train,
            validation_data=(X_val.values if hasattr(X_val, 'values') else X_val, y_val.values if hasattr(y_val, 'values') else y_val)
        )
        
        # 5. 评估模型
        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate(self.model, X_val, y_val)
        
        _m = getattr(evaluation, 'metrics', evaluation)
        sharpe = getattr(_m, 'sharpe_ratio', 0) if hasattr(_m, 'sharpe_ratio') else (float(_m.get('sharpe_ratio', 0)) if isinstance(_m, dict) else 0)
        logger.info(f"模型训练完成，验证集夏普比率: {sharpe:.4f}")
        
        return {
            'training': training_result,
            'evaluation': evaluation
        }
    
    def _prepare_training_data(self, factor_data: Dict) -> tuple:
        """准备训练数据"""
        import pandas as pd
        import numpy as np
        
        all_features = []
        all_targets = []
        
        for symbol, df in factor_data.items():
            # 选择数值型特征列
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['open', 'high', 'low', 'close', 'volume']]
            
            # 计算未来收益率作为目标
            df['target'] = df['close'].pct_change(self.trading_config.prediction_horizon).shift(-self.trading_config.prediction_horizon)
            
            # 删除NaN
            df_clean = df[feature_cols + ['target']].dropna()
            
            all_features.append(df_clean[feature_cols])
            all_targets.append(df_clean['target'])
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # 划分训练集和验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    async def generate_signals(self) -> Dict[str, Dict]:
        """
        生成交易信号
        
        Returns:
            各币种的交易信号
        """
        signals = {}
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        for symbol in self.trading_config.symbols:
            try:
                # 获取最新数据
                df = self.data_manager.get_data(
                    symbol=symbol,
                    timeframe='1m',
                    start_time=start_time,
                    end_time=end_time
                )
                
                # 计算因子
                if df.index.name is None and 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                factors_df = self.factor_pool.compute(df, verbose=False)
                df_with_factors = pd.concat([df, factors_df], axis=1) if factors_df is not None and not factors_df.empty else df
                
                # 准备特征
                feature_cols = df_with_factors.select_dtypes(include=[float, int]).columns.tolist()
                feature_cols = [c for c in feature_cols if c not in ['open', 'high', 'low', 'close', 'volume']]
                latest_features = df_with_factors[feature_cols].iloc[-1:].values
                
                # 预测
                prediction = self.model.predict(latest_features)[0]
                
                # 生成信号
                signal = self._interpret_prediction(prediction, df)
                signals[symbol] = signal
                
            except Exception as e:
                logger.error(f"生成{symbol}信号失败: {e}")
                signals[symbol] = {'action': 'hold', 'confidence': 0}
        
        self.signals = signals
        return signals
    
    def _interpret_prediction(self, prediction: float, df: pd.DataFrame) -> Dict:
        """解释预测结果为交易信号"""
        current_price = df['close'].iloc[-1]
        volatility = df['close'].pct_change().std() * np.sqrt(1440)  # 日波动率
        
        # 根据预测值和波动率确定信号
        threshold = volatility * 0.5  # 阈值设为波动率的一半
        
        if prediction > threshold:
            return {
                'action': 'buy',
                'confidence': min(abs(prediction) / volatility, 1.0),
                'predicted_return': prediction,
                'current_price': current_price
            }
        elif prediction < -threshold:
            return {
                'action': 'sell',
                'confidence': min(abs(prediction) / volatility, 1.0),
                'predicted_return': prediction,
                'current_price': current_price
            }
        else:
            return {
                'action': 'hold',
                'confidence': 0,
                'predicted_return': prediction,
                'current_price': current_price
            }
    
    async def execute_signals(self, signals: Dict[str, Dict]):
        """
        执行交易信号
        
        Args:
            signals: 交易信号字典
        """
        for symbol, signal in signals.items():
            if signal['action'] == 'hold':
                continue
            
            # 风控检查（approve_trade 返回 TradeApproval 对象）
            risk_check = self.risk_manager.approve_trade(
                symbol=symbol,
                side=signal['action'],
                entry_price=float(signal['current_price'])
            )
            
            if not risk_check.approved:
                logger.info(f"{symbol} 交易被风控拒绝: {risk_check.message}")
                continue
            
            # 执行交易（通过 order_manager 市价单）
            try:
                from decimal import Decimal
                side = OrderSide.BUY if signal['action'] == 'buy' else OrderSide.SELL
                order = await self._order_manager.place_market_order(
                    symbol=symbol,
                    side=side,
                    amount=Decimal(str(risk_check.position_size))
                )
                logger.info(f"{symbol} 订单已提交: {order.id if order else 'N/A'}")
            except Exception as e:
                logger.error(f"{symbol} 订单执行失败: {e}")
    
    async def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        save_report: bool = True,
        use_local_data_only: bool = False,
        report_suffix: Optional[str] = None,
        symbols_to_use: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            start_date: 开始日期 '2024-01-01'
            end_date: 结束日期 '2024-12-31'
            strategy_params: 策略参数字典。可选键含 inverse_strategy（True=金叉开空/死叉平空）
            save_report: 是否生成并保存 HTML 报告
            use_local_data_only: True 时仅从 data/csv 加载，无本地数据则跳过该品种
            report_suffix: 报告文件名后缀，如 'original' -> backtest_2026-03-01_2026-03-05_original.html
            symbols_to_use: 仅对指定品种回测（如 ['BTC_USDT']），None 则用 trading_config.symbols[:5]
            
        Returns:
            回测结果报告
        """
        # 策略参数默认值，可被 strategy_params 覆盖
        sp = strategy_params or {}
        def _p(key: str, default: Any):
            return sp.get(key, default)
        logger.info(f"开始回测: {start_date} 至 {end_date}")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # 初始化回测引擎（使用 backtest 模块内的 CostModel）
        cost_model = CostModel(
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage_model='volatility'
        )
        from backtest.engine.backtest_engine import (
            TradeType, OrderType, OrderSide, PositionSide,
        )
        trade_type = TradeType.FUTURES if self.trading_config.trade_type == 'futures' else TradeType.SPOT
        self.backtest_engine = BacktestEngine(
            initial_capital=100000.0,
            trade_type=trade_type,
            cost_model=cost_model
        )

        # 加载数据：symbols_to_use 指定则只加载这些品种，否则用 trading_config.symbols[:5]
        symbols_to_load = symbols_to_use if symbols_to_use is not None else self.trading_config.symbols[:5]
        for symbol in symbols_to_load:
            df = _load_local_csv(symbol, '1m', start_dt, end_dt)
            if (df is None or df.empty) and not use_local_data_only:
                df = self.data_manager.get_data(
                    symbol=symbol,
                    timeframe='1m',
                    start_time=start_dt,
                    end_time=end_dt
                )
            if df is not None and not df.empty:
                # 确保有 timestamp 列（部分数据源为 index 或列名 datetime）
                if 'datetime' in df.columns and 'timestamp' not in df.columns:
                    df = df.rename(columns={'datetime': 'timestamp'})
                if 'timestamp' not in df.columns:
                    df = df.reset_index()
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'timestamp'})
                self.backtest_engine.load_data(symbol, df)

        if not self.backtest_engine.market_data:
            logger.warning("无可用行情数据，跳过回测")
            return {'performance': {}, 'trade_stats': {}, 'results': {}, 'report_path': None}

        # 策略参数（可被 strategy_params 覆盖，用于调参与参数搜索）
        rsi_period = _p("rsi_period", 14)
        rsi_max_open = _p("rsi_max_open", 70)
        volume_ma_period = _p("volume_ma_period", 20)
        use_volume_filter = _p("use_volume_filter", True)
        stop_loss_pct = _p("stop_loss_pct", 0.005)
        take_profit_pct = _p("take_profit_pct", 0.015)
        max_positions = _p("max_positions", 3)
        max_exposure_pct = _p("max_exposure_pct", 0.6)
        short_period = _p("short_period", 5)
        long_period = _p("long_period", 40)   # 默认 40，网格搜索中长均线表现更稳
        position_pct = _p("position_pct", 0.2)
        inverse_strategy = _p("inverse_strategy", False)  # True=金叉开空/死叉平空，False=金叉开多/死叉平多

        def _ma_crossover_strategy(engine, portfolio, market_data, timestamp):
            need_bars = max(long_period + 5, rsi_period + 2, volume_ma_period + 2)
            total_equity = portfolio.total_value
            open_count = sum(1 for p in portfolio.positions.values() if p.quantity and abs(p.quantity) > 1e-12)
            exposure = sum(
                p.quantity * (market_data[p.symbol].close if p.symbol in market_data else p.entry_price)
                for p in portfolio.positions.values() if p.symbol in market_data
            )
            exposure_pct = exposure / total_equity if total_equity else 0
            opened_this_bar = 0

            for symbol, data in market_data.items():
                df = engine.market_data.get(symbol)
                if df is None or len(df) < need_bars:
                    continue
                hist = df[df["timestamp"] <= timestamp].tail(need_bars)
                if len(hist) < long_period:
                    continue
                close = hist["close"].astype(float)
                volume = hist["volume"].astype(float)
                short_ma = close.rolling(short_period, min_periods=1).mean().iloc[-1]
                long_ma = close.rolling(long_period, min_periods=1).mean().iloc[-1]
                prev_short = close.rolling(short_period, min_periods=1).mean().iloc[-2]
                prev_long = close.rolling(long_period, min_periods=1).mean().iloc[-2]
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = (-delta).where(delta < 0, 0.0)
                avg_gain = gain.rolling(rsi_period, min_periods=1).mean().iloc[-1]
                avg_loss = loss.rolling(rsi_period, min_periods=1).mean().iloc[-1]
                rs = avg_gain / avg_loss if avg_loss and avg_loss > 0 else 999.0
                rsi = 100 - (100 / (1 + rs))
                volume_ma = volume.rolling(volume_ma_period, min_periods=1).mean().iloc[-1]
                current_volume = float(data.volume) if hasattr(data, 'volume') else volume.iloc[-1]
                volume_ok = (not use_volume_filter) or (volume_ma and current_volume >= volume_ma * 0.9)
                position = portfolio.get_position(symbol)
                price = float(data.close)
                low = float(data.low) if hasattr(data, 'low') else price
                high = float(data.high) if hasattr(data, 'high') else price

                # 有持仓：止损/止盈 + 死叉平仓
                if position and position.quantity != 0:
                    entry = position.entry_price
                    if inverse_strategy and position.side == PositionSide.SHORT:
                        stop_price = entry * (1 + stop_loss_pct)
                        tp_price = entry * (1 - take_profit_pct)
                        if high >= stop_price:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.BUY, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                            continue
                        if low <= tp_price:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.BUY, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                            continue
                        if prev_short >= prev_long and short_ma < long_ma:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.BUY, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                        continue
                    if not inverse_strategy and position.side == PositionSide.LONG:
                        stop_price = entry * (1 - stop_loss_pct)
                        tp_price = entry * (1 + take_profit_pct)
                        if low <= stop_price:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.SELL, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                            continue
                        if high >= tp_price:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.SELL, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                            continue
                        if prev_short >= prev_long and short_ma < long_ma:
                            order = engine.execution_engine.create_order(
                                symbol=symbol, order_type=OrderType.MARKET,
                                side=OrderSide.SELL, quantity=abs(position.quantity),
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                        continue

                # 开仓：金叉 + RSI + 量能 + 仓位上限
                if (open_count + opened_this_bar >= max_positions) or (exposure_pct >= max_exposure_pct):
                    continue
                if not position or position.quantity == 0:
                    if (
                        prev_short <= prev_long and short_ma > long_ma
                        and rsi < rsi_max_open and volume_ok and price > 0
                    ):
                        cash_use = portfolio.cash * position_pct
                        quantity = cash_use / price
                        if quantity > 0:
                            side = OrderSide.SELL if inverse_strategy else OrderSide.BUY
                            order = engine.execution_engine.create_order(
                                symbol=symbol,
                                order_type=OrderType.MARKET,
                                side=side,
                                quantity=quantity,
                                trade_type=engine.trade_type,
                            )
                            engine.submit_order(order)
                            opened_this_bar += 1
                            exposure_pct = (exposure + quantity * price) / total_equity if total_equity else 0
        self.backtest_engine.set_strategy(_ma_crossover_strategy)

        # 运行回测
        results = self.backtest_engine.run()

        # 从权益曲线计算收益率
        equity = results.get('equity_curve')
        if equity is not None and not equity.empty and 'equity' in equity.columns:
            returns = equity['equity'].astype(float).pct_change().dropna()
            equity_series = equity['equity'].astype(float)
        else:
            returns = __import__('pandas').Series(dtype=float)
            equity_series = None

        # 绩效分析
        perf_analyzer = PerformanceAnalyzer()
        performance = perf_analyzer.analyze(returns, equity_curve=equity_series)
        performance_dict = vars(performance) if hasattr(performance, '__dataclass_fields__') else {'sharpe_ratio': getattr(performance, 'sharpe_ratio', 0), 'total_return': getattr(performance, 'total_return', 0), 'max_drawdown': getattr(performance, 'max_drawdown', 0), 'annualized_return': getattr(performance, 'annualized_return', 0), 'calmar_ratio': getattr(performance, 'calmar_ratio', 0)}

        # 交易分析：优先使用回合数据 round_trips，得到准确胜率、盈亏比、持仓时长
        trade_analyzer = TradeAnalyzer()
        round_trips_df = results.get('round_trips')
        if round_trips_df is not None and not round_trips_df.empty:
            try:
                trade_analyzer.add_trades_from_dataframe(round_trips_df)
            except Exception as e:
                logger.warning(f"回合数据接入 TradeAnalyzer 失败: {e}")
        else:
            trades_df = results.get('trades')
            if trades_df is not None and not trades_df.empty:
                if 'entry_time' not in trades_df.columns and 'timestamp' in trades_df.columns:
                    trades_df = trades_df.rename(columns={'timestamp': 'entry_time'})
                    trades_df['exit_time'] = trades_df['entry_time']
                    trades_df['entry_price'] = trades_df.get('price', 0)
                    trades_df['exit_price'] = trades_df.get('price', 0)
                if 'entry_time' in trades_df.columns and 'entry_price' in trades_df.columns:
                    try:
                        trade_analyzer.add_trades_from_dataframe(trades_df)
                    except Exception:
                        pass
        trade_stats_obj = trade_analyzer.analyze()
        trade_stats = vars(trade_stats_obj) if hasattr(trade_stats_obj, '__dataclass_fields__') else {}

        # 生成报告并写入文件（参数搜索时可跳过）
        report_name = f'backtest_{start_date}_{end_date}.html'
        if report_suffix:
            report_name = f'backtest_{start_date}_{end_date}_{report_suffix}.html'
        report_path = str(PROJECT_ROOT / 'reports' / report_name)
        if save_report:
            report_generator = ReportGenerator(str(PROJECT_ROOT / 'reports'))
            html_content = report_generator.generate_html_report(
                performance_dict, trade_stats, None, None, None
            )
            Path(report_path).write_text(html_content, encoding='utf-8')
            logger.info(f"报告已保存: {report_path}")

        logger.info(f"回测完成，夏普比率: {performance_dict.get('sharpe_ratio', 0):.4f}")

        return {
            'performance': performance_dict,
            'trade_stats': trade_stats,
            'results': results,
            'report_path': str(report_path)
        }
    
    async def run_live(self):
        """运行实盘交易"""
        logger.info("启动实盘交易...")
        self.is_running = True
        
        while self.is_running:
            try:
                # 1. 生成信号
                signals = await self.generate_signals()
                
                # 2. 执行信号
                await self.execute_signals(signals)
                
                # 3. 更新持仓
                self.positions = await self.execution_engine.get_positions()
                
                # 4. 风控监控
                risk_status = self.risk_manager.monitor_risk(
                    positions=self.positions,
                    account_info=await self.execution_engine.get_account_info()
                )
                
                if risk_status['alert_level'] == 'critical':
                    logger.warning("触发风控警报，暂停交易")
                    await self.emergency_close_all()
                    break
                
                # 等待下一分钟
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                await asyncio.sleep(60)
    
    async def emergency_close_all(self):
        """紧急平仓"""
        logger.warning("执行紧急平仓...")
        for symbol, position in self.positions.items():
            if position['amount'] != 0:
                side = 'sell' if position['amount'] > 0 else 'buy'
                await self.execution_engine.place_order(
                    symbol=symbol,
                    side=side,
                    amount=abs(position['amount']),
                    order_type='market'
                )
    
    def stop(self):
        """停止交易系统"""
        logger.info("停止交易系统...")
        self.is_running = False


async def main():
    """主函数"""
    # 创建日志目录
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'reports').mkdir(exist_ok=True)
    
    # 初始化系统
    system = CryptoQuantSystem()
    await system.initialize()
    
    compare_inverse = os.environ.get('COMPARE_INVERSE', '').strip().lower() in ('1', 'true', 'yes')
    # 默认回测区间：过去约 30 天，与 data/csv 五品种一个月数据一致
    _end = datetime.now(timezone.utc)
    _start = _end - timedelta(days=30)
    start_date = os.environ.get('BACKTEST_START') or _start.strftime('%Y-%m-%d')
    end_date = os.environ.get('BACKTEST_END') or _end.strftime('%Y-%m-%d')

    if compare_inverse:
        # 仅用本地 CSV 跑两版：原多空 vs 反向，同一数据同一区间对比
        training_result = await system.train_model(days=1)
        print("\n[1/2] 原方向（金叉开多、死叉平多）...")
        result_original = await system.run_backtest(
            start_date=start_date,
            end_date=end_date,
            strategy_params={'inverse_strategy': False},
            use_local_data_only=True,
            report_suffix='original'
        )
        print("[2/2] 反向（金叉开空、死叉平空）...")
        result_inverse = await system.run_backtest(
            start_date=start_date,
            end_date=end_date,
            strategy_params={'inverse_strategy': True},
            use_local_data_only=True,
            report_suffix='inverse'
        )
        p0 = result_original.get('performance') or {}
        p1 = result_inverse.get('performance') or {}
        t0 = result_original.get('trade_stats') or {}
        t1 = result_inverse.get('trade_stats') or {}
        print("\n" + "=" * 70)
        print("策略对比（数据: 仅本地 CSV 2026-03-01 ~ 2026-03-05）")
        print("=" * 70)
        print(f"{'指标':<20} {'原方向(多)':>18} {'反向(空)':>18} {'镜像关系':>14}")
        print("-" * 70)
        sr0 = p0.get('sharpe_ratio') or 0
        sr1 = p1.get('sharpe_ratio') or 0
        print(f"{'夏普比率':<20} {sr0:>18.4f} {sr1:>18.4f} {'≈ -原' if abs(sr1 + sr0) < 1.5 else '':>14}")
        r0 = (p0.get('total_return') or 0) * 100
        r1 = (p1.get('total_return') or 0) * 100
        print(f"{'总收益率 %':<20} {r0:>17.2f}% {r1:>17.2f}% {'≈ -原' if abs(r1 + r0) < 3 else '':>14}")
        d0 = (p0.get('max_drawdown') or 0) * 100
        d1 = (p1.get('max_drawdown') or 0) * 100
        print(f"{'最大回撤 %':<20} {d0:>17.2f}% {d1:>17.2f}%")
        print(f"{'总交易次数':<20} {t0.get('total_trades', 0):>18} {t1.get('total_trades', 0):>18}")
        print("=" * 70)
        print(f"报告: reports/backtest_{start_date}_{end_date}_original.html")
        print(f"      reports/backtest_{start_date}_{end_date}_inverse.html")
        return

    # 训练模型（演示用较少天数，便于本地快速跑通）
    training_result = await system.train_model(days=1)
    
    # 运行回测（若有 data/csv 下的 OKX 数据则用 2026-03-01~05，否则用模拟数据）
    backtest_result = await system.run_backtest(
        start_date=start_date,
        end_date=end_date
    )
    
    # 输出结果
    performance = backtest_result['performance']
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"总收益率: {performance.get('total_return', 0)*100:.2f}%")
    print(f"年化收益率: {performance.get('annualized_return', performance.get('annual_return', 0))*100:.2f}%")
    print(f"夏普比率: {performance.get('sharpe_ratio', 0):.4f}")
    print(f"最大回撤: {performance.get('max_drawdown', 0)*100:.2f}%")
    print(f"Calmar比率: {performance.get('calmar_ratio', 0):.4f}")
    print("=" * 60)
    
    # 如果夏普 > 4，可以启动实盘
    if performance.get('sharpe_ratio', 0) > 4:
        print("\n夏普比率达标 (>4)，可以启动实盘交易！")
    else:
        print("\n夏普比率未达标，建议继续优化策略")


if __name__ == '__main__':
    asyncio.run(main())
