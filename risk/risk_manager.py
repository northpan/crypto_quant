"""
风控管理器主模块

整合所有风控功能，提供统一的风控管理接口
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# 导入子模块
from .metrics.risk_metrics import RiskMetricsCalculator, RiskMetricsResult, quick_metrics
from .position.position_sizer import (
    PositionSizer, DynamicPositionSizer, PositionSize, PositionSizingMethod
)

# 使用DynamicPositionSizer作为默认仓位管理器
PositionSizer = DynamicPositionSizer
from .stop_loss.stop_manager import (
    StopManager, AdvancedStopManager, StopLevel, TakeProfitLevel, PositionStops,
    StopType, TakeProfitType, quick_stop_loss, quick_take_profit
)
from .drawdown.drawdown_controller import (
    DrawdownController, DynamicDrawdownController, MultiTierDrawdownController,
    DrawdownAction, DrawdownLevel, DrawdownState, calculate_drawdown
)
from .portfolio.portfolio_risk import (
    PortfolioRiskManager, CrossExchangePortfolioRiskManager,
    Position, MarginInfo, PortfolioRiskMetrics, ContractType
)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskCheckResult:
    """风险检查结果"""
    passed: bool
    risk_level: RiskLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeApproval:
    """交易审批结果"""
    approved: bool
    position_size: float
    max_leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: float = 0.0
    message: str = ""


class RiskManager:
    """风控管理器主类"""
    
    def __init__(
        self,
        account_balance: float = 10000.0,
        risk_mode: str = "moderate",  # conservative, moderate, aggressive
        contract_type: str = "spot"
    ):
        """
        初始化风控管理器
        
        Args:
            account_balance: 账户余额
            risk_mode: 风险模式
            contract_type: 合约类型
        """
        self.account_balance = account_balance
        self.risk_mode = risk_mode
        self.contract_type = contract_type
        
        # 初始化各子模块
        self._init_modules()
        
        # 设置风险参数
        self._setup_risk_parameters()
        
        # 风险事件回调
        self.risk_callbacks: Dict[str, List[Callable]] = {
            'warning': [],
            'limit_breach': [],
            'emergency': []
        }
        
        # 历史记录
        self.risk_events: List[Dict] = []
        self.daily_pnl: List[Tuple[datetime, float]] = []
        
        # 状态
        self.is_initialized = True
        self.trading_enabled = True
    
    def _init_modules(self):
        """初始化各风控模块"""
        # 风险指标计算器
        self.metrics_calculator = RiskMetricsCalculator()
        
        # 仓位管理器
        self.position_sizer = PositionSizer(
            account_balance=self.account_balance,
            contract_type=self.contract_type
        )
        
        # 止损止盈管理器
        self.stop_manager = AdvancedStopManager()
        
        # 回撤控制器
        self.drawdown_controller = DynamicDrawdownController()
        
        # 组合风险管理器
        self.portfolio_manager = PortfolioRiskManager(
            account_balance=self.account_balance
        )
    
    def _setup_risk_parameters(self):
        """设置风险参数"""
        # 根据风险模式设置参数
        risk_params = {
            'conservative': {
                'max_drawdown': 0.10,
                'daily_drawdown': 0.05,
                'max_leverage': 2.0,
                'max_position_pct': 0.20,
                'risk_per_trade': 0.01,
                'var_limit': 0.02
            },
            'moderate': {
                'max_drawdown': 0.20,
                'daily_drawdown': 0.10,
                'max_leverage': 5.0,
                'max_position_pct': 0.30,
                'risk_per_trade': 0.02,
                'var_limit': 0.05
            },
            'aggressive': {
                'max_drawdown': 0.30,
                'daily_drawdown': 0.15,
                'max_leverage': 10.0,
                'max_position_pct': 0.50,
                'risk_per_trade': 0.03,
                'var_limit': 0.08
            }
        }
        
        params = risk_params.get(self.risk_mode, risk_params['moderate'])
        
        # 应用参数
        self.max_drawdown = params['max_drawdown']
        self.daily_drawdown = params['daily_drawdown']
        self.max_leverage = params['max_leverage']
        self.max_position_pct = params['max_position_pct']
        self.risk_per_trade = params['risk_per_trade']
        self.var_limit = params['var_limit']
        
        # 更新各模块参数
        self.position_sizer.max_leverage = self.max_leverage
        self.position_sizer.max_position_pct = self.max_position_pct
        self.position_sizer.risk_per_trade = self.risk_per_trade
        
        self.drawdown_controller.max_drawdown_limit = self.max_drawdown
        self.drawdown_controller.daily_drawdown_limit = self.daily_drawdown
        
        self.portfolio_manager.max_total_leverage = self.max_leverage
        self.portfolio_manager.max_single_position_pct = self.max_position_pct
    
    def update_account_balance(self, balance: float):
        """更新账户余额"""
        self.account_balance = balance
        self.position_sizer.set_account_balance(balance)
        self.portfolio_manager.account_balance = balance
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """更新权益"""
        self.drawdown_controller.update_equity(equity, timestamp)
        self.position_sizer.update_equity(equity)
    
    def check_trade_risk(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        position_size: float,
        stop_loss: Optional[float] = None,
        leverage: float = 1.0
    ) -> RiskCheckResult:
        """
        检查交易风险
        
        Args:
            symbol: 交易对
            side: 方向 (long/short)
            entry_price: 入场价格
            position_size: 仓位大小
            stop_loss: 止损价格
            leverage: 杠杆倍数
            
        Returns:
            风险检查结果
        """
        details = {}
        
        # 1. 检查回撤限制
        allowed, reason = self.drawdown_controller.check_trading_allowed()
        details['drawdown_check'] = {'passed': allowed, 'reason': reason}
        
        if not allowed:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                message=reason,
                details=details
            )
        
        # 2. 检查杠杆限制
        notional = position_size * entry_price
        trade_leverage = notional / self.account_balance if self.account_balance > 0 else 0
        
        if trade_leverage > self.max_leverage:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                message=f"杠杆超过限制: {trade_leverage:.2f}x > {self.max_leverage:.2f}x",
                details=details
            )
        
        # 3. 检查仓位限制
        position_pct = notional / self.account_balance if self.account_balance > 0 else 0
        
        if position_pct > self.max_position_pct:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                message=f"仓位比例超过限制: {position_pct*100:.1f}% > {self.max_position_pct*100:.1f}%",
                details=details
            )
        
        # 4. 检查风险金额
        if stop_loss:
            if side == "long":
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price
            
            risk_amount = abs(risk_per_unit * position_size)
            risk_pct = risk_amount / self.account_balance if self.account_balance > 0 else 0
            
            details['risk_amount'] = risk_amount
            details['risk_pct'] = risk_pct
            
            if risk_pct > self.risk_per_trade:
                return RiskCheckResult(
                    passed=False,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"单笔风险超过限制: {risk_pct*100:.2f}% > {self.risk_per_trade*100:.2f}%",
                    details=details
                )
        
        # 5. 检查组合限制
        limit_checks = self.portfolio_manager.check_all_limits()
        details['portfolio_checks'] = limit_checks
        
        failed_checks = [k for k, v in limit_checks.items() if not v[0]]
        if failed_checks:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                message=f"组合限制检查失败: {failed_checks}",
                details=details
            )
        
        return RiskCheckResult(
            passed=True,
            risk_level=RiskLevel.LOW,
            message="风险检查通过",
            details=details
        )
    
    def approve_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_reward: float = 2.0,
        method: PositionSizingMethod = PositionSizingMethod.PERCENT_RISK
    ) -> TradeApproval:
        """
        审批交易
        
        Args:
            symbol: 交易对
            side: 方向
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            risk_reward: 风险回报比
            method: 仓位计算方法
            
        Returns:
            交易审批结果
        """
        # 计算止损价格
        if stop_loss is None:
            stop_pct = 0.05
            if side == "long":
                stop_loss = entry_price * (1 - stop_pct)
            else:
                stop_loss = entry_price * (1 + stop_pct)
        
        # 计算仓位
        position_size_result = self.position_sizer.calculate_position_size(
            method=method,
            current_price=entry_price,
            stop_price=stop_loss
        )
        
        position_size = position_size_result.size_in_units
        
        # 检查风险
        risk_check = self.check_trade_risk(
            symbol, side, entry_price, position_size, stop_loss
        )
        
        if not risk_check.passed:
            return TradeApproval(
                approved=False,
                position_size=0,
                max_leverage=0,
                message=risk_check.message
            )
        
        # 计算止盈
        if take_profit is None:
            if side == "long":
                risk = entry_price - stop_loss
                take_profit = entry_price + risk * risk_reward
            else:
                risk = stop_loss - entry_price
                take_profit = entry_price - risk * risk_reward
        
        return TradeApproval(
            approved=True,
            position_size=position_size,
            max_leverage=position_size_result.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=position_size_result.risk_amount,
            message="交易已批准"
        )
    
    def register_position(
        self,
        position_id: str,
        symbol: str,
        entry_price: float,
        position_size: float,
        side: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> PositionStops:
        """
        注册持仓
        
        Args:
            position_id: 持仓ID
            symbol: 交易对
            entry_price: 入场价格
            position_size: 仓位大小
            side: 方向
            stop_loss: 止损价格
            take_profit: 止盈价格
            
        Returns:
            持仓止损止盈设置
        """
        is_long = side == "long"
        
        # 注册到止损管理器
        pos_stops = self.stop_manager.register_position(
            position_id, entry_price, position_size, is_long
        )
        
        # 设置止损
        if stop_loss:
            self.stop_manager.set_fixed_stop(position_id, stop_loss)
        
        # 设置止盈
        if take_profit:
            self.stop_manager.set_fixed_take_profit(position_id, take_profit)
        
        # 注册到组合管理器
        position = Position(
            symbol=symbol,
            size=position_size if is_long else -position_size,
            entry_price=entry_price,
            current_price=entry_price,
            side=side
        )
        self.portfolio_manager.add_position(position)
        
        return pos_stops
    
    def update_position(self, position_id: str, current_price: float):
        """更新持仓价格"""
        # 更新组合管理器
        if position_id in self.portfolio_manager.positions:
            self.portfolio_manager.update_position_price(position_id, current_price)
        
        # 检查止损止盈
        stop, tp = self.stop_manager.check_all_stops(position_id, current_price)
        
        if stop:
            self._handle_stop_triggered(position_id, stop)
        
        if tp:
            self._handle_tp_triggered(position_id, tp)
    
    def _handle_stop_triggered(self, position_id: str, stop: StopLevel):
        """处理止损触发"""
        event = {
            'type': 'stop_loss',
            'position_id': position_id,
            'trigger_price': stop.trigger_price,
            'stop_price': stop.price,
            'timestamp': datetime.now()
        }
        self.risk_events.append(event)
        
        # 触发回调
        for callback in self.risk_callbacks['limit_breach']:
            callback(event)
    
    def _handle_tp_triggered(self, position_id: str, tp: TakeProfitLevel):
        """处理止盈触发"""
        event = {
            'type': 'take_profit',
            'position_id': position_id,
            'trigger_price': tp.trigger_price,
            'tp_price': tp.price,
            'timestamp': datetime.now()
        }
        self.risk_events.append(event)
    
    def close_position(self, position_id: str):
        """平仓"""
        self.stop_manager.remove_position(position_id)
        
        if position_id in self.portfolio_manager.positions:
            self.portfolio_manager.remove_position(position_id)
    
    def calculate_portfolio_metrics(self, returns_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算组合风险指标
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            风险指标字典
        """
        metrics = self.portfolio_manager.get_portfolio_metrics(returns_df)
        
        return {
            'total_exposure': metrics.total_exposure,
            'net_exposure': metrics.net_exposure,
            'gross_exposure': metrics.gross_exposure,
            'concentration_risk': metrics.concentration_risk,
            'correlation_risk': metrics.correlation_risk,
            'sector_exposure': metrics.sector_exposure,
            'var_95': metrics.var_95,
            'expected_shortfall': metrics.expected_shortfall
        }
    
    def get_risk_report(self) -> Dict:
        """
        获取风险报告
        
        Returns:
            风险报告
        """
        return {
            'account_balance': self.account_balance,
            'risk_mode': self.risk_mode,
            'trading_enabled': self.trading_enabled,
            'drawdown': self.drawdown_controller.get_drawdown_report(),
            'portfolio': self.portfolio_manager.get_portfolio_report(),
            'margin': self.portfolio_manager.get_margin_report(),
            'position_scale': self.drawdown_controller.get_position_scale()
        }
    
    def generate_daily_report(self) -> str:
        """
        生成日报
        
        Returns:
            报告文本
        """
        report = self.get_risk_report()
        
        lines = [
            "=" * 50,
            "风险日报",
            "=" * 50,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"账户余额: {report['account_balance']:,.2f}",
            f"风险模式: {report['risk_mode']}",
            f"交易状态: {'启用' if report['trading_enabled'] else '禁用'}",
            "",
            "--- 回撤信息 ---",
            f"当前回撤: {report['drawdown']['current_drawdown']*100:.2f}%",
            f"最大回撤: {report['drawdown']['max_drawdown']*100:.2f}%",
            f"日回撤: {report['drawdown']['daily_drawdown']*100:.2f}%",
            f"仓位缩放: {report['position_scale']*100:.0f}%",
            "",
            "--- 组合信息 ---",
            f"持仓数量: {report['portfolio']['total_positions']}",
            f"总敞口: {report['portfolio']['total_exposure']:,.2f}",
            f"净敞口: {report['portfolio']['net_exposure']:,.2f}",
            f"集中度风险: {report['portfolio']['concentration_risk']:.3f}",
            "",
            "--- 保证金信息 ---",
            f"总保证金: {report['margin']['total_margin']:,.2f}",
            f"已用保证金: {report['margin']['used_margin']:,.2f}",
            f"可用保证金: {report['margin']['free_margin']:,.2f}",
            f"保证金水平: {report['margin']['margin_level']:.1f}%",
            "",
            "=" * 50
        ]
        
        return "\n".join(lines)
    
    def save_config(self, filepath: str):
        """保存配置"""
        config = {
            'account_balance': self.account_balance,
            'risk_mode': self.risk_mode,
            'contract_type': self.contract_type,
            'max_drawdown': self.max_drawdown,
            'daily_drawdown': self.daily_drawdown,
            'max_leverage': self.max_leverage,
            'max_position_pct': self.max_position_pct,
            'risk_per_trade': self.risk_per_trade
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: str):
        """加载配置"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.account_balance = config.get('account_balance', self.account_balance)
        self.risk_mode = config.get('risk_mode', self.risk_mode)
        self.contract_type = config.get('contract_type', self.contract_type)
        
        self._setup_risk_parameters()
    
    def register_risk_callback(self, event_type: str, callback: Callable):
        """注册风险事件回调"""
        if event_type in self.risk_callbacks:
            self.risk_callbacks[event_type].append(callback)


class MultiAccountRiskManager:
    """多账户风控管理器"""
    
    def __init__(self):
        """初始化多账户风控管理器"""
        self.risk_managers: Dict[str, RiskManager] = {}
        self.total_balance = 0.0
    
    def add_account(
        self,
        account_id: str,
        balance: float,
        risk_mode: str = "moderate"
    ):
        """添加账户"""
        self.risk_managers[account_id] = RiskManager(
            account_balance=balance,
            risk_mode=risk_mode
        )
        self.total_balance += balance
    
    def update_account_balance(self, account_id: str, balance: float):
        """更新账户余额"""
        if account_id in self.risk_managers:
            old_balance = self.risk_managers[account_id].account_balance
            self.risk_managers[account_id].update_account_balance(balance)
            self.total_balance += balance - old_balance
    
    def get_combined_risk_report(self) -> Dict:
        """获取组合风险报告"""
        reports = {}
        total_exposure = 0.0
        
        for account_id, manager in self.risk_managers.items():
            report = manager.get_risk_report()
            reports[account_id] = report
            total_exposure += report['portfolio']['total_exposure']
        
        return {
            'accounts': reports,
            'total_balance': self.total_balance,
            'total_exposure': total_exposure,
            'overall_leverage': total_exposure / self.total_balance if self.total_balance > 0 else 0
        }


# 便捷函数
def create_risk_manager(
    account_balance: float = 10000.0,
    risk_mode: str = "moderate"
) -> RiskManager:
    """
    快速创建风控管理器
    
    Args:
        account_balance: 账户余额
        risk_mode: 风险模式
        
    Returns:
        风控管理器实例
    """
    return RiskManager(account_balance, risk_mode)


def quick_risk_check(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    position_size: float
) -> bool:
    """
    快速风险检查
    
    Args:
        account_balance: 账户余额
        entry_price: 入场价格
        stop_loss: 止损价格
        position_size: 仓位大小
        
    Returns:
        是否通过
    """
    risk_amount = abs(entry_price - stop_loss) * position_size
    risk_pct = risk_amount / account_balance if account_balance > 0 else 1.0
    
    return risk_pct <= 0.02  # 2%风险限制
