"""
组合风险管理模块

提供多币种相关性管理、行业分散、杠杆控制、保证金监控等功能
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ContractType(Enum):
    """合约类型"""
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    MARGIN = "margin"


@dataclass
class Position:
    """持仓信息"""
    symbol: str  # 交易对
    size: float  # 持仓数量
    entry_price: float  # 入场价格
    current_price: float  # 当前价格
    side: str = "long"  # 方向: long, short
    contract_type: ContractType = ContractType.SPOT
    leverage: float = 1.0
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    sector: str = "other"  # 行业分类


@dataclass
class MarginInfo:
    """保证金信息"""
    total_margin: float  # 总保证金
    used_margin: float  # 已用保证金
    free_margin: float  # 可用保证金
    margin_level: float  # 保证金水平
    maintenance_margin: float  # 维持保证金
    liquidation_price: Optional[float] = None  # 强平价格
    margin_ratio: float = 0.0  # 保证金比率


@dataclass
class PortfolioRiskMetrics:
    """组合风险指标"""
    total_exposure: float  # 总敞口
    net_exposure: float  # 净敞口
    gross_exposure: float  # 总敞口（绝对值）
    concentration_risk: float  # 集中度风险
    correlation_risk: float  # 相关性风险
    sector_exposure: Dict[str, float]  # 行业敞口
    var_95: float  # 组合VaR
    expected_shortfall: float  # 预期亏损
    portfolio_beta: float  # 组合Beta


class PortfolioRiskManager:
    """组合风险管理器"""
    
    def __init__(
        self,
        account_balance: float = 10000.0,
        max_total_leverage: float = 3.0,
        max_single_position_pct: float = 0.3,
        max_sector_exposure_pct: float = 0.5,
        max_correlation_exposure: float = 0.6
    ):
        """
        初始化组合风险管理器
        
        Args:
            account_balance: 账户余额
            max_total_leverage: 最大总杠杆
            max_single_position_pct: 最大单币种仓位比例
            max_sector_exposure_pct: 最大行业敞口比例
            max_correlation_exposure: 最大相关性敞口
        """
        self.account_balance = account_balance
        self.max_total_leverage = max_total_leverage
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_correlation_exposure = max_correlation_exposure
        
        # 持仓
        self.positions: Dict[str, Position] = {}
        
        # 行业分类
        self.sector_map: Dict[str, str] = {}
        
        # 相关性矩阵
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # 保证金信息
        self.margin_info = MarginInfo(
            total_margin=account_balance,
            used_margin=0.0,
            free_margin=account_balance,
            margin_level=0.0,
            maintenance_margin=0.0
        )
        
        # 风险限制
        self.position_limits: Dict[str, float] = {}
        self.sector_limits: Dict[str, float] = {}
    
    def add_position(self, position: Position):
        """添加持仓"""
        self.positions[position.symbol] = position
        self._update_margin_info()
    
    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_margin_info()
    
    def update_position_price(self, symbol: str, current_price: float):
        """更新持仓价格"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            
            # 计算未实现盈亏
            if pos.side == "long":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
            
            self._update_margin_info()
    
    def set_sector_map(self, sector_map: Dict[str, str]):
        """设置行业分类映射"""
        self.sector_map = sector_map
    
    def set_correlation_matrix(self, corr_matrix: pd.DataFrame):
        """设置相关性矩阵"""
        self.correlation_matrix = corr_matrix
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            相关性矩阵
        """
        corr = returns_df.corr()
        self.correlation_matrix = corr
        return corr
    
    def _update_margin_info(self):
        """更新保证金信息"""
        total_margin = self.account_balance
        used_margin = 0.0
        maintenance_margin = 0.0
        
        for pos in self.positions.values():
            notional = abs(pos.size) * pos.current_price
            
            if pos.contract_type != ContractType.SPOT:
                # 合约仓位
                pos.margin = notional / pos.leverage
                used_margin += pos.margin
                
                # 维持保证金（假设0.5%）
                maintenance_margin += notional * 0.005
        
        free_margin = total_margin - used_margin
        margin_level = (total_margin / used_margin * 100) if used_margin > 0 else float('inf')
        margin_ratio = used_margin / total_margin if total_margin > 0 else 0
        
        self.margin_info = MarginInfo(
            total_margin=total_margin,
            used_margin=used_margin,
            free_margin=free_margin,
            margin_level=margin_level,
            maintenance_margin=maintenance_margin,
            margin_ratio=margin_ratio
        )
    
    def get_position_exposure(self, symbol: str) -> float:
        """
        获取单个持仓敞口
        
        Args:
            symbol: 交易对
            
        Returns:
            敞口金额
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        return abs(pos.size) * pos.current_price
    
    def get_total_exposure(self) -> float:
        """获取总敞口"""
        return sum(self.get_position_exposure(sym) for sym in self.positions)
    
    def get_net_exposure(self) -> float:
        """获取净敞口"""
        net = 0.0
        for pos in self.positions.values():
            exposure = pos.size * pos.current_price
            if pos.side == "short":
                exposure = -exposure
            net += exposure
        return net
    
    def get_gross_exposure(self) -> float:
        """获取总敞口（绝对值）"""
        return sum(abs(pos.size) * pos.current_price for pos in self.positions.values())
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """获取各行业敞口"""
        sector_exp = {}
        
        for symbol, pos in self.positions.items():
            sector = self.sector_map.get(symbol, "other")
            exposure = abs(pos.size) * pos.current_price
            
            if sector not in sector_exp:
                sector_exp[sector] = 0.0
            sector_exp[sector] += exposure
        
        return sector_exp
    
    def get_concentration_risk(self) -> float:
        """
        计算集中度风险（赫芬达尔指数）
        
        Returns:
            集中度风险 (0-1)
        """
        total = self.get_total_exposure()
        
        if total == 0:
            return 0.0
        
        # 计算各持仓权重
        weights = []
        for pos in self.positions.values():
            weight = (abs(pos.size) * pos.current_price) / total
            weights.append(weight)
        
        # 赫芬达尔指数
        hhi = sum(w ** 2 for w in weights)
        
        return hhi
    
    def get_correlation_risk(self) -> float:
        """
        计算相关性风险
        
        Returns:
            相关性风险
        """
        if self.correlation_matrix is None or len(self.positions) < 2:
            return 0.0
        
        # 获取持仓的相关性
        symbols = list(self.positions.keys())
        available_symbols = [s for s in symbols if s in self.correlation_matrix.columns]
        
        if len(available_symbols) < 2:
            return 0.0
        
        # 计算平均相关性
        corr_subset = self.correlation_matrix.loc[available_symbols, available_symbols]
        
        # 排除对角线
        mask = ~np.eye(len(available_symbols), dtype=bool)
        avg_corr = corr_subset.values[mask].mean()
        
        return avg_corr
    
    def get_portfolio_var(
        self,
        returns_df: pd.DataFrame,
        confidence: float = 0.95
    ) -> float:
        """
        计算组合VaR
        
        Args:
            returns_df: 收益率数据框
            confidence: 置信水平
            
        Returns:
            组合VaR
        """
        if len(self.positions) == 0:
            return 0.0
        
        # 计算持仓权重
        weights = []
        available_symbols = []
        
        for symbol in returns_df.columns:
            if symbol in self.positions:
                pos = self.positions[symbol]
                weight = (pos.size * pos.current_price) / self.account_balance
                weights.append(weight)
                available_symbols.append(symbol)
            else:
                weights.append(0.0)
        
        weights = np.array(weights)
        
        # 计算组合收益率
        portfolio_returns = returns_df.dot(weights)
        
        # 计算VaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        return var * self.account_balance
    
    def get_portfolio_beta(self, market_returns: pd.Series) -> float:
        """
        计算组合Beta
        
        Args:
            market_returns: 市场收益率
            
        Returns:
            组合Beta
        """
        if len(self.positions) == 0:
            return 1.0
        
        # 简化计算：按市值加权平均
        total_exposure = self.get_total_exposure()
        
        if total_exposure == 0:
            return 1.0
        
        # 这里简化处理，实际应该计算各资产Beta后加权
        return 1.0
    
    def check_position_limit(self, symbol: str) -> Tuple[bool, str]:
        """
        检查持仓限制
        
        Args:
            symbol: 交易对
            
        Returns:
            (是否通过, 原因)
        """
        exposure = self.get_position_exposure(symbol)
        exposure_pct = exposure / self.account_balance if self.account_balance > 0 else 0
        
        if exposure_pct > self.max_single_position_pct:
            return False, f"持仓{symbol}超过限制: {exposure_pct*100:.1f}% > {self.max_single_position_pct*100:.1f}%"
        
        return True, "通过"
    
    def check_leverage_limit(self) -> Tuple[bool, str]:
        """
        检查杠杆限制
        
        Returns:
            (是否通过, 原因)
        """
        total_exposure = self.get_total_exposure()
        leverage = total_exposure / self.account_balance if self.account_balance > 0 else 0
        
        if leverage > self.max_total_leverage:
            return False, f"总杠杆超过限制: {leverage:.2f}x > {self.max_total_leverage:.2f}x"
        
        return True, "通过"
    
    def check_sector_limit(self) -> Tuple[bool, str]:
        """
        检查行业敞口限制
        
        Returns:
            (是否通过, 原因)
        """
        sector_exp = self.get_sector_exposure()
        
        for sector, exposure in sector_exp.items():
            exposure_pct = exposure / self.account_balance if self.account_balance > 0 else 0
            
            if exposure_pct > self.max_sector_exposure_pct:
                return False, f"行业{sector}敞口超过限制: {exposure_pct*100:.1f}% > {self.max_sector_exposure_pct*100:.1f}%"
        
        return True, "通过"
    
    def check_correlation_limit(self) -> Tuple[bool, str]:
        """
        检查相关性限制
        
        Returns:
            (是否通过, 原因)
        """
        corr_risk = self.get_correlation_risk()
        
        if corr_risk > self.max_correlation_exposure:
            return False, f"相关性风险超过限制: {corr_risk:.2f} > {self.max_correlation_exposure:.2f}"
        
        return True, "通过"
    
    def check_margin_safety(self) -> Tuple[bool, str]:
        """
        检查保证金安全
        
        Returns:
            (是否通过, 原因)
        """
        # 如果没有合约持仓，保证金检查通过
        if self.margin_info.used_margin == 0:
            return True, "通过"
        
        # 检查保证金水平
        if self.margin_info.margin_level < 110:
            return False, f"保证金水平过低: {self.margin_info.margin_level:.1f}% < 110%"
        
        if self.margin_info.free_margin < self.margin_info.maintenance_margin:
            return False, "可用保证金低于维持保证金"
        
        return True, "通过"
    
    def check_all_limits(self) -> Dict[str, Tuple[bool, str]]:
        """
        检查所有限制
        
        Returns:
            检查结果字典
        """
        results = {}
        
        # 检查杠杆
        results['leverage'] = self.check_leverage_limit()
        
        # 检查各持仓
        for symbol in self.positions:
            results[f'position_{symbol}'] = self.check_position_limit(symbol)
        
        # 检查行业
        results['sector'] = self.check_sector_limit()
        
        # 检查相关性
        results['correlation'] = self.check_correlation_limit()
        
        # 检查保证金
        results['margin'] = self.check_margin_safety()
        
        return results
    
    def get_portfolio_metrics(self, returns_df: Optional[pd.DataFrame] = None) -> PortfolioRiskMetrics:
        """
        获取组合风险指标
        
        Args:
            returns_df: 收益率数据框
            
        Returns:
            组合风险指标
        """
        # 计算VaR
        var_95 = 0.0
        expected_shortfall = 0.0
        if returns_df is not None:
            var_95 = self.get_portfolio_var(returns_df, 0.95)
            expected_shortfall = self.get_portfolio_var(returns_df, 0.99)
        
        return PortfolioRiskMetrics(
            total_exposure=self.get_total_exposure(),
            net_exposure=self.get_net_exposure(),
            gross_exposure=self.get_gross_exposure(),
            concentration_risk=self.get_concentration_risk(),
            correlation_risk=self.get_correlation_risk(),
            sector_exposure=self.get_sector_exposure(),
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            portfolio_beta=self.get_portfolio_beta(pd.Series())
        )
    
    def get_margin_report(self) -> Dict:
        """
        获取保证金报告
        
        Returns:
            保证金报告
        """
        return {
            'total_margin': self.margin_info.total_margin,
            'used_margin': self.margin_info.used_margin,
            'free_margin': self.margin_info.free_margin,
            'margin_level': self.margin_info.margin_level,
            'margin_ratio': self.margin_info.margin_ratio,
            'maintenance_margin': self.margin_info.maintenance_margin,
            'is_safe': self.margin_info.margin_level > 150
        }
    
    def get_portfolio_report(self) -> Dict:
        """
        获取组合报告
        
        Returns:
            组合报告
        """
        metrics = self.get_portfolio_metrics()
        
        return {
            'total_positions': len(self.positions),
            'total_exposure': metrics.total_exposure,
            'net_exposure': metrics.net_exposure,
            'gross_exposure': metrics.gross_exposure,
            'concentration_risk': metrics.concentration_risk,
            'correlation_risk': metrics.correlation_risk,
            'sector_exposure': metrics.sector_exposure,
            'margin_report': self.get_margin_report(),
            'limit_checks': self.check_all_limits()
        }
    
    def suggest_position_adjustments(self) -> List[Dict]:
        """
        建议仓位调整
        
        Returns:
            调整建议列表
        """
        suggestions = []
        
        # 检查集中度
        concentration = self.get_concentration_risk()
        if concentration > 0.3:  # 高集中度
            # 找出最大持仓
            max_symbol = max(
                self.positions.keys(),
                key=lambda s: self.get_position_exposure(s)
            )
            suggestions.append({
                'type': 'reduce_concentration',
                'symbol': max_symbol,
                'message': f'集中度风险较高，建议减少{max_symbol}持仓'
            })
        
        # 检查行业敞口
        sector_exp = self.get_sector_exposure()
        for sector, exposure in sector_exp.items():
            exposure_pct = exposure / self.account_balance if self.account_balance > 0 else 0
            if exposure_pct > self.max_sector_exposure_pct:
                suggestions.append({
                    'type': 'reduce_sector',
                    'sector': sector,
                    'message': f'行业{sector}敞口过高，建议分散投资'
                })
        
        # 检查杠杆
        leverage_check = self.check_leverage_limit()
        if not leverage_check[0]:
            suggestions.append({
                'type': 'reduce_leverage',
                'message': '总杠杆过高，建议降低仓位'
            })
        
        return suggestions


class CrossExchangePortfolioRiskManager(PortfolioRiskManager):
    """跨交易所组合风险管理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 按交易所分组
        self.exchange_positions: Dict[str, Dict[str, Position]] = {}
        self.exchange_balances: Dict[str, float] = {}
    
    def add_exchange(self, exchange: str, balance: float):
        """添加交易所"""
        self.exchange_positions[exchange] = {}
        self.exchange_balances[exchange] = balance
    
    def add_position_to_exchange(self, exchange: str, position: Position):
        """向交易所添加持仓"""
        if exchange not in self.exchange_positions:
            self.add_exchange(exchange, 0.0)
        
        self.exchange_positions[exchange][position.symbol] = position
        
        # 同时更新总持仓
        self.positions[position.symbol] = position
    
    def get_exchange_exposure(self, exchange: str) -> float:
        """获取交易所敞口"""
        if exchange not in self.exchange_positions:
            return 0.0
        
        return sum(
            abs(pos.size) * pos.current_price
            for pos in self.exchange_positions[exchange].values()
        )
    
    def get_exchange_report(self) -> Dict[str, Dict]:
        """获取各交易所报告"""
        reports = {}
        
        for exchange in self.exchange_positions:
            exposure = self.get_exchange_exposure(exchange)
            balance = self.exchange_balances.get(exchange, 0.0)
            
            reports[exchange] = {
                'balance': balance,
                'exposure': exposure,
                'leverage': exposure / balance if balance > 0 else 0,
                'positions': len(self.exchange_positions[exchange])
            }
        
        return reports


# 便捷函数
def calculate_portfolio_weights(positions: Dict[str, float], total_value: float) -> Dict[str, float]:
    """
    计算组合权重
    
    Args:
        positions: 持仓价值字典
        total_value: 总价值
        
    Returns:
        权重字典
    """
    if total_value == 0:
        return {k: 0.0 for k in positions}
    
    return {k: v / total_value for k, v in positions.items()}


def check_correlation_clustering(correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Set[str]]:
    """
    检查相关性聚类
    
    Args:
        correlation_matrix: 相关性矩阵
        threshold: 相关性阈值
        
    Returns:
        高相关性资产组列表
    """
    clusters = []
    visited = set()
    
    for symbol in correlation_matrix.columns:
        if symbol in visited:
            continue
        
        # 找到高相关性资产
        cluster = {symbol}
        visited.add(symbol)
        
        for other in correlation_matrix.columns:
            if other in visited:
                continue
            
            if abs(correlation_matrix.loc[symbol, other]) >= threshold:
                cluster.add(other)
                visited.add(other)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters
