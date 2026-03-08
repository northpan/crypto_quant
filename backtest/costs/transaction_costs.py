"""
交易成本模型
包含手续费、滑点、冲击成本和资金费率模型
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeeType(Enum):
    """手续费类型"""
    FIXED = "fixed"           # 固定费率
    TIERED = "tiered"         # 阶梯费率
    MAKER_TAKER = "maker_taker"  # 挂单/吃单费率


class SlippageModel(Enum):
    """滑点模型"""
    FIXED = "fixed"                    # 固定滑点
    PERCENTAGE = "percentage"          # 百分比滑点
    VOLATILITY_BASED = "volatility"    # 基于波动率的滑点
    VOLUME_BASED = "volume"            # 基于成交量的滑点
    SPREAD_BASED = "spread"            # 基于买卖价差的滑点
    ADVANCED = "advanced"              # 高级滑点模型


class ImpactModel(Enum):
    """冲击成本模型"""
    NONE = "none"
    LINEAR = "linear"                  # 线性冲击
    SQUARE_ROOT = "square_root"        # 平方根冲击
    POWER = "power"                    # 幂律冲击
    ALMGREN_CHRISS = "almgren_chriss"  # Almgren-Chriss模型


@dataclass
class FeeStructure:
    """手续费结构"""
    maker_fee: float = 0.001      # 挂单费率
    taker_fee: float = 0.001      # 吃单费率
    withdrawal_fee: float = 0.0   # 提现费
    min_fee: float = 0.0          # 最小手续费
    
    # 阶梯费率设置
    tiered_thresholds: List[float] = None
    tiered_maker_fees: List[float] = None
    tiered_taker_fees: List[float] = None


@dataclass
class SlippageParams:
    """滑点参数"""
    base_slippage: float = 0.0005     # 基础滑点
    volatility_factor: float = 0.1    # 波动率因子
    volume_factor: float = 0.1        # 成交量因子
    spread_factor: float = 0.5        # 价差因子
    max_slippage: float = 0.01        # 最大滑点 (1%)


@dataclass
class ImpactParams:
    """冲击成本参数"""
    eta: float = 0.1                  # 永久冲击系数
    gamma: float = 0.1                # 临时冲击系数
    beta: float = 0.6                 # 幂律指数
    market_depth: float = 1000000.0   # 市场深度


class TransactionCostModel:
    """
    交易成本模型
    
    综合考虑:
    - 手续费 (挂单/吃单)
    - 滑点成本
    - 市场冲击成本
    - 资金费率 (合约)
    """
    
    def __init__(
        self,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: SlippageModel = SlippageModel.FIXED,
        slippage_params: Optional[SlippageParams] = None,
        impact_model: ImpactModel = ImpactModel.SQUARE_ROOT,
        impact_params: Optional[ImpactParams] = None,
        funding_rate: float = 0.0,
        funding_interval: int = 8  # 小时
    ):
        self.fee_structure = fee_structure or FeeStructure()
        self.slippage_model = slippage_model
        self.slippage_params = slippage_params or SlippageParams()
        self.impact_model = impact_model
        self.impact_params = impact_params or ImpactParams()
        self.funding_rate = funding_rate
        self.funding_interval = funding_interval
        
        # 成本统计
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_impact = 0.0
        self.total_funding = 0.0
        self.cost_history: List[Dict] = []
    
    def calculate_fee(
        self,
        notional: float,
        is_maker: bool = False,
        volume_30d: float = 0.0
    ) -> float:
        """
        计算手续费
        
        Args:
            notional: 名义价值
            is_maker: 是否为挂单
            volume_30d: 30天交易量 (用于阶梯费率)
        
        Returns:
            手续费金额
        """
        # 阶梯费率
        if self.fee_structure.tiered_thresholds:
            fee_rate = self._get_tiered_fee(volume_30d, is_maker)
        else:
            fee_rate = self.fee_structure.maker_fee if is_maker else self.fee_structure.taker_fee
        
        fee = notional * fee_rate
        return max(fee, self.fee_structure.min_fee)
    
    def _get_tiered_fee(self, volume_30d: float, is_maker: bool) -> float:
        """获取阶梯费率"""
        thresholds = self.fee_structure.tiered_thresholds
        maker_fees = self.fee_structure.tiered_maker_fees
        taker_fees = self.fee_structure.tiered_taker_fees
        
        if not thresholds or not maker_fees or not taker_fees:
            return self.fee_structure.taker_fee
        
        fees = maker_fees if is_maker else taker_fees
        
        for i, threshold in enumerate(thresholds):
            if volume_30d < threshold:
                return fees[i]
        
        return fees[-1]
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        volume: float,
        volatility: float = 0.0,
        spread: float = 0.0,
        side: str = "buy"
    ) -> float:
        """
        计算滑点
        
        Args:
            price: 当前价格
            quantity: 交易数量
            volume: 市场成交量
            volatility: 波动率 (如ATR/价格)
            spread: 买卖价差
            side: 交易方向 (buy/sell)
        
        Returns:
            滑点金额
        """
        params = self.slippage_params
        
        if self.slippage_model == SlippageModel.FIXED:
            slippage = params.base_slippage
        
        elif self.slippage_model == SlippageModel.PERCENTAGE:
            slippage = price * params.base_slippage
        
        elif self.slippage_model == SlippageModel.VOLATILITY_BASED:
            slippage = price * volatility * params.volatility_factor
        
        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            volume_ratio = min(quantity / max(volume, 1e-10), 1.0)
            slippage = price * params.base_slippage * (1 + volume_ratio * params.volume_factor)
        
        elif self.slippage_model == SlippageModel.SPREAD_BASED:
            slippage = spread * params.spread_factor
        
        elif self.slippage_model == SlippageModel.ADVANCED:
            # 综合模型
            vol_slippage = price * volatility * params.volatility_factor
            volume_ratio = min(quantity / max(volume, 1e-10), 1.0)
            vol_based_slippage = price * params.base_slippage * (1 + volume_ratio * params.volume_factor)
            spread_slippage = spread * params.spread_factor
            
            slippage = max(vol_slippage, vol_based_slippage, spread_slippage)
        
        else:
            slippage = params.base_slippage
        
        # 限制最大滑点
        slippage = min(slippage, price * params.max_slippage)
        
        return slippage
    
    def calculate_impact_cost(
        self,
        price: float,
        quantity: float,
        market_depth: Optional[float] = None,
        volatility: float = 0.0
    ) -> Tuple[float, float]:
        """
        计算市场冲击成本
        
        Args:
            price: 当前价格
            quantity: 交易数量
            market_depth: 市场深度
            volatility: 波动率
        
        Returns:
            (永久冲击成本, 临时冲击成本)
        """
        if market_depth is None:
            market_depth = self.impact_params.market_depth
        
        params = self.impact_params
        
        if self.impact_model == ImpactModel.NONE:
            return 0.0, 0.0
        
        elif self.impact_model == ImpactModel.LINEAR:
            # 线性冲击模型
            participation = quantity / max(market_depth, 1e-10)
            permanent_impact = price * params.eta * participation
            temporary_impact = price * params.gamma * participation
        
        elif self.impact_model == ImpactModel.SQUARE_ROOT:
            # 平方根冲击模型
            participation = quantity / max(market_depth, 1e-10)
            permanent_impact = price * params.eta * np.sqrt(participation)
            temporary_impact = price * params.gamma * np.sqrt(participation)
        
        elif self.impact_model == ImpactModel.POWER:
            # 幂律冲击模型
            participation = quantity / max(market_depth, 1e-10)
            permanent_impact = price * params.eta * (participation ** params.beta)
            temporary_impact = price * params.gamma * (participation ** params.beta)
        
        elif self.impact_model == ImpactModel.ALMGREN_CHRISS:
            # Almgren-Chriss模型
            participation = quantity / max(market_depth, 1e-10)
            permanent_impact = price * params.eta * participation
            temporary_impact = price * params.gamma * np.sign(quantity) * (participation ** 0.6) * (volatility ** 0.6)
        
        else:
            permanent_impact = 0.0
            temporary_impact = 0.0
        
        return permanent_impact, temporary_impact
    
    def calculate_funding_cost(
        self,
        position_value: float,
        hours_held: float,
        custom_rate: Optional[float] = None
    ) -> float:
        """
        计算资金费率成本
        
        Args:
            position_value: 持仓价值
            hours_held: 持仓小时数
            custom_rate: 自定义资金费率
        
        Returns:
            资金费率成本
        """
        rate = custom_rate if custom_rate is not None else self.funding_rate
        intervals = hours_held / self.funding_interval
        return position_value * rate * intervals
    
    def calculate_total_cost(
        self,
        price: float,
        quantity: float,
        volume: float,
        is_maker: bool = False,
        volatility: float = 0.0,
        spread: float = 0.0,
        side: str = "buy",
        market_depth: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算总交易成本
        
        Returns:
            包含各项成本的字典
        """
        notional = price * quantity
        
        # 手续费
        fee = self.calculate_fee(notional, is_maker)
        
        # 滑点
        slippage = self.calculate_slippage(price, quantity, volume, volatility, spread, side)
        
        # 冲击成本
        permanent_impact, temporary_impact = self.calculate_impact_cost(
            price, quantity, market_depth, volatility
        )
        
        total_cost = fee + slippage * quantity + (permanent_impact + temporary_impact) * quantity
        
        cost_breakdown = {
            "fee": fee,
            "slippage": slippage * quantity,
            "permanent_impact": permanent_impact * quantity,
            "temporary_impact": temporary_impact * quantity,
            "total_cost": total_cost,
            "cost_bps": (total_cost / notional) * 10000 if notional > 0 else 0
        }
        
        # 更新统计
        self.total_fees += fee
        self.total_slippage += slippage * quantity
        self.total_impact += (permanent_impact + temporary_impact) * quantity
        self.cost_history.append(cost_breakdown)
        
        return cost_breakdown
    
    def get_cost_summary(self) -> Dict:
        """获取成本统计摘要"""
        if not self.cost_history:
            return {
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "total_impact": 0.0,
                "total_funding": 0.0,
                "avg_cost_bps": 0.0
            }
        
        avg_cost_bps = np.mean([c["cost_bps"] for c in self.cost_history])
        
        return {
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_impact": self.total_impact,
            "total_funding": self.total_funding,
            "avg_cost_bps": avg_cost_bps,
            "num_trades": len(self.cost_history)
        }
    
    def reset(self):
        """重置成本统计"""
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_impact = 0.0
        self.total_funding = 0.0
        self.cost_history = []


class ExchangeCostProfile:
    """
    交易所成本配置
    预定义主流交易所的手续费结构
    """
    
    @staticmethod
    def binance_spot() -> FeeStructure:
        """币安现货费率"""
        return FeeStructure(
            maker_fee=0.001,
            taker_fee=0.001,
            withdrawal_fee=0.0
        )
    
    @staticmethod
    def binance_futures() -> FeeStructure:
        """币安合约费率"""
        return FeeStructure(
            maker_fee=0.0002,
            taker_fee=0.0005,
            withdrawal_fee=0.0
        )
    
    @staticmethod
    def okx_spot() -> FeeStructure:
        """OKX现货费率"""
        return FeeStructure(
            maker_fee=0.0008,
            taker_fee=0.001,
            withdrawal_fee=0.0
        )
    
    @staticmethod
    def okx_futures() -> FeeStructure:
        """OKX合约费率"""
        return FeeStructure(
            maker_fee=0.0002,
            taker_fee=0.0005,
            withdrawal_fee=0.0
        )
    
    @staticmethod
    def bybit_spot() -> FeeStructure:
        """Bybit现货费率"""
        return FeeStructure(
            maker_fee=0.001,
            taker_fee=0.001,
            withdrawal_fee=0.0
        )
    
    @staticmethod
    def bybit_futures() -> FeeStructure:
        """Bybit合约费率"""
        return FeeStructure(
            maker_fee=0.0001,
            taker_fee=0.0006,
            withdrawal_fee=0.0
        )


def estimate_realistic_costs(
    symbol: str,
    trade_size_usd: float,
    exchange: str = "binance",
    market_type: str = "futures"
) -> Dict[str, float]:
    """
    估算实际交易成本
    
    Args:
        symbol: 交易对
        trade_size_usd: 交易金额 (USD)
        exchange: 交易所名称
        market_type: 市场类型 (spot/futures)
    
    Returns:
        成本估算
    """
    # 获取交易所费率
    if exchange.lower() == "binance":
        fee_structure = ExchangeCostProfile.binance_futures() if market_type == "futures" else ExchangeCostProfile.binance_spot()
    elif exchange.lower() == "okx":
        fee_structure = ExchangeCostProfile.okx_futures() if market_type == "futures" else ExchangeCostProfile.okx_spot()
    elif exchange.lower() == "bybit":
        fee_structure = ExchangeCostProfile.bybit_futures() if market_type == "futures" else ExchangeCostProfile.bybit_spot()
    else:
        fee_structure = FeeStructure()
    
    # 创建成本模型
    cost_model = TransactionCostModel(
        fee_structure=fee_structure,
        slippage_model=SlippageModel.VOLUME_BASED,
        impact_model=ImpactModel.SQUARE_ROOT
    )
    
    # 估算价格 (假设)
    price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 1
    quantity = trade_size_usd / price
    
    # 估算成本
    costs = cost_model.calculate_total_cost(
        price=price,
        quantity=quantity,
        volume=trade_size_usd * 10,  # 假设成交量是交易量的10倍
        is_maker=False,
        volatility=0.02,
        spread=price * 0.0001,
        side="buy"
    )
    
    return costs


if __name__ == "__main__":
    # 测试成本模型
    print("交易成本模型测试")
    print("=" * 60)
    
    # 创建成本模型
    cost_model = TransactionCostModel(
        fee_structure=ExchangeCostProfile.binance_futures(),
        slippage_model=SlippageModel.ADVANCED,
        impact_model=ImpactModel.SQUARE_ROOT
    )
    
    # 测试交易成本计算
    price = 50000
    quantity = 0.1
    volume = 1000000
    
    costs = cost_model.calculate_total_cost(
        price=price,
        quantity=quantity,
        volume=volume,
        is_maker=False,
        volatility=0.02,
        spread=price * 0.0001,
        side="buy"
    )
    
    print(f"\n交易参数:")
    print(f"  价格: ${price:,.2f}")
    print(f"  数量: {quantity} BTC")
    print(f"  名义价值: ${price * quantity:,.2f}")
    
    print(f"\n成本分解:")
    for key, value in costs.items():
        print(f"  {key}: ${value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 测试资金费率
    funding_cost = cost_model.calculate_funding_cost(
        position_value=5000,
        hours_held=24
    )
    print(f"\n资金费率成本 (24小时): ${funding_cost:.4f}")
    
    # 测试成本摘要
    summary = cost_model.get_cost_summary()
    print(f"\n成本统计摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 测试交易所成本估算
    print("\n" + "=" * 60)
    print("交易所成本估算")
    print("=" * 60)
    
    for exchange in ["binance", "okx", "bybit"]:
        for mtype in ["spot", "futures"]:
            costs = estimate_realistic_costs("BTCUSDT", 10000, exchange, mtype)
            print(f"\n{exchange.upper()} {mtype.upper()}:")
            print(f"  总成本: ${costs['total_cost']:.4f} ({costs['cost_bps']:.2f} bps)")
