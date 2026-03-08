"""
滑点模型模块
- 滑点估计
- 冲击成本模型
- 大单拆分
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import defaultdict
import math

from ..exchange.exchange_client import OrderSide

logger = logging.getLogger(__name__)


class SlippageModelType(Enum):
    """滑点模型类型"""
    FIXED = "fixed"                 # 固定滑点
    LINEAR = "linear"               # 线性滑点
    SQUARE_ROOT = "square_root"     # 平方根模型
    LOGARITHMIC = "logarithmic"     # 对数模型
    ADAPTIVE = "adaptive"           # 自适应模型


class MarketImpactType(Enum):
    """市场冲击类型"""
    TEMPORARY = "temporary"         # 临时冲击
    PERMANENT = "permanent"         # 永久冲击
    TOTAL = "total"                 # 总冲击


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: Decimal
    amount: Decimal
    cumulative_amount: Decimal = Decimal("0")


@dataclass
class SlippageEstimate:
    """滑点估计结果"""
    estimated_slippage: Decimal
    estimated_price: Decimal
    confidence: float
    factors: Dict[str, float]


@dataclass
class MarketImpact:
    """市场冲击"""
    temporary_impact: Decimal
    permanent_impact: Decimal
    total_impact: Decimal
    decay_time: float  # 冲击衰减时间(秒)


@dataclass
class OrderSplitResult:
    """订单拆分结果"""
    slices: List[Decimal]
    total_expected_slippage: Decimal
    execution_time: float
    optimal_slice_count: int


class SlippageModel:
    """滑点模型"""
    
    def __init__(
        self,
        model_type: SlippageModelType = SlippageModelType.ADAPTIVE,
        base_slippage: Decimal = Decimal("0.001"),
        volatility_factor: float = 1.0,
        spread_factor: float = 0.5,
        depth_factor: float = 1.0
    ):
        self.model_type = model_type
        self.base_slippage = base_slippage
        self.volatility_factor = volatility_factor
        self.spread_factor = spread_factor
        self.depth_factor = depth_factor
        
        # 历史数据
        self.slippage_history: Dict[str, List[Decimal]] = defaultdict(list)
        self.spread_history: Dict[str, List[Decimal]] = defaultdict(list)
        self.volatility_history: Dict[str, List[Decimal]] = defaultdict(list)
        
        # 模型参数
        self.model_params: Dict[str, Dict] = defaultdict(lambda: {
            'alpha': 0.5,      # 订单规模系数
            'beta': 0.6,       # 非线性系数
            'gamma': 0.3       # 波动率系数
        })
    
    def estimate_slippage(
        self,
        symbol: str,
        order_size: Decimal,
        side: OrderSide,
        reference_price: Decimal,
        orderbook: Optional[Dict] = None,
        volatility: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None
    ) -> SlippageEstimate:
        """估计滑点"""
        factors = {}
        
        # 1. 基础滑点
        base = self.base_slippage
        factors['base'] = float(base)
        
        # 2. 订单规模因子
        if avg_volume and avg_volume > 0:
            size_ratio = order_size / avg_volume
            size_factor = self._calculate_size_factor(size_ratio)
        else:
            size_factor = Decimal("1.0")
        factors['size'] = float(size_factor)
        
        # 3. 波动率因子
        if volatility:
            vol_factor = Decimal("1.0") + volatility * Decimal(str(self.volatility_factor))
        else:
            vol_factor = Decimal("1.0")
        factors['volatility'] = float(vol_factor)
        
        # 4. 订单簿深度因子
        depth_factor = Decimal("1.0")
        if orderbook:
            depth_factor = self._calculate_depth_factor(order_size, side, orderbook)
        factors['depth'] = float(depth_factor)
        
        # 计算总滑点
        if self.model_type == SlippageModelType.FIXED:
            slippage = base * size_factor
        elif self.model_type == SlippageModelType.LINEAR:
            slippage = base * size_factor * vol_factor * depth_factor
        elif self.model_type == SlippageModelType.SQUARE_ROOT:
            slippage = base * (size_factor.sqrt()) * vol_factor * depth_factor
        elif self.model_type == SlippageModelType.LOGARITHMIC:
            if size_factor > 0:
                slippage = base * (Decimal("1.0") + size_factor.ln()) * vol_factor * depth_factor
            else:
                slippage = base * vol_factor * depth_factor
        else:  # ADAPTIVE
            slippage = self._adaptive_slippage(symbol, base, size_factor, vol_factor, depth_factor)
        
        # 计算估计价格
        if side == OrderSide.BUY:
            estimated_price = reference_price * (Decimal("1.0") + slippage)
        else:
            estimated_price = reference_price * (Decimal("1.0") - slippage)
        
        # 计算置信度
        confidence = self._calculate_confidence(symbol, factors)
        
        return SlippageEstimate(
            estimated_slippage=slippage,
            estimated_price=estimated_price,
            confidence=confidence,
            factors=factors
        )
    
    def _calculate_size_factor(self, size_ratio: Decimal) -> Decimal:
        """计算订单规模因子"""
        if size_ratio <= Decimal("0.01"):
            return Decimal("1.0")
        elif size_ratio <= Decimal("0.05"):
            return Decimal("1.0") + size_ratio * Decimal("10.0")
        elif size_ratio <= Decimal("0.1"):
            return Decimal("1.5") + size_ratio * Decimal("20.0")
        else:
            return Decimal("3.5") + size_ratio.sqrt() * Decimal("5.0")
    
    def _calculate_depth_factor(
        self,
        order_size: Decimal,
        side: OrderSide,
        orderbook: Dict
    ) -> Decimal:
        """计算订单簿深度因子"""
        if side == OrderSide.BUY:
            levels = orderbook.get('asks', [])
        else:
            levels = orderbook.get('bids', [])
        
        if not levels:
            return Decimal("2.0")  # 无深度数据，假设高滑点
        
        # 计算达到订单规模需要的价格移动
        cumulative = Decimal("0")
        for level in levels:
            price = Decimal(str(level[0]))
            amount = Decimal(str(level[1]))
            cumulative += amount
            
            if cumulative >= order_size:
                # 找到足够的深度
                return Decimal("1.0")
        
        # 深度不足
        shortfall = (order_size - cumulative) / order_size
        return Decimal("1.0") + shortfall * Decimal("5.0")
    
    def _adaptive_slippage(
        self,
        symbol: str,
        base: Decimal,
        size_factor: Decimal,
        vol_factor: Decimal,
        depth_factor: Decimal
    ) -> Decimal:
        """自适应滑点模型"""
        params = self.model_params[symbol]
        alpha = Decimal(str(params['alpha']))
        beta = Decimal(str(params['beta']))
        gamma = Decimal(str(params['gamma']))
        
        # 非线性组合
        slippage = base * (Decimal("1.0") + alpha * (size_factor ** beta - Decimal("1.0")))
        slippage *= vol_factor ** gamma
        slippage *= depth_factor
        
        return slippage
    
    def _calculate_confidence(self, symbol: str, factors: Dict[str, float]) -> float:
        """计算置信度"""
        # 基于历史数据的质量计算置信度
        history_size = len(self.slippage_history.get(symbol, []))
        
        if history_size < 10:
            return 0.3
        elif history_size < 50:
            return 0.5
        elif history_size < 100:
            return 0.7
        else:
            return 0.85
    
    def record_actual_slippage(
        self,
        symbol: str,
        expected_price: Decimal,
        actual_price: Decimal,
        side: OrderSide
    ):
        """记录实际滑点"""
        if expected_price > 0:
            if side == OrderSide.BUY:
                slippage = (actual_price - expected_price) / expected_price
            else:
                slippage = (expected_price - actual_price) / expected_price
            
            self.slippage_history[symbol].append(slippage)
            
            # 保持历史记录在合理范围内
            if len(self.slippage_history[symbol]) > 1000:
                self.slippage_history[symbol] = self.slippage_history[symbol][-500:]
            
            # 更新模型参数
            self._update_model_params(symbol)
    
    def _update_model_params(self, symbol: str):
        """更新模型参数"""
        history = self.slippage_history.get(symbol, [])
        if len(history) < 20:
            return
        
        # 简单的在线学习
        recent_slippage = sum(history[-20:]) / 20
        
        # 根据实际滑点调整参数
        params = self.model_params[symbol]
        if recent_slippage > self.base_slippage * Decimal("2.0"):
            params['alpha'] = min(params['alpha'] * 1.1, 1.0)
        elif recent_slippage < self.base_slippage * Decimal("0.5"):
            params['alpha'] = max(params['alpha'] * 0.9, 0.1)
    
    def get_average_slippage(self, symbol: str, lookback: int = 100) -> Decimal:
        """获取平均滑点"""
        history = self.slippage_history.get(symbol, [])
        if not history:
            return self.base_slippage
        
        recent = history[-lookback:]
        return sum(recent) / len(recent)
    
    def get_slippage_stats(self, symbol: str) -> Dict:
        """获取滑点统计"""
        history = self.slippage_history.get(symbol, [])
        if not history:
            return {
                'mean': float(self.base_slippage),
                'std': 0.0,
                'min': float(self.base_slippage),
                'max': float(self.base_slippage),
                'count': 0
            }
        
        mean = sum(history) / len(history)
        variance = sum((s - mean) ** 2 for s in history) / len(history)
        std = variance.sqrt()
        
        return {
            'mean': float(mean),
            'std': float(std),
            'min': float(min(history)),
            'max': float(max(history)),
            'count': len(history)
        }


class MarketImpactModel:
    """市场冲击模型"""
    
    def __init__(
        self,
        temporary_impact_coeff: float = 0.1,
        permanent_impact_coeff: float = 0.05,
        decay_halftime: float = 300.0  # 5分钟
    ):
        self.temporary_impact_coeff = temporary_impact_coeff
        self.permanent_impact_coeff = permanent_impact_coeff
        self.decay_halftime = decay_halftime
        
        # 冲击历史
        self.impact_history: List[Dict] = []
    
    def calculate_impact(
        self,
        order_size: Decimal,
        avg_daily_volume: Decimal,
        volatility: Decimal,
        execution_time: float
    ) -> MarketImpact:
        """计算市场冲击"""
        if avg_daily_volume <= 0:
            return MarketImpact(Decimal("0"), Decimal("0"), Decimal("0"), 0.0)
        
        # 订单规模占比
        participation_rate = float(order_size / avg_daily_volume)
        
        # 临时冲击 (与执行速度相关)
        temp_impact = self._calculate_temporary_impact(
            participation_rate, float(volatility), execution_time
        )
        
        # 永久冲击 (与订单规模相关)
        perm_impact = self._calculate_permanent_impact(
            participation_rate, float(volatility)
        )
        
        # 总冲击
        total_impact = temp_impact + perm_impact
        
        # 衰减时间
        decay_time = self.decay_halftime * (1 + participation_rate * 10)
        
        return MarketImpact(
            temporary_impact=Decimal(str(temp_impact)),
            permanent_impact=Decimal(str(perm_impact)),
            total_impact=Decimal(str(total_impact)),
            decay_time=decay_time
        )
    
    def _calculate_temporary_impact(
        self,
        participation_rate: float,
        volatility: float,
        execution_time: float
    ) -> float:
        """计算临时冲击"""
        # Almgren-Chriss 模型简化版
        urgency = 1.0 / max(execution_time / 60.0, 0.1)  # 执行紧迫性
        temp_impact = (
            self.temporary_impact_coeff *
            volatility *
            (participation_rate ** 0.6) *
            (urgency ** 0.5)
        )
        return temp_impact
    
    def _calculate_permanent_impact(
        self,
        participation_rate: float,
        volatility: float
    ) -> float:
        """计算永久冲击"""
        # 永久冲击与订单规模成正比
        perm_impact = (
            self.permanent_impact_coeff *
            volatility *
            participation_rate
        )
        return perm_impact
    
    def estimate_price_recovery(
        self,
        impact: MarketImpact,
        time_elapsed: float
    ) -> Decimal:
        """估计价格恢复程度"""
        if impact.decay_time <= 0:
            return Decimal("1.0")
        
        # 指数衰减
        decay = 1.0 - math.exp(-time_elapsed / impact.decay_time)
        recovery = impact.temporary_impact * Decimal(str(decay))
        
        return recovery
    
    def record_impact(self, order_data: Dict):
        """记录冲击数据"""
        self.impact_history.append({
            'timestamp': order_data.get('timestamp'),
            'symbol': order_data.get('symbol'),
            'size': order_data.get('size'),
            'expected_price': order_data.get('expected_price'),
            'actual_price': order_data.get('actual_price'),
            'execution_time': order_data.get('execution_time')
        })
        
        # 限制历史记录大小
        if len(self.impact_history) > 1000:
            self.impact_history = self.impact_history[-500:]


class OrderSplitter:
    """订单拆分器"""
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        impact_model: Optional[MarketImpactModel] = None
    ):
        self.slippage_model = slippage_model or SlippageModel()
        self.impact_model = impact_model or MarketImpactModel()
    
    def split_order(
        self,
        symbol: str,
        total_amount: Decimal,
        side: OrderSide,
        reference_price: Decimal,
        max_slice_size: Optional[Decimal] = None,
        max_slippage: Decimal = Decimal("0.01"),
        avg_volume: Optional[Decimal] = None,
        target_time: float = 300.0
    ) -> OrderSplitResult:
        """拆分订单"""
        # 如果没有指定最大切片大小，根据市场容量计算
        if max_slice_size is None:
            if avg_volume:
                max_slice_size = avg_volume * Decimal("0.05")  # 5%的市场容量
            else:
                max_slice_size = total_amount / Decimal("10")
        
        # 计算最优切片数量
        optimal_slices = self._calculate_optimal_slices(
            total_amount, max_slice_size, max_slippage, avg_volume
        )
        
        # 生成切片
        slices = self._generate_slices(
            total_amount, optimal_slices, max_slice_size
        )
        
        # 估计总滑点
        total_slippage = self._estimate_total_slippage(
            symbol, slices, side, reference_price, avg_volume
        )
        
        # 估计执行时间
        execution_time = self._estimate_execution_time(
            slices, avg_volume, target_time
        )
        
        return OrderSplitResult(
            slices=slices,
            total_expected_slippage=total_slippage,
            execution_time=execution_time,
            optimal_slice_count=len(slices)
        )
    
    def _calculate_optimal_slices(
        self,
        total_amount: Decimal,
        max_slice_size: Decimal,
        max_slippage: Decimal,
        avg_volume: Optional[Decimal]
    ) -> int:
        """计算最优切片数量"""
        # 基于最大切片大小的最小切片数
        min_slices = int((total_amount / max_slice_size).to_integral_value())
        
        # 基于滑点限制的切片数
        if avg_volume and avg_volume > 0:
            participation = max_slice_size / avg_volume
            # 如果参与度过高，增加切片数
            if participation > Decimal("0.1"):
                min_slices = max(min_slices, int((total_amount / (avg_volume * Decimal("0.05"))).to_integral_value()))
        
        # 限制切片数量在合理范围内
        return max(2, min(min_slices, 50))
    
    def _generate_slices(
        self,
        total_amount: Decimal,
        num_slices: int,
        max_slice_size: Decimal
    ) -> List[Decimal]:
        """生成切片"""
        base_size = total_amount / num_slices
        slices = []
        remaining = total_amount
        
        for i in range(num_slices - 1):
            # 添加随机变化，避免模式识别
            variation = Decimal(str(0.9 + (i % 3) * 0.1))  # 0.9-1.1的变化
            slice_size = min(base_size * variation, max_slice_size, remaining)
            slices.append(slice_size)
            remaining -= slice_size
        
        # 最后一个切片包含余数
        slices.append(remaining)
        
        return slices
    
    def _estimate_total_slippage(
        self,
        symbol: str,
        slices: List[Decimal],
        side: OrderSide,
        reference_price: Decimal,
        avg_volume: Optional[Decimal]
    ) -> Decimal:
        """估计总滑点"""
        total_slippage = Decimal("0")
        
        for slice_size in slices:
            estimate = self.slippage_model.estimate_slippage(
                symbol=symbol,
                order_size=slice_size,
                side=side,
                reference_price=reference_price,
                avg_volume=avg_volume
            )
            # 加权平均
            weight = slice_size / sum(slices)
            total_slippage += estimate.estimated_slippage * weight
        
        return total_slippage
    
    def _estimate_execution_time(
        self,
        slices: List[Decimal],
        avg_volume: Optional[Decimal],
        target_time: float
    ) -> float:
        """估计执行时间"""
        if not avg_volume or avg_volume <= 0:
            return target_time
        
        # 基于市场容量估计
        total_volume = sum(slices)
        participation_rate = float(total_volume / avg_volume)
        
        # 考虑市场冲击，延长执行时间
        base_time = target_time
        impact_factor = 1.0 + participation_rate * 2.0
        
        return base_time * impact_factor
    
    def optimize_slice_sizes(
        self,
        slices: List[Decimal],
        market_depth: List[OrderBookLevel]
    ) -> List[Decimal]:
        """根据市场深度优化切片大小"""
        if not market_depth:
            return slices
        
        optimized = []
        
        for slice_size in slices:
            # 找到最适合的切片大小
            optimal_size = slice_size
            
            for level in market_depth:
                if level.amount >= slice_size * Decimal("0.5"):
                    # 该层级有足够深度
                    optimal_size = min(slice_size, level.amount * Decimal("0.8"))
                    break
            
            optimized.append(optimal_size)
        
        return optimized


# 便捷函数
def estimate_slippage(
    order_size: Decimal,
    avg_volume: Decimal,
    volatility: Decimal = Decimal("0.02"),
    spread: Decimal = Decimal("0.001")
) -> Decimal:
    """快速估计滑点"""
    model = SlippageModel()
    
    estimate = model.estimate_slippage(
        symbol="",
        order_size=order_size,
        side=OrderSide.BUY,
        reference_price=Decimal("1"),
        volatility=volatility
    )
    
    return estimate.estimated_slippage


def split_large_order(
    total_amount: Decimal,
    max_slice: Decimal,
    min_slices: int = 2
) -> List[Decimal]:
    """快速拆分大单"""
    splitter = OrderSplitter()
    
    result = splitter.split_order(
        symbol="",
        total_amount=total_amount,
        side=OrderSide.BUY,
        reference_price=Decimal("1"),
        max_slice_size=max_slice
    )
    
    return result.slices


# 示例用法
if __name__ == "__main__":
    # 测试滑点模型
    model = SlippageModel()
    
    estimate = model.estimate_slippage(
        symbol="BTC/USDT",
        order_size=Decimal("10"),
        side=OrderSide.BUY,
        reference_price=Decimal("50000"),
        avg_volume=Decimal("1000"),
        volatility=Decimal("0.03")
    )
    
    print(f"估计滑点: {estimate.estimated_slippage}")
    print(f"估计价格: {estimate.estimated_price}")
    print(f"置信度: {estimate.confidence}")
    print(f"因子: {estimate.factors}")
    
    # 测试订单拆分
    splitter = OrderSplitter(model)
    
    result = splitter.split_order(
        symbol="BTC/USDT",
        total_amount=Decimal("100"),
        side=OrderSide.BUY,
        reference_price=Decimal("50000"),
        max_slice_size=Decimal("10"),
        avg_volume=Decimal("1000")
    )
    
    print(f"\n订单拆分:")
    print(f"切片数量: {result.optimal_slice_count}")
    print(f"切片大小: {[float(s) for s in result.slices]}")
    print(f"预计总滑点: {result.total_expected_slippage}")
    print(f"预计执行时间: {result.execution_time}秒")
