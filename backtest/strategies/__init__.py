"""
策略注册表
"""

from .base import BaseStrategy
from .ma_cross import MACrossStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .rsi import RSIStrategy
from .composite import CompositeStrategy
from .advanced_swing import (
    AdvancedSwingStrategy,
    TrendFollowingStrategy,
    VolumeBreakoutStrategy,
)
from .pattern_swing import PatternEnhancedStrategy


# 所有可用策略
STRATEGIES = {
    # ---- 基础策略 ----
    "ma_cross": MACrossStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
    "rsi": RSIStrategy,
    "composite": CompositeStrategy,
    # ---- 高级策略 ----
    "advanced_swing": AdvancedSwingStrategy,
    "trend_following": TrendFollowingStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    # ---- K线形态策略 ----
    "pattern_enhanced": PatternEnhancedStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """根据名称获取策略实例"""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}. 可用: {list(STRATEGIES.keys())}")
    return cls(**kwargs)


def list_strategies():
    """列出所有策略"""
    for name, cls in STRATEGIES.items():
        inst = cls()
        print(f"  {name:20s} - {inst.description}")
