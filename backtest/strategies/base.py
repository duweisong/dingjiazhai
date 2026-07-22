"""
策略基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class Signal:
    """交易信号"""
    date: pd.Timestamp
    signal_type: str        # "BUY", "SELL", "HOLD"
    price: float
    reason: str = ""
    confidence: float = 1.0  # 0~1 信度


@dataclass
class Position:
    """持仓状态"""
    in_position: bool = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    shares: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    highest_since_entry: float = 0.0    # 持仓期间最高价 (用于移动止损)


class BaseStrategy(ABC):
    """策略抽象基类"""

    name: str = "base"
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self._signals: List[Signal] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成信号序列。

        Returns
        -------
        pd.Series with index matching df
            1 = 买入, -1 = 卖出, 0 = 持有
        """
        ...

    def get_params_str(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    def __repr__(self):
        return f"{self.name}({self.get_params_str()})"
