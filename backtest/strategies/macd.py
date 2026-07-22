"""
MACD 策略 —— 趋势波段策略

逻辑:
  - 买入: DIF 上穿 DEA (金叉) 且处于零轴下方或刚上零轴
  - 卖出: DIF 下穿 DEA (死叉) 或 高位死叉
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    name = "macd"
    description = "MACD金叉死叉波段策略"

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9,
                 zero_line_filter: bool = True):
        super().__init__(params={
            "fast": fast, "slow": slow, "signal": signal_period,
            "zero_filter": zero_line_filter,
        })
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
        self.zero_line_filter = zero_line_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        dif = df["macd_dif"].values
        dea = df["macd_dea"].values

        for i in range(1, len(df)):
            idx = df.index[i]
            # 金叉: DIF上穿DEA
            golden_cross = (dif[i-1] <= dea[i-1]) and (dif[i] > dea[i])
            # 死叉: DIF下穿DEA
            dead_cross = (dif[i-1] >= dea[i-1]) and (dif[i] < dea[i])

            if golden_cross:
                if self.zero_line_filter and dif[i] < -0.05:
                    # 零轴下方较远处金叉，可能是反弹不是反转，降低信度
                    signals.iloc[i] = 1
                else:
                    signals.iloc[i] = 1

            elif dead_cross:
                signals.iloc[i] = -1

        return signals
