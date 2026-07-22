"""
RSI 策略 —— 超买超卖反转波段

逻辑:
  - 买入: RSI 从超卖区(<30)回升
  - 卖出: RSI 从超买区(>70)回落 或 价格跌破止损
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    name = "rsi"
    description = "RSI超买超卖反转策略"

    def __init__(self, period: int = 14, oversold: int = 30,
                 overbought: int = 70, divergence_lookback: int = 5):
        super().__init__(params={
            "period": period, "oversold": oversold,
            "overbought": overbought,
        })
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        rsi = df["rsi"].values
        close = df["close"].values

        for i in range(1, len(df)):
            idx = df.index[i]

            # 买入: RSI 从超卖区上穿 + 价格收阳确认
            if (rsi[i-1] <= self.oversold and rsi[i] > self.oversold and
                close[i] > close[i-1]):
                signals.iloc[i] = 1
            # 或 RSI < 35 且连续下跌后企稳
            elif (rsi[i] < 35 and rsi[i] > rsi[i-1] and
                  close[i] > close[i-1] and
                  close[i-1] < close[i-2] and close[i-2] < close[i-3]):
                signals.iloc[i] = 1

            # 卖出: RSI 从超买区下穿
            if (rsi[i-1] >= self.overbought and rsi[i] < self.overbought and
                close[i] < close[i-1]):
                signals.iloc[i] = -1
            # 或 RSI > 65 且阴线
            elif rsi[i] > 65 and close[i] < close[i-1]:
                signals.iloc[i] = -1

        return signals
