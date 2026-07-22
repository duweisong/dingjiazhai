"""
布林带策略 —— 均值回归波段策略

逻辑:
  - 买入: 价格触及/跌破下轨 + 企稳信号 (次日反弹)
  - 卖出: 价格触及/突破上轨 或 回到中轨下方
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class BollingerStrategy(BaseStrategy):
    name = "bollinger"
    description = "布林带均值回归波段策略"

    def __init__(self, period: int = 20, std: float = 2.0,
                 confirm_bars: int = 1):
        super().__init__(params={
            "period": period, "std": std, "confirm_bars": confirm_bars,
        })
        self.period = period
        self.std = std
        self.confirm_bars = confirm_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        close = df["close"].values
        upper = df["boll_upper"].values
        lower = df["boll_lower"].values
        mid = df["boll_mid"].values

        for i in range(1, len(df)):
            idx = df.index[i]

            # 买入: 价格跌破下轨后回升
            if close[i-1] <= lower[i-1] and close[i] > lower[i-1]:
                signals.iloc[i] = 1
            # 或价格接近下轨 + 阳线确认
            elif (close[i-1] <= lower[i-1] * 1.02 and
                  close[i] > close[i-1] and
                  close[i] > (close[i] + df["open"].iloc[i]) / 2):
                signals.iloc[i] = 1

            # 卖出: 价格突破上轨后回落
            if close[i-1] >= upper[i-1] and close[i] < upper[i-1]:
                signals.iloc[i] = -1
            # 或价格从上方跌破中轨
            elif (close[i-1] >= mid[i-1] and close[i] < mid[i]):
                signals.iloc[i] = -1

        return signals
