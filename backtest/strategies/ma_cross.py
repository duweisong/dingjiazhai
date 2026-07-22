"""
均线交叉策略 —— 波段经典策略

逻辑:
  - 买入: 短期均线上穿长期均线 + 放量确认
  - 卖出: 短期均线下穿长期均线 或 移动止损
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class MACrossStrategy(BaseStrategy):
    name = "ma_cross"
    description = "双均线交叉波段策略"

    def __init__(self, ma_short: int = 5, ma_long: int = 20,
                 vol_confirm: bool = True, stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15):
        super().__init__(params={
            "ma_short": ma_short, "ma_long": ma_long,
            "vol_confirm": vol_confirm, "stop_loss": stop_loss_pct,
            "take_profit": take_profit_pct,
        })
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_confirm = vol_confirm
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        close = df["close"].values
        ma_s = df["ma_short"].values
        ma_l = df["ma_long"].values

        # 均线位置关系 (前一周期)
        above_prev = ma_s[:-1] > ma_l[:-1]
        cross_up = (ma_s[:-1] <= ma_l[:-1]) & (ma_s[1:] > ma_l[1:])
        cross_down = (ma_s[:-1] >= ma_l[:-1]) & (ma_s[1:] < ma_l[1:])

        for i in range(1, len(df)):
            idx = df.index[i]

            if cross_up[i-1]:
                # 上穿买入信号
                if self.vol_confirm:
                    vol_ok = df["volume"].iloc[i] > df["vol_ma_5"].iloc[i]
                else:
                    vol_ok = True

                if vol_ok:
                    signals.iloc[i] = 1

            elif cross_down[i-1]:
                # 下穿卖出信号
                signals.iloc[i] = -1

        return signals
