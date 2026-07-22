"""
综合策略 —— 多信号共振波段策略

逻辑：
  - 买入: 均线多头排列 + MACD金叉 + RSI不超买 + 放量
  - 卖出: 任一下跌信号触发 (均线死叉 / RSI超买回落 / 跌破中轨)
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class CompositeStrategy(BaseStrategy):
    name = "composite"
    description = "多指标共振波段策略"

    def __init__(self, ma_short: int = 5, ma_long: int = 20,
                 stop_loss_pct: float = 0.06,
                 take_profit_pct: float = 0.20):
        super().__init__(params={
            "ma_short": ma_short, "ma_long": ma_long,
            "stop_loss": stop_loss_pct, "take_profit": take_profit_pct,
        })
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        close = df["close"].values
        dif = df["macd_dif"].values
        dea = df["macd_dea"].values
        rsi = df["rsi"].values
        vol = df["volume"].values
        vol_ma = df["vol_ma_5"].values

        for i in range(3, len(df)):
            idx = df.index[i]

            # ========== 买入条件 (需同时满足4项) ==========
            buy_score = 0

            # 1. 均线多头排列: MA5 > MA20 > MA60
            if (df["ma_short"].iloc[i] > df["ma_long"].iloc[i] and
                df["ma_long"].iloc[i] > df["ma_60"].iloc[i]):
                buy_score += 1

            # 2. MACD 金叉或即将金叉
            if dif[i] > dea[i] and dif[i-1] <= dea[i-1]:
                buy_score += 1
            elif dif[i] > dea[i] and dif[i] > dif[i-1]:
                buy_score += 0.5

            # 3. RSI 不超买 (< 65)
            if rsi[i] < 65 and rsi[i] > 30:
                buy_score += 1

            # 4. 成交量配合 (放量)
            if vol[i] > vol_ma[i] * 1.2:
                buy_score += 1

            # 买入: >= 3 分
            if buy_score >= 3:
                signals.iloc[i] = 1

            # ========== 卖出条件 ==========
            sell_score = 0

            # 均线死叉
            if (df["ma_short"].iloc[i] < df["ma_long"].iloc[i] and
                df["ma_short"].iloc[i-1] >= df["ma_long"].iloc[i-1]):
                sell_score += 2

            # RSI 超买回落
            if rsi[i] < rsi[i-1] and rsi[i-1] > 75:
                sell_score += 1

            # 价格跌破布林中轨
            if (close[i] < df["boll_mid"].iloc[i] and
                close[i-1] >= df["boll_mid"].iloc[i-1]):
                sell_score += 1

            # 卖出: >= 2 分
            if sell_score >= 2:
                signals.iloc[i] = -1

        return signals
