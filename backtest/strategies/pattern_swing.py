"""
K线形态增强波段策略

将K线形态识别与均线交叉策略结合:
  - 均线交叉 = 主信号
  - K线形态 = 确认/提前入场
  - 连续K线分析 = 过滤假信号
  - 自适应仓位 = Kelly + ATR动态调整
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from patterns import CandlestickPatterns, ConsecutiveBarAnalyzer
from filters import VolumePriceAnalyzer, ATRDynamicStops


class PatternEnhancedStrategy(BaseStrategy):
    """
    K线形态增强策略

    开仓逻辑 (信号叠加):
      1. 主信号: MA5上穿MA20 (与最佳策略一致)
      2. 形态增强: 如果在交叉前1-2天出现了看涨形态，提前入场
      3. K线过滤: 连阴>3天后出现锤子线/吞没 → 反弹信号
      4. 量价确认: 放量上涨 + 非出货期

    平仓逻辑:
      1. MA死叉 (主要)
      2. 黄昏之星/看跌吞没 + 跌破5日线 (加速)
      3. ATR动态止损 (辅助)
    """

    name = "pattern_enhanced"
    description = "K线形态+均线交叉+量价共振策略"

    def __init__(
        self,
        ma_short: int = 5,
        ma_long: int = 20,
        # 止损
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        # 形态权重
        pattern_weight: float = 0.3,   # 形态在入场信号中的权重
        # 量价
        vol_surge: float = 1.2,
    ):
        super().__init__(params={
            "ma": f"({ma_short},{ma_long})",
            "sl": stop_loss_pct,
            "tp": take_profit_pct,
            "ptn_w": pattern_weight,
        })
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.pattern_weight = pattern_weight
        self.vol_surge = vol_surge

        self.candles = CandlestickPatterns()
        self.bars = ConsecutiveBarAnalyzer()
        self.vp = VolumePriceAnalyzer(vol_ma_period=20, surge_multiplier=vol_surge)
        self.atr_stops = ATRDynamicStops(atr_mult_stop=2.0, atr_mult_target=3.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)

        # 各子模块
        patterns = self.candles.detect_all(df)
        bars = self.bars.detect(df)
        vp_entry = self.vp.get_entry_filters(df)
        atr_stops_df = self.atr_stops.compute_stops(df)
        self._stop_info = atr_stops_df

        close = df["close"].values
        open_p = df["open"].values
        ma_s = df["ma_short"].values
        ma_l = df["ma_long"].values
        rsi = df["rsi"].values
        vol = df["volume"].values
        vol_ma5 = df["vol_ma_5"].values

        n = len(df)

        for i in range(3, n):
            idx = df.index[i]

            # ========================================
            # 第一层: 主信号 — MA金叉/死叉
            # ========================================
            golden = (ma_s[i] > ma_l[i] and ma_s[i-1] <= ma_l[i-1])
            dead = (ma_s[i] < ma_l[i] and ma_s[i-1] >= ma_l[i-1])

            # ========================================
            # 第二层: K线形态信号
            # ========================================
            pattern = patterns["pattern"].iloc[i]
            pattern_type = patterns["pattern_type"].iloc[i]
            pattern_conf = patterns["pattern_confidence"].iloc[i]
            pattern_score = patterns["pattern_score"].iloc[i]

            # 前一天形态 (提前入场逻辑)
            pattern_prev = patterns["pattern"].iloc[i-1]
            pattern_type_prev = patterns["pattern_type"].iloc[i-1]
            pattern_score_prev = patterns["pattern_score"].iloc[i-1]

            # 连阳/连阴
            cons_up = bars["consecutive_up"].iloc[i]
            cons_down = bars["consecutive_down"].iloc[i]
            nr = bars["nr_pattern"].iloc[i]
            vol_dry = bars["vol_dry_up"].iloc[i]

            # 量价
            vol_ok = vp_entry["vol_ok"].iloc[i]
            vp_conf = vp_entry["confidence"].iloc[i]

            # ========================================
            # 买入信号
            # ========================================
            buy_score = 0.0
            buy_reasons = []

            # 主信号: MA金叉
            if golden:
                buy_score += 0.6
                buy_reasons.append("MA金叉")

            # 形态: 前一天出看涨形态 + 今天阳线确认
            if (pattern_type_prev == "bullish" and
                close[i] > close[i-1] and
                close[i] > open_p[i]):
                buy_score += self.pattern_weight * pattern_conf
                buy_reasons.append(f"形态:{pattern_prev}")

            # 形态: 今天出强看涨形态
            if pattern_type == "bullish" and pattern_score >= 1.5:
                buy_score += 0.3
                buy_reasons.append(f"当日:{pattern}")

            # 连续K线: 连阴后企稳
            if cons_down >= 3 and close[i] > close[i-1] and close[i] > open_p[i]:
                buy_score += 0.2
                buy_reasons.append("连阴企稳")

            # 缩量连阳: 主力吸筹
            if vol_dry:
                buy_score += 0.15
                buy_reasons.append("缩量吸筹")

            # NR后放量突破
            nr_prev = bars["nr_pattern"].iloc[i-2:i].any()
            if nr_prev and vol[i] > vol_ma5[i] * 1.3 and close[i] > close[i-1]:
                buy_score += 0.2
                buy_reasons.append("窄幅突破")

            # 量价确认
            if vol_ok and close[i] > close[i-1]:
                buy_score += 0.15

            # RSI 不在超买区
            if rsi[i] < 65:
                buy_score += 0.1

            # 买入阈值
            if buy_score >= 0.6:
                signals.iloc[i] = 1

            # ========================================
            # 卖出信号
            # ========================================
            sell_score = 0.0
            sell_reasons = []

            # MA死叉 (主信号)
            if dead:
                sell_score += 0.6
                sell_reasons.append("MA死叉")

            # 看跌形态
            if pattern_type == "bearish" and pattern_score <= -1.5:
                sell_score += 0.4
                sell_reasons.append(f"看跌形态:{pattern}")

            # 黄昏之星
            if pattern == "evening_star":
                sell_score += 0.5
                sell_reasons.append("黄昏之星")

            # 看跌吞没 + 跌破5日线
            if pattern == "bearish_engulfing" and close[i] < ma_s[i]:
                sell_score += 0.5
                sell_reasons.append("吞没+破MA5")

            # RSI超买
            if rsi[i] > 75 and rsi[i] < rsi[i-1]:
                sell_score += 0.3
                sell_reasons.append("RSI超买")

            # 连阳后乏力
            if cons_up >= 5 and close[i] < close[i-1]:
                sell_score += 0.3
                sell_reasons.append("连阳乏力")

            # 卖出阈值
            if sell_score >= 0.6:
                signals.iloc[i] = -1

        return signals

    def get_dynamic_stops(self) -> pd.DataFrame:
        if hasattr(self, "_stop_info"):
            return self._stop_info
        return None
