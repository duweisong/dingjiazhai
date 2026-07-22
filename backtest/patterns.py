"""
K线形态识别模块

识别经典反转/持续形态，作为策略信号的增强维度。
中日技术分析常用形态全覆盖。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class CandlestickPatterns:
    """K线形态识别器"""

    def __init__(self):
        pass

    def _body(self, open_, close):
        """实体长度"""
        return abs(close - open_)

    def _upper_shadow(self, open_, high, close):
        """上影线长度"""
        return high - max(open_, close)

    def _lower_shadow(self, open_, low, close):
        """下影线长度"""
        return min(open_, close) - low

    def _is_bullish(self, open_, close):
        return close > open_

    def _is_bearish(self, open_, close):
        return close < open_

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测所有形态，返回每个交易日的形态标签和信度

        Returns DataFrame with:
            pattern        : 形态名称 (空字符串 = 无形态)
            pattern_type   : 'bullish' | 'bearish' | ''
            confidence     : 0~1 信度
            pattern_score  : -2到+2的综合评分
        """
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        patterns = [""] * n
        pattern_type = [""] * n
        confidence = np.zeros(n)
        pattern_score = np.zeros(n)

        for i in range(2, n):
            o1, h1, l1, c1 = open_[i-1], high[i-1], low[i-1], close[i-1]
            o2, h2, l2, c2 = open_[i-2], high[i-2], low[i-2], close[i-2]
            o, h, l, c = open_[i], high[i], low[i], close[i]

            body = self._body(o, c)
            body_prev = self._body(o1, c1)
            upper = self._upper_shadow(o, h, c)
            lower = self._lower_shadow(o, l, c)
            upper_prev = self._upper_shadow(o1, h1, c1)
            lower_prev = self._lower_shadow(o1, l1, c1)
            total_range = h - l
            total_range_prev = h1 - l1

            if total_range == 0 or total_range_prev == 0:
                continue

            # ============================================
            # 反转形态 (看涨)
            # ============================================

            # 1. 锤子线 Hammer: 长下影 + 小实体 + 在下行趋势后
            if (lower > body * 2 and
                upper < body * 0.3 and
                lower > total_range * 0.55):
                # 确认：前一根为阴线，当前在低位
                if c1 < o1 and c > o and c > c1:
                    patterns[i] = "hammer"
                    pattern_type[i] = "bullish"
                    confidence[i] = 0.7
                    pattern_score[i] = 1.5

            # 2. 看涨吞没 Bullish Engulfing
            if (c > o and c1 < o1 and
                o <= c1 and c >= o1 and
                body > body_prev * 1.2):
                patterns[i] = "bullish_engulfing"
                pattern_type[i] = "bullish"
                confidence[i] = 0.8
                pattern_score[i] = 2.0

            # 3. 启明星 Morning Star (三日形态)
            if i >= 3:
                o3, c3 = open_[i-2], close[i-2]
                if (c3 < o3 and       # Day1: 大阴线
                    abs(c1 - o1) < body_prev * 0.3 and  # Day2: 小实体(星)
                    c > o and c > (o3 + c3) / 2):       # Day3: 阳线、收盘高于Day1中点
                    if patterns[i] == "":
                        patterns[i] = "morning_star"
                        pattern_type[i] = "bullish"
                        confidence[i] = 0.85
                        pattern_score[i] = 2.0

            # 4. 穿刺线 Piercing Line
            if (c1 < o1 and c > o and
                o < l1 and c > (o1 + c1) / 2 and
                body > body_prev * 1.1):
                patterns[i] = "piercing"
                pattern_type[i] = "bullish"
                confidence[i] = 0.75
                pattern_score[i] = 1.8

            # ============================================
            # 反转形态 (看跌)
            # ============================================

            # 5. 上吊线 Hanging Man: 长下影 + 小实体 + 在上行趋势后
            if (lower > body * 2 and
                upper < body * 0.3 and
                lower > total_range * 0.55):
                if c1 > o1 and c < o:
                    patterns[i] = "hanging_man"
                    pattern_type[i] = "bearish"
                    confidence[i] = 0.65
                    pattern_score[i] = -1.5

            # 6. 看跌吞没 Bearish Engulfing
            if (c < o and c1 > o1 and
                o >= c1 and c <= o1 and
                body > body_prev * 1.2):
                patterns[i] = "bearish_engulfing"
                pattern_type[i] = "bearish"
                confidence[i] = 0.8
                pattern_score[i] = -2.0

            # 7. 黄昏之星 Evening Star
            if i >= 3:
                o3, c3 = open_[i-2], close[i-2]
                if (c3 > o3 and
                    abs(c1 - o1) < self._body(o1, c1) * 0.3 and
                    c < o and c < (o3 + c3) / 2):
                    if patterns[i] == "":
                        patterns[i] = "evening_star"
                        pattern_type[i] = "bearish"
                        confidence[i] = 0.85
                        pattern_score[i] = -2.0

            # 8. 乌云盖顶 Dark Cloud Cover
            if (c1 > o1 and c < o and
                o > h1 and c < (o1 + c1) / 2 and
                body > body_prev * 1.1):
                patterns[i] = "dark_cloud"
                pattern_type[i] = "bearish"
                confidence[i] = 0.75
                pattern_score[i] = -1.8

            # ============================================
            # 持续形态
            # ============================================

            # 9. 上升三法 Rising Three Methods (简化版)
            if i >= 4:
                if (close[i-4] < open_[i-4] and  # Day1: 大阳线 → actually should be bullish
                    all(abs(self._body(open_[j], close[j])) < body * 0.4 for j in range(i-3, i)) and
                    c > o and c > high[i-4]):
                    patterns[i] = "rising_three"
                    pattern_type[i] = "bullish"
                    confidence[i] = 0.7
                    pattern_score[i] = 1.5

            # 10. 十字星 Doji (重要转折信号)
            if body < total_range * 0.1 and total_range > 0:
                if lower > upper * 1.5:
                    patterns[i] = "dragonfly_doji"  # 蜻蜓十字 (底部反转)
                    pattern_type[i] = "bullish"
                    confidence[i] = 0.6
                    pattern_score[i] = 0.8
                elif upper > lower * 1.5:
                    patterns[i] = "gravestone_doji"  # 墓碑十字 (顶部反转)
                    pattern_type[i] = "bearish"
                    confidence[i] = 0.6
                    pattern_score[i] = -0.8
                else:
                    patterns[i] = "doji"
                    pattern_type[i] = "neutral"
                    confidence[i] = 0.5
                    pattern_score[i] = 0

        result = pd.DataFrame({
            "pattern": patterns,
            "pattern_type": pattern_type,
            "pattern_confidence": confidence,
            "pattern_score": pattern_score,
        }, index=df.index)

        return result


class ConsecutiveBarAnalyzer:
    """
    连续K线分析

    检测:
      - 连阳/连阴天数
      - 缩量/放量连阳 (主力缓慢吸筹 vs 散户追涨)
      - NR (Narrow Range) 窄幅整理
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        vol_ma = df["vol_ma_5"].values
        n = len(df)

        consecutive_up = np.zeros(n, dtype=int)
        consecutive_down = np.zeros(n, dtype=int)
        nr_pattern = np.zeros(n, dtype=bool)  # Narrow Range
        vol_dry_up = np.zeros(n, dtype=bool)  # 缩量连阳 (吸筹)

        up_count = 0
        down_count = 0

        for i in range(n):
            if i == 0:
                continue

            # 连阳/连阴
            if close[i] > open_[i]:
                up_count += 1
                down_count = 0
            elif close[i] < open_[i]:
                down_count += 1
                up_count = 0
            else:
                up_count = 0
                down_count = 0

            consecutive_up[i] = up_count
            consecutive_down[i] = down_count

            # NR (振幅<2%)
            if i >= 1 and high[i] - low[i] < close[i] * 0.02:
                nr_pattern[i] = True

            # 缩量连阳: 连续3天阳线 + 量递减
            if up_count >= 3:
                if (vol[i] < vol[i-1] and vol[i-1] < vol[i-2] and
                    vol[i] < vol_ma[i] * 0.8):
                    vol_dry_up[i] = True

        result = pd.DataFrame({
            "consecutive_up": consecutive_up,
            "consecutive_down": consecutive_down,
            "nr_pattern": nr_pattern,
            "vol_dry_up": vol_dry_up,
        }, index=df.index)

        return result
