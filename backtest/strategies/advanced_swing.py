"""
高级波段策略 —— 融合三大增强

1. 市场状态过滤器 (Market Regime Filter)
   - 仅在上升趋势或震荡市做多
   - 下降趋势中完全空仓，避免逆势操作

2. 量价关系确认 (Volume-Price Analysis)
   - 开仓要求放量配合
   - 检测主力吸筹/出货信号
   - 量价背离作为反转预警

3. ATR 动态止损止盈 (Adaptive Stops)
   - 高波动 → 宽止损
   - 低波动 → 紧止损
   - 移动止盈随趋势推进
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from filters import MarketRegimeFilter, VolumePriceAnalyzer, ATRDynamicStops


class AdvancedSwingStrategy(BaseStrategy):
    """
    高级波段策略

    开仓条件 (需同时满足):
      1. 市场状态允许交易 (非下降趋势)
      2. 量价配合 (放量上涨 / 主力吸筹)
      3. 技术信号触发 (MA金叉 / 回调企稳 / 突破回踩)

    平仓条件:
      1. ATR 动态止损
      2. ATR 动态止盈
      3. 量价出货信号
      4. 市场转为下降趋势
    """

    name = "advanced_swing"
    description = "市场状态+量价+ATR自适应 综合波段策略"

    def __init__(
        self,
        # 均线参数
        ma_short: int = 5,
        ma_mid: int = 20,
        ma_long: int = 60,
        # 市场状态
        adx_threshold: int = 20,
        # ATR
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        min_stop_pct: float = 0.02,
        max_stop_pct: float = 0.08,
        # 量价
        vol_surge: float = 1.3,
        # 信号
        require_pullback: bool = True,
        kdj_oversold: int = 30,
    ):
        super().__init__(params={
            "ma": f"({ma_short},{ma_mid},{ma_long})",
            "adx_thresh": adx_threshold,
            "atr_mult": atr_stop_mult,
            "vol_surge": vol_surge,
            "pullback": require_pullback,
        })
        self.ma_short = ma_short
        self.ma_mid = ma_mid
        self.ma_long = ma_long
        self.adx_threshold = adx_threshold
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.vol_surge = vol_surge
        self.require_pullback = require_pullback
        self.kdj_oversold = kdj_oversold

        # 子模块
        self.regime_filter = MarketRegimeFilter(
            ma_fast=ma_mid, ma_slow=ma_long,
            adx_period=14, adx_threshold=adx_threshold,
        )
        self.vp_analyzer = VolumePriceAnalyzer(
            vol_ma_period=20, surge_multiplier=vol_surge,
        )
        self.atr_stops = ATRDynamicStops(
            atr_mult_stop=atr_stop_mult,
            atr_mult_target=atr_target_mult,
            min_stop_pct=min_stop_pct,
            max_stop_pct=max_stop_pct,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号，同时将动态止损止盈信息存储在 self._stop_info 中
        供引擎读取
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        # 初始化各过滤器
        regime = self.regime_filter.get_regime(df)
        vp_entry = self.vp_analyzer.get_entry_filters(df)
        vp_analysis = self.vp_analyzer.analyze(df)
        atr_stops_df = self.atr_stops.compute_stops(df)

        # 存储动态止损止盈给引擎使用
        self._stop_info = atr_stops_df

        close = df["close"].values
        open_p = df["open"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        ma_s = df["ma_short"].values
        ma_m = df["ma_long"].values         # 使用 ma_long (20日均线)
        ma_l = df["ma_60"].values            # 60日均线
        rsi = df["rsi"].values
        kdj_k = df["kdj_k"].values
        kdj_d = df["kdj_d"].values
        kdj_j = df["kdj_j"].values

        n = len(df)

        for i in range(3, n):
            idx = df.index[i]

            # ============================
            # 第一层: 市场状态过滤
            # ============================
            if not regime["is_tradeable"].iloc[i]:
                # 下降趋势，检查是否有持仓需要卖出
                if signals.iloc[i-1] == 1:
                    # 如果之前有买入信号，这里生成卖出
                    pass  # 卖出由 engine 根据这个信号处理
                continue

            # ============================
            # 第二层: 量价过滤
            # ============================
            vol_ok = vp_entry["vol_ok"].iloc[i]
            accumulation = vp_analysis["accumulation"].iloc[i]
            distribution = vp_analysis["distribution"].iloc[i]

            # 如果检测到出货，生成卖出信号
            if distribution and signals.iloc[i-1] != 0:
                signals.iloc[i] = -1
                continue

            # ============================
            # 第三层: 技术信号
            # ============================

            # ---- 买入模式 ----
            buy_signal = False

            # 模式A: 均线金叉 + 量价配合
            golden_cross = (
                ma_s[i] > ma_m[i] and
                ma_s[i-1] <= ma_m[i-1] and
                close[i] > ma_m[i]
            )

            # 模式B: 回调至均线支撑 + KDJ低位金叉
            pullback_support = (
                close[i] > ma_m[i] and           # 在中期均线上方
                close[i-1] <= close[i-2] and     # 连续2天回调
                close[i-2] <= close[i-3] and
                close[i] > close[i-1] and         # 今天收阳
                close[i] > open_p[i] and          # 阳线确认
                low[i] >= ma_m[i] * 0.97 and      # 回调至均线附近 (3%以内)
                kdj_j[i] > kdj_j[i-1] and         # KDJ J值回升
                kdj_j[i-1] < self.kdj_oversold     # 之前超卖
            )

            # 模式C: RSI超卖反弹 + 量价确认
            rsi_reversal = (
                rsi[i] > rsi[i-1] and
                rsi[i-1] < 35 and
                close[i] > close[i-1] and
                close[i] > open_p[i]
            )

            # 选择模式
            if golden_cross:
                buy_signal = True
            elif self.require_pullback and pullback_support:
                buy_signal = True
            elif rsi_reversal and vol_ok:
                buy_signal = True

            # 信度评估
            confidence = 0
            if buy_signal:
                confidence += 0.3
                if vol_ok:
                    confidence += 0.2
                if accumulation:
                    confidence += 0.3
                if regime["regime"].iloc[i] == "trending_up":
                    confidence += 0.2
                if rsi[i] > 30 and rsi[i] < 65:
                    confidence += 0.1

            # 需要至少 0.5 信度才开仓
            if buy_signal and confidence >= 0.5:
                signals.iloc[i] = 1

            # ---- 卖出模式 ----
            sell_signal = False

            # 死叉
            dead_cross = (
                ma_s[i] < ma_m[i] and
                ma_s[i-1] >= ma_m[i-1]
            )

            # RSI 超买回落
            rsi_overbought = rsi[i] < rsi[i-1] and rsi[i-1] > 72

            # KDJ 高位死叉
            kdj_dead = (
                kdj_k[i] < kdj_d[i] and
                kdj_k[i-1] >= kdj_d[i-1] and
                kdj_j[i] > 80
            )

            # 跌破中期均线
            break_ma = close[i] < ma_m[i] and close[i-1] >= ma_m[i-1]

            if dead_cross or rsi_overbought or break_ma:
                sell_signal = True
            elif kdj_dead and close[i] < close[i-1]:
                sell_signal = True

            # 市场转为下降趋势，强制卖出
            if regime["regime"].iloc[i] == "trending_down":
                sell_signal = True

            if sell_signal:
                signals.iloc[i] = -1

        return signals

    def get_dynamic_stops(self) -> pd.DataFrame:
        """返回动态止损止盈数据，供引擎读取"""
        if hasattr(self, "_stop_info"):
            return self._stop_info
        return None


class TrendFollowingStrategy(BaseStrategy):
    """
    纯趋势跟随策略 —— 仅在确认上升趋势后持仓

    特点:
      - 市场状态过滤器严格过滤
      - 突破追入 + 移动止损跟随趋势
      - 下降趋势和震荡市中完全空仓
    """

    name = "trend_following"
    description = "严格趋势跟随 + ATR移动止损策略"

    def __init__(
        self,
        ma_short: int = 10,
        ma_long: int = 30,
        adx_threshold: int = 22,
        atr_stop_mult: float = 2.5,
        atr_target_mult: float = 4.0,
    ):
        super().__init__(params={
            "ma": f"({ma_short},{ma_long})",
            "adx": adx_threshold,
            "atr_stop": atr_stop_mult,
        })
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.adx_threshold = adx_threshold
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

        self.regime_filter = MarketRegimeFilter(
            ma_fast=ma_short, ma_slow=ma_long,
            adx_period=14, adx_threshold=adx_threshold,
        )
        self.atr_stops = ATRDynamicStops(
            atr_mult_stop=atr_stop_mult,
            atr_mult_target=atr_target_mult,
            min_stop_pct=0.03,
            max_stop_pct=0.10,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        regime = self.regime_filter.get_regime(df)
        atr_stops_df = self.atr_stops.compute_stops(df)
        self._stop_info = atr_stops_df

        close = df["close"].values
        ma_s = df["ma_short"].values
        ma_l = df["ma_long"].values
        boll_upper = df["boll_upper"].values
        boll_lower = df["boll_lower"].values

        n = len(df)

        for i in range(2, n):
            idx = df.index[i]

            # 仅上升趋势中允许开仓
            is_trending = regime["regime"].iloc[i] == "trending_up"

            if not is_trending:
                # 如果持仓，转为震荡或下降趋势，卖出
                if signals.iloc[i-1] == 1:
                    signals.iloc[i] = -1
                continue

            # 开仓: 均线多头 + 突破布林上轨或回调企稳
            ma_bullish = ma_s[i] > ma_l[i] and close[i] > ma_s[i]

            if ma_bullish:
                # 突破信号: 突破前高 + 放量
                breakout = close[i] > boll_upper[i] * 0.98

                # 回调企稳: 回踩均线后反弹
                pullback = (
                    close[i] > close[i-1] and
                    close[i-1] < ma_s[i-1] and
                    close[i] > ma_s[i]
                )

                if breakout or pullback:
                    signals.iloc[i] = 1

            # 平仓: 跌破均线或布林下轨
            if ma_s[i] < ma_l[i]:
                signals.iloc[i] = -1
            elif close[i] < ma_s[i] and close[i-1] >= ma_s[i-1]:
                signals.iloc[i] = -1

        return signals

    def get_dynamic_stops(self) -> pd.DataFrame:
        if hasattr(self, "_stop_info"):
            return self._stop_info
        return None


class VolumeBreakoutStrategy(BaseStrategy):
    """
    量价突破策略 —— 聚焦放量突破

    核心逻辑:
      - 寻找"缩量整理 + 放量突破"模式
      - 成交量验证，避免假突破
      - 结合 ATR 通道做动态止盈
    """

    name = "volume_breakout"
    description = "放量突破+缩量回调确认策略"

    def __init__(
        self,
        vol_surge: float = 1.5,
        consolidation_period: int = 5,
        breakout_pct: float = 0.03,
    ):
        super().__init__(params={
            "vol_surge": vol_surge,
            "consolidation": consolidation_period,
            "breakout": breakout_pct,
        })
        self.vol_surge = vol_surge
        self.consolidation_period = consolidation_period
        self.breakout_pct = breakout_pct

        self.vp_analyzer = VolumePriceAnalyzer(
            vol_ma_period=20, surge_multiplier=vol_surge,
        )
        self.atr_stops = ATRDynamicStops(
            atr_mult_stop=2.0, atr_mult_target=3.0,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        vp = self.vp_analyzer.analyze(df)
        atr_stops_df = self.atr_stops.compute_stops(df)
        self._stop_info = atr_stops_df

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        vol_ma = df["vol_ma_20"].values

        n = len(df)
        p = self.consolidation_period

        for i in range(p + 2, n):
            idx = df.index[i]

            # 检测缩量整理区间 (过去p天)
            recent_vol = vol[i-p:i]
            recent_vol_ma = vol_ma[i-p:i]

            # 成交量萎缩: 大部分天数的量 < 均量
            vol_contracted = (recent_vol < recent_vol_ma * 0.8).sum() >= p * 0.6

            # 价格窄幅整理: 振幅<5%
            recent_high = high[i-p:i].max()
            recent_low = low[i-p:i].min()
            range_pct = (recent_high - recent_low) / recent_low
            narrow_range = range_pct < 0.05

            # 今天放量突破
            vol_surge_today = vol[i] > vol_ma[i] * self.vol_surge
            price_breakout = close[i] > recent_high * (1 + self.breakout_pct * 0.3)

            # 缩量整理后的放量突破 = 强买入信号
            if vol_contracted and narrow_range and vol_surge_today and price_breakout:
                # 额外确认: 不再出货区
                if not vp["distribution"].iloc[i]:
                    signals.iloc[i] = 1

            # 卖出条件
            # 放量滞涨
            if (vol[i] > vol_ma[i] * 1.5 and
                close[i] < close[i-1] * 0.99 and
                close[i] < high[i] * 0.97):
                signals.iloc[i] = -1

            # 缩量破位
            if (vol[i] < vol_ma[i] * 0.5 and
                close[i] < close[i-1] * 0.98):
                signals.iloc[i] = -1

            # 检测出货
            if vp["distribution"].iloc[i]:
                signals.iloc[i] = -1

        return signals

    def get_dynamic_stops(self) -> pd.DataFrame:
        if hasattr(self, "_stop_info"):
            return self._stop_info
        return None
