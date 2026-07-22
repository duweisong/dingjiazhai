"""
市场状态过滤器 & 量价分析模块

提供:
  1. MarketRegimeFilter  - 市场状态识别 (趋势/震荡/高波动)
  2. VolumePriceAnalyzer - 量价关系分析 (主力资金流向)
  3. ATRDynamicStops     - ATR自适应止损止盈
"""

import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional


class MarketRegimeFilter:
    """
    市场状态过滤器

    判断当前市场处于:
      - trending_up   : 上升趋势
      - trending_down : 下降趋势
      - ranging       : 震荡
      - high_vol      : 高波动
    """

    def __init__(
        self,
        ma_fast: int = 20,
        ma_slow: int = 60,
        adx_period: int = 14,
        adx_threshold: int = 20,
        vol_period: int = 20,
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_period = vol_period

    def compute_adx(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 ADX (平均趋向指数)"""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        n = len(df)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

        # Wilder's smoothing
        atr = pd.Series(tr).ewm(alpha=1/self.adx_period, adjust=False).mean().values
        plus_di = pd.Series(plus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean().values
        minus_di = pd.Series(minus_dm).ewm(alpha=1/self.adx_period, adjust=False).mean().values

        # ADX
        dx = np.zeros(n)
        for i in range(n):
            denom = plus_di[i] + minus_di[i]
            dx[i] = abs(plus_di[i] - minus_di[i]) / denom * 100 if denom > 0 else 0

        adx = pd.Series(dx).ewm(alpha=1/self.adx_period, adjust=False).mean().values

        return adx, plus_di, minus_di

    def get_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        返回每个交易日的市场状态标签

        Returns DataFrame with columns:
            regime       : 'trending_up' | 'trending_down' | 'ranging'
            is_tradeable : bool - 是否适合交易
            trend_strength : float - 趋势强度 (ADX归一化)
        """
        close = df["close"].values
        n = len(df)

        adx, plus_di, minus_di = self.compute_adx(df)
        ma_f = df["close"].rolling(self.ma_fast).mean().values
        ma_s = df["close"].rolling(self.ma_slow).mean().values

        regimes = []
        tradeable = []
        strengths = []

        for i in range(n):
            if i < self.ma_slow:
                regimes.append("unknown")
                tradeable.append(False)
                strengths.append(0)
                continue

            # 趋势判断
            price_above_long = close[i] > ma_s[i]
            ma_bullish = ma_f[i] > ma_s[i]
            adx_strong = adx[i] > self.adx_threshold
            di_bullish = plus_di[i] > minus_di[i]

            # 分类
            if adx_strong and ma_bullish and di_bullish:
                regimes.append("trending_up")
                tradeable.append(True)
            elif adx_strong and not ma_bullish and not di_bullish:
                regimes.append("trending_down")
                tradeable.append(False)  # A股只能做多，下降趋势不开仓
            else:
                regimes.append("ranging")
                tradeable.append(True)   # 震荡市可以做波段

            strengths.append(adx[i] / 100)

        result = pd.DataFrame({
            "regime": regimes,
            "is_tradeable": tradeable,
            "trend_strength": strengths,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "ma_fast": ma_f,
            "ma_slow": ma_s,
        }, index=df.index)

        return result


class VolumePriceAnalyzer:
    """
    量价关系分析器

    检测:
      - 放量上涨 (主力吸筹)
      - 放量下跌 (主力出货)
      - 缩量调整 (正常回调)
      - 量价背离 (反转信号)
    """

    def __init__(self, vol_ma_period: int = 20, surge_multiplier: float = 1.5):
        self.vol_ma_period = vol_ma_period
        self.surge_multiplier = surge_multiplier

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析量价关系

        Returns DataFrame with:
            vol_ratio     : 相对均量的倍数
            vp_score      : 量价综合评分 (-1到+1，正=看涨，负=看跌)
            accumulation  : 是否在吸筹
            distribution  : 是否在出货
            volume_climax : 量能高潮
        """
        close = df["close"].values
        vol = df["volume"].values
        n = len(df)

        vol_ma = df["volume"].rolling(self.vol_ma_period).mean().values
        price_chg = df["close"].pct_change().values

        vol_ratio = np.zeros(n)
        vp_score = np.zeros(n)
        accumulation = np.zeros(n, dtype=bool)
        distribution = np.zeros(n, dtype=bool)
        volume_climax = np.zeros(n, dtype=bool)
        money_flow = np.zeros(n)

        for i in range(1, n):
            if vol_ma[i] > 0:
                vol_ratio[i] = vol[i] / vol_ma[i]
            else:
                vol_ratio[i] = 1.0

            # 量价评分
            if vol_ratio[i] > self.surge_multiplier:
                if price_chg[i] > 0.02:       # 放量涨2%+
                    vp_score[i] = 0.8
                    accumulation[i] = True
                    money_flow[i] = vol[i] * price_chg[i] * close[i]
                elif price_chg[i] < -0.02:     # 放量跌2%+
                    vp_score[i] = -0.8
                    distribution[i] = True
                    money_flow[i] = vol[i] * price_chg[i] * close[i]
                elif price_chg[i] > 0:          # 放量小涨
                    vp_score[i] = 0.4
                else:                            # 放量小跌
                    vp_score[i] = -0.4
            elif vol_ratio[i] < 0.5:             # 缩量
                if price_chg[i] > 0.01:
                    vp_score[i] = 0.3           # 缩量上涨 = 健康
                elif price_chg[i] < -0.01:
                    vp_score[i] = -0.1          # 缩量下跌 = 正常调整
                else:
                    vp_score[i] = 0.1
            else:
                if price_chg[i] > 0:
                    vp_score[i] = 0.2
                else:
                    vp_score[i] = -0.2

            # 量能高潮检测 (成交量是均量的3倍以上)
            if vol_ratio[i] > 3.0:
                volume_climax[i] = True

            # 累积资金流
            if i >= 1 and vol_ma[i-1] > 0:
                money_flow[i] = money_flow[i-1] + (vol[i] - vol_ma[i]) * price_chg[i]

        result = pd.DataFrame({
            "vol_ratio": vol_ratio,
            "vp_score": vp_score,
            "accumulation": accumulation,
            "distribution": distribution,
            "volume_climax": volume_climax,
            "money_flow_cum": np.cumsum(money_flow),
            "money_flow": money_flow,
        }, index=df.index)

        return result

    def get_entry_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        返回开仓过滤条件

        Returns DataFrame with:
            allow_long  : 是否允许做多
            confidence  : 信度 0-1
            reason      : 原因
            vol_ok      : 量能是否配合
        """
        vp = self.analyze(df)
        close = df["close"].values
        n = len(df)

        allow_long = np.zeros(n, dtype=bool)
        confidence = np.zeros(n)
        reason = [""] * n

        for i in range(1, n):
            # 禁止开仓的情况
            if vp["distribution"].iloc[i]:
                allow_long[i] = False
                reason[i] = "主力出货迹象"
                confidence[i] = 0
            elif vp["volume_climax"].iloc[i] and close[i] < close[i-1]:
                allow_long[i] = False
                reason[i] = "量能高潮+下跌，出货"
                confidence[i] = 0
            elif vp["accumulation"].iloc[i]:
                allow_long[i] = True
                reason[i] = "主力吸筹迹象"
                confidence[i] = 0.8
            elif vp["vol_ratio"].iloc[i] > 1.0 and close[i] > close[i-1]:
                allow_long[i] = True
                reason[i] = "放量上涨"
                confidence[i] = 0.5
            elif vp["vol_ratio"].iloc[i] < 0.5 and close[i] < close[i-1]:
                allow_long[i] = False
                reason[i] = "缩量下跌"
                confidence[i] = 0.1
            else:
                allow_long[i] = True
                reason[i] = "正常"
                confidence[i] = 0.3

        result = pd.DataFrame({
            "allow_long": allow_long,
            "confidence": confidence,
            "reason": reason,
            "vol_ok": vp["vol_ratio"].values > 0.8,
        }, index=df.index)

        return result


class ATRDynamicStops:
    """
    ATR 自适应止损止盈

    根据当前波动率动态调整止损/止盈幅度
    高波动 → 宽止损 (避免被噪音震出)
    低波动 → 紧止损 (保护利润)
    """

    def __init__(self, atr_mult_stop: float = 2.0, atr_mult_target: float = 3.0,
                 min_stop_pct: float = 0.02, max_stop_pct: float = 0.08):
        self.atr_mult_stop = atr_mult_stop
        self.atr_mult_target = atr_mult_target
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct

    def compute_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个交易日的动态止损止盈水平

        Returns DataFrame with:
            atr_stop_pct    : 动态止损比例
            atr_target_pct  : 动态止盈比例
            atr_value       : 当日ATR值
        """
        atr = df["atr"].values
        close = df["close"].values
        n = len(df)

        stop_pcts = np.zeros(n)
        target_pcts = np.zeros(n)
        atr_pcts = np.zeros(n)

        for i in range(n):
            if i < 20 or close[i] <= 0:
                stop_pcts[i] = self.min_stop_pct
                target_pcts[i] = self.min_stop_pct * 2
                continue

            atr_pct = atr[i] / close[i]

            # 止损: ATR * 倍数, 限制在 min-max 范围内
            raw_stop = atr_pct * self.atr_mult_stop
            stop_pcts[i] = np.clip(raw_stop, self.min_stop_pct, self.max_stop_pct)

            # 止盈: 止损的 2-3 倍
            raw_target = atr_pct * self.atr_mult_target
            target_pcts[i] = np.clip(raw_target, self.min_stop_pct * 2, self.max_stop_pct * 2)

            atr_pcts[i] = atr_pct

        result = pd.DataFrame({
            "atr_stop_pct": stop_pcts,
            "atr_target_pct": target_pcts,
            "atr_pct": atr_pcts,
        }, index=df.index)

        return result
