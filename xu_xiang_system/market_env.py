
"""
市场环境分类器 —— 徐翔体系中"永远顺势"的量化实现

将市场分为 A/B/C 三级:
  A级: 中期均线多头 + 成交额>万亿 + 涨停家数>60 → 积极做多，可追板
  B级: 震荡 + 量能5000-10000亿 + 板块轮动 → 潜伏低吸，不追板
  C级: 均线空头 + 缩量 + 跌停潮 → 空仓/逆回购
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional

EnvLevel = Literal["A", "B", "C", "unknown"]


@dataclass
class MarketSnapshot:
    """市场快照"""
    date: pd.Timestamp
    level: EnvLevel
    index_close: float
    ma_60: float
    ma_direction: str
    volume_100m: float
    limit_up_count: int
    limit_down_count: int
    adv_decline_ratio: float
    score: float
    is_tradeable: bool
    suggested_action: str


class MarketEnvClassifier:
    """市场环境 A/B/C 分级分类器"""

    def __init__(self, ma_period=60, vol_threshold_a=1.5, vol_threshold_b=1.0,
                 limit_up_a=50, limit_up_b=25):
        self.ma_period = ma_period
        self.vol_threshold_a = vol_threshold_a
        self.vol_threshold_b = vol_threshold_b
        self.limit_up_a = limit_up_a
        self.limit_up_b = limit_up_b

    def classify_daily(self, date, close, ma_60, volume_100m,
                       limit_up, limit_down, adv_count, decline_count):
        """
        Classify single day's market environment.

        Parameters
        ----------
        volume_100m : float
            Volume ratio (current / 20-day avg). Values > 1.5 = high volume,
            < 0.7 = low volume. NOT absolute turnover in 100M yuan.
        """
        # 趋势分数 (0-40)
        if pd.notna(ma_60) and ma_60 > 0:
            price_vs_ma = close / ma_60
            if price_vs_ma > 1.08:
                trend_score = 38; ma_direction = "up"
            elif price_vs_ma > 1.03:
                trend_score = 30; ma_direction = "up"
            elif price_vs_ma > 1.00:
                trend_score = 22; ma_direction = "up"
            elif price_vs_ma > 0.97:
                trend_score = 15; ma_direction = "flat"
            elif price_vs_ma > 0.92:
                trend_score = 8; ma_direction = "down"
            else:
                trend_score = 2; ma_direction = "down"
        else:
            trend_score = 0; ma_direction = "flat"

        # 量能分数 (0-30) - uses volume RATIO not absolute
        if volume_100m >= 2.0:
            vol_score = 30
        elif volume_100m >= self.vol_threshold_a:  # 1.5
            vol_score = 25
        elif volume_100m >= self.vol_threshold_b:  # 1.0
            vol_score = 18
        elif volume_100m >= 0.7:
            vol_score = 10
        else:
            vol_score = 3

        # 情绪分数 (0-20) - based on estimated limit up/down
        if limit_up >= self.limit_up_a and limit_down < 15:
            sentiment_score = 20
        elif limit_up >= self.limit_up_b:
            sentiment_score = 14
        elif limit_up >= 15:
            sentiment_score = 8
        else:
            sentiment_score = 3
        if limit_down > 80:
            sentiment_score = max(0, sentiment_score - 10)
        elif limit_down > 40:
            sentiment_score = max(0, sentiment_score - 5)

        # 涨跌比分数 (0-10)
        total = adv_count + decline_count
        ratio = adv_count / total if total > 0 else 0.5
        ratio_score = min(10, ratio * 12)

        total_score = trend_score + vol_score + sentiment_score + ratio_score

        if total_score >= 70:
            level = "A"
        elif total_score >= 40:
            level = "B"
        else:
            level = "C"

        if level == "A":
            action = "积极做多: 可追板、可接力、可重仓"
        elif level == "B":
            action = "谨慎做多: 潜伏低吸为主，不追板"
        else:
            action = "空仓观望: 逆回购，不做任何交易"

        return MarketSnapshot(
            date=date, level=level, index_close=close, ma_60=ma_60,
            ma_direction=ma_direction, volume_100m=volume_100m,
            limit_up_count=limit_up, limit_down_count=limit_down,
            adv_decline_ratio=ratio, score=total_score,
            is_tradeable=(level != "C"), suggested_action=action)

    def classify_series(self, df):
        """对整个时间序列进行环境分类"""
        snapshots = []
        for i in range(len(df)):
            row = df.iloc[i]
            snap = self.classify_daily(
                date=row.get("date", pd.Timestamp.now()),
                close=row.get("close", 0),
                ma_60=row.get("ma_60", 0),
                volume_100m=row.get("volume_100m", 0),
                limit_up=int(row.get("limit_up", 0)),
                limit_down=int(row.get("limit_down", 0)),
                adv_count=int(row.get("adv_count", 0)),
                decline_count=int(row.get("dec_count", 0)))
            snapshots.append(snap)

        return pd.DataFrame([{
            "date": s.date, "env_level": s.level, "env_score": s.score,
            "is_tradeable": s.is_tradeable, "ma_direction": s.ma_direction,
            "volume_100m": s.volume_100m, "limit_up": s.limit_up_count,
            "limit_down": s.limit_down_count,
            "adv_dec_ratio": s.adv_decline_ratio,
            "suggested_action": s.suggested_action} for s in snapshots])

    def get_env_stats(self, env_df):
        """统计各环境级别占比"""
        total = len(env_df)
        if total == 0:
            return {}
        stats = {}
        for level in ["A", "B", "C"]:
            days = (env_df["env_level"] == level).sum()
            stats[level] = {"days": int(days), "pct": round(days / total * 100, 1)}
        return stats


def build_index_env_df(index_df):
    """
    Build environment analysis DataFrame from index data.

    Since akshare index data only has OHLCV (no market breadth),
    we estimate advance/decline and limit up/down counts from
    the index's own price behavior.

    Estimation logic:
      - Big up days (>2%) → lots of limit ups, few limit downs
      - Big down days (<-2%) → lots of limit downs
      - Volume amplifies the signal
    """
    df = index_df.copy()
    df["ma_60"] = df["close"].rolling(60).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    pct_chg = df["close"].pct_change().fillna(0)
    abs_chg = pct_chg.abs()

    # Use volume ratio (current/20d avg) as the primary volume metric
    # This works with any volume unit (shares, lots, yuan)
    if "volume" in df.columns:
        vol_ma20 = df["volume"].rolling(20).mean()
        df["volume_100m"] = (df["volume"] / vol_ma20.replace(0, np.nan)).fillna(1.0)
        vol_ratio = df["volume_100m"].values
    elif "amount" in df.columns:
        amt_ma20 = df["amount"].rolling(20).mean()
        df["volume_100m"] = (df["amount"] / amt_ma20.replace(0, np.nan)).fillna(1.0)
        vol_ratio = df["volume_100m"].values
    else:
        df["volume_100m"] = 1.0
        vol_ratio = np.ones(len(df))

    # --- Estimate advance/decline counts ---
    # Base: 2000 advance, 2000 decline (neutral)
    # Strong day: 3000+ advance
    # Weak day: 3000+ decline
    adv_base = 2000
    dec_base = 2000

    # Adjust based on index return
    adv_adjust = np.clip(pct_chg * 50000, -1500, 2000)
    adv = (adv_base + adv_adjust).astype(int)
    dec = (dec_base - adv_adjust).astype(int)

    # Floor at 100
    adv = np.maximum(adv, 100)
    dec = np.maximum(dec, 100)
    df["adv_count"] = adv
    df["dec_count"] = dec

    # --- Estimate limit up/down counts ---
    # Index up 3%+ with high volume → ~200 limit ups (A-share frenzy)
    # Index up 1-3% → ~60-120 limit ups
    # Index flat → ~30-50 limit ups
    # Index down 1-3% → ~20-40 limit ups, more limit downs
    # Index down 3%+ → ~100-500 limit downs (panic)

    pct = pct_chg.fillna(0).values * 100  # percentage

    # Limit ups scale with index return and volume
    limit_up = np.zeros(len(df), dtype=int)
    limit_down = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        p = pct[i]
        vr = vol_ratio[i] if i >= 20 else 1.0

        if p > 3:
            limit_up[i] = int(150 + p * 20 * vr)
            limit_down[i] = max(0, int(10 - p * 2))
        elif p > 2:
            limit_up[i] = int(80 + p * 15 * vr)
            limit_down[i] = int(15 - p * 2)
        elif p > 1:
            limit_up[i] = int(40 + p * 20 * vr)
            limit_down[i] = int(25 - p * 5)
        elif p > 0:
            limit_up[i] = int(30 + p * 10 * vr)
            limit_down[i] = int(30 - p * 5)
        elif p > -1:
            limit_up[i] = int(30 + p * 5)
            limit_down[i] = int(30 - p * 10)
        elif p > -2:
            limit_up[i] = max(0, int(20 + p * 5))
            limit_down[i] = int(40 - p * 15 * vr)
        elif p > -3:
            limit_up[i] = max(0, int(10 + p * 2))
            limit_down[i] = int(60 - p * 20 * vr)
        else:
            limit_up[i] = max(0, int(5))
            limit_down[i] = int(100 - p * 30 * vr)

        limit_up[i] = max(0, limit_up[i])
        limit_down[i] = max(0, limit_down[i])

    df["limit_up"] = limit_up
    df["limit_down"] = limit_down

    return df


if __name__ == "__main__":
    classifier = MarketEnvClassifier()
    test_cases = [
        ("2020-07-06", 3500, 3450, 15000, 150, 5, 3500, 500),
        ("2023-03-15", 3250, 3200, 8500, 40, 15, 2200, 1800),
        ("2018-10-11", 2600, 3000, 3500, 10, 200, 300, 3700),
        ("2024-09-30", 3350, 3100, 26000, 200, 0, 4500, 100),
    ]
    for date_str, close, ma60, vol, lu, ld, adv, dec in test_cases:
        snap = classifier.classify_daily(
            pd.Timestamp(date_str), close, ma60, vol, lu, ld, adv, dec)
        print(f"{snap.date.date()} | {snap.level} | score={snap.score:.0f} | {snap.suggested_action}")
