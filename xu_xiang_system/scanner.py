"""
XuanStock Scanner - Xu Xiang Upgraded Edition
Five-dimension scoring system for A-share stock selection.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from .market_env import MarketEnvClassifier, MarketSnapshot
from .capital_structure import CapitalStructureAnalyzer, quick_profile


@dataclass
class ScanResult:
    symbol: str = ""
    name: str = ""
    score: float = 0
    env_level: str = ""
    trend_score: float = 0
    volume_score: float = 0
    pattern_score: float = 0
    capital_score: float = 0
    theme_score: float = 0
    close: float = 0
    mcap: float = 0
    turnover: float = 0
    ret_5d: float = 0
    ret_20d: float = 0
    ma_bullish: bool = False
    vol_ratio: float = 0
    capital_profile: dict = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    action: str = ""


class XuXiangScanner:
    def __init__(self):
        self.env_classifier = MarketEnvClassifier()
        self.capital_analyzer = CapitalStructureAnalyzer()

    def scan_single(self, row, env_snapshot, theme_score=0.5):
        close = float(row.get("close", 0))
        flags = []
        trend_score = 0.0
        if row.get("ma_bullish", 0) == 1:
            trend_score += 15
        ma5 = row.get("ma_5", 0) or 0
        ma20 = row.get("ma_20", 0) or 0
        if ma5 > ma20:
            trend_score += 5
        ma60 = row.get("ma_60", close * 2) or close * 2
        if close > ma60:
            trend_score += 5
        ret5 = row.get("ret_5d", 0) or 0
        if ret5 > 0.05:
            trend_score += 5
        elif ret5 > 0.02:
            trend_score += 3
        elif ret5 < -0.05:
            trend_score -= 5
            flags.append("drop_too_fast")
        vol_ratio = row.get("vol_ratio", 0) or 1.0
        pct_chg = row.get("pct_chg", 0) or 0
        if vol_ratio > 2.0:
            volume_score = 20 if pct_chg > 0.03 else 15
        elif vol_ratio > 1.3:
            volume_score = 12
        elif vol_ratio > 0.8:
            volume_score = 8
        else:
            volume_score = 3
        pattern_score = 0.0
        hh20 = row.get("hh_20d", 0) or 0
        if hh20 and close >= hh20 * 0.98:
            pattern_score += 12
            flags.append("near_20d_high")
        boll_w = row.get("boll_width", 0.2) or 0.2
        if boll_w < 0.1:
            pattern_score += 5
        macd_h = row.get("macd_hist", 0) or 0
        prev_m = row.get("macd_hist_prev", 0) or 0
        if macd_h > 0 and prev_m <= 0:
            pattern_score += 5
            flags.append("macd_golden")
        rsi = row.get("rsi", 50) or 50
        if 40 < rsi < 70:
            pattern_score += 3
        cap_profile = quick_profile(row)
        dom = cap_profile.get("dominant", "mixed")
        if dom == "retail": capital_score = 18
        elif dom == "mixed": capital_score = 14
        elif dom == "north": capital_score = 10
        else: capital_score = 5
        if cap_profile.get("quant_vul"):
            capital_score -= 8
            flags.append("quant_heavy")
        flags.extend(cap_profile.get("flags", []))
        theme = min(15, theme_score * 15)
        raw_score = trend_score + volume_score + pattern_score + capital_score + theme
        if env_snapshot.level == "A": env_mult = 1.0
        elif env_snapshot.level == "B":
            env_mult = 0.8
            if vol_ratio > 2.0: raw_score *= 0.7
        else: env_mult = 0.0
        final_score = raw_score * env_mult
        if final_score >= 70 and env_snapshot.level == "A": action = "chase_board"
        elif final_score >= 55: action = "dip_buy"
        elif final_score >= 35 and env_snapshot.level != "C": action = "watchlist"
        else: action = "skip"
        return ScanResult(
            symbol=str(row.get("symbol", "")), name=str(row.get("name", "")),
            score=round(final_score, 1), env_level=env_snapshot.level,
            trend_score=trend_score, volume_score=volume_score,
            pattern_score=pattern_score, capital_score=capital_score,
            theme_score=theme, close=close,
            mcap=float(row.get("mcap", 0) or 0),
            turnover=float(row.get("turnover", 0) or 0),
            ret_5d=ret5, ret_20d=float(row.get("ret_20d", 0) or 0),
            ma_bullish=bool(row.get("ma_bullish", 0)),
            vol_ratio=vol_ratio, capital_profile=cap_profile,
            flags=flags, action=action)

    def scan_batch(self, stocks_df, index_df, target_date=None):
        env_df = self.env_classifier.classify_series(index_df)
        if target_date is None: target_date = stocks_df["date"].max()
        env_row = env_df[env_df["date"] == target_date]
        if len(env_row) == 0:
            target_date = env_df["date"].max()
            env_row = env_df[env_df["date"] == target_date]
        er = env_row.iloc[0]
        snap = MarketSnapshot(
            date=target_date, level=str(er["env_level"]),
            index_close=0, ma_60=0, ma_direction="",
            volume_100m=0, limit_up_count=0, limit_down_count=0,
            adv_decline_ratio=0.5, score=float(er["env_score"]),
            is_tradeable=bool(er["is_tradeable"]),
            suggested_action=str(er.get("suggested_action", "")))
        day_data = stocks_df[stocks_df["date"] == target_date]
        results = []
        for _, row in day_data.iterrows():
            try: results.append(self.scan_single(row, snap))
            except Exception: continue
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def print_results(self, results, top_n=20):
        if not results:
            print("\n  No matching stocks")
            return
        print()
        print("=" * 90)
        print(f"  XuXiang Scanner | Env: {results[0].env_level}")
        print("=" * 90)
        for i, r in enumerate(results[:top_n], 1):
            print(f"  {i:<4}{r.symbol:<8}{r.name:<8} score={r.score:<6.1f} "
                  f"5d={r.ret_5d:>+6.1%} action={r.action}")
            if r.flags:
                print(f"       flags: {', '.join(r.flags[:4])}")
        print(f"\n  Total: {len(results)}, showing top {min(top_n, len(results))}")
