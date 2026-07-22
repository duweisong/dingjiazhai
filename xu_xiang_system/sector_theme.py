"""
Sector Theme Heat Detection Module

Detects which industry sectors are "hot" (in rotation) and assigns
theme scores to individual stocks based on their sector membership.

Core capabilities:
  1. Sector momentum ranking (multi-timeframe)
  2. Theme rotation speed detection
  3. Individual stock theme score assignment
  4. Theme fade / overcrowding warning

This replaces the crude "题材热度" proxy in the original scanner
with a data-driven approach.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================
# Sector Definitions (Shenwan Industry Classification)
# ============================================================
SECTOR_STOCKS = {
    "food_beverage": {
        "name": "食品饮料",
        "stocks": ["600519", "000858", "000568", "600809", "002304"],
    },
    "electronics": {
        "name": "电子",
        "stocks": ["002036", "002475", "002241", "300433", "603501"],
    },
    "semiconductor": {
        "name": "半导体",
        "stocks": ["002371", "688981", "603986", "300782", "688012"],
    },
    "pharma": {
        "name": "医药生物",
        "stocks": ["600276", "300760", "000538", "300122", "002007"],
    },
    "new_energy": {
        "name": "新能源",
        "stocks": ["300750", "002594", "601012", "688599", "300274"],
    },
    "finance": {
        "name": "金融",
        "stocks": ["601318", "600036", "601688", "600030", "000001"],
    },
    "media_internet": {
        "name": "传媒互联网",
        "stocks": ["300364", "002624", "300418", "002555", "300251"],
    },
    "building_materials": {
        "name": "建材",
        "stocks": ["600585", "000786", "002271", "600176", "000877"],
    },
    "auto": {
        "name": "汽车",
        "stocks": ["002594", "000625", "601238", "600104", "000800"],
    },
    "computer": {
        "name": "计算机",
        "stocks": ["002230", "600570", "300033", "002410", "600536"],
    },
}


# ============================================================
# SectorThemeAnalyzer
# ============================================================
@dataclass
class SectorSnapshot:
    """Snapshot of a sector's momentum at a point in time"""
    sector_key: str
    sector_name: str
    ret_5d: float           # 5-day return
    ret_20d: float          # 20-day return
    ret_60d: float          # 60-day return
    momentum_score: float   # composite momentum 0-100
    vol_ratio: float        # relative volume
    rank_5d: int            # ranking among all sectors
    rank_20d: int
    is_hot: bool            # is this sector "hot" right now?
    rotation_speed: float   # how fast themes are rotating (cross-sector)


@dataclass
class ThemeSignal:
    """Theme signal for a stock"""
    symbol: str
    sector_key: str
    sector_name: str
    theme_score: float          # 0-1, how hot is the theme
    sector_momentum: float      # sector's own momentum
    relative_strength: float    # stock vs sector strength
    is_leader: bool             # is this stock leading its sector?
    crowding_warning: bool      # is the theme getting crowded?
    fade_risk: float            # 0-1, probability of theme fading soon


class SectorThemeAnalyzer:
    """
    Sector and theme heat analyzer.

    Works by:
      1. Computing sector aggregate momentum from constituent stocks
      2. Ranking sectors by multi-timeframe momentum
      3. Detecting rotation speed (how fast money moves between sectors)
      4. Scoring individual stocks based on sector heat + relative strength
    """

    def __init__(self):
        self.sector_map = {
            s: info for s, info in SECTOR_STOCKS.items()
        }
        # Build reverse map: stock -> sector
        self.stock_sector = {}
        for sk, info in self.sector_map.items():
            for sym in info["stocks"]:
                self.stock_sector[sym] = sk

    def compute_sector_momentum(self, sector_data: Dict[str, pd.DataFrame],
                                 target_date) -> List[SectorSnapshot]:
        """
        Compute momentum scores for all sectors at a target date.

        Parameters
        ----------
        sector_data : dict of sector_key -> DataFrame with date, close columns
        target_date : the date to snapshot

        Returns
        -------
        List of SectorSnapshot sorted by momentum_score descending
        """
        snapshots = []

        for sk, info in self.sector_map.items():
            if sk not in sector_data:
                continue

            df = sector_data[sk]
            df_sub = df[df["date"] <= target_date]

            if len(df_sub) < 60:
                continue

            close = df_sub["close"]
            ret_5d = (close.iloc[-1] / close.iloc[-min(6, len(close))] - 1)
            ret_20d = (close.iloc[-1] / close.iloc[-min(21, len(close))] - 1)
            ret_60d = (close.iloc[-1] / close.iloc[-min(61, len(close))] - 1)

            # Multi-timeframe momentum score (0-100)
            mom = (
                max(0, min(100, ret_5d * 300 + 50)) * 0.3 +
                max(0, min(100, ret_20d * 150 + 50)) * 0.4 +
                max(0, min(100, ret_60d * 80 + 50)) * 0.3
            )

            # Volume ratio
            if "volume" in df_sub.columns:
                vol_ma20 = df_sub["volume"].rolling(20).mean()
                vol_ratio = (df_sub["volume"].iloc[-1] /
                             vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] > 0 else 1.0
            else:
                vol_ratio = 1.0

            snapshots.append(SectorSnapshot(
                sector_key=sk,
                sector_name=info["name"],
                ret_5d=ret_5d,
                ret_20d=ret_20d,
                ret_60d=ret_60d,
                momentum_score=mom,
                vol_ratio=vol_ratio,
                rank_5d=0,
                rank_20d=0,
                is_hot=False,
                rotation_speed=0.0,
            ))

        # Rank by momentum
        snapshots.sort(key=lambda s: s.momentum_score, reverse=True)
        for i, snap in enumerate(snapshots):
            snap.rank_5d = i + 1

        snapshots_20d = sorted(snapshots, key=lambda s: s.ret_20d, reverse=True)
        for i, snap in enumerate(snapshots_20d):
            snap.rank_20d = i + 1

        # Mark hot sectors (top 3 by momentum, or score > 70)
        for snap in snapshots:
            snap.is_hot = (snap.rank_5d <= 3 and snap.momentum_score > 50)

        # Rotation speed: std of sector returns (high = fast rotation)
        if len(snapshots) >= 3:
            rets_5d = [s.ret_5d for s in snapshots]
            rotation_speed = np.std(rets_5d) * 5  # scale up
            for snap in snapshots:
                snap.rotation_speed = rotation_speed

        return snapshots

    def get_sector_index(self, sector_key: str,
                          stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build a synthetic sector index from constituent stocks.
        Equal-weighted average of normalized prices.
        """
        stocks = self.sector_map.get(sector_key, {}).get("stocks", [])
        available = [s for s in stocks if s in stock_data]

        if not available:
            return pd.DataFrame()

        # Align all stock data to common dates
        dfs = []
        for sym in available:
            df = stock_data[sym][["date", "close"]].copy()
            df = df.rename(columns={"close": sym})
            dfs.append(df)

        # Merge on date
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="date", how="inner")

        if len(merged) < 10:
            return pd.DataFrame()

        # Equal-weighted composite (normalized to 100 at start)
        stock_cols = [s for s in available if s in merged.columns]
        normed = merged[stock_cols].div(merged[stock_cols].iloc[0]) * 100
        merged["close"] = normed.mean(axis=1)

        # Synthetic volume: sum of volumes (if available)
        # For now, use 1.0 as placeholder volume
        merged["volume"] = 1.0

        return merged[["date", "close", "volume"]]

    def score_stock(self, symbol: str, stock_row,
                     sector_snapshots: List[SectorSnapshot]) -> ThemeSignal:
        """
        Score a single stock's theme heat.

        Parameters
        ----------
        symbol : stock code
        stock_row : single row of stock data (contains ret_5d, etc.)
        sector_snapshots : list of current sector snapshots

        Returns
        -------
        ThemeSignal
        """
        sector_key = self.stock_sector.get(symbol, "unknown")

        # Find sector snapshot
        sec_snap = None
        for s in sector_snapshots:
            if s.sector_key == sector_key:
                sec_snap = s
                break

        if sec_snap is None:
            return ThemeSignal(
                symbol=symbol, sector_key=sector_key,
                sector_name="unknown", theme_score=0.3,
                sector_momentum=50, relative_strength=1.0,
                is_leader=False, crowding_warning=False, fade_risk=0.5)

        # Theme score components:
        # 1. Sector momentum (0-50 points)
        sector_score = sec_snap.momentum_score * 0.5

        # 2. Hot sector bonus (0-20 points)
        hot_bonus = 20 if sec_snap.is_hot else 0

        # 3. Stock relative strength vs sector (0-20 points)
        stock_ret5 = stock_row.get("ret_5d", 0) or 0
        rel_strength = 1.0
        if sec_snap.ret_5d != 0:
            rel_strength = (1 + stock_ret5) / (1 + sec_snap.ret_5d)
        rs_score = min(20, max(0, (rel_strength - 1) * 200 + 10))

        # 4. Volume confirmation (0-10 points)
        vol_ratio = stock_row.get("vol_ratio", 1.0) or 1.0
        vol_score = min(10, max(0, (vol_ratio - 0.8) * 10))

        theme_score = (sector_score + hot_bonus + rs_score + vol_score) / 100
        theme_score = max(0.05, min(1.0, theme_score))

        # Is leader?
        is_leader = (stock_ret5 > sec_snap.ret_5d * 1.2 and
                     vol_ratio > 1.3 and
                     sec_snap.is_hot)

        # Crowding warning
        crowding = (sec_snap.rotation_speed > 15 and
                    sec_snap.momentum_score > 70)

        # Fade risk
        fade_risk = 0.0
        if sec_snap.ret_5d < sec_snap.ret_20d * 0.3:
            fade_risk += 0.3  # short-term momentum fading
        if sec_snap.ret_20d < 0:
            fade_risk += 0.3  # medium-term negative
        if sec_snap.rotation_speed > 20:
            fade_risk += 0.3  # fast rotation
        fade_risk = min(1.0, fade_risk)

        return ThemeSignal(
            symbol=symbol, sector_key=sector_key,
            sector_name=sec_snap.sector_name,
            theme_score=theme_score,
            sector_momentum=sec_snap.momentum_score,
            relative_strength=rel_strength,
            is_leader=is_leader,
            crowding_warning=crowding,
            fade_risk=fade_risk,
        )

    def analyze_market(self, stock_data: Dict[str, pd.DataFrame],
                        target_date) -> Dict:
        """
        Full market theme analysis at target_date.

        Returns dict with:
          - sectors: List[SectorSnapshot] ranked by momentum
          - stock_themes: Dict[symbol -> ThemeSignal]
          - rotation_speed: float
          - dominant_theme: str (name of hottest sector)
          - theme_count: int (number of hot sectors)
        """
        # Build sector indices
        sector_data = {}
        for sk in self.sector_map:
            sec_df = self.get_sector_index(sk, stock_data)
            if len(sec_df) > 0:
                sector_data[sk] = sec_df

        # Compute sector momentum
        snapshots = self.compute_sector_momentum(sector_data, target_date)

        # Score individual stocks
        stock_themes = {}
        for sym in self.stock_sector:
            if sym in stock_data:
                df = stock_data[sym]
                row = df[df["date"] == target_date]
                if len(row) > 0:
                    stock_themes[sym] = self.score_stock(
                        sym, row.iloc[0], snapshots)

        # Aggregate
        avg_rotation = (np.mean([s.rotation_speed for s in snapshots])
                        if snapshots else 0)
        dominant = snapshots[0].sector_name if snapshots else "none"
        hot_count = sum(1 for s in snapshots if s.is_hot)

        return {
            "sectors": snapshots,
            "stock_themes": stock_themes,
            "rotation_speed": avg_rotation,
            "dominant_theme": dominant,
            "theme_count": hot_count,
        }

    def print_sector_heatmap(self, snapshots: List[SectorSnapshot]):
        """Print sector momentum heatmap"""
        print()
        print("=" * 80)
        print("  Sector Theme Heatmap")
        print("=" * 80)
        print(f"  {'Sector':<16}{'Mom':<8}{'5d':<10}{'20d':<10}{'60d':<10}"
              f"{'Vol':<8}{'Hot':<6}{'Rank':<6}")
        print(f"  {'-'*70}")
        for s in snapshots:
            hot_mark = "HOT" if s.is_hot else ""
            print(f"  {s.sector_name:<16}{s.momentum_score:<8.0f}"
                  f"{s.ret_5d:>+8.1%}  {s.ret_20d:>+8.1%}  "
                  f"{s.ret_60d:>+8.1%}  {s.vol_ratio:<8.2f}"
                  f"{hot_mark:<6}{s.rank_5d:<6}")
        print(f"  Rotation speed: {snapshots[0].rotation_speed:.1f}" if snapshots else "")


# ============================================================
# Theme History Tracker
# ============================================================
class ThemeHistoryTracker:
    """
    Tracks theme rotation over time to detect patterns:
      - How long themes typically last
      - When themes are accelerating or decelerating
      - Sector correlation changes
    """

    def __init__(self, lookback=60):
        self.lookback = lookback
        self.history: List[Dict] = []  # List of daily snapshots

    def record(self, date, analysis: Dict):
        """Record a daily theme snapshot"""
        self.history.append({
            "date": date,
            "dominant": analysis["dominant_theme"],
            "hot_count": analysis["theme_count"],
            "rotation_speed": analysis["rotation_speed"],
            "sector_scores": {
                s.sector_key: s.momentum_score
                for s in analysis["sectors"]
            },
        })
        # Keep only lookback window
        if len(self.history) > self.lookback:
            self.history = self.history[-self.lookback:]

    def get_theme_duration(self, sector_key: str) -> int:
        """How many consecutive days has this sector been dominant?"""
        count = 0
        for h in reversed(self.history):
            if h["dominant"] == sector_key:
                count += 1
            else:
                break
        return count

    def is_accelerating(self, sector_key: str) -> bool:
        """Is the sector's momentum accelerating?"""
        if len(self.history) < 5:
            return False
        recent = [
            h["sector_scores"].get(sector_key, 0)
            for h in self.history[-5:]
        ]
        return (recent[-1] > recent[0] and
                recent[-1] > np.mean(recent[:-1]))


if __name__ == "__main__":
    # Demo
    analyzer = SectorThemeAnalyzer()
    print("Sector definitions loaded:")
    for sk, info in analyzer.sector_map.items():
        print(f"  {sk}: {info['name']} ({len(info['stocks'])} stocks)")
    print(f"  Total stock-sector mappings: {len(analyzer.stock_sector)}")
