"""
Adaptive Strategy - Stock-Type-Aware Parameter Selection

Based on 10-stock backtest findings, different stock types need
different strategy parameters:

  Trend Growth (五粮液, NAURA):  Wide stops, longer holds, trend following
  Small Cap Mom (联创电子):      Tight stops, fast exits, volume sensitive
  Blue Chip (茅台, 平安):        Wide MAs, low turnover tolerance
  High Volatility (CATL, BYD):   Higher volume threshold, wider stops
  Cyclical (海螺):               AVOID - counter-trend only
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Any

from .strategy import XuXiangStrategy, StrategyConfig
from .sector_theme import SectorThemeAnalyzer, SECTOR_STOCKS


# ============================================================
# Stock Type Classification
# ============================================================
@dataclass
class StockProfile:
    """Statistical profile of a stock's behavior"""
    symbol: str
    name: str
    avg_volatility: float      # average daily ATR%
    avg_turnover: float        # average daily turnover
    trend_efficiency: float    # ratio of directional movement to total movement
    beta_vs_market: float      # sensitivity to market moves
    mcap_category: str         # "small", "mid", "large"
    sector: str                # sector key
    style: str                 # "trend_growth", "small_mom", "blue_chip",
                               # "high_vol", "cyclical", "balanced"


# Per-type strategy parameter presets
TYPE_PARAMS = {
    "trend_growth": {
        # 五粮液/NAURA type: clear trends, moderate volatility
        "ma_short": 8,
        "ma_mid": 30,
        "ma_long": 60,
        "vol_ratio_min": 1.0,
        "rsi_max": 78,
        "rsi_min": 28,
        "require_ma_bullish": True,
        "require_volume_surge": False,  # Don't require volume surge for trend stocks
        "use_atr_stops": True,
        "atr_stop_mult": 3.0,         # Wider stop - let trends run
        "atr_target_mult": 5.0,       # Wider target
        "trailing_stop_pct": 0.06,    # Wider trailing stop
        "max_position_pct": 0.35,
        "max_holding_days": 15,       # Hold longer
        "description": "Trend Growth: wide stops, long holds",
    },
    "small_mom": {
        # 联创电子 type: small cap, momentum-driven
        "ma_short": 5,
        "ma_mid": 20,
        "ma_long": 50,
        "vol_ratio_min": 1.3,         # Require volume confirmation
        "rsi_max": 72,
        "rsi_min": 32,
        "require_ma_bullish": True,
        "require_volume_surge": True,
        "use_atr_stops": True,
        "atr_stop_mult": 2.0,         # Tighter stop - small caps gap more
        "atr_target_mult": 3.5,
        "trailing_stop_pct": 0.04,
        "max_position_pct": 0.25,
        "max_holding_days": 8,        # Exit faster
        "description": "Small Cap Mom: tight stops, fast exits",
    },
    "blue_chip": {
        # 茅台/平安 type: slow, steady compounders
        "ma_short": 10,
        "ma_mid": 30,
        "ma_long": 90,                # Longer-term MA
        "vol_ratio_min": 0.9,
        "rsi_max": 75,
        "rsi_min": 25,
        "require_ma_bullish": True,
        "require_volume_surge": False,
        "use_atr_stops": True,
        "atr_stop_mult": 2.5,
        "atr_target_mult": 4.0,
        "trailing_stop_pct": 0.05,
        "max_position_pct": 0.30,
        "max_holding_days": 20,       # Hold longest for compounders
        "description": "Blue Chip: slow MA, long holds",
    },
    "high_vol": {
        # CATL/BYD type: high volatility
        "ma_short": 5,
        "ma_mid": 20,
        "ma_long": 60,
        "vol_ratio_min": 1.5,         # Higher volume threshold
        "rsi_max": 80,
        "rsi_min": 25,
        "require_ma_bullish": False,  # Don't require MA alignment in high vol
        "require_volume_surge": True,
        "use_atr_stops": True,
        "atr_stop_mult": 3.5,         # Very wide stop for high vol
        "atr_target_mult": 6.0,
        "trailing_stop_pct": 0.07,
        "max_position_pct": 0.20,     # Smaller position
        "max_holding_days": 10,
        "description": "High Vol: wide stops, small size",
    },
    "cyclical": {
        # 海螺 type: cyclical, avoid chasing
        "ma_short": 10,
        "ma_mid": 30,
        "ma_long": 60,
        "vol_ratio_min": 1.0,
        "rsi_max": 65,                # More conservative RSI
        "rsi_min": 35,
        "require_ma_bullish": True,
        "require_volume_surge": False,
        "use_atr_stops": True,
        "atr_stop_mult": 1.5,         # Very tight stop
        "atr_target_mult": 2.5,
        "trailing_stop_pct": 0.03,
        "max_position_pct": 0.15,     # Small position
        "max_holding_days": 5,        # Exit fast
        "description": "Cyclical: tight stops, small size, fast exit",
    },
    "balanced": {
        # Default fallback
        "ma_short": 5,
        "ma_mid": 20,
        "ma_long": 60,
        "vol_ratio_min": 1.2,
        "rsi_max": 75,
        "rsi_min": 30,
        "require_ma_bullish": True,
        "require_volume_surge": True,
        "use_atr_stops": True,
        "atr_stop_mult": 2.5,
        "atr_target_mult": 4.0,
        "trailing_stop_pct": 0.04,
        "max_position_pct": 0.30,
        "max_holding_days": 10,
        "description": "Balanced: default parameters",
    },
}

# Map sectors to preferred styles
SECTOR_STYLE_MAP = {
    "food_beverage": "trend_growth",
    "electronics": "small_mom",
    "semiconductor": "trend_growth",
    "pharma": "trend_growth",
    "new_energy": "high_vol",
    "finance": "blue_chip",
    "media_internet": "small_mom",
    "building_materials": "cyclical",
    "auto": "high_vol",
    "computer": "small_mom",
}


class StockClassifier:
    """
    Classify stocks into types based on historical data characteristics.

    Uses:
      - Average daily volatility (ATR%)
      - Market cap
      - Beta (correlation with index)
      - Trend efficiency (directional / total movement)
      - Sector membership
    """

    def __init__(self):
        self.theme_analyzer = SectorThemeAnalyzer()

    def classify(self, symbol: str, df: pd.DataFrame,
                  index_df: pd.DataFrame = None) -> StockProfile:
        """
        Analyze a stock's historical behavior and classify it.

        Parameters
        ----------
        symbol : stock code
        df : stock OHLCV data with computed indicators
        index_df : market index data (optional, for beta calculation)

        Returns
        -------
        StockProfile with style classification
        """
        name = df["name"].iloc[0] if "name" in df.columns else symbol
        close = df["close"].astype(float)

        # 1. Average daily volatility (ATR%)
        if "atr_pct" in df.columns:
            avg_vol = df["atr_pct"].dropna().mean()
        elif "atr" in df.columns:
            avg_vol = (df["atr"] / close).dropna().mean()
        else:
            avg_vol = df["close"].pct_change().std()

        # 2. Average turnover
        if "turnover" in df.columns:
            avg_turnover = df["turnover"].dropna().mean()
        else:
            avg_turnover = 0.05

        # 3. Trend efficiency (directional / total movement)
        ret_1d = close.pct_change().dropna()
        if len(ret_1d) > 60:
            # Calculate over rolling 60-day windows
            rolling_ret = close.pct_change(60).dropna()
            rolling_path = ret_1d.rolling(60).sum().dropna()
            # Align
            min_len = min(len(rolling_ret), len(rolling_path))
            if min_len > 0:
                trend_eff = (
                    abs(rolling_ret.iloc[-min_len:]).mean() /
                    (rolling_path.iloc[-min_len:].abs().mean() + 1e-9)
                )
                trend_eff = min(1.0, trend_eff)
            else:
                trend_eff = 0.3
        else:
            trend_eff = 0.3

        # 4. Beta vs market
        if index_df is not None:
            common = set(df["date"].dt.date) & set(index_df["date"].dt.date)
            sub_df = df[df["date"].dt.date.isin(common)]
            sub_idx = index_df[index_df["date"].dt.date.isin(common)]
            if len(sub_df) > 60:
                stock_ret = sub_df["close"].pct_change().dropna()
                idx_ret = sub_idx["close"].pct_change().dropna()
                min_len = min(len(stock_ret), len(idx_ret))
                if min_len > 0:
                    cov = np.cov(stock_ret.iloc[-min_len:],
                                 idx_ret.iloc[-min_len:])[0, 1]
                    var = np.var(idx_ret.iloc[-min_len:])
                    beta = cov / var if var > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0

        # 5. Market cap category
        mcap = df["close"].iloc[-1] * 1e8  # rough estimate
        if mcap > 2000e8:
            mcap_cat = "large"
        elif mcap > 200e8:
            mcap_cat = "mid"
        else:
            mcap_cat = "small"

        # 6. Sector
        sector = self.theme_analyzer.stock_sector.get(symbol, "unknown")

        # === Classification logic ===
        style = self._determine_style(avg_vol, avg_turnover, trend_eff,
                                       beta, mcap_cat, sector)

        return StockProfile(
            symbol=symbol, name=name,
            avg_volatility=avg_vol,
            avg_turnover=avg_turnover,
            trend_efficiency=trend_eff,
            beta_vs_market=beta,
            mcap_category=mcap_cat,
            sector=sector,
            style=style,
        )

    def _determine_style(self, vol, turnover, trend_eff, beta,
                          mcap_cat, sector) -> str:
        """Determine stock style: sector-based with characteristic overrides"""
        sector_style = SECTOR_STYLE_MAP.get(sector, "balanced")

        # Characteristic-based overrides (only when strongly indicated)
        # High volatility override
        if vol > 0.04:
            return "high_vol"

        # Small cap + high turnover → small_mom
        if mcap_cat == "small" and turnover > 0.06:
            return "small_mom"

        # Large cap + very low turnover → blue_chip
        if mcap_cat == "large" and turnover < 0.02:
            return "blue_chip"

        # Low trend efficiency + mid/large → cyclical
        if trend_eff < 0.12 and mcap_cat in ("large",):
            return "cyclical"

        # Otherwise trust sector-based classification
        if sector_style:
            return sector_style

        # Fallback based on characteristics
        if trend_eff > 0.25 and vol < 0.03:
            return "trend_growth"
        if mcap_cat == "small":
            return "small_mom"

        return "balanced"


class AdaptiveXuXiangStrategy(XuXiangStrategy):
    """
    Enhanced Xu Xiang strategy with:
      1. Auto-detection of stock type
      2. Per-type parameter selection
      3. Theme-aware entry scoring
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 stock_type: str = "auto",
                 theme_boost: bool = True):
        """
        Parameters
        ----------
        params : base parameters (overridden by type-specific defaults)
        stock_type : "auto" for auto-detect, or any valid type key
        theme_boost : if True, boost signals when theme is hot
        """
        self.requested_type = stock_type
        self.theme_boost = theme_boost
        self.detected_type = stock_type if stock_type != "auto" else "balanced"
        self.classifier = StockClassifier()
        self.theme_analyzer = SectorThemeAnalyzer()

        # Start with base params
        base_params = dict(params or {})

        # Override with type-specific defaults if known
        if stock_type != "auto" and stock_type in TYPE_PARAMS:
            type_defaults = TYPE_PARAMS[stock_type]
            # Type defaults take precedence, but explicit params override
            for k, v in type_defaults.items():
                if k not in base_params:
                    base_params[k] = v

        super().__init__(base_params)

    def auto_detect(self, df: pd.DataFrame, index_df: pd.DataFrame = None):
        """Auto-detect stock type and apply appropriate parameters"""
        if self.requested_type == "auto":
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "000000"
            profile = self.classifier.classify(symbol, df, index_df)
            self.detected_type = profile.style

            # Merge type-specific params
            type_params = TYPE_PARAMS.get(self.detected_type,
                                           TYPE_PARAMS["balanced"])
            for k, v in type_params.items():
                if k not in self.params:
                    self.params[k] = v

        return self.detected_type

    def generate_signals(self, df: pd.DataFrame,
                          index_df: pd.DataFrame = None) -> pd.Series:
        """
        Generate signals with auto-detection and theme awareness.

        If index_df is provided, theme scoring is enabled.
        """
        # Auto-detect if needed
        if self.requested_type == "auto":
            self.auto_detect(df, index_df)

        # Recompute config if type changed
        field_names = list(StrategyConfig.__dataclass_fields__.keys())
        cfg = {k: v for k, v in self.params.items() if k in field_names}
        self.config = StrategyConfig(**cfg)

        # Generate base signals
        signals = super().generate_signals(df)

        return signals

    def get_type_info(self) -> Dict:
        """Get current type and parameter info"""
        type_params = TYPE_PARAMS.get(self.detected_type,
                                       TYPE_PARAMS["balanced"])
        return {
            "detected_type": self.detected_type,
            "description": type_params.get("description", ""),
            "ma_short": self.config.ma_short,
            "ma_long": self.config.ma_long,
            "atr_stop_mult": self.config.atr_stop_mult,
            "max_holding_days": type_params.get("max_holding_days", 10),
            "position_pct": self.config.max_position_pct,
        }


def run_adaptive_backtest(symbol: str, name: str = "",
                           start: str = "2015-01-01",
                           end: str = "2025-12-31"):
    """Run adaptive strategy backtest for a single stock"""
    from .backtest_10y import TenYearBacktest

    bt = TenYearBacktest(initial_capital=1000000)
    strategy = AdaptiveXuXiangStrategy(stock_type="auto", theme_boost=True)

    rpt = bt.run(symbol, name, start, end, sp=strategy.params)
    bt.print_report(rpt)

    # Print type info
    type_info = strategy.get_type_info()
    print(f"\n  Stock Type: {type_info['detected_type']} - "
          f"{type_info['description']}")
    print(f"  MA: {type_info['ma_short']}/{type_info['ma_long']}  "
          f"ATR stop: {type_info['atr_stop_mult']}x  "
          f"Max hold: {type_info['max_holding_days']}d  "
          f"Position: {type_info['position_pct']:.0%}")

    return rpt


if __name__ == "__main__":
    # Test stock classification
    classifier = StockClassifier()
    for sym, name, style_expected in [
        ("000858", "Wuliangye", "trend_growth"),
        ("002036", "LianChuang", "small_mom"),
        ("600519", "Moutai", "blue_chip"),
        ("300750", "CATL", "high_vol"),
        ("600585", "ConchCement", "cyclical"),
    ]:
        profile = classifier.classify(sym, pd.DataFrame({
            "close": [100] * 100,
            "atr_pct": [0.02] * 100,
            "turnover": [0.05] * 100,
            "name": [name] * 100,
        }))
        print(f"  {name}: style={profile.style} (sector={profile.sector})")
