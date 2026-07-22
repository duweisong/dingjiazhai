"""
Factor library — computes standard risk premia factors for A-shares.

Five factor categories (Fama-French inspired + A-share specific):
1. Momentum — Cross-sectional momentum (past returns)
2. Value — Cheap vs expensive (PE, PB, PS, dividend yield)
3. Quality — Profitable vs unprofitable (ROE, gross margin, debt ratio)
4. Size — Large cap vs small cap (market capitalization)
5. Volatility — Low vol vs high vol (realized volatility)

Each factor is computed cross-sectionally (per date, across stocks) and
returns a FactorData object compatible with the alpha research pipeline.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import FactorData, MultiStockData
from ..utils.logger import get_logger
from ..config import get_config

logger = get_logger(__name__)


class FactorBuilder:
    """Builds standard factor time series from price and fundamental data.

    AQR approach: "Factors are not just academic abstractions.
    Each factor must be investable, liquid, and robust to implementation."
    """

    def __init__(self):
        self.config = get_config()

    def build_all(
        self, data: MultiStockData, include_fundamentals: bool = False
    ) -> List[FactorData]:
        """Build all standard factors.

        Args:
            data: Multi-stock price + indicator data.
            include_fundamentals: If True, include value/quality factors
                                 that require fundamental data.

        Returns:
            List of FactorData objects.
        """
        factors: List[FactorData] = []

        # Momentum factors
        factors.extend(self.build_momentum_factors(data))

        # Volatility factors
        factors.extend(self.build_volatility_factors(data))

        # Size factor
        factors.append(self.build_size_factor(data))

        # Value and Quality (require fundamentals)
        if include_fundamentals:
            factors.extend(self.build_value_factors(data))
            factors.extend(self.build_quality_factors(data))

        logger.info(f"Built {len(factors)} factors: {[f.name for f in factors]}")
        return factors

    def build_momentum_factors(self, data: MultiStockData) -> List[FactorData]:
        """Build momentum factors at multiple horizons.

        Computation: cumulative return over lookback period, skipping
        the most recent month (to avoid short-term reversal).

        Returns:
            List of momentum FactorData (1M, 3M, 6M, 12M).
        """
        factors = []
        periods = self.config.factor.momentum_periods  # [20, 60, 120]

        closes = {}
        for code, df in data.prices.items():
            if "close" in df.columns:
                closes[code] = df["close"]
        if not closes:
            return factors

        close_df = pd.DataFrame(closes)

        for period in periods:
            # Momentum = return over past `period` days, skip last 5 days
            momentum = close_df.pct_change(5).shift(5).rolling(period - 5).apply(
                lambda x: (1 + x).prod() - 1
            )

            factors.append(FactorData(
                name=f"momentum_{period}d",
                display_name=f"{period//21}M Momentum" if period >= 21 else f"{period}D Momentum",
                direction=1,  # Long high-momentum stocks
                values=momentum,
                category="momentum",
            ))

        return factors

    def build_volatility_factors(self, data: MultiStockData) -> List[FactorData]:
        """Build volatility factors.

        Low volatility anomaly: low-vol stocks tend to outperform
        high-vol stocks on a risk-adjusted basis.

        Returns:
            List of volatility FactorData (1M, 3M realized vol).
        """
        factors = []
        periods = self.config.factor.volatility_periods  # [20, 60]

        closes = {}
        for code, df in data.prices.items():
            if "close" in df.columns:
                closes[code] = df["close"]
        if not closes:
            return factors

        close_df = pd.DataFrame(closes)
        returns = close_df.pct_change()

        for period in periods:
            # Realized volatility (annualized)
            realized_vol = returns.rolling(period).std() * np.sqrt(244)

            # Negate so that low vol = high factor value
            neg_vol = -realized_vol

            factors.append(FactorData(
                name=f"volatility_{period}d",
                display_name=f"{period//21}M Realized Volatility",
                direction=-1,  # Long low-vol (negated: higher factor = lower vol)
                values=neg_vol,
                category="volatility",
            ))

        return factors

    def build_size_factor(self, data: MultiStockData) -> List[FactorData]:
        """Build size (market cap) factor.

        Small-cap premium: smaller stocks tend to outperform larger
        stocks over the long term (though this has weakened recently).

        Returns:
            Size FactorData — negated log market cap (higher = smaller).
        """
        # Size proxy: use average daily volume × price as liquidity proxy
        # and negate log of it (higher factor value = smaller stock)
        volumes = {}
        for code, df in data.prices.items():
            if "close" in df.columns and "volume" in df.columns:
                # Approximate market cap proxy (volume × close)
                volumes[code] = np.log(df["volume"] * df["close"] + 1)

        if not volumes:
            return []

        size_df = pd.DataFrame(volumes)
        # Negate: higher factor = smaller stock
        neg_size = -size_df

        return FactorData(
            name="size",
            display_name="Size (Small-Cap)",
            direction=1,  # Long small-caps (higher factor = smaller)
            values=neg_size,
            category="size",
        )

    def build_value_factors(self, data: MultiStockData) -> List[FactorData]:
        """Build value factors using fundamental data.

        Note: Requires fundamental data (PE, PB ratios) which
        must be fetched from akshare separately.

        Returns:
            List of value FactorData (placeholder — needs fundamentals).
        """
        logger.info("Value factors require fundamental data (PE, PB, etc.)")
        logger.info("Use akshare to fetch stock financial data first.")
        return []  # Placeholder — needs actual fundamental data pipeline

    def build_quality_factors(self, data: MultiStockData) -> List[FactorData]:
        """Build quality factors using fundamental data.

        Note: Requires fundamental data (ROE, margins, debt ratios)
        which must be fetched from akshare separately.

        Returns:
            List of quality FactorData (placeholder — needs fundamentals).
        """
        logger.info("Quality factors require fundamental data (ROE, margins, etc.)")
        logger.info("Use akshare to fetch stock financial data first.")
        return []  # Placeholder — needs actual fundamental data pipeline

    def build_custom_factor(
        self,
        name: str,
        values: pd.DataFrame,
        category: str = "custom",
        direction: int = 1,
        display_name: str = "",
    ) -> FactorData:
        """Build a custom factor from a pre-computed DataFrame.

        Args:
            name: Factor identifier.
            values: DataFrame (dates x stocks) of factor values.
            category: Factor category label.
            direction: +1 = long high, -1 = long low.
            display_name: Human-readable name.

        Returns:
            FactorData object.
        """
        return FactorData(
            name=name,
            display_name=display_name or name,
            direction=direction,
            values=values,
            category=category,
        )

    def to_signal_matrix(
        self, factors: List[FactorData], method: str = "rank"
    ) -> pd.DataFrame:
        """Convert multiple factors to a combined signal matrix.

        Args:
            factors: List of factors.
            method: "rank" (cross-sectional percentile) or "zscore".

        Returns:
            Combined signal DataFrame (dates x stocks).
        """
        if not factors:
            return pd.DataFrame()

        combined = None
        for f in factors:
            vals = f.values.copy()
            if method == "rank":
                vals = vals.rank(axis=1, pct=True)
            else:
                vals = vals.sub(vals.mean(axis=1), axis=0).div(vals.std(axis=1), axis=0)

            # Apply direction
            vals = vals * f.direction

            if combined is None:
                combined = vals
            else:
                combined = combined.add(vals, fill_value=0)

        return combined.div(len(factors)) if combined is not None else pd.DataFrame()
