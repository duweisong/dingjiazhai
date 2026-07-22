"""
Signal pipeline — composes factor exposures into tradeable signals.

Takes a StrategySpec with factor exposures and signal weights,
runs factors through the alpha research pipeline, and produces
per-stock trading signals.

Goldman Sachs approach: "The signal pipeline is the assembly line
of quantitative investing. Every station must be independently
verified before the output is trusted."
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import FactorData, Signal, SignalResult, MultiStockData
from ..utils.logger import get_logger
from .spec_parser import StrategySpec

logger = get_logger(__name__)


class SignalPipeline:
    """Converts strategy specs + factor data into trading signals."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec

    def generate(
        self,
        factors: List[FactorData],
        data: Optional[MultiStockData] = None,
        date: Optional[pd.Timestamp] = None,
        top_n: int = 20,
    ) -> SignalResult:
        """Generate trading signals from factors.

        Args:
            factors: List of FactorData objects.
            data: Multi-stock data (optional, for additional filters).
            date: Generate signals for this date (default: last date).
            top_n: Maximum number of BUY signals to return.

        Returns:
            SignalResult with ranked signals.
        """
        # Filter factors to those in the spec
        relevant_factors = [
            f for f in factors
            if f.name in self.spec.factor_exposures
        ]

        if not relevant_factors:
            logger.warning("No matching factors found for this strategy spec")
            return SignalResult(date=date or pd.Timestamp.now(), signals=[])

        # Determine date
        if date is None:
            common_dates = relevant_factors[0].values.index
            for f in relevant_factors[1:]:
                common_dates = common_dates.intersection(f.values.index)
            if len(common_dates) == 0:
                return SignalResult(date=pd.Timestamp.now(), signals=[])
            date = common_dates[-1]

        # Build composite score per stock
        composite = self._compose_factors(relevant_factors, date)

        if composite is None or len(composite) == 0:
            return SignalResult(date=date, signals=[])

        # Convert composite scores to signals
        signals = self._scores_to_signals(composite, date, top_n)

        return SignalResult(date=date, signals=signals)

    def _compose_factors(
        self, factors: List[FactorData], date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Combine factor values into a composite score per stock."""
        scores = None

        for f in factors:
            if date not in f.values.index:
                continue

            row = f.values.loc[date].dropna()
            if len(row) == 0:
                continue

            # Normalize to z-score cross-sectionally
            z = (row - row.mean()) / (row.std() + 1e-10)
            # Apply direction
            z = z * f.direction

            # Weight by spec exposure
            weight = self.spec.factor_exposures.get(f.name, 0.0)
            if weight == 0:
                continue

            weighted = z * weight

            if scores is None:
                scores = weighted
            else:
                scores = scores.add(weighted, fill_value=0)

        return scores

    def _scores_to_signals(
        self,
        scores: pd.Series,
        date: pd.Timestamp,
        top_n: int,
    ) -> List[Signal]:
        """Convert composite scores to BUY/SELL/HOLD signals."""
        signals = []
        min_strength = self.spec.min_signal_strength

        # Rank scores
        ranked = scores.sort_values(ascending=False)
        n = len(ranked)

        for i, (code, score) in enumerate(ranked.items()):
            # Percentile rank as confidence
            pct_rank = 1.0 - i / n

            # Normalize score to [-1, 1] range
            if n > 1:
                norm_score = score / (abs(ranked).max() + 1e-10)
            else:
                norm_score = 0.0

            if norm_score > min_strength and i < top_n:
                signals.append(Signal(
                    date=date,
                    code=code,
                    signal_type="BUY",
                    strength=float(np.clip(norm_score, -1, 1)),
                    confidence=float(pct_rank),
                    reason=f"Composite score: {score:.3f}, rank: {i+1}/{n}",
                ))
            elif norm_score < -min_strength:
                signals.append(Signal(
                    date=date,
                    code=code,
                    signal_type="SELL",
                    strength=float(np.clip(norm_score, -1, 1)),
                    confidence=float(1 - pct_rank),
                    reason=f"Composite score: {score:.3f}, rank: {i+1}/{n}",
                ))

        return signals
