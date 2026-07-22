"""
Signal decay analysis.

Measures how a signal's predictive power decays over increasing
forward horizons. This answers the critical question:
"How long is my edge valid for?"

A fast-decaying signal (IC drops to zero in 3 days) requires
high-frequency rebalancing. A slow-decaying signal (IC persists
for 20+ days) works well with weekly or monthly execution.

Citadel-style: "A signal without a measured half-life is a
signal you can't size properly."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..utils.types import FactorData
from .signal_tester import SignalTester, ICResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecayResult:
    """Signal decay analysis over multiple horizons."""
    factor_name: str
    horizons: List[int]
    ic_means: List[float]                       # Mean IC at each horizon
    ic_irs: List[float]                         # IC_IR at each horizon
    half_life: Optional[int]                    # Horizon where IC drops to 50% of peak
    zero_ic_horizon: Optional[int]              # Horizon where IC crosses zero
    optimal_horizon: int                        # Horizon with maximum IC_IR
    decay_rate: float                           # Average IC decay per day
    summary: str


class DecayAnalyzer:
    """Analyzes how signal predictive power decays over time.

    Computes IC at multiple forward horizons (1, 5, 10, 20, 40 days)
    and fits a decay curve to estimate the signal half-life.
    """

    def __init__(self):
        self.tester = SignalTester()

    def analyze(
        self,
        factor: FactorData,
        forward_returns: pd.DataFrame,
        horizons: Optional[List[int]] = None,
    ) -> DecayResult:
        """Analyze signal decay across multiple horizons.

        Args:
            factor: Factor to analyze.
            forward_returns: Forward return matrix.
            horizons: List of horizons to test. Default: [1, 3, 5, 10, 20].

        Returns:
            DecayResult with IC decay curve and half-life estimate.
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        ic_means = []
        ic_irs = []
        ic_results: List[ICResult] = []

        for h in horizons:
            ic = self.tester.information_coefficient(factor, forward_returns, horizon=h)
            ic_means.append(ic.pearson_ic_mean)
            ic_irs.append(ic.pearson_ic_ir)
            ic_results.append(ic)

        # Estimate half-life: horizon at which IC drops to 50% of peak
        peak_ic = max(ic_means) if ic_means else 0.0
        half_life = None
        if peak_ic > 0:
            half_ic = peak_ic / 2
            for h, ic_m in zip(horizons, ic_means):
                if ic_m <= half_ic:
                    half_life = h
                    break
            if half_life is None and ic_means[-1] > half_ic:
                half_life = horizons[-1]  # Slower decay than tested horizons

        # Zero-crossing horizon
        zero_horizon = None
        for h, ic_m in zip(horizons, ic_means):
            if ic_m <= 0:
                zero_horizon = h
                break

        # Optimal horizon (max IC_IR)
        if ic_irs:
            opt_idx = np.argmax(ic_irs)
            optimal_h = horizons[opt_idx]
        else:
            optimal_h = horizons[0] if horizons else 1

        # Decay rate
        if len(horizons) >= 2:
            decay_per_day = (ic_means[0] - ic_means[-1]) / (horizons[-1] - horizons[0])
        else:
            decay_per_day = 0.0

        # Summary
        lines = [f"Decay Analysis: {factor.name}"]
        lines.append(f"  {'Horizon':<10} {'IC Mean':<12} {'IC_IR':<10}")
        lines.append(f"  {'-'*8}   {'-'*10}   {'-'*8}")
        for h, ic_m, ic_ir in zip(horizons, ic_means, ic_irs):
            marker = " ← BEST" if h == optimal_h else ""
            lines.append(f"  {h}d        {ic_m:.4f}       {ic_ir:.3f}{marker}")

        lines.append(f"\n  Half-life: {half_life}d" if half_life else "\n  Half-life: >20d")
        lines.append(f"  Optimal rebalance: every {optimal_h} days")
        lines.append(f"  Decay rate: {decay_per_day:.5f} IC/day")

        rec = "DAILY rebalancing recommended" if optimal_h <= 3 else (
            "WEEKLY rebalancing recommended" if optimal_h <= 7 else
            "BIWEEKLY rebalancing recommended" if optimal_h <= 14 else
            "MONTHLY rebalancing recommended"
        )
        lines.append(f"  → {rec}")

        return DecayResult(
            factor_name=factor.name,
            horizons=horizons,
            ic_means=ic_means,
            ic_irs=ic_irs,
            half_life=half_life,
            zero_ic_horizon=zero_horizon,
            optimal_horizon=optimal_h,
            decay_rate=decay_per_day,
            summary="\n".join(lines),
        )

    def compare_factor_decay(
        self,
        factors: List[FactorData],
        forward_returns: pd.DataFrame,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Compare decay profiles across multiple factors.

        Returns:
            DataFrame with IC_IR per factor per horizon.
        """
        if horizons is None:
            horizons = [1, 5, 10, 20]

        results = {}
        for f in factors:
            decay = self.analyze(f, forward_returns, horizons)
            results[f.name] = dict(zip(decay.horizons, decay.ic_irs))

        df = pd.DataFrame(results, index=horizons)
        df.index.name = "horizon"
        return df
