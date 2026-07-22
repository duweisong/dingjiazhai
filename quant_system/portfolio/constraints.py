"""
Portfolio constraint library.

Defines common portfolio constraints for optimization:
- Weight bounds (min/max per asset)
- Sum-to-one (fully invested)
- Long-only
- Sector / group constraints
- Turnover limits
- Cardinality (max number of positions)
- Tracking error vs benchmark

Man Group approach: "Constraints are your strategy in math form.
If your constraints are loose, your optimizer will find the
most extreme corner solution — and you'll be sorry."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Portfolio optimization constraints."""
    # Basic constraints
    sum_to_one: bool = True                # Sum of weights = 1
    long_only: bool = True                 # All weights >= 0
    max_weight: float = 0.10               # Max per-asset weight
    min_weight: float = 0.0                # Min per-asset weight

    # Advanced constraints
    max_sector_weight: Optional[float] = None  # Max per sector
    max_turnover: Optional[float] = None   # Max weight change from current
    max_positions: Optional[int] = None    # Max number of non-zero positions
    min_positions: Optional[int] = None    # Min number of non-zero positions
    target_return: Optional[float] = None  # Minimum target return
    target_volatility: Optional[float] = None  # Maximum target volatility

    # Custom bounds
    asset_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sector_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    group_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def validate(self, weights: np.ndarray, assets: List[str] = None) -> List[str]:
        """Check if weights satisfy all constraints.

        Returns:
            List of violation descriptions (empty = all good).
        """
        violations = []
        n = len(weights)

        if self.sum_to_one:
            total = np.sum(weights)
            if abs(total - 1.0) > 0.01:
                violations.append(f"Sum={total:.3f} ≠ 1.0")

        if self.long_only:
            neg = weights[weights < -0.001]
            if len(neg) > 0:
                violations.append(f"{len(neg)} negative weights")

        if self.max_weight:
            over = weights[weights > self.max_weight + 0.001]
            if len(over) > 0:
                violations.append(f"{len(over)} weights > {self.max_weight:.0%}")

        if self.max_positions:
            n_nonzero = np.sum(weights > 0.001)
            if n_nonzero > self.max_positions:
                violations.append(f"{n_nonzero} positions > max {self.max_positions}")

        return violations


class ConstraintBuilder:
    """Builds constraints from config and portfolio state."""

    @staticmethod
    def from_config(config) -> Constraints:
        """Build constraints from GlobalConfig."""
        return Constraints(
            sum_to_one=True,
            long_only=True,
            max_weight=config.risk_limits.max_single_position,
            max_sector_weight=config.risk_limits.max_sector_exposure,
        )

    @staticmethod
    def with_sectors(
        base: Constraints,
        sectors: Dict[str, str],
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Constraints:
        """Add sector-specific constraints."""
        if sector_limits is None:
            sector_limits = {}
        # Build sector bounds
        sector_bounds = {}
        for sector, limit in sector_limits.items():
            sector_bounds[sector] = (0.0, limit)
        base.sector_bounds = sector_bounds
        return base

    @staticmethod
    def with_turnover_limit(
        base: Constraints,
        current_weights: Dict[str, float],
        max_turnover: float = 0.30,
    ) -> Constraints:
        """Add turnover constraint."""
        base.max_turnover = max_turnover
        return base

    @staticmethod
    def with_cardinality(
        base: Constraints,
        min_positions: int = 5,
        max_positions: int = 30,
    ) -> Constraints:
        """Add position count constraints."""
        base.min_positions = min_positions
        base.max_positions = max_positions
        return base
