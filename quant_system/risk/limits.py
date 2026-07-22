"""
Exposure limits — enforce risk constraints.

Checks portfolio positions against predefined limits:
- Single stock concentration
- Sector concentration
- Leverage
- Liquidity (position size vs daily volume)
- Drawdown halt trigger

Two Sigma approach: "Limits are not suggestions. Violating a limit
is a process failure, not a trading decision."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


@dataclass
class LimitCheckResult:
    """Result of a limit check."""
    passed: bool
    violations: List[Dict]        # List of {limit, actual, threshold, severity}
    warnings: List[Dict]
    summary: str


class ExposureLimits:
    """Portfolio exposure limit enforcement."""

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()
        self.limits = self.config.risk_limits

    def check_all(
        self,
        weights: PortfolioWeights,
        sectors: Optional[Dict[str, str]] = None,
        market_caps: Optional[Dict[str, float]] = None,
        daily_volumes: Optional[Dict[str, float]] = None,
        portfolio_value: float = 1_000_000,
    ) -> LimitCheckResult:
        """Run all exposure limit checks.

        Returns:
            LimitCheckResult with pass/fail and details.
        """
        violations = []
        warnings = []

        # 1. Single stock concentration
        for code, w in weights.weights.items():
            if w > self.limits.max_single_position:
                violations.append({
                    "limit": "Single stock concentration",
                    "code": code,
                    "actual": w,
                    "threshold": self.limits.max_single_position,
                    "severity": "HIGH",
                    "message": f"{code}: {w:.1%} > {self.limits.max_single_position:.0%} limit",
                })

        # 2. Sector concentration
        if sectors:
            sector_weights: Dict[str, float] = {}
            for code, w in weights.weights.items():
                sec = sectors.get(code, "Unknown")
                sector_weights[sec] = sector_weights.get(sec, 0.0) + w

            for sec, w in sector_weights.items():
                if w > self.limits.max_sector_exposure:
                    violations.append({
                        "limit": "Sector concentration",
                        "sector": sec,
                        "actual": w,
                        "threshold": self.limits.max_sector_exposure,
                        "severity": "HIGH",
                        "message": f"{sec}: {w:.1%} > {self.limits.max_sector_exposure:.0%} limit",
                    })
                elif w > self.limits.max_sector_exposure * 0.8:
                    warnings.append({
                        "limit": "Sector concentration (warning)",
                        "sector": sec,
                        "actual": w,
                        "threshold": self.limits.max_sector_exposure * 0.8,
                        "message": f"{sec}: {w:.1%} approaching {self.limits.max_sector_exposure:.0%} limit",
                    })

        # 3. Cash allocation
        if weights.cash_pct < 0:
            violations.append({
                "limit": "Leverage",
                "actual": weights.cash_pct,
                "threshold": 0.0,
                "severity": "CRITICAL",
                "message": f"Negative cash ({weights.cash_pct:.1%}) = leverage detected!",
            })

        # 4. Number of positions
        n_positions = len([w for w in weights.weights.values() if w > 0.01])
        if n_positions < 3:
            warnings.append({
                "limit": "Diversification",
                "actual": n_positions,
                "threshold": 3,
                "message": f"Only {n_positions} positions — poor diversification",
            })
        if n_positions > 50:
            warnings.append({
                "limit": "Position bloat",
                "actual": n_positions,
                "threshold": 50,
                "message": f"{n_positions} positions — may be over-diversified",
            })

        # 5. Liquidity check (position size vs daily volume)
        if daily_volumes:
            for code, w in weights.weights.items():
                position_value = portfolio_value * w
                daily_vol = daily_volumes.get(code, 0)
                if daily_vol > 0:
                    vol_ratio = position_value / daily_vol
                    if vol_ratio > 0.10:  # Position > 10% of daily volume
                        warnings.append({
                            "limit": "Liquidity",
                            "code": code,
                            "actual": vol_ratio,
                            "threshold": 0.10,
                            "message": f"{code}: position is {vol_ratio:.0%} of daily volume",
                        })

        passed = len(violations) == 0

        # Summary
        high_severity = [v for v in violations if v["severity"] in ("CRITICAL", "HIGH")]
        summary = (
            f"Limits Check: {'PASS' if passed else 'FAIL'}\n"
            f"  Violations: {len(violations)} ({len(high_severity)} HIGH/CRITICAL)\n"
            f"  Warnings: {len(warnings)}"
        )

        return LimitCheckResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            summary=summary,
        )

    def drawdown_halt(
        self, current_drawdown: float
    ) -> Tuple[bool, str]:
        """Check if global drawdown limit has been breached.

        Returns:
            (should_halt, message)
        """
        if abs(current_drawdown) > self.limits.max_drawdown_limit:
            return True, (
                f"⚠️ DRAWDOWN HALT: {current_drawdown:.1%} exceeds "
                f"{self.limits.max_drawdown_limit:.0%} limit. "
                f"Reduce positions or stop trading."
            )
        elif abs(current_drawdown) > self.limits.max_drawdown_limit * 0.7:
            return False, (
                f"⚠️ WARNING: Drawdown {current_drawdown:.1%} approaching "
                f"{self.limits.max_drawdown_limit:.0%} limit."
            )
        return False, ""
