"""
Performance attribution — factor-based return decomposition.

Decomposes portfolio returns into:
- Factor contributions (how much each factor explained)
- Specific return / alpha (what's left after accounting for factors)
- Interaction effects

AQR approach: "If you can't attribute your returns, you don't know
whether you're being paid for factor exposure or for genuine skill."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AttributionReport:
    """Performance attribution report."""
    total_return: float
    factor_contributions: Dict[str, float]  # factor_name -> return contribution
    specific_return: float                   # Residual / alpha
    r_squared: float                         # Fraction explained by factors
    tracking_error: float                    # Volatility of specific return
    information_ratio: float                 # Specific return / tracking error
    summary: str


class PerformanceAttribution:
    """Factor-based performance attribution (Brinson-style)."""

    def attribute(
        self,
        portfolio_returns: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> AttributionReport:
        """Attribute portfolio returns to factor exposures.

        Args:
            portfolio_returns: Daily portfolio return series.
            factor_exposures: DataFrame (dates x factors) of portfolio
                             exposures to each factor.
            factor_returns: DataFrame (dates x factors) of factor returns.

        Returns:
            AttributionReport with factor contributions and specific return.
        """
        # Align all data
        common_dates = portfolio_returns.index.intersection(
            factor_exposures.index
        ).intersection(factor_returns.index)

        if len(common_dates) < 10:
            return AttributionReport(
                total_return=0.0,
                factor_contributions={},
                specific_return=0.0,
                r_squared=0.0,
                tracking_error=0.0,
                information_ratio=0.0,
                summary="Insufficient data for attribution.",
            )

        port_rets = portfolio_returns[common_dates]
        exposures = factor_exposures.loc[common_dates]
        factor_rets = factor_returns.loc[common_dates]

        # Factor contributions: exposure * factor_return per period
        contributions = {}
        for factor in factor_rets.columns:
            if factor in exposures.columns:
                contrib_series = exposures[factor] * factor_rets[factor]
                contributions[factor] = float(contrib_series.sum())

        total_factor_return = sum(contributions.values())
        total_portfolio_return = float(port_rets.sum())

        # Specific return = actual - factor-explained
        specific_return = total_portfolio_return - total_factor_return

        # R-squared: fraction of variance explained by factors
        predicted = pd.Series(0.0, index=common_dates)
        for factor in factor_rets.columns:
            if factor in exposures.columns:
                predicted += exposures[factor] * factor_rets[factor]

        ss_res = float(((port_rets - predicted) ** 2).sum())
        ss_tot = float(((port_rets - port_rets.mean()) ** 2).sum())
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Tracking error and IR
        specific_series = port_rets - predicted
        tracking_error = float(specific_series.std() * np.sqrt(244))
        ann_specific = float(specific_series.mean() * 244)
        information_ratio = ann_specific / tracking_error if tracking_error > 0 else 0.0

        # Summary
        lines = ["Performance Attribution Report", "=" * 40]
        lines.append(f"  Total Return: {total_portfolio_return:.2%}")
        lines.append(f"  Factor-Explained: {total_factor_return:.2%}")
        lines.append(f"  Specific (Alpha): {specific_return:.2%}")
        lines.append(f"  R²: {r_squared:.3f}")
        lines.append(f"  Tracking Error: {tracking_error:.2%}")
        lines.append(f"  Information Ratio: {information_ratio:.2f}")
        lines.append(f"\n  Factor Contributions:")
        for name, contrib in sorted(contributions.items(), key=lambda x: -abs(x[1])):
            lines.append(f"    {name}: {contrib:.2%}")
        if specific_return != 0:
            lines.append(f"    [Alpha/Residual]: {specific_return:.2%}")

        return AttributionReport(
            total_return=total_portfolio_return,
            factor_contributions=contributions,
            specific_return=specific_return,
            r_squared=r_squared,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            summary="\n".join(lines),
        )

    def rolling_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Rolling attribution over time.

        Shows how factor contributions and alpha evolve.

        Returns:
            DataFrame with rolling factor contributions and specific return.
        """
        results = []

        for end in range(window, len(portfolio_returns)):
            start = end - window
            window_returns = portfolio_returns.iloc[start:end]
            window_exposures = factor_exposures.iloc[start:end]
            window_factor_rets = factor_returns.iloc[start:end]

            report = self.attribute(window_returns, window_exposures, window_factor_rets)
            row = {"date": portfolio_returns.index[end - 1]}
            row.update(report.factor_contributions)
            row["specific_return"] = report.specific_return
            row["r_squared"] = report.r_squared
            results.append(row)

        return pd.DataFrame(results).set_index("date")
