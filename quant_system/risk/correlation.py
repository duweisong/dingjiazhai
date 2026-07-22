"""
Correlation monitoring — detect when diversification breaks down.

Monitors cross-stock and cross-factor correlations. When correlations
spike (crisis mode), diversification benefits disappear and risk
concentrates — this is when portfolios blow up.

Two Sigma approach: "The worst time to discover your positions
are all correlated is during a crisis. Monitor correlations daily."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationReport:
    """Correlation monitoring report."""
    date: pd.Timestamp
    avg_correlation: float               # Average pairwise correlation
    max_correlation: float               # Highest pairwise correlation
    correlation_alarm: bool              # True if avg correlation > threshold
    top_correlated: List[Tuple[str, str, float]]  # Top 3 most correlated pairs
    correlation_regime: str              # "normal" | "elevated" | "crisis"


class CorrelationMonitor:
    """Monitors correlation structure of the portfolio."""

    def __init__(
        self,
        window: int = 60,
        alarm_threshold: float = 0.6,
        crisis_threshold: float = 0.8,
    ):
        self.window = window
        self.alarm_threshold = alarm_threshold
        self.crisis_threshold = crisis_threshold

    def compute_correlation_matrix(
        self, returns: pd.DataFrame, date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Compute correlation matrix from returns.

        Args:
            returns: DataFrame (dates x stocks).
            date: If provided, use rolling window ending at date.

        Returns:
            Stock correlation matrix.
        """
        if date is not None:
            end_idx = returns.index.searchsorted(date, side="right")
            start_idx = max(0, end_idx - self.window)
            returns = returns.iloc[start_idx:end_idx]

        return returns.corr()

    def check(
        self, returns: pd.DataFrame, date: Optional[pd.Timestamp] = None
    ) -> CorrelationReport:
        """Check current correlation state.

        Args:
            returns: Stock return matrix (dates x stocks).
            date: Check date (default: last date).

        Returns:
            CorrelationReport.
        """
        if date is None:
            date = returns.index[-1] if len(returns) > 0 else pd.Timestamp.now()

        corr_matrix = self.compute_correlation_matrix(returns, date)

        if corr_matrix.empty or corr_matrix.shape[0] < 2:
            return CorrelationReport(
                date=date,
                avg_correlation=0.0,
                max_correlation=0.0,
                correlation_alarm=False,
                top_correlated=[],
                correlation_regime="unknown",
            )

        # Average pairwise correlation (upper triangle)
        n = corr_matrix.shape[0]
        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]

        avg_corr = float(np.mean(upper_tri))
        max_corr = float(np.max(upper_tri))

        # Find top 3 most correlated pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((
                    corr_matrix.index[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j],
                ))
        top_pairs = sorted(pairs, key=lambda x: -abs(x[2]))[:3]

        # Regime classification
        if avg_corr > self.crisis_threshold:
            regime = "crisis"
        elif avg_corr > self.alarm_threshold:
            regime = "elevated"
        else:
            regime = "normal"

        alarm = regime != "normal"

        return CorrelationReport(
            date=date,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            correlation_alarm=alarm,
            top_correlated=top_pairs,
            correlation_regime=regime,
        )

    def rolling_avg_correlation(
        self, returns: pd.DataFrame
    ) -> pd.Series:
        """Compute rolling average pairwise correlation over time.

        This is a powerful early warning indicator: rising average
        correlation precedes most major drawdowns.

        Returns:
            Series of average pairwise correlation at each date.
        """
        results = []
        dates = returns.index[self.window:]

        for date in dates:
            report = self.check(returns, date)
            results.append({
                "date": date,
                "avg_correlation": report.avg_correlation,
                "regime": report.correlation_regime,
            })

        df = pd.DataFrame(results).set_index("date")
        return df["avg_correlation"]

    def correlation_breakdown(
        self, returns: pd.DataFrame, sectors: Optional[Dict[str, str]] = None
    ) -> Dict:
        """Breakdown correlations by sector (intra vs inter).

        During normal markets, intra-sector correlations should be
        higher than inter-sector. When inter-sector correlations
        rise, systemic risk is building.

        Returns:
            Dict with intra_sector and inter_sector avg correlations.
        """
        if sectors is None:
            return {"intra_sector": 0.0, "inter_sector": 0.0}

        corr_matrix = self.compute_correlation_matrix(returns)
        stocks = corr_matrix.index.tolist()

        intra_corrs = []
        inter_corrs = []

        for i, s1 in enumerate(stocks):
            for j, s2 in enumerate(stocks):
                if i >= j:
                    continue
                sec1 = sectors.get(s1, "Unknown")
                sec2 = sectors.get(s2, "Unknown")
                corr = corr_matrix.loc[s1, s2]

                if sec1 == sec2:
                    intra_corrs.append(corr)
                else:
                    inter_corrs.append(corr)

        return {
            "intra_sector": float(np.mean(intra_corrs)) if intra_corrs else 0.0,
            "inter_sector": float(np.mean(inter_corrs)) if inter_corrs else 0.0,
            "ratio": (
                float(np.mean(intra_corrs) / np.mean(inter_corrs))
                if intra_corrs and inter_corrs and np.mean(inter_corrs) != 0
                else 0.0
            ),
        }
