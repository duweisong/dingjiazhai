"""
Factor correlation analysis.

Analyzes correlations between factors to:
- Identify redundant factors (highly correlated = same bet)
- Ensure factor diversification
- Detect factor crowding (correlations rising over time)

AQR approach: "If two factors have correlation > 0.7, you're
paying twice for the same bet."
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import FactorData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FactorCorrelationAnalyzer:
    """Analyzes correlation structure across factors."""

    def __init__(self, rolling_window: int = 60):
        self.rolling_window = rolling_window

    def factor_correlation_matrix(
        self, factors: List[FactorData]
    ) -> pd.DataFrame:
        """Compute full-sample factor correlation matrix.

        For each pair of factors, computes the cross-sectional
        average correlation of their values over time.

        Args:
            factors: List of FactorData objects.

        Returns:
            Correlation matrix (factor x factor).
        """
        names = [f.name for f in factors]
        corr_matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors):
                if i >= j:
                    continue

                # Align factors
                common_dates = f1.values.index.intersection(f2.values.index)
                correlations = []

                for date in common_dates:
                    v1 = f1.values.loc[date].dropna()
                    v2 = f2.values.loc[date].dropna()
                    common = v1.index.intersection(v2.index)

                    if len(common) < 10:
                        continue

                    corr = v1[common].corr(v2[common])
                    if not np.isnan(corr):
                        correlations.append(corr)

                avg_corr = np.mean(correlations) if correlations else 0.0
                corr_matrix.loc[names[i], names[j]] = avg_corr
                corr_matrix.loc[names[j], names[i]] = avg_corr

        return corr_matrix

    def rolling_correlation(
        self, f1: FactorData, f2: FactorData
    ) -> pd.Series:
        """Compute rolling cross-sectional correlation between two factors.

        Args:
            f1, f2: Two factors to compare.

        Returns:
            Series of rolling correlations over time.
        """
        common_dates = f1.values.index.intersection(f2.values.index)
        rolling_corrs = []

        for date in common_dates:
            v1 = f1.values.loc[date].dropna()
            v2 = f2.values.loc[date].dropna()
            common = v1.index.intersection(v2.index)

            if len(common) < 10:
                rolling_corrs.append(np.nan)
                continue

            corr = v1[common].corr(v2[common])
            rolling_corrs.append(corr)

        return pd.Series(rolling_corrs, index=common_dates)

    def detect_redundant_factors(
        self, factors: List[FactorData], threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """Identify pairs of factors with correlation above threshold.

        Returns:
            List of (factor1, factor2, correlation) tuples for redundant pairs.
        """
        corr_matrix = self.factor_correlation_matrix(factors)
        redundant = []

        names = corr_matrix.index.tolist()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > threshold:
                    redundant.append((names[i], names[j], corr))

        return sorted(redundant, key=lambda x: -x[2])

    def factor_pca(self, factors: List[FactorData]) -> Dict:
        """PCA decomposition of factor structure.

        Identifies how many independent "bets" the factor set represents.

        Returns:
            Dict with eigenvalues, explained variance, and effective N.
        """
        corr_matrix = self.factor_correlation_matrix(factors)

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
        eigenvalues = eigenvalues[::-1]  # Descending
        explained_var = eigenvalues / np.sum(eigenvalues)
        cumulative_var = np.cumsum(explained_var)

        # Effective number of independent factors (1 / sum(w^2))
        weights = np.ones(len(factors)) / len(factors)
        effective_n = 1.0 / np.sum(weights @ corr_matrix.values * weights)

        return {
            "n_factors": len(factors),
            "effective_n": float(effective_n),
            "eigenvalues": eigenvalues.tolist(),
            "explained_variance": explained_var.tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "top3_explain_pct": float(cumulative_var[2]) if len(cumulative_var) >= 3 else 1.0,
            "summary": (
                f"Factor PCA: {len(factors)} factors → {effective_n:.1f} effective. "
                f"Top 3 PCs explain {cumulative_var[2]*100:.0f}% of variance "
                f"(vs {3/len(factors)*100:.0f}% expected under independence)."
            ),
        }
