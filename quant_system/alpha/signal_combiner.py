"""
Signal combination methods.

Combines multiple alpha signals into a single composite signal
using various weighting schemes:
- Equal weight: Simple average
- IC-weighted: Weight by historical Information Coefficient
- Eigenvector: Use first principal component weights
- ML voting: Ensemble approach (future extension)

The art of signal combination is avoiding double-counting correlated
signals while preserving diversification benefits.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import FactorData, Signal
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SignalCombiner:
    """Combines multiple alpha signals into a composite signal.

    "Signals are like uncorrelated bets. Combining them increases
    your Information Ratio by sqrt(N) — but only if they're truly
    uncorrelated." — Citadel wisdom
    """

    def __init__(self):
        pass

    def combine(
        self,
        signals: Dict[str, pd.DataFrame],
        method: str = "ic_weighted",
        ic_history: Optional[Dict[str, float]] = None,
        recent_performance: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Combine multiple signals into one.

        Args:
            signals: Dict of factor_name -> DataFrame (dates x stocks).
            method: "equal" | "ic_weighted" | "eigenvector" | "performance".
            ic_history: Dict of factor_name -> mean IC (for IC-weighted).
            recent_performance: Dict of factor_name -> recent return series
                               (for performance-weighted).

        Returns:
            Combined signal DataFrame (dates x stocks).
        """
        if method == "equal":
            return self.equal_weight(signals)
        elif method == "ic_weighted":
            return self.ic_weighted(signals, ic_history or {})
        elif method == "eigenvector":
            return self.eigenvector(signals)
        elif method == "performance":
            return self.performance_weighted(signals, recent_performance or {})
        else:
            logger.warning(f"Unknown method '{method}', using equal weight")
            return self.equal_weight(signals)

    def equal_weight(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simple equal-weighted average."""
        if not signals:
            raise ValueError("No signals provided")

        # Align all signals to common index
        result = pd.DataFrame(0.0, index=self._common_index(signals),
                              columns=self._common_columns(signals))

        for name, df in signals.items():
            aligned = df.reindex(index=result.index, columns=result.columns)
            result += aligned.fillna(0)

        return result / len(signals)

    def ic_weighted(
        self,
        signals: Dict[str, pd.DataFrame],
        ic_history: Dict[str, float],
    ) -> pd.DataFrame:
        """Weight signals by their historical Information Coefficient.

        Uses absolute IC as weight (direction is preserved in the signal).
        """
        if not signals:
            raise ValueError("No signals provided")

        # Default: equal weight if no IC history
        if not ic_history:
            logger.warning("No IC history provided, falling back to equal weight")
            return self.equal_weight(signals)

        # Compute weights from IC (shrinkage toward equal weight)
        abs_ics = {name: abs(ic_history.get(name, 0.0)) for name in signals}
        total_ic = sum(abs_ics.values())

        if total_ic == 0:
            return self.equal_weight(signals)

        weights = {
            name: (ic / total_ic * 0.8 + 0.2 / len(signals))  # 20% shrinkage to equal
            for name, ic in abs_ics.items()
        }

        result = pd.DataFrame(0.0, index=self._common_index(signals),
                              columns=self._common_columns(signals))

        for name, df in signals.items():
            w = weights.get(name, 1.0 / len(signals))
            aligned = df.reindex(index=result.index, columns=result.columns)
            result += aligned.fillna(0) * w

        return result

    def eigenvector(
        self, signals: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Use first principal component as composite signal.

        This naturally down-weights signals that are highly correlated
        with others, avoiding double-counting. The first PC captures
        the maximum common variance across all signals.

        For each date, computes the first eigenvector of the
        cross-signal covariance matrix and uses it as weights.
        """
        if len(signals) < 2:
            return self.equal_weight(signals)

        idx = self._common_index(signals)
        cols = self._common_columns(signals)
        result = pd.DataFrame(0.0, index=idx, columns=cols)

        for date in idx:
            # Stack factor values for this date
            factor_vals = {}
            for name, df in signals.items():
                if date in df.index:
                    row = df.loc[date]
                    factor_vals[name] = row.values

            if len(factor_vals) < 2:
                continue

            # Build matrix: stocks x factors
            common_stocks = None
            for vals in factor_vals.values():
                finite_mask = np.isfinite(vals)
                if common_stocks is None:
                    common_stocks = finite_mask
                else:
                    common_stocks = common_stocks & finite_mask

            if common_stocks is None or common_stocks.sum() < 5:
                continue

            X = np.column_stack([
                vals[common_stocks] for vals in factor_vals.values()
            ])

            # Covariance matrix (factor x factor)
            X_centered = X - X.mean(axis=0)
            cov = X_centered.T @ X_centered / (X.shape[0] - 1)

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # First PC weights (eigenvector with largest eigenvalue)
                weights = eigenvectors[:, -1]
                weights = weights / np.sum(np.abs(weights))  # Normalize
            except np.linalg.LinAlgError:
                weights = np.ones(len(factor_vals)) / len(factor_vals)

            # Composite signal = weighted sum
            composite = np.zeros(common_stocks.sum())
            for i, (name, vals) in enumerate(factor_vals.items()):
                composite += weights[i] * vals[common_stocks]

            # Store
            stock_subset = cols[common_stocks] if hasattr(common_stocks, '__len__') else []
            if len(stock_subset) > 0:
                for j, stock in enumerate(stock_subset):
                    result.loc[date, stock] = composite[j]

        return result

    def performance_weighted(
        self,
        signals: Dict[str, pd.DataFrame],
        recent_performance: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Weight signals by recent performance (momentum of signals).

        Signals that have worked recently get higher weight.
        This is a regime-adaptive approach.
        """
        if not recent_performance:
            return self.equal_weight(signals)

        # Compute weights from recent Sharpe-like metric
        perf_scores = {}
        for name, rets in recent_performance.items():
            if len(rets) > 5:
                sharpe = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
                perf_scores[name] = max(sharpe, 0.0)  # Floor at zero

        total = sum(perf_scores.values())
        if total == 0:
            return self.equal_weight(signals)

        weights = {name: score / total for name, score in perf_scores.items()}

        result = pd.DataFrame(0.0, index=self._common_index(signals),
                              columns=self._common_columns(signals))

        for name, df in signals.items():
            w = weights.get(name, 0.0)
            if w > 0:
                aligned = df.reindex(index=result.index, columns=result.columns)
                result += aligned.fillna(0) * w

        return result

    def _common_index(self, signals: Dict[str, pd.DataFrame]) -> pd.Index:
        """Find common date index across all signal DataFrames."""
        indices = [df.index for df in signals.values() if not df.empty]
        if not indices:
            return pd.DatetimeIndex([])
        common = indices[0]
        for idx in indices[1:]:
            common = common.intersection(idx)
        return common

    def _common_columns(self, signals: Dict[str, pd.DataFrame]) -> pd.Index:
        """Find common stock columns across all signal DataFrames."""
        cols = [df.columns for df in signals.values() if not df.empty]
        if not cols:
            return pd.Index([])
        common = cols[0]
        for c in cols[1:]:
            common = common.intersection(c)
        return common
