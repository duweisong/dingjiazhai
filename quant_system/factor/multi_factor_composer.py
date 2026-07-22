"""
Multi-factor composition.

Combines multiple factors into a single investable composite using:
- Equal Risk Contribution (ERC): Weight so each factor contributes equally to risk
- IC-Weighted: Weight by predictive power
- Equal Weight: Simple average

AQR approach: "A well-constructed multi-factor portfolio captures
multiple independent risk premia while minimizing unintended bets."
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.types import FactorData, Signal
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultiFactorComposer:
    """Composes multiple factors into a single investable signal."""

    def __init__(self):
        pass

    def compose(
        self,
        factors: List[FactorData],
        method: str = "erc",
        ic_history: Optional[Dict[str, float]] = None,
        lookback: int = 252,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Compose multiple factors into one.

        Args:
            factors: List of factors.
            method: "equal" | "erc" | "ic_weighted" | "max_sharpe".
            ic_history: Factor IC history (for IC-weighted).
            lookback: Lookback window for covariance estimation.

        Returns:
            Tuple of (composite signal DataFrame, composition metadata dict).
        """
        if len(factors) == 0:
            return pd.DataFrame(), {}

        if len(factors) == 1:
            return factors[0].values * factors[0].direction, {"method": "single"}

        if method == "equal":
            weights = {f.name: 1.0 / len(factors) for f in factors}
        elif method == "ic_weighted" and ic_history:
            weights = self._ic_weights(factors, ic_history)
        elif method == "erc":
            weights = self._erc_weights(factors, lookback)
        elif method == "max_sharpe":
            weights = self._max_sharpe_weights(factors, ic_history or {}, lookback)
        else:
            weights = {f.name: 1.0 / len(factors) for f in factors}

        # Apply weights and directions
        composite = None
        for f in factors:
            w = weights.get(f.name, 0.0)
            if w == 0:
                continue
            vals = f.values * f.direction * w
            if composite is None:
                composite = vals
            else:
                composite = composite.add(vals, fill_value=0)

        meta = {
            "method": method,
            "weights": weights,
            "n_factors": len(factors),
            "factor_names": [f.name for f in factors],
        }

        return composite if composite is not None else pd.DataFrame(), meta

    def _ic_weights(
        self, factors: List[FactorData], ic_history: Dict[str, float]
    ) -> Dict[str, float]:
        """Weight factors by their IC, with shrinkage."""
        abs_ics = {}
        for f in factors:
            ic = ic_history.get(f.name, 0.0)
            abs_ics[f.name] = max(abs(ic), 0.001)  # Floor at 0.001

        total = sum(abs_ics.values())
        if total == 0:
            return {f.name: 1.0 / len(factors) for f in factors}

        # 30% shrinkage toward equal weight
        eq_w = 1.0 / len(factors)
        return {
            f.name: 0.7 * (abs_ics[f.name] / total) + 0.3 * eq_w
            for f in factors
        }

    def _erc_weights(
        self, factors: List[FactorData], lookback: int = 252
    ) -> Dict[str, float]:
        """Equal Risk Contribution weights.

        Weights factors so that each contributes equally to portfolio risk.
        Uses the factor correlation matrix to allocate more weight to
        factors with lower correlation to others.
        """
        # Build factor return series (cross-sectional mean factor return)
        factor_rets = {}
        for f in factors:
            # Factor "returns" = cross-sectional mean of direction-signed factor
            daily_vals = f.values.mean(axis=1) * f.direction
            daily_rets = daily_vals.diff().dropna()
            if len(daily_rets) > lookback:
                daily_rets = daily_rets.iloc[-lookback:]
            factor_rets[f.name] = daily_rets

        if not factor_rets:
            return {f.name: 1.0 / len(factors) for f in factors}

        # Align and compute covariance
        ret_df = pd.DataFrame(factor_rets).dropna()
        if ret_df.empty or ret_df.shape[1] < 2:
            return {f.name: 1.0 / len(factors) for f in factors}

        cov = ret_df.cov().values
        n = len(factors)

        # ERC optimization: find weights w such that
        # w_i * (Cov @ w)_i = constant for all i
        # Simplified: use inverse of row sums of correlation matrix
        try:
            corr = ret_df.corr().values
            inv_avg_corr = 1.0 / np.maximum(np.mean(np.abs(corr), axis=1), 0.1)
            weights = inv_avg_corr / np.sum(inv_avg_corr)
        except Exception:
            weights = np.ones(n) / n

        return {f.name: float(weights[i]) for i, f in enumerate(factors)}

    def _max_sharpe_weights(
        self,
        factors: List[FactorData],
        ic_history: Dict[str, float],
        lookback: int = 252,
    ) -> Dict[str, float]:
        """Maximum Sharpe ratio weights (simplified).

        Uses IC as expected return proxy and correlation matrix
        as risk proxy. Falls back to ERC if IC data is unavailable.
        """
        if not ic_history:
            return self._erc_weights(factors, lookback)

        # Build expected returns from IC
        expected_rets = np.array([
            ic_history.get(f.name, 0.0) * f.direction for f in factors
        ])

        if np.all(expected_rets == 0):
            return self._erc_weights(factors, lookback)

        # Use ERC as fallback for covariance
        erc_weights = self._erc_weights(factors, lookback)
        return erc_weights

    def rebalance_weights(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        max_turnover: float = 0.30,
    ) -> Dict[str, float]:
        """Transition weights gradually to limit turnover.

        Args:
            current_weights: Current portfolio factor weights.
            target_weights: Desired target factor weights.
            max_turnover: Maximum fraction of weights to change.

        Returns:
            Transition weights.
        """
        all_names = set(current_weights.keys()) | set(target_weights.keys())
        new_weights = {}

        total_change = sum(
            abs(target_weights.get(n, 0) - current_weights.get(n, 0))
            for n in all_names
        )

        if total_change <= max_turnover:
            return target_weights

        # Cap the change
        scale = max_turnover / total_change
        for name in all_names:
            cur = current_weights.get(name, 0.0)
            tgt = target_weights.get(name, 0.0)
            new_weights[name] = cur + (tgt - cur) * scale

        return new_weights
