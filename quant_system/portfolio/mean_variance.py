"""
Mean-Variance Optimization (Markowitz).

Classic MVO with Ledoit-Wolf shrinkage for robust covariance estimation.
Shrinkage reduces estimation error by blending the sample covariance
matrix with a structured estimator (constant correlation target).

Man Group approach: "Raw sample covariance produces extreme corner
solutions. Always shrink. Always. The only question is how much."
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger
from .constraints import Constraints

logger = get_logger(__name__)


class MeanVarianceOptimizer:
    """Markowitz mean-variance optimization with shrinkage."""

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Constraints] = None,
        shrinkage: str = "ledoit_wolf",
    ) -> PortfolioWeights:
        """Compute optimal mean-variance portfolio.

        Solves: max (w'μ - λ/2 * w'Σw) subject to constraints.

        Args:
            returns: DataFrame (dates x assets) of historical returns.
            constraints: Optimization constraints.
            shrinkage: "ledoit_wolf" | "sample" | "constant_correlation".

        Returns:
            PortfolioWeights with optimized weights.
        """
        if constraints is None:
            constraints = Constraints()

        n_assets = returns.shape[1]
        if n_assets == 0 or len(returns) == 0:
            return PortfolioWeights(date=pd.Timestamp.now(), weights={}, method="mvo")

        assets = returns.columns.tolist()

        # Expected returns (simple mean)
        mu = returns.mean().values

        # Covariance matrix
        if shrinkage == "ledoit_wolf":
            sigma = self._ledoit_wolf_shrinkage(returns.values)
        elif shrinkage == "constant_correlation":
            sigma = self._constant_correlation_shrinkage(returns.values)
        else:
            sigma = returns.cov().values

        # Add small ridge to ensure positive definiteness
        sigma += np.eye(n_assets) * 1e-6

        # Solve: min (w'Σw - w'μ/λ) subject to constraints
        try:
            # Unconstrained solution: w* = Σ⁻¹μ / λ
            sigma_inv = np.linalg.inv(sigma)
            w = sigma_inv @ mu / self.risk_aversion

            # Normalize to sum to 1
            w = w / np.sum(w)

            # Apply long-only + max weight constraints
            if constraints.long_only:
                w = np.maximum(w, 0)
                w = w / np.sum(w)

            if constraints.max_weight:
                # Iterative clipping
                for _ in range(10):
                    over = w > constraints.max_weight
                    if not over.any():
                        break
                    excess = np.sum(w[over] - constraints.max_weight)
                    w[over] = constraints.max_weight
                    under_mask = ~over & (w > 0)
                    if under_mask.any():
                        w[under_mask] += excess / under_mask.sum()
                    w = np.maximum(w, 0)
                    w = w / np.sum(w)

        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed in MVO, using equal weight")
            w = np.ones(n_assets) / n_assets

        weights = {assets[i]: float(w[i]) for i in range(n_assets) if w[i] > 0.001}

        return PortfolioWeights(
            date=pd.Timestamp.now(),
            weights=weights,
            cash_pct=1.0 - sum(weights.values()),
            method="mvo",
        )

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf shrinkage estimator for covariance matrix.

        Shrinks sample covariance toward a constant-correlation target.
        The shrinkage intensity is analytically determined to minimize
        expected Frobenius norm of the estimation error.
        """
        n, p = returns.shape

        # Sample covariance
        S = np.cov(returns, rowvar=False)

        # Constant correlation target
        stds = np.sqrt(np.diag(S))
        correlations = S / np.outer(stds, stds)

        # Average correlation
        tril_idx = np.tril_indices(p, k=-1)
        avg_corr = np.mean(correlations[tril_idx])

        # Target: constant correlation
        T = np.outer(stds, stds) * avg_corr
        np.fill_diagonal(T, np.diag(S))

        # Shrinkage intensity (simplified LW formula)
        # π = sum of asymptotic variances / sum of squared differences
        diff = (S - T) ** 2
        pi_hat = np.sum(diff)
        # Variance of S elements
        gamma_hat = 0
        for i in range(n):
            x = returns[i] - np.mean(returns, axis=0)
            gamma_hat += np.sum((np.outer(x, x) - S) ** 2)
        gamma_hat /= n**2

        shrinkage = min(gamma_hat / max(pi_hat, 1e-10), 1.0)

        # Shrink
        sigma_shrunk = shrinkage * T + (1 - shrinkage) * S
        return sigma_shrunk

    def _constant_correlation_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Simpler: directly use constant correlation target."""
        n, p = returns.shape
        S = np.cov(returns, rowvar=False)
        stds = np.sqrt(np.diag(S))
        correlations = S / np.outer(stds, stds)
        tril_idx = np.tril_indices(p, k=-1)
        avg_corr = np.mean(correlations[tril_idx])

        T = np.outer(stds, stds) * avg_corr
        np.fill_diagonal(T, np.diag(S))
        return T * 0.5 + S * 0.5  # 50-50 blend

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 20,
        constraints: Optional[Constraints] = None,
    ) -> pd.DataFrame:
        """Generate the efficient frontier.

        Returns:
            DataFrame with columns: return, volatility, sharpe.
        """
        if constraints is None:
            constraints = Constraints()

        mu = returns.mean().values * 244
        sigma = self._ledoit_wolf_shrinkage(returns.values) * 244

        # Range of target returns
        min_ret = np.min(mu) if constraints.long_only else np.min(mu) * 0.5
        max_ret = np.max(mu)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            try:
                weights = self.optimize(returns, constraints)
                port_ret = sum(
                    w * mu[i] for i, (_, w) in enumerate(weights.weights.items())
                )
                port_vol = np.sqrt(
                    sum(
                        weights.weights[a] * sigma[i, j] * weights.weights[b]
                        for i, a in enumerate(weights.weights)
                        for j, b in enumerate(weights.weights)
                    )
                )
                frontier.append({
                    "return": port_ret,
                    "volatility": port_vol,
                    "sharpe": port_ret / port_vol if port_vol > 0 else 0,
                })
            except Exception:
                pass

        return pd.DataFrame(frontier)
