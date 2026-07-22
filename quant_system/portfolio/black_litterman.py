"""
Black-Litterman portfolio optimization.

Combines market equilibrium returns (prior) with investor views
to produce a posterior return distribution. The key insight:
you don't need return forecasts for ALL assets — only for the
ones you have views on.

Formula: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

Man Group approach: "Black-Litterman turns MVO from a garbage-in-
garbage-out machine into something you can actually use. The prior
stabilizes the optimization; the views add your edge."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class View:
    """An investor view on asset returns.

    Examples:
        "Stock A will outperform Stock B by 5%" →
            View(assets=["A", "B"], weights=[1, -1], value=0.05, confidence=0.3)

        "Stock C will return 10%" →
            View(assets=["C"], weights=[1], value=0.10, confidence=0.5)
    """
    assets: List[str]            # Asset identifiers
    weights: List[float]         # Portfolio weights for the view (sum to 0 for relative)
    value: float                 # Expected return of the view portfolio
    confidence: float            # 0.0 (no confidence) to 1.0 (certain)


class BlackLittermanOptimizer:
    """Black-Litterman model optimizer."""

    def __init__(self, tau: float = 0.05):
        """
        Args:
            tau: Uncertainty scaling factor for the prior covariance.
                 Smaller tau = more weight on prior (equilibrium).
                 Typical: 0.01 ~ 0.05.
        """
        self.tau = tau

    def optimize(
        self,
        returns: pd.DataFrame,
        market_weights: Optional[Dict[str, float]] = None,
        views: Optional[List[View]] = None,
        risk_aversion: float = 2.5,
    ) -> PortfolioWeights:
        """Compute Black-Litterman optimal weights.

        Args:
            returns: Historical returns (dates x assets).
            market_weights: Market cap weights (prior). If None, use equal weight.
            views: List of investor views.
            risk_aversion: Risk aversion parameter (δ).

        Returns:
            PortfolioWeights with BL-optimal weights.
        """
        assets = returns.columns.tolist()
        n = len(assets)

        # Covariance matrix
        sigma = returns.cov().values
        sigma = sigma + np.eye(n) * 1e-6

        # Prior (equilibrium) returns: Π = δ Σ w_market
        if market_weights:
            w_mkt = np.array([market_weights.get(a, 0.0) for a in assets])
            w_mkt = w_mkt / np.sum(w_mkt)
        else:
            w_mkt = np.ones(n) / n

        pi = risk_aversion * sigma @ w_mkt  # Equilibrium excess returns

        if not views:
            # No views → pure equilibrium
            posterior_mu = pi
        else:
            # Build view matrices
            P, Q, Omega = self._build_view_matrices(views, assets, sigma)

            # Posterior: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
            tau_sigma_inv = np.linalg.inv(self.tau * sigma)
            omega_inv = np.linalg.inv(Omega)

            M = tau_sigma_inv + P.T @ omega_inv @ P
            b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

            try:
                posterior_mu = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                logger.warning("BL posterior computation failed, using prior")
                posterior_mu = pi

        # Unconstrained BL weights: w* = (δΣ)⁻¹ μ_posterior
        try:
            sigma_inv = np.linalg.inv(sigma)
            w = sigma_inv @ posterior_mu / risk_aversion
            w = np.maximum(w, 0)  # Long-only
            w = w / np.sum(w)
        except np.linalg.LinAlgError:
            logger.warning("BL weight computation failed, using prior weights")
            w = w_mkt

        weights = {assets[i]: float(w[i]) for i in range(n) if w[i] > 0.001}

        return PortfolioWeights(
            date=pd.Timestamp.now(),
            weights=weights,
            cash_pct=1.0 - sum(weights.values()),
            method="black_litterman",
        )

    def _build_view_matrices(
        self, views: List[View], assets: List[str], sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the P (pick), Q (value), and Ω (uncertainty) matrices.

        P: k×n pick matrix (each row is a view portfolio)
        Q: k×1 vector of view expected returns
        Ω: k×k diagonal matrix of view uncertainties
        """
        k = len(views)
        n = len(assets)
        asset_to_idx = {a: i for i, a in enumerate(assets)}

        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))

        for i, view in enumerate(views):
            for j, asset in enumerate(view.assets):
                if asset in asset_to_idx:
                    idx = asset_to_idx[asset]
                    P[i, idx] = view.weights[j]

            Q[i] = view.value

            # Uncertainty: Ω_ii = (1/confidence - 1) * P_i Σ P_i'
            view_var = P[i] @ sigma @ P[i]
            Omega[i, i] = (1.0 / max(view.confidence, 0.01) - 1) * view_var

        return P, Q, Omega
