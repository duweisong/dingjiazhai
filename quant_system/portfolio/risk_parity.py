"""
Risk Parity / Equal Risk Contribution (ERC) optimization.

Allocates weights so that each asset contributes equally to
portfolio risk. More robust than MVO because it doesn't require
expected return estimates (which are notoriously noisy).

Risk contribution of asset i = w_i * (Σw)_i / sqrt(w'Σw)

Man Group approach: "Risk parity is MVO with the implicit assumption
that all Sharpe ratios are equal. It's almost always better than
raw MVO in practice."
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger
from .constraints import Constraints

logger = get_logger(__name__)


class RiskParityOptimizer:
    """Equal Risk Contribution portfolio optimizer."""

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Constraints] = None,
    ) -> PortfolioWeights:
        """Compute ERC portfolio weights.

        Minimizes: Σ_i Σ_j (RC_i - RC_j)²
        where RC_i = w_i * (Σw)_i is the risk contribution of asset i.

        Args:
            returns: Historical returns DataFrame (dates x assets).
            constraints: Optional constraints.

        Returns:
            PortfolioWeights with ERC weights.
        """
        if constraints is None:
            constraints = Constraints()

        assets = returns.columns.tolist()
        n = len(assets)

        if n == 0:
            return PortfolioWeights(date=pd.Timestamp.now(), weights={}, method="erc")
        if n == 1:
            return PortfolioWeights(
                date=pd.Timestamp.now(),
                weights={assets[0]: 1.0},
                method="erc",
            )

        sigma = returns.cov().values

        # Objective: minimize variance of risk contributions
        def risk_parity_objective(w):
            w = np.abs(w)
            w = w / np.sum(w)
            port_vol = np.sqrt(w @ sigma @ w)
            if port_vol < 1e-10:
                return 1e10
            # Marginal risk contributions
            mrc = sigma @ w
            # Risk contributions
            rc = w * mrc / port_vol
            # Target: equal contribution
            target_rc = 1.0 / n
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        cons = []
        if constraints.sum_to_one:
            cons.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

        bounds = [(0.0, constraints.max_weight) if constraints.long_only
                  else (None, None) for _ in range(n)]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        try:
            result = minimize(
                risk_parity_objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": self.max_iter, "ftol": 1e-12},
            )

            if result.success:
                w = np.abs(result.x)
                w = w / np.sum(w)
            else:
                logger.warning(f"ERC optimization did not converge: {result.message}")
                w = w0
        except Exception as e:
            logger.warning(f"ERC optimization failed: {e}, using equal weight")
            w = w0

        weights = {assets[i]: float(w[i]) for i in range(n) if w[i] > 0.001}

        return PortfolioWeights(
            date=pd.Timestamp.now(),
            weights=weights,
            cash_pct=1.0 - sum(weights.values()),
            method="erc",
        )

    def risk_contributions(
        self, weights: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Compute risk contribution of each asset.

        RC_i = w_i * (Σw)_i / σ_portfolio
        """
        port_vol = np.sqrt(weights @ sigma @ weights)
        if port_vol < 1e-10:
            return np.zeros_like(weights)
        mrc = sigma @ weights  # Marginal risk contributions
        rc = weights * mrc / port_vol
        return rc
