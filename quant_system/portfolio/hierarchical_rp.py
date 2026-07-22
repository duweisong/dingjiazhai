"""
Hierarchical Risk Parity (HRP) — Lopez de Prado algorithm.

HRP addresses three major flaws of traditional MVO:
1. Instability: Small changes in inputs → large changes in weights
2. Concentration: MVO tends to allocate heavily to a few assets
3. Sensitivity: Requires inverting covariance matrix (ill-conditioned)

HRP uses hierarchical clustering on the correlation matrix, then
applies risk parity recursively within each cluster. No matrix
inversion required.

Man Group approach: "HRP is the most robust allocation method
we've tested. It's our default for multi-asset portfolios."
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger
from .constraints import Constraints

logger = get_logger(__name__)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity portfolio optimizer."""

    def __init__(
        self,
        linkage_method: str = "ward",
        distance_metric: str = "correlation",
    ):
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Constraints] = None,
    ) -> PortfolioWeights:
        """Compute HRP portfolio weights.

        Steps:
        1. Compute correlation matrix → distance matrix
        2. Hierarchical clustering → tree structure
        3. Quasi-diagonalization (reorder assets by tree)
        4. Recursive bisection: allocate risk equally at each split

        Args:
            returns: Historical returns (dates x assets).
            constraints: Optional constraints.

        Returns:
            PortfolioWeights with HRP weights.
        """
        if constraints is None:
            constraints = Constraints()

        assets = returns.columns.tolist()
        n = len(assets)

        if n <= 2:
            w = np.ones(n) / n
            weights = {assets[i]: float(w[i]) for i in range(n)}
            return PortfolioWeights(
                date=pd.Timestamp.now(), weights=weights, method="hrp"
            )

        # Step 1: Correlation → distance matrix
        corr = returns.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Step 2: Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        clusters = linkage(condensed_dist, method=self.linkage_method)

        # Step 3: Quasi-diagonalization (get sorted asset indices)
        sorted_idx = self._quasi_diagonalize(clusters, n)

        # Step 4: Recursive bisection
        cov = returns.cov().values
        w = self._recursive_bisection(cov, sorted_idx)

        weights = {assets[i]: float(w[i]) for i in range(n) if w[i] > 0.001}

        return PortfolioWeights(
            date=pd.Timestamp.now(),
            weights=weights,
            cash_pct=1.0 - sum(weights.values()),
            method="hrp",
        )

    def _quasi_diagonalize(self, clusters: np.ndarray, n: int) -> List[int]:
        """Reorder assets so similar ones are adjacent (quasi-diagonal)."""
        # The last 2*(n-1) cluster indices contain the ordering
        sorted_idx = []

        def traverse(node):
            if node < n:
                sorted_idx.append(int(node))
            else:
                left = int(clusters[node - n, 0])
                right = int(clusters[node - n, 1])
                traverse(left)
                traverse(right)

        traverse(2 * n - 2)  # Root node
        return sorted_idx

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_idx: List[int]
    ) -> np.ndarray:
        """Recursively split the sorted list and apply inverse-variance
        allocation at each split.
        """
        n = len(sorted_idx)
        w = np.ones(n) / n

        if n <= 1:
            return w

        # Recursively process clusters
        def _bisect(indices):
            if len(indices) <= 1:
                return

            # Variance of each cluster: w_sub' Σ_sub w_sub
            # Split into two halves
            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            # Sub-covariance matrices
            cov_left = cov[np.ix_(left, left)]
            cov_right = cov[np.ix_(right, right)]

            # Inverse variance allocation within each cluster
            # w_i ∝ 1/σ_i for assets within cluster
            # Cluster weight ∝ 1/σ_cluster
            try:
                ivp_left = 1.0 / np.diag(cov_left)
                ivp_left = ivp_left / np.sum(ivp_left)
                w_left = ivp_left
            except Exception:
                w_left = np.ones(len(left)) / len(left)

            try:
                ivp_right = 1.0 / np.diag(cov_right)
                ivp_right = ivp_right / np.sum(ivp_right)
                w_right = ivp_right
            except Exception:
                w_right = np.ones(len(right)) / len(right)

            # Cluster weights from inverse variance of cluster portfolios
            var_left = w_left @ cov_left @ w_left
            var_right = w_right @ cov_right @ w_right

            alpha_left = (1.0 / max(var_left, 1e-10)) if var_left > 0 else 0.5
            alpha_right = (1.0 / max(var_right, 1e-10)) if var_right > 0 else 0.5
            total = alpha_left + alpha_right
            alpha_left /= total
            alpha_right /= total

            # Assign weights
            for i, idx in enumerate(left):
                w[idx] = w[idx] * alpha_left * w_left[i]
            for i, idx in enumerate(right):
                w[idx] = w[idx] * alpha_right * w_right[i]

            # Recurse
            _bisect(left)
            _bisect(right)

        _bisect(sorted_idx)

        # Normalize
        w = w / np.sum(w)
        return w

    def get_cluster_structure(
        self, returns: pd.DataFrame
    ) -> Dict:
        """Return cluster tree structure for visualization.

        Returns:
            Dict with linkage matrix and asset labels.
        """
        corr = returns.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        condensed_dist = squareform(dist, checks=False)
        clusters = linkage(condensed_dist, method=self.linkage_method)

        return {
            "linkage": clusters.tolist(),
            "assets": returns.columns.tolist(),
            "method": self.linkage_method,
        }
