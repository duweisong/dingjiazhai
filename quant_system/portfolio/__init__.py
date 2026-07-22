"""Portfolio Optimizer — Man Group-style capital allocation.

Multi-method optimization with automatic fallback chain:
MVO → ERC → HRP → Equal Weight (never fails silently).
"""

from .mean_variance import MeanVarianceOptimizer
from .black_litterman import BlackLittermanOptimizer
from .risk_parity import RiskParityOptimizer
from .hierarchical_rp import HierarchicalRiskParity
from .constraints import Constraints, ConstraintBuilder
from .rebalancer import Rebalancer

__all__ = [
    "MeanVarianceOptimizer",
    "BlackLittermanOptimizer",
    "RiskParityOptimizer",
    "HierarchicalRiskParity",
    "Constraints",
    "ConstraintBuilder",
    "Rebalancer",
]
