"""
Value at Risk (VaR) calculation — the cornerstone of risk management.

Four methods implemented:
1. Historical VaR — Empirical percentile of historical returns
2. Parametric VaR — Normal distribution assumption
3. Cornish-Fisher VaR — Adjusts for skewness and kurtosis
4. Monte Carlo VaR — Simulated from fitted distribution

Two Sigma approach: "Never trust a single VaR number. If four methods
disagree, the conservative one is probably right."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VaRResult:
    """Single VaR calculation result."""
    method: str
    var: float                      # Value at Risk (positive = loss)
    cvar: float                     # Conditional VaR (Expected Shortfall)
    confidence: float               # e.g. 0.95
    horizon: int                    # Days
    var_pct: float                  # VaR as % of portfolio value
    cvar_pct: float                 # CVaR as % of portfolio value


class VaRCalculator:
    """Multi-method VaR calculator.

    Computes Value at Risk and Conditional Value at Risk
    (Expected Shortfall) using four complementary approaches.
    """

    def __init__(self, confidence: float = 0.95, horizon: int = 1):
        self.confidence = confidence
        self.horizon = horizon

    def compute_all(
        self, returns: pd.Series, portfolio_value: float = 1.0
    ) -> Dict[str, VaRResult]:
        """Compute VaR using all four methods.

        Returns:
            Dict mapping method name -> VaRResult.
        """
        returns = returns.dropna()
        if len(returns) < 30:
            logger.warning("Insufficient return history for reliable VaR")
            return {}

        results = {}
        methods = {
            "historical": self.historical,
            "parametric": self.parametric,
            "cornish_fisher": self.cornish_fisher,
            "monte_carlo": self.monte_carlo,
        }

        for name, method in methods.items():
            try:
                results[name] = method(returns, portfolio_value)
            except Exception as e:
                logger.warning(f"VaR method '{name}' failed: {e}")

        return results

    def historical(
        self, returns: pd.Series, portfolio_value: float = 1.0
    ) -> VaRResult:
        """Historical VaR — empirical percentile.

        Simplest and most assumption-free method. Uses the empirical
        distribution of past returns. Best when you have 500+ observations.
        """
        alpha = 1 - self.confidence
        var = -float(np.percentile(returns, alpha * 100)) * np.sqrt(self.horizon)
        cvar = -float(returns[returns <= -var / np.sqrt(self.horizon)].mean()) * np.sqrt(self.horizon)

        return VaRResult(
            method="historical",
            var=var,
            cvar=cvar,
            confidence=self.confidence,
            horizon=self.horizon,
            var_pct=var,
            cvar_pct=cvar,
        )

    def parametric(
        self, returns: pd.Series, portfolio_value: float = 1.0
    ) -> VaRResult:
        """Parametric VaR — assumes normal distribution.

        VaR = -(μ + σ * z_α) * sqrt(horizon)
        where z_α is the normal distribution quantile.

        Fast and widely used, but underestimates tail risk
        when returns are fat-tailed (kurtosis > 3).
        """
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        # z_α = Φ⁻¹(α), e.g., for 95% VaR, z = 1.645
        z = stats.norm.ppf(self.confidence)

        # VaR_α = -(μ - z_α * σ) = z_α * σ - μ (positive = loss)
        var = (z * sigma - mu) * np.sqrt(self.horizon)
        var = max(var, 0.0)  # Floor at zero
        # CVaR for normal: σ * φ(z) / (1-α) - μ
        cvar = (sigma * stats.norm.pdf(z) / (1 - self.confidence) - mu) * np.sqrt(self.horizon)
        cvar = max(cvar, var)

        return VaRResult(
            method="parametric",
            var=var,
            cvar=cvar,
            confidence=self.confidence,
            horizon=self.horizon,
            var_pct=var,
            cvar_pct=cvar,
        )

    def cornish_fisher(
        self, returns: pd.Series, portfolio_value: float = 1.0
    ) -> VaRResult:
        """Cornish-Fisher VaR — adjusts normal quantile for skewness and kurtosis.

        More accurate than parametric when returns are non-normal
        (which they almost always are in real markets).
        """
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # Excess kurtosis

        z = stats.norm.ppf(self.confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        var = (z_cf * sigma - mu) * np.sqrt(self.horizon)
        var = max(var, 0.0)
        cvar = var * 1.3  # Conservative CVaR estimate under CF
        cvar = max(cvar, var)

        return VaRResult(
            method="cornish_fisher",
            var=var,
            cvar=cvar,
            confidence=self.confidence,
            horizon=self.horizon,
            var_pct=var,
            cvar_pct=cvar,
        )

    def monte_carlo(
        self, returns: pd.Series, portfolio_value: float = 1.0, n_sims: int = 10000
    ) -> VaRResult:
        """Monte Carlo VaR — simulate from fitted t-distribution.

        Fits a Student's t-distribution (which has fatter tails than normal)
        and simulates future returns.
        """
        # Fit t-distribution
        params = stats.t.fit(returns)
        df, loc, scale = params

        # Simulate
        sim_returns = stats.t.rvs(df=df, loc=loc, scale=scale, size=n_sims)

        alpha = 1 - self.confidence
        var = -float(np.percentile(sim_returns, alpha * 100)) * np.sqrt(self.horizon)
        cvar = -float(sim_returns[sim_returns <= -var / np.sqrt(self.horizon)].mean()) * np.sqrt(self.horizon)

        return VaRResult(
            method="monte_carlo",
            var=var,
            cvar=cvar,
            confidence=self.confidence,
            horizon=self.horizon,
            var_pct=var,
            cvar_pct=cvar,
        )

    def best_estimate(self, results: Dict[str, VaRResult]) -> VaRResult:
        """Select the most conservative (highest) VaR estimate.

        Two Sigma principle: when methods disagree, the conservative
        one is least likely to surprise you.
        """
        if not results:
            return VaRResult("none", 0, 0, self.confidence, self.horizon, 0, 0)

        # Pick highest VaR (most conservative)
        best = max(results.values(), key=lambda r: r.var)
        return best

    def summary(self, results: Dict[str, VaRResult]) -> str:
        """Generate human-readable VaR summary."""
        lines = [f"VaR Analysis (confidence={self.confidence:.0%}, horizon={self.horizon}d)"]
        lines.append("-" * 50)
        for name, r in results.items():
            lines.append(
                f"  {name:<18} VaR={r.var_pct:.4%}  CVaR={r.cvar_pct:.4%}"
            )

        best = self.best_estimate(results)
        lines.append(f"\n  → Conservative estimate: VaR={best.var_pct:.4%} ({best.method})")
        lines.append(f"    Meaning: {best.confidence:.0%} chance daily loss ≤ {best.var_pct:.4%}")

        return "\n".join(lines)
