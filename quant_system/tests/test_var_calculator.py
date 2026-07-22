"""Tests for VaR Calculator."""

import pytest
import numpy as np
import pandas as pd

from quant_system.risk.var_calculator import VaRCalculator


class TestVaRCalculator:
    """Test VaR calculation methods."""

    def test_historical_var(self):
        """Historical VaR should return a positive loss number."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.randn(500) * 0.02)  # 2% daily vol

        result = calc.historical(returns)
        assert result.var > 0, "VaR should be positive (representing potential loss)"
        assert result.cvar >= result.var, "CVaR should be >= VaR"
        assert result.var_pct < 0.10, "VaR should be reasonable for 2% daily vol"

    def test_parametric_var(self):
        """Parametric VaR under normal distribution."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.normal(0, 0.02, 500))

        result = calc.parametric(returns)
        assert result.var > 0
        # For normal returns with 2% vol, 95% VaR ≈ 1.645 * 2% ≈ 3.3%
        assert 0.02 < result.var_pct < 0.05

    def test_cornish_fisher_var(self):
        """Cornish-Fisher should handle non-normal returns."""
        calc = VaRCalculator(confidence=0.95)
        # Fat-tailed returns (t-distribution with df=3)
        returns = pd.Series(np.random.standard_t(3, 500) * 0.02)

        result = calc.cornish_fisher(returns)
        assert result.var > 0

    def test_monte_carlo_var(self):
        """Monte Carlo VaR should produce reasonable estimates."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.randn(500) * 0.02)

        result = calc.monte_carlo(returns, n_sims=5000)
        assert result.var > 0
        assert result.cvar >= result.var

    def test_compute_all_returns_dict(self):
        """compute_all should return results for multiple methods."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.randn(300) * 0.02)

        results = calc.compute_all(returns)
        assert len(results) >= 3  # Should have at least 3 methods
        for method, result in results.items():
            assert result.var > 0

    def test_best_estimate_picks_conservative(self):
        """Best estimate should pick the most conservative VaR."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.randn(300) * 0.02)

        results = calc.compute_all(returns)
        best = calc.best_estimate(results)
        max_var_result = max(results.values(), key=lambda r: r.var)
        assert best.var == max_var_result.var

    def test_empty_returns(self):
        """Should handle insufficient data gracefully."""
        calc = VaRCalculator(confidence=0.95)
        returns = pd.Series(np.random.randn(10) * 0.02)

        results = calc.compute_all(returns)
        assert len(results) == 0  # Too few observations
