"""Tests for Signal Tester (IC analysis)."""

import pytest
import numpy as np
import pandas as pd

from quant_system.utils.types import FactorData
from quant_system.alpha.signal_tester import SignalTester


class TestSignalTester:
    """Test signal testing framework."""

    @pytest.fixture
    def tester(self):
        return SignalTester(min_stocks_per_period=5)

    @pytest.fixture
    def predictive_factor(self) -> FactorData:
        """Create a factor that actually predicts returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        stocks = [f"STOCK{i}" for i in range(20)]

        values = pd.DataFrame(
            np.random.randn(200, 20),
            index=dates,
            columns=stocks,
        )

        return FactorData(
            name="test_momentum",
            display_name="Test Momentum",
            direction=1,
            values=values,
            category="momentum",
        )

    @pytest.fixture
    def forward_returns(self, predictive_factor) -> pd.DataFrame:
        """Create forward returns correlated with the factor."""
        np.random.seed(42)
        values = predictive_factor.values * 0.01 + np.random.randn(200, 20) * 0.005
        return pd.DataFrame(values, index=predictive_factor.values.index, columns=predictive_factor.values.columns)

    def test_ic_computation(self, tester, predictive_factor, forward_returns):
        """IC should be computed and return reasonable values."""
        result = tester.information_coefficient(
            predictive_factor, forward_returns, horizon=1
        )

        assert result.factor_name == "test_momentum"
        assert result.horizon == 1
        assert -1 <= result.pearson_ic_mean <= 1
        assert result.ic_t_stat != 0  # Should have some t-stat
        assert len(result.ic_series) > 0

    def test_ic_on_noisy_factor(self, tester):
        """A pure-noise factor should have IC close to zero."""
        np.random.seed(99)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        stocks = [f"S{i}" for i in range(20)]

        noise = FactorData(
            name="noise",
            display_name="Noise",
            direction=1,
            values=pd.DataFrame(np.random.randn(200, 20), index=dates, columns=stocks),
            category="custom",
        )
        fwd = pd.DataFrame(np.random.randn(200, 20) * 0.01, index=dates, columns=stocks)

        result = tester.information_coefficient(noise, fwd)
        # Noise IC should be close to 0
        assert abs(result.pearson_ic_mean) < 0.10

    def test_quantile_analysis(self, tester, predictive_factor, forward_returns):
        """Quantile analysis should return per-quantile metrics."""
        result = tester.quantile_analysis(
            predictive_factor, forward_returns, n_quantiles=5
        )

        assert result.n_quantiles == 5
        assert len(result.quantile_returns) > 0
        assert -1 <= result.monotonicity <= 1

    def test_fama_macbeth(self, tester, predictive_factor, forward_returns):
        """Fama-MacBeth should return coefficient estimates."""
        result = tester.fama_macbeth(
            [predictive_factor], forward_returns
        )

        assert len(result.coefficients) == 1
        assert "test_momentum" in result.coefficients
        assert result.n_periods > 0

    def test_insufficient_data(self, tester):
        """Should handle very small datasets gracefully."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        small = FactorData(
            name="tiny",
            display_name="Tiny",
            direction=1,
            values=pd.DataFrame(np.random.randn(10, 2), index=dates, columns=["A", "B"]),
            category="test",
        )
        fwd = pd.DataFrame(np.random.randn(10, 2) * 0.01, index=dates, columns=["A", "B"])

        result = tester.information_coefficient(small, fwd)
        # Should return zero-filled result for insufficient data
        assert result.pearson_ic_mean == 0.0
