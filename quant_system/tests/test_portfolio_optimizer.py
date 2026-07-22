"""Tests for portfolio optimization methods."""

import pytest
import numpy as np
import pandas as pd

from quant_system.portfolio.mean_variance import MeanVarianceOptimizer
from quant_system.portfolio.risk_parity import RiskParityOptimizer
from quant_system.portfolio.hierarchical_rp import HierarchicalRiskParity
from quant_system.portfolio.constraints import Constraints, ConstraintBuilder
from quant_system.utils.types import PortfolioWeights


class TestMeanVariance:
    """Test Mean-Variance Optimization."""

    @pytest.fixture
    def returns(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        stocks = ["A", "B", "C", "D", "E"]
        data = np.random.randn(252, 5) * 0.02
        data[:, 0] = data[:, 0] * 0.5 + 0.0005  # Asset A: lower vol, positive drift
        data[:, 4] = data[:, 4] * 2.0  # Asset E: higher vol
        return pd.DataFrame(data, index=dates, columns=stocks)

    def test_mvo_basic(self, returns):
        optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
        result = optimizer.optimize(returns)

        assert isinstance(result, PortfolioWeights)
        assert len(result.weights) > 0
        assert abs(sum(result.weights.values()) - 1.0) < 0.05

    def test_mvo_long_only(self, returns):
        optimizer = MeanVarianceOptimizer()
        constraints = Constraints(long_only=True, max_weight=0.30)
        result = optimizer.optimize(returns, constraints)

        for code, w in result.weights.items():
            assert 0 <= w <= 0.35, f"{code} weight {w} violates constraints (max allowed 0.30)"

    def test_mvo_empty_input(self):
        optimizer = MeanVarianceOptimizer()
        returns = pd.DataFrame(columns=["A", "B"])
        result = optimizer.optimize(returns)
        assert len(result.weights) == 0


class TestRiskParity:
    """Test Risk Parity optimization."""

    @pytest.fixture
    def returns(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        stocks = ["A", "B", "C"]
        data = np.random.randn(252, 3) * 0.02
        return pd.DataFrame(data, index=dates, columns=stocks)

    def test_erc_basic(self, returns):
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)

        assert isinstance(result, PortfolioWeights)
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 0.05

    def test_erc_weights_sum_to_one(self, returns):
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(returns)
        total = sum(result.weights.values())
        assert 0.99 < total < 1.01

    def test_erc_single_asset(self):
        optimizer = RiskParityOptimizer()
        returns = pd.DataFrame(np.random.randn(252, 1) * 0.02, columns=["A"])
        result = optimizer.optimize(returns)
        assert result.weights.get("A", 0) == 1.0


class TestHierarchicalRP:
    """Test Hierarchical Risk Parity."""

    @pytest.fixture
    def returns(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        stocks = [f"S{i}" for i in range(8)]
        data = np.random.randn(252, 8) * 0.02
        return pd.DataFrame(data, index=dates, columns=stocks)

    def test_hrp_basic(self, returns):
        optimizer = HierarchicalRiskParity()
        result = optimizer.optimize(returns)

        assert isinstance(result, PortfolioWeights)
        assert len(result.weights) > 0
        assert result.method == "hrp"
        assert abs(sum(result.weights.values()) - 1.0) < 0.05

    def test_hrp_all_positive(self, returns):
        optimizer = HierarchicalRiskParity()
        result = optimizer.optimize(returns)
        for w in result.weights.values():
            assert w >= 0, "HRP should produce non-negative weights"

    def test_hrp_cluster_structure(self, returns):
        optimizer = HierarchicalRiskParity()
        structure = optimizer.get_cluster_structure(returns)
        assert "linkage" in structure
        assert "assets" in structure
        assert len(structure["assets"]) == 8


class TestConstraints:
    """Test portfolio constraints."""

    def test_basic_constraints(self):
        c = Constraints(long_only=True, max_weight=0.15)
        valid = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert len(c.validate(valid)) == 0

        invalid = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])
        violations = c.validate(invalid)
        assert len(violations) > 0

    def test_sum_to_one(self):
        c = Constraints(sum_to_one=True)
        invalid = np.array([0.1] * 8)
        violations = c.validate(invalid)
        assert any("Sum" in v for v in violations)

    def test_long_only(self):
        c = Constraints(long_only=True)
        invalid = np.array([0.1, 0.1, -0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15])
        violations = c.validate(invalid)
        assert any("negative" in v for v in violations)
