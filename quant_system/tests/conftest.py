"""Shared test fixtures for the quant system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Generate synthetic return data for 10 stocks over 500 days."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    stocks = [f"00000{i}.SZ" for i in range(10)]

    # Generate correlated returns
    n = len(dates)
    market_return = np.random.normal(0.0005, 0.015, n)  # ~8% annual, 24% vol

    data = {}
    for stock in stocks:
        stock_specific = np.random.normal(0, 0.01, n)
        stock_return = market_return * (0.8 + 0.4 * np.random.random()) + stock_specific
        data[stock] = stock_return

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_prices(sample_returns) -> pd.DataFrame:
    """Generate synthetic price data from returns."""
    prices = (1 + sample_returns).cumprod() * 100
    return prices


@pytest.fixture
def sample_weights():
    """Generate sample portfolio weights."""
    return {
        "0000001.SZ": 0.15,
        "0000002.SZ": 0.15,
        "0000003.SZ": 0.12,
        "0000004.SZ": 0.12,
        "0000005.SZ": 0.10,
        "0000006.SZ": 0.10,
        "0000007.SZ": 0.08,
        "0000008.SZ": 0.08,
        "0000009.SZ": 0.05,
        "0000010.SZ": 0.05,
    }


@pytest.fixture
def sample_factor_values(sample_returns) -> pd.DataFrame:
    """Generate synthetic factor values (dates x stocks)."""
    np.random.seed(123)
    return pd.DataFrame(
        np.random.randn(500, 10) * 0.5 + 0.1,
        index=sample_returns.index,
        columns=sample_returns.columns,
    )
