"""Backtest engine extensions for multi-stock portfolio simulation.

Renaissance-style rigorous backtesting with walk-forward validation,
Monte Carlo simulation, statistical significance testing, and
survivorship bias correction.
"""

from .multi_stock_engine import MultiStockBacktestEngine
from .forward_walk import WalkForwardValidator
from .monte_carlo import MonteCarloSimulator
from .significance import SignificanceTester
from .survivorship import SurvivorshipAdjuster

__all__ = [
    "MultiStockBacktestEngine",
    "WalkForwardValidator",
    "MonteCarloSimulator",
    "SignificanceTester",
    "SurvivorshipAdjuster",
]
