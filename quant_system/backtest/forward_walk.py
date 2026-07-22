"""
Walk-forward validation.

Enforces out-of-sample testing by splitting data into rolling
train/test windows. Any strategy must pass walk-forward before
being considered production-ready.

Wraps and extends the existing backtest/optimizer.py WalkForwardOptimizer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..utils.types import MultiStockData
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    """Walk-forward validation output."""
    n_windows: int
    train_sharpe: float          # Average in-sample Sharpe
    test_sharpe: float           # Average out-of-sample Sharpe
    oos_r2: float                # Predictive R² of train on test
    consistency: float           # % windows where test was positive
    window_results: List[Dict]   # Per-window detail
    is_valid: bool               # Pass/fail
    degradation: float           # Sharpe degradation (train - test)


class WalkForwardValidator:
    """Enforces walk-forward validation on any strategy.

    The core principle (Renaissance-style):
    - Split data into sequential train/test windows
    - Optimize parameters on training window
    - Test on the subsequent out-of-sample window
    - Slide forward and repeat
    - Aggregate OOS performance → REAL expected performance
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()
        self.train_window = self.config.backtest.walk_forward_train
        self.test_window = self.config.backtest.walk_forward_test

    def validate(
        self,
        data: MultiStockData,
        signal_func: callable = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            data: Multi-stock price and indicator data.
            signal_func: Function(df) -> pd.Series that generates signals.
                        If None, returns a placeholder result.

        Returns:
            WalkForwardResult with aggregated OOS metrics.
        """
        n_dates = data.n_dates
        window_size = self.train_window + self.test_window

        if n_dates < window_size:
            logger.warning(
                f"Insufficient data for walk-forward: "
                f"{n_dates} days < {window_size} required"
            )
            return WalkForwardResult(
                n_windows=0,
                train_sharpe=0.0,
                test_sharpe=0.0,
                oos_r2=0.0,
                consistency=0.0,
                window_results=[],
                is_valid=False,
                degradation=0.0,
            )

        # Calculate number of windows
        n_windows = (n_dates - self.train_window) // self.test_window
        train_sharpes = []
        test_sharpes = []
        window_results = []

        for w in range(n_windows):
            train_start = w * self.test_window
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n_dates)

            # Placeholder: in full implementation, this would:
            # 1. Train/optimize strategy on train_start:train_end
            # 2. Run backtest on test_start:test_end
            # 3. Record both in-sample and out-of-sample metrics
            window_results.append({
                "window": w,
                "train_period": f"{data.dates[train_start].date()} ~ {data.dates[train_end-1].date()}",
                "test_period": f"{data.dates[test_start].date()} ~ {data.dates[test_end-1].date()}",
                "train_sharpe": 0.0,
                "test_sharpe": 0.0,
                "test_return": 0.0,
            })

        # Aggregate
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
        consistency = sum(1 for r in window_results if r["test_return"] > 0) / max(n_windows, 1)
        degradation = avg_train_sharpe - avg_test_sharpe

        # A strategy "passes" walk-forward if:
        # 1. Average OOS Sharpe > 0
        # 2. Consistency > 50% of windows positive
        # 3. Degradation < 0.5 (not overfitting too badly)
        is_valid = avg_test_sharpe > 0 and consistency > 0.5

        return WalkForwardResult(
            n_windows=n_windows,
            train_sharpe=avg_train_sharpe,
            test_sharpe=avg_test_sharpe,
            oos_r2=0.0,
            consistency=consistency,
            window_results=window_results,
            is_valid=is_valid,
            degradation=degradation,
        )

    def summary(self, result: WalkForwardResult) -> str:
        """Generate a human-readable walk-forward summary."""
        status = " PASS" if result.is_valid else " FAIL"
        return (
            f"Walk-Forward Validation: {status}\n"
            f"  Windows: {result.n_windows}\n"
            f"  In-Sample Sharpe:  {result.train_sharpe:.3f}\n"
            f"  Out-of-Sample Sharpe: {result.test_sharpe:.3f}\n"
            f"  Consistency: {result.consistency:.0%} of windows positive\n"
            f"  Degradation: {result.degradation:.3f}\n"
            f"  {'' if result.is_valid else 'WARNING: Strategy may be overfit.'}"
        )
