"""
Monte Carlo simulation for backtest validation.

Uses block bootstrap resampling of the trade sequence to estimate
the distribution of possible outcomes and assess statistical significance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..utils.types import MultiStockBacktestResult
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


class MonteCarloSimulator:
    """Block bootstrap Monte Carlo simulation.

    Resamples the sequence of trades (preserving within-trade structure)
    to generate a distribution of possible backtest outcomes. This answers:
    "Was my result luck, or is the strategy genuinely profitable?"
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()
        self.n_sims = self.config.backtest.monte_carlo_sims

    def run(
        self,
        result: MultiStockBacktestResult,
        method: str = "trade_block",
        block_size: int = 5,
    ) -> Dict:
        """Run Monte Carlo simulation on a backtest result.

        Args:
            result: Completed backtest result with daily returns.
            method: "trade_block" | "return_shuffle" | "parametric".
            block_size: Number of consecutive trades per block.

        Returns:
            Dict with MC distribution statistics.
        """
        returns = result.daily_returns.dropna().values

        if len(returns) < 30:
            logger.warning("Too few returns for meaningful MC simulation")
            return self._empty_result()

        if method == "return_shuffle":
            sim_metrics = self._shuffle_returns(returns)
        elif method == "parametric":
            sim_metrics = self._parametric_bootstrap(returns)
        else:
            # Default: trade block bootstrap
            sim_metrics = self._block_bootstrap(returns, block_size)

        return self._summarize(sim_metrics, result)

    def _shuffle_returns(self, returns: np.ndarray) -> List[Dict]:
        """Simple return shuffling (breaks autocorrelation structure)."""
        results = []
        for _ in range(self.n_sims):
            shuffled = np.random.permutation(returns)
            metrics = self._compute_path_metrics(shuffled)
            results.append(metrics)
        return results

    def _block_bootstrap(
        self, returns: np.ndarray, block_size: int
    ) -> List[Dict]:
        """Block bootstrap preserving local correlation structure."""
        n = len(returns)
        n_blocks = n // block_size + 1

        results = []
        for _ in range(self.n_sims):
            # Sample blocks with replacement
            block_starts = np.random.randint(0, max(n - block_size, 1), size=n_blocks)
            sampled = []
            for start in block_starts:
                end = min(start + block_size, n)
                sampled.extend(returns[start:end])
                if len(sampled) >= n:
                    break
            sampled = np.array(sampled[:n])
            metrics = self._compute_path_metrics(sampled)
            results.append(metrics)

        return results

    def _parametric_bootstrap(self, returns: np.ndarray) -> List[Dict]:
        """Parametric bootstrap: fit distribution, resample from fit."""
        mu = np.mean(returns)
        sigma = np.std(returns)

        results = []
        for _ in range(self.n_sims):
            sim_returns = np.random.normal(mu, sigma, len(returns))
            metrics = self._compute_path_metrics(sim_returns)
            results.append(metrics)

        return results

    def _compute_path_metrics(self, returns: np.ndarray) -> Dict:
        """Compute key metrics for a simulated return path."""
        total_ret = float(np.prod(1 + returns) - 1)
        ann_ret = float(np.mean(returns) * 244)
        ann_vol = float(np.std(returns) * np.sqrt(244))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

        # Drawdown
        equity = np.cumprod(1 + np.insert(returns, 0, 0))
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / peak))

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    def _summarize(
        self, sim_results: List[Dict], actual: MultiStockBacktestResult
    ) -> Dict:
        """Summarize MC simulation distribution."""
        if not sim_results:
            return self._empty_result()

        sharpes = [r["sharpe"] for r in sim_results]
        returns = [r["total_return"] for r in sim_results]
        drawdowns = [r["max_drawdown"] for r in sim_results]

        sharpe_median = float(np.median(sharpes))
        sharpe_p5 = float(np.percentile(sharpes, 5))
        sharpe_p95 = float(np.percentile(sharpes, 95))

        # How does actual compare to MC distribution?
        actual_sharpe = actual.sharpe_ratio
        prob_positive = float(np.mean(np.array(sharpes) > 0))
        actual_percentile = float(np.mean(np.array(sharpes) < actual_sharpe))

        return {
            "n_simulations": self.n_sims,
            "sharpe_median": sharpe_median,
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_p5": sharpe_p5,
            "sharpe_p95": sharpe_p95,
            "return_median": float(np.median(returns)),
            "drawdown_median": float(np.median(drawdowns)),
            "drawdown_p5": float(np.percentile(drawdowns, 5)),
            "prob_positive": prob_positive,
            "actual_sharpe": actual_sharpe,
            "actual_percentile": actual_percentile,
            "is_significant": actual_percentile > 0.95,  # > 95th percentile
            "summary": (
                f"MC ({self.n_sims} sims): Sharpe median={sharpe_median:.2f}, "
                f"5th={sharpe_p5:.2f}, 95th={sharpe_p95:.2f}. "
                f"Actual Sharpe ({actual_sharpe:.2f}) is at "
                f"{actual_percentile:.0%} percentile. "
                f"{'Statistically significant.' if actual_percentile > 0.95 else 'Not significant at 95% level.'}"
            ),
        }

    def _empty_result(self) -> Dict:
        return {
            "n_simulations": 0,
            "sharpe_median": 0.0,
            "sharpe_p5": 0.0,
            "sharpe_p95": 0.0,
            "prob_positive": 0.0,
            "actual_sharpe": 0.0,
            "actual_percentile": 0.0,
            "is_significant": False,
            "summary": "Insufficient data for Monte Carlo simulation.",
        }
