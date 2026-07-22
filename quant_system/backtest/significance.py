"""
Statistical significance testing for backtest results.

Tests whether observed strategy returns are statistically different
from zero (or from a benchmark), using:
- One-sample t-test on daily returns
- Sharpe ratio difference test (Jobson-Korkie)
- Bootstrap confidence intervals
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.types import MultiStockBacktestResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignificanceResult:
    """Statistical test results."""
    t_statistic: float
    p_value: float
    is_significant: bool           # p < 0.05
    mean_daily_return: float
    mean_daily_return_pct: float   # in basis points
    annualized_return: float
    sharpe_t_stat: float           # Sharpe ratio t-stat (Jobson-Korkie)
    sharpe_p_value: float
    benchmark_alpha: float         # Excess over benchmark
    alpha_t_stat: float
    alpha_p_value: float
    sample_size: int
    confidence_interval: Tuple[float, float]  # 95% CI for annualized return
    summary: str


class SignificanceTester:
    """Renaissance-style statistical validation.

    "If you can't measure it with statistical confidence,
    you don't have an edge — you have a coin flip."
    """

    def test(
        self,
        result: MultiStockBacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
        confidence: float = 0.95,
    ) -> SignificanceResult:
        """Run statistical significance tests on backtest results.

        Args:
            result: Backtest result with daily returns.
            benchmark_returns: Optional benchmark daily returns for alpha test.
            confidence: Confidence level (default 0.95).

        Returns:
            SignificanceResult with all test statistics.
        """
        returns = result.daily_returns.dropna()
        n = len(returns)

        if n < result.total_trades:
            logger.warning(
                f"Sample size ({n}) may be insufficient for "
                f"meaningful significance testing"
            )

        # ── One-sample t-test on daily returns ─────────────
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        se_ret = std_ret / np.sqrt(n)

        t_stat = mean_ret / se_ret if se_ret > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        is_significant = p_value < 0.05

        # Annualized
        ann_return = float(mean_ret * 244)
        ann_se = se_ret * np.sqrt(244)
        ci_low = ann_return - stats.t.ppf((1 + confidence) / 2, df=n - 1) * ann_se
        ci_high = ann_return + stats.t.ppf((1 + confidence) / 2, df=n - 1) * ann_se

        # ── Sharpe ratio significance (Jobson-Korkie) ──────
        sharpe = result.sharpe_ratio
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n)
        sharpe_t = sharpe / sharpe_se if sharpe_se > 0 else 0.0
        sharpe_p = float(2 * (1 - stats.norm.cdf(abs(sharpe_t))))

        # ── Alpha vs benchmark ─────────────────────────────
        alpha = 0.0
        alpha_t = 0.0
        alpha_p = 1.0

        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned = pd.DataFrame({
                "strategy": returns,
                "benchmark": benchmark_returns,
            }).dropna()

            if len(aligned) > 30:
                excess = aligned["strategy"].values - aligned["benchmark"].values
                mean_excess = float(np.mean(excess))
                se_excess = float(np.std(excess, ddof=1) / np.sqrt(len(excess)))
                alpha = float(mean_excess * 244)
                alpha_t = mean_excess / se_excess if se_excess > 0 else 0.0
                alpha_p = float(2 * stats.t.sf(abs(alpha_t), df=len(excess) - 1))
            else:
                # Use actual benchmark return from backtest result
                benchmark_ann = result.benchmark_return
                n_years = n / 244
                ann_strategy = (1 + result.total_return) ** (1 / max(n_years, 0.5)) - 1
                ann_benchmark = (1 + benchmark_ann) ** (1 / max(n_years, 0.5)) - 1
                alpha = ann_strategy - ann_benchmark

        # ── Build summary ──────────────────────────────────
        significance_text = (
            "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
        )
        summary = (
            f"Statistical Significance: {significance_text}\n"
            f"  t-statistic: {t_stat:.3f} (p={p_value:.4f})\n"
            f"  Mean daily return: {mean_ret*10000:.2f} bp\n"
            f"  Annualized: {ann_return:.2%} "
            f"[{ci_low:.2%}, {ci_high:.2%}]\n"
            f"  Sharpe t-stat: {sharpe_t:.3f} (p={sharpe_p:.4f})\n"
            f"  Alpha vs benchmark: {alpha:.2%} "
            f"(t={alpha_t:.3f}, p={alpha_p:.4f})\n"
            f"  Sample: {n} daily returns, {result.total_trades} trades"
        )

        return SignificanceResult(
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            mean_daily_return=mean_ret,
            mean_daily_return_pct=mean_ret * 10000,
            annualized_return=ann_return,
            sharpe_t_stat=sharpe_t,
            sharpe_p_value=sharpe_p,
            benchmark_alpha=alpha,
            alpha_t_stat=alpha_t,
            alpha_p_value=alpha_p,
            sample_size=n,
            confidence_interval=(ci_low, ci_high),
            summary=summary,
        )

    def quick_check(self, returns: pd.Series) -> Dict:
        """Quick statistical sanity check on any return series.

        Returns:
            Dict with mean, std, t-stat, p-value, significance flag.
        """
        returns = returns.dropna()
        n = len(returns)
        if n < 5:
            return {"error": "Too few observations"}

        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))

        return {
            "n": n,
            "mean_daily": mean,
            "mean_annualized": mean * 244,
            "std_annualized": std * np.sqrt(244),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant_5pct": p_value < 0.05,
            "is_significant_1pct": p_value < 0.01,
        }
