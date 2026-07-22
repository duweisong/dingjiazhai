"""
Signal testing framework — the heart of alpha research.

Tests whether a signal (factor) has predictive power for future returns
using the gold-standard metrics:
- Information Coefficient (IC): Pearson + Spearman rank correlation
- IC_IR: Information Coefficient / std(IC) — measures consistency
- Quantile analysis: Returns by factor quintile, spread, monotonicity
- Fama-MacBeth regression: Cross-sectional regression of returns on factors

Citadel-style: "A signal that can't survive IC testing across regimes
is not a signal — it's a historical coincidence."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from ..utils.types import FactorData
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ICResult:
    """Information Coefficient analysis result."""
    factor_name: str
    horizon: int                                  # Forward return horizon in days
    pearson_ic_mean: float
    pearson_ic_std: float
    pearson_ic_ir: float                          # IC / std(IC)
    spearman_ic_mean: float
    spearman_ic_ir: float
    ic_series: pd.Series                          # IC over time
    ic_t_stat: float
    ic_p_value: float
    positive_ic_pct: float                        # % of periods with positive IC
    summary: str = ""


@dataclass
class QuantileResult:
    """Quantile (quintile) analysis result."""
    factor_name: str
    horizon: int
    n_quantiles: int
    quantile_returns: Dict[int, float]            # quantile -> mean forward return
    spread: float                                 # Q5 - Q1 return spread
    spread_t_stat: float
    monotonicity: float                           # Spearman correlation of Q-order vs return
    quantile_sharpes: Dict[int, float]
    summary: str = ""


@dataclass
class FMResult:
    """Fama-MacBeth regression result."""
    factor_names: List[str]
    horizon: int
    coefficients: Dict[str, float]
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    n_periods: int
    avg_r_squared: float
    summary: str = ""


class SignalTester:
    """Comprehensive signal testing suite.

    Tests whether a factor predicts future returns with statistical
    significance, using IC analysis, quantile analysis, and
    Fama-MacBeth regression.
    """

    def __init__(self, min_stocks_per_period: int = 10):
        self.min_stocks = min_stocks_per_period

    def information_coefficient(
        self,
        factor: FactorData,
        forward_returns: pd.DataFrame,
        horizon: int = 1,
    ) -> ICResult:
        """Compute Information Coefficient for a factor.

        IC = correlation(factor_value_t, forward_return_t+horizon)
        Computed cross-sectionally for each date, then averaged over time.

        Args:
            factor: Factor data (dates x stocks).
            forward_returns: Forward returns DataFrame (dates x stocks).
                             forward_returns[t] = return from t to t+horizon.
            horizon: Forward return horizon in days.

        Returns:
            ICResult with mean IC, IC_IR, significance, etc.
        """
        values = factor.values
        fwd = forward_returns.shift(-horizon)

        pearson_ics = []
        spearman_ics = []
        dates = []

        common_dates = values.index.intersection(fwd.index)
        for date in common_dates:
            f_vals = values.loc[date].dropna()
            r_vals = fwd.loc[date].dropna()

            common = f_vals.index.intersection(r_vals.index)
            if len(common) < self.min_stocks:
                continue

            f = f_vals[common].values
            r = r_vals[common].values

            # Remove inf/nan
            mask = np.isfinite(f) & np.isfinite(r)
            if mask.sum() < self.min_stocks:
                continue

            f, r = f[mask], r[mask]

            # Pearson IC
            pearson_ic, _ = scipy_stats.pearsonr(f, r)
            pearson_ics.append(pearson_ic)

            # Spearman (rank) IC
            spearman_ic, _ = scipy_stats.spearmanr(f, r)
            spearman_ics.append(spearman_ic)

            dates.append(date)

        ic_series = pd.Series(pearson_ics, index=dates)

        if len(pearson_ics) == 0:
            return ICResult(
                factor_name=factor.name,
                horizon=horizon,
                pearson_ic_mean=0.0,
                pearson_ic_std=0.0,
                pearson_ic_ir=0.0,
                spearman_ic_mean=0.0,
                spearman_ic_ir=0.0,
                ic_series=ic_series,
                ic_t_stat=0.0,
                ic_p_value=1.0,
                positive_ic_pct=0.0,
                summary="Insufficient data for IC calculation.",
            )

        pearson_mean = float(np.mean(pearson_ics))
        pearson_std = float(np.std(pearson_ics, ddof=1))
        pearson_ir = pearson_mean / pearson_std if pearson_std > 0 else 0.0

        spearman_mean = float(np.mean(spearman_ics))
        spearman_std = float(np.std(spearman_ics, ddof=1))
        spearman_ir = spearman_mean / spearman_std if spearman_std > 0 else 0.0

        # t-test: is the mean IC significantly different from zero?
        n = len(pearson_ics)
        ic_t = pearson_mean / (pearson_std / np.sqrt(n)) if pearson_std > 0 else 0.0
        ic_p = float(2 * scipy_stats.t.sf(abs(ic_t), df=n - 1))

        positive_pct = float(np.mean(np.array(pearson_ics) > 0))

        # Quality assessment
        if pearson_ir > 0.5 and ic_p < 0.01:
            quality = "STRONG signal — IC consistent and significant"
        elif pearson_ir > 0.2 and ic_p < 0.05:
            quality = "MODERATE signal — usable with caution"
        elif pearson_ir > 0:
            quality = "WEAK signal — consider combining with other factors"
        else:
            quality = "NEGATIVE or ZERO signal — not usable"

        summary = (
            f"IC Analysis: {factor.name} (horizon={horizon}d)\n"
            f"  Pearson IC: {pearson_mean:.4f} (±{pearson_std:.4f}), "
            f"IR={pearson_ir:.3f}\n"
            f"  Spearman IC: {spearman_mean:.4f}, IR={spearman_ir:.3f}\n"
            f"  t-stat={ic_t:.3f}, p={ic_p:.4f}\n"
            f"  Positive IC: {positive_pct:.0%} of periods\n"
            f"  Assessment: {quality}"
        )

        return ICResult(
            factor_name=factor.name,
            horizon=horizon,
            pearson_ic_mean=pearson_mean,
            pearson_ic_std=pearson_std,
            pearson_ic_ir=pearson_ir,
            spearman_ic_mean=spearman_mean,
            spearman_ic_ir=spearman_ir,
            ic_series=ic_series,
            ic_t_stat=ic_t,
            ic_p_value=ic_p,
            positive_ic_pct=positive_pct,
            summary=summary,
        )

    def quantile_analysis(
        self,
        factor: FactorData,
        forward_returns: pd.DataFrame,
        horizon: int = 1,
        n_quantiles: int = 5,
    ) -> QuantileResult:
        """Analyze forward returns by factor quantile.

        Stocks are sorted into quantiles based on factor value at each date.
        Top quintile (Q5) returns vs bottom quintile (Q1) returns gives
        the "quantile spread" — a measure of the factor's economic value.

        Args:
            factor: Factor data.
            forward_returns: Forward returns.
            horizon: Return horizon.
            n_quantiles: Number of quantile buckets.

        Returns:
            QuantileResult with per-quantile returns and spread analysis.
        """
        values = factor.values
        fwd = forward_returns.shift(-horizon)

        quantile_returns: Dict[int, List[float]] = {q: [] for q in range(1, n_quantiles + 1)}
        spreads = []

        common_dates = values.index.intersection(fwd.index)
        for date in common_dates:
            f_vals = values.loc[date].dropna()
            r_vals = fwd.loc[date].dropna()
            common = f_vals.index.intersection(r_vals.index)

            if len(common) < n_quantiles * self.min_stocks:
                continue

            f = f_vals[common]
            r = r_vals[common]

            # Assign quantiles
            try:
                q_labels = pd.qcut(f, n_quantiles, labels=False, duplicates="drop") + 1
            except ValueError:
                continue

            for q in range(1, n_quantiles + 1):
                mask = q_labels == q
                if mask.sum() > 0:
                    quantile_returns[q].append(float(r[mask].mean()))

            # Spread: top - bottom
            if n_quantiles in quantile_returns and 1 in quantile_returns:
                if len(quantile_returns[n_quantiles]) > len(spreads) + 1:
                    continue
                top_ret = r[q_labels == n_quantiles].mean()
                bottom_ret = r[q_labels == 1].mean()
                spreads.append(top_ret - bottom_ret)

        # Aggregate
        q_ret_mean = {
            q: float(np.mean(rets)) if rets else 0.0
            for q, rets in quantile_returns.items()
        }

        spread = q_ret_mean.get(n_quantiles, 0.0) - q_ret_mean.get(1, 0.0)
        spread_t = (
            float(np.mean(spreads) / (np.std(spreads, ddof=1) / np.sqrt(len(spreads))))
            if len(spreads) > 1 and np.std(spreads) > 0
            else 0.0
        )

        # Monotonicity: should increase from Q1 to Q5
        q_order = sorted(q_ret_mean.keys())
        q_rets_ordered = [q_ret_mean[q] for q in q_order]
        if len(q_rets_ordered) >= 3 and np.std(q_rets_ordered) > 0:
            mono_corr, _ = scipy_stats.spearmanr(range(len(q_rets_ordered)), q_rets_ordered)
            if np.isnan(mono_corr):
                mono_corr = 0.0
        else:
            mono_corr = 0.0

        # Quantile Sharpes (approximate)
        q_sharpes = {}
        for q in q_order:
            rets = quantile_returns[q]
            if rets and len(rets) > 1:
                q_sharpes[q] = float(np.mean(rets) / np.std(rets)) if np.std(rets) > 0 else 0.0

        summary = (
            f"Quantile Analysis: {factor.name} (horizon={horizon}d, {n_quantiles}Q)\n"
            + "\n".join(
                f"  Q{q}: {q_ret_mean.get(q, 0.0)*100:.3f}% "
                f"(Sharpe={q_sharpes.get(q, 0.0):.2f})"
                for q in q_order
            )
            + f"\n  Long-Short Spread: {spread*100:.3f}% (t={spread_t:.2f})"
            + f"\n  Monotonicity: {mono_corr:.2f}"
        )

        return QuantileResult(
            factor_name=factor.name,
            horizon=horizon,
            n_quantiles=n_quantiles,
            quantile_returns=q_ret_mean,
            spread=spread,
            spread_t_stat=spread_t,
            monotonicity=float(mono_corr),
            quantile_sharpes=q_sharpes,
            summary=summary,
        )

    def fama_macbeth(
        self,
        factors: List[FactorData],
        returns: pd.DataFrame,
        horizon: int = 1,
    ) -> FMResult:
        """Fama-MacBeth two-pass regression.

        Stage 1 (time-series): For each stock, regress returns on factors
                               to get factor betas (exposures).
        Stage 2 (cross-sectional): For each date, regress returns on betas
                                    to get factor risk premia.

        This isolates whether a factor earns a risk premium AFTER
        controlling for other factors.

        Args:
            factors: List of factors to test.
            returns: Returns DataFrame (dates x stocks).
            horizon: Return horizon.

        Returns:
            FMResult with coefficients (risk premia) and t-statistics.
        """
        fwd_ret = returns.shift(-horizon)

        # Simplified single-pass Fama-MacBeth:
        # Cross-sectional regression of returns on factor values at each date
        coeffs: Dict[str, List[float]] = {f.name: [] for f in factors}
        r_squareds = []

        common_dates = fwd_ret.index
        for f in factors:
            common_dates = common_dates.intersection(f.values.index)

        for date in common_dates:
            r_vals = fwd_ret.loc[date].dropna()
            stocks = r_vals.index.tolist()

            # Gather factor values for this date
            X_cols = []
            valid_factors = []
            for f in factors:
                f_vals = f.values.loc[date].reindex(stocks)
                if f_vals.notna().sum() >= self.min_stocks:
                    X_cols.append(f_vals.values)
                    valid_factors.append(f.name)

            if len(X_cols) < 1 or len(stocks) < self.min_stocks:
                continue

            X = np.column_stack(X_cols)
            X = np.column_stack([np.ones(len(stocks)), X])  # Intercept
            y = r_vals.values

            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if mask.sum() < self.min_stocks:
                continue

            X, y = X[mask], y[mask]

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                for i, name in enumerate(valid_factors):
                    coeffs[name].append(beta[i + 1])  # Skip intercept

                # R-squared
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r_squareds.append(r2)
            except np.linalg.LinAlgError:
                continue

        # Aggregate
        result_coeffs = {}
        result_t_stats = {}
        result_p_vals = {}
        for name in [f.name for f in factors]:
            c = coeffs.get(name, [])
            if len(c) > 1:
                mean_c = float(np.mean(c))
                std_c = float(np.std(c, ddof=1))
                t_c = mean_c / (std_c / np.sqrt(len(c))) if std_c > 0 else 0.0
                p_c = float(2 * scipy_stats.t.sf(abs(t_c), df=len(c) - 1))
            else:
                mean_c, t_c, p_c = 0.0, 0.0, 1.0
            result_coeffs[name] = mean_c
            result_t_stats[name] = t_c
            result_p_vals[name] = p_c

        avg_r2 = float(np.mean(r_squareds)) if r_squareds else 0.0

        # Summary
        lines = [f"Fama-MacBeth: {len([f.name for f in factors])} factors, horizon={horizon}d"]
        for name in result_coeffs:
            sig = "***" if result_p_vals[name] < 0.01 else ("**" if result_p_vals[name] < 0.05 else ("*" if result_p_vals[name] < 0.1 else ""))
            lines.append(
                f"  {name}: {result_coeffs[name]*10000:.2f} bp/day "
                f"(t={result_t_stats[name]:.2f}) {sig}"
            )
        lines.append(f"  Avg R²: {avg_r2:.4f}")

        return FMResult(
            factor_names=[f.name for f in factors],
            horizon=horizon,
            coefficients=result_coeffs,
            t_statistics=result_t_stats,
            p_values=result_p_vals,
            n_periods=len(r_squareds),
            avg_r_squared=avg_r2,
            summary="\n".join(lines),
        )
