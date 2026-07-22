"""
Portfolio rebalancer — transaction-cost-aware weight transitions.

Manages the transition from current portfolio weights to target
weights, considering:
- Transaction costs (commission + slippage)
- Minimum trade size
- Turnover constraints
- Tax considerations (placeholder)

Man Group approach: "The best rebalance is sometimes no rebalance.
If the cost of trading exceeds the expected benefit of the new
weights, stay put."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.types import PortfolioWeights
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


@dataclass
class RebalancePlan:
    """A rebalancing plan with trade list and cost estimates."""
    current_weights: PortfolioWeights
    target_weights: PortfolioWeights
    trades: List[Dict]              # List of {code, action, weight_change, cost}
    total_turnover: float           # Sum of absolute weight changes
    total_cost: float               # Estimated transaction cost
    is_worth_it: bool               # Whether the rebalance is cost-effective
    summary: str


class Rebalancer:
    """Cost-aware portfolio rebalancing."""

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()

    def plan(
        self,
        current: PortfolioWeights,
        target: PortfolioWeights,
        portfolio_value: float = 1_000_000,
        min_trade_size: float = 0.005,    # Don't trade if weight change < 0.5%
        max_turnover: float = 0.50,       # Max 50% one-way turnover
    ) -> RebalancePlan:
        """Generate a rebalancing plan.

        Args:
            current: Current portfolio weights.
            target: Target portfolio weights.
            portfolio_value: Total portfolio value in RMB.
            min_trade_size: Minimum weight change to execute a trade.
            max_turnover: Maximum allowed turnover.

        Returns:
            RebalancePlan with trade list and cost estimates.
        """
        all_codes = set(current.weights.keys()) | set(target.weights.keys())
        trades = []
        total_turnover = 0.0
        total_cost = 0.0

        for code in all_codes:
            cur_w = current.weights.get(code, 0.0)
            tgt_w = target.weights.get(code, 0.0)
            change = tgt_w - cur_w

            if abs(change) < min_trade_size:
                continue

            trade_value = portfolio_value * abs(change)
            # Estimated cost: commission (both sides) + slippage
            cost = trade_value * (
                self.config.costs.commission_rate * 2
                + self.config.costs.slippage_rate * 2
            )
            # A-share stamp tax if selling
            if change < 0:
                cost += trade_value * self.config.costs.stamp_tax_rate

            trades.append({
                "code": code,
                "action": "BUY" if change > 0 else "SELL",
                "weight_current": cur_w,
                "weight_target": tgt_w,
                "weight_change": change,
                "value": trade_value,
                "estimated_cost": cost,
            })

            total_turnover += abs(change)
            total_cost += cost

        # Cap turnover
        if total_turnover > max_turnover:
            scale = max_turnover / total_turnover
            for t in trades:
                t["weight_change"] *= scale
                t["value"] *= scale
                t["estimated_cost"] *= scale
            total_turnover = max_turnover
            total_cost *= scale
            logger.info(f"Turnover capped at {max_turnover:.0%} (was {total_turnover/scale:.0%})")

        # Cost-effectiveness check
        cost_pct = total_cost / portfolio_value if portfolio_value > 0 else 0.0
        expected_benefit_pct = 0.001  # Placeholder: should be estimated from alpha model
        is_worth_it = cost_pct < expected_benefit_pct or len(trades) <= 1

        summary = (
            f"Rebalance Plan: {len(trades)} trades, "
            f"turnover={total_turnover:.1%}, "
            f"cost=¥{total_cost:,.0f} ({cost_pct:.3%}) → "
            f"{'EXECUTE' if is_worth_it else 'SKIP (cost > benefit)'}"
        )

        return RebalancePlan(
            current_weights=current,
            target_weights=target,
            trades=trades,
            total_turnover=total_turnover,
            total_cost=total_cost,
            is_worth_it=is_worth_it,
            summary=summary,
        )

    def execute(
        self,
        plan: RebalancePlan,
        dry_run: bool = True,
    ) -> PortfolioWeights:
        """Execute (or simulate) a rebalancing plan.

        Args:
            plan: The rebalancing plan to execute.
            dry_run: If True, only simulate.

        Returns:
            New portfolio weights after execution.
        """
        if dry_run:
            logger.info(f"DRY RUN: {plan.summary}")
            return plan.target_weights

        # In a full system, this would:
        # 1. Send orders to broker API
        # 2. Wait for fills
        # 3. Update position tracking
        # 4. Handle partial fills and rejections
        logger.info(f"EXECUTING: {plan.summary}")
        return plan.target_weights

    def optimal_rebalance_frequency(
        self,
        returns: pd.Series,
        cost_per_trade_pct: float = 0.002,
        n_freqs: List[int] = None,
    ) -> Dict:
        """Estimate the optimal rebalancing frequency.

        Balances the benefit of staying close to target weights
        against the cost of rebalancing.

        Returns:
            Dict with optimal frequency and cost-benefit analysis.
        """
        if n_freqs is None:
            n_freqs = [1, 5, 10, 20, 40, 60]  # Days

        results = []
        for freq in n_freqs:
            # Rebalance every `freq` days
            n_rebalances = len(returns) / freq
            total_cost = n_rebalances * cost_per_trade_pct

            # Benefit: lower tracking error vs target
            tracking_error = returns.rolling(freq).std().mean()

            results.append({
                "frequency_days": freq,
                "n_rebalances_per_year": 244 / freq,
                "annual_cost_pct": total_cost / (len(returns) / 244),
                "tracking_error": tracking_error,
                "cost_per_te": total_cost / max(tracking_error, 1e-6),
            })

        df = pd.DataFrame(results)
        best = df.loc[df["cost_per_te"].idxmin()]

        return {
            "optimal_frequency_days": int(best["frequency_days"]),
            "optimal_cost_pct": float(best["annual_cost_pct"]),
            "analysis": df.to_dict("records"),
        }
