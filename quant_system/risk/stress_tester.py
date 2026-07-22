"""
Stress testing — replay historical crash scenarios on current portfolio.

Pre-defined scenarios:
- 2008 Global Financial Crisis
- 2015 A-share Crash (Chinese stock market bubble burst)
- 2020 COVID-19 Crash
- Custom user-defined scenarios

Two Sigma approach: "VaR tells you about normal days.
Stress tests tell you about the days that matter."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StressScenario:
    """A historical or hypothetical stress scenario."""
    name: str
    description: str
    date_start: str
    date_end: str
    # Market moves (if hypothetical)
    equity_shock: float = 0.0      # e.g. -0.30 = -30%
    bond_shock: float = 0.0
    fx_shock: float = 0.0
    vol_shock: float = 0.0         # VIX multiplier


@dataclass
class StressResult:
    """Results of a single stress test."""
    scenario: str
    portfolio_pnl: float            # Absolute P&L
    portfolio_pnl_pct: float        # % P&L
    worst_day: float                # Worst single-day return
    max_drawdown: float             # Max drawdown during scenario
    days_to_recovery: int            # Days to recover (if applicable)
    exceeds_var: bool               # Whether loss exceeds VaR estimate


class StressTester:
    """Historical and hypothetical stress testing engine."""

    def __init__(self):
        self._scenarios = self._load_default_scenarios()

    def _load_default_scenarios(self) -> Dict[str, StressScenario]:
        """Load built-in stress scenarios."""
        return {
            "china_2015_crash": StressScenario(
                name="2015 A-Share Crash",
                description="Chinese stock market bubble burst, June-August 2015",
                date_start="2015-06-12",
                date_end="2015-08-26",
                equity_shock=-0.43,
            ),
            "covid_2020": StressScenario(
                name="COVID-19 Crash",
                description="Global pandemic market crash, Feb-March 2020",
                date_start="2020-02-20",
                date_end="2020-03-23",
                equity_shock=-0.34,
            ),
            "global_2008": StressScenario(
                name="2008 Financial Crisis",
                description="Global financial crisis, Sept-Nov 2008",
                date_start="2008-09-01",
                date_end="2008-11-20",
                equity_shock=-0.50,
            ),
            "trade_war_2018": StressScenario(
                name="2018 Trade War",
                description="US-China trade war escalation, 2018",
                date_start="2018-03-01",
                date_end="2018-12-31",
                equity_shock=-0.25,
            ),
            "black_monday": StressScenario(
                name="Black Monday Style",
                description="Hypothetical -20% single-day crash",
                date_start="",
                date_end="",
                equity_shock=-0.20,
                vol_shock=3.0,
            ),
            "rates_spike": StressScenario(
                name="Interest Rate Spike",
                description="Hypothetical 200bp rate hike shock",
                date_start="",
                date_end="",
                bond_shock=-0.10,
                equity_shock=-0.15,
            ),
        }

    def list_scenarios(self) -> Dict[str, str]:
        """List available stress scenarios."""
        return {name: s.description for name, s in self._scenarios.items()}

    def add_scenario(self, scenario: StressScenario):
        """Add a custom stress scenario."""
        self._scenarios[scenario.name] = scenario

    def run_historical(
        self,
        portfolio_returns: pd.Series,
        scenario_name: str,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Optional[StressResult]:
        """Replay a historical scenario on the current portfolio.

        Maps the scenario period's benchmark returns onto the
        portfolio's current risk characteristics.

        Args:
            portfolio_returns: Current portfolio daily returns.
            scenario_name: Name of the scenario to replay.
            benchmark_returns: Benchmark returns (e.g. CSI 300) for
                              mapping scenario returns to portfolio.

        Returns:
            StressResult with estimated P&L impact.
        """
        scenario = self._scenarios.get(scenario_name)
        if scenario is None:
            logger.warning(f"Unknown scenario: {scenario_name}")
            return None

        # If we have actual benchmark data for the scenario period
        if benchmark_returns is not None and scenario.date_start:
            try:
                bench_scenario = benchmark_returns[
                    (benchmark_returns.index >= scenario.date_start) &
                    (benchmark_returns.index <= scenario.date_end)
                ]
                if len(bench_scenario) > 0:
                    return self._map_benchmark_to_portfolio(
                        portfolio_returns, bench_scenario, scenario
                    )
            except Exception:
                pass

        # Fallback: use shock-based estimate
        return self._shock_estimate(portfolio_returns, scenario)

    def _shock_estimate(
        self, returns: pd.Series, scenario: StressScenario
    ) -> StressResult:
        """Estimate P&L from shock parameters."""
        portfolio_beta = 1.0  # Default: assume beta=1
        pnl_pct = scenario.equity_shock * portfolio_beta

        worst_day = scenario.equity_shock * portfolio_beta
        max_dd = abs(pnl_pct * 1.3)  # Crisis drawdowns typically 30% worse than peak-to-trough

        return StressResult(
            scenario=scenario.name,
            portfolio_pnl=pnl_pct,
            portfolio_pnl_pct=pnl_pct,
            worst_day=worst_day,
            max_drawdown=max_dd,
            days_to_recovery=0,
            exceeds_var=abs(pnl_pct) > 0.05,
        )

    def _map_benchmark_to_portfolio(
        self,
        portfolio_returns: pd.Series,
        benchmark_scenario: pd.Series,
        scenario: StressScenario,
    ) -> StressResult:
        """Map benchmark scenario returns to portfolio using beta."""
        # Estimate portfolio beta to benchmark
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_scenario.reindex(portfolio_returns.index),
        }).dropna()

        if len(aligned) > 30:
            beta = float(np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1] /
                        np.var(aligned["benchmark"]))
            beta = max(beta, 0.1)
        else:
            beta = 1.0

        # Apply beta to scenario benchmark returns
        scenario_returns = benchmark_scenario * beta
        cumulative = (1 + scenario_returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)
        worst_day = float(scenario_returns.min())

        # Max drawdown during scenario
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min())

        return StressResult(
            scenario=scenario.name,
            portfolio_pnl=total_return,
            portfolio_pnl_pct=total_return,
            worst_day=worst_day,
            max_drawdown=max_dd,
            days_to_recovery=0,
            exceeds_var=abs(total_return) > 0.05,
        )

    def run_all(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        var_estimate: float = 0.05,
    ) -> List[StressResult]:
        """Run all stress scenarios and return results.

        Returns:
            List of StressResult, sorted by worst P&L first.
        """
        results = []
        for name in self._scenarios:
            r = self.run_historical(portfolio_returns, name, benchmark_returns)
            if r is not None:
                r.exceeds_var = abs(r.portfolio_pnl_pct) > var_estimate
                results.append(r)

        return sorted(results, key=lambda r: r.portfolio_pnl_pct)

    def summary_table(self, results: List[StressResult]) -> str:
        """Generate a human-readable stress test summary table."""
        lines = ["Stress Test Results", "=" * 65]
        lines.append(f"  {'Scenario':<25} {'P&L':>8} {'Worst Day':>10} {'Breaches VaR':>12}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*12}")

        for r in results:
            breach = "⚠️ YES" if r.exceeds_var else "OK"
            lines.append(
                f"  {r.scenario:<25} {r.portfolio_pnl_pct:>7.1%} "
                f"{r.worst_day:>9.1%} {breach:>12}"
            )

        worst = results[0] if results else None
        if worst:
            lines.append(f"\n  Worst case: {worst.scenario} ({worst.portfolio_pnl_pct:.1%})")

        return "\n".join(lines)
