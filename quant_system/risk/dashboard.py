"""
Risk dashboard — one-page risk summary.

Generates a concise daily risk report covering all dimensions:
VaR, stress tests, correlations, exposure limits, and drawdown.

Two Sigma approach: "If you can't read your risk dashboard in 30 seconds,
you won't read it at all. Make it short, make it visual, make it count."
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.types import RiskReport
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RiskDashboard:
    """Daily risk dashboard generator."""

    def generate(
        self,
        risk_report: RiskReport,
        var_results: Optional[Dict] = None,
        stress_results: Optional[List] = None,
        correlation_report: Optional[object] = None,
        limits_check: Optional[object] = None,
        portfolio_value: float = 1_000_000,
        format: str = "text",
    ) -> str:
        """Generate a complete risk dashboard.

        Args:
            risk_report: Core risk report.
            var_results: VaR calculation results.
            stress_results: Stress test results.
            correlation_report: Correlation monitor report.
            limits_check: Exposure limits check result.
            portfolio_value: Current portfolio value.
            format: "text" | "html".

        Returns:
            Formatted risk dashboard string.
        """
        if format == "html":
            return self._generate_html(risk_report, var_results, stress_results,
                                      correlation_report, limits_check, portfolio_value)
        return self._generate_text(risk_report, var_results, stress_results,
                                   correlation_report, limits_check, portfolio_value)

    def _generate_text(
        self,
        risk_report: RiskReport,
        var_results: Optional[Dict],
        stress_results: Optional[List],
        correlation_report,
        limits_check,
        portfolio_value: float,
    ) -> str:
        """Generate text dashboard."""
        lines = [
            "=" * 60,
            f"  RISK DASHBOARD — {risk_report.date.strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            f"  Portfolio Value: ¥{portfolio_value:,.0f}",
            f"  VaR (95% 1d):    {risk_report.var_95:.2%}  (¥{portfolio_value * risk_report.var_95:,.0f})",
            f"  CVaR (95% 1d):   {risk_report.cvar_95:.2%}  (¥{portfolio_value * risk_report.cvar_95:,.0f})",
            f"  Max Drawdown:    {risk_report.max_drawdown:.2%}",
            f"  Volatility (ann):{risk_report.volatility_annual:.2%}",
            f"  Sharpe Ratio:    {risk_report.sharpe_ratio:.2f}",
            "",
        ]

        # Correlation status
        if correlation_report:
            alarm_icon = "⚠️" if correlation_report.correlation_alarm else "✅"
            lines.append(f"  Correlation: {alarm_icon} {correlation_report.correlation_regime.upper()}")
            lines.append(f"    Avg: {correlation_report.avg_correlation:.3f} | Max: {correlation_report.max_correlation:.3f}")
            if correlation_report.top_correlated:
                top = correlation_report.top_correlated[0]
                lines.append(f"    Most correlated: {top[0]} ↔ {top[1]} ({top[2]:.3f})")
            lines.append("")

        # Stress test highlights
        if stress_results:
            lines.append("  Stress Test (worst 3):")
            for r in stress_results[:3]:
                icon = "⚠️" if r.exceeds_var else "  "
                lines.append(f"    {icon} {r.scenario:<25} {r.portfolio_pnl_pct:>7.1%}")
            lines.append("")

        # Limits
        if limits_check:
            status = "✅ PASS" if limits_check.passed else "❌ FAIL"
            lines.append(f"  Limits: {status}")
            for v in limits_check.violations[:5]:
                lines.append(f"    ❌ {v.get('message', str(v))}")
            for w in limits_check.warnings[:5]:
                lines.append(f"    ⚠️  {w.get('message', str(w))}")
            lines.append("")

        # Alerts summary
        alerts = []
        if risk_report.correlation_alarm:
            alerts.append("⚠️ Correlation alarm — diversification breaking down")
        if abs(risk_report.max_drawdown) > 0.15:
            alerts.append("⚠️ Drawdown > 15% — review position sizes")
        if risk_report.var_95 > 0.03:
            alerts.append("⚠️ VaR > 3% — portfolio risk elevated")
        if risk_report.sharpe_ratio < 0:
            alerts.append("⚠️ Negative Sharpe — strategy underperforming risk-free")

        if alerts:
            lines.append("  ALERTS:")
            for a in alerts:
                lines.append(f"    {a}")
        else:
            lines.append("  ✅ No active alerts.")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_html(
        self,
        risk_report: RiskReport,
        var_results: Optional[Dict],
        stress_results: Optional[List],
        correlation_report,
        limits_check,
        portfolio_value: float,
    ) -> str:
        """Generate HTML dashboard."""
        lines = [
            '<div style="font-family: monospace; max-width: 700px;">',
            f'<h2>⚠️ Risk Dashboard — {risk_report.date.strftime("%Y-%m-%d")}</h2>',
            '<table style="border-collapse: collapse; width: 100%;">',
            f'<tr><td>Portfolio Value</td><td>¥{portfolio_value:,.0f}</td></tr>',
            f'<tr><td>VaR (95% 1d)</td><td style="color:red">{risk_report.var_95:.2%}</td></tr>',
            f'<tr><td>Max Drawdown</td><td>{risk_report.max_drawdown:.2%}</td></tr>',
            f'<tr><td>Volatility (ann)</td><td>{risk_report.volatility_annual:.2%}</td></tr>',
            f'<tr><td>Sharpe Ratio</td><td>{risk_report.sharpe_ratio:.2f}</td></tr>',
            '</table>',
        ]

        if risk_report.summary:
            lines.append(f"<p>{risk_report.summary}</p>")

        lines.append("</div>")
        return "\n".join(lines)
