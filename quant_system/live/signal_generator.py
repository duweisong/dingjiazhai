"""
Weekly signal generator.

Runs the complete quantitative pipeline to produce weekly trading
signals: data → factors → alpha testing → factor composition →
signal generation → portfolio optimization → risk report.

This is the "assembly line" that connects all 6 roles.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..utils.types import (
    StockPool, MultiStockData, FactorData, Signal,
    PortfolioWeights, RiskReport, WeeklySignalReport,
)
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


class WeeklySignalGenerator:
    """Generates weekly trading signals using the full quant pipeline."""

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()

    def generate(
        self,
        data: MultiStockData,
        pool: StockPool,
        strategy_spec=None,  # StrategySpec
        date: Optional[pd.Timestamp] = None,
    ) -> WeeklySignalReport:
        """Generate weekly signals.

        Full pipeline:
        1. Load latest data
        2. Build factors
        3. Generate composite signals
        4. Optimize portfolio weights
        5. Generate risk report
        6. Package into WeeklySignalReport

        Args:
            data: Multi-stock price and indicator data.
            pool: Stock pool metadata.
            strategy_spec: Strategy specification.
            date: Reference date (default: last available).

        Returns:
            WeeklySignalReport ready for PushPlus delivery.
        """
        if date is None:
            date = data.dates[-1] if len(data.dates) > 0 else pd.Timestamp.now()

        # Determine week boundaries
        week_start = date - pd.Timedelta(days=date.dayofweek)
        week_end = week_start + pd.Timedelta(days=4)

        logger.info(f"Generating signals for week {week_start.date()} ~ {week_end.date()}")

        # Step 1: Build factors
        from ..factor.factor_definitions import FactorBuilder
        builder = FactorBuilder()
        factors = builder.build_all(data)
        logger.info(f"Built {len(factors)} factors")

        # Step 2: Generate signals from strategy spec
        signals = []
        if strategy_spec:
            from ..strategy.signal_pipeline import SignalPipeline
            pipeline = SignalPipeline(strategy_spec)
            result = pipeline.generate(factors, data, date=date)
            signals = result.signals
        else:
            # Default: simple momentum-based signals
            signals = self._default_signals(factors, data, date)

        # Step 3: Build portfolio weights
        weights = self._build_portfolio(signals, data, date)

        # Step 4: Risk report
        risk = self._build_risk_report(weights, data)

        # Step 5: Commentary
        commentary = self._generate_commentary(signals, weights, risk)

        return WeeklySignalReport(
            generated_at=pd.Timestamp.now(),
            week_start=week_start,
            week_end=week_end,
            portfolio=weights,
            buy_signals=[s for s in signals if s.signal_type == "BUY"],
            sell_signals=[s for s in signals if s.signal_type == "SELL"],
            risk_report=risk,
            commentary=commentary,
        )

    def _default_signals(
        self, factors: List[FactorData], data: MultiStockData, date: pd.Timestamp
    ) -> List[Signal]:
        """Generate default momentum-based signals when no strategy spec is provided."""
        momentum_factors = [f for f in factors if f.category == "momentum"]
        if not momentum_factors:
            return []

        # Average all momentum factors
        composite = None
        for f in momentum_factors:
            if date in f.values.index:
                row = f.values.loc[date].dropna() * f.direction
                z = (row - row.mean()) / (row.std() + 1e-10)
                if composite is None:
                    composite = z
                else:
                    composite = composite.add(z, fill_value=0)

        if composite is None:
            return []

        # Sort and pick top N
        ranked = composite.sort_values(ascending=False)
        top_n = self.config.live.top_n_stocks
        signals = []

        for i, (code, score) in enumerate(ranked.head(top_n).items()):
            signals.append(Signal(
                date=date,
                code=code,
                signal_type="BUY",
                strength=float(score / max(abs(ranked).max(), 1e-10)),
                confidence=float(1.0 - i / len(ranked)),
                reason=f"Default momentum composite: {score:.3f}",
            ))

        return signals

    def _build_portfolio(
        self, signals: List[Signal], data: MultiStockData, date: pd.Timestamp
    ) -> PortfolioWeights:
        """Build portfolio weights from signals."""
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        if not buy_signals:
            return PortfolioWeights(date=date, weights={}, cash_pct=1.0, method="equal")

        # Signal-strength-weighted allocation
        strengths = {s.code: s.strength for s in buy_signals}
        total = sum(strengths.values())

        weights = {
            code: s / total for code, s in strengths.items()
        } if total > 0 else {
            s.code: 1.0 / len(buy_signals) for s in buy_signals
        }

        return PortfolioWeights(
            date=date,
            weights=weights,
            cash_pct=0.0,
            method="signal_strength",
        )

    def _build_risk_report(
        self, weights: PortfolioWeights, data: MultiStockData
    ) -> Optional[RiskReport]:
        """Build a risk report for the current portfolio."""
        try:
            from ..risk.var_calculator import VaRCalculator
            from ..risk.correlation import CorrelationMonitor

            # Get portfolio returns
            return_matrix = pd.DataFrame({
                code: df["close"].pct_change()
                for code, df in data.prices.items()
                if code in weights.weights
            }).dropna()

            if return_matrix.empty:
                return None

            portfolio_returns = pd.Series(0.0, index=return_matrix.index)
            for code, w in weights.weights.items():
                if code in return_matrix.columns:
                    portfolio_returns += return_matrix[code].fillna(0) * w

            # VaR
            calc = VaRCalculator()
            var_results = calc.compute_all(portfolio_returns)
            best = calc.best_estimate(var_results)

            # Drawdown
            equity = (1 + portfolio_returns).cumprod()
            cummax = equity.expanding().max()
            dd = (equity - cummax) / cummax
            max_dd = float(dd.min())

            # Volatility
            ann_vol = float(portfolio_returns.std() * np.sqrt(244))
            sharpe = float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(244)) if portfolio_returns.std() > 0 else 0.0

            # Correlation
            monitor = CorrelationMonitor()
            corr_report = monitor.check(return_matrix)

            return RiskReport(
                date=pd.Timestamp.now(),
                var_95=best.var_pct,
                var_99=0.0,  # Not computed by default
                cvar_95=best.cvar_pct,
                max_drawdown=max_dd,
                volatility_annual=ann_vol,
                sharpe_ratio=sharpe,
                correlation_alarm=corr_report.correlation_alarm,
                summary=(
                    f"VaR(95%): {best.var_pct:.2%} | "
                    f"MaxDD: {max_dd:.2%} | "
                    f"Vol: {ann_vol:.1%} | "
                    f"Sharpe: {sharpe:.2f}"
                ),
            )
        except Exception as e:
            logger.warning(f"Risk report generation failed: {e}")
            return None

    def _generate_commentary(
        self,
        signals: List[Signal],
        weights: PortfolioWeights,
        risk: Optional[RiskReport],
    ) -> str:
        """Generate natural language commentary for the weekly report."""
        buy_count = len([s for s in signals if s.signal_type == "BUY"])
        sell_count = len([s for s in signals if s.signal_type == "SELL"])
        n_positions = len(weights.weights)

        lines = [
            f"本周策略信号已生成。",
            f"买入信号: {buy_count} 个，卖出信号: {sell_count} 个。",
            f"建议持仓: {n_positions} 只股票。",
        ]

        if risk:
            if risk.correlation_alarm:
                lines.append("⚠️ 相关性和风险水平处于高位，请关注组合集中度。")
            if abs(risk.max_drawdown) > 0.10:
                lines.append("⚠️ 当前回撤超过10%，建议检查风控参数。")

        lines.append(f"\n（系统自动生成于 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}）")
        return "\n".join(lines)
