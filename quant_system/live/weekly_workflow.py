"""
Weekly workflow — end-to-end orchestration.

Coordinates the complete weekly signal generation cycle:
1. Refresh stock pool (fetch latest index constituents)
2. Load/refresh market data
3. Build factors
4. Run signal pipeline
5. Optimize portfolio
6. Generate risk report
7. Push signals via PushPlus

This is the production entry point for the entire quant system.
"""

from typing import Dict, Optional
import pandas as pd

from ..utils.types import WeeklySignalReport
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


class WeeklyWorkflow:
    """End-to-end weekly workflow orchestrator.

    Runs the complete pipeline from data to push notification.
    Designed to be called by:
    - CLI: `python -m quant_system.main live --push`
    - Cron: Scheduled weekly (Thursday evening, execute Friday)
    - Manual: `quant_system live --dry-run`
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        push: bool = False,
        backtest_first: bool = False,
        strategy_name: str = "multi_factor",
    ) -> WeeklySignalReport:
        """Run the complete weekly workflow.

        Args:
            push: If True, push signals via PushPlus.
            backtest_first: If True, run backtest before generating signals.
            strategy_name: Name of strategy template to use.

        Returns:
            WeeklySignalReport.
        """
        logger.info("=" * 50)
        logger.info("Starting weekly workflow...")
        logger.info("=" * 50)

        # Step 1: Load strategy
        logger.info("Step 1/6: Loading strategy...")
        from ..strategy.spec_parser import StrategySpecParser
        parser = StrategySpecParser()
        spec = parser.load_template(strategy_name)
        logger.info(f"  Strategy: {spec.name} v{spec.version}")

        # Step 2: Load stock pool
        logger.info("Step 2/6: Loading stock pool...")
        from ..data.stock_pool import get_stock_pool
        pool = get_stock_pool(
            self.config.stock_pool_index,
            refresh=self.config.live.weekly_refresh_stock_pool,
        )
        logger.info(f"  Pool: {len(pool.codes)} stocks")

        # Step 3: Load market data
        logger.info("Step 3/6: Loading market data...")
        from ..data.multi_stock_loader import MultiStockLoader
        loader = MultiStockLoader(self.config)
        # For live mode, load recent 2 years for factor computation
        data = loader.load(
            pool,
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d"),
            end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            max_stocks=50,  # Top 50 by market cap for performance
        )
        logger.info(f"  Data: {data.n_stocks} stocks, {data.n_dates} trading days")

        # Step 4: (Optional) Run backtest
        if backtest_first:
            logger.info("Step 4a/6: Running backtest...")
            from ..backtest.multi_stock_engine import MultiStockBacktestEngine
            engine = MultiStockBacktestEngine(self.config)
            bt_result = engine.run(data)
            logger.info(f"  Backtest: Return={bt_result.total_return:.2%}, Sharpe={bt_result.sharpe_ratio:.2f}")

        # Step 5: Generate signals
        logger.info("Step 5/6: Generating signals...")
        from ..live.signal_generator import WeeklySignalGenerator
        generator = WeeklySignalGenerator(self.config)
        report = generator.generate(data, pool, spec)
        logger.info(f"  Buy signals: {len(report.buy_signals)}")
        logger.info(f"  Sell signals: {len(report.sell_signals)}")
        logger.info(f"  Positions: {len(report.portfolio.weights)}")

        # Step 6: Push via PushPlus
        if push:
            logger.info("Step 6/6: Pushing signals via PushPlus...")
            from ..live.pushplus import PushPlusConnector
            connector = PushPlusConnector()
            if connector.is_configured:
                success = connector.send_signal_report(report)
                if success:
                    logger.info("  PushPlus notification sent!")
                else:
                    logger.warning("  PushPlus push failed")
            else:
                logger.info("  PushPlus not configured. Run setup:")
                logger.info("    python -m quant_system.main live --init-config")
        else:
            logger.info("Step 6/6: Dry run — not pushing. Use --push to enable.")
            # Print summary to console
            self._print_summary(report)

        logger.info("=" * 50)
        logger.info("Weekly workflow complete.")
        logger.info("=" * 50)

        return report

    def _print_summary(self, report: WeeklySignalReport):
        """Print a text summary of the weekly report to console."""
        print(f"\n  Quant Weekly Report")
        print(f"  {'=' * 50}")
        print(f"  Period: {report.week_start.date()} ~ {report.week_end.date()}")
        print(f"  Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
        print()

        if report.buy_signals:
            print(f"  BUY Signals ({len(report.buy_signals)}):")
            for s in report.buy_signals[:10]:
                print(f"    {s.code:<12} strength={s.strength:.2f} confidence={s.confidence:.0%}")
        else:
            print("  No buy signals.")

        if report.sell_signals:
            print(f"\n  SELL Signals ({len(report.sell_signals)}):")
            for s in report.sell_signals:
                print(f"    {s.code}")

        if report.portfolio.weights:
            print(f"\n  Suggested Portfolio ({len(report.portfolio.weights)} stocks):")
            for code, w in sorted(report.portfolio.weights.items(), key=lambda x: -x[1]):
                print(f"    {code:<12} {w:.1%}")

        if report.risk_report:
            r = report.risk_report
            print(f"\n  Risk Overview:")
            print(f"    VaR(95%): {r.var_95:.2%}")
            print(f"    Max DD:   {r.max_drawdown:.2%}")
            print(f"    Vol:      {r.volatility_annual:.1%}")
            print(f"    Sharpe:   {r.sharpe_ratio:.2f}")

        if report.commentary:
            print(f"\n  Commentary: {report.commentary}")
        print()
