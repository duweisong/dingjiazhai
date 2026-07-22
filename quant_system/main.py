"""
CLI entry point for the quant system.

Six subcommands mapping to the six core roles:
  architect  — Strategy design and specification
  backtest   — Multi-stock backtesting
  risk       — Risk analysis and dashboard
  alpha      — Alpha signal research
  factor     — Factor model building
  optimize   — Portfolio optimization
  live       — Weekly signal generation + PushPlus push
"""

import sys
from pathlib import Path

# Ensure parent is on path for imports
_PARENT = Path(__file__).parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


def _get_click():
    """Lazy import click with helpful error message."""
    try:
        import click
        return click
    except ImportError:
        print("Error: 'click' is required. Install with: pip install click")
        sys.exit(1)


def main():
    """Main entry point — dispatches to subcommands."""
    click = _get_click()

    @click.group()
    @click.version_option(version="0.1.0", prog_name="quant_system")
    @click.pass_context
    def cli(ctx):
        """Quant System — Multi-Role Quantitative Trading Framework.

        Inspired by institutional quant pipelines at Goldman Sachs,
        Renaissance, Citadel, Two Sigma, AQR, and Man Group.
        """
        ctx.ensure_object(dict)

    # ── Subcommand: architect ──────────────────────────────
    @cli.command()
    @click.option(
        "--spec", "-s",
        type=click.Path(exists=True),
        help="Path to strategy YAML specification.",
    )
    @click.option(
        "--template", "-t",
        type=click.Choice(["momentum", "value_quality", "low_vol", "multi_factor"]),
        help="Use a built-in strategy template.",
    )
    @click.option("--validate-only", is_flag=True, help="Validate spec without running.")
    @click.pass_context
    def architect(ctx, spec, template, validate_only):
        """[Role 1] Strategy Architect — Design and validate trading strategies.

        Load a strategy specification from YAML or use a built-in template.
        Validates factor exposures, signal weights, constraints, and regime gates.
        """
        click.echo("  Role 1: Strategy Architect (Goldman Sachs style)")
        click.echo("=" * 60)
        try:
            from quant_system.strategy.spec_parser import StrategySpecParser
            parser = StrategySpecParser()

            if template:
                spec_obj = parser.load_template(template)
            elif spec:
                spec_obj = parser.load(spec)
            else:
                click.echo("Usage: architect --spec <file.yaml> OR --template <name>")
                click.echo("Templates: momentum, value_quality, low_vol, multi_factor")
                return

            errors = parser.validate(spec_obj)
            if errors:
                click.echo(f"\n  Validation errors ({len(errors)}):")
                for e in errors:
                    click.echo(f"    - {e}")
            else:
                click.echo(f"\n  Strategy: {spec_obj.name} v{spec_obj.version}")
                click.echo(f"  Universe: {spec_obj.universe_filter}")
                click.echo(f"  Factors: {list(spec_obj.factor_exposures.keys())}")
                click.echo(f"  Signals: {list(spec_obj.signal_weights.keys())}")
                click.echo(f"  Constraints: {list(spec_obj.constraints.keys())}")
                click.echo("\n  Validation passed!")

                if not validate_only:
                    click.echo("\n  Generated signal pipeline ready for backtesting.")
                    click.echo("  Next: quant_system backtest --spec <file.yaml>")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: backtest ───────────────────────────────
    @cli.command()
    @click.option("--spec", "-s", type=click.Path(exists=True), help="Strategy YAML spec.")
    @click.option("--strategy", "-t", type=click.Choice(["momentum", "value_quality", "low_vol", "multi_factor"]), help="Built-in strategy.")
    @click.option("--walk-forward", is_flag=True, help="Enforce walk-forward validation.")
    @click.option("--monte-carlo", is_flag=True, help="Run Monte Carlo simulations.")
    @click.option("--output", "-o", default="quant_system/output/reports", help="Output directory.")
    @click.pass_context
    def backtest(ctx, spec, strategy, walk_forward, monte_carlo, output):
        """[Role 2] Backtest Engine — Multi-stock portfolio backtesting.

        Runs a Renaissance-style rigorous backtest with:
        - Multi-stock portfolio simulation
        - Walk-forward validation (enforce with --walk-forward)
        - Monte Carlo simulation (--monte-carlo)
        - Statistical significance testing
        - Survivorship bias correction
        """
        click.echo("  Role 2: Backtest Engine (Renaissance style)")
        click.echo("=" * 60)
        try:
            from quant_system.config import get_config
            from quant_system.data.stock_pool import get_stock_pool
            from quant_system.data.multi_stock_loader import MultiStockLoader
            from quant_system.backtest.multi_stock_engine import MultiStockBacktestEngine

            cfg = get_config()
            click.echo(f"  Stock pool: {cfg.stock_pool_index}")
            click.echo(f"  Period: {cfg.backtest.start_date} ~ {cfg.backtest.end_date}")
            click.echo(f"  Capital: {cfg.backtest.initial_capital:,.0f} RMB")

            # Load stock pool
            pool = get_stock_pool(cfg.stock_pool_index)
            click.echo(f"  Universe: {len(pool.codes)} stocks")

            # Load data
            loader = MultiStockLoader(cfg)
            data = loader.load(pool, max_stocks=20)  # Default: top 20 for speed
            click.echo(f"  Data: {data.n_stocks} stocks, {data.n_dates} days")

            # Run backtest
            engine = MultiStockBacktestEngine(cfg)
            result = engine.run(data)

            click.echo(f"\n  Results:")
            click.echo(f"    Total Return:  {result.total_return:.2%}")
            click.echo(f"    Annual Return: {result.annual_return:.2%}")
            click.echo(f"    Sharpe Ratio:  {result.sharpe_ratio:.2f}")
            click.echo(f"    Max Drawdown:  {result.max_drawdown:.2%}")
            click.echo(f"    Win Rate:      {result.win_rate:.1%}")
            click.echo(f"    Total Trades:  {result.total_trades}")

            if walk_forward:
                click.echo("\n  Running walk-forward validation...")
                from quant_system.backtest.forward_walk import WalkForwardValidator
                wf = WalkForwardValidator(cfg)
                wf_result = wf.validate(data)
                click.echo(f"    Walk-forward Sharpe: {wf_result.sharpe:.2f}")
                click.echo(f"    Out-of-sample R²:    {wf_result.oos_r2:.3f}")

            if monte_carlo:
                click.echo("\n  Running Monte Carlo simulation...")
                from quant_system.backtest.monte_carlo import MonteCarloSimulator
                mc = MonteCarloSimulator(cfg)
                mc_result = mc.run(result)
                click.echo(f"    MC Sharpe (median): {mc_result['sharpe_median']:.2f}")
                click.echo(f"    MC Sharpe (5th pct): {mc_result['sharpe_p5']:.2f}")
                click.echo(f"    Prob(positive): {mc_result['prob_positive']:.1%}")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: risk ───────────────────────────────────
    @cli.command()
    @click.option("--portfolio", "-p", default="current", help="Portfolio state to analyze.")
    @click.option("--var", "var_method", type=click.Choice(["historical", "parametric", "cornish_fisher", "monte_carlo"]), default="historical", help="VaR method.")
    @click.option("--stress", is_flag=True, help="Run stress test scenarios.")
    @click.option("--dashboard", is_flag=True, help="Show risk dashboard.")
    @click.pass_context
    def risk(ctx, portfolio, var_method, stress, dashboard):
        """[Role 3] Risk Manager — Two Sigma-style risk analysis.

        Computes VaR, stress tests, correlation monitoring, and
        exposure limit checks.
        """
        click.echo("  Role 3: Risk Manager (Two Sigma style)")
        click.echo("=" * 60)
        try:
            from quant_system.risk.var_calculator import VaRCalculator
            from quant_system.risk.stress_tester import StressTester

            calc = VaRCalculator()
            click.echo(f"  VaR method: {var_method}")
            click.echo(f"  Confidence: 95% (1-day horizon)")
            click.echo(f"\n  [Module ready — supply portfolio returns to compute VaR]")
            click.echo(f"  Usage: quant_system risk --var monte_carlo --stress --dashboard")

            if stress:
                tester = StressTester()
                click.echo(f"\n  Stress scenarios available:")
                for name, desc in tester.list_scenarios().items():
                    click.echo(f"    - {name}: {desc}")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: alpha ──────────────────────────────────
    @cli.command()
    @click.option("--factor", "-f", "factor_name", help="Factor name to test.")
    @click.option("--ic-test", is_flag=True, help="Run Information Coefficient analysis.")
    @click.option("--decay", is_flag=True, help="Analyze signal decay over horizons.")
    @click.option("--combine", is_flag=True, help="Combine multiple signals.")
    @click.pass_context
    def alpha(ctx, factor_name, ic_test, decay, combine):
        """[Role 4] Alpha Signal Researcher — Citadel-style signal discovery.

        Systematic signal research pipeline:
        - Feature engineering (standardize, winsorize, neutralize)
        - IC analysis (Pearson + Spearman, quantile returns)
        - Signal decay analysis over multiple horizons
        - Signal combination methods
        - Regime detection
        """
        click.echo("  Role 4: Alpha Signal Researcher (Citadel style)")
        click.echo("=" * 60)
        try:
            from quant_system.alpha.signal_tester import SignalTester
            from quant_system.alpha.feature_engineering import FeatureEngineer

            if factor_name:
                click.echo(f"  Testing factor: {factor_name}")
                if ic_test:
                    click.echo(f"  IC analysis ready for {factor_name}")
                if decay:
                    click.echo(f"  Decay analysis ready for {factor_name}")
            elif combine:
                click.echo(f"  Signal combiner ready (equal / IC-weighted / eigenvector)")
            else:
                click.echo(f"  Usage: quant_system alpha --factor momentum_20d --ic-test --decay")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: factor ─────────────────────────────────
    @cli.command()
    @click.option("--compose", "-c", help="Comma-separated factor names to combine.")
    @click.option("--method", "-m", type=click.Choice(["erc", "ic_weighted", "equal"]), default="erc", help="Combination method.")
    @click.option("--attribute", is_flag=True, help="Run performance attribution.")
    @click.pass_context
    def factor(ctx, compose, method, attribute):
        """[Role 5] Factor Model Builder — AQR-style multi-factor construction.

        Builds and combines factor portfolios:
        - Factor definitions (momentum, value, quality, size, volatility)
        - Factor correlation analysis
        - Multi-factor combination (ERC, IC-weighted, equal)
        - Performance attribution (factor return vs residual)
        """
        click.echo("  Role 5: Factor Model Builder (AQR style)")
        click.echo("=" * 60)
        try:
            if compose:
                factors = [f.strip() for f in compose.split(",")]
                click.echo(f"  Factors: {factors}")
                click.echo(f"  Method: {method}")
                click.echo(f"\n  [Module ready — supply factor data to compose]")
            else:
                click.echo(f"  Available built-in factors:")
                click.echo(f"    momentum_20d, momentum_60d, momentum_120d")
                click.echo(f"    value_pe, value_pb, value_dy")
                click.echo(f"    quality_roe, quality_gross_margin, quality_debt_ratio")
                click.echo(f"    size_market_cap")
                click.echo(f"    volatility_20d, volatility_60d")
                click.echo(f"\n  Usage: quant_system factor --compose value_pe,momentum_60d,quality_roe")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: optimize ───────────────────────────────
    @cli.command()
    @click.option("--method", "-m", type=click.Choice(["mvo", "bl", "erc", "hrp"]), default="hrp", help="Optimization method.")
    @click.option("--returns", "-r", type=click.Path(exists=True), help="Path to returns matrix (Parquet).")
    @click.option("--constraints", "-c", type=click.Path(exists=True), help="Path to constraints YAML.")
    @click.pass_context
    def optimize(ctx, method, returns, constraints):
        """[Role 6] Portfolio Optimizer — Man Group-style capital allocation.

        Multi-method portfolio optimization:
        - Mean-Variance Optimization (MVO) with Ledoit-Wolf shrinkage
        - Black-Litterman with explicit views
        - Equal Risk Contribution (ERC)
        - Hierarchical Risk Parity (HRP)
        With automatic fallback chain: MVO → ERC → HRP → Equal Weight
        """
        click.echo("  Role 6: Portfolio Optimizer (Man Group style)")
        click.echo("=" * 60)
        try:
            click.echo(f"  Method: {method}")
            click.echo(f"  Fallback chain: MVO → ERC → HRP → Equal Weight")
            click.echo(f"\n  [Module ready — supply returns matrix to optimize]")
            click.echo(f"  Usage: quant_system optimize --method hrp --returns returns.parquet")

            methods_desc = {
                "mvo": "Mean-Variance Optimization (Ledoit-Wolf shrinkage)",
                "bl": "Black-Litterman (requires view specification)",
                "erc": "Equal Risk Contribution",
                "hrp": "Hierarchical Risk Parity (Lopez de Prado)",
            }
            click.echo(f"\n  About {method}: {methods_desc.get(method, '')}")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # ── Subcommand: live ───────────────────────────────────
    @cli.command()
    @click.option("--push", is_flag=True, help="Push weekly signals via PushPlus.")
    @click.option("--dry-run", is_flag=True, help="Generate signals without pushing.")
    @click.option("--backtest", "run_backtest", is_flag=True, help="Run backtest before signal generation.")
    @click.pass_context
    def live(ctx, push, dry_run, run_backtest):
        """Live signal generation + PushPlus push.

        Generates weekly trading signals and optionally pushes them
        via PushPlus (WeChat notification).
        """
        click.echo("  Live Signal Generator")
        click.echo("=" * 60)
        try:
            from quant_system.live.weekly_workflow import WeeklyWorkflow

            wf = WeeklyWorkflow()
            if dry_run:
                click.echo("  Mode: DRY RUN (no push)")
            elif push:
                click.echo("  Mode: LIVE (signals will be pushed)")

            report = wf.run(push=push and not dry_run, backtest_first=run_backtest)
            click.echo(f"\n  Week: {report.week_start.date()} ~ {report.week_end.date()}")
            click.echo(f"  Positions: {len(report.portfolio.weights)} stocks")
            click.echo(f"  Buy signals: {len(report.buy_signals)}")
            click.echo(f"  Sell signals: {len(report.sell_signals)}")
            if report.risk_report:
                click.echo(f"  VaR(95%): {report.risk_report.var_95:.2%}")

            if push and not dry_run:
                click.echo("\n  Signals pushed to PushPlus!")

        except ImportError as e:
            click.echo(f"  [NOT YET IMPLEMENTED] {e}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    return cli()


if __name__ == "__main__":
    main()
