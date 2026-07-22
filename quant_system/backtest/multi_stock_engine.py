"""
Multi-stock backtest engine.

Wraps the existing single-stock BacktestEngine (from backtest/engine.py)
for portfolio-level simulation. Each stock runs independently with its
allocated capital, and results are aggregated at the portfolio level.

Key features:
- Per-stock capital allocation (equal, signal-strength-weighted, or optimized)
- Portfolio-level equity curve aggregation
- Rebalancing at configurable frequency
- Full T+1, cost, and stop-loss simulation per stock (reused)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Reuse existing backtest engine
from backtest.engine import BacktestEngine, BacktestResult as SingleStockResult, Trade
from backtest.strategies.base import BaseStrategy, Signal

from ..utils.types import MultiStockData, MultiStockBacktestResult, PortfolioWeights
from ..utils.logger import get_logger
from ..config import GlobalConfig, get_config

logger = get_logger(__name__)


class SignalAdapterStrategy(BaseStrategy):
    """Adapts pre-computed float signals to the BaseStrategy interface.

    Converts continuous signals (-1 to +1) to discrete buy/sell/hold
    decisions using configurable thresholds.
    """

    def __init__(
        self,
        signal_series: pd.Series,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        super().__init__()
        self.signal_series = signal_series
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Convert float signals to 1/0/-1 based on thresholds."""
        result = pd.Series(0, index=df.index, dtype=int)
        aligned = self.signal_series.reindex(df.index, fill_value=0.0)

        result[aligned >= self.buy_threshold] = 1
        result[aligned <= self.sell_threshold] = -1
        return result


@dataclass
class PortfolioBook:
    """Portfolio-level bookkeeping across stocks."""

    initial_capital: float
    cash: float
    positions: Dict[str, Dict] = field(default_factory=dict)
    equity_history: List[Dict] = field(default_factory=list)

    def allocate_capital(
        self,
        codes: List[str],
        method: str = "equal_weight",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Allocate capital to each stock.

        Args:
            codes: List of stock codes.
            method: "equal_weight" | "signal_strength" | "custom".
            weights: Custom weights dict (for "custom" method).

        Returns:
            Dict mapping code -> allocated capital.
        """
        n = len(codes)
        if n == 0:
            return {}

        if method == "custom" and weights:
            alloc = {code: self.initial_capital * weights.get(code, 0.0) for code in codes}
        elif method == "signal_strength":
            # Weights passed via the weights parameter as signal strengths
            total = sum(weights.values()) if weights else n
            alloc = {
                code: self.initial_capital * (weights.get(code, 0.0) / max(total, 1e-10))
                for code in codes
            }
        else:
            # Equal weight
            alloc = {code: self.initial_capital / n for code in codes}

        self.cash = sum(alloc.values())
        return alloc

    def update_position(self, code: str, trade: Trade):
        """Record a trade for a stock."""
        if code not in self.positions:
            self.positions[code] = {"trades": [], "current_shares": 0}
        self.positions[code]["trades"].append(trade)

    def record_equity(self, date: pd.Timestamp, equity: float):
        """Record daily portfolio equity."""
        self.equity_history.append({"date": date, "equity": equity})

    def get_equity_curve(self) -> pd.Series:
        """Return portfolio equity curve as a Series."""
        if not self.equity_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df["equity"]


class MultiStockBacktestEngine:
    """Portfolio-level backtest engine.

    Creates one BacktestEngine per stock, runs simulations independently,
    and aggregates results at the portfolio level.
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        data: MultiStockData,
        signals: Optional[Dict[str, pd.Series]] = None,
        allocation_method: Optional[str] = None,
        allocation_weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ) -> MultiStockBacktestResult:
        """Run multi-stock backtest.

        Args:
            data: Multi-stock OHLCV + indicator data.
            signals: Pre-computed float signals per stock (optional).
                     If None, uses equal-weight allocation only.
            allocation_method: "equal_weight" | "signal_strength" | "custom".
            allocation_weights: Custom weights for allocation.
            buy_threshold: Float signal threshold for BUY.
            sell_threshold: Float signal threshold for SELL.

        Returns:
            MultiStockBacktestResult with aggregated metrics.
        """
        allocation_method = allocation_method or self.config.backtest.position_method

        # Allocate capital
        book = PortfolioBook(initial_capital=self.config.backtest.initial_capital)
        capital_alloc = book.allocate_capital(
            data.codes,
            method=allocation_method,
            weights=allocation_weights,
        )

        per_stock_results: Dict[str, SingleStockResult] = {}
        per_stock_equity: Dict[str, pd.Series] = {}
        all_trades: List[Trade] = []

        for code in data.codes:
            df = self._merge_price_and_indicators(data, code)
            if df is None or len(df) < 50:
                logger.warning(f"Skipping {code}: insufficient data")
                continue

            capital = capital_alloc.get(code, self.config.backtest.initial_capital / len(data.codes))

            # Build strategy adapter
            if signals and code in signals:
                strategy = SignalAdapterStrategy(
                    signals[code],
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                )
            else:
                # No signals → hold (zero-signal strategy)
                strategy = SignalAdapterStrategy(
                    pd.Series(0.0, index=df.index),
                )

            # Run single-stock backtest
            engine = BacktestEngine(
                initial_capital=capital,
                position_size_pct=0.95,  # Use most of allocated capital
                commission_rate=self.config.costs.commission_rate,
                stamp_tax_rate=self.config.costs.stamp_tax_rate,
                min_commission=self.config.costs.min_commission,
                slippage=self.config.costs.slippage_rate,
                t_plus_one=self.config.backtest.t_plus_one,
            )

            try:
                result = engine.run(
                    df=df,
                    strategy=strategy,
                    stop_loss_pct=self.config.risk_limits.stop_loss_pct,
                    take_profit_pct=self.config.risk_limits.take_profit_pct,
                    trailing_stop_pct=self.config.risk_limits.trailing_stop_pct,
                )
                per_stock_results[code] = result
                per_stock_equity[code] = result.equity_curve.set_index("date")["equity"]
                all_trades.extend(result.trades)

                # Tag trades with stock code
                for t in result.trades:
                    book.update_position(code, t)

            except Exception as e:
                logger.warning(f"Backtest failed for {code}: {e}")

        # Aggregate portfolio equity curve
        portfolio_equity = self._aggregate_equity(per_stock_equity, data.dates)
        portfolio_returns = portfolio_equity.pct_change().dropna()

        # Compile results
        result = self._compile_result(
            equity_curve=portfolio_equity,
            daily_returns=portfolio_returns,
            per_stock_equity=per_stock_equity,
            all_trades=all_trades,
            per_stock_results=per_stock_results,
            benchmark_return=self._compute_benchmark_return(data),
        )

        logger.info(
            f"Multi-stock backtest complete: "
            f"Return={result.total_return:.2%}, "
            f"Sharpe={result.sharpe_ratio:.2f}, "
            f"MaxDD={result.max_drawdown:.2%}"
        )
        return result

    def _merge_price_and_indicators(
        self, data: MultiStockData, code: str
    ) -> Optional[pd.DataFrame]:
        """Merge price and indicator DataFrames for a single stock."""
        price_df = data.prices.get(code)
        ind_df = data.indicators.get(code)

        if price_df is None or price_df.empty:
            return None

        if ind_df is not None and not ind_df.empty:
            # Avoid column name collisions
            overlap = set(price_df.columns) & set(ind_df.columns)
            if overlap:
                ind_df = ind_df.drop(columns=list(overlap), errors="ignore")
            df = pd.concat([price_df, ind_df], axis=1)
        else:
            df = price_df.copy()

        # Ensure 'date' column exists
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index

        return df

    def _aggregate_equity(
        self,
        per_stock_equity: Dict[str, pd.Series],
        dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Aggregate per-stock equity curves into portfolio equity."""
        if not per_stock_equity:
            return pd.Series(self.config.backtest.initial_capital, index=dates)

        # Build a DataFrame: dates x stocks
        eq_df = pd.DataFrame(per_stock_equity, index=dates)
        portfolio_equity = eq_df.sum(axis=1)
        return portfolio_equity.fillna(method="ffill").fillna(self.config.backtest.initial_capital)

    def _compute_benchmark_return(self, data: MultiStockData) -> float:
        """Compute benchmark (equal-weight buy-and-hold) return."""
        if not data.prices:
            return 0.0
        try:
            closes = {}
            for code, df in data.prices.items():
                if "close" in df.columns:
                    closes[code] = df["close"]
            if not closes:
                return 0.0
            bench_df = pd.DataFrame(closes)
            bench_ret = bench_df.mean(axis=1).pct_change().dropna()
            return float((1 + bench_ret).prod() - 1)
        except Exception:
            return 0.0

    def _compile_result(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        per_stock_equity: Dict[str, pd.Series],
        all_trades: List[Trade],
        per_stock_results: Dict[str, SingleStockResult],
        benchmark_return: float,
    ) -> MultiStockBacktestResult:
        """Compile aggregated performance metrics."""
        if len(daily_returns) == 0:
            return MultiStockBacktestResult(
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                per_stock_equity=per_stock_equity,
            )

        total_return = float((1 + daily_returns).prod() - 1)
        n_years = len(daily_returns) / 244  # Trading days per year
        ann_return = float((1 + total_return) ** (1 / max(n_years, 0.5)) - 1)
        ann_vol = float(daily_returns.std() * np.sqrt(244))
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(244)) if daily_returns.std() > 0 else 0.0

        # Drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = float(drawdown.min())

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Sortino
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() * np.sqrt(244) if len(downside) > 0 else 0.0
        sortino = ann_return / downside_std if downside_std > 0 else 0.0

        # Trade stats
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        win_rate = len(wins) / len(all_trades) if all_trades else 0.0

        # Holding days
        avg_hold = (
            np.mean([t.holding_days for t in all_trades if t.exit_date is not None])
            if all_trades else 0.0
        )

        # Turnover
        total_turnover = sum(abs(t.pnl) for t in all_trades) if all_trades else 0
        turnover_annual = total_turnover / (self.config.backtest.initial_capital * max(n_years, 0.5))

        excess_return = total_return - benchmark_return

        return MultiStockBacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            per_stock_equity=per_stock_equity,
            total_trades=len(all_trades),
            win_rate=win_rate,
            total_return=total_return,
            annual_return=ann_return,
            annual_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            avg_holding_days=avg_hold,
            turnover_annual=turnover_annual,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
        )
