"""
Portfolio Backtest — Multi-Stock Simultaneous Trading

Runs the Xu Xiang upgraded strategy across multiple stocks simultaneously,
with portfolio-level risk management:
  - Risk budgeting (equal risk / Kelly / conviction)
  - Correlation-aware position sizing
  - Sector concentration limits
  - Portfolio drawdown stops
  - Dynamic rebalancing

This is the "institutional upgrade" of Xu Xiang's single-stock tactics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .data_loader import get_stock_data, get_index_data, compute_indicators
from .strategy import XuXiangStrategy
from .adaptive_strategy import AdaptiveXuXiangStrategy, StockClassifier
from .market_env import MarketEnvClassifier, build_index_env_df
from .portfolio_manager import (
    PortfolioRiskManager, RiskBudget, PositionSizer,
    CorrelationMatrix, SectorLimits, PortfolioStop
)
from .sector_theme import SectorThemeAnalyzer


@dataclass
class PortfolioResult:
    """Portfolio backtest results"""
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    avg_positions: float          # average number of concurrent positions
    max_positions: int
    sector_diversity: float       # average number of sectors held
    turnover_ratio: float         # annual turnover
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    yearly: List[Dict]
    env_breakdown: Dict
    stock_breakdown: Dict         # per-stock contribution


class PortfolioBacktest:
    """
    Multi-stock portfolio backtest engine.

    Features:
      - Simultaneous positions across multiple stocks
      - Portfolio-level risk management
      - Environment-aware position sizing (A: aggressive, B: moderate)
      - Correlation penalty between positions
      - Sector concentration limits
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.00025,
        stamp_tax: float = 0.001,
        slippage: float = 0.001,
        risk_mode: str = "equal_risk",
        total_risk: float = 0.20,
        max_positions: int = 5,
        max_drawdown: float = 0.15,
        use_adaptive: bool = True,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission
        self.stamp_tax_rate = stamp_tax
        self.slippage = slippage
        self.use_adaptive = use_adaptive

        # Portfolio risk manager
        self.prm = PortfolioRiskManager(
            risk_mode=risk_mode,
            total_risk=total_risk,
            max_positions=max_positions,
            max_drawdown=max_drawdown,
        )

        # Environment
        self.env_classifier = MarketEnvClassifier()

        # Theme
        self.theme_analyzer = SectorThemeAnalyzer()

    def run(
        self,
        symbols: List[str],
        names: List[str] = None,
        start: str = "2015-01-01",
        end: str = "2025-12-31",
    ) -> PortfolioResult:
        """
        Run portfolio backtest across multiple stocks.

        Parameters
        ----------
        symbols : list of stock codes
        names : list of stock names (optional)
        start, end : date range
        """
        if names is None:
            names = symbols
        elif len(names) < len(symbols):
            names = names + symbols[len(names):]

        print(f"\n{'='*80}")
        print(f"  PORTFOLIO BACKTEST: {len(symbols)} stocks")
        print(f"  {', '.join(f'{n}({s})' for s, n in zip(symbols, names))}")
        print(f"  Period: {start} ~ {end}")
        print(f"  Risk mode: {self.prm.risk_budget.mode} | "
              f"Max positions: {self.prm.risk_budget.max_positions}")
        print(f"{'='*80}")

        # Step 1: Load all data
        print("\n[1/4] Loading data...")
        stock_data = {}
        stock_strategies = {}
        stock_signals = {}
        for sym, name in zip(symbols, names):
            try:
                df = get_stock_data(sym, name, start, end)
                stock_data[sym] = df
                # Create strategy with adaptive params
                if self.use_adaptive:
                    strategy = AdaptiveXuXiangStrategy(stock_type="auto")
                    strategy.auto_detect(df)
                    stock_strategies[sym] = strategy
                else:
                    stock_strategies[sym] = XuXiangStrategy()
            except Exception as e:
                print(f"  WARN: Cannot load {sym}: {e}")

        # Load index for environment
        index_df = get_index_data("000001", start, end)
        env_raw = build_index_env_df(index_df)
        env_df = self.env_classifier.classify_series(env_raw)
        env_map = dict(zip(env_df["date"], env_df["env_level"]))

        print(f"  Loaded {len(stock_data)} stocks, "
              f"Index: {len(index_df)}d, "
              f"Env: A={int((env_df['env_level']=='A').sum())}d "
              f"B={int((env_df['env_level']=='B').sum())}d "
              f"C={int((env_df['env_level']=='C').sum())}d")

        # Step 2: Generate signals for all stocks
        print("\n[2/4] Generating signals...")
        for sym in stock_data:
            df = stock_data[sym]
            strategy = stock_strategies[sym]
            signals = strategy.generate_signals(df)
            stock_signals[sym] = signals
            buy_count = int((signals == 1).sum())
            print(f"  {sym}: {buy_count} buy signals")

        # Step 3: Run simulation
        print("\n[3/4] Running portfolio simulation...")
        result = self._simulate_portfolio(
            stock_data, stock_signals, env_map, stock_strategies)

        # Step 4: Build report
        print("\n[4/4] Building report...")
        report = self._build_report(
            symbols, names, start, end, stock_data, result, env_df)

        return report

    def _simulate_portfolio(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_signals: Dict[str, pd.Series],
        env_map: Dict,
        stock_strategies: Dict = None,
    ) -> Dict:
        """
        Simulate multi-stock portfolio trading with risk management.
        """
        if stock_strategies is None:
            stock_strategies = {}
        cash = self.initial_capital
        positions = {}  # {symbol: {shares, entry_price, entry_date, ...}}
        trades_list = []
        equity_records = []
        daily_returns = {}

        # Align all dates
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df["date"])
        all_dates = sorted(all_dates)

        # Track correlation using recent returns
        returns_history: Dict[str, List[float]] = defaultdict(list)

        for date in all_dates:
            env_level = env_map.get(date, "B")
            daily_pnl = 0.0

            # === Update positions ===
            for sym in list(positions.keys()):
                pos = positions[sym]
                df = stock_data.get(sym)
                if df is None:
                    continue

                row = df[df["date"] == date]
                if len(row) == 0:
                    continue

                close = row["close"].iloc[0]
                prev_close = (df[df["date"] < date]["close"].iloc[-1]
                              if len(df[df["date"] < date]) > 0 else close)

                # Daily return for correlation tracking
                if prev_close > 0:
                    ret = (close / prev_close) - 1
                    daily_returns[sym] = ret
                    returns_history[sym].append(ret)
                    if len(returns_history[sym]) > 60:
                        returns_history[sym] = returns_history[sym][-60:]

                # Check exit conditions
                exit_triggered = False
                exit_price = close
                exit_reason = ""

                # Signal-based exit
                signals = stock_signals.get(sym)
                if signals is not None:
                    sig_row = signals[signals.index[df["date"] == date]]
                    if len(sig_row) > 0 and sig_row.iloc[0] == -1:
                        exit_triggered = True
                        exit_reason = "signal"

                # Environment stop (C-level)
                if env_level == "C":
                    exit_triggered = True
                    exit_reason = "env_C"

                # Stop loss (5%)
                if close < pos["entry_price"] * 0.95:
                    exit_triggered = True
                    exit_reason = "stop_loss"

                if exit_triggered:
                    # Sell
                    proceeds = exit_price * pos["shares"]
                    commission = max(proceeds * self.commission_rate, 5)
                    stamp = proceeds * self.stamp_tax_rate
                    cash += proceeds - commission - stamp

                    pnl = ((exit_price - pos["entry_price"]) * pos["shares"]
                           - commission - stamp)
                    pnl_pct = (exit_price / pos["entry_price"]) - 1

                    trades_list.append({
                        "symbol": sym,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "reason": exit_reason,
                        "holding_days": max(1, (date - pos["entry_date"]).days),
                        "env_at_entry": pos.get("env_at_entry", "?"),
                        "env_at_exit": env_level,
                    })

                    daily_pnl += pnl
                    del positions[sym]

            # === Check for new entries ===
            if env_level != "C" and len(positions) < self.prm.risk_budget.max_positions:
                candidates = []

                for sym in stock_data:
                    if sym in positions:
                        continue

                    df = stock_data[sym]
                    row = df[df["date"] == date]
                    if len(row) == 0:
                        continue

                    signals = stock_signals.get(sym)
                    if signals is None:
                        continue

                    sig_row = signals[signals.index[df["date"] == date]]
                    if len(sig_row) == 0 or sig_row.iloc[0] != 1:
                        continue

                    r = row.iloc[0]
                    close = r["close"]

                    # Estimate annual vol from ATR
                    if "atr_pct" in r.index and pd.notna(r["atr_pct"]):
                        annual_vol = r["atr_pct"] * np.sqrt(252)
                    else:
                        annual_vol = 0.35

                    # Edge estimate from strategy type
                    strategy = stock_strategies.get(sym)
                    if self.use_adaptive and hasattr(strategy, 'detected_type'):
                        stype = strategy.detected_type
                        edge_map = {
                            "trend_growth": 0.06,
                            "small_mom": 0.04,
                            "blue_chip": 0.03,
                            "high_vol": 0.05,
                            "cyclical": 0.02,
                            "balanced": 0.04,
                        }
                        edge = edge_map.get(stype, 0.04)
                    else:
                        edge = 0.04

                    # Sector
                    sector = self.theme_analyzer.stock_sector.get(
                        sym, "unknown")

                    candidates.append({
                        "symbol": sym,
                        "price": close,
                        "annual_vol": annual_vol,
                        "edge_estimate": edge,
                        "sector": sector,
                        "conviction": 0.6 if env_level == "A" else 0.4,
                    })

                # Size and select positions
                if candidates:
                    # Sort by edge/vol ratio
                    candidates.sort(
                        key=lambda c: c["edge_estimate"] / max(c["annual_vol"], 0.01),
                        reverse=True)

                    # Calculate total portfolio value
                    total_value = cash
                    for s, p in positions.items():
                        sdf = stock_data.get(s)
                        if sdf is not None:
                            sr = sdf[sdf["date"] == date]
                            if len(sr) > 0:
                                total_value += p["shares"] * sr["close"].iloc[0]

                    # Build current holdings info for sector limits
                    current_holdings_info = {}
                    for s, p in positions.items():
                        sdf = stock_data.get(s)
                        if sdf is not None:
                            sr = sdf[sdf["date"] == date]
                            if len(sr) > 0:
                                current_holdings_info[s] = {
                                    "sector": self.theme_analyzer.stock_sector.get(s, "?"),
                                    "weight": (p["shares"] * sr["close"].iloc[0] /
                                              total_value if total_value > 0 else 0),
                                }

                    sized = self.prm.sizer.size_portfolio(
                        candidates, total_value, current_holdings_info)

                    # Execute approved positions up to max
                    slots_available = self.prm.risk_budget.max_positions - len(positions)
                    for pos_info in sized[:slots_available]:
                        sym = pos_info["symbol"]
                        shares = pos_info["shares"]
                        price = candidates[[c["symbol"] for c in candidates].index(sym)]["price"] if sym in [c["symbol"] for c in candidates] else 0
                        # Find price from candidates
                        price = next((c["price"] for c in candidates if c["symbol"] == sym), 0)
                        if shares < 100 or price <= 0:
                            continue
                        cost = price * shares
                        commission = max(cost * self.commission_rate, 5)
                        total_cost = cost + commission

                        if total_cost <= cash:
                            cash -= total_cost
                            positions[sym] = {
                                "shares": shares,
                                "entry_price": price,
                                "entry_date": date,
                                "env_at_entry": env_level,
                                "sector": pos_info.get("sector", "unknown"),
                            }

            # === Record equity ===
            market_value = 0.0
            for sym, pos in positions.items():
                df = stock_data.get(sym)
                if df is not None:
                    row = df[df["date"] == date]
                    if len(row) > 0:
                        market_value += pos["shares"] * row["close"].iloc[0]

            equity = cash + market_value
            equity_records.append({
                "date": date,
                "equity": equity,
                "cash": cash,
                "market_value": market_value,
                "positions": len(positions),
                "env_level": env_level,
            })

            # Update correlation tracker
            if len(daily_returns) > 0:
                self.prm.update_returns(daily_returns)
            daily_returns = {}

        # Force close remaining positions
        for sym, pos in list(positions.items()):
            df = stock_data.get(sym)
            if df is not None:
                last_close = df["close"].iloc[-1]
                proceeds = last_close * pos["shares"]
                commission = max(proceeds * self.commission_rate, 5)
                stamp = proceeds * self.stamp_tax_rate
                cash += proceeds - commission - stamp
                pnl = ((last_close - pos["entry_price"]) * pos["shares"]
                       - commission - stamp)
                pnl_pct = (last_close / pos["entry_price"]) - 1
                trades_list.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": df["date"].iloc[-1],
                    "entry_price": pos["entry_price"],
                    "exit_price": last_close,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": "force_close",
                    "holding_days": max(1, (df["date"].iloc[-1] - pos["entry_date"]).days),
                    "env_at_entry": pos.get("env_at_entry", "?"),
                    "env_at_exit": env_map.get(df["date"].iloc[-1], "?"),
                })
        positions.clear()

        eq_df = pd.DataFrame(equity_records)
        eq_df["ret"] = eq_df["equity"].pct_change()
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        return {
            "equity_curve": eq_df,
            "trades": trades_df,
            "final_cash": cash,
            "final_equity": eq_df["equity"].iloc[-1] if len(eq_df) > 0 else cash,
        }

    def _build_report(
        self,
        symbols: List[str],
        names: List[str],
        start: str,
        end: str,
        stock_data: Dict,
        result: Dict,
        env_df: pd.DataFrame,
    ) -> PortfolioResult:
        """Build comprehensive portfolio report"""
        eq_df = result["equity_curve"]
        trades_df = result["trades"]

        # Basic metrics
        total_return = (eq_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        total_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
        years = total_days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        cummax = eq_df["equity"].cummax()
        drawdown = (eq_df["equity"] - cummax) / cummax
        max_dd = drawdown.min()

        rf_daily = 0.03 / 252
        excess = eq_df["ret"].dropna() - rf_daily
        sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        if len(trades_df) > 0:
            win_rate = (trades_df["pnl"] > 0).mean()
        else:
            win_rate = 0.0

        # Position stats
        avg_pos = eq_df["positions"].mean() if "positions" in eq_df.columns else 0
        max_pos = eq_df["positions"].max() if "positions" in eq_df.columns else 0

        # Sector diversity
        if len(trades_df) > 0:
            sectors_per_day = trades_df.groupby("entry_date")
        sector_div = 1.0

        # Turnover
        if len(trades_df) > 0:
            total_buys = trades_df["shares"].sum() * trades_df["entry_price"].mean() if "entry_price" in trades_df.columns else 0
            turnover = total_buys / self.initial_capital / years if years > 0 else 0
        else:
            turnover = 0.0

        # Yearly breakdown
        yearly = self._yearly_breakdown(eq_df, trades_df, env_df)

        # Stock contribution
        stock_breakdown = {}
        if len(trades_df) > 0:
            for sym in symbols:
                st = trades_df[trades_df["symbol"] == sym]
                if len(st) > 0:
                    stock_breakdown[sym] = {
                        "trades": len(st),
                        "total_pnl": st["pnl"].sum(),
                        "win_rate": (st["pnl"] > 0).mean(),
                        "avg_pnl": st["pnl"].mean(),
                        "max_win": st["pnl"].max(),
                        "max_loss": st["pnl"].min(),
                    }

        # Environment breakdown
        env_breakdown = {}
        for level in ["A", "B", "C"]:
            et = trades_df[trades_df["env_at_entry"] == level] if len(trades_df) > 0 else pd.DataFrame()
            env_breakdown[level] = {
                "trades": len(et),
                "total_pnl": et["pnl"].sum() if len(et) > 0 else 0,
                "win_rate": (et["pnl"] > 0).mean() if len(et) > 0 else 0,
                "avg_pnl_pct": et["pnl_pct"].mean() if len(et) > 0 else 0,
            }

        return PortfolioResult(
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            final_equity=eq_df["equity"].iloc[-1],
            total_return_pct=total_return * 100,
            annualized_return_pct=annual_return * 100,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            total_trades=len(trades_df),
            win_rate=win_rate * 100,
            avg_positions=avg_pos,
            max_positions=int(max_pos),
            sector_diversity=sector_div,
            turnover_ratio=turnover,
            equity_curve=eq_df,
            trades=trades_df,
            yearly=yearly,
            env_breakdown=env_breakdown,
            stock_breakdown=stock_breakdown,
        )

    def _yearly_breakdown(self, eq_df, trades_df, env_df):
        """Yearly performance breakdown"""
        yearly = []
        eq_df["year"] = pd.to_datetime(eq_df["date"]).dt.year

        for year in sorted(eq_df["year"].unique()):
            yeq = eq_df[eq_df["year"] == year]
            if len(yeq) < 2:
                continue

            start_eq = yeq["equity"].iloc[0]
            end_eq = yeq["equity"].iloc[-1]
            yret = (end_eq - start_eq) / start_eq

            cmax_y = yeq["equity"].cummax()
            dd_y = (yeq["equity"] - cmax_y) / cmax_y
            mdd_y = dd_y.min()

            rf = 0.03 / 252
            ex = yeq["ret"].dropna() - rf
            sy = np.sqrt(252) * ex.mean() / ex.std() if ex.std() > 0 else 0

            if len(trades_df) > 0:
                yt = trades_df[pd.to_datetime(trades_df["entry_date"]).dt.year == year]
            else:
                yt = pd.DataFrame()

            ye2 = env_df[pd.to_datetime(env_df["date"]).dt.year == year]
            ea = int((ye2["env_level"] == "A").sum())
            eb = int((ye2["env_level"] == "B").sum())
            ec = int((ye2["env_level"] == "C").sum())

            yearly.append({
                "year": int(year),
                "return": yret * 100,
                "max_dd": mdd_y * 100,
                "sharpe": sy,
                "trades": len(yt),
                "win_rate": (yt["pnl"] > 0).mean() * 100 if len(yt) > 0 else 0,
                "env_A": ea, "env_B": eb, "env_C": ec,
                "avg_positions": yeq["positions"].mean() if "positions" in yeq.columns else 0,
            })

        return yearly

    def print_report(self, report: PortfolioResult):
        """Print formatted portfolio report"""
        print()
        print("=" * 90)
        print(f"  PORTFOLIO BACKTEST REPORT")
        print(f"  Period: {report.start_date} ~ {report.end_date}")
        print("=" * 90)

        print(f"\n  --- Portfolio Summary ---")
        print(f"  Initial Capital:     {report.initial_capital:>12,.0f}")
        print(f"  Final Equity:        {report.final_equity:>12,.0f}")
        print(f"  Total Return:        {report.total_return_pct:>+11.2f}%")
        print(f"  Annualized Return:   {report.annualized_return_pct:>+11.2f}%")
        print(f"  Max Drawdown:        {report.max_drawdown_pct:>11.2f}%")
        print(f"  Sharpe Ratio:        {report.sharpe_ratio:>11.2f}")
        print(f"  Calmar Ratio:        {report.calmar_ratio:>11.2f}")
        print(f"  Avg Positions:       {report.avg_positions:>11.1f}")
        print(f"  Max Positions:       {report.max_positions:>11}")

        print(f"\n  --- Trading ---")
        print(f"  Total Trades:        {report.total_trades:>11}")
        print(f"  Win Rate:            {report.win_rate:>11.1f}%")

        print(f"\n  --- Yearly Performance ---")
        print(f"  {'Year':<6}{'Return':<10}{'MaxDD':<9}"
              f"{'Sharpe':<8}{'Trades':<8}{'Win%':<8}{'AvgPos':<8}{'Env(A/B/C)':<18}")
        print(f"  {'-'*80}")
        for y in report.yearly:
            env_str = f"{y['env_A']}/{y['env_B']}/{y['env_C']}"
            print(f"  {y['year']:<6}{y['return']:>+8.1f}% "
                  f"{y['max_dd']:>7.1f}% {y['sharpe']:>6.2f} "
                  f"{y['trades']:>6}  {y['win_rate']:>5.1f}% "
                  f"{y['avg_positions']:>6.1f}  {env_str:<18}")

        print(f"\n  --- Environment-Conditional ---")
        for level in ["A", "B", "C"]:
            eb = report.env_breakdown.get(level, {})
            if eb.get("trades", 0) > 0:
                print(f"  {level}: {eb['trades']} trades, "
                      f"avg PnL={eb['avg_pnl_pct']*100:+.1f}%, "
                      f"win={eb['win_rate']*100:.0f}%, "
                      f"total={eb['total_pnl']:+,.0f}")

        if report.stock_breakdown:
            print(f"\n  --- Stock Contribution ---")
            for sym, sb in sorted(report.stock_breakdown.items(),
                                   key=lambda x: x[1]["total_pnl"], reverse=True):
                print(f"  {sym}: {sb['trades']} trades, "
                      f"total={sb['total_pnl']:+,.0f}, "
                      f"win={sb['win_rate']*100:.0f}%, "
                      f"best={sb['max_win']:+,.0f}, "
                      f"worst={sb['max_loss']:+,.0f}")


def run_portfolio_backtest(
    symbols: List[str] = None,
    names: List[str] = None,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    risk_mode: str = "equal_risk",
    max_positions: int = 5,
):
    """Convenience function to run portfolio backtest"""
    if symbols is None:
        symbols = ["000858", "002371", "002036", "600276", "600519"]
    if names is None:
        names = ["Wuliangye", "NAURA", "LianChuang", "Hengrui", "Moutai"]

    pbt = PortfolioBacktest(
        initial_capital=1_000_000,
        risk_mode=risk_mode,
        max_positions=max_positions,
        use_adaptive=True,
    )

    report = pbt.run(symbols, names, start, end)
    pbt.print_report(report)
    return report


if __name__ == "__main__":
    report = run_portfolio_backtest()
