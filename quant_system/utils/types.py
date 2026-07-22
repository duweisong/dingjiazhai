"""
Shared data types for the quant system.

All data contracts between the 6 roles are defined here as immutable
or well-typed dataclasses, ensuring type safety across module boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


# ── Universe & Data ────────────────────────────────────────────

@dataclass
class StockPool:
    """Stock universe definition."""
    index: str                              # "hs300" | "csi500" | "custom"
    codes: List[str]                        # Stock codes (e.g. "000001.SZ")
    names: Dict[str, str]                   # code -> name
    industries: Dict[str, str]              # code -> Shenwan industry name
    market_caps: Dict[str, float]           # code -> market cap (in 100M CNY)
    as_of_date: pd.Timestamp


@dataclass
class MultiStockData:
    """Container for multi-stock OHLCV + computed indicators."""
    prices: Dict[str, pd.DataFrame]         # code -> DataFrame[date, o, h, l, c, volume]
    indicators: Dict[str, pd.DataFrame]     # code -> DataFrame with all indicators
    dates: pd.DatetimeIndex
    codes: List[str]

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def close_matrix(self) -> pd.DataFrame:
        """Return close prices as (dates x codes) matrix."""
        closes = {}
        for code, df in self.prices.items():
            closes[code] = df["close"]
        return pd.DataFrame(closes, index=self.dates)


# ── Factor & Signal ────────────────────────────────────────────

@dataclass(frozen=True)
class FactorData:
    """Single factor time series across stocks."""
    name: str                               # e.g. "momentum_20d"
    display_name: str                       # e.g. "20-Day Momentum"
    direction: int                          # +1 = long high-value, -1 = long low-value
    values: pd.DataFrame                    # index=date, columns=stock_codes
    category: str                           # "momentum" | "value" | "quality" | "size" | "volatility"

    def __post_init__(self):
        if self.values.index.name != "date":
            object.__setattr__(self, 'values', self.values.copy())


@dataclass(frozen=True)
class Signal:
    """Trading signal for a single stock at a point in time."""
    date: pd.Timestamp
    code: str
    signal_type: str                        # "BUY" | "SELL" | "HOLD"
    strength: float                         # -1.0 to +1.0 (normalized signal)
    confidence: float                       # 0.0 to 1.0
    factors: Dict[str, float] = field(default_factory=dict)  # factor_name -> contribution
    reason: str = ""


@dataclass
class SignalResult:
    """Aggregated signals for one rebalance date."""
    date: pd.Timestamp
    signals: List[Signal]
    buy_signals: List[Signal] = field(default_factory=list)
    sell_signals: List[Signal] = field(default_factory=list)

    def __post_init__(self):
        self.buy_signals = [s for s in self.signals if s.signal_type == "BUY"]
        self.sell_signals = [s for s in self.signals if s.signal_type == "SELL"]


# ── Portfolio & Risk ───────────────────────────────────────────

@dataclass
class PortfolioWeights:
    """Portfolio allocation snapshot."""
    date: pd.Timestamp
    weights: Dict[str, float]               # code -> weight (sums to ~1.0)
    cash_pct: float = 0.0                   # unallocated cash fraction
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    method: str = ""                        # "mvo" | "erc" | "hrp" | "bl" | "equal"


@dataclass
class RiskReport:
    """Risk analysis output for a single date."""
    date: pd.Timestamp
    var_95: float                           # 95% 1-day VaR
    var_99: float                           # 99% 1-day VaR
    cvar_95: float                          # 95% Conditional VaR (Expected Shortfall)
    max_drawdown: float                     # Current drawdown from peak
    portfolio_beta: float = 0.0             # Beta to benchmark
    volatility_annual: float = 0.0          # Annualized volatility
    sharpe_ratio: float = 0.0               # Rolling Sharpe
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    top_contributors: Dict[str, float] = field(default_factory=dict)  # top risk contributors
    correlation_alarm: bool = False         # Elevated cross-stock correlation
    stress_results: Dict[str, float] = field(default_factory=dict)  # scenario -> pnl_pct
    summary: str = ""                       # One-line risk summary


@dataclass
class WeeklySignalReport:
    """Weekly signal output for PushPlus delivery."""
    generated_at: pd.Timestamp
    week_start: pd.Timestamp
    week_end: pd.Timestamp
    portfolio: PortfolioWeights
    buy_signals: List[Signal] = field(default_factory=list)
    sell_signals: List[Signal] = field(default_factory=list)
    risk_report: Optional[RiskReport] = None
    commentary: str = ""

    def to_html(self) -> str:
        """Render as HTML for PushPlus."""
        lines = [
            f"<h2> 量化周报 {self.week_start.strftime('%Y-%m-%d')} ~ {self.week_end.strftime('%Y-%m-%d')}</h2>",
            f"<p>生成时间: {self.generated_at.strftime('%Y-%m-%d %H:%M')}</p>",
            "<h3> 建议持仓</h3><ul>",
        ]
        for code, w in sorted(self.portfolio.weights.items(), key=lambda x: -x[1]):
            if w > 0.01:
                lines.append(f"<li>{code}: {w*100:.1f}%</li>")
        lines.append("</ul>")

        if self.buy_signals:
            lines.append("<h3> 买入信号</h3><ul>")
            for s in self.buy_signals:
                lines.append(f"<li>{s.code} (强度:{s.strength:.2f}, 置信度:{s.confidence:.0%})</li>")
            lines.append("</ul>")

        if self.sell_signals:
            lines.append("<h3> 卖出信号</h3><ul>")
            for s in self.sell_signals:
                lines.append(f"<li>{s.code}</li>")
            lines.append("</ul>")

        if self.risk_report:
            r = self.risk_report
            lines.append(
                f"<h3>⚠️ 风险概览</h3>"
                f"<p>VaR(95%): {r.var_95:.2%} | "
                f"最大回撤: {r.max_drawdown:.2%} | "
                f"波动率(年): {r.volatility_annual:.2%}</p>"
            )

        if self.commentary:
            lines.append(f"<p>{self.commentary}</p>")

        return "\n".join(lines)


# ── Backtest Results ───────────────────────────────────────────

@dataclass
class MultiStockBacktestResult:
    """Aggregated backtest result for a multi-stock portfolio."""
    equity_curve: pd.Series                 # Portfolio-level equity over time
    daily_returns: pd.Series                # Portfolio daily returns
    per_stock_equity: Dict[str, pd.Series]  # Per-stock equity curves
    total_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_holding_days: float = 0.0
    turnover_annual: float = 0.0
    benchmark_return: float = 0.0           # e.g. CSI 300 return over same period
    excess_return: float = 0.0
    monte_carlo_results: Optional[Dict] = None  # MC simulation output
    significance_tests: Optional[Dict] = None    # Statistical test results
