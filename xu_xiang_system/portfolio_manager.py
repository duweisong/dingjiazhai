"""
Portfolio Risk Manager — Multi-Stock Position Sizing & Risk Budgeting

Elevates Xu Xiang's single-stock tactics to institutional portfolio management:

Core Capabilities:
  1. Risk Budgeting — allocate risk (not capital) across positions
  2. Correlation-Aware Sizing — reduce exposure to overlapping bets
  3. Kelly-Inspired Fractional Sizing — scale by edge estimate
  4. Portfolio-Level Stops — max drawdown, sector concentration limits
  5. Equal Risk Contribution — balance risk evenly across holdings
  6. Dynamic Rebalancing — adjust as volatility regimes shift

Architecture:
  PortfolioRiskManager
  ├── RiskBudget        — defines how much risk each position gets
  ├── PositionSizer     — translates risk budget into share counts
  ├── CorrelationMatrix — tracks pairwise correlations
  ├── SectorLimits      — enforces sector concentration caps
  └── PortfolioStop     — portfolio-level drawdown protection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================
# Risk Budget
# ============================================================
@dataclass
class RiskBudget:
    """
    Defines how risk is allocated across the portfolio.

    Three modes:
      - equal_risk: each position contributes equal risk (vol * size)
      - kelly: size proportional to estimated edge
      - conviction: manual override per position
    """
    mode: str = "equal_risk"        # "equal_risk" | "kelly" | "conviction"
    total_risk_budget: float = 0.20  # 20% annualized portfolio vol target
    max_single_risk: float = 0.05    # max 5% risk contribution per position
    max_positions: int = 5           # max concurrent positions
    max_sector_risk: float = 0.10    # max 10% risk per sector

    # Kelly parameters
    kelly_fraction: float = 0.25     # use 1/4 Kelly for safety
    min_win_rate: float = 0.40       # minimum required win rate
    min_profit_factor: float = 1.2   # minimum required profit factor

    def allocate(self, candidates: List[Dict]) -> Dict[str, float]:
        """
        Allocate risk budget to candidates.

        Each candidate dict should have:
          - symbol: str
          - volatility: float (annualized)
          - edge_estimate: float (0-1, estimated alpha)
          - sector: str
          - conviction: float (0-1, manual override)
        """
        if not candidates:
            return {}

        if self.mode == "equal_risk":
            return self._equal_risk_alloc(candidates)
        elif self.mode == "kelly":
            return self._kelly_alloc(candidates)
        elif self.mode == "conviction":
            return self._conviction_alloc(candidates)
        else:
            return self._equal_risk_alloc(candidates)

    def _equal_risk_alloc(self, candidates: List[Dict]) -> Dict[str, float]:
        """Equal risk contribution allocation"""
        n = min(len(candidates), self.max_positions)
        risk_per = min(
            self.total_risk_budget / n,
            self.max_single_risk
        )
        return {c["symbol"]: risk_per for c in candidates[:n]}

    def _kelly_alloc(self, candidates: List[Dict]) -> Dict[str, float]:
        """
        Kelly-inspired allocation:
          f = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
          scaled by kelly_fraction and capped at max_single_risk
        """
        allocation = {}
        for c in candidates[:self.max_positions]:
            edge = c.get("edge_estimate", 0.02)
            vol = c.get("volatility", 0.30)
            if vol <= 0:
                vol = 0.30

            # Simplified Kelly: edge / variance
            kelly_f = edge / (vol ** 2)
            kelly_f = max(0, kelly_f)

            # Apply fraction
            risk = kelly_f * self.kelly_fraction

            # Cap
            risk = min(risk, self.max_single_risk)
            risk = max(risk, 0.005)  # minimum 0.5% risk

            allocation[c["symbol"]] = risk

        # Scale to total risk budget if exceeded
        total = sum(allocation.values())
        if total > self.total_risk_budget and total > 0:
            scale = self.total_risk_budget / total
            allocation = {k: v * scale for k, v in allocation.items()}

        return allocation

    def _conviction_alloc(self, candidates: List[Dict]) -> Dict[str, float]:
        """Conviction-weighted allocation"""
        total_conv = sum(c.get("conviction", 0.5) for c in candidates[:self.max_positions])
        if total_conv <= 0:
            return {}

        allocation = {}
        for c in candidates[:self.max_positions]:
            conv = c.get("conviction", 0.5)
            risk = self.total_risk_budget * (conv / total_conv)
            risk = min(risk, self.max_single_risk)
            allocation[c["symbol"]] = risk

        return allocation


# ============================================================
# Correlation Matrix
# ============================================================
class CorrelationMatrix:
    """
    Tracks pairwise correlations between portfolio holdings.

    Used to:
      - Detect overcrowded bets (high correlation = same bet)
      - Reduce position sizes for highly correlated pairs
      - Flag correlation regime changes
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.returns_cache: Dict[str, pd.Series] = {}
        self.corr_matrix: Optional[pd.DataFrame] = None

    def update(self, returns: Dict[str, pd.Series]):
        """Update with latest return series"""
        self.returns_cache.update(returns)

        # Trim to lookback
        for sym in self.returns_cache:
            self.returns_cache[sym] = self.returns_cache[sym].iloc[-self.lookback:]

        # Compute correlation matrix
        symbols = list(self.returns_cache.keys())
        if len(symbols) >= 2:
            df = pd.DataFrame(self.returns_cache)
            self.corr_matrix = df.corr()

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get pairwise correlation"""
        if self.corr_matrix is None:
            return 0.0
        if sym1 in self.corr_matrix.index and sym2 in self.corr_matrix.columns:
            return self.corr_matrix.loc[sym1, sym2]
        return 0.0

    def get_cluster_penalty(self, symbol: str, holdings: List[str]) -> float:
        """
        Calculate correlation penalty for a new position.
        Penalty increases if the symbol is highly correlated with existing holdings.

        Returns 0-1 multiplier (1 = no penalty, 0.5 = 50% size reduction)
        """
        if not holdings:
            return 1.0

        correlations = []
        for h in holdings:
            corr = abs(self.get_correlation(symbol, h))
            correlations.append(corr)

        if not correlations:
            return 1.0

        avg_corr = np.mean(correlations)
        # Penalty formula: if avg correlation > 0.5, start reducing
        if avg_corr > 0.8:
            penalty = 0.4   # 60% reduction
        elif avg_corr > 0.6:
            penalty = 0.6   # 40% reduction
        elif avg_corr > 0.4:
            penalty = 0.8   # 20% reduction
        else:
            penalty = 1.0

        return penalty

    def is_overcrowded(self, holdings: List[str]) -> bool:
        """Check if portfolio is overcrowded (too many correlated bets)"""
        if len(holdings) < 3 or self.corr_matrix is None:
            return False

        subset = [h for h in holdings if h in self.corr_matrix.index]
        if len(subset) < 3:
            return False

        sub_corr = self.corr_matrix.loc[subset, subset]
        # Average pairwise correlation (excluding diagonal)
        n = len(subset)
        if n < 2:
            return False
        avg = (sub_corr.values.sum() - n) / (n * (n - 1))
        return avg > 0.7


# ============================================================
# Sector Limits
# ============================================================
@dataclass
class SectorLimits:
    """Enforces sector concentration limits"""
    max_sectors: int = 3                # max different sectors
    max_per_sector: float = 0.35        # max 35% capital per sector
    max_per_sector_positions: int = 2   # max 2 stocks per sector

    def check(self, holdings: Dict[str, Dict]) -> Tuple[bool, str]:
        """
        Check if adding a position would violate sector limits.

        holdings: {symbol: {sector: str, weight: float}}

        Returns (allowed, reason)
        """
        sectors = defaultdict(lambda: {"count": 0, "weight": 0.0})
        for sym, info in holdings.items():
            sec = info.get("sector", "unknown")
            sectors[sec]["count"] += 1
            sectors[sec]["weight"] += info.get("weight", 0)

        # Check distinct sectors
        if len(sectors) > self.max_sectors:
            return False, f"Too many sectors ({len(sectors)} > {self.max_sectors})"

        # Check per-sector
        for sec, info in sectors.items():
            if info["count"] > self.max_per_sector_positions:
                return False, f"Too many positions in {sec}"
            if info["weight"] > self.max_per_sector:
                return False, f"Sector {sec} overweight ({info['weight']:.0%})"

        return True, "OK"

    def get_available_sectors(self, current_sectors: List[str]) -> int:
        """How many more sectors can be added"""
        return max(0, self.max_sectors - len(set(current_sectors)))


# ============================================================
# Position Sizer
# ============================================================
class PositionSizer:
    """
    Translates risk budgets into actual share counts.

    Input:
      - Risk budget per symbol (e.g., 4% of portfolio risk)
      - Stock volatility
      - Portfolio capital
      - Correlation penalty

    Output:
      - Number of shares
      - Capital allocation
      - Expected risk contribution
    """

    def __init__(self, risk_budget: RiskBudget = None):
        self.risk_budget = risk_budget or RiskBudget()
        self.corr_matrix = CorrelationMatrix()
        self.sector_limits = SectorLimits()

    def size_position(
        self,
        symbol: str,
        price: float,
        annual_vol: float,
        capital: float,
        edge_estimate: float = 0.02,
        sector: str = "unknown",
        conviction: float = 0.5,
        current_holdings: Dict[str, Dict] = None,
    ) -> Dict:
        """
        Calculate position size for a single candidate.

        Returns dict with:
          shares, capital_alloc, weight_pct, risk_contrib,
          correlation_penalty, approved, reason
        """
        current_holdings = current_holdings or {}

        # Step 1: Check sector limits
        proposed = {
            **current_holdings,
            symbol: {"sector": sector, "weight": 0}
        }
        allowed, reason = self.sector_limits.check(proposed)

        # Step 2: Calculate risk budget for this position
        candidates = [{
            "symbol": symbol,
            "volatility": annual_vol,
            "edge_estimate": edge_estimate,
            "sector": sector,
            "conviction": conviction,
        }]
        budget = self.risk_budget.allocate(candidates)
        risk_alloc = budget.get(symbol, self.risk_budget.max_single_risk)

        # Step 3: Correlation penalty
        existing_symbols = list(current_holdings.keys())
        corr_penalty = self.corr_matrix.get_cluster_penalty(
            symbol, existing_symbols)

        # Step 4: Calculate position size
        # Risk contribution = weight * volatility
        # weight = risk_alloc / annual_vol
        base_weight = risk_alloc / annual_vol if annual_vol > 0 else 0.10
        adjusted_weight = base_weight * corr_penalty

        # Cap at max single position
        adjusted_weight = min(adjusted_weight, 0.30)

        # Minimum position size
        if adjusted_weight < 0.01:
            adjusted_weight = 0.0

        capital_alloc = capital * adjusted_weight
        shares = int(capital_alloc / price / 100) * 100  # round to lots

        # Recalculate actual weight
        actual_capital = shares * price
        actual_weight = actual_capital / capital if capital > 0 else 0
        actual_risk_contrib = actual_weight * annual_vol

        return {
            "symbol": symbol,
            "shares": shares,
            "capital_alloc": actual_capital,
            "weight_pct": actual_weight * 100,
            "risk_contrib_pct": actual_risk_contrib * 100,
            "correlation_penalty": corr_penalty,
            "annual_vol": annual_vol,
            "approved": shares >= 100 and allowed,
            "reason": reason if not allowed else "OK",
        }

    def size_portfolio(
        self,
        candidates: List[Dict],
        capital: float,
        current_holdings: Dict[str, Dict] = None,
    ) -> List[Dict]:
        """
        Size an entire portfolio of candidates.

        candidates: [{symbol, price, annual_vol, edge_estimate, sector, conviction}]
        """
        current_holdings = current_holdings or {}
        results = []

        # Sort by conviction * edge / vol (best risk-adjusted first)
        scored = []
        for c in candidates:
            score = (c.get("conviction", 0.5) * c.get("edge_estimate", 0.02) /
                     max(c.get("annual_vol", 0.30), 0.01))
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Allocate risk budget
        risk_alloc = self.risk_budget.allocate([
            {"symbol": c["symbol"], "volatility": c.get("annual_vol", 0.30),
             "edge_estimate": c.get("edge_estimate", 0.02),
             "sector": c.get("sector", "unknown"),
             "conviction": c.get("conviction", 0.5)}
            for _, c in scored
        ])

        remaining_capital = capital
        for _, c in scored[:self.risk_budget.max_positions]:
            if remaining_capital <= 0:
                break

            result = self.size_position(
                symbol=c["symbol"],
                price=c["price"],
                annual_vol=c.get("annual_vol", 0.30),
                capital=capital,
                edge_estimate=c.get("edge_estimate", 0.02),
                sector=c.get("sector", "unknown"),
                conviction=c.get("conviction", 0.5),
                current_holdings=current_holdings,
            )

            if result["approved"]:
                results.append(result)

        return results


# ============================================================
# Portfolio Stop
# ============================================================
@dataclass
class PortfolioStop:
    """Portfolio-level risk controls"""
    max_drawdown: float = 0.15         # 15% max portfolio drawdown
    daily_loss_limit: float = 0.05     # 5% daily loss → reduce
    weekly_loss_limit: float = 0.08    # 8% weekly loss → reduce
    max_leverage: float = 1.0          # no leverage
    volatility_target: float = 0.20    # 20% annualized vol target

    def check(self, portfolio_value: float, peak_value: float,
              daily_return: float, weekly_return: float) -> Tuple[bool, str]:
        """
        Check portfolio stop conditions.

        Returns (should_reduce, reason)
        """
        # Drawdown stop
        if peak_value > 0:
            dd = (portfolio_value - peak_value) / peak_value
            if dd < -self.max_drawdown:
                return True, f"Portfolio drawdown {dd:.1%} exceeds limit"

        # Daily loss
        if daily_return < -self.daily_loss_limit:
            return True, f"Daily loss {daily_return:.1%} exceeds limit"

        # Weekly loss
        if weekly_return < -self.weekly_loss_limit:
            return True, f"Weekly loss {weekly_return:.1%} exceeds limit"

        return False, "OK"

    def get_target_exposure(self, current_vol: float) -> float:
        """Calculate target exposure given current volatility"""
        if current_vol <= 0:
            return 1.0
        target = self.volatility_target / current_vol
        return min(target, self.max_leverage)


# ============================================================
# Portfolio Risk Manager (Main Orchestrator)
# ============================================================
class PortfolioRiskManager:
    """
    Top-level portfolio risk orchestrator.

    Integrates:
      - RiskBudget: how risk is allocated
      - PositionSizer: how much to buy
      - CorrelationMatrix: diversification penalty
      - SectorLimits: concentration control
      - PortfolioStop: drawdown protection
    """

    def __init__(
        self,
        risk_mode: str = "equal_risk",
        total_risk: float = 0.20,
        max_positions: int = 5,
        max_drawdown: float = 0.15,
    ):
        self.risk_budget = RiskBudget(
            mode=risk_mode,
            total_risk_budget=total_risk,
            max_positions=max_positions,
        )
        self.sizer = PositionSizer(risk_budget=self.risk_budget)
        self.portfolio_stop = PortfolioStop(max_drawdown=max_drawdown)
        self.corr_matrix = self.sizer.corr_matrix
        self.sector_limits = self.sizer.sector_limits

        # State
        self.holdings: Dict[str, Dict] = {}
        self.peak_value: float = 0.0
        self.daily_returns: List[float] = []
        self.weekly_returns: List[float] = []

    def evaluate_candidates(
        self,
        candidates: List[Dict],
        capital: float,
        portfolio_value: float,
    ) -> List[Dict]:
        """
        Evaluate and size a set of candidates.

        Returns list of approved positions with sizing.
        """
        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        # Check portfolio stops
        daily_ret = self.daily_returns[-1] if self.daily_returns else 0
        weekly_ret = (sum(self.weekly_returns[-5:]) if len(self.weekly_returns) >= 5
                      else sum(self.weekly_returns))
        should_reduce, reason = self.portfolio_stop.check(
            portfolio_value, self.peak_value, daily_ret, weekly_ret)

        if should_reduce:
            print(f"  [Portfolio Stop] {reason} — reducing exposure")
            return []  # No new positions

        # Size the portfolio
        return self.sizer.size_portfolio(candidates, capital, self.holdings)

    def update_returns(self, returns: Dict[str, float]):
        """Update with daily returns for correlation tracking"""
        self.daily_returns.append(sum(returns.values()))

        # Update correlation matrix returns cache
        ret_series = {}
        for sym, ret in returns.items():
            ret_series[sym] = pd.Series([ret])
        self.corr_matrix.update(ret_series)

    def get_portfolio_risk(self) -> Dict:
        """Calculate current portfolio risk metrics"""
        if not self.holdings:
            return {"total_risk": 0, "num_positions": 0, "sectors": [],
                    "avg_correlation": 0, "concentration_ok": True}

        # Total risk contribution
        total_risk = sum(
            h.get("risk_contrib_pct", 0) / 100
            for h in self.holdings.values()
        )

        # Sectors
        sectors = list(set(
            h.get("sector", "unknown") for h in self.holdings.values()
        ))

        # Average correlation
        symbols = list(self.holdings.keys())
        avg_corr = 0.0
        if len(symbols) >= 2:
            corrs = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    corrs.append(abs(self.corr_matrix.get_correlation(
                        symbols[i], symbols[j])))
            avg_corr = np.mean(corrs) if corrs else 0.0

        # Concentration check
        _, conc_msg = self.sector_limits.check({
            sym: {"sector": h.get("sector", "unknown"),
                  "weight": h.get("weight_pct", 0) / 100}
            for sym, h in self.holdings.items()
        })

        return {
            "total_risk": round(total_risk * 100, 1),
            "num_positions": len(self.holdings),
            "sectors": sectors,
            "avg_correlation": round(avg_corr, 3),
            "concentration_ok": conc_msg == "OK",
            "concentration_msg": conc_msg,
        }

    def print_portfolio_report(self):
        """Print current portfolio status"""
        risk = self.get_portfolio_risk()
        print()
        print("=" * 70)
        print("  PORTFOLIO RISK REPORT")
        print("=" * 70)
        print(f"  Positions: {risk['num_positions']}  |  "
              f"Risk Contrib: {risk['total_risk']}%  |  "
              f"Avg Corr: {risk['avg_correlation']}")
        print(f"  Sectors: {risk['sectors']}  |  "
              f"Concentration: {'OK' if risk['concentration_ok'] else 'WARN'}")
        print(f"  Portfolio DD Limit: {self.portfolio_stop.max_drawdown:.0%}")
        print(f"  Risk Mode: {self.risk_budget.mode}  |  "
              f"Max Positions: {self.risk_budget.max_positions}")

        if self.holdings:
            print(f"\n  {'Symbol':<10}{'Weight':<10}{'Risk%':<10}"
                  f"{'Vol':<10}{'CorrPen':<10}{'Sector':<15}")
            print(f"  {'-'*60}")
            for sym, h in self.holdings.items():
                print(f"  {sym:<10}{h.get('weight_pct',0):>7.1f}%  "
                      f"{h.get('risk_contrib_pct',0):>7.1f}%  "
                      f"{h.get('annual_vol',0)*100:>7.1f}%  "
                      f"{h.get('correlation_penalty',1):>7.2f}x  "
                      f"{h.get('sector','?'):<15}")


if __name__ == "__main__":
    # Demo
    prm = PortfolioRiskManager(
        risk_mode="equal_risk",
        total_risk=0.20,
        max_positions=5,
    )

    # Simulate candidates
    candidates = [
        {"symbol": "000858", "price": 150.0, "annual_vol": 0.32,
         "edge_estimate": 0.05, "sector": "food_beverage", "conviction": 0.8},
        {"symbol": "002371", "price": 280.0, "annual_vol": 0.40,
         "edge_estimate": 0.06, "sector": "semiconductor", "conviction": 0.7},
        {"symbol": "002036", "price": 12.5, "annual_vol": 0.38,
         "edge_estimate": 0.04, "sector": "electronics", "conviction": 0.6},
        {"symbol": "600519", "price": 1500.0, "annual_vol": 0.28,
         "edge_estimate": 0.03, "sector": "food_beverage", "conviction": 0.5},
        {"symbol": "600276", "price": 45.0, "annual_vol": 0.33,
         "edge_estimate": 0.04, "sector": "pharma", "conviction": 0.6},
    ]

    positions = prm.evaluate_candidates(candidates, capital=1_000_000,
                                         portfolio_value=1_000_000)

    # Update mock holdings
    for pos in positions:
        prm.holdings[pos["symbol"]] = {
            **pos,
            "sector": [c["sector"] for c in candidates
                       if c["symbol"] == pos["symbol"]][0],
        }

    prm.print_portfolio_report()
