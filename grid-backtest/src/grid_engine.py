"""
网格交易回测系统 — 核心引擎

模拟在价格区间内设置 N 档网格挂单，价格触及网格线自动触发买卖。
支持 T+1、交易成本、几何/算术网格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    GridConfig, COMMISSION_RATE, STAMP_TAX_RATE,
    MIN_COMMISSION, T_PLUS_ONE,
)


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class Trade:
    """单笔交易记录（不可变）"""
    date: pd.Timestamp
    action: str          # "BUY" | "SELL"
    price: float
    shares: int
    amount: float        # 成交金额
    commission: float    # 佣金
    stamp_tax: float     # 印花税
    grid_level: int      # 触发网格线编号
    cash_after: float
    shares_after: int


@dataclass(frozen=True)
class GridResult:
    """网格回测结果（不可变）"""
    config: GridConfig
    equity_curve: pd.DataFrame      # 每日权益
    trades: List[Trade]             # 所有交易
    grid_lines: np.ndarray          # 网格线价格
    total_return_pct: float         # 总收益率
    annualized_return_pct: float    # 年化收益率
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float                 # 胜率（盈利交易占比）
    total_trades: int
    profit_trades: int
    avg_profit_per_trade: float
    max_holding_shares: int
    max_holding_value: float
    grid_utilization: float         # 网格利用率（被触发的网格比例）
    final_cash: float
    final_shares: int
    final_value: float


# ============================================================
# 网格线生成
# ============================================================

def build_grid_lines(price_low: float, price_high: float,
                     step: float, mode: str = "geometric") -> np.ndarray:
    """
    生成网格线数组。

    Args:
        price_low: 价格下限
        price_high: 价格上限
        step: 步长（百分比，如 0.02 = 2%）
        mode: "geometric" 等比网格 | "arithmetic" 等差网格

    Returns:
        np.ndarray: 从低到高排列的网格线价格
    """
    if mode == "geometric":
        # 等比：grid[i] = price_low * (1+step)^i
        n = int(np.ceil(np.log(price_high / price_low) / np.log(1 + step)))
        n = max(n, 2)  # 至少 2 条线
        lines = price_low * (1 + step) ** np.arange(n + 1)
        # 裁剪到 [price_low, price_high]
        lines = lines[(lines >= price_low * 0.999) & (lines <= price_high * 1.001)]
    else:
        # 等差：grid[i] = price_low + i * Δ
        n = int(np.ceil((price_high - price_low) / (price_low * step)))
        n = max(n, 2)
        delta = price_low * step
        lines = price_low + delta * np.arange(n + 1)
        lines = lines[(lines >= price_low * 0.999) & (lines <= price_high * 1.001)]

    return np.round(lines, 3)


def get_grid_level(price: float, grid_lines: np.ndarray) -> int:
    """返回价格所在的网格区间索引（下方网格线编号）"""
    idx = np.searchsorted(grid_lines, price, side="right") - 1
    return max(0, min(idx, len(grid_lines) - 2))


# ============================================================
# 网格回测引擎
# ============================================================

class GridBacktestEngine:
    """网格交易回测引擎"""

    def __init__(self):
        self.cash: float = 0.0
        self.shares: int = 0
        self.trades: List[Trade] = []
        self.equity_records: List[dict] = []

        # 网格状态
        self.grid_lines: np.ndarray = np.array([])
        self.bought_at_grid: Dict[int, float] = {}   # grid_idx -> buy_price
        self.shares_at_grid: Dict[int, int] = {}     # grid_idx -> shares held
        self.t1_lock: Dict[int, int] = {}            # grid_idx -> shares (T+1锁定)

        # 调试信息
        self._cross_events: List[str] = []

    def _commission(self, amount: float) -> float:
        """计算佣金"""
        return max(amount * COMMISSION_RATE, MIN_COMMISSION)

    def _stamp_tax(self, amount: float) -> float:
        """计算印花税（仅卖出）"""
        return amount * STAMP_TAX_RATE

    def _available_shares(self) -> int:
        """可卖出的股数（排除 T+1 锁定）"""
        locked = sum(self.t1_lock.values())
        return max(0, self.shares - locked)

    def _record_equity(self, date: pd.Timestamp, close: float):
        """记录当日权益"""
        total_value = self.cash + self.shares * close
        self.equity_records.append({
            "date": date,
            "close": close,
            "cash": self.cash,
            "shares": self.shares,
            "total_value": total_value,
        })

    def _execute_buy(self, date: pd.Timestamp, price: float,
                     grid_idx: int) -> Optional[Trade]:
        """在网格线 grid_idx 执行买入"""
        # 该网格线是否已有持仓
        if grid_idx in self.shares_at_grid and self.shares_at_grid[grid_idx] > 0:
            return None  # 已买过，不重复

        cfg = self._config
        # 计算买入金额
        buy_amount = cfg.initial_capital * cfg.position_per_grid

        if self.cash < buy_amount + self._commission(buy_amount):
            # 现金不足，用剩余现金买
            buy_amount = self.cash / (1 + COMMISSION_RATE) * 0.95
            if buy_amount < price * 100:  # 不够 1 手
                return None

        shares = int(buy_amount / price / 100) * 100  # 整手
        if shares < 100:
            return None

        actual_amount = shares * price
        commission = self._commission(actual_amount)
        total_cost = actual_amount + commission

        if total_cost > self.cash:
            # 重新以可用现金计算
            shares = int((self.cash / (1 + COMMISSION_RATE)) / price / 100) * 100
            if shares < 100:
                return None
            actual_amount = shares * price
            commission = self._commission(actual_amount)
            total_cost = actual_amount + commission

        # 执行买入
        self.cash -= total_cost
        self.shares += shares
        self.bought_at_grid[grid_idx] = price
        self.shares_at_grid[grid_idx] = shares

        if T_PLUS_ONE:
            self.t1_lock[grid_idx] = self.t1_lock.get(grid_idx, 0) + shares

        trade = Trade(
            date=date, action="BUY", price=price,
            shares=shares, amount=actual_amount,
            commission=commission, stamp_tax=0.0,
            grid_level=grid_idx,
            cash_after=self.cash, shares_after=self.shares,
        )
        self.trades.append(trade)
        return trade

    def _execute_sell(self, date: pd.Timestamp, price: float,
                      grid_idx: int, buy_grid_idx: int) -> Optional[Trade]:
        """在网格线 grid_idx 卖出在 buy_grid_idx 买入的持仓"""
        if buy_grid_idx not in self.shares_at_grid:
            return None

        shares = self.shares_at_grid.get(buy_grid_idx, 0)
        if shares <= 0:
            return None

        # T+1 检查
        if T_PLUS_ONE:
            locked = self.t1_lock.get(buy_grid_idx, 0)
            if locked >= shares:
                return None  # 全部锁定
            if locked > 0:
                shares = shares - locked  # 部分卖出

        if shares < 100:
            return None

        # 确保整手
        shares = (shares // 100) * 100
        if shares < 100:
            return None

        actual_amount = shares * price
        commission = self._commission(actual_amount)
        stamp_tax = self._stamp_tax(actual_amount)
        total_revenue = actual_amount - commission - stamp_tax

        # 执行卖出
        self.cash += total_revenue
        self.shares -= shares

        # 更新网格状态
        remaining = self.shares_at_grid[buy_grid_idx] - shares
        if remaining <= 0:
            del self.shares_at_grid[buy_grid_idx]
            del self.bought_at_grid[buy_grid_idx]
            if buy_grid_idx in self.t1_lock:
                del self.t1_lock[buy_grid_idx]
        else:
            self.shares_at_grid[buy_grid_idx] = remaining
            self.t1_lock[buy_grid_idx] = max(0, self.t1_lock.get(buy_grid_idx, 0) - shares)

        trade = Trade(
            date=date, action="SELL", price=price,
            shares=shares, amount=actual_amount,
            commission=commission, stamp_tax=stamp_tax,
            grid_level=grid_idx,
            cash_after=self.cash, shares_after=self.shares,
        )
        self.trades.append(trade)
        return trade

    def _release_t1(self):
        """释放 T+1 锁定（次日生效）"""
        self.t1_lock.clear()

    def _check_grid_cross(self, prev_close: float, today_low: float,
                          today_high: float, today_close: float,
                          date: pd.Timestamp):
        """检查当日价格是否穿越网格线，执行交易"""
        n = len(self.grid_lines)

        for i in range(n):
            line = self.grid_lines[i]

            # 价格向下穿越网格线 → 买入信号
            if today_low <= line <= prev_close:
                self._execute_buy(date, line, i)

            # 价格向上穿越网格线 → 卖出信号
            if prev_close <= line <= today_high:
                # 寻找下方最近的已买入网格
                for j in range(i - 1, -1, -1):
                    if j in self.shares_at_grid and self.shares_at_grid[j] > 0:
                        self._execute_sell(date, line, i, j)
                        break

    def run(self, df: pd.DataFrame, config: GridConfig) -> GridResult:
        """
        执行网格回测。

        Args:
            df: 日线数据 (date, open, close, high, low)
            config: 网格配置

        Returns:
            GridResult 包含完整的回测结果
        """
        self._config = config
        self.cash = config.initial_capital
        self.shares = 0
        self.trades = []
        self.equity_records = []
        self.bought_at_grid = {}
        self.shares_at_grid = {}
        self.t1_lock = {}
        self._cross_events = []

        df = df.copy()
        df = df[(df["date"] >= pd.Timestamp(config.start_date))
                & (df["date"] <= pd.Timestamp(config.end_date))]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("回测区间内无数据")

        # 确定价格区间
        if config.price_low is not None and config.price_high is not None:
            price_low = config.price_low
            price_high = config.price_high
        else:
            # 自动计算：取回测初期价格的 ±15%
            mid = float(df["close"].iloc[:20].mean())
            margin = config.position_per_grid * 3  # 基于仓位推算默认边距
            if margin < 0.05:
                margin = 0.15
            price_low = mid * (1 - margin)
            price_high = mid * (1 + margin)

        # 构建网格线
        self.grid_lines = build_grid_lines(
            price_low, price_high, config.grid_step, config.grid_mode,
        )

        # 初始化起点的网格买入（在第一条网格线以下时提前建仓）
        first_close = float(df["close"].iloc[0])
        init_grid = get_grid_level(first_close, self.grid_lines)
        for g in range(init_grid + 1):
            if self.cash > config.initial_capital * 0.1:
                self._execute_buy(df["date"].iloc[0], self.grid_lines[g], g)

        prev_close = first_close

        # ============================================================
        # 逐日回测
        # ============================================================
        for idx in range(len(df)):
            row = df.iloc[idx]
            date = row["date"]
            today_open = float(row["open"])
            today_high = float(row["high"])
            today_low = float(row["low"])
            today_close = float(row["close"])

            # T+1 锁释放
            if T_PLUS_ONE and idx > 0:
                self._release_t1()

            # 用开盘价判断穿越
            self._check_grid_cross(prev_close, today_low, today_high,
                                    today_close, date)

            # 止损检查：价格跌破下限过多
            stop_price = price_low * (1 - config.stop_loss_pct)
            if today_low <= stop_price and self.shares > 0:
                # 清仓
                sell_price = max(today_open, stop_price)
                while self.shares >= 100:
                    sell_shares = (self.shares // 100) * 100
                    if sell_shares < 100:
                        sell_shares = self.shares
                    actual_amount = sell_shares * sell_price
                    commission = self._commission(actual_amount)
                    stamp = self._stamp_tax(actual_amount)
                    self.cash += actual_amount - commission - stamp
                    self.shares -= sell_shares
                    self.trades.append(Trade(
                        date=date, action="SELL(STOP)", price=sell_price,
                        shares=sell_shares, amount=actual_amount,
                        commission=commission, stamp_tax=stamp,
                        grid_level=-1,
                        cash_after=self.cash, shares_after=self.shares,
                    ))
                self.bought_at_grid.clear()
                self.shares_at_grid.clear()
                self.t1_lock.clear()

            # 记录当日权益
            self._record_equity(date, today_close)

            # 更新 prev_close（考虑日内高低点的穿越后，用收盘价作为次日参考）
            prev_close = today_close

        # ============================================================
        # 计算绩效指标
        # ============================================================
        equity = pd.DataFrame(self.equity_records)
        if equity.empty:
            raise ValueError("回测结果为空")

        # 最后一天清仓计算最终价值
        final_close = float(equity["close"].iloc[-1])
        final_value = self.cash + self.shares * final_close
        total_ret = (final_value / config.initial_capital) - 1

        # 年化收益
        days = len(equity)
        years = days / 252
        cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

        # 日收益率序列
        equity["daily_ret"] = equity["total_value"].pct_change()
        daily_rets = equity["daily_ret"].dropna()

        # 夏普比率
        if daily_rets.std() > 0:
            sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # 最大回撤
        peak = equity["total_value"].expanding().max()
        drawdown = (equity["total_value"] - peak) / peak
        max_dd = float(drawdown.min())

        # 卡尔玛比率
        calmar = cagr / abs(max_dd) if max_dd < -0.001 else 0.0

        # 交易分析
        buy_trades = [t for t in self.trades if t.action == "BUY"]
        sell_trades = [t for t in self.trades if t.action.startswith("SELL")]
        total_trades = len(self.trades)

        # 按网格层级配对：sell 在 grid_i → 对应 buy 在 grid_{i-1}
        profit_trades = 0
        total_profit = 0.0
        paired_buys = set()  # 已配对的买入交易（按对象id）

        for st in sell_trades:
            # 找同一天之前、最近的低一档买入
            best_buy = None
            for bt in buy_trades:
                if id(bt) in paired_buys:
                    continue
                if bt.grid_level < st.grid_level and bt.date <= st.date:
                    if best_buy is None or bt.grid_level > best_buy.grid_level:
                        best_buy = bt
            if best_buy is not None:
                paired_buys.add(id(best_buy))
                if st.price > best_buy.price:
                    profit_trades += 1
                total_profit += (st.price - best_buy.price) * min(st.shares, best_buy.shares)

        win_rate = profit_trades / len(sell_trades) * 100 if sell_trades else 0
        avg_profit = total_profit / len(sell_trades) if sell_trades else 0

        # 网格利用率：被触发过的网格层级 / 总网格区间数，封顶 100%
        total_intervals = max(len(self.grid_lines) - 1, 1)
        used_levels = len(set(t.grid_level for t in self.trades if t.grid_level >= 0))
        grid_util = min(used_levels / total_intervals, 1.0)

        # 最大持仓
        max_shares = max((r["shares"] for r in self.equity_records), default=0)
        max_hold_val = max((r["shares"] * r["close"] for r in self.equity_records), default=0)

        result = GridResult(
            config=config,
            equity_curve=equity,
            trades=self.trades,
            grid_lines=self.grid_lines,
            total_return_pct=total_ret * 100,
            annualized_return_pct=cagr * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_trades=profit_trades,
            avg_profit_per_trade=avg_profit,
            max_holding_shares=max_shares,
            max_holding_value=max_hold_val,
            grid_utilization=grid_util * 100,
            final_cash=self.cash,
            final_shares=self.shares,
            final_value=final_value,
        )

        return result


def run_grid_backtest(df: pd.DataFrame, config: GridConfig) -> GridResult:
    """便捷函数：运行一次网格回测"""
    engine = GridBacktestEngine()
    return engine.run(df, config)
