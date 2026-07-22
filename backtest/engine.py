"""
回测引擎 —— 模拟A股T+1制度下的交易执行
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from config import (
    INITIAL_CAPITAL, POSITION_SIZE_PCT, MIN_SHARES,
    COMMISSION_RATE, STAMP_TAX_RATE, MIN_COMMISSION, SLIPPAGE,
    T_PLUS_ONE,
)
from strategies.base import BaseStrategy


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    holding_days: int = 0
    max_drawdown_pct: float = 0.0      # 持仓期间最大浮亏%
    max_profit_pct: float = 0.0         # 持仓期间最大浮盈%

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    def summary(self) -> str:
        if self.exit_date is None:
            return f"{self.entry_date.date()} BUY {self.shares}股@{self.entry_price:.2f} (持仓中)"
        tag = "赢" if self.is_win else "亏"
        return (
            f"{self.entry_date.date()} → {self.exit_date.date()} | "
            f"{self.entry_price:.2f} → {self.exit_price:.2f} | "
            f"{self.shares}股 | {tag} {self.pnl:+.2f} ({self.pnl_pct:+.2%}) | "
            f"持{self.holding_days}天 | {self.reason}"
        )


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    strategy_params: Dict[str, Any]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_holding_days: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    equity_curve: pd.DataFrame
    trades: List[Trade]
    benchmark_return: float


class BacktestEngine:
    """
    A股波段回测引擎

    特性:
      - T+1 制度: 当日买入，最早次日卖出
      - 交易成本: 佣金(买卖) + 印花税(卖出) + 滑点
      - 整数手交易: 100股为一手
      - 止损/止盈: 日内监控
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        position_size_pct: float = POSITION_SIZE_PCT,
        commission_rate: float = COMMISSION_RATE,
        stamp_tax_rate: float = STAMP_TAX_RATE,
        min_commission: float = MIN_COMMISSION,
        slippage: float = SLIPPAGE,
        t_plus_one: bool = T_PLUS_ONE,
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.t_plus_one = t_plus_one

    def run(self, df: pd.DataFrame, strategy: BaseStrategy,
            stop_loss_pct: float = 0.06,
            take_profit_pct: float = 0.18,
            trailing_stop_pct: float = 0.05) -> BacktestResult:
        """
        执行回测。

        Parameters
        ----------
        df : DataFrame with columns: date, open, high, low, close, ... (all indicators)
        strategy : 策略对象
        stop_loss_pct : 硬止损比例
        take_profit_pct : 硬止盈比例
        trailing_stop_pct : 移动止损回撤比例

        Returns
        -------
        BacktestResult
        """
        # 生成信号
        signals = strategy.generate_signals(df)

        # 初始化状态
        cash = self.initial_capital
        position = 0               # 持仓股数
        entry_price = 0
        entry_date = None
        highest_since_entry = 0
        can_sell_today = False     # T+1: 今天能否卖
        last_buy_date = None

        trades: List[Trade] = []
        current_trade: Optional[Trade] = None

        equity_records = []        # 每日权益记录

        for i in range(len(df)):
            date = df["date"].iloc[i]
            open_p = df["open"].iloc[i]
            high_p = df["high"].iloc[i]
            low_p = df["low"].iloc[i]
            close_p = df["close"].iloc[i]
            signal = signals.iloc[i]

            # ---- 持仓更新 ----
            if position > 0:
                # 更新最高价
                highest_since_entry = max(highest_since_entry, high_p)

                # 获取动态止损止盈 (如果策略提供)
                if hasattr(strategy, 'get_dynamic_stops') and strategy.get_dynamic_stops() is not None:
                    dyn_stops = strategy.get_dynamic_stops()
                    dyn_stop_pct = dyn_stops["atr_stop_pct"].iloc[i]
                    dyn_target_pct = dyn_stops["atr_target_pct"].iloc[i]
                    effective_stop_pct = dyn_stop_pct
                    effective_target_pct = dyn_target_pct
                else:
                    effective_stop_pct = stop_loss_pct
                    effective_target_pct = take_profit_pct

                # 计算当日止损价格
                hard_stop = entry_price * (1 - effective_stop_pct)
                trail_stop = highest_since_entry * (1 - trailing_stop_pct)
                effective_stop = max(hard_stop, trail_stop)

                # 计算当日止盈价格
                take_profit = entry_price * (1 + effective_target_pct)

                sell_triggered = False
                sell_reason = ""
                sell_price = 0.0

                # 止损检查（按优先级）
                if low_p <= effective_stop and can_sell_today:
                    sell_triggered = True
                    sell_price = min(open_p, effective_stop * (1 - self.slippage))
                    sell_reason = f"止损({effective_stop:.2f})"

                # 止盈检查
                elif high_p >= take_profit and can_sell_today:
                    sell_triggered = True
                    sell_price = max(open_p, take_profit * (1 + self.slippage))
                    sell_reason = f"止盈({take_profit:.2f})"

                # 信号卖出
                elif signal == -1 and can_sell_today:
                    sell_triggered = True
                    sell_price = close_p * (1 - self.slippage)
                    sell_reason = "策略信号卖出"

                # 执行卖出
                if sell_triggered and position > 0:
                    proceeds = sell_price * position
                    commission = max(proceeds * self.commission_rate, self.min_commission)
                    stamp_tax = proceeds * self.stamp_tax_rate
                    cash += proceeds - commission - stamp_tax

                    pnl = (sell_price - entry_price) * position - commission - stamp_tax
                    pnl_pct = (sell_price - entry_price) / entry_price

                    # 完成交易记录
                    if current_trade:
                        current_trade.exit_date = date
                        current_trade.exit_price = sell_price
                        current_trade.pnl = pnl
                        current_trade.pnl_pct = pnl_pct
                        current_trade.reason = sell_reason
                        trading_days = len(df[(df["date"] > current_trade.entry_date) &
                                              (df["date"] <= date)])
                        current_trade.holding_days = max(1, trading_days)
                        trades.append(current_trade)

                    position = 0
                    current_trade = None
                    entry_price = 0
                    highest_since_entry = 0
                    can_sell_today = False

            # ---- 开仓逻辑 ----
            if signal == 1 and position == 0:
                buy_price = close_p * (1 + self.slippage)
                budget = cash * self.position_size_pct
                raw_shares = int(budget / buy_price)
                shares = (raw_shares // MIN_SHARES) * MIN_SHARES

                if shares >= MIN_SHARES:
                    cost = buy_price * shares
                    commission = max(cost * self.commission_rate, self.min_commission)
                    total_cost = cost + commission

                    if total_cost <= cash:
                        cash -= total_cost
                        position = shares
                        entry_price = buy_price
                        entry_date = date
                        highest_since_entry = high_p
                        last_buy_date = date
                        can_sell_today = not self.t_plus_one

                        current_trade = Trade(
                            entry_date=date,
                            entry_price=buy_price,
                            shares=shares,
                        )

            # T+1 解锁: 到了下一个交易日
            if self.t_plus_one and position > 0 and date > last_buy_date:
                can_sell_today = True

            # ---- 持仓期浮盈浮亏追踪 ----
            if position > 0 and current_trade:
                current_pnl = (close_p - entry_price) / entry_price
                current_trade.max_drawdown_pct = min(
                    current_trade.max_drawdown_pct,
                    (low_p - entry_price) / entry_price
                )
                current_trade.max_profit_pct = max(
                    current_trade.max_profit_pct,
                    (high_p - entry_price) / entry_price
                )

            # ---- 记录每日权益 ----
            market_value = position * close_p if position > 0 else 0
            equity = cash + market_value

            equity_records.append({
                "date": date,
                "equity": equity,
                "cash": cash,
                "position": position,
                "market_value": market_value,
                "close": close_p,
                "signal": signal,
            })

        # ---- 强制平仓 (回测期末) ----
        if position > 0:
            last_close = df["close"].iloc[-1]
            proceeds = last_close * position
            commission = max(proceeds * self.commission_rate, self.min_commission)
            stamp_tax = proceeds * self.stamp_tax_rate
            cash += proceeds - commission - stamp_tax

            if current_trade:
                current_trade.exit_date = df["date"].iloc[-1]
                current_trade.exit_price = last_close
                pnl = (last_close - entry_price) * position - commission - stamp_tax
                current_trade.pnl = pnl
                current_trade.pnl_pct = (last_close - entry_price) / entry_price
                current_trade.reason = "回测结束强制平仓"
                trading_days = len(df[(df["date"] > current_trade.entry_date)])
                current_trade.holding_days = max(1, trading_days)
                trades.append(current_trade)

            position = 0

        # ---- 汇总统计 ----
        equity_df = pd.DataFrame(equity_records)
        equity_df["ret"] = equity_df["equity"].pct_change()

        # 回测区间总收益率
        total_return = (cash - self.initial_capital) / self.initial_capital

        # 年化
        total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        years = total_days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        cummax = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # 最大回撤持续期 (从开始回撤到创新高)
        max_dd_duration = 0
        if pd.notna(max_dd_idx):
            if max_dd_idx > 0:
                peak_idx = cummax[:max_dd_idx].idxmax()
            else:
                peak_idx = max_dd_idx
            peak_date = equity_df["date"].iloc[peak_idx]
            recovery = equity_df["equity"].iloc[max_dd_idx:]
            recovery_idx = (recovery >= cummax.iloc[max_dd_idx])
            if recovery_idx.any() and len(recovery_idx) > 0:
                try:
                    recover_date = equity_df["date"].iloc[max_dd_idx:].iloc[recovery_idx.argmax()]
                    max_dd_duration = (recover_date - peak_date).days
                except (ValueError, IndexError):
                    max_dd_duration = (equity_df["date"].iloc[-1] - peak_date).days
            else:
                max_dd_duration = (equity_df["date"].iloc[-1] - peak_date).days

        # 夏普比率 (假设无风险利率3%)
        rf_daily = 0.03 / 252
        excess = equity_df["ret"].dropna() - rf_daily
        sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

        # 索提诺比率 (只考虑下行波动)
        downside = excess[excess < 0]
        sortino = np.sqrt(252) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

        # 卡尔玛比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # 盈亏统计
        winning = [t for t in trades if t.is_win]
        losing = [t for t in trades if not t.is_win]
        win_rate = len(winning) / len(trades) if trades else 0
        total_wins = sum(t.pnl for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0

        # 基准收益 (买入持有)
        bench_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

        return BacktestResult(
            strategy_name=strategy.name,
            strategy_params=strategy.params,
            start_date=df["date"].iloc[0],
            end_date=df["date"].iloc[-1],
            initial_capital=self.initial_capital,
            final_equity=cash,
            total_pnl=cash - self.initial_capital,
            total_return_pct=total_return * 100,
            annualized_return_pct=annual_return * 100,
            max_drawdown_pct=max_dd * 100,
            max_drawdown_duration=int(max_dd_duration),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            avg_win_pct=avg_win * 100,
            avg_loss_pct=avg_loss * 100,
            avg_holding_days=avg_hold,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            equity_curve=equity_df,
            trades=trades,
            benchmark_return=bench_ret * 100,
        )
