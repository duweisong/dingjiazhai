"""
A股网格交易回测 — 网格引擎

基于 entry.py 的网格策略逻辑，自包含实现:
  - 动态网格: N 日 Highest/Lowest → 中枢 → 等比网格线
  - T+1 制度: 当日买入次日才能卖出
  - 整手交易: 100 股整数倍
  - 交易成本: 佣金 + 印花税 + 过户费

策略逻辑:
  1. 计算 N 日最高/最低价的中枢 (mid)
  2. 从中枢向上/下等比展开 N 档网格线
  3. 初始建仓: 找到当前价下方最近的网格线，按比例建仓
  4. 后续交易: 价升穿上一档 → 卖出; 价跌破下一档 → 买入
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .commissions import calc_commission, calc_stamp_tax, calc_transfer_fee
from .config import (
    GridConfig,
    MIN_SHARES,
    T_PLUS_ONE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trade:
    """单笔交易记录 (不可变)"""

    date: pd.Timestamp
    action: str  # "INIT_BUY" | "BUY" | "SELL" | "SELL(FINAL)"
    price: float
    shares: int
    amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    grid_level: int
    target_pct: float
    cash_after: float
    shares_after: int


@dataclass
class GridResult:
    """网格回测结果"""

    config: GridConfig
    equity_curve: pd.DataFrame
    trades: list[Trade]
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    profit_trades: int
    final_value: float
    final_cash: float
    final_shares: int
    benchmark_return_pct: float = 0.0

    def summary(self) -> str:
        """生成可读摘要"""
        return (
            f"\n{'='*60}\n"
            f"  {self.config.symbol} 网格回测结果\n"
            f"{'='*60}\n"
            f"  网格: {self.config.grid_period}日周期  "
            f"{self.config.grid_levels}档  "
            f"{self.config.grid_step:.1%}步长\n"
            f"  区间: {self.config.start_date} ~ {self.config.end_date}\n"
            f"{'─'*60}\n"
            f"  总收益率:      {self.total_return_pct:>+8.2f}%\n"
            f"  年化收益率:    {self.annualized_return_pct:>+8.2f}%\n"
            f"  夏普比率:      {self.sharpe_ratio:>8.2f}\n"
            f"  最大回撤:      {self.max_drawdown_pct:>8.2f}%\n"
            f"  卡尔玛比率:    {self.calmar_ratio:>8.2f}\n"
            f"{'─'*60}\n"
            f"  最终资金:      {self.final_value:>10,.2f}\n"
            f"  总交易次数:    {self.total_trades:>10}\n"
            f"  胜率:          {self.win_rate:>9.1f}%\n"
            f"  基准收益:      {self.benchmark_return_pct:>+8.2f}% (买入持有)\n"
            f"{'='*60}\n"
        )


class GridEngine:
    """A股网格交易回测引擎 (自包含，无框架依赖)"""

    def __init__(self) -> None:
        self.cash: float = 0.0
        self.shares: int = 0
        self.trades: list[Trade] = []
        self.equity_records: list[dict] = []

        # 网格状态
        self.price_levels: list[float] = []
        self.last_level_index: int | None = None

        # T+1 追踪 (bar index of last buy)
        self._buy_bar: int | None = None

        # 配置引用
        self._config: GridConfig | None = None

    # ============================================================
    # 网格计算
    # ============================================================

    def _build_grid(self, high: float, low: float) -> list[float]:
        """基于最高/最低价中枢，构建网格线数组 (与 entry.py 一致)"""
        cfg = self._config
        mid = (high + low) / 2.0

        levels = []
        for i in range(cfg.grid_levels):
            offset = cfg.grid_top_pct - i * cfg.grid_step
            levels.append(mid * (1.0 + offset))

        # 验证: 最低档 = 中点 × (1 + grid_bot_pct)
        expected_bot = mid * (1.0 + cfg.grid_bot_pct)
        actual_bot = levels[-1]
        if abs(actual_bot - expected_bot) / abs(expected_bot) > 0.001:
            logger.warning(
                "网格最低档 %.4f 与 grid_bot_pct 推导值 %.4f 不一致，"
                "请确认 grid_top_pct / grid_step / grid_levels 一致性",
                actual_bot, expected_bot,
            )

        return levels

    # ============================================================
    # T+1 & 闸门
    # ============================================================

    def _can_sell(self, bar_idx: int) -> bool:
        """T+1: 买入当天不能卖，次日可卖"""
        if not T_PLUS_ONE:
            return True
        if self._buy_bar is None:
            return True
        return bar_idx > self._buy_bar

    def _gate_allows_buy(self, price: float, ma_val: float | None) -> bool:
        """闸门检查 (处理 NaN)"""
        cfg = self._config
        if not cfg.use_ma_gate:
            return True
        if ma_val is None or pd.isna(ma_val):
            return True

        if cfg.ma_gate_rule == "below":
            return price >= ma_val
        elif cfg.ma_gate_rule == "above":
            return price <= ma_val
        return True

    # ============================================================
    # 仓位计算
    # ============================================================

    def _current_equity(self, price: float) -> float:
        """当前总权益 (现金 + 持仓市值)"""
        return self.cash + self.shares * price

    def _shares_for_value(self, target_value: float, price: float) -> int:
        """将目标金额转为整手股数"""
        if target_value <= 0 or price <= 0:
            return 0
        raw = int(target_value / price)
        lots = raw // MIN_SHARES
        if lots < 1:
            return 0
        return lots * MIN_SHARES

    def _max_affordable_shares(self, price: float) -> int:
        """在交易成本约束下，最多能买多少股"""
        # 二分查找最大可负担股数
        lo, hi = 0, int(self.cash / price / MIN_SHARES) * MIN_SHARES
        hi = max(hi, MIN_SHARES)
        best = 0
        while lo <= hi:
            mid = ((lo // MIN_SHARES + hi // MIN_SHARES) // 2) * MIN_SHARES
            if mid < MIN_SHARES:
                break
            amount = mid * price
            total_cost = amount + calc_commission(amount) + calc_transfer_fee(amount)
            if total_cost <= self.cash:
                best = mid
                lo = mid + MIN_SHARES
            else:
                hi = mid - MIN_SHARES
        return best

    # ============================================================
    # 执行交易
    # ============================================================

    def _record_equity(self, date: pd.Timestamp, close: float) -> None:
        """记录当日权益"""
        self.equity_records.append({
            "date": date,
            "close": close,
            "cash": self.cash,
            "shares": self.shares,
            "total_value": self._current_equity(close),
        })

    def _execute_trade(
        self, date: pd.Timestamp, price: float,
        action: str, target_shares: int, grid_level: int,
        target_pct: float, bar_idx: int,
    ) -> None:
        """执行买卖，记录交易"""
        current_shares = self.shares

        if action in ("INIT_BUY", "BUY"):
            buy_shares = target_shares - current_shares
            if buy_shares < MIN_SHARES:
                return
            buy_shares = (buy_shares // MIN_SHARES) * MIN_SHARES

            # 确保买得起
            amount = buy_shares * price
            comm = calc_commission(amount)
            transfer = calc_transfer_fee(amount)
            if amount + comm + transfer > self.cash:
                buy_shares = self._max_affordable_shares(price)
                if buy_shares < MIN_SHARES:
                    return
                amount = buy_shares * price
                comm = calc_commission(amount)
                transfer = calc_transfer_fee(amount)

            total_cost = amount + comm + transfer
            self.cash -= total_cost
            self.shares += buy_shares
            self._buy_bar = bar_idx

            self.trades.append(Trade(
                date=date, action=action, price=price,
                shares=buy_shares, amount=amount,
                commission=comm, stamp_tax=0.0,
                transfer_fee=transfer,
                grid_level=grid_level, target_pct=target_pct,
                cash_after=self.cash, shares_after=self.shares,
            ))

        elif action == "SELL":
            sell_shares = current_shares - target_shares
            if sell_shares < MIN_SHARES:
                return
            sell_shares = (sell_shares // MIN_SHARES) * MIN_SHARES
            amount = sell_shares * price
            comm = calc_commission(amount)
            stamp = calc_stamp_tax(amount)
            transfer = calc_transfer_fee(amount)
            revenue = amount - comm - stamp - transfer

            self.cash += revenue
            self.shares -= sell_shares

            self.trades.append(Trade(
                date=date, action=action, price=price,
                shares=sell_shares, amount=amount,
                commission=comm, stamp_tax=stamp,
                transfer_fee=transfer,
                grid_level=grid_level, target_pct=target_pct,
                cash_after=self.cash, shares_after=self.shares,
            ))

    # ============================================================
    # 主回测循环
    # ============================================================

    def run(self, df: pd.DataFrame, config: GridConfig) -> GridResult:
        """
        执行网格回测。

        Args:
            df: 日线数据 (date, open, high, low, close, volume, amount)
            config: 网格配置

        Returns:
            GridResult
        """
        self._config = config
        self.cash = config.initial_capital
        self.shares = 0
        self.trades = []
        self.equity_records = []
        self.price_levels = []
        self.last_level_index = None
        self._buy_bar = None

        # 过滤日期范围
        df = df.copy()
        df = df[(df["date"] >= pd.Timestamp(config.start_date))
                & (df["date"] <= pd.Timestamp(config.end_date))]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("回测区间内无数据")

        # 预计算指标
        high_series = df["high"].astype(float)
        low_series = df["low"].astype(float)
        close_series = df["close"].astype(float)

        # 滚动高低点
        rolling_high = (
            high_series.rolling(config.grid_period, min_periods=config.grid_period)
            .max()
        )
        rolling_low = (
            low_series.rolling(config.grid_period, min_periods=config.grid_period)
            .min()
        )

        # 可选: 均线闸门
        if config.use_ma_gate:
            ma_series = close_series.rolling(config.ma_gate_period).mean()
        else:
            ma_series = pd.Series([None] * len(df), dtype=float)

        # ---- 逐日回测 ----
        for idx in range(len(df)):
            date = df["date"].iloc[idx]
            close = float(close_series.iloc[idx])

            # 数据不足时跳过
            if idx < config.grid_period:
                self._record_equity(date, close)
                continue

            current_high = float(rolling_high.iloc[idx])
            current_low = float(rolling_low.iloc[idx])
            ma_val = float(ma_series.iloc[idx]) if config.use_ma_gate else None

            if pd.isna(current_high) or pd.isna(current_low):
                self._record_equity(date, close)
                continue

            if config.use_ma_gate and ma_val is not None and pd.isna(ma_val):
                # MA 尚未就绪: 放行
                pass

            # 构建当前网格线
            self.price_levels = self._build_grid(current_high, current_low)

            # ---- 初始建仓 ----
            if self.last_level_index is None:
                for i in range(len(self.price_levels)):
                    if close > self.price_levels[i]:
                        self.last_level_index = i
                        target_pct = i / (len(self.price_levels) - 1)

                        if not self._gate_allows_buy(close, ma_val):
                            self._record_equity(date, close)
                            break

                        equity = self._current_equity(close)
                        target_value = equity * target_pct
                        target_shares = self._shares_for_value(target_value, close)

                        if target_shares >= MIN_SHARES:
                            self._execute_trade(
                                date, close, "INIT_BUY",
                                target_shares, i, target_pct, idx,
                            )
                        self._record_equity(date, close)
                        break
                else:
                    self._record_equity(date, close)
                continue

            # ---- 持续交易: 检测网格穿越 ----
            signal = False
            while True:
                upper: float | None = None
                lower: float | None = None

                if self.last_level_index > 0:
                    upper = self.price_levels[self.last_level_index - 1]
                if self.last_level_index < len(self.price_levels) - 1:
                    lower = self.price_levels[self.last_level_index + 1]

                # 价格涨穿上一档 → 卖 (需 T+1 解锁)
                if upper is not None and close > upper:
                    if self._can_sell(idx):
                        self.last_level_index -= 1
                        signal = True
                        continue

                # 价格跌破下一档 → 买
                if lower is not None and close < lower:
                    if self._gate_allows_buy(close, ma_val):
                        self.last_level_index += 1
                        signal = True
                        continue

                break

            # 执行调仓
            if signal:
                target_pct = self.last_level_index / (len(self.price_levels) - 1)
                equity = self._current_equity(close)
                target_value = equity * target_pct
                target_shares = self._shares_for_value(target_value, close)

                if target_shares > self.shares:
                    self._execute_trade(
                        date, close, "BUY",
                        target_shares, self.last_level_index, target_pct, idx,
                    )
                elif target_shares < self.shares:
                    if self._can_sell(idx):
                        self._execute_trade(
                            date, close, "SELL",
                            target_shares, self.last_level_index, target_pct, idx,
                        )

            # 记录当日权益
            self._record_equity(date, close)

        # ---- 最后一天强制平仓 ----
        if self.shares > 0:
            final_close = float(df["close"].iloc[-1])
            final_date = df["date"].iloc[-1]
            amount = self.shares * final_close
            comm = calc_commission(amount)
            stamp = calc_stamp_tax(amount)
            transfer = calc_transfer_fee(amount)
            self.cash += amount - comm - stamp - transfer

            self.trades.append(Trade(
                date=final_date, action="SELL(FINAL)", price=final_close,
                shares=self.shares, amount=amount,
                commission=comm, stamp_tax=stamp,
                transfer_fee=transfer,
                grid_level=-1, target_pct=0.0,
                cash_after=self.cash, shares_after=0,
            ))
            self.shares = 0

        # ---- 计算绩效 ----
        return self._compute_metrics(df, config)

    def _compute_metrics(
        self, df: pd.DataFrame, config: GridConfig
    ) -> GridResult:
        """计算回测绩效指标"""
        equity = pd.DataFrame(self.equity_records)
        final_value = self.cash + self.shares * float(df["close"].iloc[-1])
        total_ret = (final_value / config.initial_capital) - 1

        # 年化
        days = len(equity)
        years = max(days / 252, 0.1)
        if total_ret <= -1:
            cagr = -1.0
        else:
            cagr = (1 + total_ret) ** (1 / years) - 1

        # 日收益率序列
        equity["daily_ret"] = equity["total_value"].pct_change()
        daily_rets = equity["daily_ret"].dropna()

        # 夏普比率 (无风险 3%)
        rf_daily = 0.03 / 252
        excess = daily_rets - rf_daily
        sharpe = (
            float(excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 0
            else 0.0
        )

        # 最大回撤
        peak = equity["total_value"].expanding().max()
        drawdown = (equity["total_value"] - peak) / peak
        max_dd = float(drawdown.min())

        # 卡尔玛
        calmar = cagr / abs(max_dd) if max_dd < -0.001 else 0.0

        # 交易统计: 基于实际现金流计算盈亏
        all_trades = self.trades
        sells = [t for t in all_trades if t.action.startswith("SELL")]
        buys = {t.grid_level: t for t in all_trades if t.action in ("INIT_BUY", "BUY")}

        profit_count = 0
        for st in sells:
            for offset in range(1, len(self.price_levels)):
                buy_level = st.grid_level + offset
                if buy_level in buys:
                    if st.price > buys[buy_level].price:
                        profit_count += 1
                    break

        total_sells = len(sells)
        win_rate = (profit_count / total_sells * 100) if total_sells > 0 else 0.0

        # 基准收益
        bench_ret = (
            (float(df["close"].iloc[-1]) - float(df["close"].iloc[0]))
            / float(df["close"].iloc[0])
        )

        return GridResult(
            config=config,
            equity_curve=equity,
            trades=self.trades,
            total_return_pct=total_ret * 100,
            annualized_return_pct=cagr * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar,
            win_rate=win_rate,
            total_trades=len(all_trades),
            profit_trades=profit_count,
            final_value=final_value,
            final_cash=self.cash,
            final_shares=self.shares,
            benchmark_return_pct=bench_ret * 100,
        )


def run_grid_backtest(df: pd.DataFrame, config: GridConfig) -> GridResult:
    """便捷函数: 运行一次网格回测"""
    engine = GridEngine()
    return engine.run(df, config)
