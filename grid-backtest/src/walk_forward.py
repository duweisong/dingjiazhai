"""
网格交易回测系统 — Walk-Forward 分析 + 动态网格 + MA 趋势闸门

Walk-Forward: 滚动窗口优化 → 验证参数在样本外的时间稳定性。
动态网格: 定期根据最新价格重新居中网格线。
MA 闸门: 用移动均线过滤单边趋势市，仅在震荡市启用网格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time

import numpy as np
import pandas as pd

from .config import GridConfig
from .grid_engine import GridBacktestEngine, GridResult
from .optimizer import GridOptimizer


# ============================================================
# Walk-Forward 分析
# ============================================================

@dataclass(frozen=True)
class WFWindow:
    """单个 Walk-Forward 窗口结果"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict
    train_score: float
    test_return: float
    test_sharpe: float
    test_max_dd: float
    test_trades: int


@dataclass
class WalkForwardResult:
    """Walk-Forward 分析完整结果"""
    symbol: str
    windows: List[WFWindow]
    total_return_pct: float
    avg_sharpe: float
    avg_max_dd: float
    param_stability: Dict[str, float]  # 参数 → 被选中的频率
    is_robust: bool  # 参数是否稳健


class WalkForwardAnalyzer:
    """
    Walk-Forward 滚动窗口分析。

    将数据切分为多个 (训练窗, 测试窗) 对，
    在每个训练窗内优化参数，在测试窗内验证，
    最终评估参数的时间稳定性。
    """

    def __init__(self, train_months: int = 12, test_months: int = 3,
                 step_months: int = 3):
        """
        Args:
            train_months: 训练窗口长度（月）
            test_months: 测试窗口长度（月）
            step_months: 滚动步长（月）
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def _make_windows(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, ...]]:
        """生成滚动窗口日期边界"""
        dates = df["date"].sort_values()
        start = dates.iloc[0]
        end = dates.iloc[-1]
        total_months = (end.year - start.year) * 12 + (end.month - start.month)

        windows = []
        cursor = start
        while True:
            train_end = cursor + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)
            if test_end > end:
                break

            windows.append((
                cursor,
                train_end,
                train_end + pd.Timedelta(days=1),
                test_end,
            ))
            cursor += pd.DateOffset(months=self.step_months)

        return windows

    def run(self, df: pd.DataFrame, base_config: GridConfig,
            param_space: Optional[Dict[str, List]] = None,
            space_type: str = "coarse",
            ) -> WalkForwardResult:
        """
        执行 Walk-Forward 分析。

        Args:
            df: 日线数据
            base_config: 基础配置
            param_space: 参数空间
            space_type: 参数空间精度

        Returns:
            WalkForwardResult
        """
        if param_space is None:
            from .grid_strategy import get_param_space
            param_space = get_param_space(space_type)

        windows = self._make_windows(df)
        if len(windows) < 2:
            raise ValueError(f"数据不足以做 Walk-Forward（仅{len(windows)}个窗口）")

        print(f"  Walk-Forward: {len(windows)} 窗口 "
              f"(训练{self.train_months}月/测试{self.test_months}月/步长{self.step_months}月)")

        optimizer = GridOptimizer(scoring="calmar_sharpe")
        engine = GridBacktestEngine()
        wf_windows: List[WFWindow] = []
        param_counts: Dict[str, int] = {}

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            tr_s_str = tr_s.strftime("%Y-%m-%d")
            tr_e_str = tr_e.strftime("%Y-%m-%d")
            te_s_str = te_s.strftime("%Y-%m-%d")
            te_e_str = te_e.strftime("%Y-%m-%d")

            # 训练
            train_df = df[(df["date"] >= tr_s) & (df["date"] <= tr_e)]
            if len(train_df) < 60:
                continue

            train_cfg = base_config.with_updates(
                start_date=tr_s_str, end_date=tr_e_str,
            )

            try:
                report = optimizer.optimize(train_df, train_cfg,
                                             param_space=param_space, top_n=1)
                best = report.best
                best_params = {
                    "grid_step": best.config.grid_step,
                    "grid_num": best.config.grid_num,
                    "position_per_grid": best.config.position_per_grid,
                    "grid_mode": best.config.grid_mode,
                }
            except Exception:
                continue

            # 记录参数选择
            param_key = (f"s={best_params['grid_step']:.3f}"
                         f"_n={best_params['grid_num']}"
                         f"_p={best_params['position_per_grid']:.2f}")
            param_counts[param_key] = param_counts.get(param_key, 0) + 1

            # 测试
            test_df = df[(df["date"] >= te_s) & (df["date"] <= te_e)]
            if len(test_df) < 20:
                continue

            test_cfg = best.config.with_updates(
                start_date=te_s_str, end_date=te_e_str,
            )

            try:
                test_result = engine.run(test_df, test_cfg)
            except Exception:
                continue

            wf_windows.append(WFWindow(
                window_id=i + 1,
                train_start=tr_s_str, train_end=tr_e_str,
                test_start=te_s_str, test_end=te_e_str,
                best_params=best_params,
                train_score=best.score,
                test_return=test_result.total_return_pct,
                test_sharpe=test_result.sharpe_ratio,
                test_max_dd=test_result.max_drawdown_pct,
                test_trades=test_result.total_trades,
            ))

        if not wf_windows:
            raise ValueError("Walk-Forward 无有效窗口")

        # 汇总
        cum_ret = np.prod([1 + w.test_return / 100 for w in wf_windows]) - 1
        avg_sharpe = np.mean([w.test_sharpe for w in wf_windows])
        avg_dd = np.mean([w.test_max_dd for w in wf_windows])

        # 参数稳定性：最常用参数的出现频率
        total_selections = sum(param_counts.values())
        param_stability = {k: v / total_selections for k, v
                           in sorted(param_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:5]}

        # 稳健性判断：最常用参数是否占 >40%
        top_freq = max(param_stability.values()) if param_stability else 0
        is_robust = top_freq > 0.4 and len(wf_windows) >= 3

        return WalkForwardResult(
            symbol=base_config.symbol,
            windows=wf_windows,
            total_return_pct=cum_ret * 100,
            avg_sharpe=avg_sharpe,
            avg_max_dd=avg_dd,
            param_stability=param_stability,
            is_robust=is_robust,
        )


# ============================================================
# 动态网格引擎
# ============================================================

class DynamicGridEngine(GridBacktestEngine):
    """
    动态网格引擎 — 定期重新居中网格线。

    当价格偏离原始网格中心超过阈值时，自动重新计算网格区间，
    确保网格始终围绕当前价格水平。
    """

    def __init__(self, recenter_threshold: float = 0.10,
                 recenter_freq_days: int = 60):
        """
        Args:
            recenter_threshold: 价格偏离中心超过此比例触发重新居中
            recenter_freq_days: 最小重新居中间隔（交易日）
        """
        super().__init__()
        self.recenter_threshold = recenter_threshold
        self.recenter_freq_days = recenter_freq_days
        self._last_recenter_idx: int = 0
        self._initial_price_mid: float = 0.0

    def run(self, df: pd.DataFrame, config: GridConfig) -> GridResult:
        """运行动态网格回测"""
        self._config = config
        self.cash = config.initial_capital
        self.shares = 0
        self.trades = []
        self.equity_records = []
        self.bought_at_grid = {}
        self.shares_at_grid = {}
        self.t1_lock = {}
        self._last_recenter_idx = 0

        df = df.copy()
        df = df[(df["date"] >= pd.Timestamp(config.start_date))
                & (df["date"] <= pd.Timestamp(config.end_date))]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("回测区间内无数据")

        # 初始价格区间（基于前 20 日均价）
        init_mid = float(df["close"].iloc[:20].mean())
        margin = max(config.position_per_grid * 3, 0.10)
        price_low = init_mid * (1 - margin)
        price_high = init_mid * (1 + margin)
        self._initial_price_mid = init_mid

        self.grid_lines = self._build_lines(price_low, price_high, config)

        # 初始化建仓
        prev_close = float(df["close"].iloc[0])
        init_grid = np.searchsorted(self.grid_lines, prev_close, side="right") - 1
        init_grid = max(0, min(init_grid, len(self.grid_lines) - 2))
        for g in range(init_grid + 1):
            if self.cash > config.initial_capital * 0.1:
                self._execute_buy(df["date"].iloc[0], self.grid_lines[g], g)

        for idx in range(len(df)):
            row = df.iloc[idx]
            date = row["date"]
            today_high = float(row["high"])
            today_low = float(row["low"])
            today_close = float(row["close"])

            if self.t1_lock and idx > 0:
                self._release_t1()

            # 动态重新居中检查
            self._maybe_recenter(today_close, config, idx)

            # 穿越检测
            self._check_grid_cross(prev_close, today_low, today_high,
                                    today_close, date)

            # 止损
            stop_price = self.grid_lines[0] * (1 - config.stop_loss_pct)
            if today_low <= stop_price and self.shares > 0:
                self._emergency_sell(date, max(today_low, stop_price))

            self._record_equity(date, today_close)
            prev_close = today_close

        return self._build_result(config, df)

    def _maybe_recenter(self, current_price: float, config: GridConfig, idx: int):
        """检查是否需要重新居中网格"""
        grid_mid = (self.grid_lines[0] + self.grid_lines[-1]) / 2
        deviation = abs(current_price - grid_mid) / grid_mid

        if (deviation > self.recenter_threshold
                and (idx - self._last_recenter_idx) > self.recenter_freq_days):
            # 保存持仓状态
            old_bought = dict(self.bought_at_grid)
            old_shares_at = dict(self.shares_at_grid)

            # 新网格中心 = 当前价格
            new_mid = current_price
            new_low = new_mid * (self.grid_lines[0] / grid_mid)
            new_high = new_mid * (self.grid_lines[-1] / grid_mid)

            self.grid_lines = self._build_lines(new_low, new_high, config)
            self._last_recenter_idx = idx

    def _build_lines(self, low, high, config):
        """构建网格线（兼容 GridConfig）"""
        from .grid_engine import build_grid_lines
        return build_grid_lines(low, high, config.grid_step, config.grid_mode)

    def _emergency_sell(self, date, price):
        """紧急清仓"""
        from .grid_engine import Trade, COMMISSION_RATE, STAMP_TAX_RATE, MIN_COMMISSION
        while self.shares >= 100:
            shares = (self.shares // 100) * 100
            if shares < 100:
                shares = self.shares
            amount = shares * price
            comm = max(amount * COMMISSION_RATE, MIN_COMMISSION)
            stamp = amount * STAMP_TAX_RATE
            self.cash += amount - comm - stamp
            self.shares -= shares
            self.trades.append(Trade(
                date=date, action="SELL(STOP)", price=price,
                shares=shares, amount=amount, commission=comm,
                stamp_tax=stamp, grid_level=-1,
                cash_after=self.cash, shares_after=self.shares,
            ))
        self.bought_at_grid.clear()
        self.shares_at_grid.clear()
        self.t1_lock.clear()

    def _build_result(self, config, df):
        """构造结果（复用父类逻辑）"""
        from .grid_engine import GridResult
        equity = pd.DataFrame(self.equity_records)

        final_close = float(equity["close"].iloc[-1])
        final_value = self.cash + self.shares * final_close
        total_ret = (final_value / config.initial_capital) - 1

        days = len(equity)
        years = days / 252
        cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

        equity["daily_ret"] = equity["total_value"].pct_change()
        daily_rets = equity["daily_ret"].dropna()
        sharpe = (float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
                  if daily_rets.std() > 0 else 0.0)

        peak = equity["total_value"].expanding().max()
        max_dd = float(((equity["total_value"] - peak) / peak).min())
        calmar = cagr / abs(max_dd) if max_dd < -0.001 else 0.0

        sell_trades = [t for t in self.trades if t.action.startswith("SELL")]
        profit_trades = sum(1 for t in sell_trades
                            if any(b.price < t.price
                                   for b in self.trades
                                   if b.action == "BUY" and b.date <= t.date))
        win_rate = profit_trades / len(sell_trades) * 100 if sell_trades else 0

        total_intervals = max(len(self.grid_lines) - 1, 1)
        used = len(set(t.grid_level for t in self.trades if t.grid_level >= 0))
        grid_util = min(used / total_intervals, 1.0)

        max_s = max((r["shares"] for r in self.equity_records), default=0)
        max_v = max((r["shares"] * r["close"] for r in self.equity_records), default=0)

        return GridResult(
            config=config, equity_curve=equity, trades=self.trades,
            grid_lines=self.grid_lines,
            total_return_pct=total_ret * 100,
            annualized_return_pct=cagr * 100,
            sharpe_ratio=sharpe, max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar, win_rate=win_rate,
            total_trades=len(self.trades), profit_trades=profit_trades,
            avg_profit_per_trade=0, max_holding_shares=max_s,
            max_holding_value=max_v, grid_utilization=grid_util * 100,
            final_cash=self.cash, final_shares=self.shares,
            final_value=final_value,
        )


# ============================================================
# MA 趋势闸门
# ============================================================

class GatedGridEngine(DynamicGridEngine):
    """
    MA 趋势闸门网格引擎。

    当价格在均线之下时暂停买入（单边下跌保护），
    仅在价格 > MA 时启用网格，避免在趋势市中持续接飞刀。
    """

    def __init__(self, ma_period: int = 60,
                 recenter_threshold: float = 0.10,
                 recenter_freq_days: int = 60):
        """
        Args:
            ma_period: 均线周期（默认 60 日 = 季度线）
            recenter_threshold: 动态网格重居中阈值
            recenter_freq_days: 重居中最小间隔
        """
        super().__init__(recenter_threshold, recenter_freq_days)
        self.ma_period = ma_period
        self._ma: Optional[np.ndarray] = None

    def run(self, df: pd.DataFrame, config: GridConfig) -> GridResult:
        """运行带 MA 闸门的网格回测"""
        # 预计算 MA
        df = df.copy()
        df["ma"] = df["close"].rolling(self.ma_period).mean()
        self._ma = df["ma"].values

        # 调用父亲的 run（覆盖 _check_grid_cross 逻辑）
        self._config = config
        self.cash = config.initial_capital
        self.shares = 0
        self.trades = []
        self.equity_records = []
        self.bought_at_grid = {}
        self.shares_at_grid = {}
        self.t1_lock = {}
        self._last_recenter_idx = 0

        df = df[(df["date"] >= pd.Timestamp(config.start_date))
                & (df["date"] <= pd.Timestamp(config.end_date))]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("回测区间内无数据")

        init_mid = float(df["close"].iloc[:20].mean())
        margin = max(config.position_per_grid * 3, 0.10)
        price_low = init_mid * (1 - margin)
        price_high = init_mid * (1 + margin)

        self.grid_lines = self._build_lines(price_low, price_high, config)

        prev_close = float(df["close"].iloc[0])
        # 仅在价格>MA时初始建仓
        init_ma = df["ma"].iloc[0]
        if pd.notna(init_ma) and prev_close > init_ma:
            init_grid = np.searchsorted(self.grid_lines, prev_close, side="right") - 1
            init_grid = max(0, min(init_grid, len(self.grid_lines) - 2))
            for g in range(init_grid + 1):
                if self.cash > config.initial_capital * 0.1:
                    self._execute_buy(df["date"].iloc[0], self.grid_lines[g], g)

        gate_closed_days = 0

        for idx in range(len(df)):
            row = df.iloc[idx]
            date = row["date"]
            today_high = float(row["high"])
            today_low = float(row["low"])
            today_close = float(row["close"])
            today_ma = df["ma"].iloc[idx]

            if self.t1_lock and idx > 0:
                self._release_t1()

            # MA 闸门：价格<MA 时只卖不买
            gate_open = pd.isna(today_ma) or today_close >= today_ma
            if not gate_open:
                gate_closed_days += 1

            # 卖出始终允许
            self._check_sell_only(prev_close, today_high, date)

            # 买入仅在闸门开启时
            if gate_open:
                self._check_buy_only(prev_close, today_low, date)
                self._maybe_recenter(today_close, config, idx)

            # 止损
            stop_price = self.grid_lines[0] * (1 - config.stop_loss_pct)
            if today_low <= stop_price and self.shares > 0:
                self._emergency_sell(date, max(today_low, stop_price))

            self._record_equity(date, today_close)
            prev_close = today_close

        result = self._build_result(config, df)
        # 添加闸门信息
        object.__setattr__(result, "gate_closed_days", gate_closed_days)
        object.__setattr__(result, "gate_closed_pct",
                           gate_closed_days / len(df) * 100 if len(df) > 0 else 0)
        return result

    def _check_sell_only(self, prev_close, today_high, date):
        """只检查卖出穿越"""
        for i in range(len(self.grid_lines)):
            line = self.grid_lines[i]
            if prev_close <= line <= today_high:
                for j in range(i - 1, -1, -1):
                    if j in self.shares_at_grid and self.shares_at_grid[j] > 0:
                        self._execute_sell(date, line, i, j)
                        break

    def _check_buy_only(self, prev_close, today_low, date):
        """只检查买入穿越"""
        for i in range(len(self.grid_lines)):
            line = self.grid_lines[i]
            if today_low <= line <= prev_close:
                self._execute_buy(date, line, i)
