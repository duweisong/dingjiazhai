"""
滚动窗口优化 & 自适应仓位管理

1. WalkForwardOptimizer  - 滚动窗口参数优化，避免静态参数过拟合
2. AdaptivePositionSizer - Kelly公式 + 波动率自适应仓位
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from copy import deepcopy

from engine import BacktestEngine, BacktestResult
from strategies import get_strategy


class WalkForwardOptimizer:
    """
    滚动窗口优化器

    做法:
      将历史数据分为多个窗口，每个窗口:
        训练期(如2年) → 参数优化
        测试期(如1年) → 用最优参数交易
      下一窗口向前滚动，每次重新优化参数

    优势:
      - 模拟真实交易中的参数自适应
      - 避免静态参数在历史全局最优但未来失效
    """

    def __init__(
        self,
        train_years: float = 2.0,
        test_years: float = 1.0,
        step_years: float = 0.5,
    ):
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years
        self.engine = BacktestEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        param_grid: Dict[str, List],
        fixed_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        执行滚动窗口优化

        Returns:
            dict with:
                windows: 每个窗口的详情
                combined_equity: 拼接的权益曲线
                total_return: 总收益
                total_trades: 总交易数
        """
        dates = df["date"].values
        start_date = dates[0]
        end_date = dates[-1]
        total_days = (end_date - start_date).astype('timedelta64[D]').astype(int)

        train_days = int(self.train_years * 252)
        test_days = int(self.test_years * 252)
        step_days = int(self.step_years * 252)

        windows = []
        all_trades = []
        all_equity_segments = []

        # 滑动窗口
        window_start_day = 0  # 从数据起始开始

        window_num = 0
        while window_start_day + train_days + test_days <= len(df):
            window_num += 1

            # 训练集
            train_end = window_start_day + train_days
            train_df = df.iloc[window_start_day:train_end].copy()

            # 测试集
            test_end = min(train_end + test_days, len(df))
            test_df = df.iloc[train_end:test_end].copy()

            if len(test_df) < 60:  # 最少3个月测试
                break

            # ---- 参数网格搜索 ----
            best_params = None
            best_score = -999

            keys = list(param_grid.keys())
            values = list(param_grid.values())

            for combo in product(*values):
                strat_params = {}
                other_params = {}
                for k, v in zip(keys, combo):
                    if k in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct"):
                        other_params[k] = v
                    else:
                        strat_params[k] = v

                if fixed_params:
                    strat_params.update(fixed_params)

                sl = other_params.get("stop_loss_pct", 0.06)
                tp = other_params.get("take_profit_pct", 0.18)
                ts = other_params.get("trailing_stop_pct", 0.05)

                try:
                    strategy = get_strategy(strategy_name, **strat_params)
                    result = self.engine.run(train_df, strategy,
                                            stop_loss_pct=sl,
                                            take_profit_pct=tp,
                                            trailing_stop_pct=ts)
                    # 综合评分
                    score = (result.calmar_ratio * 0.4 +
                            result.sharpe_ratio * 0.3 +
                            result.win_rate / 100 * 0.2 +
                            min(result.total_trades / 10, 1.0) * 0.1)
                except Exception:
                    continue

                if score > best_score:
                    best_score = score
                    best_params = {**strat_params,
                                   "stop_loss_pct": sl,
                                   "take_profit_pct": tp,
                                   "trailing_stop_pct": ts}

            if best_params is None:
                window_start_day += step_days
                continue

            # ---- 用最优参数在测试集上交易 ----
            strategy = get_strategy(strategy_name, **{
                k: v for k, v in best_params.items()
                if k not in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct")
            })
            test_result = self.engine.run(
                test_df, strategy,
                stop_loss_pct=best_params.get("stop_loss_pct", 0.06),
                take_profit_pct=best_params.get("take_profit_pct", 0.18),
                trailing_stop_pct=best_params.get("trailing_stop_pct", 0.05),
            )

            train_start_date = df["date"].iloc[window_start_day]
            train_end_date = df["date"].iloc[train_end - 1]
            test_end_date = df["date"].iloc[test_end - 1]

            windows.append({
                "window": window_num,
                "train_period": f"{train_start_date.date()} ~ {train_end_date.date()}",
                "test_period": f"{df['date'].iloc[train_end].date()} ~ {test_end_date.date()}",
                "best_params": best_params,
                "best_train_score": best_score,
                "test_return": test_result.total_return_pct,
                "test_sharpe": test_result.sharpe_ratio,
                "test_max_dd": test_result.max_drawdown_pct,
                "test_trades": test_result.total_trades,
                "test_win_rate": test_result.win_rate,
            })

            all_trades.extend(test_result.trades)
            all_equity_segments.append(test_result.equity_curve)

            print(f"  Window {window_num}: train={windows[-1]['train_period']} | "
                  f"test={windows[-1]['test_period']} | "
                  f"params={best_params} | "
                  f"return={test_result.total_return_pct:+.2f}% | "
                  f"sharpe={test_result.sharpe_ratio:.2f} | "
                  f"trades={test_result.total_trades}")

            # 向前滑动
            window_start_day += step_days

        # ---- 合并结果 ----
        if not windows:
            return {"windows": [], "total_return": 0, "total_trades": 0}

        # 拼接权益曲线 (简单拼接，每段按前一段最终权益rebased)
        if all_equity_segments:
            combined_equity = all_equity_segments[0].copy()
            for seg in all_equity_segments[1:]:
                seg = seg.copy()
                rebase = combined_equity["equity"].iloc[-1] / seg["equity"].iloc[0]
                seg["equity"] = seg["equity"] * rebase
                combined_equity = pd.concat([combined_equity, seg.iloc[1:]])
            combined_equity.reset_index(drop=True, inplace=True)
        else:
            combined_equity = None

        total_return = sum(w["test_return"] for w in windows)

        return {
            "windows": windows,
            "combined_equity": combined_equity,
            "total_return": total_return,
            "total_trades": sum(w["test_trades"] for w in windows),
            "avg_win_rate": np.mean([w["test_win_rate"] for w in windows]),
            "best_overall_params": self._most_common_params(windows),
        }

    def _most_common_params(self, windows: List[Dict]) -> Dict:
        """找出最常出现的最优参数"""
        param_counts = {}
        for w in windows:
            key = str(w["best_params"])
            param_counts[key] = param_counts.get(key, 0) + 1
        best_key = max(param_counts, key=param_counts.get)
        return eval(best_key)


class AdaptivePositionSizer:
    """
    自适应仓位管理器

    三种模式:
      1. Kelly Criterion:  f = (p*b - q) / b  (盈利概率p, 赔率b, 亏损概率q)
      2. Volatility Targeting: 仓位与ATR反比 (高波动→轻仓, 低波动→重仓)
      3. Confidence-based: 根据策略信度动态调整
    """

    def __init__(
        self,
        method: str = "kelly",  # "kelly" | "volatility" | "confidence" | "hybrid"
        base_size: float = 0.8,
        min_size: float = 0.2,
        max_size: float = 0.95,
        kelly_fraction: float = 0.25,  # 半凯利 (保守)
    ):
        self.method = method
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        self.kelly_fraction = kelly_fraction

        # 用于 Kelly 计算的滚动统计
        self._recent_trades = []

    def update_trade(self, is_win: bool, pnl_pct: float):
        """记录交易结果，用于Kelly更新"""
        self._recent_trades.append((is_win, pnl_pct))
        if len(self._recent_trades) > 20:
            self._recent_trades.pop(0)

    def get_size(
        self,
        atr_pct: float = None,
        confidence: float = None,
        regime: str = "ranging",
    ) -> float:
        """
        计算建议仓位比例

        Parameters:
            atr_pct: 当前ATR/价格比率
            confidence: 策略信度 0~1
            regime: 市场状态 'trending_up' | 'trending_down' | 'ranging'
        """
        if self.method == "kelly":
            return self._kelly_size()
        elif self.method == "volatility":
            return self._vol_size(atr_pct)
        elif self.method == "confidence":
            return self._conf_size(confidence)
        elif self.method == "hybrid":
            k = self._kelly_size()
            v = self._vol_size(atr_pct) if atr_pct else self.base_size
            c = self._conf_size(confidence) if confidence else self.base_size
            size = (k * 0.4 + v * 0.3 + c * 0.3)
            return np.clip(size, self.min_size, self.max_size)

        return self.base_size

    def _kelly_size(self) -> float:
        """Kelly公式仓位"""
        if len(self._recent_trades) < 5:
            return self.base_size * 0.5  # 初始保守

        wins = [t for t in self._recent_trades if t[0]]
        losses = [t for t in self._recent_trades if not t[0]]

        p = len(wins) / len(self._recent_trades)
        q = 1 - p

        avg_win = np.mean([t[1] for t in wins]) if wins else 0.01
        avg_loss = abs(np.mean([t[1] for t in losses])) if losses else 0.01

        if avg_loss < 0.001:
            avg_loss = 0.01

        b = avg_win / avg_loss  # 赔率

        # Kelly: f = (p*b - q) / b
        kelly = (p * b - q) / b if b > 0 else 0

        # 分数凯利 (保守)
        size = kelly * self.kelly_fraction

        # 限制范围
        return np.clip(size, self.min_size, self.max_size)

    def _vol_size(self, atr_pct: float) -> float:
        """波动率倒数仓位 (ATR越大,仓位越小)"""
        if atr_pct is None or atr_pct <= 0:
            return self.base_size

        # 目标波动率: 假设3% ATR = base_size
        target_vol = 0.03
        size = self.base_size * (target_vol / atr_pct)
        return np.clip(size, self.min_size, self.max_size)

    def _conf_size(self, confidence: float) -> float:
        """基于信度的仓位"""
        if confidence is None:
            return self.base_size
        size = self.base_size * confidence
        return np.clip(size, self.min_size, self.max_size)
