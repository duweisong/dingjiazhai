"""
绩效指标分析模块
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_monthly(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """月度收益率分析"""
    df = equity_curve.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        start_equity=("equity", "first"),
        end_equity=("equity", "last"),
    )
    monthly["ret"] = (monthly["end_equity"] - monthly["start_equity"]) / monthly["start_equity"]
    monthly["win"] = monthly["ret"] > 0
    monthly["cumulative"] = (1 + monthly["ret"]).cumprod()
    return monthly


def analyze_yearly(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """年度收益率分析"""
    df = equity_curve.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg(
        start_equity=("equity", "first"),
        end_equity=("equity", "last"),
        high=("equity", "max"),
        low=("equity", "min"),
    )
    yearly["ret"] = (yearly["end_equity"] - yearly["start_equity"]) / yearly["start_equity"]
    yearly["drawdown"] = (yearly["low"] - yearly["high"]) / yearly["high"]
    yearly["cumulative"] = (1 + yearly["ret"]).cumprod()
    return yearly


def trade_analysis(trades: list) -> Dict[str, Any]:
    """交易行为深度分析"""
    if not trades:
        return {}

    holding_days = [t.holding_days for t in trades]
    wins = [t.pnl_pct for t in trades if t.is_win]
    losses = [t.pnl_pct for t in trades if not t.is_win]

    # 连续盈亏统计
    streak = []
    current_streak = 0
    for t in trades:
        if t.is_win:
            if current_streak > 0:
                current_streak += 1
            else:
                if current_streak < 0:
                    streak.append(current_streak)
                current_streak = 1
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                if current_streak > 0:
                    streak.append(current_streak)
                current_streak = -1
    if current_streak != 0:
        streak.append(current_streak)

    return {
        "avg_holding_days": np.mean(holding_days),
        "median_holding_days": np.median(holding_days),
        "max_holding_days": max(holding_days),
        "min_holding_days": min(holding_days),
        "max_win": max(wins) * 100 if wins else 0,
        "max_loss": min(losses) * 100 if losses else 0,
        "max_consecutive_wins": max([s for s in streak if s > 0], default=0),
        "max_consecutive_losses": abs(min([s for s in streak if s < 0], default=0)),
        "avg_max_drawdown_in_trade": np.mean([t.max_drawdown_pct for t in trades]) * 100,
        "avg_max_profit_in_trade": np.mean([t.max_profit_pct for t in trades]) * 100,
    }


def performance_summary(result) -> str:
    """生成可打印的绩效摘要"""
    lines = []
    w = 42
    lines.append("=" * 70)
    lines.append(f"  策略绩效报告: {result.strategy_name}")
    lines.append("=" * 70)
    lines.append(f"  {'回测区间:':<{w}} {result.start_date.date()} ~ {result.end_date.date()}")
    lines.append(f"  {'策略参数:':<{w}} {result.strategy_params}")
    lines.append("-" * 70)
    lines.append(f"  {'初始资金:':<{w}} 元{result.initial_capital:,.0f}")
    lines.append(f"  {'最终权益:':<{w}} 元{result.final_equity:,.0f}")
    lines.append(f"  {'总盈亏:':<{w}} 元{result.total_pnl:+,.0f}")
    lines.append(f"  {'总收益率:':<{w}} {result.total_return_pct:+.2f}%")
    lines.append(f"  {'年化收益率:':<{w}} {result.annualized_return_pct:+.2f}%")
    lines.append(f"  {'基准(买入持有)收益率:':<{w}} {result.benchmark_return:+.2f}%")
    lines.append(f"  {'超额收益:':<{w}} {(result.total_return_pct - result.benchmark_return):+.2f}%")
    lines.append("-" * 70)
    lines.append(f"  {'最大回撤:':<{w}} {result.max_drawdown_pct:.2f}%")
    lines.append(f"  {'最大回撤持续天数:':<{w}} {result.max_drawdown_duration}")
    lines.append(f"  {'夏普比率:':<{w}} {result.sharpe_ratio:.3f}")
    lines.append(f"  {'索提诺比率:':<{w}} {result.sortino_ratio:.3f}")
    lines.append(f"  {'卡尔玛比率:':<{w}} {result.calmar_ratio:.3f}")
    lines.append("-" * 70)
    lines.append(f"  {'总交易次数:':<{w}} {result.total_trades}")
    lines.append(f"  {'盈利次数 / 亏损次数:':<{w}} {result.winning_trades} / {result.losing_trades}")
    lines.append(f"  {'胜率:':<{w}} {result.win_rate:.1f}%")
    lines.append(f"  {'盈亏比 (Profit Factor):':<{w}} {result.profit_factor:.2f}")
    lines.append(f"  {'平均盈利:':<{w}} {result.avg_win_pct:+.2f}%")
    lines.append(f"  {'平均亏损:':<{w}} {result.avg_loss_pct:+.2f}%")
    lines.append(f"  {'平均持仓天数:':<{w}} {result.avg_holding_days:.1f} 天")
    lines.append("=" * 70)
    return "\n".join(lines)
