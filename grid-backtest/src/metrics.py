"""
网格交易回测系统 — 绩效指标计算
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .grid_engine import GridResult


def performance_summary(result: GridResult) -> str:
    """生成简洁绩效摘要"""
    return (
        f"\n{'='*65}\n"
        f"  {result.config.symbol} 网格回测结果\n"
        f"{'='*65}\n"
        f"  网格参数: 步长={result.config.grid_step:.1%}  "
        f"网格数={result.config.grid_num}  "
        f"单格仓位={result.config.position_per_grid:.0%}  "
        f"模式={result.config.grid_mode}\n"
        f"  网格线数: {len(result.grid_lines)}  "
        f"价格区间: [{result.grid_lines[0]:.2f} ~ {result.grid_lines[-1]:.2f}]\n"
        f"{'─'*65}\n"
        f"  📈 总收益率:     {result.total_return_pct:>+8.2f}%\n"
        f"  📈 年化收益率:   {result.annualized_return_pct:>+8.2f}%\n"
        f"  📊 夏普比率:     {result.sharpe_ratio:>8.2f}\n"
        f"  📉 最大回撤:     {result.max_drawdown_pct:>8.2f}%\n"
        f"  🎯 卡尔玛比率:   {result.calmar_ratio:>8.2f}\n"
        f"{'─'*65}\n"
        f"  💰 最终资金:     {result.final_value:>10,.2f}\n"
        f"  💵 现金余额:     {result.final_cash:>10,.2f}\n"
        f"  📦 持仓市值:     {result.final_value - result.final_cash:>10,.2f}\n"
        f"  📦 持仓股数:     {result.final_shares:>10,}\n"
        f"{'─'*65}\n"
        f"  🔄 总交易次数:   {result.total_trades:>10}\n"
        f"  ✅ 盈利笔数:     {result.profit_trades:>10}\n"
        f"  🎯 胜率:         {result.win_rate:>9.1f}%\n"
        f"  💹 笔均盈利:     {result.avg_profit_per_trade:>10,.2f}\n"
        f"  📊 网格利用率:   {result.grid_utilization:>9.1f}%\n"
        f"  📊 最大持仓:     {result.max_holding_shares:>10,}股  "
        f"({result.max_holding_value:>10,.2f}元)\n"
        f"{'='*65}\n"
    )


def analyze_monthly(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """月度收益分析"""
    df = equity_curve.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        start_value=("total_value", "first"),
        end_value=("total_value", "last"),
    )
    monthly["ret"] = monthly["end_value"] / monthly["start_value"] - 1
    monthly["cumulative"] = (1 + monthly["ret"]).cumprod()
    return monthly


def analyze_yearly(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """年度收益分析"""
    df = equity_curve.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg(
        start_value=("total_value", "first"),
        end_value=("total_value", "last"),
        max_value=("total_value", "max"),
    )
    yearly["ret"] = yearly["end_value"] / yearly["start_value"] - 1
    yearly["drawdown"] = (yearly["end_value"] - yearly["max_value"]) / yearly["max_value"]
    return yearly


def generate_metrics_dict(result: GridResult) -> Dict:
    """生成指标字典（供报告使用）"""
    return {
        "标的代码": result.config.symbol,
        "网格步长": f"{result.config.grid_step:.1%}",
        "网格数量": result.config.grid_num,
        "单格仓位": f"{result.config.position_per_grid:.0%}",
        "网格模式": result.config.grid_mode,
        "网格线数": len(result.grid_lines),
        "价格下限": f"{result.grid_lines[0]:.2f}",
        "价格上限": f"{result.grid_lines[-1]:.2f}",
        "总收益率": f"{result.total_return_pct:.2f}%",
        "年化收益率": f"{result.annualized_return_pct:.2f}%",
        "夏普比率": f"{result.sharpe_ratio:.2f}",
        "最大回撤": f"{result.max_drawdown_pct:.2f}%",
        "卡尔玛比率": f"{result.calmar_ratio:.2f}",
        "胜率": f"{result.win_rate:.1f}%",
        "总交易次数": result.total_trades,
        "盈利笔数": result.profit_trades,
        "笔均盈利": f"{result.avg_profit_per_trade:.2f}",
        "网格利用率": f"{result.grid_utilization:.1f}%",
        "最终资金": f"{result.final_value:,.2f}",
        "最大持仓市值": f"{result.max_holding_value:,.2f}",
    }
