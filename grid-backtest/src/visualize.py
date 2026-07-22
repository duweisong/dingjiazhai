"""
网格交易回测系统 — 可视化模块

生成权益曲线、网格叠加、交易标记、参数热力图等图表。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .grid_engine import GridResult

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_equity_curve(result: GridResult, save: bool = True) -> str:
    """绘制权益曲线 + 网格覆盖"""
    equity = result.equity_curve
    initial = result.config.initial_capital

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [2, 1, 1]},
                              sharex=True)

    # ── 子图1: 权益曲线 + 交易标记 ──
    ax1 = axes[0]
    ax1.plot(equity["date"], equity["total_value"],
             color="#2563eb", linewidth=1.2, label="权益曲线", zorder=2)
    ax1.axhline(y=initial, color="gray", linestyle="--", alpha=0.5,
                label=f"初始资金 {initial:,.0f}")

    # 标记买入/卖出
    buys = equity[equity["date"].isin(
        [t.date for t in result.trades if t.action == "BUY"])]
    sells = equity[equity["date"].isin(
        [t.date for t in result.trades if t.action.startswith("SELL")])]
    if len(buys) > 0:
        ax1.scatter(buys["date"], buys["total_value"],
                    color="red", marker="^", s=40, alpha=0.7,
                    label=f"买入 ({len(buys)})", zorder=3)
    if len(sells) > 0:
        ax1.scatter(sells["date"], sells["total_value"],
                    color="green", marker="v", s=40, alpha=0.7,
                    label=f"卖出 ({len(sells)})", zorder=3)

    ax1.set_ylabel("权益 (元)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"{result.config.symbol} 网格回测 — "
        f"步长{result.config.grid_step:.1%} "
        f"{result.config.grid_num}格 "
        f"单格{result.config.position_per_grid:.0%} | "
        f"收益{result.total_return_pct:+.1f}% "
        f"夏普{result.sharpe_ratio:.2f}",
        fontsize=13, fontweight="bold",
    )

    # ── 子图2: 回撤 ──
    ax2 = axes[1]
    peak = equity["total_value"].expanding().max()
    dd = (equity["total_value"] - peak) / peak * 100
    ax2.fill_between(equity["date"], 0, dd, color="#ef4444", alpha=0.3)
    ax2.plot(equity["date"], dd, color="#dc2626", linewidth=0.8)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_ylabel("回撤 (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ── 子图3: 日收益率 ──
    ax3 = axes[2]
    daily_ret = equity["daily_ret"].dropna() * 100
    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in daily_ret]
    ax3.bar(daily_ret.index, daily_ret.values, color=colors, width=1, alpha=0.7)
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax3.set_ylabel("日收益 (%)", fontsize=11)
    ax3.set_xlabel("日期", fontsize=11)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / f"grid_equity_{result.config.symbol}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def plot_price_with_grid(result: GridResult, df: pd.DataFrame,
                          save: bool = True) -> str:
    """价格走势 + 网格线叠加"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 价格
    ax.plot(df["date"], df["close"], color="#2563eb", linewidth=0.8,
            alpha=0.7, label="收盘价")

    # 网格线
    for i, line in enumerate(result.grid_lines):
        ls = "-" if i == 0 or i == len(result.grid_lines) - 1 else ":"
        lw = 1.2 if i == 0 or i == len(result.grid_lines) - 1 else 0.5
        ax.axhline(y=line, color="#f59e0b", linestyle=ls,
                   linewidth=lw, alpha=0.5)

    ax.set_ylabel("价格 (元)", fontsize=11)
    ax.set_title(
        f"{result.config.symbol} 价格走势 + 网格线 "
        f"({len(result.grid_lines)} 条, 步长 {result.config.grid_step:.1%})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / f"grid_price_{result.config.symbol}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def plot_optimization_heatmap(report, save: bool = True) -> str:
    """参数优化热力图"""
    heatmaps = report.param_heatmap
    params = list(heatmaps.keys())
    n = len(params)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        data = heatmaps[param]
        labels = list(data.keys())
        scores = [data[k]["avg_score"] for k in labels]

        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{param} → 平均评分", fontsize=11)
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis="y")

        # 数值标注
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.2f}", ha="center", fontsize=7)

    fig.suptitle(f"{report.symbol} 参数优化热力图", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = OUTPUT_DIR / f"grid_heatmap_{report.symbol}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def plot_multi_symbol_comparison(results: dict, save: bool = True) -> str:
    """
    多标的网格回测对比。

    Args:
        results: {symbol: GridResult} 字典
    """
    if not results:
        return ""

    symbols = list(results.keys())
    metrics_list = [
        ("total_return_pct", "总收益率 (%)"),
        ("sharpe_ratio", "夏普比率"),
        ("max_drawdown_pct", "最大回撤 (%)"),
        ("calmar_ratio", "卡尔玛比率"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))

    for ax, (attr, title) in zip(axes.flat, metrics_list):
        values = [getattr(results[s], attr) for s in symbols]
        bars = ax.barh(symbols, values, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, values):
            x_pos = bar.get_width()
            ax.text(x_pos + (max(values) * 0.01), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}" if attr != "sharpe_ratio" else f"{val:.2f}",
                    va="center", fontsize=8)

    fig.suptitle("多标的网格回测对比", fontsize=15, fontweight="bold")
    fig.tight_layout()

    path = OUTPUT_DIR / "grid_multi_comparison.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return str(path)
