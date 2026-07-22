"""
A股网格交易回测 — 可视化模块

生成权益曲线、回撤曲线、交易分布等图表。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grid_engine import GridResult

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _ensure_matplotlib():
    """延迟导入 matplotlib"""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError:
        raise ImportError("matplotlib 未安装，请运行: pip install matplotlib")


def plot_equity_curve(
    result: GridResult,
    save: bool = True,
    show: bool = False,
) -> str | None:
    """
    绘制权益曲线 + 回撤曲线。
    """
    plt = _ensure_matplotlib()

    equity = result.equity_curve
    if equity.empty:
        print("权益曲线数据为空，跳过绘图")
        return None

    equity["date"] = pd.to_datetime(equity["date"])
    equity.set_index("date", inplace=True)

    peak = equity["total_value"].expanding().max()
    drawdown = (equity["total_value"] - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(equity.index, equity["total_value"], color="#2e86de", linewidth=1.2, label="策略权益")
    ax1.axhline(y=result.config.initial_capital, color="gray",
                linestyle="--", alpha=0.5, label="初始资金")
    ax1.set_ylabel("权益 (元)", fontsize=11)
    ax1.set_title(
        f"{result.config.symbol} 网格策略回测 | "
        f"总收益 {result.total_return_pct:+.2f}% | "
        f"夏普 {result.sharpe_ratio:.2f} | "
        f"回撤 {result.max_drawdown_pct:.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)

    ax2.fill_between(equity.index, 0, drawdown, color="#ee5a24", alpha=0.4)
    ax2.plot(equity.index, drawdown, color="#ee5a24", linewidth=0.8)
    ax2.set_ylabel("回撤 (%)", fontsize=11)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / f"{result.config.symbol}_equity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[图表] 已保存: {path}")
        plt.close(fig)
        return str(path)

    if show:
        plt.show()

    plt.close(fig)
    return None


def plot_trade_distribution(
    result: GridResult,
    save: bool = True,
    show: bool = False,
) -> str | None:
    """
    绘制交易买卖点分布图。
    """
    plt = _ensure_matplotlib()

    if not result.trades:
        print("无交易记录，跳过交易分布图")
        return None

    trades_data = [
        {"date": t.date, "price": t.price, "action": t.action}
        for t in result.trades
    ]
    trades_df = pd.DataFrame(trades_data)
    trades_df["date"] = pd.to_datetime(trades_df["date"])

    equity = result.equity_curve
    equity["date"] = pd.to_datetime(equity["date"])

    buys = trades_df[trades_df["action"].isin(["INIT_BUY", "BUY"])]
    sells = trades_df[trades_df["action"].str.startswith("SELL")]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(equity["date"], equity["total_value"], color="#dfe6e9", linewidth=1.5,
            alpha=0.7, label="权益曲线")

    ax_twin = ax.twinx()
    ax_twin.scatter(buys["date"], buys["price"], marker="^", c="#00b894",
                    s=50, alpha=0.8, label="买入", zorder=5)
    ax_twin.scatter(sells["date"], sells["price"], marker="v", c="#d63031",
                    s=50, alpha=0.8, label="卖出", zorder=5)

    ax_twin.set_ylabel("成交价格 (元)", fontsize=11)
    ax.set_ylabel("权益 (元)", fontsize=11)
    ax.set_title(
        f"{result.config.symbol} 交易分布 | 共 {len(trades_df)} 笔",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / f"{result.config.symbol}_trades.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[图表] 已保存: {path}")
        plt.close(fig)
        return str(path)

    if show:
        plt.show()

    plt.close(fig)
    return None


def plot_all(result: GridResult, show: bool = False) -> dict[str, str]:
    """生成所有图表"""
    paths: dict[str, str] = {}
    p1 = plot_equity_curve(result, save=True, show=show)
    if p1:
        paths["equity"] = p1
    p2 = plot_trade_distribution(result, save=True, show=show)
    if p2:
        paths["trades"] = p2
    return paths
