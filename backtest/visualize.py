"""
可视化模块 —— 回测结果图表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_equity_curve(result, save: bool = True) -> str:
    """绘制权益曲线 + 回撤 + 交易标记"""
    eq = result.equity_curve
    benchmark = eq["close"] / eq["close"].iloc[0] * result.initial_capital

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    # ---- 图1: 权益曲线 ----
    ax1 = axes[0]
    ax1.plot(eq["date"], eq["equity"], color="#1a73e8", linewidth=1.5, label="策略权益")
    ax1.plot(eq["date"], benchmark, color="#999999", linewidth=1, alpha=0.7,
             label=f"买入持有基准")
    ax1.fill_between(eq["date"], result.initial_capital, eq["equity"],
                     where=eq["equity"] >= result.initial_capital,
                     color="#e8f5e9", alpha=0.5)
    ax1.fill_between(eq["date"], result.initial_capital, eq["equity"],
                     where=eq["equity"] < result.initial_capital,
                     color="#ffebee", alpha=0.5)
    ax1.axhline(result.initial_capital, color="gray", linestyle="--", linewidth=0.8)

    # 标记买卖点
    for trade in result.trades:
        ax1.scatter(trade.entry_date, trade.entry_price * trade.shares / result.initial_capital * result.initial_capital,
                    color="red", marker="^", s=60, zorder=5, alpha=0.8)
        if trade.exit_date:
            ax1.scatter(trade.exit_date, trade.exit_price * trade.shares / result.initial_capital * result.initial_capital,
                        color="green" if trade.is_win else "blue",
                        marker="v", s=60, zorder=5, alpha=0.8)

    ax1.set_ylabel("权益 (元)", fontsize=12)
    ax1.set_title(f"{result.strategy_name} — 权益曲线", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"元{x/10000:.0f}万"))
    ax1.grid(True, alpha=0.3)

    # ---- 图2: 回撤 ----
    ax2 = axes[1]
    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax * 100
    ax2.fill_between(eq["date"], 0, drawdown, color="#ff5252", alpha=0.4,
                     label=f"最大回撤: {result.max_drawdown_pct:.1f}%")
    ax2.plot(eq["date"], drawdown, color="#d32f2f", linewidth=0.5)
    ax2.set_ylabel("回撤 (%)", fontsize=12)
    ax2.set_ylim(drawdown.min() * 1.3, 2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # ---- 图3: 每日收益率 ----
    ax3 = axes[2]
    daily_ret = eq["ret"].dropna() * 100
    colors = ["#4caf50" if r >= 0 else "#f44336" for r in daily_ret]
    ax3.bar(eq["date"].iloc[1:], daily_ret, color=colors, width=1, alpha=0.8)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("日收益 (%)", fontsize=12)
    ax3.set_xlabel("日期", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / f"{result.strategy_name}_equity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_trade_analysis(result, save: bool = True) -> str:
    """绘制交易分析图表"""
    trades = result.trades
    if not trades:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 交易盈亏分布 ----
    ax = axes[0, 0]
    pnls = [t.pnl_pct * 100 for t in trades]
    colors = ["#4caf50" if p >= 0 else "#f44336" for p in pnls]
    ax.bar(range(len(trades)), pnls, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("每笔交易盈亏 (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("交易序号")
    ax.set_ylabel("盈亏 (%)")
    ax.grid(True, alpha=0.3)

    # ---- 累计盈亏 ----
    ax = axes[0, 1]
    cum_pnl = np.cumsum(pnls)
    ax.fill_between(range(len(trades)), 0, cum_pnl,
                    where=np.array(cum_pnl) >= 0, color="#e8f5e9", alpha=0.8)
    ax.fill_between(range(len(trades)), 0, cum_pnl,
                    where=np.array(cum_pnl) < 0, color="#ffebee", alpha=0.8)
    ax.plot(range(len(trades)), cum_pnl, color="#1a73e8", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"累计盈亏: {cum_pnl[-1]:+.2f}%", fontsize=12, fontweight="bold")
    ax.set_xlabel("交易序号")
    ax.set_ylabel("累计盈亏 (%)")
    ax.grid(True, alpha=0.3)

    # ---- 持仓天数分布 ----
    ax = axes[1, 0]
    holding = [t.holding_days for t in trades]
    ax.hist(holding, bins=20, color="#1a73e8", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(holding), color="red", linestyle="--",
               label=f"均值: {np.mean(holding):.1f}天")
    ax.axvline(np.median(holding), color="orange", linestyle="--",
               label=f"中位数: {np.median(holding):.0f}天")
    ax.set_title("持仓天数分布", fontsize=12, fontweight="bold")
    ax.set_xlabel("持仓天数")
    ax.set_ylabel("频次")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 胜率饼图 ----
    ax = axes[1, 1]
    wins = result.winning_trades
    losses = result.losing_trades
    sizes = [wins, losses]
    labels = [f"盈利 ({wins}笔)", f"亏损 ({losses}笔)"]
    colors_pie = ["#4caf50", "#f44336"]
    explode = (0.05, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title(f"交易胜率: {result.win_rate:.1f}%", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / f"{result.strategy_name}_trades.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_strategy_comparison(results: list, save: bool = True) -> str:
    """多策略对比图表"""
    if not results:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    colors = ["#1a73e8", "#e91e63", "#4caf50", "#ff9800", "#9c27b0"]

    # ---- 权益曲线对比 ----
    ax = axes[0, 0]
    for i, r in enumerate(results):
        eq = r.equity_curve
        norm_eq = eq["equity"] / eq["equity"].iloc[0]
        ax.plot(eq["date"], norm_eq, color=colors[i % len(colors)], linewidth=1.5,
                label=f"{r.strategy_name}")
    # 基准
    eq0 = results[0].equity_curve
    bench = eq0["close"] / eq0["close"].iloc[0]
    ax.plot(eq0["date"], bench, color="gray", linewidth=1, linestyle="--",
            label="买入持有", alpha=0.7)
    ax.set_title("策略权益曲线对比 (归一化)", fontsize=13, fontweight="bold")
    ax.set_ylabel("归一化权益")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---- 关键指标柱状图 ----
    ax = axes[0, 1]
    names = [r.strategy_name for r in results]
    x = np.arange(len(names))
    width = 0.18

    returns = [r.total_return_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    maxdds = [r.max_drawdown_pct for r in results]

    bars1 = ax.bar(x - width, returns, width, label="总收益率(%)", color="#1a73e8", alpha=0.85)
    bars2 = ax.bar(x, sharpes, width, label="夏普比率", color="#4caf50", alpha=0.85)
    bars3 = ax.bar(x + width, maxdds, width, label="最大回撤(%)", color="#f44336", alpha=0.85)

    # 数值标注
    for bar, val in zip(bars1, returns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("策略核心指标对比", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- 回撤对比 ----
    ax = axes[1, 0]
    for i, r in enumerate(results):
        eq = r.equity_curve
        cummax = eq["equity"].cummax()
        dd = (eq["equity"] - cummax) / cummax * 100
        ax.plot(eq["date"], dd, color=colors[i % len(colors)], linewidth=1,
                label=f"{r.strategy_name} (MDD: {r.max_drawdown_pct:.1f}%)")
    ax.set_title("回撤对比", fontsize=13, fontweight="bold")
    ax.set_ylabel("回撤 (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # ---- 风险收益散点 ----
    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.scatter(r.max_drawdown_pct, r.total_return_pct,
                   s=150, color=colors[i % len(colors)],
                   edgecolors="white", linewidth=1.5, zorder=5,
                   label=f"{r.strategy_name}")
        ax.annotate(r.strategy_name, (r.max_drawdown_pct, r.total_return_pct),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("最大回撤 (%)")
    ax.set_ylabel("总收益率 (%)")
    ax.set_title("风险-收益散点图", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "strategy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def generate_html_report(result, monthly_df, yearly_df, extra: dict = None) -> str:
    """生成 HTML 格式回测报告"""
    eq = result.equity_curve
    start = result.start_date.date()
    end = result.end_date.date()

    trades_html = ""
    for i, t in enumerate(result.trades[:20], 1):  # 最多显示20笔
        tag = "🟢" if t.is_win else "🔴"
        trades_html += f"""
        <tr>
            <td>{i}</td>
            <td>{t.entry_date.date()}</td>
            <td>{t.entry_price:.2f}</td>
            <td>{t.exit_date.date() if t.exit_date else '—'}</td>
            <td>{f'{t.exit_price:.2f}' if t.exit_price else '—'}</td>
            <td>{t.holding_days}</td>
            <td style="color:{'#4caf50' if t.is_win else '#f44336'}">{t.pnl_pct:+.2%}</td>
            <td>{t.reason}</td>
        </tr>"""

    monthly_html = ""
    for m, row in monthly_df.iterrows():
        color = "#4caf50" if row["ret"] > 0 else "#f44336"
        monthly_html += f"""
        <tr>
            <td>{m}</td>
            <td style="color:{color}">{row['ret']:+.2%}</td>
            <td>{row['cumulative']:.3f}</td>
        </tr>"""

    yearly_html = ""
    for y, row in yearly_df.iterrows():
        color = "#4caf50" if row["ret"] > 0 else "#f44336"
        yearly_html += f"""
        <tr>
            <td>{y}</td>
            <td style="color:{color}">{row['ret']:+.2%}</td>
            <td>{row['drawdown']:.2%}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>回测报告: {result.strategy_name}</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; background:#f5f7fa; color:#333; padding:20px; }}
    .container {{ max-width:1200px; margin:0 auto; }}
    h1 {{ color:#1a73e8; margin-bottom:10px; }}
    .subtitle {{ color:#666; margin-bottom:30px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px; }}
    .card {{ background:#fff; border-radius:12px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
    .card h3 {{ font-size:12px; color:#888; text-transform:uppercase; margin-bottom:8px; }}
    .card .value {{ font-size:28px; font-weight:700; }}
    .card .positive {{ color:#4caf50; }}
    .card .negative {{ color:#f44336; }}
    table {{ width:100%; border-collapse:collapse; margin:16px 0; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
    th {{ background:#1a73e8; color:#fff; padding:10px 12px; text-align:left; font-size:13px; }}
    td {{ padding:8px 12px; border-bottom:1px solid #eee; font-size:13px; }}
    tr:hover {{ background:#f5f8ff; }}
    .section {{ margin:30px 0 16px; }}
    .section h2 {{ font-size:18px; color:#1a73e8; border-left:4px solid #1a73e8; padding-left:12px; }}
    .flex {{ display:flex; gap:20px; flex-wrap:wrap; }}
    .flex-item {{ flex:1; min-width:400px; }}
    img {{ max-width:100%; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
    .footer {{ text-align:center; color:#aaa; font-size:12px; margin-top:40px; padding:20px; }}
</style>
</head>
<body>
<div class="container">
    <h1>📊 {result.strategy_name} 回测报告</h1>
    <p class="subtitle">{start} ~ {end} | 联创电子 (002036.SZ) | 波段策略</p>

    <div class="grid">
        <div class="card">
            <h3>总收益率</h3>
            <div class="value {'positive' if result.total_return_pct>0 else 'negative'}">{result.total_return_pct:+.2f}%</div>
        </div>
        <div class="card">
            <h3>年化收益率</h3>
            <div class="value {'positive' if result.annualized_return_pct>0 else 'negative'}">{result.annualized_return_pct:+.2f}%</div>
        </div>
        <div class="card">
            <h3>最大回撤</h3>
            <div class="value negative">{result.max_drawdown_pct:.2f}%</div>
        </div>
        <div class="card">
            <h3>夏普比率</h3>
            <div class="value">{result.sharpe_ratio:.2f}</div>
        </div>
        <div class="card">
            <h3>胜率</h3>
            <div class="value">{result.win_rate:.1f}%</div>
        </div>
        <div class="card">
            <h3>盈亏比</h3>
            <div class="value">{result.profit_factor:.2f}</div>
        </div>
    </div>

    <div class="section"><h2>权益曲线</h2></div>
    <img src="{result.strategy_name}_equity.png" alt="Equity Curve">

    <div class="section"><h2>交易分析</h2></div>
    <img src="{result.strategy_name}_trades.png" alt="Trade Analysis">

    <div class="section"><h2>交易明细 (最近20笔)</h2></div>
    <table>
        <thead><tr><th>#</th><th>买入日</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>持仓天</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>{trades_html}</tbody>
    </table>

    <div class="flex">
        <div class="flex-item">
            <div class="section"><h2>年度表现</h2></div>
            <table>
                <thead><tr><th>年份</th><th>收益率</th><th>年内回撤</th></tr></thead>
                <tbody>{yearly_html}</tbody>
            </table>
        </div>
        <div class="flex-item">
            <div class="section"><h2>月度表现</h2></div>
            <table style="max-height:400px; overflow-y:auto;">
                <thead><tr><th>月份</th><th>收益率</th><th>累计净值</th></tr></thead>
                <tbody>{monthly_html}</tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        本报告由回测引擎自动生成 | 仅供参考，不构成投资建议 | 回测收益不代表未来表现
    </div>
</div>
</body>
</html>"""

    path = OUTPUT_DIR / f"{result.strategy_name}_report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
