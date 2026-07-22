"""
网格交易回测系统 — 报告生成

生成 HTML 报告和 Excel 导出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from .grid_engine import GridResult
from .optimizer import OptimizationReport
from .metrics import generate_metrics_dict

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_html_report(result: GridResult,
                         monthly: pd.DataFrame = None,
                         yearly: pd.DataFrame = None,
                         save: bool = True) -> str:
    """生成 HTML 报告"""
    cfg = result.config
    m = generate_metrics_dict(result)

    metrics_rows = ""
    for key, val in m.items():
        metrics_rows += f"<tr><td>{key}</td><td>{val}</td></tr>\n"

    # 交易列表
    trade_rows = ""
    for t in result.trades[-50:]:  # 最近 50 笔
        color = "#22c55e" if t.action.startswith("SELL") else "#ef4444"
        trade_rows += (
            f"<tr>"
            f"<td>{t.date.strftime('%Y-%m-%d')}</td>"
            f"<td style='color:{color}'>{t.action}</td>"
            f"<td>{t.price:.2f}</td>"
            f"<td>{t.shares:,}</td>"
            f"<td>{t.amount:,.0f}</td>"
            f"<td>{t.grid_level}</td>"
            f"</tr>\n"
        )

    # 月度收益表
    monthly_rows = ""
    if monthly is not None and not monthly.empty:
        for idx, row in monthly.iterrows():
            ret_color = "#22c55e" if row["ret"] >= 0 else "#ef4444"
            monthly_rows += (
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td style='color:{ret_color}'>{row['ret']:+.2%}</td>"
                f"<td>{row['cumulative']:.4f}</td>"
                f"</tr>\n"
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>网格回测报告 — {cfg.symbol}</title>
<style>
  :root {{ --bg: #0f172a; --surface: #1e293b; --text: #e2e8f0;
          --accent: #3b82f6; --green: #22c55e; --red: #ef4444; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.3rem; margin: 1.5rem 0 0.8rem; color: var(--accent); }}
  .subtitle {{ color: #94a3b8; margin-bottom: 1.5rem; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:1rem;
           background: var(--surface); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #334155; font-weight: 600; color: #94a3b8; font-size:0.85rem; }}
  td {{ font-size: 0.9rem; }}
  .summary-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
                     gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--surface); padding: 1rem; border-radius: 8px;
           border-left: 4px solid var(--accent); }}
  .card .label {{ color: #94a3b8; font-size:0.8rem; }}
  .card .value {{ font-size:1.5rem; font-weight:700; margin-top:0.3rem; }}
  .card .value.positive {{ color: var(--green); }}
  .card .value.negative {{ color: var(--red); }}
  footer {{ margin-top:2rem; color:#64748b; font-size:0.8rem; text-align:center; }}
</style>
</head>
<body>

<h1>📊 网格交易回测报告</h1>
<p class="subtitle">标的: {cfg.symbol}.{cfg.market} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="summary-cards">
  <div class="card">
    <div class="label">总收益率</div>
    <div class="value {'positive' if result.total_return_pct>=0 else 'negative'}">
      {result.total_return_pct:+.2f}%</div>
  </div>
  <div class="card">
    <div class="label">年化收益率</div>
    <div class="value {'positive' if result.annualized_return_pct>=0 else 'negative'}">
      {result.annualized_return_pct:+.2f}%</div>
  </div>
  <div class="card">
    <div class="label">夏普比率</div>
    <div class="value">{result.sharpe_ratio:.2f}</div>
  </div>
  <div class="card">
    <div class="label">最大回撤</div>
    <div class="value negative">{result.max_drawdown_pct:.2f}%</div>
  </div>
  <div class="card">
    <div class="label">胜率</div>
    <div class="value">{result.win_rate:.1f}%</div>
  </div>
  <div class="card">
    <div class="label">最终价值</div>
    <div class="value">{result.final_value:,.0f}</div>
  </div>
</div>

<h2>📋 完整指标</h2>
<table>{metrics_rows}</table>

<h2>📈 网格参数</h2>
<table>
  <tr><th>参数</th><th>值</th></tr>
  <tr><td>步长</td><td>{cfg.grid_step:.1%}</td></tr>
  <tr><td>网格数</td><td>{cfg.grid_num}</td></tr>
  <tr><td>单格仓位</td><td>{cfg.position_per_grid:.0%}</td></tr>
  <tr><td>网格模式</td><td>{cfg.grid_mode}</td></tr>
  <tr><td>价格区间</td><td>{result.grid_lines[0]:.2f} ~ {result.grid_lines[-1]:.2f}</td></tr>
  <tr><td>网格线数</td><td>{len(result.grid_lines)}</td></tr>
  <tr><td>止损线</td><td>{cfg.stop_loss_pct:.0%}</td></tr>
</table>

<h2>🔄 最近交易记录</h2>
<table>
  <tr><th>日期</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>网格</th></tr>
  {trade_rows}
</table>

<h2>📅 月度收益</h2>
<table>
  <tr><th>月份</th><th>收益率</th><th>累计净值</th></tr>
  {monthly_rows}
</table>

<footer>
  Grid Backtest System · Generated by grid-backtest engine ·
  数据源: efinance
</footer>

</body>
</html>"""

    path = OUTPUT_DIR / f"grid_report_{cfg.symbol}.html"
    if save:
        path.write_text(html, encoding="utf-8")
    return str(path)


def export_to_excel(results: List[GridResult],
                    save: bool = True) -> str:
    """导出多标的回测结果到 Excel"""
    rows = []
    for r in results:
        rows.append({
            "标的": r.config.symbol,
            "步长": f"{r.config.grid_step:.1%}",
            "网格数": r.config.grid_num,
            "单格仓位": f"{r.config.position_per_grid:.0%}",
            "网格模式": r.config.grid_mode,
            "总收益率%": round(r.total_return_pct, 2),
            "年化收益率%": round(r.annualized_return_pct, 2),
            "夏普比率": round(r.sharpe_ratio, 2),
            "最大回撤%": round(r.max_drawdown_pct, 2),
            "卡尔玛比率": round(r.calmar_ratio, 2),
            "胜率%": round(r.win_rate, 1),
            "总交易数": r.total_trades,
            "盈利笔数": r.profit_trades,
            "网格利用率%": round(r.grid_utilization, 1),
            "最终价值": round(r.final_value, 2),
        })

    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "grid_results_summary.xlsx"
    if save:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="回测汇总", index=False)
            # 交易明细单独sheet
            for r in results:
                if r.trades:
                    tdf = pd.DataFrame([{
                        "日期": t.date.strftime("%Y-%m-%d"),
                        "方向": t.action,
                        "价格": t.price,
                        "股数": t.shares,
                        "金额": t.amount,
                        "佣金": t.commission,
                        "印花税": t.stamp_tax,
                        "网格线": t.grid_level,
                    } for t in r.trades])
                    sheet_name = f"交易_{r.config.symbol}"[:31]
                    tdf.to_excel(writer, sheet_name=sheet_name, index=False)
    return str(path)


def generate_optimization_report(report: OptimizationReport,
                                  save: bool = True) -> str:
    """生成优化报告（Markdown）"""
    lines = [
        f"# 网格参数优化报告 — {report.symbol}",
        "",
        f"**扫描组合**: {report.total_combinations} 组  |  "
        f"**有效结果**: {report.valid_results} 组  |  "
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 🏆 最优参数",
        "",
    ]

    best = report.best
    if best:
        cfg = best.config
        r = best.result
        lines += [
            f"| 参数 | 值 |",
            f"|------|-----|",
            f"| 步长 | {cfg.grid_step:.1%} |",
            f"| 网格数 | {cfg.grid_num} |",
            f"| 单格仓位 | {cfg.position_per_grid:.0%} |",
            f"| 网格模式 | {cfg.grid_mode} |",
            f"| 评分 | {best.score:.3f} |",
            f"| 年化收益率 | {r.annualized_return_pct:+.2f}% |",
            f"| 夏普比率 | {r.sharpe_ratio:.2f} |",
            f"| 最大回撤 | {r.max_drawdown_pct:.2f}% |",
            f"| 卡尔玛比率 | {r.calmar_ratio:.2f} |",
            f"| 胜率 | {r.win_rate:.1f}% |",
            f"| 网格利用率 | {r.grid_utilization:.1f}% |",
            "",
        ]

    lines += ["## 📊 Top 10 参数组合", ""]
    lines += [
        "| 排名 | 步长 | 网格数 | 仓位/格 | 模式 | 年化收益 | 夏普 | 回撤 | 评分 |",
        "|------|------|--------|---------|------|----------|------|------|------|",
    ]
    for opt_r in report.top_n:
        c = opt_r.config
        r = opt_r.result
        lines.append(
            f"| {opt_r.rank} | {c.grid_step:.1%} | {c.grid_num} | "
            f"{c.position_per_grid:.0%} | {c.grid_mode} | "
            f"{r.annualized_return_pct:+.1f}% | {r.sharpe_ratio:.2f} | "
            f"{r.max_drawdown_pct:.1f}% | {opt_r.score:.3f} |"
        )

    lines += ["", "## 📈 参数敏感性", ""]
    for param, agg in report.param_heatmap.items():
        lines += [f"### {param}", ""]
        lines += ["| 值 | 样本数 | 平均评分 | 平均收益 | 平均夏普 |",
                   "|----|--------|----------|----------|----------|"]
        for val, stats in sorted(agg.items()):
            lines.append(
                f"| {val} | {stats['count']} | {stats['avg_score']:.3f} | "
                f"{stats['avg_ret']:+.1f}% | {stats['avg_sharpe']:.2f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    path = OUTPUT_DIR / f"optimization_{report.symbol}.md"
    if save:
        path.write_text(text, encoding="utf-8")
    return str(path)
