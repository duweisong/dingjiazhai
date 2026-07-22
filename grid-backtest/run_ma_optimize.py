#!/usr/bin/env python3
"""
MA 闸门周期优化 — 扫描 MA20~MA120，找到每只 ETF 的最优过滤周期。

核心问题：之前的分析用了固定 MA60，但这真的是最优的吗？
答案：不同 ETF 的最优 MA 周期差异很大，需要逐个优化。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig
from src.grid_strategy import PRESET_STRATEGIES, template_to_config
from src.walk_forward import GatedGridEngine
from src.cache_adapter import get_all_etf_data, SECTOR_ETFS
from src.reporter import export_to_excel


def optimize_ma_for_etf(df: pd.DataFrame, code: str, market: str,
                         ma_periods: list = None) -> list:
    """
    对单只 ETF 扫描不同 MA 周期，返回排序结果。
    """
    if ma_periods is None:
        ma_periods = [20, 30, 40, 50, 60, 80, 100, 120]

    template = [t for t in PRESET_STRATEGIES if t.name == "moderate"][0]
    config = template_to_config(template, code, market)

    results = []
    for ma in ma_periods:
        try:
            engine = GatedGridEngine(ma_period=ma)
            result = engine.run(df, config)
            gate_pct = getattr(result, "gate_closed_pct", 0)
            results.append({
                "ma": ma,
                "sharpe": result.sharpe_ratio,
                "cagr": result.annualized_return_pct,
                "max_dd": result.max_drawdown_pct,
                "calmar": result.calmar_ratio,
                "trades": result.total_trades,
                "gate_closed_pct": gate_pct,
                "final_value": result.final_value,
            })
        except Exception as e:
            results.append({"ma": ma, "sharpe": -999, "cagr": 0,
                            "max_dd": -100, "error": str(e)})

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def main():
    print("加载 ETF 数据...")
    all_data = get_all_etf_data()

    # 取数据充分的 ETF（>200 交易日）
    good_data = {c: d for c, d in all_data.items() if len(d) > 200}
    # 取前 25 只
    top_etfs = dict(sorted(good_data.items(),
                           key=lambda x: len(x[1]), reverse=True)[:25])
    print(f"优化 {len(top_etfs)} 只 ETF 的 MA 闸门周期...")

    ma_periods = [20, 30, 40, 50, 60, 80, 100, 120]
    t0 = time.time()

    best_per_etf = {}
    all_rows = []

    for code, df in sorted(top_etfs.items()):
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"

        results = optimize_ma_for_etf(df, code, market, ma_periods)
        best = results[0]
        baseline = [r for r in results if r["ma"] == 60][0]

        best_per_etf[code] = {
            "name": name,
            "best_ma": best["ma"],
            "best_sharpe": best["sharpe"],
            "baseline_sharpe": baseline["sharpe"],
            "improvement": best["sharpe"] - baseline["sharpe"],
            "best_dd": best["max_dd"],
            "best_gate": best["gate_closed_pct"],
            "all_results": results,
        }

        sector = SECTOR_ETFS.get(code, (code, "未知"))[1][:6]
        arrow = "+" if best["sharpe"] > baseline["sharpe"] else " "
        print(f"  [{code} {name:<12}] MA{best['ma']:>3} (基线MA60)"
              f" 夏普{best['sharpe']:>6.2f} ({arrow}{best['sharpe']-baseline['sharpe']:+.2f})"
              f" 回撤{best['max_dd']:>6.1f}% 闸门{best['gate_closed_pct']:.0f}%"
              f" [{sector}]")

        for r in results:
            all_rows.append({
                "ETF": code,
                "名称": name,
                "MA周期": r["ma"],
                "夏普": round(r["sharpe"], 3),
                "年化%": round(r["cagr"], 1),
                "回撤%": round(r["max_dd"], 1),
                "闸门关闭%": round(r.get("gate_closed_pct", 0), 0),
            })

    elapsed = time.time() - t0

    # 汇总
    print(f"\n{'='*80}")
    print(f"  MA 周期优化汇总 ({len(best_per_etf)} 只 ETF, {elapsed:.0f}s)")
    print(f"{'='*80}")

    # 按最优 MA 分组
    ma_dist = {}
    for code, info in best_per_etf.items():
        ma = info["best_ma"]
        ma_dist[ma] = ma_dist.get(ma, 0) + 1

    print(f"\n  最优 MA 分布:")
    for ma in sorted(ma_dist.keys()):
        bar = "#" * ma_dist[ma]
        print(f"    MA{ma:>3}: {ma_dist[ma]:>2} 只 {bar}")

    # 平均提升
    improvements = [info["improvement"] for info in best_per_etf.values()]
    avg_improve = np.mean(improvements)
    print(f"\n  相对 MA60 平均夏普提升: {avg_improve:+.3f}")
    print(f"  有提升的 ETF: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    # Top improvement ETFs
    print(f"\n  提升最大的 TOP 5:")
    sorted_improve = sorted(best_per_etf.items(),
                            key=lambda x: x[1]["improvement"], reverse=True)
    for code, info in sorted_improve[:5]:
        print(f"    {code} {info['name']:<12}: "
              f"MA60→MA{info['best_ma']} "
              f"夏普 {info['baseline_sharpe']:.2f}→{info['best_sharpe']:.2f} "
              f"({info['improvement']:+.2f})")

    # 导出 Excel
    df_all = pd.DataFrame(all_rows)
    path = Path("output") / "ma_optimization_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="MA扫描结果", index=False)
        # 最优每 ETF
        best_df = pd.DataFrame([{
            "ETF": c, "名称": i["name"],
            "最优MA": i["best_ma"],
            "最优夏普": round(i["best_sharpe"], 3),
            "MA60夏普": round(i["baseline_sharpe"], 3),
            "提升": round(i["improvement"], 3),
            "最优回撤%": round(i["best_dd"], 1),
            "闸门关闭%": round(i["best_gate"], 0),
        } for c, i in best_per_etf.items()])
        best_df = best_df.sort_values("最优夏普", ascending=False)
        best_df.to_excel(w, sheet_name="每ETF最优MA", index=False)
    print(f"\n  结果导出: {path}")

    return best_per_etf


if __name__ == "__main__":
    main()
