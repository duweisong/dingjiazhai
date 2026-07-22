#!/usr/bin/env python3
"""
高级网格策略验证 — Walk-Forward + 动态网格 + MA 闸门对比

对比三种引擎在同一数据上的表现：
1. Static — 固定网格（基线）
2. Dynamic — 动态重新居中
3. Gated — MA 趋势闸门（仅震荡市交易）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig
from src.grid_engine import GridBacktestEngine
from src.grid_strategy import PRESET_STRATEGIES, template_to_config
from src.walk_forward import (
    WalkForwardAnalyzer, DynamicGridEngine, GatedGridEngine,
)
from src.cache_adapter import get_all_etf_data, SECTOR_ETFS
from src.metrics import performance_summary
from src.visualize import plot_equity_curve, plot_multi_symbol_comparison
from src.reporter import export_to_excel

OUTPUT = Path(__file__).parent / "output"


def compare_engines(data_dict: dict, top_n: int = 10):
    """对比三种引擎在同标的上表现"""
    static_engine = GridBacktestEngine()
    dynamic_engine = DynamicGridEngine(recenter_threshold=0.08)
    gated_engine = GatedGridEngine(ma_period=60)

    template = [t for t in PRESET_STRATEGIES if t.name == "moderate"][0]

    print(f"\n{'='*90}")
    print(f"  三种网格引擎对比 (Static vs Dynamic vs Gated)")
    print(f"{'='*90}")

    all_results = {"static": {}, "dynamic": {}, "gated": {}}

    for code, df in sorted(data_dict.items()):
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"
        config = template_to_config(template, code, market)

        row = f"  {code:<8}{name:<14}"

        # Static
        try:
            r_s = static_engine.run(df, config)
            all_results["static"][code] = r_s
            row += f" | S: 夏普{r_s.sharpe_ratio:>5.2f} 回撤{r_s.max_drawdown_pct:>5.1f}%"
        except Exception as e:
            row += f" | S: ERR"

        # Dynamic
        try:
            r_d = dynamic_engine.run(df, config)
            all_results["dynamic"][code] = r_d
            row += f" | D: 夏普{r_d.sharpe_ratio:>5.2f} 回撤{r_d.max_drawdown_pct:>5.1f}%"
        except Exception as e:
            row += f" | D: ERR"

        # Gated
        try:
            r_g = gated_engine.run(df, config)
            all_results["gated"][code] = r_g
            gate_pct = getattr(r_g, "gate_closed_pct", 0)
            row += f" | G: 夏普{r_g.sharpe_ratio:>5.2f} 回撤{r_g.max_drawdown_pct:>5.1f}% 闸门关闭{gate_pct:.0f}%"
        except Exception as e:
            row += f" | G: ERR"

        print(row)

    # 统计汇总
    print(f"\n{'='*90}")
    print(f"  汇总统计")
    print(f"{'='*90}")

    for engine_name, results in all_results.items():
        if not results:
            continue
        sharpes = [r.sharpe_ratio for r in results.values()]
        dds = [r.max_drawdown_pct for r in results.values()]
        rets = [r.annualized_return_pct for r in results.values()]
        pos_sharpe = sum(1 for s in sharpes if s > 0)

        # Gated 附加信息
        extra = ""
        if engine_name == "gated":
            gate_pcts = [getattr(r, "gate_closed_pct", 0) for r in results.values()]
            extra = f" 平均闸门关闭: {np.mean(gate_pcts):.0f}%"

        print(f"  [{engine_name:<8}] 平均夏普: {np.mean(sharpes):.3f}  |  "
              f"中位夏普: {np.median(sharpes):.3f}  |  "
              f"夏普>0: {pos_sharpe}/{len(sharpes)}  |  "
              f"平均回撤: {np.mean(dds):.1f}%  |  "
              f"平均年化: {np.mean(rets):.1f}%  {extra}")

    return all_results


def run_walk_forward(data_dict: dict, top_n: int = 5):
    """对前 N 只 ETF 做 Walk-Forward 分析"""
    print(f"\n{'='*90}")
    print(f"  Walk-Forward 参数稳定性分析")
    print(f"{'='*90}")

    # 取前 N 只（按数据长度排序）
    sorted_etfs = sorted(data_dict.items(),
                         key=lambda x: len(x[1]), reverse=True)[:top_n]

    analyzer = WalkForwardAnalyzer(train_months=12, test_months=3, step_months=3)
    results = {}

    for code, df in sorted_etfs:
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"
        base_cfg = GridConfig(symbol=code, market=market)

        try:
            wf_result = analyzer.run(df, base_cfg, space_type="coarse")
            results[code] = wf_result

            robust_icon = "[ROBUST]" if wf_result.is_robust else "[WEAK]"
            print(f"  [{code} {name:<12}] {robust_icon} "
                  f"稳健={wf_result.is_robust} | "
                  f"窗口={len(wf_result.windows)} | "
                  f"累计收益={wf_result.total_return_pct:+.1f}% | "
                  f"平均夏普={wf_result.avg_sharpe:.2f} | "
                  f"平均回撤={wf_result.avg_max_dd:.1f}%")

            if wf_result.param_stability:
                top_param = list(wf_result.param_stability.keys())[0]
                top_pct = list(wf_result.param_stability.values())[0]
                print(f"         最常用参数: {top_param} ({top_pct:.0%})")

        except Exception as e:
            print(f"  [{code}] WF 失败: {e}")

    return results


def main():
    print("加载 ETF 数据...")
    all_data = get_all_etf_data()
    print(f"已加载 {len(all_data)} 只 ETF")

    # 取数据长度前 20 的 ETF（数据量大，结果更可靠）
    top_etfs = dict(sorted(all_data.items(),
                           key=lambda x: len(x[1]), reverse=True)[:20])
    print(f"使用前 20 只 ETF 进行高级分析")

    t0 = time.time()

    # 1. 三引擎对比
    engine_results = compare_engines(top_etfs)

    # 2. Walk-Forward
    wf_results = run_walk_forward(top_etfs, top_n=8)

    elapsed = time.time() - t0

    # 3. 输出最佳组合
    print(f"\n{'='*90}")
    print(f"  ★ 最佳策略推荐")
    print(f"{'='*90}")

    # 从三引擎对比中挑出最佳
    if engine_results["gated"]:
        best_code = max(engine_results["gated"].items(),
                        key=lambda x: x[1].sharpe_ratio)
        r = best_code[1]
        name = SECTOR_ETFS.get(best_code[0], (best_code[0], ""))[0]
        gate_pct = getattr(r, "gate_closed_pct", 0)

        print(f"\n  推荐策略: MA 闸门网格 (Gated Grid)")
        print(f"  推荐标的: {best_code[0]} {name}")
        print(f"  夏普比率: {r.sharpe_ratio:.2f}")
        print(f"  最大回撤: {r.max_drawdown_pct:.1f}%")
        print(f"  年化收益: {r.annualized_return_pct:+.1f}%")
        print(f"  闸门关闭: {gate_pct:.0f}% 交易日（有效避跌）")

    # 4. 导出
    all_static = list(engine_results["static"].values())
    all_dynamic = list(engine_results["dynamic"].values())
    all_gated = list(engine_results["gated"].values())
    export_to_excel(all_static + all_dynamic + all_gated)

    # 动态引擎对比图
    if len(engine_results["static"]) >= 2:
        plot_multi_symbol_comparison(engine_results["static"])

    print(f"\n  总耗时: {elapsed:.0f}s")
    print(f"  输出: {OUTPUT}/")

    # Walk-Forward 详细表
    if wf_results:
        print(f"\n{'='*90}")
        print(f"  Walk-Forward 详细结果")
        print(f"{'='*90}")
        print(f"  {'代码':<8}{'窗口':<5}{'训练期':<24}{'测试期':<24}"
              f"{'收益':<8}{'夏普':<6}{'回撤':<7}{'最优参数'}")
        print(f"  {'-'*85}")
        for code, wfr in wf_results.items():
            for w in wfr.windows[:3]:  # 只显示前 3 个窗口
                params_str = f"s={w.best_params['grid_step']:.1%} " \
                             f"n={w.best_params['grid_num']} " \
                             f"p={w.best_params['position_per_grid']:.0%}"
                print(f"  {code:<8}{w.window_id:<5}"
                      f"{w.train_start}~{w.train_end}  "
                      f"{w.test_start}~{w.test_end}  "
                      f"{w.test_return:>+5.1f}%  "
                      f"{w.test_sharpe:>4.2f}  "
                      f"{w.test_max_dd:>5.1f}%  "
                      f"{params_str}")


if __name__ == "__main__":
    main()
