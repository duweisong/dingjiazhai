#!/usr/bin/env python3
"""
批量 ETF 网格回测 + 参数优化 — 一键运行脚本

用法:
    python run_batch_backtest.py              # 全 ETF 池回测
    python run_batch_backtest.py --optimize    # 全 ETF 池优化
    python run_batch_backtest.py --top 10      # 只跑头部 ETF
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig, INITIAL_CAPITAL
from src.grid_engine import GridBacktestEngine, GridResult
from src.grid_strategy import PRESET_STRATEGIES, template_to_config, get_param_space
from src.optimizer import GridOptimizer
from src.metrics import performance_summary
from src.visualize import plot_equity_curve, plot_multi_symbol_comparison
from src.reporter import export_to_excel, generate_html_report, generate_optimization_report
from src.metrics import analyze_monthly, analyze_yearly
from src.cache_adapter import get_all_etf_data, SECTOR_ETFS


def run_all_etfs(data_dict: dict, preset_name: str = "moderate",
                 ) -> dict:
    """用预设模板对所有 ETF 做一次回测"""
    engine = GridBacktestEngine()
    results = {}
    template = [t for t in PRESET_STRATEGIES if t.name == preset_name][0]

    print(f"\n{'='*70}")
    print(f"  批量网格回测 — 预设: {preset_name} ({template.description[:30]}...)")
    print(f"  标的数: {len(data_dict)}")
    print(f"{'='*70}")

    for code, df in sorted(data_dict.items()):
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"
        config = template_to_config(template, code, market)

        try:
            result = engine.run(df, config)
            results[code] = result
            print(f"  [{code} {name:<12}] "
                  f"年化={result.annualized_return_pct:>+6.1f}%  "
                  f"夏普={result.sharpe_ratio:>5.2f}  "
                  f"回撤={result.max_drawdown_pct:>5.1f}%  "
                  f"交易={result.total_trades:>4}笔  "
                  f"网格利用率={result.grid_utilization:>5.1f}%")
        except Exception as e:
            print(f"  [{code} {name:<12}] ERROR: {e}")

    return results


def run_optimization(data_dict: dict, space_type: str = "standard",
                     top_n: int = 3):
    """对每只 ETF 做参数优化"""
    optimizer = GridOptimizer(scoring="calmar_sharpe")
    all_reports = {}
    best_configs = {}

    print(f"\n{'='*70}")
    print(f"  参数优化 — 扫描空间: {space_type}")
    print(f"  标的数: {len(data_dict)}")
    print(f"{'='*70}")

    param_space = get_param_space(space_type)
    total_combos = 1
    for v in param_space.values():
        total_combos *= len(v)
    print(f"  每标的扫描组合: {total_combos}")

    for code, df in sorted(data_dict.items()):
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"

        base_cfg = GridConfig(
            symbol=code, market=market,
            start_date=str(df["date"].iloc[0].date()),
            end_date=str(df["date"].iloc[-1].date()),
        )

        t0 = time.time()
        try:
            report = optimizer.optimize(df, base_cfg, param_space=param_space, top_n=top_n)
            elapsed = time.time() - t0

            all_reports[code] = report
            best = report.best
            best_configs[code] = best

            cfg = best.config
            r = best.result
            print(f"  [{code} {name:<12}] "
                  f"最优: step={cfg.grid_step:.1%} "
                  f"grids={cfg.grid_num} "
                  f"pos={cfg.position_per_grid:.0%} "
                  f"| 年化={r.annualized_return_pct:>+6.1f}% "
                  f"夏普={r.sharpe_ratio:.2f} "
                  f"回撤={r.max_drawdown_pct:.1f}% "
                  f"({elapsed:.0f}s)")

        except Exception as e:
            print(f"  [{code} {name:<12}] ERROR: {e}")

    return all_reports, best_configs


def print_ranking(results: dict, title: str = "回测排名"):
    """打印排名表"""
    if not results:
        return

    # 按夏普排序
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1].sharpe_ratio, reverse=True)

    print(f"\n{'='*80}")
    print(f"  {title} (按夏普比率)")
    print(f"{'='*80}")
    header = (f"  {'排名':<5}{'代码':<8}{'名称':<14}{'行业':<10}"
              f"{'年化':<9}{'夏普':<7}{'回撤':<8}{'卡尔玛':<7}"
              f"{'胜率':<7}{'交易':<5}{'网格利用率':<8}")
    print(header)
    print(f"  {'-'*78}")

    for i, (code, r) in enumerate(sorted_items, 1):
        info = SECTOR_ETFS.get(code, (code, "未知"))
        name = info[0][:12]
        sector = info[1][:8]
        print(f"  {i:<5}{code:<8}{name:<14}{sector:<10}"
              f"{r.annualized_return_pct:>+7.1f}% "
              f"{r.sharpe_ratio:>5.2f} "
              f"{r.max_drawdown_pct:>6.1f}% "
              f"{r.calmar_ratio:>5.2f} "
              f"{r.win_rate:>5.0f}% "
              f"{r.total_trades:>4} "
              f"{r.grid_utilization:>6.1f}%")

    # 统计
    sharpes = [r.sharpe_ratio for _, r in sorted_items]
    rets = [r.total_return_pct for _, r in sorted_items]
    print(f"\n  平均夏普: {sum(sharpes)/len(sharpes):.2f}  |  "
          f"最高夏普: {max(sharpes):.2f}  |  "
          f"夏普>0: {sum(1 for s in sharpes if s>0)}/{len(sharpes)}  |  "
          f"正收益: {sum(1 for r in rets if r>0)}/{len(rets)}")


def print_optimization_summary(best_configs: dict):
    """打印优化结果汇总"""
    if not best_configs:
        return

    sorted_items = sorted(best_configs.items(),
                          key=lambda x: x[1].result.sharpe_ratio, reverse=True)

    print(f"\n{'='*90}")
    print(f"  ★ 参数优化结果 — 最优参数汇总")
    print(f"{'='*90}")
    header = (f"  {'代码':<8}{'名称':<14}{'行业':<8}"
              f"{'步长':<7}{'网格数':<7}{'仓位/格':<8}{'模式':<12}"
              f"{'年化':<9}{'夏普':<7}{'回撤':<8}")
    print(header)
    print(f"  {'-'*86}")

    for code, opt_r in sorted_items:
        info = SECTOR_ETFS.get(code, (code, "未知"))
        name = info[0][:12]
        sector = info[1][:8]
        c = opt_r.config
        r = opt_r.result
        print(f"  {code:<8}{name:<14}{sector:<8}"
              f"{c.grid_step:.1%}   {c.grid_num:<5}"
              f"{c.position_per_grid:.0%}     {c.grid_mode:<12}"
              f"{r.annualized_return_pct:>+7.1f}% "
              f"{r.sharpe_ratio:>5.2f} "
              f"{r.max_drawdown_pct:>6.1f}%")

    # 最优参数分布
    steps = [opt_r.config.grid_step for _, opt_r in sorted_items]
    grids = [opt_r.config.grid_num for _, opt_r in sorted_items]
    poses = [opt_r.config.position_per_grid for _, opt_r in sorted_items]

    print(f"\n  参数分布 (中位数):")
    print(f"    步长: {sorted(steps)[len(steps)//2]:.1%}  |  "
          f"网格数: {sorted(grids)[len(grids)//2]}  |  "
          f"仓位/格: {sorted(poses)[len(poses)//2]:.0%}")


def main():
    parser = argparse.ArgumentParser(description="批量 ETF 网格回测一键运行")
    parser.add_argument("--optimize", "-o", action="store_true",
                        help="参数优化模式（否则用预设模板）")
    parser.add_argument("--preset", "-p", type=str, default="moderate",
                        help="预设模板名 (default: moderate)")
    parser.add_argument("--space", type=str, default="standard",
                        choices=["coarse", "standard", "fine"],
                        help="优化扫描精度 (default: standard)")
    parser.add_argument("--top", type=int, default=0,
                        help="只跑前 N 只 ETF (0=全部)")
    parser.add_argument("--codes", nargs="*",
                        help="指定 ETF 代码列表")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录")
    args = parser.parse_args()

    start_time = time.time()

    # 加载数据
    print("加载 ETF 数据...")
    all_data = get_all_etf_data()
    print(f"已加载 {len(all_data)} 只 ETF")

    if args.codes:
        all_data = {c: all_data[c] for c in args.codes if c in all_data}
        print(f"筛选后: {len(all_data)} 只")

    if args.top > 0:
        # 先快速跑一遍按流动性排序
        all_data = dict(sorted(all_data.items())[:args.top])
        print(f"取前 {args.top} 只")

    if not all_data:
        print("无可用数据")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # ── 运行 ──
    if args.optimize:
        reports, best_configs = run_optimization(all_data, args.space)
        print_optimization_summary(best_configs)

        # 生成对比图
        best_results = {code: opt_r.result for code, opt_r in best_configs.items()}
        plot_multi_symbol_comparison(best_results)

        # 导出
        export_to_excel([opt_r.result for opt_r in best_configs.values()])

        # 为 Top 3 生成详细报告
        sorted_best = sorted(best_configs.items(),
                             key=lambda x: x[1].result.sharpe_ratio, reverse=True)
        for code, opt_r in sorted_best[:3]:
            r = opt_r.result
            cfg = opt_r.config
            # 用优化后的配置重新跑一次得到完整数据
            engine = GridBacktestEngine()
            df = all_data[code]
            result = engine.run(df, cfg)
            monthly = analyze_monthly(result.equity_curve)
            yearly = analyze_yearly(result.equity_curve)
            plot_equity_curve(result)
            generate_html_report(result, monthly, yearly)
            generate_optimization_report(reports[code])
            print(f"\n  ★ {code} 最优参数报告已生成")

    else:
        results = run_all_etfs(all_data, args.preset)
        print_ranking(results, f"网格回测排名 (预设: {args.preset})")

        # 多标的对比图
        if len(results) >= 2:
            plot_multi_symbol_comparison(results)

        # Excel 汇总导出
        export_to_excel(list(results.values()))

        # 为 Top 5 生成详细 HTML 报告
        sorted_results = sorted(results.items(),
                                key=lambda x: x[1].sharpe_ratio, reverse=True)
        for code, result in sorted_results[:5]:
            monthly = analyze_monthly(result.equity_curve)
            yearly = analyze_yearly(result.equity_curve)
            plot_equity_curve(result)
            generate_html_report(result, monthly, yearly)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  总耗时: {elapsed:.0f}s")
    print(f"  输出目录: {out_dir.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
