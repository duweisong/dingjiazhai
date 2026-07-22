#!/usr/bin/env python3
"""
A 股网格交易回测系统 — CLI 入口

用法:
    # 单标的回测
    python -m src.main --symbol 510300 --market SH

    # 多 ETF 批量回测
    python -m src.main --pool etf

    # 参数优化
    python -m src.main --symbol 510300 --optimize --space standard

    # 使用预设策略模板
    python -m src.main --symbol 002036 --preset moderate

    # 自定义网格参数
    python -m src.main --symbol 510300 --step 0.02 --grids 20 --pos 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保能导入同目录模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    GridConfig, DEFAULT_ETF_POOL, DEFAULT_STOCK_POOL,
    START_DATE, END_DATE, INITIAL_CAPITAL,
)
from src.data_loader import get_stock_data, get_etf_pool, get_multi_stock_data
from src.grid_engine import GridBacktestEngine, run_grid_backtest
from src.grid_strategy import (
    PRESET_STRATEGIES, template_to_config, get_param_space,
)
from src.optimizer import GridOptimizer
from src.metrics import (
    performance_summary, analyze_monthly, analyze_yearly,
    generate_metrics_dict,
)
from src.visualize import (
    plot_equity_curve, plot_price_with_grid,
    plot_optimization_heatmap, plot_multi_symbol_comparison,
)
from src.reporter import (
    generate_html_report, export_to_excel,
    generate_optimization_report,
)


def cmd_single(args, df):
    """单标的回测"""
    cfg = GridConfig(
        symbol=args.symbol,
        market=args.market,
        start_date=args.start or START_DATE,
        end_date=args.end or END_DATE,
        initial_capital=args.capital or INITIAL_CAPITAL,
        grid_step=args.step,
        grid_num=args.grids,
        position_per_grid=args.pos,
        grid_mode=args.mode,
        stop_loss_pct=args.stop_loss,
    )

    if args.price_low:
        cfg = cfg.with_updates(price_low=args.price_low)
    if args.price_high:
        cfg = cfg.with_updates(price_high=args.price_high)

    result = run_grid_backtest(df, cfg)
    print(performance_summary(result))

    # 图表
    eq_path = plot_equity_curve(result)
    price_path = plot_price_with_grid(result, df)
    print(f"  权益曲线: {eq_path}")
    print(f"  价格网格: {price_path}")

    # 报告
    monthly = analyze_monthly(result.equity_curve)
    yearly = analyze_yearly(result.equity_curve)
    html_path = generate_html_report(result, monthly, yearly)
    print(f"  HTML报告: {html_path}")


def cmd_preset(args, df):
    """使用预设策略模板"""
    for tpl in PRESET_STRATEGIES:
        if tpl.name == args.preset:
            cfg = template_to_config(tpl, args.symbol, args.market)
            result = run_grid_backtest(df, cfg)
            print(performance_summary(result))
            eq_path = plot_equity_curve(result)
            price_path = plot_price_with_grid(result, df)
            monthly = analyze_monthly(result.equity_curve)
            yearly = analyze_yearly(result.equity_curve)
            html_path = generate_html_report(result, monthly, yearly)
            print(f"  权益曲线: {eq_path}")
            print(f"  HTML报告: {html_path}")
            return
    print(f"未知预设: {args.preset}，可用: {[t.name for t in PRESET_STRATEGIES]}")


def cmd_optimize(args, df):
    """参数优化"""
    base_cfg = GridConfig(
        symbol=args.symbol,
        market=args.market,
        start_date=args.start or START_DATE,
        end_date=args.end or END_DATE,
        initial_capital=args.capital or INITIAL_CAPITAL,
    )

    optimizer = GridOptimizer(scoring=args.scoring)

    def progress(done, total):
        pct = done / total * 100
        if done % 50 == 0 or done == 1:
            print(f"  进度: {done}/{total} ({pct:.0f}%)")

    report = optimizer.optimize(
        df, base_cfg, space_type=args.space,
        progress_callback=progress,
        top_n=args.top_n,
    )

    # 输出最优
    best = report.best
    print(f"\n{'='*65}")
    print(f"  ★ 最优参数 [{report.symbol}] (共扫描 {report.total_combinations} 组)")
    print(f"{'='*65}")
    print(performance_summary(best.result))

    # Top N
    print(f"\n  Top {args.top_n} 参数组合:")
    print(f"  {'排名':<5}{'步长':<8}{'网格数':<7}{'仓位/格':<8}"
          f"{'模式':<12}{'年化':<10}{'夏普':<7}{'回撤':<8}{'评分':<7}")
    print(f"  {'-'*70}")
    for opt_r in report.top_n:
        c = opt_r.config
        r = opt_r.result
        print(f"  {opt_r.rank:<5}{c.grid_step:.1%}   {c.grid_num:<5}"
              f"{c.position_per_grid:.0%}     {c.grid_mode:<12}"
              f"{r.annualized_return_pct:>+7.1f}%  {r.sharpe_ratio:>5.2f}"
              f"  {r.max_drawdown_pct:>5.1f}%  {opt_r.score:>5.2f}")

    # 热力图
    heatmap_path = plot_optimization_heatmap(report)
    print(f"\n  热力图: {heatmap_path}")

    # 优化报告
    opt_report_path = generate_optimization_report(report)
    print(f"  优化报告: {opt_report_path}")


def cmd_pool(args):
    """多标的批量回测/优化"""
    # 确定标的池
    if args.pool == "etf":
        pool_codes = args.symbols or DEFAULT_ETF_POOL
        symbols = [(c, "SZ" if c.startswith("1") else "SH") for c in pool_codes]
    elif args.pool == "stock":
        symbols = args.symbols or DEFAULT_STOCK_POOL
    else:
        print(f"未知池: {args.pool}")
        return

    print(f"\n  标的池: {len(symbols)} 个")
    data_dict = get_multi_stock_data(
        symbols,
        start=args.start or START_DATE,
        end=args.end or END_DATE,
        force_refresh=args.refresh,
    )
    print(f"  数据加载完成: {len(data_dict)} 个标的")

    if not data_dict:
        print("  无可用数据")
        return

    engine = GridBacktestEngine()
    results = {}
    best_per_symbol = {}

    for symbol, df in data_dict.items():
        print(f"\n  [{symbol}] 回测中...")

        if args.optimize:
            # 优化模式
            base_cfg = GridConfig(
                symbol=symbol,
                market="SZ" if symbol.startswith("1") else "SH",
                start_date=args.start or START_DATE,
                end_date=args.end or END_DATE,
            )
            optimizer = GridOptimizer()
            report = optimizer.optimize(df, base_cfg, space_type=args.space,
                                         top_n=1)
            best = report.best
            best_per_symbol[symbol] = best.result
            cfg_label = (f"step={best.config.grid_step:.1%} "
                         f"grids={best.config.grid_num} "
                         f"pos={best.config.position_per_grid:.0%}")
            print(f"    最优: {cfg_label} | "
                  f"年化={best.result.annualized_return_pct:+.1f}% | "
                  f"夏普={best.result.sharpe_ratio:.2f} | "
                  f"回撤={best.result.max_drawdown_pct:.1f}%")
        else:
            # 使用默认/命令行参数
            cfg = GridConfig(
                symbol=symbol,
                market="SZ" if symbol.startswith("1") else "SH",
                start_date=args.start or START_DATE,
                end_date=args.end or END_DATE,
                grid_step=args.step,
                grid_num=args.grids,
                position_per_grid=args.pos,
                grid_mode=args.mode,
            )
            result = engine.run(df, cfg)
            results[symbol] = result
            print(f"    收益={result.total_return_pct:+.1f}% | "
                  f"夏普={result.sharpe_ratio:.2f} | "
                  f"回撤={result.max_drawdown_pct:.1f}%")

    # 汇总
    final_results = results if not args.optimize else best_per_symbol
    if final_results:
        print(f"\n{'='*70}")
        print(f"  标的池回测汇总 ({len(final_results)} 个)")
        print(f"{'='*70}")
        sorted_symbols = sorted(
            final_results.items(),
            key=lambda x: x[1].sharpe_ratio, reverse=True,
        )
        print(f"  {'标的':<10}{'年化收益':<12}{'夏普':<8}{'最大回撤':<10}"
              f"{'卡尔玛':<8}{'胜率':<8}{'交易数':<7}")
        print(f"  {'-'*60}")
        for sym, r in sorted_symbols:
            print(f"  {sym:<10}{r.annualized_return_pct:>+8.1f}%  "
                  f"{r.sharpe_ratio:>5.2f}  {r.max_drawdown_pct:>7.1f}%  "
                  f"{r.calmar_ratio:>5.2f}  {r.win_rate:>5.1f}%  "
                  f"{r.total_trades:>5}")

        # 对比图
        cmp_path = plot_multi_symbol_comparison(
            {s: r for s, r in sorted_symbols[:12]}
        )
        print(f"\n  对比图: {cmp_path}")

        # Excel 导出
        xlsx_path = export_to_excel([r for _, r in sorted_symbols])
        print(f"  Excel汇总: {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="A股网格交易回测系统 — 步长/持仓/区间可调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.main --symbol 510300 --market SH
  python -m src.main --symbol 002036 --step 0.02 --grids 20 --pos 0.05
  python -m src.main --symbol 510300 --optimize
  python -m src.main --pool etf --optimize
  python -m src.main --symbol 510300 --preset conservative
  python -m src.main --list-presets
        """,
    )

    # ── 标的选择 ──
    parser.add_argument("--symbol", "-s", type=str,
                        help="股票/ETF 代码 (6位)")
    parser.add_argument("--market", "-m", type=str, default="SH",
                        help="交易所: SH(上海) / SZ(深圳) (default: SH)")
    parser.add_argument("--pool", "-p", type=str,
                        choices=["etf", "stock"],
                        help="批量模式: etf / stock")
    parser.add_argument("--symbols", nargs="*",
                        help="自定义标的列表 (配合 --pool 使用)")

    # ── 网格参数 ──
    parser.add_argument("--step", type=float, default=0.02,
                        help="网格步长 (默认 0.02 = 2%%)")
    parser.add_argument("--grids", type=int, default=20,
                        help="网格数量 (默认 20)")
    parser.add_argument("--pos", type=float, default=0.05,
                        help="单格仓位比例 (默认 0.05 = 5%%)")
    parser.add_argument("--mode", type=str, default="geometric",
                        choices=["geometric", "arithmetic"],
                        help="网格模式 (默认 geometric)")
    parser.add_argument("--price-low", type=float,
                        help="价格下限 (不指定则自动)")

    parser.add_argument("--price-high", type=float,
                        help="价格上限 (不指定则自动)")

    parser.add_argument("--stop-loss", type=float, default=0.15,
                        help="止损线 (默认 0.15 = 15%%)")

    # ── 回测日期/资金 ──
    parser.add_argument("--start", type=str, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, help="初始资金")

    # ── 预设策略 ──
    parser.add_argument("--preset", type=str,
                        help="使用预设策略模板")
    parser.add_argument("--list-presets", action="store_true",
                        help="列出所有预设策略")

    # ── 优化 ──
    parser.add_argument("--optimize", "-o", action="store_true",
                        help="参数优化模式")
    parser.add_argument("--space", type=str, default="standard",
                        choices=["fine", "standard", "coarse"],
                        help="参数搜索空间 (default: standard)")
    parser.add_argument("--scoring", type=str, default="calmar_sharpe",
                        choices=["calmar_sharpe", "sharpe", "total_return", "composite"],
                        help="优化评分函数 (default: calmar_sharpe)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="保留前 N 名 (default: 10)")

    # ── 其他 ──
    parser.add_argument("--refresh", "-r", action="store_true",
                        help="强制刷新数据")

    args = parser.parse_args()

    # 列出预设
    if args.list_presets:
        print("\n可用预设策略模板:")
        print(f"  {'名称':<20}{'描述':<60}")
        print(f"  {'-'*80}")
        for t in PRESET_STRATEGIES:
            print(f"  {t.name:<20}{t.description:<60}")
        return

    # 批量模式
    if args.pool:
        cmd_pool(args)
        return

    # 单标的需要 --symbol
    if not args.symbol:
        parser.print_help()
        print("\n请指定 --symbol 或 --pool")
        return

    # 加载数据
    print(f"\n  加载 {args.symbol}.{args.market} 数据...")
    df = get_stock_data(
        args.symbol, args.market,
        start=args.start or START_DATE,
        end=args.end or END_DATE,
        force_refresh=args.refresh,
    )
    print(f"  数据: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()} "
          f"({len(df)} 个交易日)")

    # 路由
    if args.preset:
        cmd_preset(args, df)
    elif args.optimize:
        cmd_optimize(args, df)
    else:
        cmd_single(args, df)


if __name__ == "__main__":
    main()
