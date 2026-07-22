#!/usr/bin/env python
"""
A股网格交易回测 — 主入口

基于 entry.py 的网格策略逻辑，自包含引擎，无框架依赖。

用法:
    python main.py                              # 默认参数回测 (联创电子)
    python main.py --symbol 600519 --market SH   # 指定标的
    python main.py --scan                        # 参数扫描
    python main.py --pool stock                  # 标的池扫描
    python main.py --plot --show                 # 生成图表并显示
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig, STOCK_POOL, ETF_POOL
from src.engine import run_backtest, scan_params, scan_pool
from src.grid_engine import GridResult


def print_result(result: GridResult) -> None:
    """打印回测结果"""
    print(result.summary())


def cmd_single(args: argparse.Namespace) -> None:
    """单次回测"""
    symbol, market = _resolve_market(args.symbol, args.market)

    config = GridConfig(
        symbol=symbol,
        market=market,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        grid_period=args.period,
        grid_levels=args.levels,
        grid_step=args.step,
        grid_top_pct=(args.levels - 1) / 2 * args.step,
        grid_bot_pct=-(args.levels - 1) / 2 * args.step,
        use_ma_gate=args.gate,
        ma_gate_period=args.gate_period,
    )

    result = run_backtest(config)

    if args.plot:
        from src.visualize import plot_all
        plot_all(result, show=args.show)


def cmd_scan(args: argparse.Namespace) -> None:
    """参数扫描"""
    symbol, market = _resolve_market(args.symbol, args.market)

    config = GridConfig(
        symbol=symbol,
        market=market,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    steps = [float(s) for s in args.scan_steps.split(",")] if args.scan_steps else None
    periods = [int(p) for p in args.scan_periods.split(",")] if args.scan_periods else None
    levels_list = [int(l) for l in args.scan_levels.split(",")] if args.scan_levels else None

    results = scan_params(config, steps, periods, levels_list)

    # 输出 Top 10
    print(f"\n{'='*65}")
    print(f"  参数扫描完成 — Top 10 (按夏普比率)")
    print(f"{'='*65}")
    for i, r in enumerate(results[:10]):
        print(
            f"  {i+1:2d}. 步长={r.config.grid_step:.2%}  "
            f"周期={r.config.grid_period:3d}  "
            f"档位={r.config.grid_levels:2d}  "
            f"→ 年化={r.annualized_return_pct:+7.1f}%  "
            f"夏普={r.sharpe_ratio:.2f}  "
            f"回撤={r.max_drawdown_pct:.1f}%"
        )

    # 最佳参数
    if results:
        best = results[0]
        print_result(best)
        if args.plot:
            from src.visualize import plot_all
            plot_all(best, show=args.show)


def cmd_pool(args: argparse.Namespace) -> None:
    """标的池扫描"""
    pool = STOCK_POOL if args.pool_type == "stock" else ETF_POOL

    config = GridConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        grid_period=args.period,
        grid_levels=args.levels,
        grid_step=args.step,
        grid_top_pct=(args.levels - 1) / 2 * args.step,
        grid_bot_pct=-(args.levels - 1) / 2 * args.step,
    )

    results = scan_pool(pool, config)

    print(f"\n{'='*60}")
    print(f"  {'个股' if args.pool_type == 'stock' else 'ETF'}池排名 (按夏普比率)")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        name = next((n for c, m, n in pool if c == r.config.symbol), r.config.symbol)
        print(
            f"  {i+1:2d}. {name:6s} ({r.config.symbol})  "
            f"→ 年化={r.annualized_return_pct:+7.1f}%  "
            f"夏普={r.sharpe_ratio:.2f}  "
            f"回撤={r.max_drawdown_pct:.1f}%  "
            f"胜率={r.win_rate:.0f}%"
        )


def _resolve_market(symbol: str, market: str) -> tuple[str, str]:
    """从预置池查找 market"""
    for code, mkt, _ in STOCK_POOL + ETF_POOL:
        if code == symbol:
            return code, mkt
    return symbol, market


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A股网格交易回测系统 (自包含引擎)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                          # 默认参数回测 (联创电子)
  python main.py --symbol 510300 --market SH              # 沪深300ETF
  python main.py --symbol 600519 --market SH --scan       # 茅台参数扫描
  python main.py --pool stock --plot --show               # 个股池扫描 + 图表
  python main.py --step 0.01 --levels 15 --period 480     # 自定义网格参数
        """,
    )

    # 标的参数
    parser.add_argument("--symbol", default="002036", help="股票/ETF代码 (默认: 002036)")
    parser.add_argument("--market", default="SZ", help="交易所 SZ/SH (默认: SZ)")
    parser.add_argument("--start", default="2020-01-01", help="起始日期")
    parser.add_argument("--end", default="2025-12-31", help="结束日期")
    parser.add_argument("--capital", type=float, default=100_000, help="初始资金 (默认: 100000)")

    # 网格参数
    parser.add_argument("--step", type=float, default=0.005, help="网格步长 (默认: 0.005=0.5%%)")
    parser.add_argument("--levels", type=int, default=11, help="网格档位数 (默认: 11)")
    parser.add_argument("--period", type=int, default=240, help="高低点回溯周期 (默认: 240)")

    # 闸门
    parser.add_argument("--gate", action="store_true", help="启用均线闸门")
    parser.add_argument("--gate-period", type=int, default=60, help="闸门均线周期 (默认: 60)")

    # 运行模式
    parser.add_argument("--scan", action="store_true", help="参数扫描模式")
    parser.add_argument("--scan-steps", default=None, help="扫描步长列表 (逗号分隔)")
    parser.add_argument("--scan-periods", default=None, help="扫描周期列表 (逗号分隔)")
    parser.add_argument("--scan-levels", default=None, help="扫描档位列表 (逗号分隔)")
    parser.add_argument("--pool", choices=["stock", "etf"], help="标的池扫描模式")

    # 可视化
    parser.add_argument("--plot", action="store_true", help="生成图表")
    parser.add_argument("--show", action="store_true", help="弹窗显示图表")

    args = parser.parse_args()

    if args.scan:
        cmd_scan(args)
    elif args.pool:
        cmd_pool(args)
    else:
        cmd_single(args)


if __name__ == "__main__":
    main()
