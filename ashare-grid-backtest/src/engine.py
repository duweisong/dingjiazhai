"""
A股网格交易回测 — 回测运行器

封装数据加载 + 网格引擎 + 结果汇总，提供单次回测与参数扫描。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GridConfig
from .data_loader import get_stock_data
from .grid_engine import GridEngine, GridResult, run_grid_backtest


def run_backtest(config: GridConfig | None = None, **overrides) -> GridResult:
    """
    执行一次网格策略回测。

    Args:
        config: GridConfig 配置对象
        **overrides: 覆盖配置字段 (如 grid_step=0.01)

    Returns:
        GridResult 包含完整回测结果
    """
    if config is None:
        config = GridConfig()
    if overrides:
        config = config.with_updates(**overrides)

    # 加载数据
    print(f"\n{'='*60}")
    print(f"  {config.symbol}.{config.market} 网格回测")
    print(f"  参数: 周期={config.grid_period}  档位={config.grid_levels}  "
          f"步长={config.grid_step:.1%}")
    print(f"{'='*60}")

    df = get_stock_data(
        config.symbol, config.market,
        config.start_date, config.end_date,
    )
    print(f"  数据: {len(df)} 条日线  ({df['date'].min().date()} ~ {df['date'].max().date()})")

    # 执行回测
    engine = GridEngine()
    result = engine.run(df, config)

    print(result.summary())
    return result


def scan_params(
    config: GridConfig,
    grid_steps: list[float] | None = None,
    grid_periods: list[int] | None = None,
    grid_levels_list: list[int] | None = None,
) -> list[GridResult]:
    """
    参数扫描: 遍历不同网格参数组合。

    Args:
        config: 基础配置
        grid_steps: 步长列表
        grid_periods: 周期列表
        grid_levels_list: 档位列表

    Returns:
        按夏普比率降序排列的结果列表
    """
    if grid_steps is None:
        grid_steps = [0.005, 0.01, 0.015, 0.02, 0.03]
    if grid_periods is None:
        grid_periods = [120, 240, 480]
    if grid_levels_list is None:
        grid_levels_list = [7, 11, 15]

    # 预加载数据
    print(f"\n[扫描] 加载 {config.symbol}.{config.market} 数据...")
    df = get_stock_data(
        config.symbol, config.market,
        config.start_date, config.end_date,
    )

    all_results: list[GridResult] = []
    total = len(grid_steps) * len(grid_periods) * len(grid_levels_list)
    count = 0

    for step in grid_steps:
        for period in grid_periods:
            for levels in grid_levels_list:
                count += 1
                half = (levels - 1) / 2
                top_pct = half * step
                bot_pct = -half * step

                cfg = config.with_updates(
                    grid_step=step,
                    grid_period=period,
                    grid_levels=levels,
                    grid_top_pct=top_pct,
                    grid_bot_pct=bot_pct,
                )

                print(f"  [{count}/{total}] 步长={step:.1%}  周期={period}  档位={levels}  ",
                      end="", flush=True)

                try:
                    engine = GridEngine()
                    result = engine.run(df.copy(), cfg)
                    all_results.append(result)
                    print(f"→ 年化={result.annualized_return_pct:+.1f}%  "
                          f"夏普={result.sharpe_ratio:.2f}  "
                          f"回撤={result.max_drawdown_pct:.1f}%")
                except Exception as e:
                    print(f"→ ❌ {e}")

    all_results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    return all_results


def scan_pool(
    symbols: list[tuple[str, str, str]],
    config: GridConfig,
) -> list[GridResult]:
    """
    扫描多个标的。

    Args:
        symbols: [(code, market, name), ...]
        config: 网格配置 (symbol/market 字段会被覆盖)

    Returns:
        按夏普比率降序排列的结果列表
    """
    all_results: list[GridResult] = []

    for code, mkt, name in symbols:
        print(f"\n--- {name} ({code}.{mkt}) ---")
        cfg = config.with_updates(symbol=code, market=mkt)

        try:
            result = run_backtest(cfg)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    all_results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    return all_results
