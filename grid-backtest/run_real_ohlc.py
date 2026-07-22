#!/usr/bin/env python3
"""
用真实 OHLC 数据验证网格引擎 — 使用 backtest/.cache/ 中的个股日线
并与收盘价估算结果对比，评估 OHLC 估算偏差。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig
from src.grid_engine import GridBacktestEngine, GridResult
from src.grid_strategy import PRESET_STRATEGIES, template_to_config
from src.optimizer import GridOptimizer
from src.metrics import performance_summary
from src.visualize import plot_equity_curve, plot_multi_symbol_comparison
from src.reporter import export_to_excel, generate_html_report
from src.metrics import analyze_monthly, analyze_yearly
from src.cache_adapter import price_to_ohlcv

BACKTEST_CACHE = Path(__file__).parent.parent / "backtest" / ".cache"

STOCK_INFO = {
    "002036": ("联创电子", "SZ", "电子制造"),
    "000948": ("南天信息", "SZ", "信息技术"),
    "002487": ("大金重工", "SZ", "电力设备"),
    "600667": ("太极实业", "SH", "电子"),
}


def load_real_ohlc() -> dict:
    """加载 backtest 缓存的真实 OHLC 个股数据"""
    data = {}
    for code, (name, market, sector) in STOCK_INFO.items():
        for cache_file in BACKTEST_CACHE.glob(f"{code}_*.parquet"):
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
            df.attrs["name"] = name
            df.attrs["market"] = market
            df.attrs["sector"] = sector
            # 取最长的那个（全周期）
            if code not in data or len(df) > len(data[code]):
                data[code] = df
    return data


def run_comparison(data_dict: dict):
    """
    对比两种模式：
    1. 真实 OHLC（直接用缓存的 open/high/low/close）
    2. OHLC 估算（只用 close 还原 OHLC）
    """
    engine = GridBacktestEngine()
    template = [t for t in PRESET_STRATEGIES if t.name == "moderate"][0]

    print(f"\n{'='*80}")
    print(f"  真实 OHLC vs 收盘价估算 对比 (moderate 模板)")
    print(f"{'='*80}")
    print(f"  {'代码':<8}{'名称':<12}"
          f"{'真实年化':<10}{'真实夏普':<8}{'真实回撤':<8}"
          f"{'估算年化':<10}{'估算夏普':<8}{'估算回撤':<8}"
          f"{'夏普偏差':<8}")
    print(f"  {'-'*78}")

    all_real = {}
    all_est = {}

    for code, df in sorted(data_dict.items()):
        name = df.attrs.get("name", code)
        market = df.attrs.get("market", "SZ")
        config = template_to_config(template, code, market)

        # 1. 真实 OHLC
        r_real = engine.run(df, config)
        all_real[code] = r_real

        # 2. 收盘价估算 OHLC
        # 用 (date, close) 构造 Series 传给 price_to_ohlcv
        close_series = df.set_index("date")["close"]
        df_est = price_to_ohlcv(close_series)

        # 对齐日期
        r_est = engine.run(df_est, config)
        all_est[code] = r_est

        sharpe_diff = r_real.sharpe_ratio - r_est.sharpe_ratio
        print(f"  {code:<8}{name:<12}"
              f"{r_real.annualized_return_pct:>+7.1f}%  "
              f"{r_real.sharpe_ratio:>5.2f}  "
              f"{r_real.max_drawdown_pct:>5.1f}%  "
              f"{r_est.annualized_return_pct:>+7.1f}%  "
              f"{r_est.sharpe_ratio:>5.2f}  "
              f"{r_est.max_drawdown_pct:>5.1f}%  "
              f"{sharpe_diff:>+6.2f}")

    # 相关性分析
    real_sharpes = [all_real[c].sharpe_ratio for c in all_real]
    est_sharpes = [all_est[c].sharpe_ratio for c in all_est]
    if len(real_sharpes) >= 2:
        corr = np.corrcoef(real_sharpes, est_sharpes)[0, 1]
        print(f"\n  夏普相关性: {corr:.3f} (真实 vs 估算)")
        avg_bias = np.mean([a - b for a, b in zip(real_sharpes, est_sharpes)])
        print(f"  平均偏差: {avg_bias:+.3f} (正=估算偏保守)")

    return all_real, all_est


def run_optimization_real(data_dict: dict):
    """用真实 OHLC 做参数优化"""
    optimizer = GridOptimizer(scoring="calmar_sharpe")

    from src.grid_strategy import get_param_space
    param_space = get_param_space("standard")

    print(f"\n{'='*80}")
    print(f"  ★ 真实 OHLC 参数优化")
    print(f"{'='*80}")

    best_configs = {}
    for code, df in sorted(data_dict.items()):
        name = df.attrs.get("name", code)
        market = df.attrs.get("market", "SZ")

        base_cfg = GridConfig(
            symbol=code, market=market,
            start_date=str(df["date"].iloc[0].date()),
            end_date=str(df["date"].iloc[-1].date()),
        )

        t0 = time.time()
        try:
            report = optimizer.optimize(df, base_cfg, param_space=param_space, top_n=3)
            elapsed = time.time() - t0
            best = report.best
            best_configs[code] = best

            cfg = best.config
            r = best.result
            print(f"  [{code} {name:<10}] "
                  f"最优: step={cfg.grid_step:.1%} grids={cfg.grid_num} "
                  f"pos={cfg.position_per_grid:.0%} mode={cfg.grid_mode} | "
                  f"年化={r.annualized_return_pct:>+7.1f}% "
                  f"夏普={r.sharpe_ratio:.2f} "
                  f"回撤={r.max_drawdown_pct:.1f}% "
                  f"({elapsed:.0f}s)")

            # 生成完整报告
            result = GridBacktestEngine().run(df, cfg)
            monthly = analyze_monthly(result.equity_curve)
            yearly = analyze_yearly(result.equity_curve)
            plot_equity_curve(result)
            generate_html_report(result, monthly, yearly)

        except Exception as e:
            print(f"  [{code}] ERROR: {e}")

    return best_configs


def main():
    print("加载 backtest 缓存真实 OHLC 数据...")
    data = load_real_ohlc()
    print(f"已加载 {len(data)} 只个股:")
    for code, df in data.items():
        name = df.attrs.get("name", code)
        print(f"  {code} {name}: {len(df)} 条日线 "
              f"({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")

    # 1. 真实 vs 估算对比
    all_real, all_est = run_comparison(data)

    # 2. 真实 OHLC 参数优化
    best_configs = run_optimization_real(data)

    # 3. 汇总
    print(f"\n{'='*80}")
    print(f"  ★ 真实 OHLC 参数优化汇总")
    print(f"{'='*80}")
    print(f"  {'代码':<8}{'名称':<12}{'步长':<8}{'网格数':<7}"
          f"{'仓位/格':<8}{'年化':<10}{'夏普':<7}{'回撤':<8}")
    print(f"  {'-'*68}")
    for code, opt_r in sorted(best_configs.items(),
                               key=lambda x: x[1].result.sharpe_ratio, reverse=True):
        name = data[code].attrs.get("name", code)
        c = opt_r.config
        r = opt_r.result
        print(f"  {code:<8}{name:<12}{c.grid_step:.1%}   {c.grid_num:<5}"
              f"{c.position_per_grid:.0%}     "
              f"{r.annualized_return_pct:>+7.1f}%  {r.sharpe_ratio:>5.2f}"
              f"  {r.max_drawdown_pct:>5.1f}%")

    # 导出
    all_results = [opt_r.result for opt_r in best_configs.values()]
    export_to_excel(all_results)
    if len(best_configs) >= 2:
        plot_multi_symbol_comparison(
            {code: opt_r.result for code, opt_r in best_configs.items()}
        )

    # 最终对比：优化后 vs 默认模板
    print(f"\n{'='*80}")
    print(f"  优化提升幅度 (vs moderate 模板)")
    print(f"{'='*80}")
    for code in sorted(best_configs.keys()):
        if code in all_real and code in best_configs:
            real = all_real[code]
            opt = best_configs[code].result
            sharpe_boost = opt.sharpe_ratio - real.sharpe_ratio
            ret_boost = opt.annualized_return_pct - real.annualized_return_pct
            name = data[code].attrs.get("name", code)
            print(f"  {code} {name:<10}: "
                  f"夏普 {real.sharpe_ratio:.2f}→{opt.sharpe_ratio:.2f} "
                  f"({sharpe_boost:+.2f}) | "
                  f"年化 {real.annualized_return_pct:+.1f}%→"
                  f"{opt.annualized_return_pct:+.1f}% ({ret_boost:+.1f}%)")


if __name__ == "__main__":
    main()
