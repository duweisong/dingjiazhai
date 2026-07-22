#!/usr/bin/env python3
"""
多 ETF 组合网格优化 — 基于 MA 周期优化结果的最终组合

用最优 MA 周期 + 波动率倒数加权，构建多 ETF 网格组合。
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
from src.walk_forward import GatedGridEngine
from src.portfolio_grid import PortfolioGridOptimizer, PortfolioGridResult
from src.cache_adapter import get_all_etf_data, SECTOR_ETFS
from src.metrics import performance_summary
from src.reporter import export_to_excel
from src.visualize import plot_equity_curve

OUTPUT = Path(__file__).parent / "output"

# MA 优化结果：每 ETF 的最优 MA 周期
OPTIMAL_MA = {
    "159611": 60, "159852": 120, "159869": 20, "159886": 80,
    "159993": 20, "159995": 100, "512660": 20, "512670": 120,
    "512710": 20, "512760": 100, "512800": 20, "512880": 30,
    "512980": 100, "515050": 30, "515210": 20, "515220": 30,
    "515880": 20, "516160": 20, "516510": 100, "516780": 20,
    "516970": 30, "560860": 20, "561170": 120, "561330": 100,
    "562500": 80,
}


def build_top_portfolios(data_dict: dict):
    """构建多组 ETF 组合并对比"""
    # 按行业 + 夏普排序选最佳标的
    top_by_sector = {}
    for code in OPTIMAL_MA:
        if code not in data_dict:
            continue
        df = data_dict[code]
        info = SECTOR_ETFS.get(code, (code, "未知"))
        sector = info[1]
        market = "SZ" if code.startswith("1") else "SH"

        # 用优化后的 MA 跑一次
        ma = OPTIMAL_MA[code]
        engine = GatedGridEngine(ma_period=ma)
        config = GridConfig(
            symbol=code, market=market,
            grid_step=0.025, grid_num=10, position_per_grid=0.10,
        )
        try:
            result = engine.run(df, config)
            if sector not in top_by_sector or \
               result.sharpe_ratio > top_by_sector[sector]["sharpe"]:
                top_by_sector[sector] = {
                    "code": code, "sharpe": result.sharpe_ratio,
                    "name": info[0], "ma": ma,
                }
        except Exception:
            pass

    print(f"\n  各行业最优 ETF:")
    for sector, info in sorted(top_by_sector.items()):
        print(f"    {sector:<8}: {info['code']} {info['name']:<14} "
              f"夏普={info['sharpe']:.2f} MA{info['ma']}")

    # 构建多组组合
    portfolios = {
        "保守组合 (金融+公用+消费)": ["512800", "561170", "159996"],
        "成长组合 (TMT+制造+周期)": ["515880", "516780", "560860"],
        "均衡组合 (跨5行业)": ["512800", "515880", "516780", "561170", "515210"],
        "宽基组合 (银行+钢铁+通信)": ["512800", "515210", "515880"],
        "高夏普组合 (Top 5)": ["512800", "560860", "561330", "515210", "516780"],
    }

    engine_type = "gated"

    print(f"\n{'='*90}")
    print(f"  多 ETF 组合网格对比 (Gated + 优化MA + 波动率倒数加权)")
    print(f"{'='*90}")
    print(f"  {'组合':<24}{'夏普':<8}{'回撤':<8}{'年化':<10}"
          f"{'分散度':<8}{'标的'}")
    print(f"  {'-'*85}")

    all_results = {}
    best_portfolio = None
    best_sharpe = -999

    for name, pool in portfolios.items():
        valid = [c for c in pool if c in data_dict]
        if len(valid) < 2:
            continue

        configs = {}
        for code in valid:
            df = data_dict[code]
            market = "SZ" if code.startswith("1") else "SH"
            configs[code] = GridConfig(
                symbol=code, market=market,
                grid_step=0.025, grid_num=10, position_per_grid=0.10,
                start_date=str(df["date"].iloc[0].date()),
                end_date=str(df["date"].iloc[-1].date()),
            )

        opt = PortfolioGridOptimizer(
            engine_type=engine_type, ma_periods=OPTIMAL_MA,
        )
        try:
            result = opt.inverse_vol_weight(data_dict, configs)
            all_results[name] = result

            etf_names = [SECTOR_ETFS.get(c, (c, ""))[0][:4] for c in valid]
            print(f"  {name:<24}"
                  f"{result.sharpe_ratio:>5.2f}  "
                  f"{result.max_drawdown_pct:>5.1f}%  "
                  f"{result.annualized_return_pct:>+7.1f}%  "
                  f"{result.diversification_ratio:>5.2f}x  "
                  f"{','.join(etf_names)}")

            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_portfolio = (name, result)

        except Exception as e:
            print(f"  {name:<24} ERROR: {e}")

    return all_results, best_portfolio


def main():
    print("加载 ETF 数据...")
    all_data = get_all_etf_data()
    print(f"已加载 {len(all_data)} 只 ETF")

    # 只用有优化 MA 的 ETF
    optimized_data = {c: d for c, d in all_data.items() if c in OPTIMAL_MA}
    print(f"有优化MA的: {len(optimized_data)} 只")

    t0 = time.time()
    all_results, best = build_top_portfolios(optimized_data)

    if best:
        name, result = best
        print(f"\n{'='*90}")
        print(f"  ★ 最佳组合: {name}")
        print(f"{'='*90}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  年化收益: {result.annualized_return_pct:+.1f}%")
        print(f"  最大回撤: {result.max_drawdown_pct:.1f}%")
        print(f"  卡尔玛比率: {result.calmar_ratio:.2f}")
        print(f"  分散化倍数: {result.diversification_ratio:.2f}x")
        print(f"\n  资金分配:")
        for code, w in sorted(result.weights.items(),
                               key=lambda x: x[1], reverse=True):
            if code in result.results:
                r = result.results[code]
                name_str = SECTOR_ETFS.get(code, (code, ""))[0]
                ma = OPTIMAL_MA.get(code, 60)
                print(f"    {code} {name_str:<14}: {w:.0%} "
                      f"(夏普{r.sharpe_ratio:.2f} MA{ma})")

    # 导出
    all_grid_results = []
    for name, pr in all_results.items():
        for code, r in pr.results.items():
            all_grid_results.append(r)
    if all_grid_results:
        export_to_excel(all_grid_results)

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
