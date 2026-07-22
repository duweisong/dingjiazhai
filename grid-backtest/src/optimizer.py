"""
网格交易回测系统 — 参数优化器

支持网格搜索、多标的并行优化、推进分析（Walk-Forward）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import pandas as pd

from .config import GridConfig
from .grid_engine import GridBacktestEngine, GridResult
from .grid_strategy import get_param_space


@dataclass(frozen=True)
class OptResult:
    """单组参数优化结果"""
    config: GridConfig
    result: GridResult
    score: float           # 综合评分
    rank: int = 0


@dataclass
class OptimizationReport:
    """优化报告"""
    symbol: str
    total_combinations: int
    valid_results: int
    best: OptResult
    top_n: List[OptResult]
    all_results: List[OptResult]
    param_heatmap: Dict[str, Dict]  # 参数 → 聚合绩效


class GridOptimizer:
    """网格参数扫描优化器"""

    def __init__(self, scoring: str = "calmar_sharpe"):
        """
        Args:
            scoring: 评分函数
                - "calmar_sharpe": 0.5*Calmar + 0.3*Sharpe + 0.2*TotalRet
                - "sharpe": 纯夏普比率
                - "total_return": 纯总收益率
                - "composite": 综合加权
        """
        self.scoring = scoring
        self.engine = GridBacktestEngine()

    def score(self, result: GridResult) -> float:
        """计算综合评分（越高越好）"""
        if self.scoring == "sharpe":
            return result.sharpe_ratio
        elif self.scoring == "total_return":
            return result.total_return_pct
        elif self.scoring == "composite":
            return (
                result.calmar_ratio * 0.4
                + result.sharpe_ratio * 0.3
                + result.total_return_pct / 10 * 0.2
                + result.grid_utilization / 100 * 0.1
            )
        else:  # calmar_sharpe (default)
            return (
                result.calmar_ratio * 0.5
                + result.sharpe_ratio * 0.3
                + max(0, result.total_return_pct) / 50 * 0.2
            )

    def optimize(self, df: pd.DataFrame, base_config: GridConfig,
                 param_space: Optional[Dict[str, List]] = None,
                 space_type: str = "standard",
                 top_n: int = 10,
                 progress_callback: Optional[Callable] = None,
                 ) -> OptimizationReport:
        """
        参数网格搜索优化。

        Args:
            df: 日线数据
            base_config: 基础配置（symbol/market/日期等）
            param_space: 自定义参数空间（None则用预设）
            space_type: "fine"|"standard"|"coarse"
            top_n: 保留前 N 名
            progress_callback: 进度回调 fn(done, total)

        Returns:
            OptimizationReport
        """
        if param_space is None:
            param_space = get_param_space(space_type)

        keys = list(param_space.keys())
        values = list(param_space.values())
        total = 1
        for v in values:
            total *= len(v)

        all_results: List[OptResult] = []
        count = 0

        for combo in product(*values):
            count += 1
            kwargs = dict(zip(keys, combo))
            try:
                cfg = base_config.with_updates(**kwargs)
                result = self.engine.run(df, cfg)
                sc = self.score(result)
                all_results.append(OptResult(config=cfg, result=result, score=sc))
            except Exception:
                pass

            if progress_callback and count % 10 == 0:
                progress_callback(count, total)

        # 排序
        all_results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(all_results):
            object.__setattr__(r, "rank", i + 1)

        best = all_results[0] if all_results else None
        top_list = all_results[:top_n]

        if best is None:
            raise ValueError("优化无有效结果——请检查数据与参数范围")

        # 参数热力图数据
        heatmaps = {}
        for param in keys:
            agg = {}
            for opt_r in all_results:
                val = getattr(opt_r.config, param)
                key = str(val)
                if key not in agg:
                    agg[key] = {"count": 0, "total_score": 0.0,
                                "total_ret": 0.0, "total_sharpe": 0.0}
                agg[key]["count"] += 1
                agg[key]["total_score"] += opt_r.score
                agg[key]["total_ret"] += opt_r.result.total_return_pct
                agg[key]["total_sharpe"] += opt_r.result.sharpe_ratio
            # 均值
            for k in agg:
                agg[k]["avg_score"] = agg[k]["total_score"] / agg[k]["count"]
                agg[k]["avg_ret"] = agg[k]["total_ret"] / agg[k]["count"]
                agg[k]["avg_sharpe"] = agg[k]["total_sharpe"] / agg[k]["count"]
            heatmaps[param] = agg

        return OptimizationReport(
            symbol=base_config.symbol,
            total_combinations=total,
            valid_results=len(all_results),
            best=best,
            top_n=top_list,
            all_results=all_results,
            param_heatmap=heatmaps,
        )


def optimize_multi_symbols(
    data_dict: Dict[str, pd.DataFrame],
    base_config_factory: Callable[[str], GridConfig],
    space_type: str = "standard",
    top_n: int = 5,
) -> Dict[str, OptimizationReport]:
    """
    多标的并行优化。

    Args:
        data_dict: {symbol: DataFrame} 数据字典
        base_config_factory: fn(symbol) -> GridConfig 配置工厂
        space_type: 参数空间类型
        top_n: 每标的保留前 N

    Returns:
        {symbol: OptimizationReport} 字典
    """
    optimizer = GridOptimizer()
    reports = {}

    for symbol, df in data_dict.items():
        print(f"\n{'='*60}")
        print(f"  优化: {symbol} ({len(df)} 条日线)")
        print(f"{'='*60}")
        config = base_config_factory(symbol)
        try:
            report = optimizer.optimize(df, config, space_type=space_type,
                                         top_n=top_n)
            reports[symbol] = report
            best = report.best
            print(f"  最优: step={best.config.grid_step:.1%} "
                  f"grids={best.config.grid_num} "
                  f"pos/grid={best.config.position_per_grid:.0%} "
                  f"score={best.score:.3f} "
                  f"ret={best.result.total_return_pct:+.1f}% "
                  f"sharpe={best.result.sharpe_ratio:.2f}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    return reports
