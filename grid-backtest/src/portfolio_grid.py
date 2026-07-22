"""
多 ETF 组合网格优化器

在多个 ETF 之间分配资金，同时运行网格策略，
优化组合层面的资金分配和每标的网格参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import itertools

import numpy as np
import pandas as pd

from .config import GridConfig
from .grid_engine import GridBacktestEngine, GridResult
from .walk_forward import GatedGridEngine


@dataclass
class PortfolioGridResult:
    """组合网格回测结果"""
    configs: Dict[str, GridConfig]          # 每标的配置
    results: Dict[str, GridResult]          # 每标的回测结果
    weights: Dict[str, float]               # 资金权重
    combined_equity: pd.DataFrame           # 合并权益曲线
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    diversification_ratio: float            # 分散化比率


class PortfolioGridOptimizer:
    """
    组合网格优化器。

    方法：
    1. 等权组合（最简单，最鲁棒）
    2. 波动率倒数加权
    3. 夏普最大化组合
    """

    def __init__(self, engine_type: str = "gated",
                 ma_periods: Optional[Dict[str, int]] = None):
        """
        Args:
            engine_type: "static" | "gated"
            ma_periods: 每 ETF 的最优 MA 周期 {code: ma_period}
        """
        self.engine_type = engine_type
        self.ma_periods = ma_periods or {}

    def _make_engine(self, code: str):
        """创建引擎实例"""
        if self.engine_type == "gated":
            ma = self.ma_periods.get(code, 60)
            return GatedGridEngine(ma_period=ma)
        return GridBacktestEngine()

    def equal_weight(self, data_dict: Dict[str, pd.DataFrame],
                     configs: Dict[str, GridConfig],
                     ) -> PortfolioGridResult:
        """
        等权组合 — 每标的分配相同资金。
        """
        n = len(configs)
        if n == 0:
            raise ValueError("至少需要一个标的")

        weight = 1.0 / n
        weights = {c: weight for c in configs}
        return self._run_portfolio(data_dict, configs, weights)

    def inverse_vol_weight(self, data_dict: Dict[str, pd.DataFrame],
                           configs: Dict[str, GridConfig],
                           vol_lookback: int = 60,
                           ) -> PortfolioGridResult:
        """
        波动率倒数加权 — 低波动标的多配置。
        """
        vols = {}
        for code, df in data_dict.items():
            if code not in configs:
                continue
            rets = df["close"].pct_change().dropna()
            if len(rets) > vol_lookback:
                vols[code] = rets.tail(vol_lookback).std()
            else:
                vols[code] = rets.std()

        if not vols:
            return self.equal_weight(data_dict, configs)

        # 倒数归一化
        inv_vols = {c: 1.0 / max(v, 1e-8) for c, v in vols.items()}
        total = sum(inv_vols.values())
        weights = {c: v / total for c, v in inv_vols.items()}
        return self._run_portfolio(data_dict, configs, weights)

    def _run_portfolio(self, data_dict: Dict[str, pd.DataFrame],
                       configs: Dict[str, GridConfig],
                       weights: Dict[str, float],
                       ) -> PortfolioGridResult:
        """执行组合回测"""
        results = {}
        total_capital = sum(c.initial_capital for c in configs.values())

        # 按权重分配资金
        weighted_configs = {}
        for code, config in configs.items():
            w = weights.get(code, 0)
            if w <= 0:
                continue
            weighted_configs[code] = config.with_updates(
                initial_capital=total_capital * w,
            )

        # 逐标的运行回测
        for code, config in weighted_configs.items():
            if code not in data_dict:
                continue
            engine = self._make_engine(code)
            try:
                result = engine.run(data_dict[code], config)
                results[code] = result
            except Exception:
                continue

        return self._combine_results(results, weights, configs)

    def _combine_results(self, results: Dict[str, GridResult],
                         weights: Dict[str, float],
                         configs: Dict[str, GridConfig],
                         ) -> PortfolioGridResult:
        """合并多个标的的权益曲线"""
        if not results:
            raise ValueError("无有效回测结果")

        # 对齐日期，合并权益
        all_equity = {}
        for code, result in results.items():
            eq = result.equity_curve.set_index("date")["total_value"]
            all_equity[code] = eq

        combined = pd.DataFrame(all_equity)
        combined["total"] = combined.sum(axis=1)
        combined = combined.dropna(subset=["total"])
        combined = combined.reset_index()

        # 计算初始总资金
        initial = sum(c.initial_capital for c in configs.values())
        total_ret = combined["total"].iloc[-1] / initial - 1

        days = len(combined)
        years = days / 252
        cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

        daily_rets = combined["total"].pct_change().dropna()
        sharpe = (float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
                  if daily_rets.std() > 0 else 0)

        peak = combined["total"].expanding().max()
        max_dd = float(((combined["total"] - peak) / peak).min())
        calmar = cagr / abs(max_dd) if max_dd < -0.001 else 0

        # 分散化比率
        if len(results) >= 2:
            indiv_vols = []
            for code, eq in all_equity.items():
                rets = eq.pct_change().dropna()
                indiv_vols.append(rets.std() if rets.std() > 0 else 0)
            avg_indiv_vol = np.mean(indiv_vols)
            port_vol = daily_rets.std() if daily_rets.std() > 0 else 0
            div_ratio = avg_indiv_vol / max(port_vol, 1e-8) if avg_indiv_vol > 0 else 1
        else:
            div_ratio = 1.0

        return PortfolioGridResult(
            configs=configs,
            results=results,
            weights=weights,
            combined_equity=combined,
            total_return_pct=total_ret * 100,
            annualized_return_pct=cagr * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar,
            diversification_ratio=div_ratio,
        )


def optimize_portfolio(data_dict: Dict[str, pd.DataFrame],
                       etf_pool: List[str],
                       grid_step: float = 0.025,
                       grid_num: int = 10,
                       pos_per_grid: float = 0.10,
                       engine_type: str = "gated",
                       ma_periods: Optional[Dict[str, int]] = None,
                       ) -> PortfolioGridResult:
    """
    便捷函数：一键优化 ETF 组合网格。

    Args:
        data_dict: {code: ohlcv_df} 数据字典
        etf_pool: 入选组合的 ETF 代码列表
        grid_step: 统一网格步长
        grid_num: 统一网格数
        pos_per_grid: 统一单格仓位
        engine_type: 引擎类型
        ma_periods: 每 ETF 最优 MA 周期
    """
    configs = {}
    for code in etf_pool:
        if code not in data_dict:
            continue
        df = data_dict[code]
        market = "SZ" if code.startswith("1") else "SH"
        configs[code] = GridConfig(
            symbol=code, market=market,
            grid_step=grid_step, grid_num=grid_num,
            position_per_grid=pos_per_grid,
            start_date=str(df["date"].iloc[0].date()),
            end_date=str(df["date"].iloc[-1].date()),
        )

    opt = PortfolioGridOptimizer(engine_type=engine_type, ma_periods=ma_periods)
    return opt.inverse_vol_weight(data_dict, configs)
