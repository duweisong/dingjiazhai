"""
网格交易回测系统 — 策略参数定义

提供预设网格策略模板和参数空间定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from .config import GridConfig


# ============================================================
# 预设策略模板
# ============================================================

@dataclass(frozen=True)
class GridStrategyTemplate:
    """网格策略模板"""
    name: str
    description: str
    grid_step: float
    grid_num: int
    position_per_grid: float
    grid_mode: str = "geometric"
    stop_loss_pct: float = 0.15
    notes: str = ""


# 预设模板
PRESET_STRATEGIES: List[GridStrategyTemplate] = [
    GridStrategyTemplate(
        name="conservative",
        description="保守型：小步长(1%)，多网格(30)，轻仓(3%/格)",
        grid_step=0.01, grid_num=30, position_per_grid=0.03,
        notes="适合低波动标的，频繁交易积累微利",
    ),
    GridStrategyTemplate(
        name="moderate",
        description="稳健型：中等步长(2%)，中网格(20)，中仓(5%/格)",
        grid_step=0.02, grid_num=20, position_per_grid=0.05,
        notes="平衡收益与换手，适合大多数 ETF",
    ),
    GridStrategyTemplate(
        name="aggressive",
        description="激进型：大步长(3%)，少网格(10)，重仓(10%/格)",
        grid_step=0.03, grid_num=10, position_per_grid=0.10,
        notes="适合高波动标的，单笔利润大但触发少",
    ),
    GridStrategyTemplate(
        name="micro_step",
        description="超短步长：极密网格(0.5%)，密集交易，微仓(2%/格)",
        grid_step=0.005, grid_num=50, position_per_grid=0.02,
        notes="极端高频，适合震荡市中波动率稳定的标的",
    ),
    GridStrategyTemplate(
        name="wide_range",
        description="宽幅震荡：大步长(4%)，宽间距，重仓(8%/格)",
        grid_step=0.04, grid_num=8, position_per_grid=0.08,
        notes="适合周期性标的，长周期持股",
    ),
    GridStrategyTemplate(
        name="arithmetic_dense",
        description="等差密集：用等差网格(2%步长)，25格，中仓",
        grid_step=0.02, grid_num=25, position_per_grid=0.04,
        grid_mode="arithmetic",
        notes="等差网格在低价区的绝对间距更小",
    ),
]


def template_to_config(template: GridStrategyTemplate,
                       symbol: str, market: str = "SZ",
                       **overrides) -> GridConfig:
    """将策略模板转为具体配置"""
    return GridConfig(
        symbol=symbol,
        market=market,
        grid_step=template.grid_step,
        grid_num=template.grid_num,
        position_per_grid=template.position_per_grid,
        grid_mode=template.grid_mode,
        stop_loss_pct=template.stop_loss_pct,
        **overrides,
    )


# ============================================================
# 参数空间
# ============================================================

def get_param_space(strategy_type: str = "standard") -> Dict[str, List]:
    """
    获取参数搜索空间。

    Args:
        strategy_type: "fine" 精细 | "standard" 标准 | "coarse" 粗扫

    Returns:
        {param_name: [values]} 字典
    """
    spaces = {
        "fine": {
            "grid_step": [0.005, 0.008, 0.01, 0.012, 0.015, 0.018,
                          0.02, 0.022, 0.025, 0.028, 0.03],
            "grid_num": [10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40],
            "position_per_grid": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
            "grid_mode": ["geometric", "arithmetic"],
        },
        "standard": {
            "grid_step": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
            "grid_num": [10, 15, 20, 25, 30, 40, 50],
            "position_per_grid": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
            "grid_mode": ["geometric"],
        },
        "coarse": {
            "grid_step": [0.01, 0.02, 0.03, 0.05],
            "grid_num": [10, 20, 30, 50],
            "position_per_grid": [0.03, 0.05, 0.10],
            "grid_mode": ["geometric"],
        },
    }
    return spaces.get(strategy_type, spaces["standard"])
