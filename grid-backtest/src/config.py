"""
网格交易回测系统 — 全局配置
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# 回测标的 — ETF + 个股
# ============================================================

# 默认 ETF 候选池（流动性好的主流 ETF）
DEFAULT_ETF_POOL: List[str] = [
    # 宽基
    "510300",  # 沪深300ETF
    "510500",  # 中证500ETF
    "510050",  # 上证50ETF
    "159915",  # 创业板ETF
    "588000",  # 科创50ETF
    # 行业
    "512880",  # 证券ETF
    "512660",  # 军工ETF
    "512760",  # 半导体ETF
    "515050",  # 5GETF
    "159995",  # 芯片ETF
    "512010",  # 医药ETF
    "512690",  # 酒ETF
    "516390",  # 新能源汽车ETF
    "159857",  # 光伏ETF
    "512800",  # 银行ETF
    "515220",  # 煤炭ETF
    "159611",  # 电力ETF
    "562500",  # 机器人ETF
    "159869",  # 游戏ETF
    "512200",  # 房地产ETF
]

# 单股票标的
DEFAULT_STOCK_POOL: List[str] = [
    ("002036", "SZ"),  # 联创电子
    ("600519", "SH"),  # 贵州茅台
    ("000858", "SZ"),  # 五粮液
    ("300750", "SZ"),  # 宁德时代
    ("601318", "SH"),  # 中国平安
]

# ============================================================
# 数据源
# ============================================================
DATA_SOURCE = "efinance"  # "efinance" | "akshare"
FORCE_REFRESH = False

# 回测时间范围
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# ============================================================
# 资金与交易成本
# ============================================================
INITIAL_CAPITAL = 100_000  # 初始资金 (元)

# 交易成本 (A股)
COMMISSION_RATE = 0.00025  # 佣金 万分之2.5 (买卖双向)
STAMP_TAX_RATE = 0.001     # 印花税 千分之1 (仅卖出)
MIN_COMMISSION = 5.0       # 最低佣金 5元

# ============================================================
# T+1 制度
# ============================================================
T_PLUS_ONE = True  # T+1: 当日买入次日才能卖出

# ============================================================
# 网格参数默认值
# ============================================================
DEFAULT_GRID_STEP = 0.02        # 默认步长 2%（每格价格间距）
DEFAULT_GRID_NUM = 20           # 默认网格数量（单边10格）
DEFAULT_POSITION_PER_GRID = 0.05  # 单格仓位 5%（占总资金比例）
DEFAULT_PRICE_RANGE_MARGIN = 0.15  # 区间边距 ±15%（基于回测起点价格）


@dataclass(frozen=True)
class GridConfig:
    """网格策略配置（不可变）"""
    symbol: str                          # 标的代码
    market: str = "SZ"                   # 交易所 SZ/SH
    start_date: str = START_DATE
    end_date: str = END_DATE
    initial_capital: float = INITIAL_CAPITAL

    # 网格参数
    grid_step: float = DEFAULT_GRID_STEP       # 步长（百分比）
    grid_num: int = DEFAULT_GRID_NUM           # 网格数
    position_per_grid: float = DEFAULT_POSITION_PER_GRID  # 单格仓位
    price_low: Optional[float] = None           # 区间下限（None=自动计算）
    price_high: Optional[float] = None          # 区间上限（None=自动计算）
    grid_mode: str = "geometric"               # "geometric" | "arithmetic"

    # 风控
    stop_loss_pct: float = 0.15                # 总止损线（跌破区间下限再加这个幅度）
    max_position_pct: float = 0.95             # 最大持仓比例

    def with_updates(self, **kwargs) -> "GridConfig":
        """返回修改后的新配置（不可变模式）"""
        current = {f.name: getattr(self, f.name)
                   for f in self.__dataclass_fields__.values()}
        current.update(kwargs)
        return GridConfig(**current)


# ============================================================
# 优化器默认参数搜索空间
# ============================================================
OPTIMIZER_PARAM_GRID = {
    "grid_step": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    "grid_num": [10, 15, 20, 25, 30, 40, 50],
    "position_per_grid": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
}
