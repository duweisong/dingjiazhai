"""
A股网格交易回测 — 全局配置
"""

from dataclasses import dataclass, replace

# ============================================================
# 回测标的
# ============================================================
STOCK_CODE = "002036"       # 默认股票代码
STOCK_NAME = "联创电子"
MARKET = "SZ"               # SZ=深圳, SH=上海

# ============================================================
# 数据源 & 时间范围
# ============================================================
DATA_SOURCE = "efinance"    # "efinance" | "akshare"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# ============================================================
# 资金 & 交易成本 (A股)
# ============================================================
INITIAL_CAPITAL = 100_000.0     # 初始资金 (元)
COMMISSION_RATE = 0.00025       # 佣金 万2.5 (买卖双向)
STAMP_TAX_RATE = 0.001          # 印花税 千1 (仅卖出)
TRANSFER_FEE_RATE = 0.00001     # 过户费 万0.1 (买卖双向)
MIN_COMMISSION = 5.0            # 最低佣金 5元
MIN_SHARES = 100                # A股最小交易单位 (1手=100股)

# 滑点
SLIPPAGE = 0.001                # 0.1%

# ============================================================
# T+1 制度
# ============================================================
T_PLUS_ONE = True               # T+1: 当日买入次日才能卖出

# ============================================================
# 网格策略参数 (参考 entry.py 设计)
# ============================================================
GRID_PERIOD = 240               # 计算高低点的回溯周期 (日线≈1年)
GRID_LEVELS = 11                # 网格档位数 (含中点)
GRID_STEP = 0.005               # 每档间距 0.5%
GRID_TOP_PCT = 0.025            # 最高档偏离中点 +2.5%
GRID_BOT_PCT = -0.025           # 最低档偏离中点 -2.5%

# ============================================================
# 闸门过滤器 (可选)
# ============================================================
USE_MA_GATE = False             # 是否启用均线闸门
MA_GATE_PERIOD = 60             # 闸门均线周期
MA_GATE_RULE = "below"          # "below"=价在线上不交易, "above"=价在线下不交易


@dataclass(frozen=True)
class GridConfig:
    """网格策略配置（不可变）"""

    symbol: str = STOCK_CODE
    market: str = MARKET
    start_date: str = START_DATE
    end_date: str = END_DATE
    initial_capital: float = INITIAL_CAPITAL

    # 网格参数
    grid_period: int = GRID_PERIOD          # 高低点回溯周期
    grid_levels: int = GRID_LEVELS          # 档位数
    grid_step: float = GRID_STEP            # 每档百分比间距
    grid_top_pct: float = GRID_TOP_PCT      # 最高档偏移
    grid_bot_pct: float = GRID_BOT_PCT      # 最低档偏移

    # 闸门
    use_ma_gate: bool = USE_MA_GATE
    ma_gate_period: int = MA_GATE_PERIOD
    ma_gate_rule: str = MA_GATE_RULE

    # 风控
    stop_loss_pct: float = 0.10             # 总止损线

    def with_updates(self, **kwargs) -> "GridConfig":
        """返回修改后的新配置（不可变模式）"""
        return replace(self, **kwargs)


# ============================================================
# 常用标的池
# ============================================================
STOCK_POOL: list[tuple[str, str, str]] = [
    ("002036", "SZ", "联创电子"),
    ("600519", "SH", "贵州茅台"),
    ("000858", "SZ", "五粮液"),
    ("300750", "SZ", "宁德时代"),
    ("601318", "SH", "中国平安"),
    ("000333", "SZ", "美的集团"),
    ("600036", "SH", "招商银行"),
    ("002415", "SZ", "海康威视"),
    ("300059", "SZ", "东方财富"),
    ("601012", "SH", "隆基绿能"),
]

ETF_POOL: list[tuple[str, str, str]] = [
    ("510300", "SH", "沪深300ETF"),
    ("510500", "SH", "中证500ETF"),
    ("510050", "SH", "上证50ETF"),
    ("159915", "SZ", "创业板ETF"),
    ("588000", "SH", "科创50ETF"),
    ("512880", "SH", "证券ETF"),
    ("512660", "SH", "军工ETF"),
    ("159995", "SZ", "芯片ETF"),
    ("512010", "SH", "医药ETF"),
    ("512800", "SH", "银行ETF"),
]
