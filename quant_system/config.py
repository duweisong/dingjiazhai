"""
Global configuration for the quant system.

Loads from config.yaml with sensible defaults.
All modules receive config as a frozen dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass(frozen=True)
class DataSourceConfig:
    """Data source configuration."""
    primary: str = "baostock"              # Primary data source
    fallback_chain: List[str] = field(default_factory=lambda: [
        "baostock", "tushare", "akshare", "efinance"
    ])
    tushare_token: str = ""                # Tushare API token
    cache_dir: str = "quant_system/data/.cache"
    max_retries: int = 2
    request_delay: float = 0.1             # Seconds between requests


@dataclass(frozen=True)
class CostConfig:
    """Trading cost parameters."""
    commission_rate: float = 0.00025       # 0.025% per side
    stamp_tax_rate: float = 0.001          # 0.1% sell only
    min_commission: float = 5.0            # Minimum 5 RMB per trade
    slippage_rate: float = 0.001           # 0.1% slippage per side


@dataclass(frozen=True)
class RiskLimits:
    """Risk control limits."""
    max_single_position: float = 0.10      # Max 10% per stock
    max_sector_exposure: float = 0.30      # Max 30% per sector
    max_drawdown_limit: float = 0.25       # Global max drawdown halt
    max_leverage: float = 1.0              # No leverage by default
    var_confidence: float = 0.95           # VaR confidence level
    var_horizon: int = 1                   # VaR horizon in days
    stop_loss_pct: float = 0.08            # Hard stop-loss at 8%
    take_profit_pct: float = 0.25          # Take-profit at 25%
    trailing_stop_pct: float = 0.06        # Trailing stop at 6%


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest parameters."""
    initial_capital: float = 1_000_000     # 1M RMB
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "000300.SH"           # CSI 300 as benchmark
    rebalance_freq: str = "weekly"         # "daily" | "weekly" | "monthly"
    t_plus_one: bool = True
    position_method: str = "equal_weight"  # "equal_weight" | "signal_strength" | "optimized"
    walk_forward_train: int = 504          # ~2 years
    walk_forward_test: int = 126           # ~6 months
    monte_carlo_sims: int = 1000
    min_trades_for_significance: int = 30


@dataclass(frozen=True)
class FactorConfig:
    """Factor model parameters."""
    momentum_periods: List[int] = field(default_factory=lambda: [20, 60, 120])
    value_metrics: List[str] = field(default_factory=lambda: ["pe", "pb", "ps", "dy"])
    quality_metrics: List[str] = field(default_factory=lambda: ["roe", "gross_margin", "debt_ratio"])
    volatility_periods: List[int] = field(default_factory=lambda: [20, 60])
    ic_test_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    neutralization: bool = True            # Sector + size neutralization
    winsorize_pct: float = 0.01            # Winsorize at 1%/99%
    min_stocks_per_quantile: int = 5


@dataclass(frozen=True)
class LiveConfig:
    """Live trading / signal generation parameters."""
    pushplus_token: str = ""
    pushplus_topic: str = ""
    signal_day: str = "Friday"             # Day to generate signals
    top_n_stocks: int = 10                 # Max stocks to hold
    min_signal_strength: float = 0.3       # Minimum signal strength to trade
    weekly_refresh_stock_pool: bool = True


@dataclass(frozen=True)
class GlobalConfig:
    """Immutable global configuration aggregating all sub-configs."""

    # Stock universe
    stock_pool_index: str = "hs300"        # "hs300" | "csi500" | "hs300,csi500"

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("quant_system/data/.cache"))
    output_dir: Path = field(default_factory=lambda: Path("quant_system/output"))

    # Sub-configs
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @classmethod
    def from_yaml(cls, path: str = "quant_system/config.yaml") -> "GlobalConfig":
        """Load configuration from a YAML file, falling back to defaults."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            stock_pool_index=raw.get("stock_pool_index", "hs300"),
            data_source=DataSourceConfig(**raw.get("data_source", {})),
            costs=CostConfig(**raw.get("costs", {})),
            risk_limits=RiskLimits(**raw.get("risk_limits", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            factor=FactorConfig(**raw.get("factor", {})),
            live=LiveConfig(**raw.get("live", {})),
        )

    def ensure_dirs(self):
        """Create required directories."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "logs").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "reports").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "charts").mkdir(parents=True, exist_ok=True)


# Singleton accessor
_config: Optional[GlobalConfig] = None


def get_config(config_path: str = "quant_system/config.yaml") -> GlobalConfig:
    """Get or load the global configuration singleton."""
    global _config
    if _config is None:
        _config = GlobalConfig.from_yaml(config_path)
        _config.ensure_dirs()
    return _config


def reload_config(config_path: str = "quant_system/config.yaml") -> GlobalConfig:
    """Force reload configuration from file."""
    global _config
    _config = GlobalConfig.from_yaml(config_path)
    _config.ensure_dirs()
    return _config
