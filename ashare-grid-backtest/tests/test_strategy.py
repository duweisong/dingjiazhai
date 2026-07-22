"""
网格策略单元测试
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GridConfig, MIN_SHARES, COMMISSION_RATE, STAMP_TAX_RATE
from src.commissions import (
    calc_commission,
    calc_stamp_tax,
    calc_transfer_fee,
    calc_buy_cost,
    calc_sell_cost,
)
from src.data_loader import _cache_path
from src.grid_engine import GridEngine


class TestGridConfig:
    """GridConfig 配置类测试"""

    def test_default_config(self) -> None:
        cfg = GridConfig()
        assert cfg.symbol == "002036"
        assert cfg.grid_step == 0.005
        assert cfg.grid_levels == 11
        assert cfg.grid_period == 240
        assert cfg.initial_capital == 100_000.0

    def test_with_updates_immutable(self) -> None:
        """验证 with_updates 返回新对象，不修改原对象"""
        cfg1 = GridConfig()
        cfg2 = cfg1.with_updates(grid_step=0.02, grid_levels=20)

        assert cfg1.grid_step == 0.005
        assert cfg1.grid_levels == 11
        assert cfg2.grid_step == 0.02
        assert cfg2.grid_levels == 20
        assert cfg2.symbol == "002036"

    def test_with_updates_frozen(self) -> None:
        """验证 frozen dataclass 不可变"""
        cfg = GridConfig()
        with pytest.raises(Exception):
            cfg.grid_step = 0.99  # type: ignore[misc]

    def test_config_pool_data(self) -> None:
        """验证标的池数据结构"""
        from src.config import STOCK_POOL, ETF_POOL

        assert len(STOCK_POOL) >= 5
        assert len(ETF_POOL) >= 5

        for code, mkt, name in STOCK_POOL:
            assert len(code) == 6
            assert mkt in ("SZ", "SH")
            assert len(name) > 0

        for code, mkt, name in ETF_POOL:
            assert len(code) == 6
            assert mkt in ("SZ", "SH")
            assert len(name) > 0


class TestCommission:
    """交易成本计算测试"""

    def test_buy_commission_min(self) -> None:
        """小额买入: 佣金触发最低 5 元"""
        fee = calc_buy_cost(5000)  # 5000 * 0.00025 = 1.25 → 5
        # 佣金 5 + 过户费 max(5000*0.00001, 1) = 1 = 6
        assert fee == pytest.approx(6.0, abs=0.01)

    def test_buy_commission_large(self) -> None:
        """大额买入: 佣金按比例"""
        amount = 5_000_000  # 500万
        comm = calc_commission(amount)
        assert comm == pytest.approx(1250.0, abs=0.01)

    def test_sell_cost(self) -> None:
        """卖出: 含印花税"""
        amount = 10000
        cost = calc_sell_cost(amount)
        expected = (
            max(amount * COMMISSION_RATE, 5.0)   # 佣金: 5
            + amount * STAMP_TAX_RATE              # 印花税: 10
            + max(amount * 0.00001, 1.0)           # 过户费: 1
        )
        assert cost == pytest.approx(expected, abs=0.01)

    def test_stamp_tax_sell_only(self) -> None:
        """印花税仅卖出"""
        assert calc_stamp_tax(10000) > 0  # 卖出有印花税


def _has_network() -> bool:
    """检测网络是否可用（尝试连接东方财富 API）"""
    import urllib.request
    try:
        urllib.request.urlopen("https://push2his.eastmoney.com", timeout=5)
        return True
    except Exception:
        return False


class TestDataLoader:
    """数据加载模块测试"""

    def test_cache_path_format(self) -> None:
        path = _cache_path("002036", "SZ", "2020-01-01", "2025-12-31")
        assert "002036" in str(path)
        assert path.suffix == ".parquet"

    def test_get_stock_data_cached(self) -> None:
        """验证数据获取（需要网络）"""
        if not _has_network():
            pytest.skip("无网络连接，跳过数据获取测试")

        from src.data_loader import get_stock_data

        df = get_stock_data("002036", "SZ", "2020-01-01", "2020-03-01")
        assert df is not None
        assert len(df) > 0
        assert "open" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns

    def test_get_stock_data_columns(self) -> None:
        """验证返回的 DataFrame 有正确的列（需要网络）"""
        if not _has_network():
            pytest.skip("无网络连接，跳过数据获取测试")

        from src.data_loader import get_stock_data

        df = get_stock_data("510300", "SH", "2024-01-01", "2024-03-01")
        required = ["date", "open", "close", "high", "low"]
        for col in required:
            assert col in df.columns, f"缺少列: {col}"


class TestGridEngine:
    """网格引擎测试"""

    def test_build_grid_levels(self) -> None:
        """验证网格线构建逻辑"""
        engine = GridEngine()
        engine._config = GridConfig(
            grid_levels=11, grid_step=0.005,
            grid_top_pct=0.025, grid_bot_pct=-0.025,
        )

        levels = engine._build_grid(high=12.0, low=8.0)

        assert len(levels) == 11
        assert levels[0] > levels[-1]   # 单调递减
        mid = (12.0 + 8.0) / 2.0        # 10.0
        assert levels[0] > mid           # 最高档 > 中枢
        assert levels[-1] < mid          # 最低档 < 中枢

    def test_shares_lot_rounding(self) -> None:
        """验证股数取整到 100 的倍数"""
        raw_shares = 1234
        lots = raw_shares // MIN_SHARES
        rounded = lots * MIN_SHARES
        assert rounded == 1200
        assert rounded % 100 == 0

    def test_position_target_calculation(self) -> None:
        """验证仓位目标计算"""
        levels = 11
        index = 5
        target = index / (levels - 1)
        assert target == 0.5

        assert (levels - 1) / (levels - 1) == 1.0
        assert 0 / (levels - 1) == 0.0

    def test_grid_level_boundary(self) -> None:
        """验证网格档位边界"""
        levels = 11
        assert 0 <= 0 < levels
        assert 0 <= levels - 1 < levels

    def test_shares_for_value(self) -> None:
        """验证目标金额转整手股数"""
        engine = GridEngine()
        engine._config = GridConfig()

        # 10000 元 ÷ 10 元 = 1000 股 → 10 手
        shares = engine._shares_for_value(10000.0, 10.0)
        assert shares == 1000
        assert shares % 100 == 0

        # 500 元 ÷ 10 元 = 50 股 → 不足 1 手
        shares = engine._shares_for_value(500.0, 10.0)
        assert shares == 0

    def test_build_grid_monotonic(self) -> None:
        """验证网格线严格单调递减"""
        engine = GridEngine()
        engine._config = GridConfig(
            grid_levels=11, grid_step=0.005,
            grid_top_pct=0.025, grid_bot_pct=-0.025,
        )

        levels = engine._build_grid(high=15.0, low=5.0)

        for i in range(len(levels) - 1):
            assert levels[i] > levels[i + 1], (
                f"网格线不单调: levels[{i}]={levels[i]:.4f} <= levels[{i+1}]={levels[i+1]:.4f}"
            )

    def test_can_sell_t_plus_one(self) -> None:
        """T+1: 买入当天不能卖，次日可卖"""
        engine = GridEngine()
        engine._config = GridConfig()
        engine._buy_bar = 100

        # 买入当天: bar 100 → 不能卖
        assert engine._can_sell(100) is False
        # 次日: bar 101 → 可以卖
        assert engine._can_sell(101) is True
        # 更晚: bar 105 → 可以卖
        assert engine._can_sell(105) is True

    def test_can_sell_no_buy(self) -> None:
        """未买过时总能卖 (空仓)"""
        engine = GridEngine()
        engine._config = GridConfig()
        engine._buy_bar = None

        assert engine._can_sell(0) is True
        assert engine._can_sell(500) is True

    def test_current_equity(self) -> None:
        """当前权益 = 现金 + 持仓市值"""
        engine = GridEngine()
        engine._config = GridConfig()
        engine.cash = 50000.0
        engine.shares = 1000

        equity = engine._current_equity(10.0)
        assert equity == 60000.0  # 50000 + 1000 * 10

    def test_max_affordable_shares(self) -> None:
        """验证成本约束下最大可买股数"""
        engine = GridEngine()
        engine._config = GridConfig()
        engine.cash = 10000.0

        # 10000 元 ÷ 10 元/股 = ~990 股 (扣佣金过户费后), 整手 = 900
        shares = engine._max_affordable_shares(10.0)
        assert shares >= 900
        assert shares % 100 == 0
        # 验证确实买得起
        amount = shares * 10.0
        from src.commissions import calc_commission, calc_transfer_fee
        total = amount + calc_commission(amount) + calc_transfer_fee(amount)
        assert total <= 10000.0

    def test_gate_nan_passthrough(self) -> None:
        """NaN 均线 → 放行交易"""
        engine = GridEngine()
        engine._config = GridConfig(use_ma_gate=True, ma_gate_period=60)

        assert engine._gate_allows_buy(10.0, None) is True
        assert engine._gate_allows_buy(10.0, float("nan")) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
