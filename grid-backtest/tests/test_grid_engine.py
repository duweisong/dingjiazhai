"""
网格交易回测系统 — 单元测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.config import GridConfig
from src.grid_engine import (
    build_grid_lines, get_grid_level, GridBacktestEngine, GridResult,
)
from src.grid_strategy import PRESET_STRATEGIES, template_to_config


# ============================================================
# 造测试数据
# ============================================================

def make_test_df(days: int = 500, start_price: float = 50.0,
                 volatility: float = 0.02, seed: int = 42):
    """生成模拟日线数据（随机游走）"""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="B")
    rets = np.random.normal(0.0002, volatility, days)
    prices = start_price * np.cumprod(1 + rets)

    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.normal(0, 0.005, days)),
        "close": prices,
        "high": np.maximum(prices, prices * (1 + volatility * 0.7)) * 1.005,
        "low": np.minimum(prices, prices * (1 - volatility * 0.7)) * 0.995,
        "volume": np.random.randint(1_000_000, 10_000_000, days),
        "amount": prices * np.random.randint(1_000_000, 10_000_000, days),
    })
    return df


# ============================================================
# 网格线生成
# ============================================================

class TestBuildGridLines:
    def test_geometric_grid(self):
        lines = build_grid_lines(40, 60, 0.02, "geometric")
        assert len(lines) > 2
        assert lines[0] >= 40 * 0.999
        assert lines[-1] <= 60 * 1.001
        # 验证等比关系
        ratios = lines[1:] / lines[:-1]
        np.testing.assert_allclose(ratios, 1.02, rtol=0.005)

    def test_arithmetic_grid(self):
        lines = build_grid_lines(40, 60, 0.02, "arithmetic")
        assert len(lines) > 2
        assert lines[0] >= 40 * 0.999
        assert lines[-1] <= 60 * 1.001
        # 验证等差关系
        diffs = np.diff(lines)
        np.testing.assert_allclose(diffs, diffs[0], rtol=0.01)

    def test_minimum_grids(self):
        """至少返回 2 条线"""
        lines = build_grid_lines(40, 41, 0.001, "geometric")
        assert len(lines) >= 2

    def test_narrow_range(self):
        """窄区间处理"""
        lines = build_grid_lines(50, 52, 0.02, "geometric")
        assert len(lines) >= 2
        assert lines[0] <= 50
        assert lines[-1] >= 52


class TestGetGridLevel:
    def test_below_first(self):
        lines = np.array([40, 42, 44, 46, 48, 50])
        assert get_grid_level(38, lines) == 0

    def test_above_last(self):
        lines = np.array([40, 42, 44, 46, 48, 50])
        assert get_grid_level(55, lines) == len(lines) - 2

    def test_in_middle(self):
        lines = np.array([40, 42, 44, 46, 48, 50])
        assert get_grid_level(45, lines) == 2  # 在 44~46 之间


# ============================================================
# 网格引擎
# ============================================================

class TestGridEngine:
    def test_basic_run(self):
        """基本回测能否跑通"""
        df = make_test_df(500)
        config = GridConfig(
            symbol="TEST",
            market="SZ",
            start_date="2022-01-01",
            end_date="2023-12-31",
            grid_step=0.02,
            grid_num=20,
            position_per_grid=0.05,
            initial_capital=100_000,
        )
        engine = GridBacktestEngine()
        result = engine.run(df, config)

        assert isinstance(result, GridResult)
        assert len(result.equity_curve) > 0
        assert result.final_value > 0
        assert result.total_trades >= 0
        assert len(result.grid_lines) > 2

    def test_result_metrics_valid(self):
        """验证指标值有效"""
        df = make_test_df(300, start_price=100)
        config = GridConfig(
            symbol="TEST",
            market="SH",
            grid_step=0.02,
            grid_num=15,
            position_per_grid=0.05,
        )
        engine = GridBacktestEngine()
        result = engine.run(df, config)

        assert -100 <= result.total_return_pct <= 1000
        assert -100 <= result.annualized_return_pct <= 500
        assert -100 <= result.max_drawdown_pct <= 0
        assert 0 <= result.win_rate <= 100
        assert 0 <= result.grid_utilization <= 100

    def test_trades_have_consistent_state(self):
        """交易后 cash + shares*price 一致性"""
        df = make_test_df(200)
        config = GridConfig(
            symbol="TEST", market="SZ",
            grid_step=0.02, grid_num=10, position_per_grid=0.05,
        )
        engine = GridBacktestEngine()
        result = engine.run(df, config)

        # 最后一天权益应等于 cash + shares * close
        last = result.equity_curve.iloc[-1]
        expected_value = result.final_cash + result.final_shares * last["close"]
        assert abs(result.final_value - expected_value) < 1.0

    def test_conservative_template(self):
        """保守模板能正常跑"""
        df = make_test_df(252)
        tpl = [t for t in PRESET_STRATEGIES if t.name == "conservative"][0]
        config = template_to_config(tpl, "ETF300", "SH")
        engine = GridBacktestEngine()
        result = engine.run(df, config)
        assert result.final_value > 0

    def test_aggressive_template(self):
        """激进出价能正常跑"""
        df = make_test_df(252)
        tpl = [t for t in PRESET_STRATEGIES if t.name == "aggressive"][0]
        config = template_to_config(tpl, "ETF300", "SH")
        engine = GridBacktestEngine()
        result = engine.run(df, config)
        assert result.final_value > 0

    def test_arithmetic_mode(self):
        """等差网格模式"""
        df = make_test_df(300)
        config = GridConfig(
            symbol="TEST", market="SZ",
            grid_step=0.02, grid_num=15,
            position_per_grid=0.05, grid_mode="arithmetic",
        )
        engine = GridBacktestEngine()
        result = engine.run(df, config)
        assert result.config.grid_mode == "arithmetic"
        assert len(result.grid_lines) > 2

    def test_with_custom_range(self):
        """自定义价格区间"""
        df = make_test_df(300, start_price=80)
        config = GridConfig(
            symbol="TEST", market="SZ",
            grid_step=0.03, grid_num=10,
            position_per_grid=0.05,
            price_low=70, price_high=90,
        )
        engine = GridBacktestEngine()
        result = engine.run(df, config)
        assert result.grid_lines[0] >= 70 * 0.99
        assert result.grid_lines[-1] <= 90 * 1.01

    def test_config_immutability(self):
        """配置不可变"""
        cfg = GridConfig(symbol="TEST", market="SZ", grid_step=0.02)
        cfg2 = cfg.with_updates(grid_step=0.03)
        assert cfg.grid_step == 0.02
        assert cfg2.grid_step == 0.03
        assert cfg2.symbol == "TEST"


# ============================================================
# 策略模板
# ============================================================

class TestPresets:
    def test_all_templates_valid(self):
        """所有预设模板参数合法"""
        for tpl in PRESET_STRATEGIES:
            assert 0.001 <= tpl.grid_step <= 0.10
            assert 3 <= tpl.grid_num <= 100
            assert 0.01 <= tpl.position_per_grid <= 0.50
            assert tpl.grid_mode in ("geometric", "arithmetic")

    def test_template_to_config(self):
        tpl = PRESET_STRATEGIES[0]
        cfg = template_to_config(tpl, "510300", "SH")
        assert cfg.symbol == "510300"
        assert cfg.market == "SH"
        assert cfg.grid_step == tpl.grid_step


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
