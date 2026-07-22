
"""
五维风控系统 —— 取代徐翔原始的"跌3%砍仓"

维度:
  1. 价格止损: 跌破关键支撑位
  2. 时间止损: 持仓超过预期周期仍未启动
  3. 逻辑止损: 买入时的核心逻辑被证伪
  4. 环境止损: 大盘环境从A降到C，清掉所有仓位
  5. 相关性止损: 同板块其他票集体走弱
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import timedelta


@dataclass
class RiskState:
    """持仓风控状态"""
    entry_date: pd.Timestamp
    entry_price: float
    current_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    holding_days: int = 0
    # 五维标记
    price_stop_hit: bool = False
    time_stop_hit: bool = False
    logic_stop_hit: bool = False
    env_stop_hit: bool = False
    correlation_stop_hit: bool = False
    # 动态止损价
    dynamic_stop_price: float = 0.0
    trailing_stop_price: float = 0.0


@dataclass
class RiskConfig:
    """风控参数配置"""
    # 1. 价格止损
    price_stop_pct: float = 0.05        # 硬止损5%
    atr_stop_mult: float = 2.0          # ATR止损倍数
    trailing_stop_pct: float = 0.04     # 移动止损回撤4%
    support_break_pct: float = 0.02     # 跌破支撑位2%

    # 2. 时间止损
    max_holding_days: int = 10          # 最大持仓天数
    min_profit_by_day5: float = 0.01    # 第5天至少盈利1%

    # 3. 逻辑止损
    sector_drop_threshold: float = 0.05  # 板块指数跌5%→逻辑证伪
    theme_fade_days: int = 5            # 题材热度消退天数

    # 4. 环境止损
    env_min_level: str = "B"            # 最低可接受环境级别

    # 5. 相关性止损
    corr_sector_threshold: float = -0.03  # 同板块跌3%→预警
    corr_leader_threshold: float = -0.05  # 龙头跌5%→连带风险

    # 仓位风控
    max_single_position: float = 0.30   # 单票最大仓位
    max_total_position: float = 0.70    # 总仓位上限
    max_consecutive_loss: int = 2       # 连亏2笔停止
    daily_drawdown_limit: float = 0.03  # 日内回撤限制


class RiskManager:
    """
    五维风控管理器

    在每个交易日检查持仓状态，触发任一维度即发出减仓/清仓信号。
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.consecutive_losses = 0
        self.daily_start_equity = 0.0

    def check_position(
        self,
        position: RiskState,
        current_date: pd.Timestamp,
        current_close: float,
        current_high: float,
        current_low: float,
        atr: float,
        env_level: str,
        sector_return: float,
        leader_return: float,
        entry_logic_valid: bool,
        support_level: float = None,
    ) -> dict:
        """
        检查持仓是否需要退出。
        返回: {should_exit: bool, reasons: list, urgency: str}
        """
        reasons = []
        urgency = "normal"

        # 更新状态
        position.current_price = current_close
        position.highest_since_entry = max(position.highest_since_entry, current_high)
        position.lowest_since_entry = min(position.lowest_since_entry, current_low)
        position.holding_days = (current_date - position.entry_date).days

        # === 1. 价格止损 ===
        hard_stop = position.entry_price * (1 - self.config.price_stop_pct)
        trail_stop = position.highest_since_entry * (1 - self.config.trailing_stop_pct)
        atr_stop = position.entry_price - atr * self.config.atr_stop_mult
        effective_stop = max(hard_stop, trail_stop, atr_stop)
        position.dynamic_stop_price = effective_stop
        position.trailing_stop_price = trail_stop

        if current_low <= effective_stop:
            position.price_stop_hit = True
            reasons.append(f"价格止损触发: {effective_stop:.2f} "
                         f"(硬止损{self.config.price_stop_pct:.0%}/ATR止损/移动止损)")
            urgency = "high"

        # 支撑位止损
        if support_level and current_close < support_level * (1 - self.config.support_break_pct):
            position.price_stop_hit = True
            reasons.append(f"关键支撑位跌破: {support_level:.2f}")
            urgency = "high"

        # === 2. 时间止损 ===
        if position.holding_days >= self.config.max_holding_days:
            pnl = (current_close / position.entry_price) - 1
            if pnl < self.config.min_profit_by_day5:
                position.time_stop_hit = True
                reasons.append(f"时间止损: 持仓{position.holding_days}天未达预期")
                urgency = "medium"

        # === 3. 逻辑止损 ===
        if not entry_logic_valid:
            position.logic_stop_hit = True
            reasons.append("逻辑止损: 买入逻辑被证伪")
            urgency = "high"

        if abs(sector_return) > self.config.sector_drop_threshold and sector_return < 0:
            position.logic_stop_hit = True
            reasons.append(f"逻辑止损: 板块指数跌{sector_return:.1%}")
            urgency = "high"

        # === 4. 环境止损 ===
        if env_level == "C":
            position.env_stop_hit = True
            reasons.append("环境止损: 大盘环境降为C级，清仓")
            urgency = "high"

        # === 5. 相关性止损 ===
        if sector_return < self.config.corr_sector_threshold:
            position.correlation_stop_hit = True
            reasons.append(f"相关性止损: 同板块跌{sector_return:.1%}")
            urgency = "medium"

        if leader_return < self.config.corr_leader_threshold:
            position.correlation_stop_hit = True
            reasons.append(f"相关性止损: 龙头跌{leader_return:.1%}")
            urgency = "high"

        should_exit = any([
            position.price_stop_hit,
            position.time_stop_hit,
            position.logic_stop_hit,
            position.env_stop_hit,
            position.correlation_stop_hit,
        ])

        return {
            "should_exit": should_exit,
            "reasons": reasons,
            "urgency": urgency,
            "dynamic_stop": position.dynamic_stop_price,
            "trailing_stop": position.trailing_stop_price,
        }

    def update_consecutive_losses(self, was_loss: bool):
        """更新连续亏损计数"""
        if was_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def can_trade(self, env_level: str) -> bool:
        """检查是否可以交易"""
        if self.consecutive_losses >= self.config.max_consecutive_loss:
            return False
        if env_level == "C":
            return False
        return True

    def get_position_size(self, env_level: str, volatility: float) -> float:
        """
        根据环境和波动率计算仓位。
        A级 + 低波动 → 满仓30%
        B级 + 高波动 → 减半
        """
        base = self.config.max_single_position
        if env_level == "B":
            base *= 0.6
        if volatility > 0.04:  # 日波动>4%
            base *= 0.5
        return base

    def reset_daily(self, equity: float):
        """每日开盘重置"""
        self.daily_start_equity = equity


def create_risk_state(entry_date, entry_price) -> RiskState:
    """创建初始风控状态"""
    return RiskState(
        entry_date=entry_date,
        entry_price=entry_price,
        highest_since_entry=entry_price,
        lowest_since_entry=entry_price)


if __name__ == "__main__":
    rm = RiskManager()
    pos = create_risk_state(pd.Timestamp("2025-06-15"), 10.0)

    # 模拟几个场景
    checks = [
        ("正常持仓", 10.50, 10.60, 10.30, 0.20, "A", 0.01, 0.02, True, 9.50),
        ("跌破止损", 9.40, 9.50, 9.30, 0.25, "B", -0.02, -0.01, True, 9.50),
        ("环境恶化", 10.20, 10.30, 10.10, 0.30, "C", -0.06, -0.07, True, 9.50),
      ]

    for label, close, high, low, atr, env, sec, lead, logic, sup in checks:
        result = rm.check_position(pos, pd.Timestamp("2025-06-25"),
            close, high, low, atr, env, sec, lead, logic, sup)
        status = "EXIT" if result["should_exit"] else "HOLD"
        print(f"{label}: {status} → {result['reasons']}")
