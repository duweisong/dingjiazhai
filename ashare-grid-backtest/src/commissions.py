"""
A股网格交易回测 — A股交易成本计算

费用明细:
  - 佣金: 成交金额 × 万2.5 (买卖双向), 最低 5 元
  - 印花税: 成交金额 × 千1 (仅卖出)
  - 过户费: 成交金额 × 万0.1 (买卖双向), 最低 1 元
"""

from __future__ import annotations

from .config import (
    COMMISSION_RATE,
    STAMP_TAX_RATE,
    TRANSFER_FEE_RATE,
    MIN_COMMISSION,
)


def calc_commission(amount: float) -> float:
    """计算佣金 (买卖双向)"""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def calc_stamp_tax(amount: float) -> float:
    """计算印花税 (仅卖出)"""
    return amount * STAMP_TAX_RATE


def calc_transfer_fee(amount: float) -> float:
    """计算过户费 (买卖双向)"""
    return max(amount * TRANSFER_FEE_RATE, 1.0)


def calc_buy_cost(amount: float) -> float:
    """计算买入总费用: 佣金 + 过户费"""
    return calc_commission(amount) + calc_transfer_fee(amount)


def calc_sell_cost(amount: float) -> float:
    """计算卖出总费用: 佣金 + 印花税 + 过户费"""
    return calc_commission(amount) + calc_stamp_tax(amount) + calc_transfer_fee(amount)


def cost_summary(amount: float, is_sell: bool = False) -> dict[str, float]:
    """详细费用分解"""
    return {
        "amount": amount,
        "commission": calc_commission(amount),
        "stamp_tax": calc_stamp_tax(amount) if is_sell else 0.0,
        "transfer_fee": calc_transfer_fee(amount),
        "total": calc_sell_cost(amount) if is_sell else calc_buy_cost(amount),
    }
