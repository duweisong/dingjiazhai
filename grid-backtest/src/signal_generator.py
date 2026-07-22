"""
网格交易信号生成器 — 对接 PushPlus 微信推送

根据当前价格与网格线的相对位置，生成买入/卖出/等待信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class GridSignal:
    """单条网格交易信号"""
    symbol: str
    name: str
    current_price: float
    grid_levels: List[float]          # 所有网格线
    current_grid: int                 # 当前所在网格区间
    active_buys: List[float]          # 等待触发的买入价
    active_sells: List[float]         # 等待触发的卖出价
    ma_value: float                   # 当前 MA 值
    ma_period: int
    gate_status: str                  # "OPEN" | "CLOSED"
    signal: str                       # "BUY" | "SELL" | "WAIT" | "HOLD"
    urgency: str                      # "HIGH" | "MEDIUM" | "LOW"


class SignalGenerator:
    """
    网格信号生成器。

    用法：
        gen = SignalGenerator()
        signal = gen.generate(df, GridConfig(...))
        message = gen.format_pushplus(signals)
    """

    def generate(self, df: pd.DataFrame, config, ma_period: int = 60) -> GridSignal:
        """
        根据最新数据和网格配置生成交易信号。
        """
        from .grid_engine import build_grid_lines, get_grid_level

        latest = df.iloc[-1]
        current_price = float(latest["close"])

        # 网格线
        mid = float(df["close"].tail(60).mean())
        margin = config.position_per_grid * 3
        if margin < 0.05:
            margin = 0.10
        price_low = mid * (1 - margin)
        price_high = mid * (1 + margin)
        grid_lines = build_grid_lines(price_low, price_high,
                                       config.grid_step, config.grid_mode)

        current_grid = get_grid_level(current_price, grid_lines)

        # 活跃的买卖挂单价位
        active_buys = []
        active_sells = []
        for i, line in enumerate(grid_lines):
            if i <= current_grid:
                active_buys.append(round(line, 3))
            if i > current_grid:
                active_sells.append(round(line, 3))

        # MA 计算
        ma_val = float(df["close"].tail(ma_period).mean())
        gate_open = current_price >= ma_val

        # 信号判定
        if not gate_open:
            signal = "WAIT"   # 闸门关闭，观望
            urgency = "LOW"
        elif active_buys and current_price <= active_buys[-1] * 1.005:
            signal = "BUY"    # 接近买入线
            urgency = "HIGH" if current_price <= active_buys[-1] * 1.002 else "MEDIUM"
        elif active_sells and current_price >= active_sells[0] * 0.995:
            signal = "SELL"   # 接近卖出线
            urgency = "HIGH" if current_price >= active_sells[0] * 0.998 else "MEDIUM"
        else:
            signal = "HOLD"   # 持有，无操作
            urgency = "LOW"

        return GridSignal(
            symbol=config.symbol,
            name=getattr(df, "attrs", {}).get("name", config.symbol),
            current_price=current_price,
            grid_levels=grid_lines.tolist(),
            current_grid=current_grid,
            active_buys=active_buys[-3:] if len(active_buys) > 3 else active_buys,
            active_sells=active_sells[:3] if len(active_sells) > 3 else active_sells,
            ma_value=round(ma_val, 3),
            ma_period=ma_period,
            gate_status="OPEN" if gate_open else "CLOSED",
            signal=signal,
            urgency=urgency,
        )

    def format_pushplus(self, signals: List[GridSignal],
                        title: str = "网格交易信号") -> str:
        """
        格式化为 PushPlus 微信推送消息（Markdown 格式）。

        Args:
            signals: 多个标的的信号列表
            title: 推送标题

        Returns:
            Markdown 格式的推送内容
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"## 🤖 {title}",
            f"",
            f"> 更新时间：{now}",
            f"> 策略：MA 闸门网格（Gated Grid）",
            f"",
            f"---",
            f"",
        ]

        # 高优先级信号
        urgent = [s for s in signals if s.urgency == "HIGH"]
        if urgent:
            lines.append("### 🚨 高优先级")
            lines.append("")
            for s in urgent:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪", "WAIT": "⏸️"}
                lines.append(
                    f"- {emoji.get(s.signal, '')} **{s.signal}** "
                    f"{s.name}({s.symbol}) @ {s.current_price:.2f}"
                )
                lines.append(f"  闸门: {s.gate_status} | MA{s.ma_period}={s.ma_value:.2f}")
                if s.signal == "BUY":
                    lines.append(f"  买入线: {[f'{p:.2f}' for p in s.active_buys]}")
                elif s.signal == "SELL":
                    lines.append(f"  卖出线: {[f'{p:.2f}' for p in s.active_sells]}")
            lines.append("")

        # 完整信号表
        lines.append("### 📊 全部信号")
        lines.append("")
        lines.append("| 标的 | 价格 | 信号 | 闸门 | 网格区间 |")
        lines.append("|------|------|------|------|----------|")
        for s in signals:
            grid_range = f"{s.grid_levels[0]:.2f}~{s.grid_levels[-1]:.2f}"
            lines.append(
                f"| {s.name}({s.symbol}) | {s.current_price:.2f} | "
                f"{s.signal} | {s.gate_status} | {grid_range} |"
            )

        lines += [
            "",
            "---",
            f"> 回测引擎: grid-backtest v1.0",
            f"> 免责声明: 仅供研究参考，不构成投资建议",
        ]

        return "\n".join(lines)

    def push_to_pushplus(self, content: str, title: str,
                         token: str = None,
                         template: str = "markdown") -> bool:
        """
        通过 PushPlus API 发送微信推送。

        Args:
            content: 消息内容（Markdown）
            title: 消息标题
            token: PushPlus Token（None 则从环境变量读取）
            template: 消息模板类型

        Returns:
            是否发送成功
        """
        import os
        import requests

        if token is None:
            token = os.environ.get("PUSHPLUS_TOKEN", "")

        if not token:
            print("[PushPlus] 未配置 PUSHPLUS_TOKEN，跳过推送")
            print(content)
            return False

        try:
            resp = requests.post(
                "https://www.pushplus.plus/send",
                json={
                    "token": token,
                    "title": title,
                    "content": content,
                    "template": template,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == 200:
                print(f"[PushPlus] 推送成功: {title}")
                return True
            else:
                print(f"[PushPlus] 推送失败: {data}")
                return False
        except Exception as e:
            print(f"[PushPlus] 请求异常: {e}")
            return False
