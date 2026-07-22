#!/usr/bin/env python3
"""
每日网格交易信号生成 — 可挂 Windows 定时任务，盘后自动运行

用法:
    python run_daily_signal.py                    # 用默认策略 + 缓存数据
    python run_daily_signal.py --push             # 同时推送到 PushPlus
    python run_daily_signal.py --token YOUR_TOKEN  # 指定 PushPlus Token
    python run_daily_signal.py --live             # 用 efinance 实时数据（需 SSL 正常）

部署 (Windows 任务计划):
    schtasks /create /tn "GridSignal" /tr "python C:/AI/grid-backtest/run_daily_signal.py --push" /sc daily /st 15:30
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import GridConfig, INITIAL_CAPITAL
from src.cache_adapter import get_all_etf_data, SECTOR_ETFS
from src.signal_generator import SignalGenerator, GridSignal

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 生产策略配置 ──
PORTFOLIO = {
    "512800": {"weight": 0.52, "ma": 20, "step": 0.025, "grids": 10, "pos": 0.10},
    "515210": {"weight": 0.34, "ma": 20, "step": 0.025, "grids": 10, "pos": 0.10},
    "515880": {"weight": 0.14, "ma": 20, "step": 0.025, "grids": 10, "pos": 0.10},
}

# ── 持仓跟踪文件 ──
POSITION_FILE = Path(__file__).parent / "position.json"


def setup_logging():
    """配置日志"""
    today = datetime.now().strftime("%Y%m%d")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"signal_{today}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_position() -> Dict:
    """加载当前持仓"""
    if POSITION_FILE.exists():
        with open(POSITION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "updated": "",
        "cash": INITIAL_CAPITAL,
        "holdings": {},  # {code: {shares, avg_cost, grid_level, buy_date}}
        "trade_history": [],
    }


def save_position(pos: Dict):
    """保存持仓"""
    pos["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(pos, f, ensure_ascii=False, indent=2)


def load_data(live: bool = False) -> Dict[str, pd.DataFrame]:
    """加载数据"""
    if live:
        try:
            from src.data_loader import get_etf_pool
            return get_etf_pool(list(PORTFOLIO.keys()),
                                force_refresh=True)
        except Exception as e:
            logging.warning(f"实时数据获取失败: {e}，回退缓存")
    return get_all_etf_data()


def generate_signals(data_dict: Dict[str, pd.DataFrame]) -> List[GridSignal]:
    """为组合内每只 ETF 生成交易信号"""
    gen = SignalGenerator()
    signals = []

    for code, params in PORTFOLIO.items():
        if code not in data_dict:
            logging.warning(f"{code} 无数据，跳过")
            continue

        df = data_dict[code]
        name = df.attrs.get("name", code)
        market = "SZ" if code.startswith("1") else "SH"

        config = GridConfig(
            symbol=code, market=market,
            grid_step=params["step"],
            grid_num=params["grids"],
            position_per_grid=params["pos"],
        )

        try:
            signal = gen.generate(df, config, ma_period=params["ma"])
            signals.append(signal)
            logging.info(f"{code} {name}: {signal.signal} "
                         f"@{signal.current_price:.2f} "
                         f"闸门={signal.gate_status} "
                         f"MA{params['ma']}={signal.ma_value:.2f}")
        except Exception as e:
            logging.error(f"{code} 信号生成失败: {e}")

    return signals


def print_signal_summary(signals: List[GridSignal]):
    """打印信号摘要"""
    print(f"\n{'='*70}")
    print(f"  网格交易信号 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"  {'标的':<16}{'价格':<8}{'信号':<8}{'闸门':<8}{'紧迫':<8}{'MA值'}")
    print(f"  {'-'*60}")
    for s in signals:
        urgency_mark = {"HIGH": "!!!", "MEDIUM": "!!", "LOW": ""}
        print(f"  {s.name:<14}{s.current_price:>7.2f} "
              f"{s.signal:<8}{s.gate_status:<8}"
              f"{urgency_mark.get(s.urgency, ''):<8}"
              f"MA{s.ma_period}={s.ma_value:.2f}")

    # 执行建议
    urgent = [s for s in signals if s.urgency == "HIGH"]
    if urgent:
        print(f"\n  🚨 高优先级操作:")
        for s in urgent:
            if s.signal == "BUY":
                print(f"    → 买入 {s.name}({s.symbol}) @ ~{s.current_price:.2f}")
                print(f"      买入线: {[f'{p:.2f}' for p in s.active_buys[:3]]}")
            elif s.signal == "SELL":
                print(f"    → 卖出 {s.name}({s.symbol}) @ ~{s.current_price:.2f}")
                print(f"      卖出线: {[f'{p:.2f}' for p in s.active_sells[:3]]}")

    print(f"\n  下次更新: {datetime.now() + timedelta(days=1):%Y-%m-%d} 盘后")


def push_signals(signals: List[GridSignal], token: str = None):
    """推送到 PushPlus"""
    gen = SignalGenerator()
    today = datetime.now().strftime("%m/%d")
    content = gen.format_pushplus(signals, title=f"网格交易信号 {today}")
    gen.push_to_pushplus(content, f"网格信号 {today}", token=token)


def main():
    parser = argparse.ArgumentParser(description="每日网格交易信号生成")
    parser.add_argument("--push", action="store_true",
                        help="推送到 PushPlus 微信")
    parser.add_argument("--token", type=str,
                        help="PushPlus Token")
    parser.add_argument("--live", action="store_true",
                        help="使用实时数据（需 SSL 正常）")
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 50)
    logging.info("网格交易信号生成启动")

    # 1. 加载数据
    logging.info("加载数据...")
    data = load_data(live=args.live)
    logging.info(f"已加载 {len(data)} 只 ETF")

    # 2. 加载持仓
    position = load_position()
    logging.info(f"当前持仓: {len(position.get('holdings', {}))} 个标的")

    # 3. 生成信号
    signals = generate_signals(data)
    print_signal_summary(signals)

    # 4. 推送
    if args.push:
        push_signals(signals, token=args.token)

    # 5. 保存持仓快照
    save_position(position)

    logging.info("信号生成完成")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
