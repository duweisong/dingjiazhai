#!/usr/bin/env python3
"""
Xu Xiang Upgraded System - Main Entry Point

Usage:
  python -m xu_xiang_system.main scan       # Real-time stock scan (demo)
  python -m xu_xiang_system.main backtest   # 10-year backtest
  python -m xu_xiang_system.main env        # Market environment check
  python -m xu_xiang_system.main demo       # Overview & system info
"""

import sys
import argparse
import pandas as pd


def cmd_scan():
    """Run stock scanner (demo mode with mock data)"""
    from .scanner import XuXiangScanner
    from .market_env import MarketSnapshot

    scanner = XuXiangScanner()
    env = MarketSnapshot(
        date=pd.Timestamp.now(), level="B", index_close=3300,
        ma_60=3200, ma_direction="up", volume_100m=8500,
        limit_up_count=45, limit_down_count=12,
        adv_decline_ratio=0.55, score=55,
        is_tradeable=True, suggested_action="moderate")

    mock = pd.DataFrame([dict(
        symbol="002036", name="LianChuang", close=12.5,
        ma_5=12.0, ma_20=11.5, ma_60=10.8, ma_bullish=1,
        ret_5d=0.08, ret_20d=0.15, vol_ratio=1.8,
        mcap=80e8, turnover=0.08, pct_chg=0.04,
        boll_width=0.08, macd_hist=0.15, macd_hist_prev=-0.05,
        rsi=58, hh_20d=12.6,
        north_bound_pct=0.02, inst_holding=0.08, top10_pct=0.35)])

    results = [scanner.scan_single(row, env, 0.7)
               for _, row in mock.iterrows()]
    scanner.print_results(results)


def cmd_backtest():
    """Run 10-year backtest"""
    from .backtest_10y import run_backtest
    run_backtest()


def cmd_env():
    """Check market environment classification"""
    from .market_env import MarketEnvClassifier

    c = MarketEnvClassifier()
    tests = [
        ("2020-07-06", 3500, 3450, 15000, 150, 5, 3500, 500),
        ("2018-10-11", 2600, 3000, 3500, 10, 200, 300, 3700),
        ("2024-09-30", 3350, 3100, 26000, 200, 0, 4500, 100),
        ("2022-04-25", 2900, 3100, 6000, 25, 80, 1500, 2500),
        ("2023-06-01", 3200, 3150, 9000, 50, 10, 2300, 1700),
    ]

    print("\n" + "=" * 70)
    print("  Market Environment Classification Demo")
    print("=" * 70)
    print(f"  {'Date':<12}{'Level':<7}{'Score':<7}Action")
    print(f"  {'-'*50}")
    for d, cl, ma, v, lu, ld, a, dc in tests:
        s = c.classify_daily(pd.Timestamp(d), cl, ma, v, lu, ld, a, dc)
        print(f"  {str(s.date.date()):<12}{s.level:<7}"
              f"{s.score:<7.0f}{s.suggested_action}")


def cmd_demo():
    """Show system overview"""
    print()
    print("=" * 60)
    print("  Xu Xiang Upgraded Quant System v1.0")
    print("  Based on: Xu Xiang's core philosophy")
    print("  Upgraded for: 2015-2025 market evolution")
    print("=" * 60)
    print()
    print("  Core Modules:")
    print("    market_env.py       - A/B/C market environment classifier")
    print("    capital_structure.py - Capital participant analysis (5 types)")
    print("    scanner.py          - 5-dimension stock scoring engine")
    print("    risk_manager.py     - 5-dimension risk management")
    print("    strategy.py         - Core trading signal generator")
    print("    backtest_10y.py     - 10-year historical backtest engine")
    print("    data_loader.py      - A-share data fetching (akshare)")
    print()
    print("  Quick Start:")
    print("    python -m xu_xiang_system.main demo       # This overview")
    print("    python -m xu_xiang_system.main env        # Market env check")
    print("    python -m xu_xiang_system.main scan       # Stock scanner demo")
    print("    python -m xu_xiang_system.main backtest   # 10-year backtest")
    print()
    print("  Key Innovations over Original Xu Xiang:")
    print("    1. Env classification replaces blind trend-following")
    print("    2. Capital structure analysis avoids quant traps")
    print("    3. 5D risk management replaces simple 3% stop-loss")
    print("    4. ATR dynamic stops adapt to volatility")
    print("    5. Time + logic stops catch failing theses early")


def main():
    parser = argparse.ArgumentParser(
        description="Xu Xiang Upgraded Quant System")
    parser.add_argument(
        "cmd", nargs="?", default="demo",
        choices=["scan", "backtest", "env", "demo"],
        help="Command to run")
    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "env": cmd_env,
        "demo": cmd_demo,
    }
    commands[args.cmd]()


if __name__ == "__main__":
    main()
