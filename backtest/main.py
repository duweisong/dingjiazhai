#!/usr/bin/env python3
"""
联创电子 (002036.SZ) 波段策略回测系统

用法:
    python main.py                    # 运行所有策略回测
    python main.py --strategy ma_cross  # 运行指定策略
    python main.py --optimize         # 参数优化模式
    python main.py --refresh          # 强制刷新数据

策略列表:
    ma_cross    - 双均线交叉波段
    macd        - MACD金叉死叉波段
    bollinger   - 布林带均值回归波段
    rsi         - RSI超买超卖反转
    composite   - 多指标共振波段
"""

import argparse
import sys
from pathlib import Path

# 确保能导入同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from data_loader import get_data
from engine import BacktestEngine, BacktestResult
from strategies import get_strategy, list_strategies, STRATEGIES
from metrics import (
    performance_summary, analyze_monthly, analyze_yearly,
    trade_analysis,
)
from visualize import (
    plot_equity_curve, plot_trade_analysis, plot_strategy_comparison,
    generate_html_report,
)
from optimizer import WalkForwardOptimizer, AdaptivePositionSizer


def run_single_strategy(df, strategy_name: str, params: dict = None,
                        stop_loss: float = 0.06, take_profit: float = 0.18,
                        trailing_stop: float = 0.05) -> BacktestResult:
    """运行单个策略回测"""
    strategy = get_strategy(strategy_name, **(params or {}))
    engine = BacktestEngine()
    result = engine.run(
        df, strategy,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        trailing_stop_pct=trailing_stop,
    )
    return result


def run_all_strategies(df) -> list:
    """运行全部策略并返回排序结果"""
    print("\n" + "=" * 70)
    print("  联创电子 (002036.SZ) 多策略波段回测")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  初始资金: 元{INITIAL_CAPITAL:,}")
    print("=" * 70)

    results = []
    strategy_configs = [
        # ---- 基础策略 ----
        ("ma_cross", {"ma_short": 5, "ma_long": 20}, 0.05, 0.15, 0.04),
        ("ma_cross", {"ma_short": 10, "ma_long": 30}, 0.05, 0.15, 0.04),
        ("macd", {}, 0.05, 0.15, 0.04),
        ("bollinger", {}, 0.04, 0.12, 0.03),
        ("rsi", {"period": 14, "oversold": 30, "overbought": 70}, 0.04, 0.12, 0.03),
        ("composite", {}, 0.06, 0.20, 0.05),
        # ---- 高级策略 (自带ATR动态止损，sl/tp为fallback) ----
        ("advanced_swing", {"ma_short": 5, "ma_mid": 20, "ma_long": 60,
         "atr_stop_mult": 2.0, "atr_target_mult": 3.0, "require_pullback": True}, 0.04, 0.12, 0.03),
        ("advanced_swing", {"ma_short": 8, "ma_mid": 30, "ma_long": 60,
         "atr_stop_mult": 2.5, "atr_target_mult": 4.0, "require_pullback": False}, 0.05, 0.15, 0.04),
        ("trend_following", {"ma_short": 10, "ma_long": 30, "adx_threshold": 22,
         "atr_stop_mult": 2.5, "atr_target_mult": 4.0}, 0.05, 0.18, 0.04),
        ("trend_following", {"ma_short": 20, "ma_long": 60, "adx_threshold": 20,
         "atr_stop_mult": 3.0, "atr_target_mult": 5.0}, 0.06, 0.20, 0.05),
        ("volume_breakout", {"vol_surge": 1.5, "consolidation_period": 5,
         "breakout_pct": 0.03}, 0.04, 0.12, 0.03),
        ("volume_breakout", {"vol_surge": 2.0, "consolidation_period": 8,
         "breakout_pct": 0.04}, 0.05, 0.15, 0.04),
    ]

    for name, params, sl, tp, ts in strategy_configs:
        label = f"{name}"
        if params:
            label += f"({','.join(f'{k}={v}' for k,v in params.items())})"

        print(f"\n[{label}] 回测中...")
        try:
            result = run_single_strategy(df, name, params, sl, tp, ts)
            results.append(result)
            print(f"  总收益: {result.total_return_pct:+.2f}%  |  "
                  f"年化: {result.annualized_return_pct:+.2f}%  |  "
                  f"夏普: {result.sharpe_ratio:.2f}  |  "
                  f"最大回撤: {result.max_drawdown_pct:.2f}%  |  "
                  f"胜率: {result.win_rate:.1f}%  |  "
                  f"交易: {result.total_trades}笔")
        except Exception as e:
            print(f"  [ERROR] 错误: {e}")

    # 按年化收益排序
    results.sort(key=lambda r: r.calmar_ratio, reverse=True)
    return results


def run_optimization(df, strategy_name: str):
    """参数网格搜索优化"""
    print(f"\n{'='*70}")
    print(f"  参数优化: {strategy_name}")
    print(f"{'='*70}")

    if strategy_name == "ma_cross":
        param_grid = {
            "ma_short": [3, 5, 8, 10, 13],
            "ma_long": [15, 20, 30, 40, 50],
            "stop_loss_pct": [0.04, 0.06, 0.08],
            "take_profit_pct": [0.10, 0.15, 0.20],
        }
    elif strategy_name == "composite":
        param_grid = {
            "ma_short": [5, 8, 10],
            "ma_long": [20, 30, 40],
            "stop_loss_pct": [0.04, 0.06, 0.08],
            "take_profit_pct": [0.12, 0.18, 0.24],
        }
    elif strategy_name == "macd":
        param_grid = {
            "fast": [8, 12, 16],
            "slow": [21, 26, 30],
            "signal": [6, 9, 12],
            "stop_loss_pct": [0.04, 0.06],
        }
    elif strategy_name == "bollinger":
        param_grid = {
            "period": [14, 20, 26],
            "std": [1.5, 2.0, 2.5],
            "stop_loss_pct": [0.03, 0.05, 0.07],
        }
    elif strategy_name == "rsi":
        param_grid = {
            "period": [10, 14, 20],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
            "stop_loss_pct": [0.03, 0.05],
        }
    elif strategy_name == "advanced_swing":
        param_grid = {
            "ma_short": [5, 8, 10],
            "ma_mid": [20, 30],
            "atr_stop_mult": [1.5, 2.0, 2.5],
            "atr_target_mult": [2.5, 3.5, 4.5],
            "require_pullback": [True, False],
            "stop_loss_pct": [0.04, 0.06],
        }
    elif strategy_name == "trend_following":
        param_grid = {
            "ma_short": [10, 20],
            "ma_long": [30, 40, 60],
            "adx_threshold": [18, 22, 25],
            "atr_stop_mult": [2.0, 2.5, 3.0],
            "atr_target_mult": [3.0, 4.0, 5.0],
            "stop_loss_pct": [0.05, 0.07],
        }
    elif strategy_name == "volume_breakout":
        param_grid = {
            "vol_surge": [1.2, 1.5, 2.0],
            "consolidation_period": [3, 5, 8, 12],
            "breakout_pct": [0.02, 0.03, 0.05],
            "stop_loss_pct": [0.03, 0.05, 0.07],
        }
    else:
        print(f"未知策略: {strategy_name}")
        return

    best_result = None
    best_score = -999

    # 生成参数组合
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"  共 {total} 组参数组合\n")

    count = 0
    engine = BacktestEngine()

    for combo in product(*values):
        count += 1
        strategy_params = {}
        other_params = {}
        for k, v in zip(keys, combo):
            if k in ("stop_loss_pct", "take_profit_pct"):
                other_params[k] = v
            else:
                strategy_params[k] = v

        sl = other_params.get("stop_loss_pct", 0.06)
        tp = other_params.get("take_profit_pct", 0.18)

        try:
            strategy = get_strategy(strategy_name, **strategy_params)
            result = engine.run(df, strategy, stop_loss_pct=sl, take_profit_pct=tp)

            # 综合评分: 卡尔玛比率 + 夏普加权
            score = result.calmar_ratio * 0.6 + result.sharpe_ratio * 0.4

            if score > best_score:
                best_score = score
                best_result = result
                best_result.strategy_params = strategy_params
                best_result.strategy_params.update({"stop_loss": sl, "take_profit": tp})
                # 重命名以便区分
                best_result.strategy_name = f"{strategy_name}_optimized"

            pct_done = count / total * 100
            if count % 20 == 0 or count == 1:
                print(f"  [{count}/{total} ({pct_done:.0f}%)] "
                      f"当前最优: score={best_score:.3f} "
                      f"年化={best_result.annualized_return_pct:.1f}% "
                      f"夏普={best_result.sharpe_ratio:.2f}")

        except Exception:
            continue

    if best_result:
        print(f"\n{'='*70}")
        print(f"  最优参数: {best_result.strategy_params}")
        print(performance_summary(best_result))

        monthly = analyze_monthly(best_result.equity_curve)
        yearly = analyze_yearly(best_result.equity_curve)
        extra = trade_analysis(best_result.trades)

        eq_path = plot_equity_curve(best_result)
        tr_path = plot_trade_analysis(best_result)
        html_path = generate_html_report(best_result, monthly, yearly, extra)
        print(f"\n  图表已保存: {eq_path}")
        print(f"  报告已保存: {html_path}")


def run_walk_forward(df, strategy_name: str):
    """滚动窗口优化"""
    print(f"\n{'='*70}")
    print(f"  滚动窗口优化 (Walk-Forward): {strategy_name}")
    print(f"{'='*70}")

    if strategy_name == "ma_cross":
        param_grid = {
            "ma_short": [5, 8, 10],
            "ma_long": [15, 20, 30],
            "stop_loss_pct": [0.04, 0.06, 0.08],
            "take_profit_pct": [0.10, 0.15, 0.20],
        }
    elif strategy_name == "pattern_enhanced":
        param_grid = {
            "ma_short": [5, 8],
            "ma_long": [20, 30],
            "pattern_weight": [0.2, 0.3, 0.4],
            "stop_loss_pct": [0.04, 0.06],
            "take_profit_pct": [0.12, 0.18],
        }
    else:
        param_grid = {
            "stop_loss_pct": [0.04, 0.06, 0.08],
            "take_profit_pct": [0.10, 0.15, 0.20],
        }

    optimizer = WalkForwardOptimizer(
        train_years=2.0,
        test_years=1.0,
        step_years=0.5,
    )

    wf_result = optimizer.run(df, strategy_name, param_grid)

    if not wf_result["windows"]:
        print("  滚动窗口优化失败 (数据不足)")
        return

    print(f"\n{'='*70}")
    print(f"  滚动窗口结果汇总")
    print(f"{'='*70}")

    print(f"\n  {'窗口':<6}{'训练期':<28}{'测试期':<28}{'收益':<10}{'夏普':<8}{'回撤':<8}{'交易':<6}")
    print(f"  {'-'*86}")
    for w in wf_result["windows"]:
        print(f"  {w['window']:<6}{w['train_period']:<28}{w['test_period']:<28}"
              f"{w['test_return']:>+7.2f}%  {w['test_sharpe']:>5.2f}  "
              f"{w['test_max_dd']:>5.1f}%  {w['test_trades']:>4}")

    print(f"\n  总收益率 (各窗口累加): {wf_result['total_return']:+.2f}%")
    print(f"  总交易次数: {wf_result['total_trades']}")
    print(f"  平均胜率: {wf_result['avg_win_rate']:.1f}%")
    print(f"  最常用参数: {wf_result['best_overall_params']}")

    # 各窗口参数变化趋势
    print(f"\n  参数演变:")
    for w in wf_result["windows"]:
        params = w["best_params"]
        print(f"    {w['test_period']:<28} → {params}")


def main():
    parser = argparse.ArgumentParser(
        description="联创电子波段策略回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", "-s", type=str, default=None,
                        help="指定策略名称 (不指定则运行全部)")
    parser.add_argument("--optimize", "-o", action="store_true",
                        help="参数网格搜索优化模式")
    parser.add_argument("--refresh", "-r", action="store_true",
                        help="强制重新下载数据")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有可用策略")
    parser.add_argument("--stop-loss", type=float, default=0.06,
                        help="止损比例 (默认0.06)")
    parser.add_argument("--take-profit", type=float, default=0.18,
                        help="止盈比例 (默认0.18)")
    parser.add_argument("--walk-forward", "-w", action="store_true",
                        help="滚动窗口优化 (Walk-Forward Analysis)")

    args = parser.parse_args()

    if args.list:
        print("\n可用波段策略:")
        list_strategies()
        return

    # 加载数据
    print(f"\n{'='*70}")
    print(f"  联创电子 (002036.SZ) 波段回测系统")
    print(f"{'='*70}")
    df = get_data(force_refresh=args.refresh)
    print(f"  数据: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()} "
          f"({len(df)} 个交易日)")

    if args.optimize:
        strategy_name = args.strategy or "composite"
        run_optimization(df, strategy_name)
        return

    if args.walk_forward:
        strategy_name = args.strategy or "ma_cross"
        run_walk_forward(df, strategy_name)
        return

    if args.strategy:
        # 单策略模式
        print(f"\n  策略: {args.strategy}")
        result = run_single_strategy(
            df, args.strategy,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
        )
        print(performance_summary(result))

        monthly = analyze_monthly(result.equity_curve)
        yearly = analyze_yearly(result.equity_curve)
        extra = trade_analysis(result.trades)

        print(f"\n  年度表现:")
        for y, row in yearly.iterrows():
            tag = "+" if row["ret"] > 0 else "-"
            print(f"    {y}: {tag} {row['ret']:+.2%}  (年内回撤: {row['drawdown']:.2%})")

        print(f"\n  交易行为分析:")
        for k, v in extra.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.2f}")
            else:
                print(f"    {k}: {v}")

        eq_path = plot_equity_curve(result)
        tr_path = plot_trade_analysis(result)
        html_path = generate_html_report(result, monthly, yearly, extra)
        print(f"\n  图表: {eq_path}")
        print(f"  图表: {tr_path}")
        print(f"  报告: {html_path}")

    else:
        # 多策略对比模式
        results = run_all_strategies(df)

        print(f"\n{'='*70}")
        print(f"  策略排名 (按卡尔玛比率)")
        print(f"{'='*70}")
        print(f"  {'排名':<6}{'策略':<28}{'年化收益':<12}{'夏普':<8}{'最大回撤':<10}{'卡尔玛':<8}{'胜率':<8}")
        print(f"  {'-'*70}")
        for i, r in enumerate(results, 1):
            print(f"  {i:<6}{r.strategy_name:<28}"
                  f"{r.annualized_return_pct:>+8.1f}%  "
                  f"{r.sharpe_ratio:>5.2f}  "
                  f"{r.max_drawdown_pct:>7.1f}%  "
                  f"{r.calmar_ratio:>5.2f}  "
                  f"{r.win_rate:>5.1f}%")

        # 生成对比图
        cmp_path = plot_strategy_comparison(results)
        print(f"\n  策略对比图: {cmp_path}")

        # 为最佳策略生成完整报告
        best = results[0]
        print(f"\n{'='*70}")
        print(f"  ★ 最佳策略: {best.strategy_name}")
        print(f"{'='*70}")
        print(performance_summary(best))

        monthly = analyze_monthly(best.equity_curve)
        yearly = analyze_yearly(best.equity_curve)
        extra = trade_analysis(best.trades)

        print(f"\n  最佳策略年度表现:")
        for y, row in yearly.iterrows():
            tag = "+" if row["ret"] > 0 else "-"
            print(f"    {y}: {tag} {row['ret']:+.2%}  "
                  f"(年内回撤: {row['drawdown']:.2%})")

        eq_path = plot_equity_curve(best)
        tr_path = plot_trade_analysis(best)
        html_path = generate_html_report(best, monthly, yearly, extra)
        print(f"\n  HTML报告: {html_path}")


if __name__ == "__main__":
    main()
