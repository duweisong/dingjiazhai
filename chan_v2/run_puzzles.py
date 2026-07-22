"""
缠论 v2.0 拼图补全 — 分钟级数据验证
====================================
三块拼图：
  1. 线段完成度评分验证 (>100样本)
  2. 背驰检测调参 (MACD面积比阈值扫描)
  3. 多级别递归联立 (30m→D→W)
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chan_v2 import (
    ChanAnalyzer, LEVEL_PARAMS, process_klines, detect_fractals, detect_fractals_v1,
    detect_strokes, KLine, Fractal, Stroke, Segment, Direction, FractalType,
    _calc_obv, compute_segment_completion
)
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# 数据获取
# ============================================================================

def fetch_minute_data(symbol: str = 'sh600519', period: str = '30',
                      start_date: str = '20200101', end_date: str = '20260701'):
    """
    获取个股分钟数据。茅台(sh600519)作为流动性标杆。
    period: '1','5','15','30','60'
    """
    try:
        import akshare as ak
        print(f"[数据] 获取 {symbol} {period}分钟线 ({start_date}~{end_date})...")

        # akshare 分钟数据分期获取
        all_data = []
        # 按年分批次
        years = range(int(start_date[:4]), int(end_date[:4]) + 1)
        for yr in years:
            for suffix in ['', '0331', '0630', '0930', '1231']:
                try:
                    end_d = f'{yr}{suffix}' if suffix else f'{yr}1231'
                    if end_d < start_date or end_d > end_date:
                        continue
                    df = ak.stock_zh_a_minute(
                        symbol=symbol, period=period, adjust='qfq'
                    )
                    if df is not None and len(df) > 0:
                        # Filter by date
                        if 'day' in df.columns:
                            df = df[(df['day'] >= start_date[:8])]
                            all_data.append(df)
                        break  # Got data for this year, move to next
                except Exception:
                    continue

        if not all_data:
            # Fallback: try single call
            df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust='qfq')
            all_data = [df]

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Standardize columns
            col_map = {'day': 'date', 'open': 'open', 'high': 'high',
                       'low': 'low', 'close': 'close', 'volume': 'volume'}
            result = result.rename(columns={k: v for k, v in col_map.items() if k in result.columns})

            # Combine date + time if needed
            if 'date' in result.columns and 'time' in result.columns:
                result['date'] = pd.to_datetime(
                    result['date'].astype(str) + ' ' + result['time'].astype(str)
                )
            elif 'date' in result.columns:
                result['date'] = pd.to_datetime(result['date'])

            result = result.drop_duplicates(subset=['date']).sort_values('date')
            print(f"  获取: {len(result)} 条 {period}分钟K线")
            return result
    except Exception as e:
        print(f"  akshare分钟数据失败: {e}")
    return None


def generate_minute_sample(n: int = 5000, start_price: float = 100.0) -> pd.DataFrame:
    """
    生成有真实特征的模拟分钟数据（akshare不可用时的fallback）。
    模拟特征：趋势+震荡+噪声，对缠论框架有挑战性。
    """
    np.random.seed(42)
    # 分段模拟：牛市→熊市→震荡→牛市
    segments = [
        (0, 1500, 0.0003, 0.012),   # 温和牛市
        (1500, 2500, -0.0005, 0.018), # 剧烈熊市
        (2500, 3500, 0.00005, 0.008), # 低波动震荡
        (3500, 5000, 0.0004, 0.014),  # 恢复性牛市
    ]

    prices = np.zeros(n)
    prices[0] = start_price

    for start, end, drift, vol in segments:
        for i in range(max(1, start), min(end, n)):
            returns = np.random.randn() * vol + drift
            # 加入少量跳跃噪声（模拟量化算法）
            if np.random.random() < 0.03:
                returns += np.random.randn() * vol * 3
            prices[i] = prices[i-1] * (1 + returns)

    dates = pd.date_range('2022-01-01 09:30', periods=n, freq='30min')
    # 只保留交易时段
    trading_mask = [(d.hour >= 9 and d.hour <= 15 and not (d.hour == 11 and d.minute > 30)) for d in dates]
    dates = dates[trading_mask]
    prices = prices[:len(dates)]

    data = []
    for i in range(len(dates)):
        open_p = prices[i] * (1 + np.random.randn() * 0.002)
        high = max(open_p, prices[i]) + abs(np.random.randn()) * prices[i] * 0.008
        low = min(open_p, prices[i]) - abs(np.random.randn()) * prices[i] * 0.008
        low = max(low, 0.01)
        vol = np.random.lognormal(10, 0.8)
        data.append({
            'date': dates[i], 'open': open_p, 'high': high,
            'low': low, 'close': prices[i], 'volume': vol,
        })

    print(f"[数据] 模拟 30分钟K线: {len(data)} 条")
    return pd.DataFrame(data)


# ============================================================================
# 拼图1: 线段完成度评分验证
# ============================================================================

def validate_completion_scoring(analyzer, horizon_bars: int = 48):
    """
    验证线段完成度评分：高分→高方向准确率。
    horizon_bars = 48根30分钟K线 = 3天
    """
    klines = analyzer.merged_klines
    segments = analyzer.segments

    results = []
    for seg in segments:
        if seg.completion_score == 0:
            # 手动计算完成度
            compute_segment_completion(seg, analyzer.strokes)

        score = seg.completion_score
        end_idx = seg.strokes[-1].end_fractal.index

        if end_idx + horizon_bars >= len(klines):
            continue

        # 线段方向预测 vs 实际走势
        current_price = klines[end_idx].close
        future_price = klines[end_idx + horizon_bars].close

        if seg.direction == Direction.UP:
            prediction_correct = future_price > current_price
        else:
            prediction_correct = future_price < current_price

        results.append({
            'score': score,
            'correct': prediction_correct,
            'change_pct': (future_price - current_price) / current_price * 100,
            'strokes_in_seg': len(seg.strokes),
        })

    if not results:
        return []

    # 按评分区间统计
    score_bins = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
    summary = []
    for lo, hi in score_bins:
        bucket = [r for r in results if lo <= r['score'] < hi]
        if bucket:
            win_rate = sum(1 for r in bucket if r['correct']) / len(bucket) * 100
            avg_change = np.mean([r['change_pct'] for r in bucket])
            summary.append({
                'score_range': f'{lo}-{hi}',
                'count': len(bucket),
                'win_rate': round(win_rate, 1),
                'avg_change': round(avg_change, 2),
            })

    return summary


# ============================================================================
# 拼图2: 背驰检测MACD面积比阈值扫描
# ============================================================================

def scan_divergence_thresholds(analyzer, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]):
    """
    扫描不同MACD面积比阈值下的背驰信号质量。
    """
    klines = analyzer.merged_klines
    closes = np.array([k.close for k in klines])

    # 计算MACD
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = 2 * (dif - dea)

    results = []
    for thresh in thresholds:
        signals = []
        for seg in analyzer.segments:
            if len(seg.strokes) < 5:
                continue
            last_stroke = seg.strokes[-1]
            # 找前一同向笔
            prev_same = None
            for s in reversed(seg.strokes[:-2]):
                if s.direction == last_stroke.direction:
                    prev_same = s
                    break
            if prev_same is None:
                continue

            idx_a, idx_b = prev_same.start_fractal.index, last_stroke.end_fractal.index
            if idx_b >= len(closes) or idx_a >= idx_b:
                continue

            seg_a_bars = macd_bar.iloc[idx_a:idx_b+1]
            seg_a_area = abs(seg_a_bars.sum())
            remaining = macd_bar.iloc[idx_b:]
            seg_b_len = min(len(seg_a_bars), len(remaining))
            seg_b_area = abs(remaining.iloc[:seg_b_len].sum())

            if seg_b_area < seg_a_area * thresh:
                # 检测到了背驰
                if last_stroke.direction == Direction.DOWN:
                    # 底背驰→买点
                    if idx_b + 100 < len(closes):
                        ret = (closes[idx_b+100] - closes[idx_b]) / closes[idx_b] * 100
                        signals.append({'type': 'BUY', 'return_100bar': ret, 'win': ret > 0})
                else:
                    # 顶背驰→卖点
                    if idx_b + 100 < len(closes):
                        ret = (closes[idx_b] - closes[idx_b+100]) / closes[idx_b] * 100
                        signals.append({'type': 'SELL', 'return_100bar': ret, 'win': ret > 0})

        if signals:
            win_rate = sum(1 for s in signals if s['win']) / len(signals) * 100
            avg_ret = np.mean([s['return_100bar'] for s in signals])
            results.append({
                'threshold': thresh,
                'signals': len(signals),
                'win_rate': round(win_rate, 1),
                'avg_return': round(avg_ret, 2),
            })

    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("  缠论 v2.0 — 拼图补全")
    print("  线段评分 + 背驰调参 + 多级别联立")
    print("=" * 70)

    # --- 获取分钟数据 ---
    print("\n[数据] 获取30分钟K线...")
    df = fetch_minute_data('sh600519', '30', '20230101', '20260701')
    if df is None or len(df) < 500:
        df = generate_minute_sample(5000)
        level = '30m'
    else:
        level = '30m'

    if df is None or len(df) < 100:
        print("[错误] 数据不足，退出")
        return

    # --- 30分钟级别分析 ---
    print(f"\n[分析] {level}级别 ({len(df)}根K线)...")
    analyzer = ChanAnalyzer(level)
    analyzer.load_klines(df)
    analyzer.run()

    summary = analyzer.get_summary()
    print(f"  K线: {summary['klines_raw']} → 合并: {summary['klines_merged']}")
    print(f"  分型: {summary['fractals_total']} (能量OK: {summary['fractals_energy_ok']}, 确认: {summary['fractals_confirmed']})")
    print(f"  笔: {summary['strokes_total']} → 有效: {summary['strokes_valid']} (过滤率: {summary['filter_rate_strokes']:.1%})")
    print(f"  线段: {summary['segments_total']}")
    print(f"  子线段: {len(analyzer.sub_segments)}")
    print(f"  买卖点: {summary['bps_total']}")

    valid_strokes = analyzer.get_valid_strokes()

    # --- 拼图1: 线段完成度评分验证 (用子线段) ---
    print(f"\n{'='*60}")
    print("  拼图1: 子线段完成度评分 → 方向准确率")
    print(f"{'='*60}")

    # 用子线段替代原线段
    test_segments = analyzer.sub_segments if analyzer.sub_segments else analyzer.segments
    print(f"  可用段数: {len(test_segments)}")

    if len(test_segments) > 5:
        # 临时替换用于评分验证
        orig_segments = analyzer.segments
        analyzer.segments = test_segments
        scoring = validate_completion_scoring(analyzer, horizon_bars=48)
        analyzer.segments = orig_segments
        if scoring:
            print(f"  ┌──────────┬────────┬──────────┬──────────┐")
            print(f"  │ 评分区间 │ 样本数 │ 方向正确率│ 平均涨跌 │")
            print(f"  ├──────────┼────────┼──────────┼──────────┤")
            for r in scoring:
                bar = "█" * int(r['win_rate'] / 5)
                print(f"  │ {r['score_range']:>8} │ {r['count']:>6} │ {r['win_rate']:>5.0f}% {bar} │ {r['avg_change']:>+.2f}% │")
            print(f"  └──────────┴────────┴──────────┴──────────┘")

            # 检查单调性（核心验证）
            win_rates = [r['win_rate'] for r in scoring]
            increasing = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))
            print(f"  单调递增？ {'✓ 是 — 评分系统有效' if increasing else '✗ 否 — 需调权重'}")
        else:
            print("  线段样本不足（需horizon后仍有数据）")
    else:
        print(f"  可用段数({len(test_segments)})不够，跳过评分验证（需>5段）")

    # --- 拼图2: 背驰阈值扫描 ---
    print(f"\n{'='*60}")
    print("  拼图2: MACD面积比阈值扫描 (100根K线后收益)")
    print(f"{'='*60}")

    if len(test_segments) > 3:
        analyzer.segments = test_segments
        div_results = scan_divergence_thresholds(analyzer)
        analyzer.segments = orig_segments
        if div_results:
            print(f"  ┌──────────┬────────┬──────────┬──────────┐")
            print(f"  │ 阈值     │ 信号数 │ 胜率     │ 平均收益 │")
            print(f"  ├──────────┼────────┼──────────┼──────────┤")
            best = max(div_results, key=lambda r: r['win_rate'] * r['avg_return'] if r['avg_return'] > 0 else 0)
            for r in div_results:
                marker = " ← 最优" if r == best else ""
                print(f"  │ {r['threshold']:>8.2f} │ {r['signals']:>6} │ {r['win_rate']:>5.0f}%   │ {r['avg_return']:>+.2f}%  {marker} │")
            print(f"  └──────────┴────────┴──────────┴──────────┘")
            print(f"  推荐阈值: {best['threshold']:.2f} (胜率{best['win_rate']:.0f}%, 均收益{best['avg_return']:+.2f}%)")
        else:
            print("  无背驰信号（需要>=5笔的线段）")
    else:
        print(f"  可用段数({len(test_segments)})不足，跳过背驰扫描（需>3段）")

    # --- 拼图3: 多级别递归验证 ---
    print(f"\n{'='*60}")
    print("  拼图3: 多级别递归联立 (30m → D)")
    print(f"{'='*60}")

    # 从30分钟数据合成日线
    if df is not None and len(df) > 0:
        df_d = df.copy()
        df_d['date_d'] = pd.to_datetime(df_d['date']).dt.date
        daily = df_d.groupby('date_d').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        daily = daily.rename(columns={'date_d': 'date'})

        print(f"  30分钟数据: {len(df)} 根 → 合成日线: {len(daily)} 根")

        if len(daily) > 50:
            analyzer_d = ChanAnalyzer('D')
            analyzer_d.load_klines(daily)
            analyzer_d.run()

            d_sum = analyzer_d.get_summary()
            print(f"  日线分析:")
            print(f"    笔: {d_sum['strokes_total']} → 有效: {d_sum['strokes_valid']}")
            print(f"    线段: {d_sum['segments_total']}")
            print(f"    买卖点: {d_sum['bps_total']}")

            # 递归验证：日线笔的方向 vs 30分钟线段的方向
            d_valid = analyzer_d.get_valid_strokes()
            m30_valid = analyzer.get_valid_strokes()

            # 简化验证：检查日线向上笔期间，30分钟向上笔占优
            if d_valid and m30_valid:
                d_stroke_dirs = []
                for ds in d_valid:
                    # Use date objects for comparison
                    try:
                        d_start_d = ds.start_fractal.mid_kline.timestamp.date() if hasattr(ds.start_fractal.mid_kline.timestamp, 'date') else ds.start_fractal.mid_kline.timestamp
                        d_end_d = ds.end_fractal.mid_kline.timestamp.date() if hasattr(ds.end_fractal.mid_kline.timestamp, 'date') else ds.end_fractal.mid_kline.timestamp
                    except AttributeError:
                        d_start_d = pd.Timestamp(ds.start_fractal.mid_kline.timestamp).date()
                        d_end_d = pd.Timestamp(ds.end_fractal.mid_kline.timestamp).date()

                    m30_in = []
                    for s in m30_valid:
                        try:
                            s_d = s.start_fractal.mid_kline.timestamp.date() if hasattr(s.start_fractal.mid_kline.timestamp, 'date') else s.start_fractal.mid_kline.timestamp
                        except AttributeError:
                            s_d = pd.Timestamp(s.start_fractal.mid_kline.timestamp).date()
                        if d_start_d <= s_d <= d_end_d:
                            m30_in.append(s)
                    if m30_in:
                        up_count = sum(1 for s in m30_in if s.direction == Direction.UP)
                        dn_count = len(m30_in) - up_count
                        if ds.direction == Direction.UP:
                            d_stroke_dirs.append(up_count >= dn_count)
                        else:
                            d_stroke_dirs.append(dn_count >= up_count)

                if d_stroke_dirs:
                    multi_level_accuracy = sum(d_stroke_dirs) / len(d_stroke_dirs) * 100
                    print(f"  多级别一致性:")
                    print(f"    日线笔方向 vs 30分钟笔方向: {multi_level_accuracy:.0f}% 一致")
                    print(f"    样本: {len(d_stroke_dirs)} 根日线笔覆盖了足够的30分钟笔")
                    if multi_level_accuracy >= 70:
                        print(f"    ✓ 级别递归成立：大级别方向包含小级别方向")
                    else:
                        print(f"    ✗ 级别递归不成立：需检查参数设置")

    # --- 输出完整报告 ---
    report = f"""======================================================================
  缠论 v2.0 完整验证报告
  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}
======================================================================

数据: {level}级别, {len(df)}根K线, {datetime.now().strftime('%Y-%m')}

拼图1: 线段完成度评分 → 高分高胜率单调递增 ✓/✗ (需 >5段验证)
拼图2: MACD面积比最优阈值 → 胜率/收益最大化点
拼图3: 多级别递归 30m→D → 大方向包容小方向

已交付:
• chan_v2/chan_v2.py — 核心引擎 (~650 lines)
• chan_v2/run_backtest.py — 日线回测脚本
• chan_v2/run_puzzles.py — 分钟线验证脚本
• 级别参数表 — 6级别, 可调

待社区验证:
• 3人独立一致性测试
• 实盘paper trading
• 全A股扫描 + 分钟级回溯
======================================================================"""
    print(f"\n{report}")


if __name__ == "__main__":
    main()
