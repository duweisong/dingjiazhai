"""
缠论 v2.0 — 10股批量回测
=========================
覆盖消费/金融/科技/医药/周期五个板块，统计v2过滤效果。
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chan_v2 import ChanAnalyzer, LEVEL_PARAMS, Direction
import pandas as pd
import numpy as np
from datetime import datetime

# 10只代表性格股
STOCKS = [
    ('600519', '贵州茅台', '消费'),
    ('000858', '五粮液',   '消费'),
    ('601318', '中国平安', '金融'),
    ('600036', '招商银行', '金融'),
    ('300750', '宁德时代', '新能源'),
    ('002594', '比亚迪',   '新能源'),
    ('600276', '恒瑞医药', '医药'),
    ('000333', '美的集团', '家电'),
    ('002415', '海康威视', '科技'),
    ('600585', '海螺水泥', '周期'),
]


def fetch_one(symbol: str, start: str = '20180101', end: str = '20260701'):
    """获取个股日线 (sina source, reliable)"""
    import time
    prefix = 'sh' if symbol.startswith(('6', '9')) else 'sz'
    full_sym = f'{prefix}{symbol}'

    for attempt in range(3):
        try:
            import akshare as ak
            df = ak.stock_zh_a_daily(symbol=full_sym, adjust='qfq')
            if df is not None and len(df) > 200:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start) & (df['date'] <= end)]
                if len(df) < 50:
                    return None
                return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"    {symbol} 失败: {e}")
    return None


def analyze_one(df, symbol, name, sector):
    """运行v2分析，返回指标字典"""
    a = ChanAnalyzer('D')
    a.load_klines(df)
    a.run()

    s = a.get_summary()
    valid = a.get_valid_strokes()

    # 方向正确率
    correct = 0
    for st in valid:
        if st.direction == Direction.UP:
            correct += 1 if st.end_fractal.mid_kline.close > st.start_fractal.mid_kline.close else 0
        else:
            correct += 1 if st.end_fractal.mid_kline.close < st.start_fractal.mid_kline.close else 0

    dir_acc = round(correct / len(valid) * 100, 1) if valid else 0

    # 振幅统计
    amps = [st.amplitude for st in valid] if valid else [0]

    return {
        'symbol': symbol, 'name': name, 'sector': sector,
        'klines': s['klines_raw'],
        'merged': s['klines_merged'],
        'fractals': s['fractals_total'],
        'f_energy_ok': s['fractals_energy_ok'],
        'f_confirmed': s['fractals_confirmed'],
        'strokes_all': s['strokes_total'],
        'strokes_valid': s['strokes_valid'],
        'filter_rate': round(s['filter_rate_strokes'] * 100, 1),
        'dir_acc': dir_acc,
        'avg_amp': round(np.mean(amps), 1),
        'med_amp': round(np.median(amps), 1),
        'segments': s['segments_total'],
        'sub_segs': len(a.sub_segments),
        'bsp': s['bps_total'],
    }


def main():
    print("=" * 80)
    print("  缠论 v2.0 — 10股批量回测")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    results = []
    for symbol, name, sector in STOCKS:
        print(f"\n[{name} {symbol}] 获取数据...")
        df = fetch_one(symbol)
        if df is None or len(df) < 200:
            print(f"  跳过 (数据不足)")
            continue
        print(f"  {len(df)} 根日K线, 分析中...")
        r = analyze_one(df, symbol, name, sector)
        results.append(r)
        print(f"  笔: {r['strokes_all']}→{r['strokes_valid']} ({r['filter_rate']}%过滤) "
              f"方向:{r['dir_acc']}% 振幅均值:{r['avg_amp']}% 子段:{r['sub_segs']}")

    if not results:
        print("无有效结果")
        return

    # 汇总表
    print(f"\n{'='*80}")
    print("  10股汇总对比")
    print(f"{'='*80}")
    print(f"  {'名称':<8} {'板块':<6} {'K线':>5} {'笔(全)':>6} {'笔(有效)':>8} {'过滤%':>6} {'方向%':>6} {'均振幅%':>7} {'子段':>4}")
    print(f"  {'-'*70}")

    total_strokes_all, total_strokes_valid, total_correct = 0, 0, 0
    for r in results:
        print(f"  {r['name']:<8} {r['sector']:<6} {r['klines']:>5} {r['strokes_all']:>6} "
              f"{r['strokes_valid']:>8} {r['filter_rate']:>5.1f}% {r['dir_acc']:>5.1f}% "
              f"{r['avg_amp']:>6.1f}% {r['sub_segs']:>4}")
        total_strokes_all += r['strokes_all']
        total_strokes_valid += r['strokes_valid']

    print(f"  {'-'*70}")
    avg_filter = round((1 - total_strokes_valid / total_strokes_all) * 100, 1) if total_strokes_all else 0
    avg_dir = round(np.mean([r['dir_acc'] for r in results]), 1)
    avg_amp = round(np.mean([r['avg_amp'] for r in results]), 1)
    print(f"  {'汇总':<8} {'':<6} {'':>5} {total_strokes_all:>6} "
          f"{total_strokes_valid:>8} {avg_filter:>5.1f}% {avg_dir:>5.1f}% "
          f"{avg_amp:>6.1f}%")

    # 按板块汇总
    print(f"\n  --- 按板块 ---")
    sectors = {}
    for r in results:
        sec = r['sector']
        if sec not in sectors:
            sectors[sec] = {'all': 0, 'valid': 0, 'dirs': [], 'amps': []}
        sectors[sec]['all'] += r['strokes_all']
        sectors[sec]['valid'] += r['strokes_valid']
        sectors[sec]['dirs'].append(r['dir_acc'])
        sectors[sec]['amps'].append(r['avg_amp'])

    for sec, d in sectors.items():
        f_rate = round((1 - d['valid'] / d['all']) * 100, 1) if d['all'] else 0
        print(f"  {sec:<6}: 过滤率{f_rate:>5.1f}%  方向{np.mean(d['dirs']):>5.1f}%  "
              f"振幅{np.mean(d['amps']):>4.1f}%")

    # 结论
    print(f"\n{'='*60}")
    print(f"  结论")
    print(f"{'='*60}")
    print(f"  • 10股平均笔过滤率: {avg_filter}% (v1噪声清除)")
    print(f"  • 10股平均方向正确率: {avg_dir}% (v2有效笔)")
    print(f"  • 10股平均有效笔振幅: {avg_amp}% (非噪声信号)")
    print(f"  • 过滤一致性: {'✓ 稳定' if avg_filter > 70 else '需调参'} (跨板块)")
    print(f"  • 方向一致性: {'✓ 可靠' if avg_dir > 95 else '需检查'} (跨板块)")
    print(f"  • v2参数表跨板块通用: {'✓' if avg_dir > 90 else '需板块定制'}")


if __name__ == '__main__':
    main()
