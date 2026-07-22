"""
缠论 v2.0 — 沪深300 笔信号策略回测
===================================
策略:
  v2有效DOWN笔完成(底分型) → 买入(收盘价)
  v2有效UP笔完成(顶分型)   → 卖出(收盘价)
  多空: 仅做多 + 现金 (不做空)

对比: 买入持有 (Buy & Hold)
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chan_v2 import ChanAnalyzer, Direction
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# 数据
# ============================================================================

def fetch_csi300():
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol='sh000300')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '20150101') & (df['date'] <= '20260701')]
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ============================================================================
# 策略回测
# ============================================================================

def backtest(df):
    """基于v2笔信号的多空策略"""
    a = ChanAnalyzer('D')
    a.load_klines(df)
    a.run()

    valid = a.get_valid_strokes()
    closes = np.array([k.close for k in a.merged_klines])
    dates   = [k.timestamp for k in a.merged_klines]

    # 提取信号: (date_index, signal_type)
    # DOWN笔完成→BUY, UP笔完成→SELL
    # 执行点 = 分型结束后 +2根K线 (等分型确认, 消除前视偏差)
    signals = []
    for st in valid:
        ei = st.end_fractal.index + 2  # +2 for fractal confirmation
        if ei >= len(dates): continue
        sig = 'BUY' if st.direction == Direction.DOWN else 'SELL'
        signals.append({'idx': ei, 'date': dates[ei], 'signal': sig,
                        'price': closes[ei], 'amp': st.amplitude,
                        'stroke_end_idx': st.end_fractal.index})  # original end for hold calc

    if not signals:
        return None, None

    sig_df = pd.DataFrame(signals).sort_values('idx').reset_index(drop=True)

    # 模拟交易
    position = 0   # 0=cash, 1=long
    cash = 1.0
    shares = 0.0
    equity = np.ones(len(closes))
    trades = []
    entry_price = 0
    entry_date = None

    sig_idx = 0
    for i in range(len(closes)):
        price = closes[i]

        # 执行信号
        while sig_idx < len(sig_df) and sig_df.iloc[sig_idx]['idx'] == i:
            s = sig_df.iloc[sig_idx]
            if s['signal'] == 'BUY' and position == 0:
                shares = cash / price
                cash = 0.0
                position = 1
                entry_price = price
                entry_date = s['date']
                entry_idx = i
            elif s['signal'] == 'SELL' and position == 1:
                cash = shares * price
                shares = 0.0
                position = 0
                ret = (price - entry_price) / entry_price
                hold_bars = i - entry_idx
                trades.append({
                    'entry_date': entry_date, 'exit_date': s['date'],
                    'entry_price': entry_price, 'exit_price': price,
                    'return': ret, 'win': ret > 0,
                    'hold_days': hold_bars,
                })
            sig_idx += 1

        equity[i] = cash + shares * price

    # Buy & Hold
    bh_equity = closes / closes[0]

    # 指标
    def calc_metrics(eq, name=''):
        rets = np.diff(eq) / eq[:-1]
        # Handle zero-division
        rets = rets[np.isfinite(rets)]

        total_ret = (eq[-1] / eq[0] - 1) * 100
        years = len(eq) / 252
        cagr = ((eq[-1] / eq[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

        excess = rets - 0.02 / 252  # risk-free = 2%
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = np.min(dd) * 100

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        return {
            'name': name,
            'total_ret': round(total_ret, 1),
            'cagr': round(cagr, 2),
            'sharpe': round(sharpe, 2),
            'max_dd': round(max_dd, 1),
            'calmar': round(calmar, 2),
            'years': round(years, 1),
        }

    strat_m = calc_metrics(equity, 'v2 Stroke Strategy')
    bh_m = calc_metrics(bh_equity, 'Buy & Hold')

    # Trade stats
    if trades:
        tdf = pd.DataFrame(trades)
        win_rate = tdf['win'].mean() * 100
        avg_ret = tdf['return'].mean() * 100
        avg_win = tdf[tdf['win']]['return'].mean() * 100 if tdf['win'].any() else 0
        avg_loss = tdf[~tdf['win']]['return'].mean() * 100 if (~tdf['win']).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        avg_hold = tdf['hold_days'].mean()
        n_trades = len(trades)
    else:
        win_rate = avg_ret = avg_win = avg_loss = profit_factor = avg_hold = n_trades = 0

    trade_stats = {
        'n_trades': n_trades, 'win_rate': round(win_rate, 1),
        'avg_ret': round(avg_ret, 2), 'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2), 'profit_factor': round(profit_factor, 2) if profit_factor < 999 else 999,
        'avg_hold': round(avg_hold, 0),
    }

    return strat_m, bh_m, trade_stats, equity, bh_equity, dates, sig_df


# ============================================================================
# 分年统计
# ============================================================================

def yearly_breakdown(equity, bh_equity, dates):
    """逐年收益对比"""
    years = set()
    for d in dates:
        try:
            years.add(d.year)
        except:
            years.add(pd.Timestamp(d).year)

    print(f"\n  {'年份':<6} {'策略收益':>10} {'持有收益':>10} {'超额':>10} {'胜出':>6}")
    print(f"  {'-'*45}")
    for yr in sorted(years):
        mask = [i for i, d in enumerate(dates) if (d.year if hasattr(d,'year') else pd.Timestamp(d).year) == yr]
        if len(mask) < 10: continue
        s_ret = (equity[mask[-1]] / equity[mask[0]] - 1) * 100
        b_ret = (bh_equity[mask[-1]] / bh_equity[mask[0]] - 1) * 100
        alpha = s_ret - b_ret
        win = '✓' if alpha > 0 else ''
        print(f"  {yr:<6} {s_ret:>+9.1f}% {b_ret:>+9.1f}% {alpha:>+9.1f}% {win:>6}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  沪深300 · v2笔信号策略回测")
    print("  DOWN笔完成→买入  |  UP笔完成→卖出  |  纯多头+现金")
    print("=" * 70)

    print("\n[1] 获取沪深300数据...")
    df = fetch_csi300()
    print(f"   {len(df)} 根日K线 ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')})")

    print("\n[2] 运行v2分析...")
    a = ChanAnalyzer('D')
    a.load_klines(df)
    a.run()
    s = a.get_summary()
    valid = a.get_valid_strokes()
    print(f"   笔: {s['strokes_total']} → {s['strokes_valid']} 有效 ({s['filter_rate_strokes']:.0%} 过滤)")
    print(f"   子线段: {len(a.sub_segments)}, 买卖点: {s['bps_total']}")

    # 信号分布
    buy_sigs = sum(1 for st in valid if st.direction == Direction.DOWN)
    sell_sigs = sum(1 for st in valid if st.direction == Direction.UP)
    print(f"   信号: BUY={buy_sigs}, SELL={sell_sigs}, 配对交易≈{min(buy_sigs, sell_sigs)}")

    print("\n[3] 策略回测...")
    strat_m, bh_m, trade_stats, equity, bh_equity, dates, sig_df = backtest(df)

    if strat_m is None:
        print("   信号不足")
        return

    # 对比表
    print(f"\n{'='*70}")
    print(f"  策略 vs 基准")
    print(f"{'='*70}")
    print(f"  {'指标':<16} {'v2笔策略':>12} {'买入持有':>12} {'差异':>12}")
    print(f"  {'-'*55}")
    for key, label in [('total_ret','总收益%'), ('cagr','年化%'), ('sharpe','夏普比率'),
                        ('max_dd','最大回撤%'), ('calmar','Calmar'), ('years','回测年数')]:
        sv = strat_m[key]
        bv = bh_m[key]
        diff = sv - bv if isinstance(sv, (int, float)) else 0
        sign = '+' if diff > 0 else ''
        print(f"  {label:<16} {sv:>12} {bv:>12} {sign}{diff:>11}")

    print(f"\n  交易明细:")
    print(f"  {'指标':<16} {'值':>12}")
    for key, label in [('n_trades','交易次数'), ('win_rate','胜率%'),
                        ('avg_ret','均收益%'), ('avg_win','均盈%'),
                        ('avg_loss','均亏%'), ('profit_factor','盈亏比'),
                        ('avg_hold','均持仓(天)')]:
        print(f"  {label:<16} {trade_stats[key]:>12}")

    # 分年
    print(f"\n{'='*70}")
    print(f"  逐年收益")
    print(f"{'='*70}")
    yearly_breakdown(equity, bh_equity, dates)

    # 关键节点
    print(f"\n{'='*70}")
    print(f"  关键节点表现")
    print(f"{'='*70}")
    # 找到2015股灾、2018熊市、2020疫情、2024熊市
    key_periods = [
        ('2015股灾', '2015-06-12', '2015-08-26'),
        ('2018熊市', '2018-01-24', '2019-01-04'),
        ('2020疫情', '2020-01-14', '2020-03-23'),
        ('2024熊市', '2024-05-20', '2024-09-13'),
        ('924行情', '2024-09-24', '2024-10-08'),
    ]
    for label, start_s, end_s in key_periods:
        try:
            sd = pd.Timestamp(start_s); ed = pd.Timestamp(end_s)
            si = next((i for i,d in enumerate(dates) if d >= sd), None)
            ei = next((i for i,d in enumerate(dates) if d >= ed), None)
            if si is not None and ei is not None and si < ei:
                s_ret = (equity[ei] / equity[si] - 1) * 100
                b_ret = (bh_equity[ei] / bh_equity[si] - 1) * 100
                print(f"  {label:<12}: 策略{s_ret:>+6.1f}%  持有{b_ret:>+6.1f}%  防御{b_ret-s_ret:>+5.1f}%")
        except:
            pass

    # 结论
    print(f"\n{'='*60}")
    print(f"  结论")
    print(f"{'='*60}")
    alpha = strat_m['cagr'] - bh_m['cagr']
    dd_improve = abs(bh_m['max_dd']) - abs(strat_m['max_dd'])
    print(f"  v2笔策略 vs 买入持有 (2015-2026)")
    print(f"  年化: {strat_m['cagr']}% vs {bh_m['cagr']}% (α={alpha:+.1f}%/年)")
    print(f"  夏普: {strat_m['sharpe']} vs {bh_m['sharpe']}")
    print(f"  最大回撤: {strat_m['max_dd']}% vs {bh_m['max_dd']}% (改善{dd_improve:+.0f}%)")
    print(f"  胜率: {trade_stats['win_rate']}% · 盈亏比: {trade_stats['profit_factor']} · 交易{trade_stats['n_trades']}次")
    verdict = '✓ 策略优于持有' if alpha > 1 and strat_m['sharpe'] > bh_m['sharpe'] else ('△ 部分改善' if dd_improve > 5 else '✗ 未优于持有')
    print(f"  判定: {verdict}")


if __name__ == '__main__':
    main()
