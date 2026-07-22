"""
100万实盘模拟 · v2+趋势效率 组合策略
=====================================
选股: 沪深300中v2策略α最高5只 (2018-2023训练, 2024-2026验证)
仓位: 等权20万/只, 趋势效率调节
起始: 2024-01-02, 1,000,000
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chan_v2 import ChanAnalyzer, Direction
import pandas as pd
import numpy as np
from datetime import datetime
import time

STOCKS = [
    ('601318','中国平安'),('600036','招商银行'),('601166','兴业银行'),
    ('600030','中信证券'),('601398','工商银行'),('601939','建设银行'),
    ('000001','平安银行'),('600016','民生银行'),('601328','交通银行'),
    ('601601','中国太保'),('601628','中国人寿'),('601688','华泰证券'),
    ('600837','海通证券'),('000776','广发证券'),
    ('600519','贵州茅台'),('000858','五粮液'),('000568','泸州老窖'),
    ('600809','山西汾酒'),('002304','洋河股份'),('000596','古井贡酒'),
    ('600887','伊利股份'),('603288','海天味业'),('002714','牧原股份'),
    ('000876','新希望'),('600690','海尔智家'),('000333','美的集团'),
    ('000651','格力电器'),('002050','三花智控'),('600104','上汽集团'),
    ('000625','长安汽车'),('601238','广汽集团'),
    ('300750','宁德时代'),('002594','比亚迪'),('300014','亿纬锂能'),
    ('601012','隆基绿能'),('002459','晶澳科技'),('300274','阳光电源'),
    ('300316','晶盛机电'),('002460','赣锋锂业'),('603799','华友钴业'),
    ('002415','海康威视'),('002230','科大讯飞'),('603501','韦尔股份'),
    ('002475','立讯精密'),('688981','中芯国际'),('688111','金山办公'),
    ('688008','澜起科技'),('688012','中微公司'),('600570','恒生电子'),
    ('300454','深信服'),('002371','北方华创'),('600745','闻泰科技'),
    ('600276','恒瑞医药'),('300760','迈瑞医疗'),('603259','药明康德'),
    ('300015','爱尔眼科'),('000538','云南白药'),('600196','复星医药'),
    ('300122','智飞生物'),('600436','片仔癀'),('000999','华润三九'),
    ('600585','海螺水泥'),('600031','三一重工'),('000157','中联重科'),
    ('601668','中国建筑'),('600019','宝钢股份'),('601800','中国交建'),
    ('601985','中国核电'),('600900','长江电力'),('601857','中国石油'),
    ('600028','中国石化'),('601088','中国神华'),('600660','福耀玻璃'),
    ('603993','洛阳钼业'),('601100','恒立液压'),('600406','国电南瑞'),
    ('000002','万科A'),('600048','保利发展'),('001979','招商蛇口'),
    ('002352','顺丰控股'),('601888','中国中免'),('300059','东方财富'),
    ('000100','TCL科技'),('000725','京东方A'),('002142','宁波银行'),
    ('600050','中国联通'),('600795','国电电力'),('003816','中国广核'),
    ('600346','恒力石化'),('600845','宝信软件'),
    ('000425','徐工机械'),('002129','TCL中环'),('300124','汇川技术'),
    ('300498','温氏股份'),('000063','中兴通讯'),('000338','潍柴动力'),
    ('603369','今世缘'),('300413','芒果超媒'),
]

INITIAL = 1_000_000
WEIGHT_PER = INITIAL / 5  # 20万/只

def fetch(symbol):
    prefix = 'sh' if symbol.startswith(('6','9')) else 'sz'
    for _ in range(2):
        try:
            import akshare as ak
            df = ak.stock_zh_a_daily(symbol=f'{prefix}{symbol}', adjust='qfq')
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except: time.sleep(1)
    return None

def efficiency_ratio(prices, window=40):
    er = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        seg = prices[i-window:i+1]
        net = abs(seg[-1]-seg[0])
        path = np.sum(np.abs(np.diff(seg)))
        er[i] = net/path if path>0 else 0
    er[:window] = er[window] if window<len(er) else 0.5
    return er

def regime(er_val):
    if np.isnan(er_val): return 'semi', 0.5
    if er_val > 0.30: return 'trend', 1.0
    if er_val < 0.12: return 'chop', 1.0
    return 'semi', 0.5

def backtest_stock(df, name, test_start='2024-01-01'):
    """单只股票 v2+趋势效率, 返回日度权益曲线"""
    try:
        a = ChanAnalyzer('D'); a.load_klines(df); a.run()
    except: return None

    valid = a.get_valid_strokes()
    if len(valid) < 3: return None

    closes = np.array([k.close for k in a.merged_klines])
    dates = [k.timestamp for k in a.merged_klines]
    er = efficiency_ratio(closes, 40)

    # Find test period start
    test_start_dt = pd.Timestamp(test_start)
    start_i = 0
    for i, d in enumerate(dates):
        if d >= test_start_dt:
            start_i = i; break

    # Generate signals for entire period, execute only in test period
    signals = []
    for st in valid:
        ei = st.end_fractal.index + 2
        if ei >= len(closes) or ei >= len(er): continue
        r, w = regime(er[ei])
        sig = 'BUY' if st.direction == Direction.DOWN else 'SELL'
        signals.append({'idx': ei, 'sig': sig, 'weight': w, 'regime': r})
    signals.sort(key=lambda x: x['idx'])

    if len(signals) < 3: return None

    # Simulate full period equity (for ranking) and test-only (for portfolio)
    pos = 0; cash = 1.0; shares = 0.0
    equity = np.ones(len(closes))
    entry_price = 0; si = 0

    for i in range(len(closes)):
        price = closes[i]
        while si < len(signals) and signals[si]['idx'] <= i:
            s = signals[si]
            if s['sig'] == 'BUY' and pos == 0:
                alloc = s['weight']
                shares = (cash * alloc) / price; cash -= cash * alloc; pos = 1
                entry_price = price
            elif s['sig'] == 'SELL' and pos == 1:
                cash += shares * price; shares = 0.0; pos = 0
            si += 1
        equity[i] = cash + shares * price

    if pos == 1:
        cash += shares * closes[-1]; shares = 0.0
        equity[-1] = cash

    # Test period returns
    if start_i < len(equity):
        test_eq = equity[start_i:] / equity[start_i]
    else:
        test_eq = None

    # Full period metrics for ranking
    years_full = len(closes) / 252
    cagr = ((equity[-1]/equity[0])**(1/years_full)-1)*100 if years_full>0 else 0

    return {
        'name': name,
        'cagr_full': cagr,
        'test_eq': test_eq,
        'test_dates': dates[start_i:] if start_i < len(dates) else None,
        'n_signals': len(signals),
        'equity_full': equity,
        'closes': closes,
        'dates': dates,
    }


def main():
    print("=" * 70)
    print("  100万实盘模拟 · v2+趋势效率 组合策略")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Phase 1: Training (2018-2023) — rank stocks
    print("\n[1] 训练期 (2018-2023) 排序...")
    train_results = []
    for i, (code, name) in enumerate(STOCKS):
        if (i+1) % 20 == 0: print(f"  ... {i+1}/{len(STOCKS)}")
        df = fetch(code)
        if df is None: continue
        # Use data up to 2023-12-31 for training
        df_train = df[df['date'] <= '2023-12-31'].copy()
        if len(df_train) < 200: continue
        r = backtest_stock(df_train, name, test_start='2018-01-01')
        if r: train_results.append(r)

    # Compute buy-and-hold for alpha
    for r in train_results:
        bh_cagr = ((r['closes'][-1]/r['closes'][0])**(1/(len(r['closes'])/252))-1)*100 if len(r['closes'])>252 else 0
        r['bh_cagr'] = bh_cagr
        r['alpha'] = r['cagr_full'] - bh_cagr

    # Rank by ALPHA (not CAGR)
    train_results.sort(key=lambda x: x['alpha'], reverse=True)
    print(f"\n  训练期 TOP 10 (按策略α=超额收益):")
    for i, r in enumerate(train_results[:10]):
        print(f"  {i+1}. {r['name']:<8} α={r['alpha']:>+6.1f}%  CAGR={r['cagr_full']:>+6.1f}%  "
              f"持有={r['bh_cagr']:>+6.1f}%  信号={r['n_signals']}")

    # Select top 5
    top5 = train_results[:5]
    top5_names = [r['name'] for r in top5]
    print(f"\n  入选组合: {', '.join(top5_names)}")

    # Phase 2: Test (2024-2026)
    print(f"\n[2] 验证期 (2024-01-02 ~ 2026-07-01) 模拟...")

    # Collect all equity curves with dates
    eq_curves = {}  # date -> {stock_name: equity}
    stock_details = []

    for i, (code, name) in enumerate(STOCKS):
        if name not in top5_names: continue
        print(f"  分析 {name}...")
        df = fetch(code)
        if df is None: continue
        r = backtest_stock(df, name, test_start='2024-01-01')
        if r is None or r['test_eq'] is None: continue
        stock_details.append(r)

        for j, d in enumerate(r['test_dates']):
            d_key = d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()
            if d_key not in eq_curves:
                eq_curves[d_key] = {}
            eq_curves[d_key][r['name']] = r['test_eq'][j]

    # Build aligned DataFrame
    df_eq = pd.DataFrame(eq_curves).T.sort_index()
    df_eq = df_eq.ffill()  # Forward fill: hold last known value on non-trading days

    # Equal weight portfolio
    for name in top5_names:
        if name not in df_eq.columns:
            df_eq[name] = 1.0  # Default if missing
    df_eq = df_eq[top5_names]  # Reorder

    portfolio_eq = np.zeros(len(df_eq))
    for name in top5_names:
        portfolio_eq += df_eq[name].values * WEIGHT_PER

    portfolio_dates = df_eq.index.tolist()

    # Compute metrics
    rets_raw = np.diff(portfolio_eq) / portfolio_eq[:-1]
    rets = rets_raw[np.isfinite(rets_raw)]

    final_val = portfolio_eq[-1]
    total_ret = (final_val / INITIAL - 1) * 100
    years = len(portfolio_eq) / 252
    cagr = ((final_val / INITIAL) ** (1/years) - 1) * 100 if years > 0 else 0
    excess = rets - 0.02/252
    sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess)>0 else 0
    peak = np.maximum.accumulate(portfolio_eq)
    dd = (portfolio_eq - peak) / peak
    max_dd = float(np.min(dd[~np.isnan(dd)])) * 100 if len(dd[~np.isnan(dd)]) > 1 else 0
    win_days = (rets > 0).mean() * 100

    # CSI 300 benchmark
    df_bench = fetch('000300')  # CSI 300 itself
    bench_ret = None
    if df_bench is not None:
        b_closes = df_bench['close'].values
        b_dates = pd.to_datetime(df_bench['date'])
        b_start = 0
        for i, d in enumerate(b_dates):
            if d >= pd.Timestamp('2024-01-01'): b_start = i; break
        if b_start < len(b_closes):
            b_eq = b_closes[b_start:] / b_closes[b_start]
            # Align lengths
            min_len = min(len(portfolio_eq), len(b_eq))
            b_eq = b_eq[:min_len]
            bench_total = (b_eq[-1]/b_eq[0]-1)*100
            bench_ret = bench_total

    # Output
    print(f"\n{'='*70}")
    print(f"  100万实盘模拟结果")
    print(f"{'='*70}")
    print(f"\n  组合: {', '.join(top5_names)} (等权20万/只)")
    print(f"  策略: v2笔信号 + 趋势效率仓位调节")
    print(f"  期间: 2024-01-02 ~ {portfolio_dates[-1].strftime('%Y-%m-%d') if hasattr(portfolio_dates[-1], 'strftime') else '2026-07-01'}")

    print(f"\n  ┌──────────────────────┬──────────┐")
    print(f"  │ 指标                  │ 值       │")
    print(f"  ├──────────────────────┼──────────┤")
    print(f"  │ 初始资金              │ ¥1,000,000 │")
    print(f"  │ 最终资金              │ ¥{final_val:,.0f} │")
    print(f"  │ 总收益                │ {total_ret:+.1f}% │")
    print(f"  │ 年化收益              │ {cagr:+.1f}% │")
    print(f"  │ 夏普比率              │ {sharpe:.2f} │")
    print(f"  │ 最大回撤              │ {max_dd:.1f}% │")
    print(f"  │ 日胜率                │ {win_days:.0f}% │")
    if bench_ret is not None:
        print(f"  │ 沪深300同期           │ {bench_ret:+.1f}% │")
        print(f"  │ 超额α                 │ {total_ret-bench_ret:+.1f}% │")
    print(f"  └──────────────────────┴──────────┘")

    # Per-stock
    print(f"\n  各持仓表现:")
    print(f"  {'名称':<10} {'期初':>8} {'期末':>10} {'收益率':>8}")
    print(f"  {'-'*40}")
    for name in top5_names:
        if name in df_eq.columns:
            vals = df_eq[name].values
            start_v = vals[0] if vals[0] > 0 else 1
            end_v = vals[-1]
            ret = (end_v/start_v - 1) * 100
            print(f"  {name:<10} ¥{WEIGHT_PER/10000:>7.0f}万 ¥{end_v*WEIGHT_PER/10000:>9.0f}万 {ret:>+7.1f}%")

    # Monthly breakdown
    if len(portfolio_dates) > 0:
        print(f"\n  月度收益:")
        monthly = {}
        for i in range(1, len(portfolio_eq)):
            d = portfolio_dates[i]
            key = f"{d.year}-{d.month:02d}" if hasattr(d, 'year') else str(d)[:7]
            if key not in monthly: monthly[key] = []
            if i-1 < len(rets_raw):
                monthly[key].append(rets_raw[i-1])
        print(f"  {'月份':<8} {'收益%':>8} {'累计净值':>10}")
        cumulative = INITIAL
        for m in sorted(monthly.keys()):
            m_ret = np.sum(monthly[m]) * 100 if monthly[m] else 0
            cumulative *= (1 + m_ret/100)
            bar = '█' * max(0, min(15, int(abs(m_ret)*2))) if m_ret > 0 else '░' * max(0, min(15, int(abs(m_ret)*2)))
            sign = '+' if m_ret > 0 else ''
            print(f"  {m:<8} {sign}{m_ret:>+6.1f}% ¥{cumulative:>10,.0f} {bar}")

    # Final verdict
    print(f"\n{'='*60}")
    print(f"  结论")
    print(f"{'='*60}")
    profit = final_val - INITIAL
    print(f"  初始 ¥1,000,000 → 最终 ¥{final_val:,.0f} (盈利 ¥{profit:+,.0f})")
    print(f"  总收益率 {total_ret:+.1f}%")
    if bench_ret:
        print(f"  同期沪深300: {bench_ret:+.1f}%, 超额: {total_ret-bench_ret:+.1f}%")
    if total_ret > 0:
        print(f"  ★ 策略盈利 — 但注意: 这是TOP5精选, 不代表全样本")


if __name__ == '__main__':
    main()
