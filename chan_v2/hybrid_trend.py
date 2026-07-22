"""
v2笔 + 趋势效率比 混合策略
===========================
效率比 ER = |ΔP(N)| / Σ|ΔP(i)|  (Kaufman Efficiency Ratio)
  ER>0.4: 趋势市 → 顺势交易 (只做趋势方向的笔信号)
  ER<0.2: 震荡市 → 反转交易 (当前策略,全部信号)
  0.2-0.4: 混合市 → 减仓50%

vs 纯反转策略 (baseline).
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

# Stock list (same as before)
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


def fetch_stock(symbol):
    prefix = 'sh' if symbol.startswith(('6','9')) else 'sz'
    for _ in range(2):
        try:
            import akshare as ak
            df = ak.stock_zh_a_daily(symbol=f'{prefix}{symbol}', adjust='qfq')
            if df is not None and len(df) > 300:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= '20180101') & (df['date'] <= '20260701')]
                if len(df) > 200:
                    return df.sort_values('date').reset_index(drop=True)
        except: time.sleep(1)
    return None


def efficiency_ratio(prices, window=60):
    """Kaufman Efficiency Ratio: directness of price movement"""
    if len(prices) < window: return np.full(len(prices), 0.5)
    er = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        segment = prices[i-window:i+1]
        net_change = abs(segment[-1] - segment[0])
        total_path = np.sum(np.abs(np.diff(segment)))
        er[i] = net_change / total_path if total_path > 0 else 0
    # Forward fill
    er[:window] = er[window] if window < len(er) else 0.5
    return er


def regime(er_val):
    """放宽趋势判断阈值 (A股波动大, ER天然偏低)"""
    if np.isnan(er_val): return 'semi'
    if er_val > 0.30: return 'trend'
    if er_val < 0.12: return 'chop'
    return 'semi'


def backtest_hybrid(df, name):
    """
    v2笔 + 趋势效率 仓位调节策略 (v2)
    核心改进: 不滤信号, 调仓位。
      trend(ER>0.30): 顺势全仓 (趋势中反转信号可靠)
      chop(ER<0.12):  反转全仓 (震荡中反转信号可靠)
      semi(0.12-0.30): 半仓 (方向不明确, 减仓避险)
    """
    try:
        a = ChanAnalyzer('D'); a.load_klines(df); a.run()
    except: return None

    valid = a.get_valid_strokes()
    if len(valid) < 4: return None

    closes = np.array([k.close for k in a.merged_klines])
    er = efficiency_ratio(closes, window=40)  # Shorter window for A-share sensitivity

    signals = []
    for st in valid:
        ei = st.end_fractal.index + 2
        if ei >= len(closes) or ei >= len(er): continue

        r = regime(er[ei])
        sig = 'BUY' if st.direction == Direction.DOWN else 'SELL'

        # ALL signals pass, position size varies by regime clarity
        if r == 'trend':
            # 趋势市: v2笔反转信号在趋势中更可靠 (回调有支撑)
            signals.append({'idx': ei, 'sig': sig, 'weight': 1.0, 'regime': 'trend'})
        elif r == 'chop':
            # 震荡市: 反转就是全部, 全仓
            signals.append({'idx': ei, 'sig': sig, 'weight': 1.0, 'regime': 'chop'})
        else:  # semi
            # 半趋势: 方向不够明确, 减半仓
            signals.append({'idx': ei, 'sig': sig, 'weight': 0.5, 'regime': 'semi'})

    signals.sort(key=lambda x: x['idx'])
    if len(signals) < 4: return None

    # Simulate
    pos = 0; cash = 1.0; shares = 0.0; equity = np.ones(len(closes))
    trades = []; entry_price = 0; si = 0
    regime_counts = {'trend': 0, 'chop': 0, 'semi': 0}

    for i in range(len(closes)):
        price = closes[i]
        while si < len(signals) and signals[si]['idx'] <= i:
            s = signals[si]
            regime_counts[s['regime']] += 1
            if s['sig'] == 'BUY' and pos == 0:
                alloc = s['weight']
                shares = (cash * alloc) / price; cash -= cash * alloc; pos = 1
                entry_price = price
            elif s['sig'] == 'SELL' and pos == 1:
                cash += shares * price; shares = 0.0; pos = 0
                trades.append({'ret': (price-entry_price)/entry_price, 'win': price > entry_price,
                               'regime': s['regime']})
            si += 1
        equity[i] = cash + shares * price

    if pos == 1:
        trades.append({'ret': (closes[-1]-entry_price)/entry_price, 'win': closes[-1] > entry_price,
                       'regime': 'final'})

    bh = closes / closes[0]
    rets = np.diff(equity) / equity[:-1]; rets = rets[np.isfinite(rets)]
    if len(rets) < 10: return None

    years = len(closes) / 252
    cagr = ((equity[-1]/equity[0])**(1/years)-1)*100 if years>0 else 0
    excess = rets - 0.02/252
    sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess)>0 else 0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity-peak)/peak)*100)
    bh_cagr = ((bh[-1]/bh[0])**(1/years)-1)*100 if years>0 else 0

    if trades:
        tdf = pd.DataFrame(trades)
        wr = tdf['win'].mean()*100; avg_r = tdf['ret'].mean()*100
        avg_w = tdf[tdf['win']]['ret'].mean()*100 if tdf['win'].any() else 0
        avg_l = tdf[~tdf['win']]['ret'].mean()*100 if (~tdf['win']).any() else 0
        pf = abs(avg_w/avg_l) if avg_l != 0 else 999
    else:
        wr = avg_r = avg_w = avg_l = pf = 0

    # Regime-specific performance
    regime_perf = {}
    for r in ['trend', 'semi', 'chop']:
        rt = [t for t in trades if t.get('regime') == r]
        if rt:
            regime_perf[r] = {
                'n': len(rt), 'win_rate': sum(1 for t in rt if t['win'])/len(rt)*100,
                'avg_ret': np.mean([t['ret'] for t in rt])*100,
            }

    total_signals = len(signals)
    sig_by_regime = {r: sum(1 for s in signals if s['regime'] == r) for r in ['trend','chop','mixed']}

    return {
        'name': name,
        'cagr': round(cagr,2), 'sharpe': round(sharpe,2), 'max_dd': round(max_dd,1),
        'bh_cagr': round(bh_cagr,2), 'alpha': round(cagr-bh_cagr,2),
        'win_rate': round(wr,1), 'avg_ret': round(avg_r,2),
        'profit_factor': round(pf,2) if pf<999 else 999, 'n_trades': len(trades),
        'n_signals': total_signals, 'sig_by_regime': sig_by_regime,
        'regime_perf': regime_perf,
    }


def main():
    print("=" * 85)
    print("  v2笔 + 趋势效率比 混合策略 vs 纯反转策略")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 85)

    print("\n[1] 逐股分析 (v2+trend filter)...")
    hybrid_results = []
    for i, (code, name) in enumerate(STOCKS):
        if (i+1) % 20 == 0: print(f"  ... {i+1}/{len(STOCKS)}")
        df = fetch_stock(code)
        if df is None: continue
        r = backtest_hybrid(df, name)
        if r: hybrid_results.append(r)

    print(f"  完成: {len(hybrid_results)}只")

    dfh = pd.DataFrame(hybrid_results)
    # Filter outliers
    dfh = dfh[(dfh['cagr']>-50)&(dfh['cagr']<80)&(dfh['sharpe']>-5)&(dfh['sharpe']<10)&(dfh['n_trades']>=3)]

    # Compare with pure reversal (from previous run)
    prev_data = {
        'alpha_med': -5.9, 'alpha_mean': -5.7, 'sharpe_med': 0.21,
        'beats_bh': 38, 'vol_penalty_mean': -16.1,
    }

    print(f"\n{'='*85}")
    print(f"  混合策略 vs 纯反转策略 (N={len(dfh)})")
    print(f"{'='*85}")

    print(f"\n  整体对比:")
    print(f"  {'指标':<20} {'纯反转(旧)':>14} {'+趋势过滤(新)':>16} {'改善':>12}")
    print(f"  {'-'*65}")
    print(f"  {'中位α':<20} {prev_data['alpha_med']:>+13.1f}% {dfh['alpha'].median():>+15.1f}% "
          f"{(dfh['alpha'].median()-prev_data['alpha_med']):>+11.1f}pp")
    print(f"  {'中位夏普':<20} {prev_data['sharpe_med']:>14.2f} {dfh['sharpe'].median():>16.2f} "
          f"{(dfh['sharpe'].median()-prev_data['sharpe_med']):>+11.2f}")
    print(f"  {'α>0占比':<20} {prev_data['beats_bh']:>13.0f}% {(dfh['alpha']>0).mean()*100:>15.0f}% "
          f"{((dfh['alpha']>0).mean()*100-prev_data['beats_bh']):>+11.0f}pp")

    # Regime distribution
    all_regimes = {'trend': 0, 'chop': 0, 'semi': 0}
    for _, r in dfh.iterrows():
        for k, v in r['sig_by_regime'].items():
            if k in all_regimes: all_regimes[k] += v
    total_sigs = sum(all_regimes.values())
    print(f"\n  信号分期分布 (ER窗口=40):")
    for k, v in all_regimes.items():
        print(f"  {k}: {v} ({v/total_sigs*100:.0f}%)")

    # Regime-specific win rates
    all_rp = {'trend': {'n':0,'wins':0,'rets':[]}, 'chop': {'n':0,'wins':0,'rets':[]},
               'semi': {'n':0,'wins':0,'rets':[]}}
    for _, r in dfh.iterrows():
        for regime, perf in r['regime_perf'].items():
            all_rp[regime]['n'] += perf['n']
            all_rp[regime]['wins'] += int(perf['n'] * perf['win_rate'] / 100)
            all_rp[regime]['rets'].append(perf['avg_ret'])

    print(f"\n  分市场状态交易表现:")
    print(f"  {'状态':<10} {'交易数':>8} {'胜率':>8} {'均收益':>8}")
    for r in ['trend', 'semi', 'chop']:
        d = all_rp[r]
        if d['n'] > 0:
            wr = d['wins']/d['n']*100
            ar = np.mean(d['rets'])
            print(f"  {r:<10} {d['n']:>8} {wr:>7.1f}% {ar:>+7.2f}%")

    # Top stocks by improvement
    print(f"\n  混合策略改善最大 (vs 纯反转, 按α):")
    top = dfh.nlargest(8, 'alpha')
    for _, r in top.iterrows():
        print(f"  {r['name']:<8} α={r['alpha']:>+6.1f}%  CAGR={r['cagr']:>6.1f}%  "
              f"夏普={r['sharpe']:>.2f}  胜率={r['win_rate']:>.0f}%  交易={r['n_trades']:.0f}")

    # Key takeaway
    print(f"\n{'='*60}")
    print(f"  结论")
    print(f"{'='*60}")
    medi_alpha = dfh['alpha'].median()
    beats_bh = (dfh['alpha'] > 0).mean()*100
    print(f"  混合策略中位α: {medi_alpha:+.1f}% (纯反转: {prev_data['alpha_med']:+.1f}%)")
    print(f"  α>0占比: {beats_bh:.0f}% (纯反转: {prev_data['beats_bh']}%)")
    print(f"  趋势期(ER>0.30)全仓: {all_regimes['trend']/total_sigs*100:.0f}% 信号")
    print(f"  半趋势(ER 0.12-0.30)半仓: {all_regimes['semi']/total_sigs*100:.0f}% 信号")
    print(f"  震荡期(ER<0.12)全仓: {all_regimes['chop']/total_sigs*100:.0f}% 信号")


if __name__ == '__main__':
    main()
