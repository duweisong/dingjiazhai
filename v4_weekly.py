"""丁家山 V4 周度推荐 Top 20"""
import os, struct, pickle
import numpy as np
import pandas as pd

TDX_PATH = r'C:\zd_pazq_hy\vipdoc'
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
MIN_AVG_AMOUNT = 50_000_000
MIN_SCORE = 0.3

FACTOR_WEIGHTS = {
    'ret_5d': 0.10, 'ret_10d': 0.10, 'ret_20d': 0.10,
    'vol_ratio': 0.20, 'price_vs_ma20': 0.20, 'ma_alignment': 0.12,
    'up_days_10': 0.05, 'turnover': 0.05, 'atr_pct': -0.08,
}

POS_STRONG = 5; POS_NEUTRAL = 4; POS_BEAR = 1
SINGLE_PCT = 0.12; SINGLE_BEAR = 0.08; CASH = 0.35

def parse_day_file(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'rb') as f: raw = f.read()
    n = len(raw)//32
    if n==0: return None
    dates,opens,highs,lows,closes,amounts,volumes=[],[],[],[],[],[],[]
    for i in range(n):
        off=i*32
        date,op,hi,lo,cl,amt,vol,_=struct.unpack('IIIIIfII',raw[off:off+32])
        dates.append(str(date)); opens.append(op/100); highs.append(hi/100)
        lows.append(lo/100); closes.append(cl/100)
        amounts.append(amt); volumes.append(vol)
    return pd.DataFrame({'date':pd.to_datetime(dates,format='%Y%m%d'),
        'open':opens,'high':highs,'low':lows,'close':closes,
        'amount':amounts,'volume':volumes}).sort_values('date').reset_index(drop=True)

def get_day_path(code):
    pfx='sh' if (code.startswith('6') or code.startswith('68')) else 'sz'
    return os.path.join(TDX_PATH,pfx,'lday',f'{pfx}{code}.day')

def compute_factors(df, date_str):
    d=df[df['date']<=date_str]
    if len(d)<60: return None
    close=d['close'].values; high=d['high'].values; low=d['low'].values
    volume=d['volume'].values; n=len(d)
    ret_5d=(close[-1]/close[-6]-1)*100 if n>=6 else 0
    ret_10d=(close[-1]/close[-11]-1)*100 if n>=11 else 0
    ret_20d=(close[-1]/close[-21]-1)*100 if n>=21 else 0
    vol_20=np.mean(volume[-21:-1]) if n>=21 and np.mean(volume[-21:-1])>0 else 1
    vol_ratio=volume[-1]/vol_20 if vol_20>0 else 1
    ma_5=np.mean(close[-6:-1]); ma_10=np.mean(close[-11:-1]) if n>=11 else ma_5
    ma_20=np.mean(close[-21:-1]) if n>=21 else ma_10
    ma_60=np.mean(close[-61:-1]) if n>=61 else ma_20
    price_vs_ma20=(close[-1]/ma_20-1)*100 if ma_20>0 else 0
    ma_alignment=1.0 if (ma_5>ma_10>ma_20>ma_60) else (0.5 if ma_5>ma_20 else 0.0)
    tr_list=[]
    for i in range(max(1,n-20),n):
        tr=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
        tr_list.append(tr)
    atr_20=np.mean(tr_list) if tr_list else close[-1]*0.02
    atr_pct=(atr_20/close[-1])*100 if close[-1]>0 else 2.0
    up_days_10=sum(1 for i in range(max(1,n-10),n) if close[i]>close[i-1])
    avg_amount=np.mean(d['amount'].values[-21:-1]) if n>=21 else 0
    return {'ret_5d':ret_5d,'ret_10d':ret_10d,'ret_20d':ret_20d,
        'vol_ratio':vol_ratio,'price_vs_ma20':price_vs_ma20,
        'ma_alignment':ma_alignment,'atr_pct':atr_pct,
        'up_days_10':up_days_10,'close':close[-1],'atr_20':atr_20,
        'ma_20':ma_20,'avg_amount':avg_amount}

# Load
cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
stocks = pickle.load(open(cache_file, 'rb'))['stocks']
print(f'[LOAD] {len(stocks)} stocks...')

all_data = {}
for i, s in enumerate(stocks):
    df = parse_day_file(get_day_path(s['code']))
    if df is not None:
        df = df[df['date'] >= START_DATE].copy()
        if len(df) >= 60: all_data[s['code']] = {'name': s['name'], 'df': df}
    if (i+1) % 100 == 0: print(f'  {i+1}/{len(stocks)}...')
print(f'[DATA] {len(all_data)} loaded')

# Score
date_str = '2026-07-03'
fl = []
for code, info in all_data.items():
    f = compute_factors(info['df'], date_str)
    if f is None: continue
    f['code'] = code; f['name'] = info['name']; fl.append(f)

df = pd.DataFrame(fl)
df = df[df['avg_amount'] >= MIN_AVG_AMOUNT].copy()
sc = {}
for factor, weight in FACTOR_WEIGHTS.items():
    if factor not in df.columns: continue
    col = df[factor].copy(); m = col.mean(); s = col.std()
    z = pd.Series(0.0, index=col.index) if (s==0 or pd.isna(s)) else ((col-m)/s).clip(-3,3)
    sc[factor] = z * weight
df['score'] = sum(sc.values())
df = df.sort_values('score', ascending=False).reset_index(drop=True)
df['rank'] = range(1, len(df)+1)

# Market regime
bench_path = os.path.join(TDX_PATH, 'sh', 'lday', 'sh000300.day')
bench_df = parse_day_file(bench_path)
is_bear = False; is_strong = False
idx_close = 0; ma_val = 0
if bench_df is not None:
    bench_df = bench_df[bench_df['date'] >= START_DATE]
    bench_ma = bench_df.set_index('date')['close'].rolling(120).mean()
    dt = pd.Timestamp(date_str)
    if dt in bench_ma.index:
        idx_close = bench_df[bench_df['date'] == dt]['close'].iloc[0]
        ma_val = bench_ma.loc[dt]
        is_bear = idx_close < ma_val
        is_strong = idx_close > ma_val * 1.05

max_pos = POS_BEAR if is_bear else (POS_STRONG if is_strong else POS_NEUTRAL)
regime_name = 'BEAR(熊市)' if is_bear else ('STRONG(强牛)' if is_strong else 'NEUTRAL(中性)')
best_score = df['score'].iloc[0]

# ====== OUTPUT ======
print()
print('='*105)
print(f'  丁家山 V4 周度推荐 | {date_str} | {regime_name} | CSI300={idx_close:.0f} MA120={ma_val:.0f}')
print(f'  仓位:{max_pos}只 | 单只{12 if not is_bear else 8}% | 现金{CASH*100:.0f}% | 信号:{"达标" if best_score>=MIN_SCORE else "偏弱"}')
print('='*105)

# Top 20
print()
print('  --- 全市场 Top 20 ---')
print(f'  {"排名":<5} {"代码":<8} {"名称":<12} {"评分":>7} {"5日":>7} {"20日":>7} {"量比":>6} {"vsMA20":>8} {"均线":<6} {"波%":>5} {"建议":<14}')
print(f'  {"-"*5} {"-"*8} {"-"*12} {"-"*7} {"-"*7} {"-"*7} {"-"*6} {"-"*8} {"-"*6} {"-"*5} {"-"*14}')

for _, r in df.head(20).iterrows():
    rank = int(r['rank']); close_p = r['close']; ma20_p = r['ma_20']
    atr_p = r['atr_pct']
    align = '多头' if r['ma_alignment']>=1 else ('偏多' if r['ma_alignment']>=0.5 else '空头')

    # Risk flags
    flags = []
    if close_p < ma20_p: flags.append('破MA20')
    if rank > 80: flags.append('排>80')
    if atr_p > 5: flags.append('高波')

    if rank <= max_pos and best_score >= MIN_SCORE:
        if flags: action = f'买入⚠{",".join(flags)}'
        else: action = '买入 ✅'
    else:
        action = '候选' if best_score >= MIN_SCORE else '观望'

    print(f'  #{rank:<4} {r["code"]:<8} {r["name"]:<12} {r["score"]:>+7.3f} {r["ret_5d"]:>+6.2f}% {r["ret_20d"]:>+6.2f}% {r["vol_ratio"]:>6.2f} {r["price_vs_ma20"]:>+7.2f}% {align:<6} {atr_p:>4.1f}% {action:<14}')

# Buy plan
print()
print('  --- 买入计划 ---')
if best_score >= MIN_SCORE:
    if is_bear:
        single_pct_use = SINGLE_BEAR
        print(f'  熊市模式: {max_pos}只 x {single_pct_use*100:.0f}%仓位 + {CASH*100:.0f}%现金')
    else:
        single_pct_use = SINGLE_PCT
        print(f'  正常模式: {max_pos}只 x {single_pct_use*100:.0f}%仓位 + {CASH*100:.0f}%现金')

    print(f'  {"代码":<8} {"名称":<12} {"买入价":>8} {"止损价":>8} {"止盈":>8} {"仓位%":>7} {"金额(万)":>9}')
    print(f'  {"-"*8} {"-"*12} {"-"*8} {"-"*8} {"-"*8} {"-"*7} {"-"*9}')
    for _, r in df.head(max_pos).iterrows():
        close_p = r['close']; atr_p = r['atr_pct']
        atm = 2.0 if atr_p > 5 else (1.6 if atr_p > 2.5 else 1.2)
        stop_p = close_p - atm * r['atr_20']
        target_p = close_p * 1.12
        amt = 1_000_000 * single_pct_use / 10000
        print(f'  {r["code"]:<8} {r["name"]:<12} {close_p:>8.2f} {stop_p:>8.2f} {target_p:>8.2f} {single_pct_use*100:>6.1f}% {amt:>8.0f}')
else:
    print(f'  信号不足 (最佳{best_score:.3f}<{MIN_SCORE}) → 本周观望,保持现有持仓')

# Holdings risk check for top 20
print()
print('  --- 持仓风险监控 (若已持有Top20中任意) ---')
for _, r in df.head(20).iterrows():
    close_p = r['close']; ma20_p = r['ma_20']
    rank = int(r['rank']); atr_p = r['atr_pct']
    hard_stop = -0.03 if is_bear else -0.05

    alerts = []
    if close_p < ma20_p: alerts.append(f'跌破MA20({ma20_p:.2f})')
    if rank > 80: alerts.append(f'排名淘汰(#{rank}>80)')
    if atr_p > 5: alerts.append(f'高波动{atr_p:.1f}%')

    if alerts:
        print(f'  {r["code"]} {r["name"]:<12} 触发退出: {", ".join(alerts)}')
    else:
        atm = 2.0 if atr_p > 5 else (1.6 if atr_p > 2.5 else 1.2)
        stop_p = close_p - atm * r['atr_20']
        print(f'  {r["code"]} {r["name"]:<12} 正常 | 止损{stop_p:.2f} | 止盈{close_p*1.12:.2f} | 到期10天')

print()
print('='*105)
print('  免责: 量化策略信号,仅供参考,不构成投资建议。股市有风险,投资需谨慎。')
print('='*105)

