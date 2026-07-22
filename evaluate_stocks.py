"""丁家山 V3 个股评测"""
import os, struct, pickle
import numpy as np
import pandas as pd

TDX_PATH = r'C:\zd_pazq_hy\vipdoc'
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'

FACTOR_WEIGHTS = {
    'ret_5d': 0.12, 'ret_10d': 0.10, 'ret_20d': 0.10,
    'vol_ratio': 0.18, 'price_vs_ma20': 0.18, 'ma_alignment': 0.10,
    'up_days_10': 0.05, 'turnover': 0.05, 'atr_pct': -0.05,
    'ret_overbought': -0.07,
}

TARGETS = ['000301','601375','002926','601555','002483','000948','002036','002215','000002']

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
    d = df[df['date'] <= date_str]
    if len(d) < 60: return None
    close = d['close'].values; high = d['high'].values; low = d['low'].values
    volume = d['volume'].values; n = len(d)
    ret_5d = (close[-1]/close[-6]-1)*100 if n>=6 else 0
    ret_10d = (close[-1]/close[-11]-1)*100 if n>=11 else 0
    ret_20d = (close[-1]/close[-21]-1)*100 if n>=21 else 0
    vol_20 = np.mean(volume[-21:-1]) if n>=21 and np.mean(volume[-21:-1])>0 else 1
    vol_ratio = volume[-1]/vol_20 if vol_20>0 else 1
    ma_5 = np.mean(close[-6:-1]); ma_10 = np.mean(close[-11:-1]) if n>=11 else ma_5
    ma_20 = np.mean(close[-21:-1]) if n>=21 else ma_10
    ma_60 = np.mean(close[-61:-1]) if n>=61 else ma_20
    price_vs_ma20 = (close[-1]/ma_20-1)*100 if ma_20>0 else 0
    ma_alignment = 1.0 if (ma_5>ma_10>ma_20>ma_60) else (0.5 if ma_5>ma_20 else 0.0)
    tr_list = []
    for i in range(max(1,n-20),n):
        tr = max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
        tr_list.append(tr)
    atr_20 = np.mean(tr_list) if tr_list else close[-1]*0.02
    atr_pct = (atr_20/close[-1])*100 if close[-1]>0 else 2.0
    up_days_10 = sum(1 for i in range(max(1,n-10),n) if close[i]>close[i-1])
    ret_overbought = max(0,ret_5d-25) if ret_5d>25 else 0
    avg_amount = np.mean(d['amount'].values[-21:-1]) if n>=21 else 0
    return {'ret_5d':ret_5d,'ret_10d':ret_10d,'ret_20d':ret_20d,
        'vol_ratio':vol_ratio,'price_vs_ma20':price_vs_ma20,
        'ma_alignment':ma_alignment,'atr_pct':atr_pct,
        'turnover':1.0,'up_days_10':up_days_10,'ret_overbought':ret_overbought,
        'close':close[-1],'atr_20':atr_20,'ma_20':ma_20,'ma_60':ma_60,
        'avg_amount':avg_amount}

# Load CSI300 + targets
cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
stocks = pickle.load(open(cache_file, 'rb'))['stocks']

# Add target stocks if not in CSI300
known_names = {}
for s in stocks:
    known_names[s['code']] = s['name']
# Add any missing targets with placeholder names
extra_names = {
    '000301': '东方盛虹', '601375': '中原证券', '002926': '华西证券',
    '601555': '东吴证券', '002483': '润邦股份', '000948': '南天信息',
    '002036': '联创电子', '002215': '诺普信', '000002': '万科A',
}
for code, name in extra_names.items():
    if code not in known_names:
        stocks.append({'code': code, 'name': name})

print('[LOAD] Loading data...')
all_data = {}
start_dt = pd.Timestamp(START_DATE)
for i, s in enumerate(stocks):
    df = parse_day_file(get_day_path(s['code']))
    if df is not None:
        df = df[df['date'] >= start_dt].copy()
        if len(df) >= 60:
            all_data[s['code']] = {'name': s['name'], 'df': df}
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(stocks)}...')
print(f'[DATA] {len(all_data)} stocks loaded')

# Score on latest date
date_str = '2026-07-03'
factors_list = []
for code, info in all_data.items():
    f = compute_factors(info['df'], date_str)
    if f is None: continue
    f['code'] = code; f['name'] = info['name']
    factors_list.append(f)

df = pd.DataFrame(factors_list)
df = df[df['avg_amount'] >= 50_000_000].copy()

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
if bench_df is not None:
    bench_df = bench_df[bench_df['date'] >= START_DATE]
    bench_ma = bench_df.set_index('date')['close'].rolling(120).mean()
    dt = pd.Timestamp(date_str)
    if dt in bench_ma.index:
        idx_close = bench_df[bench_df['date'] == dt]['close'].iloc[0]
        ma_val = bench_ma.loc[dt]
        is_bear = idx_close < ma_val
        is_strong = idx_close > ma_val * 1.05

max_pos = 2 if is_bear else (5 if is_strong else 3)
regime_name = 'BEAR(熊市)' if is_bear else ('STRONG(强牛)' if is_strong else 'NEUTRAL(中性)')

# ====== OUTPUT ======
print()
print('='*90)
print('  丁家山 V3 个股评测报告')
print(f'  日期: {date_str} | 市场: {regime_name} | 最大仓位: {max_pos}只')
print('='*90)

# 1. Individual scores
print()
print('  【一】因子评分明细')
print(f'  {"代码":<8} {"名称":<10} {"评分":>8} {"全市场排名":>8} {"5日":>7} {"20日":>7} {"量比":>6} {"vsMA20":>8} {"均线":<6} {"波动%":>6} {"超买":>6}')
print(f'  {"-"*8} {"-"*10} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*6} {"-"*8} {"-"*6} {"-"*6} {"-"*6}')

for code in TARGETS:
    row = df[df['code'] == code]
    if len(row) > 0:
        r = row.iloc[0]
        align = '多头' if r['ma_alignment']>=1 else ('偏多' if r['ma_alignment']>=0.5 else '空头')
        ob = f'{r["ret_overbought"]:.1f}' if r['ret_overbought'] > 0 else '-'
        print(f'  {code:<8} {r["name"]:<10} {r["score"]:>8.3f} #{int(r["rank"]):>7} {r["ret_5d"]:>+6.2f}% {r["ret_20d"]:>+6.2f}% {r["vol_ratio"]:>6.2f} {r["price_vs_ma20"]:>+7.2f}% {align:<6} {r["atr_pct"]:>5.2f}% {ob:>6}')
    else:
        print(f'  {code:<8} {"--":<10} {"数据不足或流动性不达标":>40}')

# 2. Ranking in full market
print()
print('  【二】全市场排名位置')
print()
ranked_targets = []
for code in TARGETS:
    row = df[df['code'] == code]
    if len(row) > 0:
        ranked_targets.append((int(row.iloc[0]['rank']), code, row.iloc[0]['name'], row.iloc[0]['score']))
ranked_targets.sort()

total = len(df)
for rank, code, name, score in ranked_targets:
    pct = rank / total * 100
    bar = '█' * int(pct / 2) if pct < 100 else '█' * 50
    print(f'  #{rank:<5} {code} {name:<10} {score:+.3f} 前{pct:.1f}% {bar}')

# 3. Buy signal analysis
print()
print('  【三】买入信号判定')
best_score = df['score'].iloc[0] if len(df) > 0 else -99
print(f'  全市场最佳评分: {best_score:.3f} (买入阈值: 0.3)')
if best_score < 0.3:
    print(f'  >> 今日信号过弱,丁家山策略将跳过买入,保持空仓/现有持仓')
else:
    print(f'  >> 信号达标,最多可买入{max_pos}只')
    bought = []
    for _, row in df.iterrows():
        if len(bought) >= max_pos: break
        if row['code'] in TARGETS:
            bought.append((row['code'], row['name'], int(row['rank']), row['score']))
    if bought:
        print(f'  目标股中进入买入列表:')
        for c, n, r, s in bought:
            print(f'    --> {c} {n} 排名#{r} 评分{s:.3f}')
    else:
        print(f'  目标股均未进入Top{max_pos}')

# 4. Exit risk check
print()
print('  【四】退出风险检查 (假设今日买入)')
for code in TARGETS:
    row = df[df['code'] == code]
    if len(row) == 0: continue
    r = row.iloc[0]
    rank = int(r['rank'])
    close = r['close']; ma20 = r['ma_20']
    atr = r['atr_20']; atr_pct = r['atr_pct']

    risks = []
    if close < ma20:
        risks.append(f'收盘{close:.2f} < MA20({ma20:.2f}), 已触发均线退出')
    if rank > 100:
        risks.append(f'排名#{rank} > 100, 已触发排名淘汰')
    if atr_pct > 5:
        risks.append(f'波动率{atr_pct:.1f}%极高, 仓位将缩至60%')
    elif atr_pct > 3:
        risks.append(f'波动率{atr_pct:.1f}%偏高, 仓位将缩至80%')

    # Trailing stop estimate
    atm = 2.0 if atr_pct > 5 else (1.6 if atr_pct > 2.5 else 1.2)
    stop_price = close - atm * atr
    stop_pct = (stop_price / close - 1) * 100

    if risks:
        print(f'  {code} {r["name"]}: 风险!')
        for risk in risks:
            print(f'    - {risk}')
    else:
        print(f'  {code} {r["name"]}: 正常 | 止损位={stop_price:.2f}({stop_pct:+.1f}%) | 止盈=+15% | 到期=14天')

# 5. Score breakdown
print()
print('  【五】因子贡献分解 (Z-score * weight)')
print(f'  {"代码":<8} {"总评分":>7}', end='')
for f in FACTOR_WEIGHTS:
    print(f' {f[:8]:>8}', end='')
print()
for code in TARGETS:
    row = df[df['code'] == code]
    if len(row) == 0: continue
    r = row.iloc[0]
    print(f'  {code:<8} {r["score"]:>+7.3f}', end='')
    for factor, weight in FACTOR_WEIGHTS.items():
        if factor not in df.columns:
            print(f' {"--":>8}', end='')
            continue
        col = df[factor]
        m = col.mean(); s = col.std()
        if s == 0 or pd.isna(s):
            print(f' {"0.000":>8}', end='')
        else:
            z = (r[factor] - m) / s
            print(f' {z*weight:>+8.3f}', end='')
    print()

print()
print('='*90)
print('  免责: 本评测基于历史数据,不构成投资建议')
print('='*90)

