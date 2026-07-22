"""比较动量/反转/自适应等多种策略优化方向"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import efinance as ef
import warnings; warnings.filterwarnings('ignore')

codes = ['510880','159915','513100','518880','511010']
safe = '511010'
N_S, N_L, MOM, VOL = 25, 50, 20, 60

# ---- 数据加载 ----
dfs = {}
for code in codes:
    df = ef.fund.get_quote_history(code)
    df = df.rename(columns={'日期':'date','累计净值':'cum_nav'})
    df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce').ffill()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    dfs[code] = df['cum_nav']
data = pd.DataFrame(dfs).ffill().bfill().loc['20150101':'20250828']

# ---- 因子 ----
def trend_score(s, N):
    result = pd.Series(np.nan, index=s.index)
    vals = s.values
    for i in range(N, len(s)):
        y = vals[i-N+1:i+1] / vals[i-N+1]
        if np.isnan(y).any(): continue
        x = np.arange(1,N+1,dtype=float)
        slope,intercept = np.polyfit(x,y,1)
        y_pred = slope*x+intercept
        ss_res=np.sum((y-y_pred)**2); ss_tot=np.sum((y-np.mean(y))**2)
        r2 = 1-ss_res/ss_tot if ss_tot>0 else 0
        result.iloc[i] = slope*r2
    return result

print('[因子计算]...')
scores = pd.DataFrame(index=data.index)
for code in codes:
    p = data[code]
    scores[f'ts_{code}'] = trend_score(p,N_S)
    scores[f'tl_{code}'] = trend_score(p,N_L)
    scores[f'mom_{code}'] = p/p.shift(MOM)-1
    scores[f'vol_{code}'] = -p.pct_change().rolling(VOL).std()*np.sqrt(252)

for pf in ['ts_','tl_','mom_','vol_']:
    cols = [pf+c for c in codes]
    mu= scores[cols].mean(1); sd= scores[cols].std(1).replace(0,1)
    for col in cols: scores[col+'_z'] = (scores[col]-mu)/sd

for code in codes:
    scores[f'score_{code}'] = (0.30*scores[f'ts_{code}_z']+0.25*scores[f'tl_{code}_z']+
                               0.25*scores[f'mom_{code}_z']+0.20*scores[f'vol_{code}_z'])

scores = scores.dropna()
dret = data[codes].pct_change()
idx = scores.index.intersection(dret.index)
scores,dret = scores.loc[idx],dret.loc[idx]
fridays = set(idx[idx.dayofweek==4])
print(f'有效数据: {len(idx)} 天, 周五: {len(fridays)} 个')

# ---- 回测引擎 ----
def run_strategy(name, strategy_fn):
    h, nav, ntrades = {}, 1.0, 0
    nav_hist = []
    for i,date in enumerate(idx):
        if date in fridays and i>0:
            row = scores.loc[date]
            sc = sorted([(c,row[f'score_{c}']) for c in codes], key=lambda x:-x[1])
            all_neg = all(row[f'ts_{c}']<0 for c in codes)
            new_h = strategy_fn(sc, row, h, all_neg)
            if set(h.keys()) != set(new_h.keys()):
                ntrades += 1
                turnover = sum(abs(h.get(c,0)-new_h.get(c,0)) for c in set(list(h)+list(new_h)))
                nav *= (1-turnover*0.001)
            h = new_h
        if h:
            day_r = sum(h.get(c,0)*dret[c].iloc[i] for c in h if not np.isnan(dret[c].iloc[i]))
            nav *= (1+day_r)
        nav_hist.append(nav)
    yrs = len(nav_hist)/252
    cagr = nav**(1/yrs)-1 if nav>0 else -1
    nh = pd.Series(nav_hist)
    rets = nh.pct_change().dropna()
    sr = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    dd = float((nh/nh.expanding().max()-1).min())
    calmar = cagr/abs(dd) if dd!=0 else 0
    ndx_cagr = (dret['513100'].iloc[:len(nh)].add(1).prod())**(1/yrs)-1
    return {'name':name, 'cagr':cagr, 'sharpe':sr, 'maxdd':dd, 'calmar':calmar,
            'trades':ntrades, 'alpha':cagr-ndx_cagr, 'nav':nh}

# ---- 策略定义 ----
def top1(sc, row, h, all_neg):
    if all_neg: return {safe:1.0}
    return {sc[0][0]:1.0} if sc[0][1]>0 else {safe:1.0}

def top2_70(sc, row, h, all_neg):
    if all_neg: return {safe:1.0}
    valid = [c for c,s in sc[:2] if s>0]
    if not valid: return {safe:1.0}
    if len(valid)==1: return {valid[0]:1.0}
    return {valid[0]:0.7, valid[1]:0.3}

def bottom1(sc, row, h, all_neg):
    """反转: 买最后一名"""
    if all_neg: return {safe:1.0}
    return {sc[-1][0]:1.0}

def bottom2(sc, row, h, all_neg):
    """反转: 买倒数前2"""
    if all_neg: return {safe:1.0}
    return {sc[-1][0]:0.7, sc[-2][0]:0.3}

def inverse_weighted(sc, row, h, all_neg):
    """反转加权: 分越低仓位越重"""
    if all_neg: return {safe:1.0}
    raw = {c:max(-s, 0.001) for c,s in sc}
    total = sum(raw.values())
    return {c:v/total for c,v in raw.items()}

def score_weighted(sc, row, h, all_neg):
    """动量加权: 全ETF按分分配"""
    if all_neg: return {safe:1.0}
    raw = {c:max(s, 0.001) for c,s in sc}
    total = sum(raw.values())
    return {c:v/total for c,v in raw.items()}

def adaptive(sc, row, h, all_neg):
    """自适应: 分化大→动量; 接近→反转"""
    if all_neg: return {safe:1.0}
    sv = [s for _,s in sc]
    dispersion = np.std(sv) / (abs(np.mean(sv)) + 0.001)
    if dispersion > 1.5:
        valid = [c for c,s in sc[:2] if s>0]
        if not valid: return {safe:1.0}
        if len(valid)==1: return {valid[0]:1.0}
        return {valid[0]:0.7, valid[1]:0.3}
    else:
        return {sc[-1][0]:1.0}

def trend_filter(sc, row, h, all_neg):
    """双趋势确认: 只买短+长趋势都>0的ETF"""
    if all_neg: return {safe:1.0}
    for c,s in sc:
        if s>0 and row[f'ts_{c}']>0 and row[f'tl_{c}']>0:
            return {c:1.0}
    return {safe:1.0}

def vol_parity_top2(sc, row, h, all_neg):
    """波动率平价Top-2"""
    if all_neg: return {safe:1.0}
    valid = [(c,s) for c,s in sc[:2] if s>0]
    if not valid: return {safe:1.0}
    if len(valid)==1: return {valid[0][0]:1.0}
    # 近期波动率倒数加权
    recent = dret.iloc[max(0,len(dret)-60):]
    inv_vol = {}
    for c,_ in valid:
        v = recent[c].std()
        inv_vol[c] = 1.0 / max(v, 0.001)
    total = sum(inv_vol.values())
    return {c: v/total for c,v in inv_vol.items()}

def kelly_top2(sc, row, h, all_neg):
    """Kelly启发式: 得分越高仓位越集中"""
    if all_neg: return {safe:1.0}
    valid = [(c,s) for c,s in sc[:2] if s>0]
    if not valid: return {safe:1.0}
    if len(valid)==1: return {valid[0][0]:1.0}
    # 得分差距越大 → Top1仓位越高
    gap = valid[0][1] - valid[1][1]
    w1 = min(0.9, 0.5 + gap * 2)  # gap大→w1接近0.9
    w1 = max(0.5, w1)
    return {valid[0][0]:w1, valid[1][0]:1-w1}

def ema_trend_filter(sc, row, h, all_neg):
    """EMA趋势过滤: 价格>50日均线才买"""
    if all_neg: return {safe:1.0}
    # 用实际价格/50日均线判断
    for c,s in sc:
        if s > 0:
            price_now = data[c].iloc[-1]
            ma50 = data[c].iloc[-50:].mean()
            if price_now > ma50:
                return {c:1.0}
    return {safe:1.0}

# ---- 全量测试 ----
strategies = [
    ("Top-1 动量",            top1),
    ("Top-2 70/30",           top2_70),
    ("Bottom-1 反转",         bottom1),
    ("Bottom-2 反转",         bottom2),
    ("反转加权(全仓)",         inverse_weighted),
    ("动量加权(全仓)",         score_weighted),
    ("自适应(动量/反转)",       adaptive),
    ("趋势过滤(双确认)",        trend_filter),
    ("波动率平价Top-2",        vol_parity_top2),
    ("Kelly启发式Top-2",       kelly_top2),
    ("EMA趋势过滤",            ema_trend_filter),
]

print(f"\n{'策略':<22} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>9} {'Calmar':>7} {'年交易':>6} {'超额':>8} {'评价':>10}")
print('-'*85)

results = []
for name, fn in strategies:
    r = run_strategy(name, fn)
    results.append(r)
    # 综合评价
    grade = ""
    if r['cagr'] > 0.20 and r['sharpe'] > 1.1: grade = "优"
    elif r['cagr'] > 0.15 and r['sharpe'] > 0.9: grade = "良"
    elif r['cagr'] > 0.10: grade = "中"
    else: grade = "差"
    print(f'{name:<22} {r["cagr"]:>7.2%} {r["sharpe"]:>7.2f} {r["maxdd"]:>8.2%} {r["calmar"]:>7.2f} {r["trades"]/10.1:>6.0f} {r["alpha"]:>7.2%} {grade:>10}')

# ---- 年度相关性 ----
print(f"\n{'='*60}")
print('动量 vs 反转: 年度收益相关性')
print(f'{"="*60}')

annual_data = {}
for label, fn in [('动量Top-1', top1), ('反转Bottom-1', bottom1), ('自适应', adaptive)]:
    h,nav = {}, 1.0
    annual = {}
    for i,date in enumerate(idx):
        if date in fridays and i>0:
            row = scores.loc[date]
            sc = sorted([(c,row[f'score_{c}']) for c in codes], key=lambda x:-x[1])
            all_neg = all(row[f'ts_{c}']<0 for c in codes)
            new_h = fn(sc,row,h,all_neg)
            if set(h.keys())!=set(new_h.keys()):
                turnover = sum(abs(h.get(c,0)-new_h.get(c,0)) for c in set(list(h)+list(new_h)))
                nav *= (1-turnover*0.001)
            h = new_h
        if h:
            day_r = sum(h.get(c,0)*dret[c].iloc[i] for c in h if not np.isnan(dret[c].iloc[i]))
            nav *= (1+day_r)
        if date.month==12 and date.day>=28:
            annual[date.year] = nav
    annual_data[label] = annual

common = sorted(set(annual_data['动量Top-1']) & set(annual_data['反转Bottom-1']))
print(f"{'年份':<8} {'动量Top-1':>10} {'反转Bottom-1':>12} {'自适应':>10} {'动量-反转差':>12}")
for yr in common:
    mr = annual_data['动量Top-1'][yr] / annual_data['动量Top-1'].get(yr-1, 1) - 1 if yr-1 in annual_data['动量Top-1'] else annual_data['动量Top-1'][yr] - 1
    br = annual_data['反转Bottom-1'][yr] / annual_data['反转Bottom-1'].get(yr-1, 1) - 1 if yr-1 in annual_data['反转Bottom-1'] else annual_data['反转Bottom-1'][yr] - 1
    ar = annual_data['自适应'][yr] / annual_data['自适应'].get(yr-1, 1) - 1 if yr-1 in annual_data['自适应'] else annual_data['自适应'][yr] - 1
    diff = mr - br
    print(f'{yr:<8} {mr:>+9.2%} {br:>+11.2%} {ar:>+9.2%} {diff:>+11.2%}')

# 相关性统计
mom_rets = []
rev_rets = []
for yr in common:
    mr = annual_data['动量Top-1'][yr]/annual_data['动量Top-1'].get(yr-1,1)-1 if yr-1 in annual_data['动量Top-1'] else annual_data['动量Top-1'][yr]-1
    br = annual_data['反转Bottom-1'][yr]/annual_data['反转Bottom-1'].get(yr-1,1)-1 if yr-1 in annual_data['反转Bottom-1'] else annual_data['反转Bottom-1'][yr]-1
    mom_rets.append(mr)
    rev_rets.append(br)

corr = np.corrcoef(mom_rets, rev_rets)[0,1]
print(f'\n动量vs反转年度收益相关系数: {corr:.3f}')
print(f'→ {"负相关! 可以做多空组合" if corr < -0.2 else "正相关" if corr > 0.2 else "弱相关"}')
