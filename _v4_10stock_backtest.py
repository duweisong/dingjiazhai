# DINGJIASHAN V4.7 — Full CSI300 Pool (301 stocks) + Crash Protections
# V4.7 CHANGES (factor analysis + July crash lessons):
#  1. Factor weights: boost volume(+40%), atr positive, add retrace/gap/dist_hh
#  2. Dynamic TP: fast movers(8% in 3d) lock early, slow movers wait for 15%
#  3. Tighter threshold: MIN_SCORE_BASE=0.35 (top 30% vs top 50%)
#  4. Crash protections from V4.6: Cooldown + MA20 + DownBrake
# V4.2 -> V4.3 CHANGES:
#  1. MA20 exit: only if below MA20 for 3 consecutive days
#  2. Dynamic SL floor: never looser than -5% (hard floor)
#  3. Time-stop: if held 5+ days AND losing, force exit
# Experimental features (disabled — all underperform V4.3 in 3.5yr backtest):
#  V4.4: MA20_BUY_FILTER / MA20_EXTEND_CAP / LOSS_COOLDOWN
#  V4.5: Regime V2 / Breakeven stop / Sector cap
import json, os, pickle, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
BJT = timezone(timedelta(hours=8))  # Beijing time
from collections import Counter

C = {}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.strategy_cache')
START = pd.Timestamp('2026-06-01')
END = pd.Timestamp('2026-07-10')
POS_STRONG = 10; POS_NEUTRAL = 8; POS_BEAR = 4
BEAR_MA = 120; SHORT_MA = 20  # V4.6: short-term trend MA
MIN_AMT = 30_000_000; MIN_SCORE_BASE = 0.35  # V4.7: tighter (top ~30% vs 50%)
COOL_OFF = 2
MIN_HOLD = 1
EMERGENCY_DROP = -0.08
BEAR_FILTER = False  # Disabled: hurts recovery, bear params are better
BEAR_FILTER_MA = 60  # Use 60-day MA for bear filter (faster than 120)
SL_FLOOR = -0.05  # hard floor: never looser than -5%
DOWN_DAY_BRAKE = -0.02  # V4.6: no buys if CSI300 dropped >2 percent yesterday
TIME_LOSER = 5     # exit if held this many days AND still losing
MA20_CONSEC = 3    # consecutive days below MA20 before exit
# V4.4 new filters
MA20_BUY_FILTER = False    # disabled for V4.3 comparison
MA20_EXTEND_CAP = 0        # disabled for V4.3 comparison
LOSS_COOLDOWN = 2          # V4.6: pause after 2 consecutive losses (any exit type)
COOLDOWN_DAYS = 3          # V4.6: pause 3 days
MA20_BUY_TOLERANCE = 1.0   # close must be above MA20 (no tolerance)
# V4.5 new features
BREAKEVEN_PROFIT = 0.12     # lock in breakeven after +12% profit
BREAKEVEN_ENABLED = False   # toggle on/off
SECTOR_CAP = 0             # disabled (V4.5 experiment)
# Sector mapping for concentration limits
SECTORS = {
    'tech_semi': {'688981','688012','688521','688082','688126','688506','603986','603501','002463','300408','300394','300308','300502','002415','002475'},
    'finance': {'601318','600036','600030','600016','601398','601328','601939','601288','601166','601601','000001','000776','601066','600999','300059','601211'},
    'consumer': {'600519','000858','000568','002304','600887','600809','600690','000333','000651','002594','000002','601888'},
    'industrial': {'600031','601012','600585','601668','600028','601088','601225','600048','600104','002050','603260','000630','000301','000938','002422'},
    'energy_material': {'601857','601899','600188','603993','000657','600160','600362','600549','001965','601872'},
    'healthcare': {'600276','300015','002714','002271','603259','688506','000100'},
    'telecom_media': {'600050','600522','002558','301165','603019','002352'},
    'transport_utility': {'600900','600011','600027','600023','601138'},
}
def get_sector(code):
    for sec, codes in SECTORS.items():
        if code in codes: return sec
    return 'other'

FW = {'ret_5d':0.05,'ret_10d':0.08,'ret_20d':0.05,'vol_ratio':0.20,
      'price_vs_ma20':0.15,'ma_alignment':0.10,'up_days_10':0.05,
      'turnover':0.05,'atr_pct':0.05,'amt_ratio':0.10,'retrace':-0.10,'dist_hh20':-0.07}
EX = {'atm_low':1.2,'atm_high':2.0,'max_hold':10,'tp_std':0.12,'tp_top':0.15,
      'rank_out_ratio':0.75,'rank_drop':50,'sl_hard':-0.08,'sl_bear':-0.10,
      'ma_p':20,'atr_sl_mult':2.5,'tp_bear':0.10,
      'tp_fast':0.08,'tp_fast_days':3}  # V4.7: fast movers lock early


# Stock names loaded from cache
_NAME_CACHE = {}
_name_file = os.path.join(CACHE_DIR, 'stock_names.json')
if os.path.exists(_name_file):
    try:
        with open(_name_file, 'r', encoding='utf-8') as _f:
            _NAME_CACHE = json.load(_f)
    except Exception: pass

def _get_name(code):
    return _NAME_CACHE.get(code, code)
def load():
    skip_codes = {'000300', 'sh.000300', 'csi300_stocks'}
    for f in os.listdir(CACHE_DIR):
        if not f.endswith('.pkl'): continue
        code = f.replace('.pkl','')
        if code in skip_codes: continue
        try:
            with open(os.path.join(CACHE_DIR,f),'rb') as _pf:
                d = pickle.load(_pf)
            df = d['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # Ensure required columns exist
            required = ['open','high','low','close','volume','amount']
            if not all(c in df.columns for c in required):
                continue
            C[code] = {'name':_get_name(code),'df':df}
        except: pass
    print('Loaded',len(C),'stocks')

def bench():
    for k in ['000300']:
        bp = os.path.join(CACHE_DIR,k+'.pkl')
        if os.path.exists(bp):
            with open(bp,'rb') as _pf:
                d = pickle.load(_pf)
            df = d['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
    return None

def factors(df,idx):
    if idx<60: return None
    d = df.iloc[:idx+1]
    c=d['close'].values;h=d['high'].values;l=d['low'].values
    v=d['volume'].values;n=len(d)
    r5=(c[-1]/c[-6]-1)*100 if n>=6 else 0
    r10=(c[-1]/c[-11]-1)*100 if n>=11 else 0
    r20=(c[-1]/c[-21]-1)*100 if n>=21 else 0
    v20=np.mean(v[-21:-1]) if n>=21 else 1
    vr=v[-1]/max(v20,1) if v20>0 else 1
    ma5=np.mean(c[-6:-1]);ma10=np.mean(c[-11:-1]) if n>=11 else ma5
    ma20=np.mean(c[-21:-1]) if n>=21 else ma10
    ma60=np.mean(c[-61:-1]) if n>=61 else ma20
    pvm=(c[-1]/ma20-1)*100 if ma20>0 else 0
    mal=1 if ma5>ma10>ma20>ma60 else (0.5 if ma5>ma20 else 0)
    trs=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(max(1,n-20),n)]
    a20=np.mean(trs) if trs else c[-1]*0.02
    ap=(a20/c[-1])*100 if c[-1]>0 else 2
    upd=sum(1 for i in range(max(1,n-10),n) if c[i]>c[i-1])
    amt=np.mean(d['amount'].values[-21:-1]) if n>=21 else 0
    # V4.7 new factors
    # Amount ratio (5d vs 20d avg)
    amt5v = np.mean(d['amount'].values[-6:-1]) if n>=6 else 0
    amt20v = np.mean(d['amount'].values[-21:-1]) if n>=21 else 0
    amt_r = amt5v/max(amt20v, 1) if amt20v > 0 else 1
    # Retrace (close vs high)
    retrace = (c[-1]/h[-1] - 1)*100 if h[-1] > 0 else 0
    # Distance from 20d high
    hh20 = max(c[-21:-1]) if n>=21 else c[-1]
    dist_hh = (c[-1]/hh20 - 1)*100 if hh20 > 0 else 0
    
    return {'ret_5d':r5,'ret_10d':r10,'ret_20d':r20,'vol_ratio':vr,
        'price_vs_ma20':pvm,'ma_alignment':mal,'atr_pct':ap,
        'up_days_10':upd,'close':c[-1],'atr_20':a20,'ma_20':ma20,'avg_amount':amt,
        'amt_ratio':amt_r,'retrace':retrace,'dist_hh20':dist_hh}

def score(dt,bm):
    fl=[]
    for code,info in C.items():
        mask=info['df']['date']<=dt;idx=mask.sum()-1
        if idx<60: continue
        f=factors(info['df'],idx)
        if f is None: continue
        f['code']=code;f['name']=info['name'];fl.append(f)
    if not fl: return pd.DataFrame(),0,0,0
    df=pd.DataFrame(fl);df=df[df['avg_amount']>=MIN_AMT].copy()
    if len(df)==0: return pd.DataFrame(),0,0,0
    sc={}
    for k,w in FW.items():
        if k not in df.columns: continue
        col=df[k].copy();m=col.mean();s=col.std()
        z=pd.Series(0.0,index=col.index) if (s==0 or pd.isna(s)) else ((col-m)/s).clip(-3,3)
        sc[k]=z*w
    df['score']=sum(sc.values())
    df=df.sort_values('score',ascending=False).reset_index(drop=True)
    df['rank']=range(1,len(df)+1)
    best=df['score'].iloc[0]
    ic=0;ib=0;ir=0
    if bm is not None:
        mb=bm[bm['date']<=dt]
        if len(mb)>=BEAR_MA:
            ma=mb['close'].rolling(BEAR_MA).mean().iloc[-1]
            ic=mb['close'].iloc[-1]
            ir=1 if ic>ma*1.05 else 3 if ic<ma else 2
            ib=1 if ic<ma else 0
    return df,best,ib,ir

# === MAIN ===
load()
bm=bench()
if bm is not None: dates=bm['date'].tolist()
else:
    for v in C.values(): dates=v['df']['date'].tolist();break
td=[d for d in dates if START<=d<=END]
print(f'Backtest:',td[0].date(),'~',td[-1].date(),f'({len(td)} days)')
print('Optimizations: V4.7 | FactorWeights(v2) | DynTP(8/12/15%) | MinScore=0.35 | Cooldown+MA20+DownBrake')

pos={};all_days=[];trades=[]
banned={}  # {code: ban_until_date}  cooling-off tracking

for i,dt in enumerate(td):
    sc,best,ib,ir=score(dt,bm)
    if len(sc)==0: continue
    total_stocks = len(sc)
    # V4.6: Short-term regime filter - CSI300 vs MA20 (only in NEUTRAL/BEAR)
    ib_short = 0
    if bm is not None:
        bm_slice = bm[bm['date'] <= dt]
        if len(bm_slice) >= SHORT_MA:
            ma20_val = bm_slice['close'].rolling(SHORT_MA).mean().iloc[-1]
            idx_val = bm_slice['close'].iloc[-1]
            ib_short = 1 if idx_val < ma20_val else 0

    mx_raw = POS_BEAR if ib else (POS_STRONG if ir==1 else POS_NEUTRAL)
    # V4.6: MA20 filter only in NEUTRAL/BEAR — BULL+ stays full throttle
    if ir == 1:
        mx = mx_raw  # BULL+: no MA20 restriction
    else:
        mx = max(POS_BEAR, mx_raw // 2) if ib_short else mx_raw

    # === EXITS (V4.3 final) ===
    to_del=[]
    for code,p in list(pos.items()):
        row=sc[sc['code']==code]
        if len(row)==0: to_del.append((code,'NO DATA'));continue
        r=row.iloc[0];cl=r['close'];rk=int(r['rank']);atr=r['atr_20'];ap=r['atr_pct']
        score_now = r['score']
        p['dh']+=1;pf=cl/p['ep']-1;p['hi']=max(p.get('hi',p['ep']),cl)

        # Dynamic hard stop: bear uses wider stop, bull uses ATR-based
        atr_sl = -EX['atr_sl_mult'] * atr / p['ep']
        if ib:
            hard_sl = EX['sl_bear']  # Bear: fixed -10%
        else:
            hard_sl = max(SL_FLOOR, min(EX['sl_hard'], atr_sl))

        # Emergency check
        day_drop = (cl - p.get('prev_close', p['ep'])) / p.get('prev_close', p['ep'])
        emergency = (day_drop <= EMERGENCY_DROP)

        # MA20 consecutive days counter
        below_ma20 = (cl < r['ma_20'])
        p['ma20_cnt'] = p.get('ma20_cnt', 0) + 1 if below_ma20 else 0

        atm=EX['atm_high'] if ap>5 else ((EX['atm_low']+EX['atm_high'])/2 if ap>2.5 else EX['atm_low'])
        # V4.7: Dynamic TP - fast movers lock early
        if ib:
            tp = EX['tp_bear']  # Bear: uniform 10%
        elif p['dh'] <= EX['tp_fast_days']:
            tp = EX['tp_fast']  # Fast: 8% within 3 days
        elif p.get('er',99) <= 3:
            tp = EX['tp_top']  # Top rank: 15%
        else:
            tp = EX['tp_std']  # Standard: 12%
        max_hold = 15 if p.get('er',99) <= 5 else EX['max_hold']

        # V4.5: Track peak profit for breakeven stop
        p['peak_pf'] = max(p.get('peak_pf', pf), pf)

        go=False;reason=''
        if pf<=hard_sl: go=True;reason=f'SL({pf*100:.1f}%)'
        # V4.5: Breakeven stop — if was up enough but drops back to near entry
        elif BREAKEVEN_ENABLED and p['dh']>=MIN_HOLD and not go and p.get('peak_pf',0)>=BREAKEVEN_PROFIT and pf<=0.01:
            go=True;reason=f'BE(peak{p["peak_pf"]*100:.0f}%→{pf*100:.0f}%)'
        elif emergency: go=True;reason=f'CRASH({day_drop*100:.1f}%)'
        elif p['dh']>=MIN_HOLD:
            if pf>=tp: go=True;reason=f'TP({tp*100:.0f}%)'
            elif p['dh']>=max_hold: go=True;reason=f'TIME({p["dh"]}d)'
            elif p['dh']>=TIME_LOSER and pf<0 and p['hi']<=p['ep']: go=True;reason=f'TIME_LOSS({p["dh"]}d)'
            elif p['ma20_cnt']>=MA20_CONSEC: go=True;reason=f'MA20x{MA20_CONSEC}'
            elif rk>total_stocks*EX['rank_out_ratio']: go=True;reason='RANK'
            elif rk-p.get('er',999)>EX['rank_drop']: go=True;reason='RANK_DROP'
            elif cl<p['hi']-atm*atr: go=True;reason='TRAIL'

        if go: to_del.append((code,reason))
        else: p['prev_close'] = cl

    for code,reason in to_del:
        p=pos[code]
        row_match = sc[sc['code']==code]
        if len(row_match) == 0:
            xp = p['ep']  # Use entry price as fallback
        else:
            xp = row_match.iloc[0]['close']
        trades.append({'dt':dt,'act':'SELL','c':code,'n':p['name'],'ep':p['ep'],
            'xp':xp, 'pnl':(xp/p['ep']-1)*100,
            'dh':p['dh'],'rs':reason})
        del pos[code]
        banned[code] = dt + pd.Timedelta(days=COOL_OFF)

    # === BUYS (V4.4 refined) ===
    # V4.6: Consecutive loss cooldown — if last N exits all lost, pause buying
    loss_cooldown_active = False
    if LOSS_COOLDOWN > 0:
        recent_sells = [t for t in trades if t['act'] == 'SELL']
        if len(recent_sells) >= LOSS_COOLDOWN:
            last_n = recent_sells[-LOSS_COOLDOWN:]
            if all(t['pnl'] <= 0 for t in last_n):
                last_loss_dt = max(t['dt'] for t in last_n)
                if (dt - last_loss_dt).days <= COOLDOWN_DAYS:
                    loss_cooldown_active = True

    # Bear market filter: CSI300 < MA60 → no new buys
    bear_filter_active = False
    if BEAR_FILTER and bm is not None:
        bm_slice = bm[bm['date'] <= dt]
        if len(bm_slice) >= BEAR_FILTER_MA:
            idx_ma60 = bm_slice['close'].rolling(BEAR_FILTER_MA).mean().iloc[-1]
            idx_now = bm_slice['close'].iloc[-1]
            bear_filter_active = (idx_now < idx_ma60)

    # Dynamic min_score: higher bar when nearly full
    n_held = len(pos)
    min_sc = MIN_SCORE_BASE + 0.05 * (n_held // 3)  # 0.20 at 0-2, 0.25 at 3-5, 0.30 at 6-8
    # V4.6: Down-day brake - skip buys if CSI300 dropped >2% today
    down_day_brake = False
    if bm is not None and DOWN_DAY_BRAKE < 0:
        bm_slice = bm[bm['date'] <= dt]
        if len(bm_slice) >= 2:
            yest_close = bm_slice['close'].iloc[-2]
            today_close = bm_slice['close'].iloc[-1]
            if (today_close / yest_close - 1) < DOWN_DAY_BRAKE:
                down_day_brake = True
    
    slots = mx - len(pos)
    if bear_filter_active:
        slots = 0  # No new buys in bear market
    if loss_cooldown_active:
        slots = 0  # V4.4: pause buying after consecutive losses
    if down_day_brake:
        slots = 0  # V4.6: skip buying after >2% down day
    if slots>0 and best>=min_sc:
        held=set(pos.keys())
        for _,r in sc.iterrows():
            if slots<=0: break
            code=r['code']
            if code in held or r['score']<min_sc: continue
            # Cooling-off check
            if code in banned and dt <= banned[code]:
                continue
            # V4.4: MA20 entry filter — close within tolerance of MA20 (default 3% below)
            if MA20_BUY_FILTER and r['close'] <= r['ma_20'] * MA20_BUY_TOLERANCE:
                continue
            # V4.4: No chasing — price must not be >15% above MA20
            if MA20_EXTEND_CAP > 0 and r['price_vs_ma20'] > MA20_EXTEND_CAP:
                continue
            # V4.5: Sector concentration cap — max SECTOR_CAP per industry
            if SECTOR_CAP > 0:
                sec = get_sector(code)
                sec_count = sum(1 for pc in pos.values() if get_sector(pc.get('code','')) == sec)
                if sec_count >= SECTOR_CAP:
                    continue
            # Late-slot quality check: rank 7+ needs score > best*0.7
            if int(r['rank']) >= 7 and r['score'] < best * 0.65:
                continue
            pos[code]={'code':code,'name':r['name'],'ep':r['close'],'dh':0,'hi':r['close'],'er':int(r['rank']),'prev_close':r['close']}
            trades.append({'dt':dt,'act':'BUY','c':code,'n':r['name'],'ep':r['close'],
                'xp':r['close'],'pnl':0,'dh':0,
                'rs':f'score={r["score"]:.3f} rk=#{int(r["rank"])} min={min_sc:.2f}'})
            slots-=1

    # === RECORD ===
    holds=[]
    for code,p in pos.items():
        row=sc[sc['code']==code]
        if len(row)>0:
            r=row.iloc[0];pnl=(r['close']/p['ep']-1)*100
            holds.append({'c':code,'n':p['name'],'ep':p['ep'],'cp':r['close'],'pnl':pnl,'dh':p['dh'],'er':p['er']})
    holds.sort(key=lambda x:x['pnl'],reverse=True)
    all_days.append({'dt':dt,'n':len(holds),'mx':mx,'reg':ir,'holds':holds,
                     'banned':len([k for k,v in banned.items() if dt<=v]),
                     'bear_f':bear_filter_active})
    reg=['?','BULL+','NEUTRAL','BEAR'][ir]
    if (i+1)%3==0 or i==0:
        print(f'  [{i+1:3d}/{len(td)}] {dt.date()} {reg} [{len(holds)}/{mx}] banned:{all_days[-1]["banned"]}')

# === REPORT ===
print()
print('='*100)
avg_h=sum(d['n'] for d in all_days)/max(len(all_days),1)
total_ban_days = sum(d.get('banned',0) for d in all_days)
total_trades = len(trades)
buys = [t for t in trades if t['act']=='BUY']
sells = [t for t in trades if t['act']=='SELL']
wins = [t for t in sells if t['pnl']>0]
losses = [t for t in sells if t['pnl']<=0]
avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
win_rate = len(wins)/len(sells)*100 if sells else 0

print(f'  DINGJIASHAN V4.3.1 (10-STOCK)')
print(f'  Period: {START.date()} ~ {END.date()} | Days: {len(all_days)} | Avg Hold: {avg_h:.1f}/10')
print(f'  V4.7: FW(v2:vol+amt+retrace) | DynTP(8/12/15%) | MinScore=0.35 | Cooldown+MA20+DownBrake | DynSL(-5~-8%)')
print(f'  Trades: {total_trades} ({len(buys)}B/{len(sells)}S) | WinRate: {win_rate:.1f}% | AvgWin: {avg_win:+.1f}% | AvgLoss: {avg_loss:+.1f}%')
print('='*100)

for d in all_days:
    dt=d['dt']
    wds=['MON','TUE','WED','THU','FRI','SAT','SUN']
    wd=wds[dt.weekday()]
    rl='BEAR' if d['reg']==3 else ('BULL+' if d['reg']==1 else 'NEU')
    bar='#'*d['n']+'-'*(d['mx']-d['n']) if d['n']>0 else '.'*d['mx']
    print()
    print(f'  {dt.date()} {wd} {rl} [{d["n"]}/{d["mx"]}] {bar} ban:{d.get("banned",0)}')
    for h in d['holds']:
        if h['pnl']>0: e='G'
        elif h['pnl']<-5: e='R'
        elif h['pnl']<0: e='Y'
        else: e='O'
        tp_mark = 'T' if h.get('er',99)<=3 else ''
        print(f'    {h["c"]:<8s} {h["n"]:<6s} {h["ep"]:>7.2f}>{h["cp"]:>7.2f} [{e}]{h["pnl"]:>+6.1f}% d{h["dh"]} #{h.get("er","?")}{tp_mark}')

# Quarterly breakdown
print()
print('='*100)
print('  QUARTERLY PERFORMANCE')
print('='*100)
qtrs={}
for t in trades:
    if t['act']!='SELL': continue
    dt=t['dt']
    q=f'{dt.year}-Q{(dt.month-1)//3+1}'
    if q not in qtrs: qtrs[q]={'t':0,'w':0,'pnl':0}
    qtrs[q]['t']+=1;qtrs[q]['pnl']+=t['pnl']
    if t['pnl']>0: qtrs[q]['w']+=1

# Count trading days per quarter
qdays={}
for d in all_days:
    dt=d['dt'];q=f'{dt.year}-Q{(dt.month-1)//3+1}'
    qdays[q]=qdays.get(q,0)+1

print(f"  {'Quarter':<10} {'Days':>5} {'Sells':>6} {'Win%':>6} {'SumPnL':>8} {'Return':>8} {'BearD':>6}")
print(f"  {'-'*56}")
cum_ret=0
for q in sorted(qtrs.keys()):
    qd=qtrs[q];wr=qd['w']/qd['t']*100 if qd['t'] else 0
    ret=qd['pnl']/8
    cum_ret+=ret
    bear_days = sum(1 for d in all_days if f'{d["dt"].year}-Q{(d["dt"].month-1)//3+1}'==q and d['reg']==3)
    print(f"  {q:<10} {qdays.get(q,0):>5} {qd['t']:>6} {wr:>5.1f}% {qd['pnl']:>+7.1f}% {ret:>+7.1f}% {bear_days:>6}")
print(f"  {'-'*56}")
total_bear_days = sum(1 for d in all_days if d['reg']==3)
total_sells = sum(qd['t'] for qd in qtrs.values())
total_pnl = sum(qd['pnl'] for qd in qtrs.values())
wr_all = sum(qd['w'] for qd in qtrs.values()) / total_sells * 100 if total_sells else 0
print(f"  {'TOTAL':<10} {len(all_days):>5} {total_sells:>6} {wr_all:>5.1f}% {total_pnl:>+7.1f}% {total_pnl/8:>+7.1f}% {total_bear_days:>6}")
print(f"  Bear filter: CSI300 < MA60 → no new buys. Total bear days: {total_bear_days}/{len(all_days)}")

# Monthly
print()
print('  MONTHLY')
mons={}
for d in all_days:
    mon=d['dt'].strftime('%Y-%m')
    if mon not in mons: mons[mon]={'d':0,'h':0,'c':Counter()}
    mons[mon]['d']+=1;mons[mon]['h']+=d['n']
    for h in d['holds']:mons[mon]['c'][h['c']]+=1
for mon,md in sorted(mons.items()):
    avg=md['h']/md['d'];top=md['c'].most_common(5)
    ts=', '.join([f'{c}({n}d)' for c,n in top])
    mon_sells=sum(1 for t in trades if t['act']=='SELL' and t['dt'].strftime('%Y-%m')==mon)
    mon_pnl=sum(t['pnl'] for t in trades if t['act']=='SELL' and t['dt'].strftime('%Y-%m')==mon)
    print(f'  {mon}: {md["d"]:>2d}d avg{avg:.0f}stk sells{mon_sells:>3d} PnL{mon_pnl:>+7.1f}% | {ts}')

# Final holdings
last=all_days[-1]
print()
print('='*100)
print(f'  FINAL HOLDINGS: {last["dt"].date()} ({last["n"]}/{last["mx"]})  banned:{last.get("banned",0)}')
print(f'  {"Code":<8} {"Name":<6} {"Entry":>8} {"Close":>8} {"P&L":>8} {"Days":>5} {"Rank":>5}')
print(f'  {"-"*65}')
for h in last['holds']:
    s='+' if h['pnl']>0 else ''
    tp_tag = ' (TP15%)' if h.get('er',99)<=3 else ' (TP12%)'
    print(f'  {h["c"]:<8} {h["n"]:<6} {h["ep"]:>8.2f} {h["cp"]:>8.2f} {s}{h["pnl"]:>7.1f}% {h["dh"]:>5}d {h.get("er","?"):>5}{tp_tag}')

# Trade summary
print()
print('='*100)
print('  TRADE LOG (all entries and exits)')
print(f'  {"Date":<12} {"Act":<5} {"Code":<8} {"Name":<6} {"Entry":>8} {"Exit":>8} {"PnL":>8} {"Days":>5} {"Reason"}')
print(f'  {"-"*90}')
for t in trades:
    dt_str = str(t['dt'].date()) if hasattr(t['dt'],'date') else str(t['dt'])[:10]
    if t['act']=='BUY':
        print(f'  {dt_str:<12} BUY   {t["c"]:<8} {t["n"]:<6} {"":>8} {t["ep"]:>8.2f} {"":>8} {"":>5} {t["rs"]}')
    else:
        s='+' if t['pnl']>0 else ''
        print(f'  {dt_str:<12} SELL  {t["c"]:<8} {t["n"]:<6} {t["ep"]:>8.2f} {t["xp"]:>8.2f} {s}{t["pnl"]:>7.1f}% {t["dh"]:>5}d {t["rs"]}')

# --- AUTO PUSH ---
import subprocess

# Name mapping
NAME_MAP = {
    '603259':'药明康德','688036':'传音控股','002558':'巨人网络','600584':'长电科技',
    '301165':'锐捷网络','000776':'广发证券','601211':'国泰海通','002422':'科伦药业',
    '688506':'百利天恒','000938':'紫光股份','000301':'东方盛虹','600176':'中国巨石',
    '600030':'中信证券','000725':'京东方A','600460':'士兰微','600183':'生益科技',
    '600549':'厦门钨业','600522':'中天科技','603260':'合盛硅业','600160':'巨化股份',
    '601066':'中信建投','600999':'招商证券','688082':'盛美上海','000657':'中钨高新',
    '600188':'兖矿能源','001965':'招商公路','601138':'工业富联','300502':'新易盛',
    '300408':'三环集团','600023':'浙能电力','601872':'招商轮船','688521':'芯原股份',
    '600362':'江西铜业','603986':'兆易创新','300433':'蓝思科技','688012':'中微公司',
    '600011':'华能国际','600027':'华电国际','300394':'天孚通信','300308':'中际旭创',
    '000100':'TCL科技','002463':'沪电股份','688126':'沪硅产业','002142':'宁波银行',
    '603501':'韦尔股份','000333':'美的集团','000630':'铜陵有色','001236':'弘元绿能',
    '600031':'三一重工',
}

def get_name(code):
    return NAME_MAP.get(code, code)

# Generate markdown report
last_n = min(3, len(all_days))
recent_days = all_days[-last_n:]

md = []
dt_str = last['dt'].date()
md.append(f'# 丁家山 V4.7 每日操盘信号\n')
md.append(f'**{dt_str}** | 持仓 {last["n"]}/{last["mx"]} | 黑名单 {last.get("banned",0)}只\n')
md.append(f'\n---\n')

# Recent exits (today only)
today_exits = [t for t in trades if t['act']=='SELL' and t['dt']==last['dt']]
today_buys = [t for t in trades if t['act']=='BUY' and t['dt']==last['dt']]

if today_exits:
    md.append(f'## 🔴 今日卖出 ({len(today_exits)}笔)\n\n')
    md.append(f'| 代码 | 名称 | 盈亏 | 持天 | 原因 |\n')
    md.append(f'|------|------|------|------|------|\n')
    for t in today_exits:
        emoji = '🟢' if t['pnl']>0 else '🔴'
        md.append(f'| {t["c"]} | {get_name(t["c"])} | {emoji} {t["pnl"]:+.1f}% | {t["dh"]}d | {t["rs"]} |\n')
    md.append('\n')

if today_buys:
    md.append(f'## 🟢 今日买入 ({len(today_buys)}笔)\n\n')
    md.append(f'| 代码 | 名称 | 买入价 | 排名 |\n')
    md.append(f'|------|------|--------|------|\n')
    for t in today_buys:
        md.append(f'| {t["c"]} | {get_name(t["c"])} | {t["ep"]:.2f} | {t["rs"]} |\n')
    md.append('\n')

# Current holdings
md.append(f'---\n')
md.append(f'## 📊 当前持仓 ({last["n"]}/{last["mx"]})\n\n')
md.append(f'| 代码 | 名称 | 买入价 | 现价 | 盈亏 | 天 | 止盈 |\n')
md.append(f'|------|------|--------|------|:---:|:---:|:---:|\n')
for h in last['holds']:
    emoji = '🟢' if h['pnl']>0 else ('🔴' if h['pnl']<-5 else '🟡' if h['pnl']<0 else '⚪')
    tp_tag = '15%' if h.get('er',99)<=3 else '12%'
    md.append(f'| {h["c"]} | {get_name(h["c"])} | {h["ep"]:.2f} | {h["cp"]:.2f} | {emoji} {h["pnl"]:+.1f}% | {h["dh"]}d | {tp_tag} |\n')

# Stats
total_trades = len(trades)
sells = [t for t in trades if t['act']=='SELL']
wins = [t for t in sells if t['pnl']>0]
losses = [t for t in sells if t['pnl']<=0]
win_rate = len(wins)/len(sells)*100 if sells else 0
avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

md.append(f'\n---\n')
md.append(f'## 📈 全期统计 (6/1~{dt_str})\n\n')
md.append(f'| 交易 | 胜率 | 均盈 | 均亏 | 盈亏比 |\n')
md.append(f'|:---:|:---:|:---:|:---:|:---:|\n')
md.append(f'| {len(sells)}笔 | {win_rate:.0f}% | +{avg_win:.1f}% | {avg_loss:.1f}% | {abs(avg_win/avg_loss) if avg_loss!=0 else 99:.1f} |\n')

md.append(f'\n---\n*{datetime.now(BJT).strftime("%Y-%m-%d %H:%M")} 北京时间 自动生成 | V4.7 | FactorV2+DynTP+MinScore+Cooldown+DownBrake | 仅供参考*\n')

report_text = ''.join(md)
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'daily_reports', 'LATEST.md')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

# Git push (only for daily runs, skip full backtest)
if (END - START).days < 60:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        r1 = subprocess.run(['git', 'add', 'daily_reports/'], cwd=repo_dir, capture_output=True, text=True, timeout=30)
        r2 = subprocess.run(['git', 'commit', '-m', f'daily signal {datetime.now(BJT).strftime("%Y-%m-%d")} V4.3.1'],
                          cwd=repo_dir, capture_output=True, text=True, timeout=30)
        r3 = subprocess.run(['git', 'push'], cwd=repo_dir, capture_output=True, text=True, timeout=60)
        if r3.returncode == 0:
            print('[OK] Pushed to GitHub')
        else:
            print(f'[WARN] Push: {r3.stderr.strip()[:100]}')
    except subprocess.TimeoutExpired as e:
        print(f'[WARN] Git timeout: {e.cmd} ({e.timeout}s)')
    except Exception as e:
        print(f'[WARN] Git error: {e}')

print('='*100)
print('DONE')

