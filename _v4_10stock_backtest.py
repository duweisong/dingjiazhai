# DINGJIASHAN V4.3 FINAL - 10 Stock Backtest
# V4.2 -> V4.3 CHANGES:
#  1. MA20 exit: only if below MA20 for 3 consecutive days
#  2. Dynamic SL floor: never looser than -5% (hard floor)
#  3. Time-stop: if held 5+ days AND losing, force exit
import os, pickle, numpy as np, pandas as pd
from datetime import datetime
from collections import Counter

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.strategy_cache')
START = pd.Timestamp('2026-07-01')
END = pd.Timestamp.now()  # Dynamic: today's date for daily run
POS_STRONG = 10; POS_NEUTRAL = 8; POS_BEAR = 4
BEAR_MA = 120; MIN_AMT = 30_000_000; MIN_SCORE_BASE = 0.2
COOL_OFF = 2
MIN_HOLD = 1
EMERGENCY_DROP = -0.08
BEAR_FILTER = False  # Disabled: hurts recovery, bear params are better
BEAR_FILTER_MA = 60  # Use 60-day MA for bear filter (faster than 120)
SL_FLOOR = -0.05  # hard floor: never looser than -5%
TIME_LOSER = 5     # exit if held this many days AND still losing
MA20_CONSEC = 3    # consecutive days below MA20 before exit

FW = {'ret_5d':0.10,'ret_10d':0.10,'ret_20d':0.10,'vol_ratio':0.20,
      'price_vs_ma20':0.20,'ma_alignment':0.12,'up_days_10':0.05,
      'turnover':0.05,'atr_pct':-0.08}
EX = {'atm_low':1.2,'atm_high':2.0,'max_hold':10,'tp_std':0.12,'tp_top':0.15,
      'rank_out_ratio':0.75,'rank_drop':50,'sl_hard':-0.08,'sl_bear':-0.10,
      'ma_p':20,'atr_sl_mult':2.5,'tp_bear':0.10}  # Bear: wider SL, tighter TP

C = {}

def get_hs300_list():
    """Fetch CSI300 constituent list from baostock (cached per day)"""
    import baostock as bs
    cache_file = os.path.join(CACHE_DIR, 'hs300_list.json')
    today = pd.Timestamp.now().strftime('%Y%m%d')
    if os.path.exists(cache_file):
        import json
        cached = json.load(open(cache_file, 'r', encoding='utf-8'))
        if cached.get('date') == today:
            return cached.get('stocks', {})
    try:
        bs.login()
        rs = bs.query_hs300_stocks()
        stocks = {}
        if rs.error_code == '0':
            while rs.next():
                row = rs.get_row_data()
                # Fields: updateDate, code, code_name
                code = row[1].replace('sh.', '').replace('sz.', '')
                name = row[2]
                stocks[code] = name
        bs.logout()
        if stocks:
            import json
            json.dump({'date': today, 'stocks': stocks}, open(cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
        return stocks
    except Exception as e:
        print(f'[WARN] HS300 list fetch failed: {e}')
        return {}

def download_data():
    """Download latest stock data from baostock for all HS300 stocks + CSI300 index"""
    import baostock as bs
    os.makedirs(CACHE_DIR, exist_ok=True)
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

    # Get HS300 constituent list first (own session, cached per day)
    stock_list = get_hs300_list()
    print(f'HS300 constituents: {len(stock_list)} stocks')

    if not stock_list:
        print('[ERROR] Cannot fetch HS300 list, aborting data download')
        return

    print(f'Downloading data: {start_date} ~ {end_date}')
    bs.login()

    # Download CSI300 index benchmark
    try:
        rs = bs.query_history_k_data_plus('sh.000300',
            'date,code,open,high,low,close,volume,amount',
            start_date=start_date, end_date=end_date,
            frequency='d', adjustflag='2')
        if rs.error_code == '0':
            data = []
            while rs.next(): data.append(rs.get_row_data())
            if len(data) >= 60:
                df = pd.DataFrame(data, columns=rs.fields)
                for col in ['open','high','low','close','volume','amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                pickle.dump({'df': df}, open(os.path.join(CACHE_DIR, '000300.pkl'), 'wb'))
                print(f'  CSI300: {len(df)} rows -> {df["date"].iloc[-1].date()}')
    except Exception as e:
        print(f'  CSI300 download failed: {e}')

    # Download individual stocks
    success = 0
    for code, name in stock_list.items():
        try:
            bs_code = f'sh.{code}' if (code.startswith('6') or code.startswith('68')) else f'sz.{code}'
            rs = bs.query_history_k_data_plus(bs_code,
                'date,code,open,high,low,close,volume,amount',
                start_date=start_date, end_date=end_date,
                frequency='d', adjustflag='2')
            if rs.error_code != '0':
                continue
            data = []
            while rs.next(): data.append(rs.get_row_data())
            if len(data) < 60:
                continue
            df = pd.DataFrame(data, columns=rs.fields)
            for col in ['open','high','low','close','volume','amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # Ensure required columns
            required = ['open','high','low','close','volume','amount']
            if not all(c in df.columns for c in required):
                continue
            pickle.dump({'df': df}, open(os.path.join(CACHE_DIR, f'{code}.pkl'), 'wb'))
            success += 1
            if success % 10 == 0:
                print(f'  Downloaded {success}/{len(stock_list)} stocks...')
        except Exception as e:
            print(f'  {code} ({name}) failed: {e}')

    bs.logout()
    print(f'Downloaded {success}/{len(stock_list)} stocks successfully')

def load():
    # Load HS300 name map for display names
    import json as _json
    name_map = {}
    hs300_file = os.path.join(CACHE_DIR, 'hs300_list.json')
    if os.path.exists(hs300_file):
        try:
            name_map = _json.load(open(hs300_file, 'r', encoding='utf-8')).get('stocks', {})
        except: pass

    skip_codes = {'000300', 'sh.000300', 'csi300_stocks'}
    for f in os.listdir(CACHE_DIR):
        if not f.endswith('.pkl'): continue
        code = f.replace('.pkl','')
        if code in skip_codes: continue
        try:
            d = pickle.load(open(os.path.join(CACHE_DIR,f),'rb'))
            df = d['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # Ensure required columns exist
            required = ['open','high','low','close','volume','amount']
            if not all(c in df.columns for c in required):
                continue
            C[code] = {'name': name_map.get(code, code), 'df': df}
        except: pass
    print('Loaded',len(C),'stocks')

def bench():
    for k in ['000300']:
        bp = os.path.join(CACHE_DIR,k+'.pkl')
        if os.path.exists(bp):
            d = pickle.load(open(bp,'rb'))
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
    return {'ret_5d':r5,'ret_10d':r10,'ret_20d':r20,'vol_ratio':vr,
        'price_vs_ma20':pvm,'ma_alignment':mal,'atr_pct':ap,
        'up_days_10':upd,'close':c[-1],'atr_20':a20,'ma_20':ma20,'avg_amount':amt}

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
try:
    download_data()
except Exception as e:
    print(f'[WARN] Data download failed: {e}')
    print('[INFO] Falling back to cached data...')

load()
if len(C) == 0:
    print('[ERROR] No stock data available. Check baostock connection or cache.')
    import sys; sys.exit(1)
bm=bench()
if bm is not None: dates=bm['date'].tolist()
else:
    for v in C.values(): dates=v['df']['date'].tolist();break
td=[d for d in dates if START<=d<=END]
print(f'Backtest:',td[0].date(),'~',td[-1].date(),f'({len(td)} days)')
print('Optimizations: SL-8% | CoolOff=3d | Tiered TP(12-15%) | FlexTime(10-15d) | MinHold=2d | PropRank')

pos={};all_days=[];trades=[]
banned={}  # {code: ban_until_date}  cooling-off tracking

for i,dt in enumerate(td):
    sc,best,ib,ir=score(dt,bm)
    if len(sc)==0: continue
    total_stocks = len(sc)
    mx=POS_BEAR if ib else (POS_STRONG if ir==1 else POS_NEUTRAL)

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
        # Bear uses uniform tight TP, bull uses tiered TP
        if ib:
            tp = EX['tp_bear']  # Bear: uniform 10%
        else:
            tp = EX['tp_top'] if p.get('er',99) <= 3 else EX['tp_std']
        max_hold = 15 if p.get('er',99) <= 5 else EX['max_hold']

        go=False;reason=''
        if pf<=hard_sl: go=True;reason=f'SL({pf*100:.1f}%)'
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

    # === BUYS (V4.2 refined) ===
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
    slots = mx - len(pos)
    if bear_filter_active:
        slots = 0  # No new buys in bear market
    if slots>0 and best>=min_sc:
        held=set(pos.keys())
        for _,r in sc.iterrows():
            if slots<=0: break
            code=r['code']
            if code in held or r['score']<min_sc: continue
            # Cooling-off check
            if code in banned and dt <= banned[code]:
                continue
            # Late-slot quality check: rank 7+ needs score > best*0.7
            if int(r['rank']) >= 7 and r['score'] < best * 0.65:
                continue
            pos[code]={'name':r['name'],'ep':r['close'],'dh':0,'hi':r['close'],'er':int(r['rank']),'prev_close':r['close']}
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

print(f'  DINGJIASHAN V4.3 FINAL (10-STOCK)')
print(f'  Period: {START.date()} ~ {END.date()} | Days: {len(all_days)} | Avg Hold: {avg_h:.1f}/10')
print(f'  V4.3 BearOpt: DynSL(-5%~-8%) | BearSL(-10%) | BearTP(10%) | Crash(8%) | CoolOff2d | MA20x3d | TimeLoss(5d)')
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
print(f"  {'TOTAL':<10} {len(all_days):>5} {len(sells):>6} {win_rate:>5.1f}% {sum(t['pnl'] for t in sells):>+7.1f}% {sum(t['pnl'] for t in sells)/8:>+7.1f}% {total_bear_days:>6}")
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
import subprocess, json

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
    '601066':'中信建投','600031':'三一重工',
}

def get_name(code):
    # Try HS300 dynamic list first, then fallback map, then raw code
    if code in C and C[code].get('name', code) != code:
        return C[code]['name']
    return NAME_MAP.get(code, code)

# Generate markdown report
last_n = min(3, len(all_days))
recent_days = all_days[-last_n:]

md = []
dt_str = last['dt'].date()
md.append(f'# 丁家山 V4.3.1 每日操盘信号\n')
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

md.append(f'\n---\n*{datetime.now().strftime("%Y-%m-%d %H:%M")} 自动生成 | V4.3.1 | 仅供参考*\n')

report_text = ''.join(md)
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'daily_reports', 'LATEST.md')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

# Git push (only for daily runs, skip full backtest)
if (END - START).days < 60:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        r1 = subprocess.run(['git', 'add', 'daily_reports/'], cwd=repo_dir, capture_output=True, text=True)
        r2 = subprocess.run(['git', 'commit', '-m', f'daily signal {datetime.now().strftime("%Y-%m-%d")} V4.3.1'],
                          cwd=repo_dir, capture_output=True, text=True)
        r3 = subprocess.run(['git', 'push'], cwd=repo_dir, capture_output=True, text=True)
        if r3.returncode == 0:
            print('[OK] Pushed to GitHub')
        else:
            print(f'[WARN] Push: {r3.stderr.strip()[:100]}')
    except Exception as e:
        print(f'[WARN] Git error: {e}')

print('='*100)
print('DONE')

