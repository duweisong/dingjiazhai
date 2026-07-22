"""
丁家山 丁家山 - Further Optimized
===============================
原名: CSI300 丁家山
Improvements: graduated exposure, sector limits, score filter,
dynamic trailing stop, rank crash exit, rebalanced weights
"""

import os, struct, pickle
import numpy as np
import pandas as pd
from collections import Counter

TDX_PATH = r'C:\zd_pazq_hy\vipdoc'
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
SLIPPAGE = 0.001
BEAR_MA_PERIOD = 120
MIN_AVG_AMOUNT = 50_000_000
MIN_SCORE_THRESHOLD = 0.3
MAX_SECTOR_POSITIONS = 2

MAX_POSITIONS = 5
MAX_POSITIONS_NEUTRAL = 3
MAX_POSITIONS_BEAR = 2

FACTOR_WEIGHTS = {
    'ret_5d': 0.12, 'ret_10d': 0.10, 'ret_20d': 0.10,
    'vol_ratio': 0.18, 'price_vs_ma20': 0.18, 'ma_alignment': 0.10,
    'up_days_10': 0.05, 'turnover': 0.05, 'atr_pct': -0.05,
    'ret_overbought': -0.07,
}

EXIT_CONFIG = {
    'atr_mult_low': 1.2, 'atr_mult_high': 2.0,
    'max_hold_days': 14, 'profit_target': 0.15,
    'ma_period': 20, 'rank_exit': 100, 'rank_drop_exit': 60,
    'stop_loss_hard': -0.08, 'stop_loss_bear': -0.05,
}

def get_sector(code):
    c = code[:3]
    if c in ['600','601','603','605']: return 'SH_MAIN'
    if c in ['688']: return 'STAR'
    if c in ['000','001','002','003']: return 'SZ_MAIN'
    if c in ['300','301']: return 'CHINEXT'
    return 'OTHER'

os.makedirs(CACHE_DIR, exist_ok=True)

# ====== DATA ======
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
    df=pd.DataFrame({'date':pd.to_datetime(dates,format='%Y%m%d'),
        'open':opens,'high':highs,'low':lows,'close':closes,
        'amount':amounts,'volume':volumes})
    return df.sort_values('date').reset_index(drop=True)

def get_day_path(code):
    pfx='sh' if (code.startswith('6') or code.startswith('68')) else 'sz'
    return os.path.join(TDX_PATH,pfx,'lday',f'{pfx}{code}.day')

def load_all_data(stocks):
    all_data={}; start_dt=pd.Timestamp(START_DATE)
    for i,s in enumerate(stocks):
        df=parse_day_file(get_day_path(s['code']))
        if df is not None:
            df=df[df['date']>=start_dt].copy()
            if len(df)>=60: all_data[s['code']]={'name':s['name'],'df':df}
        if (i+1)%50==0: print(f'  Loaded {i+1}/{len(stocks)}...')
    print(f'[DATA] {len(all_data)} stocks loaded')
    return all_data

def get_trading_dates(all_data):
    dates=set()
    for info in all_data.values(): dates.update(info['df']['date'].dt.strftime('%Y-%m-%d').tolist())
    return sorted([d for d in dates if d>=START_DATE])

# ====== FACTORS ======
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
    ret_overbought=max(0,ret_5d-25) if ret_5d>25 else 0
    avg_amount=np.mean(d['amount'].values[-21:-1]) if n>=21 else 0

    return {'ret_5d':ret_5d,'ret_10d':ret_10d,'ret_20d':ret_20d,
        'vol_ratio':vol_ratio,'price_vs_ma20':price_vs_ma20,
        'ma_alignment':ma_alignment,'atr_pct':atr_pct,
        'turnover':1.0,'up_days_10':up_days_10,'ret_overbought':ret_overbought,
        'close':close[-1],'atr_20':atr_20,'ma_20':ma_20,'ma_60':ma_60,
        'avg_amount':avg_amount}

def score_stocks(all_data, date_str):
    fl=[]
    for code,info in all_data.items():
        f=compute_factors(info['df'],date_str)
        if f is None: continue
        f['code']=code; f['name']=info['name']; fl.append(f)
    if len(fl)==0: return pd.DataFrame()
    df=pd.DataFrame(fl)
    df=df[df['avg_amount']>=MIN_AVG_AMOUNT].copy()
    if len(df)==0: return pd.DataFrame()
    sc={}
    for factor,weight in FACTOR_WEIGHTS.items():
        if factor not in df.columns: continue
        col=df[factor].copy(); m=col.mean(); s=col.std()
        z=pd.Series(0.0,index=col.index) if (s==0 or pd.isna(s)) else ((col-m)/s).clip(-3,3)
        sc[factor]=z*weight
    df['score']=sum(sc.values())
    df=df.sort_values('score',ascending=False).reset_index(drop=True)
    df['rank']=range(1,len(df)+1)
    return df

# ====== PORTFOLIO ======
class Portfolio:
    def __init__(self):
        self.cash=INITIAL_CAPITAL; self.positions={}; self.trades=[]
        self.equity=[]; self.dates=[]
        self.is_bear=False; self.is_strong=False
        self.bear_days=0; self.bull_days=0; self.strong_days=0
        self.sectors=Counter()

    def max_pos(self):
        if self.is_bear: return MAX_POSITIONS_BEAR
        if self.is_strong: return MAX_POSITIONS
        return MAX_POSITIONS_NEUTRAL

    def check_exit(self, pos, close, ma20, rank, days_held, atr_pct):
        profit=(close/pos['entry_price']-1)
        highest=pos.get('highest',pos['entry_price'])
        highest=max(highest,close); pos['highest']=highest
        entry_atr=pos.get('entry_atr',pos['entry_price']*0.03)
        cfg=EXIT_CONFIG

        hard=cfg['stop_loss_bear'] if self.is_bear else cfg['stop_loss_hard']
        if atr_pct>5: atm=cfg['atr_mult_high']
        elif atr_pct>2.5: atm=(cfg['atr_mult_low']+cfg['atr_mult_high'])/2
        else: atm=cfg['atr_mult_low']

        if profit<=hard: return True,'HardStop'
        if close<highest-atm*entry_atr: return True,'TrailStop'
        if close<ma20: return True,'BelowMA20'
        if profit>=cfg['profit_target']: return True,'ProfitTgt'
        if days_held>=cfg['max_hold_days']: return True,'TimeExit'
        if rank>cfg['rank_exit']: return True,'RankDrop'
        if rank-pos.get('entry_rank',999)>cfg['rank_drop_exit'] and days_held>=2:
            return True,'RankCrash'
        return False,''

    def sell(self,code,price,reason,date_str):
        pos=self.positions[code]; shares=pos['shares']
        gross=shares*price; cost=gross*(COMMISSION+SLIPPAGE)
        self.cash+=gross-cost
        buy_cost=shares*pos['entry_price']*(1+COMMISSION+SLIPPAGE)
        pnl=((gross-cost)/buy_cost-1)*100
        regime='BEAR' if self.is_bear else ('STRONG' if self.is_strong else 'NEUTRAL')
        self.trades.append({'date':date_str,'code':code,'name':pos['name'],
            'entry_date':pos['entry_date'],'entry_price':pos['entry_price'],
            'exit_price':price,'profit_pct':pnl,'days_held':pos['days_held'],
            'reason':reason,'regime':regime})
        sec=pos.get('sector','OTHER')
        self.sectors[sec]=max(0,self.sectors.get(sec,0)-1)
        del self.positions[code]

    def buy(self,code,name,price,score,rank,date_str,atr,atr_pct):
        sec=get_sector(code)
        if self.sectors.get(sec,0)>=MAX_SECTOR_POSITIONS: return False

        # FIX: use mark-like valuation for existing positions
        total_eq=self.cash
        # Approximate - use entry prices as proxy since we don't have all_data here
        # The mark() function does proper valuation; here we use a conservative estimate
        for c,p in self.positions.items():
            total_eq+=p['shares']*p['entry_price']  # conservative: use entry not current
        max_p=self.max_pos()
        base=total_eq/max(max_p,1)

        if atr_pct>5: va=0.6
        elif atr_pct>3: va=0.8
        else: va=1.0

        target=base*va
        if target>self.cash: target=self.cash
        shares=int(target/(price*(1+COMMISSION+SLIPPAGE))/100)*100
        if shares<100: return False
        cost=shares*price*(1+COMMISSION+SLIPPAGE)
        if cost>self.cash: return False
        self.cash-=cost
        self.positions[code]={'name':name,'entry_price':price,'entry_date':date_str,
            'shares':shares,'days_held':0,'highest':price,'entry_atr':atr,
            'entry_score':score,'entry_rank':rank,'atr_pct':atr_pct,'sector':sec}
        self.sectors[sec]=self.sectors.get(sec,0)+1
        return True

    def mark(self,date_str,all_data):
        total=self.cash
        for code,pos in self.positions.items():
            if code in all_data:
                d=all_data[code]['df']; d=d[d['date']<=date_str]
                if len(d)>0: total+=pos['shares']*d['close'].iloc[-1]
        self.equity.append(total); self.dates.append(date_str)
        return total

# ====== BACKTEST ======
def run_backtest(all_data):
    dates=get_trading_dates(all_data)
    bench_path=os.path.join(TDX_PATH,'sh','lday','sh000300.day')
    bench_df=parse_day_file(bench_path)
    bench_ma=None
    if bench_df is not None:
        bench_df=bench_df[bench_df['date']>=START_DATE].copy()
        bench_ma=bench_df.set_index('date')['close'].rolling(BEAR_MA_PERIOD).mean()

    print(f'\n=== 丁家山: {dates[0]} ~ {dates[-1]} ({len(dates)} days) ===\n')
    pf=Portfolio(); buys=sells=0

    for idx,date_str in enumerate(dates):
        # Market regime: 3 tiers
        if bench_ma is not None:
            dt=pd.Timestamp(date_str)
            if dt in bench_ma.index:
                ma_val=bench_ma.loc[dt]
                ic=bench_df[bench_df['date']==dt]['close']
                if len(ic)>0:
                    idx_close=ic.iloc[0]
                    pf.is_bear=idx_close<ma_val
                    pf.is_strong=idx_close>ma_val*1.05  # 5% above MA
                    if pf.is_bear: pf.bear_days+=1
                    elif pf.is_strong: pf.strong_days+=1
                    else: pf.bull_days+=1

        scored=score_stocks(all_data,date_str)
        if len(scored)==0: continue

        # 丁家山: Score quality filter
        best_score=scored['score'].iloc[0] if len(scored)>0 else -99
        skip_buy=best_score<MIN_SCORE_THRESHOLD

        info_map={}
        for _,row in scored.iterrows():
            info_map[row['code']]={'close':row['close'],'ma20':row['ma_20'],
                'atr':row['atr_20'],'rank':int(row['rank']),'name':row['name'],
                'score':float(row['score']),'atr_pct':row['atr_pct']}

        # Exits
        to_sell=[]
        for code,pos in list(pf.positions.items()):
            if code not in info_map: continue
            inf=info_map[code]; pos['days_held']+=1
            do_exit,reason=pf.check_exit(pos,inf['close'],inf['ma20'],
                inf['rank'],pos['days_held'],inf.get('atr_pct',2))
            if do_exit: to_sell.append((code,reason))
        for code,reason in to_sell:
            pf.sell(code,info_map[code]['close'],reason,date_str); sells+=1

        # Buys
        slots=pf.max_pos()-len(pf.positions)
        if slots>0 and not skip_buy:
            held=set(pf.positions.keys())
            cand=scored[~scored['code'].isin(held)]
            for _,row in cand.iterrows():
                if slots<=0: break
                code=row['code']
                if code in info_map:
                    ok=pf.buy(code,row['name'],info_map[code]['close'],
                        float(row['score']),int(row['rank']),date_str,
                        info_map[code]['atr'],info_map[code].get('atr_pct',2))
                    if ok: buys+=1; slots-=1

        total=pf.mark(date_str,all_data)

        if (idx+1)%50==0 or idx==0:
            ret=(total/INITIAL_CAPITAL-1)*100
            regime='BEAR' if pf.is_bear else ('STRONG' if pf.is_strong else 'NEUT')
            print(f'  [{date_str}] D{idx+1}/{len(dates)} {regime} | '
                  f'Pos={len(pf.positions)}/{pf.max_pos()} | Val={total:,.0f} | Ret={ret:+.2f}%')

    # Results
    eq=pd.Series(pf.equity,index=pd.to_datetime(pf.dates))
    rets=eq.pct_change().dropna()
    total_ret=(eq.iloc[-1]/INITIAL_CAPITAL-1)*100
    ann_ret=((eq.iloc[-1]/INITIAL_CAPITAL)**(252/len(rets))-1)*100
    rf=0.03; exc=rets-rf/252
    sharpe=np.sqrt(252)*exc.mean()/exc.std() if exc.std()>0 else 0
    max_dd=((eq-eq.cummax())/eq.cummax()*100).min()

    st=pf.trades
    wins=[t for t in st if t['profit_pct']>0]
    losses=[t for t in st if t['profit_pct']<=0]
    wr=len(wins)/len(st)*100 if st else 0
    aw=np.mean([t['profit_pct'] for t in wins]) if wins else 0
    al=np.mean([t['profit_pct'] for t in losses]) if losses else 0

    bench_ret=None
    if bench_df is not None and len(bench_df)>0:
        bench_ret=(bench_df['close'].iloc[-1]/bench_df['close'].iloc[0]-1)*100

    print(f'\n{"="*60}')
    print(f'  丁家山 RESULTS')
    print(f'{"="*60}')
    print(f'  Total Return:   {total_ret:+.2f}%')
    print(f'  Annual Return:  {ann_ret:+.2f}%')
    print(f'  Sharpe Ratio:   {sharpe:.2f}')
    print(f'  Max Drawdown:   {max_dd:.2f}%')
    print(f'  Volatility:     {rets.std()*np.sqrt(252)*100:.2f}%')
    print(f'  Trades:         {len(st)} (Buy: {buys})')
    print(f'  Win Rate:       {wr:.1f}%')
    print(f'  Avg Win:        {aw:+.2f}%')
    print(f'  Avg Loss:       {al:+.2f}%')
    if al!=0: print(f'  Profit Factor:  {abs(aw/al):.2f}')
    if st: print(f'  Avg Hold Days:  {np.mean([t["days_held"] for t in st]):.1f}')
    total_days=pf.bear_days+pf.bull_days+pf.strong_days
    if total_days>0:
        print(f'')
        print(f'  Strong Bull:    {pf.strong_days}d ({pf.strong_days/total_days*100:.0f}%)')
        print(f'  Neutral:        {pf.bull_days}d ({pf.bull_days/total_days*100:.0f}%)')
        print(f'  Bear:           {pf.bear_days}d ({pf.bear_days/total_days*100:.0f}%)')
    if bench_ret is not None:
        print(f'  CSI 300 Index:  {bench_ret:+.2f}%')
        print(f'  Alpha:          {total_ret-bench_ret:+.2f}%')
    print(f'{"="*60}')

    # Yearly
    yearly=eq.resample('YE').last(); prev=INITIAL_CAPITAL
    print(f'\n  年度:')
    for dt,val in yearly.items():
        ret=(val/prev-1)*100; print(f'    {dt.year}: {ret:+.2f}%'); prev=val

    # Exit reasons
    reasons=Counter(t['reason'] for t in st)
    print(f'\n  退出:')
    for r,c in reasons.most_common():
        avg=np.mean([t['profit_pct'] for t in st if t['reason']==r])
        print(f'    {r}: {c}次 ({c/len(st)*100:.1f}%) avg={avg:+.2f}%')

    result={'strategy':'丁家山','dates':pf.dates,'equity':pf.equity,'trades':pf.trades,
        'summary':{'total_return':total_ret,'annual_return':ann_ret,'sharpe':sharpe,
        'max_drawdown':max_dd,'win_rate':wr,'avg_win':aw,'avg_loss':al,
        'total_trades':len(st),'benchmark_return':bench_ret,
        'alpha':(total_ret-bench_ret) if bench_ret else None}}
    pickle.dump(result,open(os.path.join(CACHE_DIR,'v3_result.pkl'),'wb'))
    return result

if __name__=='__main__':
    cache_file=os.path.join(CACHE_DIR,'csi300_stocks.pkl')
    stocks=pickle.load(open(cache_file,'rb'))['stocks']
    print(f'[LOAD] CSI 300: {len(stocks)} stocks')
    all_data=load_all_data(stocks)
    result=run_backtest(all_data)
