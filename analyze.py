#!/usr/bin/env python3
"""Stock analysis tool - quick scan + investment team mode"""
import sys, os, pickle, numpy as np, pandas as pd, io, argparse
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CACHE_DIR = r'C:\AI\.strategy_cache'

# Chinese name → code mapping (common A-shares)
NAME_MAP = {
    '贵州茅台':'600519','五粮液':'000858','宁德时代':'300750','比亚迪':'002594',
    '中国平安':'601318','招商银行':'600036','美的集团':'000333','格力电器':'000651',
    '恒瑞医药':'600276','药明康德':'603259','隆基绿能':'601012','中芯国际':'688981',
    '海康威视':'002415','立讯精密':'002475','京东方':'000725','京东方A':'000725',
    '紫金矿业':'601899','万华化学':'600309','长江电力':'600900','伊利股份':'600887',
    '东方财富':'300059','中信证券':'600030','牧原股份':'002714','顺丰控股':'002352',
    '泸州老窖':'000568','山西汾酒':'600809','洋河股份':'002304','海尔智家':'600690',
    '三一重工':'600031','保利发展':'600048','万科':'000002','万科A':'000002',
    '中国中免':'601888','中国神华':'601088','陕西煤业':'601225','中国石油':'601857',
    '中国石化':'600028','工商银行':'601398','建设银行':'601939','农业银行':'601288',
    '中国建筑':'601668','中国联通':'600050','上汽集团':'600104','潍柴动力':'000338',
    '爱尔眼科':'300015','宁波银行':'002142','海螺水泥':'600585','中兴通讯':'000063',
    'TCL科技':'000100','北方华创':'002371','中微公司':'688012','韦尔股份':'603501',
    '联创电子':'002036','南天信息':'000948','润邦股份':'002483','东方雨虹':'002271',
    '广发证券':'000776','招商证券':'600999','国泰海通':'601211','中信建投':'601066',
    '传音控股':'688036','巨人网络':'002558','长电科技':'600584','锐捷网络':'301165',
    '科伦药业':'002422','百利天恒':'688506','紫光股份':'000938','东方盛虹':'000301',
    '中国巨石':'600176','生益科技':'600183','士兰微':'600460','厦门钨业':'600549',
    '中天科技':'600522','合盛硅业':'603260','巨化股份':'600160','中钨高新':'000657',
    '兖矿能源':'600188','工业富联':'601138','新易盛':'300502','三环集团':'300408',
    '浙能电力':'600023','招商轮船':'601872','芯原股份':'688521','江西铜业':'600362',
    '兆易创新':'603986','蓝思科技':'300433','华能国际':'600011','华电国际':'600027',
    '天孚通信':'300394','中际旭创':'300308','沪电股份':'002463','沪硅产业':'688126',
    '铜陵有色':'000630','盛美上海':'688082','豪威集团':'603501',
}

def resolve_code(query):
    """Resolve code or Chinese name to stock code"""
    q = query.strip()
    if q.isdigit() and len(q) == 6:
        return q
    for name, code in NAME_MAP.items():
        if name in q or q in name:
            return code
    return None

def fetch_stock(code):
    """Fetch from baostock if not in cache"""
    f = os.path.join(CACHE_DIR, f'{code}.pkl')
    if os.path.exists(f):
        return pickle.load(open(f,'rb'))['df']
    try:
        import baostock as bs
        bs.login()
        bs_code = f'sh.{code}' if (code.startswith('6') or code.startswith('68')) else f'sz.{code}'
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date='2025-01-01', end_date='2026-07-07', frequency='d', adjustflag='2')
        data=[]
        while rs.next(): data.append(rs.get_row_data())
        bs.logout()
        if len(data)<60: return None
        df = pd.DataFrame(data, columns=rs.fields)
        for col in ['open','high','low','close','volume','amount','turn','pctChg']:
            df[col]=pd.to_numeric(df[col],errors='coerce')
        df=df.rename(columns={'turn':'turnover','pctChg':'pct_chg'})
        df['date']=pd.to_datetime(df['date'])
        df=df.sort_values('date').reset_index(drop=True)
        pickle.dump({'date':datetime.now().strftime('%Y-%m-%d'),'df':df}, open(f,'wb'))
        return df
    except:
        return None

def get_name(code):
    for n,c in NAME_MAP.items():
        if c==code: return n
    return code

def quick_scan(code):
    df = fetch_stock(code)
    if df is None: print(f'{code}: no data'); return
    df['date']=pd.to_datetime(df['date']); df=df.sort_values('date').reset_index(drop=True)
    last=df.iloc[-1]; c=df['close'].values; h=df['high'].values; l=df['low'].values
    v=df['volume'].values; n=len(df); cl=c[-1]; name=get_name(code)
    ma5=np.mean(c[-6:-1]); ma10=np.mean(c[-11:-1]); ma20=np.mean(c[-21:-1]); ma60=np.mean(c[-61:-1])
    r5=(cl/c[-6]-1)*100; r10=(cl/c[-11]-1)*100; r20=(cl/c[-21]-1)*100
    v20=np.mean(v[-21:-1]); vr=v[-1]/max(v20,1)
    up_days=sum(1 for i in range(max(1,n-10),n) if c[i]>c[i-1])
    atr20=np.mean([max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(max(1,n-20),n)])
    mp=(h+l)/2; ao=pd.Series(mp).rolling(5).mean()-pd.Series(mp).rolling(34).mean()
    ao_up=ao.iloc[-1]>ao.iloc[-6]
    max_dd=(c[-250:]/pd.Series(c[-250:]).cummax()-1).min()*100
    vol=pd.Series(c[-60:]).pct_change().std()*np.sqrt(252)*100
    ma_score=1.0 if ma5>ma10>ma20>ma60 else (0.5 if ma5>ma20 else 0)
    vs20=(cl/ma20-1)*100
    score=(r5*0.10+r10*0.10+r20*0.10+vr*0.20+vs20*0.20+ma_score*0.12+up_days*0.05-atr20/cl*100*0.08)
    sig='GREEN' if ao_up and vs20>-5 else ('YELLOW' if ao_up else 'RED')

    print(f"\n  {code} {name:<8s} {last['date'].date()} close={cl:.2f}")
    print(f"  5d:{r5:+.1f}% 10d:{r10:+.1f}% 20d:{r20:+.1f}% | AO:{ao.iloc[-1]:.4f} {'UP' if ao_up else 'DOWN'}")
    print(f"  MA20:{ma20:.2f} vsMA20:{vs20:+.1f}% | V4score:{score:+.3f} | Vol:{vol:.0f}% DD:{max_dd:.0f}% | Signal:{sig}")

def team_analysis(code):
    df = fetch_stock(code)
    if df is None: print(f'{code}: no data'); return
    df['date']=pd.to_datetime(df['date']); df=df.sort_values('date').reset_index(drop=True)
    last=df.iloc[-1]; c=df['close'].values; h=df['high'].values; l=df['low'].values
    v=df['volume'].values; n=len(df); cl=c[-1]; name=get_name(code)
    ma5=np.mean(c[-6:-1]); ma10=np.mean(c[-11:-1]); ma20=np.mean(c[-21:-1]); ma60=np.mean(c[-61:-1])
    r5=(cl/c[-6]-1)*100; r10=(cl/c[-11]-1)*100; r20=(cl/c[-21]-1)*100
    v20=np.mean(v[-21:-1]); vr=v[-1]/max(v20,1)
    up_days=sum(1 for i in range(max(1,n-10),n) if c[i]>c[i-1])
    atr20=np.mean([max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(max(1,n-20),n)])
    mp=(h+l)/2; ao=pd.Series(mp).rolling(5).mean()-pd.Series(mp).rolling(34).mean()
    ao_up=ao.iloc[-1]>ao.iloc[-6]
    max_dd=(c[-250:]/pd.Series(c[-250:]).cummax()-1).min()*100
    vol=pd.Series(c[-60:]).pct_change().std()*np.sqrt(252)*100
    kc_low=pd.Series((h+l+c)/3).ewm(span=20,adjust=False).mean()-2*pd.Series([max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,n)]+[atr20]*20).rolling(20).mean()

    scores=[2.0, 2.5, 1.5, 1.5, 2.0, 2.0]  # defaults
    # Adjust based on actual data
    if r5>3: scores[0]+=1  # 方新侠
    if r20>10: scores[1]+=0.5  # 宁高宁
    if ao_up and vr>1.2: scores[2]+=1  # 雷霆点睛
    if cl>ma60: scores[3]+=1  # 尚局说
    if cl>kc_low.iloc[-1]: scores[4]+=0.5  # 弈樊
    if max_dd>-30: scores[5]+=0.5  # 徐翔

    total=sum(scores); weighted=scores[0]*0.15+scores[1]*0.25+scores[2]*0.15+scores[3]*0.15+scores[4]*0.15+scores[5]*0.15
    verdict='PASS' if weighted>=2.5 else ('GRAY' if weighted>=2.0 else 'FAIL')
    vc='GREEN' if verdict=='PASS' else ('YELLOW' if verdict=='GRAY' else 'RED')

    print(f"""
+{"="*60}+
|  INVESTMENT COMMITTEE: {code} {name}  {last['date'].date()} close={cl:.2f}
+{"="*60}+
| Fang Xinxia (fund flow):  {scores[0]:.1f}/5 5d:{r5:+.1f}% vol:{vr:.2f} up:{up_days}/10
| Ning Gaoning (industry):  {scores[1]:.1f}/5 20d:{r20:+.1f}% DD:{max_dd:.0f}%
| Lei Ting (technicals):    {scores[2]:.1f}/5 AO:{ao.iloc[-1]:.4f} {'UP' if ao_up else 'DOWN'} ATR:{atr20/cl*100:.1f}%
| Shang Ju (macro trend):   {scores[3]:.1f}/5 MA60:{ma60:.2f} MA20:{ma20:.2f}
| Yi Fan (structure):       {scores[4]:.1f}/5 KC_low:{kc_low.iloc[-1]:.2f} align:{'BULL' if ma5>ma20 else 'BEAR'}
| Xu Xiang (deep research): {scores[5]:.1f}/5 Vol:{vol:.0f}% DD:{max_dd:.0f}%
+{"="*60}+
| WEIGHTED: {weighted:.2f}/5.0  VERDICT: [{vc}] {verdict}
+{"="*60}+
""")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stock analysis tool')
    p.add_argument('query', nargs='+', help='Stock code or Chinese name')
    p.add_argument('--team', '-t', action='store_true', help='Investment team mode')
    args = p.parse_args()

    for q in args.query:
        # Handle --team flag properly
        if q == '--team' or q == '-t':
            args.team = True
            continue
        code = resolve_code(q)
        if not code:
            print(f'Unknown: {q}')
            continue
        if args.team:
            team_analysis(code)
        else:
            quick_scan(code)
