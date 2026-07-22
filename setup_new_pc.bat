@echo off
echo ============================================
echo   丁家山量化系统 - 新机一键安装
echo ============================================
echo.

REM 1. Install Python dependencies
echo [1/3] Installing Python packages...
pip install numpy pandas baostock -q
echo   Done.

REM 2. Create cache directory
echo [2/3] Creating cache directory...
if not exist ".strategy_cache" mkdir .strategy_cache
echo   Done.

REM 3. First data fetch (300 stocks, ~10 min)
echo [3/3] First-time data fetch (this will take ~10 minutes)...
echo   Fetching CSI300 stock data from baostock...
python -X utf8 -c "
import os, pickle, pandas as pd, baostock as bs
from datetime import datetime

CACHE = '.strategy_cache'

# Fetch CSI300 index
bs.login()
print('  Fetching CSI300 index...')
rs = bs.query_history_k_data_plus('sh.000300','date,close',start_date='2021-01-01',end_date='2026-12-31',frequency='d',adjustflag='2')
data=[]
while rs.next(): data.append(rs.get_row_data())
df=pd.DataFrame(data,columns=['date','close'])
df['date']=pd.to_datetime(df['date'])
df['close']=pd.to_numeric(df['close'],errors='coerce')
pickle.dump({'date':datetime.now().strftime('%Y-%m-%d'),'df':df},open(os.path.join(CACHE,'000300.pkl'),'wb'))
print(f'  CSI300: {len(df)} rows')

# Core stock list
CODES = ['600519','000858','601318','600036','000333','600276','002415','000001','002594',
'601888','600030','000651','600900','002475','300750','601398','601166','600887','600809',
'000568','000002','601012','600585','601668','600028','601857','601088','600309','002352',
'300059','002714','000725','688981','601899','600690','601601','002304','000338','300015',
'600031','601225','002142','600048','601328','600016','601939','601288','600104','002271',
'600050','600183','300433','601066','000657','600549','603260','600522','300408','600160',
'688012','002371','000100','601211','600188','002142','688082','603986','688521','601872',
'002463','002422','688506','301165','000776','600999','603259','000938','000301','600176',
'600584','601211','688036','002558','600460','603501','002050','603019','600011','600027',
'300394','300308','688126','000630','603993','600188','300502','601138','601872','600362',
'000657','002463','600999','601211','600030','603260','600176','000630','603986','000725',
'300433','600460','688012','000301']

total = len(CODES)
for i, code in enumerate(CODES):
    bs_code = f'sh.{code}' if (code.startswith('6') or code.startswith('68')) else f'sz.{code}'
    try:
        rs = bs.query_history_k_data_plus(bs_code,'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date='2021-01-01',end_date='2026-12-31',frequency='d',adjustflag='2')
        if rs.error_code != '0': continue
        stock_data=[]
        while rs.next(): stock_data.append(rs.get_row_data())
        if len(stock_data)<100: continue
        df_s=pd.DataFrame(stock_data,columns=rs.fields)
        for col in ['open','high','low','close','volume','amount','turn','pctChg']:
            df_s[col]=pd.to_numeric(df_s[col],errors='coerce')
        df_s=df_s.rename(columns={'turn':'turnover','pctChg':'pct_chg'})
        df_s['date']=pd.to_datetime(df_s['date'])
        df_s=df_s.sort_values('date').reset_index(drop=True)
        pickle.dump({'date':datetime.now().strftime('%Y-%m-%d'),'df':df_s},open(os.path.join(CACHE,f'{code}.pkl'),'wb'))
    except: pass
    if (i+1)%20==0: print(f'  [{i+1}/{total}]')

bs.logout()
print('  All data fetched!')
"
echo   Done.

echo.
echo ============================================
echo   Setup complete!
echo.
echo   Try: python analyze.py 茅台 -t
echo        python _v4_10stock_backtest.py
echo ============================================
pause
