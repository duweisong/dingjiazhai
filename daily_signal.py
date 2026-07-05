"""
丁家山 V4 每日信号推送系统
==========================
每天自动:
  1. 在线获取CSI300行情 (baostock)
  2. 全市场评分排名
  3. 检查持仓退出条件
  4. 生成买入/卖出信号
  5. 输出持仓报告 → 推送GitHub
  6. 保存状态供次日对比
"""

import os, pickle, json
import numpy as np
import pandas as pd
import baostock as bs
from datetime import datetime
from collections import Counter

# ====== 配置 ======
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
SLIPPAGE = 0.001
BEAR_MA_PERIOD = 120
MIN_AVG_AMOUNT = 50_000_000
MIN_SCORE = 0.3

# 丁家山 V4 参数
FACTOR_WEIGHTS = {
    'ret_5d': 0.10, 'ret_10d': 0.10, 'ret_20d': 0.10,
    'vol_ratio': 0.20, 'price_vs_ma20': 0.20, 'ma_alignment': 0.12,
    'up_days_10': 0.05, 'turnover': 0.05, 'atr_pct': -0.08,
}

EXIT_CONFIG = {
    'atr_mult_low': 1.2, 'atr_mult_high': 2.0,
    'max_hold_days': 10, 'profit_target': 0.12,
    'ma_period': 20, 'rank_exit': 80, 'rank_drop_exit': 50,
    'stop_loss_hard': -0.05, 'stop_loss_bear': -0.03,
}

POS_STRONG = 5; POS_NEUTRAL = 4; POS_BEAR = 1
SINGLE_PCT = 0.12; SINGLE_BEAR = 0.08; CASH_RESERVE = 0.35

PORTFOLIO_FILE = os.path.join(CACHE_DIR, 'live_portfolio.json')
REPORT_DIR = r'C:\AI\daily_reports'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ====== 在线数据获取 (baostock) ======
CSI300_STOCKS = None  # lazy load

def get_csi300_list():
    """获取CSI300成分股列表 (缓存1天)"""
    cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
    today = pd.Timestamp.now().strftime('%Y%m%d')
    if os.path.exists(cache_file):
        cached = pickle.load(open(cache_file, 'rb'))
        if cached.get('date') == today:
            return cached['stocks']
    # 使用预置列表 + baostock补充
    stocks = _fallback_list()
    pickle.dump({'date': today, 'stocks': stocks}, open(cache_file, 'wb'))
    return stocks

def _baostock_code(code):
    return f'sh.{code}' if (code.startswith('6') or code.startswith('68')) else f'sz.{code}'

def _fallback_list():
    """CSI300核心成分股 (50只代表)"""
    codes = [
        ('600519','贵州茅台'),('000858','五粮液'),('601318','中国平安'),
        ('600036','招商银行'),('000333','美的集团'),('600276','恒瑞医药'),
        ('002415','海康威视'),('000001','平安银行'),('002594','比亚迪'),
        ('601888','中国中免'),('600030','中信证券'),('000651','格力电器'),
        ('600900','长江电力'),('002475','立讯精密'),('300750','宁德时代'),
        ('601398','工商银行'),('601166','兴业银行'),('600887','伊利股份'),
        ('600809','山西汾酒'),('000568','泸州老窖'),('000002','万科A'),
        ('601012','隆基绿能'),('600585','海螺水泥'),('601668','中国建筑'),
        ('600028','中国石化'),('601857','中国石油'),('601088','中国神华'),
        ('600309','万华化学'),('002352','顺丰控股'),('300059','东方财富'),
        ('002714','牧原股份'),('000725','京东方A'),('688981','中芯国际'),
        ('601899','紫金矿业'),('600690','海尔智家'),('601601','中国太保'),
        ('002304','洋河股份'),('000338','潍柴动力'),('300015','爱尔眼科'),
        ('600031','三一重工'),('601225','陕西煤业'),('002142','宁波银行'),
        ('600048','保利发展'),('601328','交通银行'),('600016','民生银行'),
        ('601939','建设银行'),('601288','农业银行'),('600104','上汽集团'),
        ('002271','东方雨虹'),('600050','中国联通'),
    ]
    return [{'code':c,'name':n} for c,n in codes]

def fetch_online_kline(code, days=150):
    """从baostock在线获取日线数据 (前复权)"""
    try:
        bs_code = _baostock_code(code)
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days+30)).strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date=start_date, end_date=end_date,
            frequency='d', adjustflag='2')
        if rs.error_code != '0': return None
        data = []
        while rs.next(): data.append(rs.get_row_data())
        if len(data) < 30: return None
        df = pd.DataFrame(data, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open','high','low','close','volume','amount','turn','pctChg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.rename(columns={'turn':'turnover','pctChg':'pct_chg'})
        return df.sort_values('date').reset_index(drop=True)
    except:
        return None

def load_data(stocks):
    """在线加载全部股票数据"""
    all_data = {}
    bs.login()
    for i, s in enumerate(stocks):
        code = s['code']
        df = fetch_online_kline(code)
        if df is not None and len(df) >= 60:
            all_data[code] = {'name': s['name'], 'df': df}
        if (i+1) % 10 == 0:
            print(f'  已加载 {i+1}/{len(stocks)}...')
    bs.logout()
    return all_data

def fetch_index_kline():
    """获取CSI300指数行情 (用于市场状态判断)"""
    try:
        bs.login()
        rs = bs.query_history_k_data_plus('sh.000300',
            'date,close', start_date='2023-01-01',
            end_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            frequency='d', adjustflag='2')
        data = []
        while rs.next(): data.append(rs.get_row_data())
        bs.logout()
        if len(data) < 60: return None
        df = pd.DataFrame(data, columns=['date','close'])
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        return df.sort_values('date').reset_index(drop=True)
    except:
        bs.logout()
        return None


# ====== 因子计算 ======
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


# ====== 评分 ======
def score_all(all_data, date_str):
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


# ====== 持仓管理 ======
class LivePortfolio:
    def __init__(self):
        self.positions = {}
        self.trade_log = []
        self.cash_pct = 0.35
        self.load()

    def load(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                data = json.load(open(PORTFOLIO_FILE, 'r', encoding='utf-8'))
                self.positions = data.get('positions', {})
                self.trade_log = data.get('trade_log', [])
            except: pass

    def save(self):
        data = {'positions': self.positions, 'trade_log': self.trade_log[-50:]}
        json.dump(data, open(PORTFOLIO_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)

    def check_exits(self, scored, all_data, date_str, is_bear):
        """检查所有持仓是否需要退出"""
        signals = []
        to_remove = []

        for code, pos in self.positions.items():
            row = scored[scored['code'] == code]
            if len(row) == 0:
                signals.append({'code': code, 'name': pos['name'], 'action': 'SELL',
                    'reason': '数据缺失', 'price': pos['entry_price']})
                to_remove.append(code)
                continue

            r = row.iloc[0]
            close = r['close']; ma20 = r['ma_20']; rank = int(r['rank'])
            atr_pct = r['atr_pct']; atr = r['atr_20']
            pos['days_held'] = pos.get('days_held', 0) + 1
            profit = (close / pos['entry_price'] - 1)
            highest = max(pos.get('highest', pos['entry_price']), close)
            pos['highest'] = highest

            hard_stop = EXIT_CONFIG['stop_loss_bear'] if is_bear else EXIT_CONFIG['stop_loss_hard']
            if atr_pct > 5: atm = EXIT_CONFIG['atr_mult_high']
            elif atr_pct > 2.5: atm = (EXIT_CONFIG['atr_mult_low'] + EXIT_CONFIG['atr_mult_high']) / 2
            else: atm = EXIT_CONFIG['atr_mult_low']

            reason = None
            if profit <= hard_stop: reason = f'硬止损({profit*100:.1f}%)'
            elif close < highest - atm * atr: reason = f'移动止损'
            elif close < ma20: reason = f'跌破MA20'
            elif profit >= EXIT_CONFIG['profit_target']: reason = f'止盈(+{profit*100:.1f}%)'
            elif pos['days_held'] >= EXIT_CONFIG['max_hold_days']: reason = f'持仓到期({pos["days_held"]}天)'
            elif rank > EXIT_CONFIG['rank_exit']: reason = f'排名淘汰(#{rank})'
            elif rank - pos.get('entry_rank', 999) > EXIT_CONFIG['rank_drop_exit']: reason = f'排名急跌'

            if reason:
                pnl_pct = (close / pos['entry_price'] - 1) * 100
                signals.append({'code': code, 'name': pos['name'], 'action': 'SELL',
                    'price': close, 'reason': reason, 'profit_pct': pnl_pct,
                    'days_held': pos['days_held'], 'entry_price': pos['entry_price']})
                self.trade_log.append({
                    'date': date_str, 'action': 'SELL', 'code': code,
                    'name': pos['name'], 'entry_price': pos['entry_price'],
                    'exit_price': close, 'profit_pct': pnl_pct,
                    'reason': reason, 'days_held': pos['days_held']
                })
                to_remove.append(code)

        for code in to_remove:
            del self.positions[code]

        return signals

    def generate_buys(self, scored, max_pos, date_str):
        """生成买入信号"""
        signals = []
        slots = max_pos - len(self.positions)
        if slots <= 0: return signals

        best_score = scored['score'].iloc[0] if len(scored) > 0 else -99
        if best_score < MIN_SCORE: return signals

        held = set(self.positions.keys())
        for _, r in scored.iterrows():
            if slots <= 0: break
            code = r['code']
            if code in held: continue
            if r['score'] < MIN_SCORE: break

            entry_price = r['close']
            self.positions[code] = {
                'name': r['name'], 'entry_price': entry_price,
                'entry_date': date_str, 'days_held': 0,
                'highest': entry_price, 'entry_rank': int(r['rank']),
                'entry_score': float(r['score'])
            }
            self.trade_log.append({
                'date': date_str, 'action': 'BUY', 'code': code,
                'name': r['name'], 'price': entry_price,
                'score': float(r['score']), 'rank': int(r['rank'])
            })
            signals.append({'code': code, 'name': r['name'], 'action': 'BUY',
                'price': entry_price, 'score': float(r['score']), 'rank': int(r['rank']),
                'atr_pct': r['atr_pct']})
            slots -= 1

        return signals


# ====== 报告生成 ======
def generate_report(date_str, market_info, sell_signals, buy_signals, holdings, top10):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    regime, idx_close, ma_val, max_pos, best_score = market_info

    lines = []
    lines.append(f'# 丁家山 V4 每日操盘信号')
    lines.append(f'')
    lines.append(f'**{date_str}** | 市场: {regime} | CSI300={idx_close:.0f} MA120={ma_val:.0f}')
    lines.append(f'仓位上限: {max_pos}只 | 单只{12}% | 现金保留35%')
    lines.append(f'')

    # 卖出信号
    lines.append(f'---')
    lines.append(f'')
    if sell_signals:
        lines.append(f'## 🔴 卖出信号 ({len(sell_signals)}笔)')
        lines.append(f'')
        lines.append(f'| 代码 | 名称 | 卖出价 | 盈亏 | 持仓天 | 原因 |')
        lines.append(f'|------|------|--------|------|--------|------|')
        for s in sell_signals:
            e = '🟢' if s.get('profit_pct', 0) > 0 else '🔴'
            lines.append(f'| {s["code"]} | {s["name"]} | {s["price"]:.2f} | {e} {s.get("profit_pct",0):+.2f}% | {s.get("days_held","-")}天 | {s["reason"]} |')
        lines.append(f'')
    else:
        lines.append(f'## 🔴 卖出信号: 无')
        lines.append(f'')

    # 买入信号
    if buy_signals:
        lines.append(f'## 🟢 买入信号 ({len(buy_signals)}笔)')
        lines.append(f'')
        lines.append(f'| 代码 | 名称 | 买入价 | 评分 | 排名 | 波动率 |')
        lines.append(f'|------|------|--------|------|------|--------|')
        for s in buy_signals:
            atr_flag = '⚠高波' if s.get('atr_pct', 0) > 5 else ''
            lines.append(f'| {s["code"]} | {s["name"]} | {s["price"]:.2f} | {s["score"]:.3f} | #{s["rank"]} | {s.get("atr_pct",0):.1f}%{atr_flag} |')
        lines.append(f'')
    else:
        if best_score < MIN_SCORE:
            lines.append(f'## 🟢 买入信号: 信号不足(最佳{best_score:.3f}<{MIN_SCORE}), 观望')
        else:
            lines.append(f'## 🟢 买入信号: 无 (仓位已满)')
        lines.append(f'')

    # 当前持仓
    lines.append(f'---')
    lines.append(f'')
    if holdings:
        lines.append(f'## 📊 当前持仓 ({len(holdings)}/{max_pos})')
        lines.append(f'')
        lines.append(f'| 代码 | 名称 | 买入价 | 现价 | 浮动盈亏 | 持仓天 | 买入排名 | 操作 |')
        lines.append(f'|------|------|--------|------|----------|--------|----------|------|')
        for h in holdings:
            pnl = h['profit_pct']
            e = '🟢' if pnl > 0 else ('🔴' if pnl < 0 else '⚪')
            days_left = EXIT_CONFIG['max_hold_days'] - h['days_held']
            action = '持有'
            if pnl >= EXIT_CONFIG['profit_target'] * 100: action = '⚠止盈'
            elif pnl <= EXIT_CONFIG['stop_loss_hard'] * 100: action = '🚨止损'
            elif days_left <= 2: action = f'⏰剩{days_left}天'
            lines.append(f'| {h["code"]} | {h["name"]} | {h["entry_price"]:.2f} | {h["current_price"]:.2f} | {e} {pnl:+.2f}% | {h["days_held"]}天 | #{h["entry_rank"]} | {action} |')
        lines.append(f'')
    else:
        lines.append(f'## 📊 当前持仓: 空仓')
        lines.append(f'')

    # Top 10 候选
    lines.append(f'---')
    lines.append(f'')
    lines.append(f'## ⭐ 今日评分 Top 10')
    lines.append(f'')
    lines.append(f'| 排名 | 代码 | 名称 | 评分 | 5日 | 20日 | 量比 | vsMA20 |')
    lines.append(f'|------|------|------|------|------|------|------|--------|')
    for _, r in top10.iterrows():
        lines.append(f'| #{int(r["rank"])} | {r["code"]} | {r["name"]} | {r["score"]:.3f} | {r["ret_5d"]:+.1f}% | {r["ret_20d"]:+.1f}% | {r["vol_ratio"]:.2f} | {r["price_vs_ma20"]:+.1f}% |')
    lines.append(f'')

    lines.append(f'---')
    lines.append(f'*{now} 自动生成 | 丁家山 V4 | 仅供参考*')

    report = '\n'.join(lines)

    # 保存
    filename = f'report_{date_str.replace("-","")}.md'
    filepath = os.path.join(REPORT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    # 最新链接
    latest_path = os.path.join(REPORT_DIR, 'LATEST.md')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


# ====== 主流程 ======
def run_daily():
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    print(f'\n{"="*50}')
    print(f'  丁家山 V4 每日信号 | {date_str} {now.strftime("%H:%M")}')
    print(f'{"="*50}\n')

    # 1. 加载股票池
    stocks = get_csi300_list()
    print(f'[1/5] 股票池: {len(stocks)}只')

    # 2. 在线加载数据
    print(f'[2/5] 在线获取行情数据...')
    all_data = load_data(stocks)
    print(f'      已加载 {len(all_data)} 只股票')

    # 3. 评分
    print(f'[3/5] 全市场评分...')
    scored = score_all(all_data, date_str)
    if len(scored) == 0:
        print('      数据不足, 终止')
        return
    best = scored['score'].iloc[0]
    print(f'      最佳: {scored["code"].iloc[0]} {scored["name"].iloc[0]} score={best:.3f}')

    # 4. 市场状态 (在线获取指数)
    bench_df = fetch_index_kline()
    is_bear = False; is_strong = False
    idx_close = 0; ma_val = 0
    if bench_df is not None:
        bench_ma = bench_df.set_index('date')['close'].rolling(BEAR_MA_PERIOD).mean()
        dt = pd.Timestamp(date_str)
        if dt in bench_ma.index:
            idx_close = bench_df[bench_df['date'] == dt]['close'].iloc[0]
            ma_val = bench_ma.loc[dt]
            is_bear = idx_close < ma_val
            is_strong = idx_close > ma_val * 1.05

    max_pos = POS_BEAR if is_bear else (POS_STRONG if is_strong else POS_NEUTRAL)
    regime = 'BEAR' if is_bear else ('STRONG_BULL' if is_strong else 'NEUTRAL')

    # 5. 持仓管理
    print(f'[4/5] 持仓管理...')
    pf = LivePortfolio()

    # 检查退出
    sell_signals = pf.check_exits(scored, all_data, date_str, is_bear)
    print(f'      卖出: {len(sell_signals)} 笔')

    # 生成买入
    buy_signals = pf.generate_buys(scored, max_pos, date_str)
    print(f'      买入: {len(buy_signals)} 笔')

    # 当前持仓详情
    holdings = []
    for code, pos in pf.positions.items():
        row = scored[scored['code'] == code]
        if len(row) > 0:
            r = row.iloc[0]
            pnl = (r['close'] / pos['entry_price'] - 1) * 100
            holdings.append({
                'code': code, 'name': pos['name'],
                'entry_price': pos['entry_price'],
                'current_price': r['close'],
                'profit_pct': pnl,
                'days_held': pos['days_held'],
                'entry_rank': pos['entry_rank']
            })
    print(f'      持仓: {len(holdings)}/{max_pos} 只')

    pf.save()

    # 6. 生成报告
    print(f'[5/5] 生成报告...')
    market_info = (regime, idx_close, ma_val, max_pos, best)
    top10 = scored.head(10)
    report = generate_report(date_str, market_info, sell_signals, buy_signals, holdings, top10)

    print(f'\n{"="*50}')
    print(f'  报告已生成')
    print(f'  卖出: {len(sell_signals)} | 买入: {len(buy_signals)} | 持仓: {len(holdings)}')
    print(f'{"="*50}\n')
    print(report)

    return report


if __name__ == '__main__':
    try:
        report = run_daily()
        if report:
            print('\n[OK] 信号生成完成')
            # 自动git push
            import subprocess
            subprocess.run(['git', 'add', 'daily_reports/', '.strategy_cache/live_portfolio.json'],
                          cwd=r'C:\AI', capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'daily signal {datetime.now().strftime("%Y-%m-%d")}'],
                          cwd=r'C:\AI', capture_output=True)
            result = subprocess.run(['git', 'push'], cwd=r'C:\AI', capture_output=True)
            if result.returncode == 0:
                print('[OK] 已推送到GitHub')
            else:
                print('[WARN] Git push失败 (检查remote配置)')
    except Exception as e:
        print(f'[ERROR] {e}')

