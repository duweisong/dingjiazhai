"""
Dingjiashan V9 Tapered - Inspired by "Tapered Language Models"
======================================================================
Paper: Tapered Language Models (arxiv.org/abs/2606.23670)
Core insight: Model layers are not equally important - early layers do creative
              understanding, later layers mostly echo/re-confirm earlier decisions.
              Shifting capacity from later to earlier layers (tapered distribution),
              zero extra cost, improves perplexity by 1.84 points.

Quant trading analogy:
  - Early layers = trend/momentum factors -> core driver, most weight
  - Mid layers   = volume/volatility factors -> auxiliary validation
  - Late layers  = confirmation/quality factors -> light confirmation (echo)

V9 Three Tapered Optimizations:
  1. Factor weights cosine-tapered - front factors get 1.5x weight of rear factors
  2. Position signal cosine decay - signal strength decays over holding period
  3. Score distribution sigmoid compression - amplify head, compress tail

Compared to V4 baseline:
  - Same number of factors (parameters unchanged)
  - Same computation cost
  - Pure redistribution gain
"""

import os, pickle, json, math
import numpy as np
import pandas as pd
import baostock as bs
from datetime import datetime
from collections import Counter

# ====== V9: Tapered Factor Weights (Cosine Decay Pattern) ======
# Factor tiers:
#   Tier-1 (front/core): ret_20d, price_vs_ma20, ma_alignment -> trend/momentum
#   Tier-2 (mid/auxiliary): ret_5d, ret_10d, vol_ratio -> volume/price integration
#   Tier-3 (rear/confirm): up_days_10, turnover, atr_pct -> quality confirmation
#
# Weight ratio: Front(0.50) : Mid(0.34) : Rear(0.16) = 3:2:1 tapered

FACTOR_WEIGHTS_TAPERED = {
    # Tier-1: Front-end core factors
    'ret_20d':        0.18,
    'price_vs_ma20':  0.18,
    'ma_alignment':   0.14,
    # Tier-2: Mid auxiliary factors
    'ret_5d':         0.12,
    'ret_10d':        0.10,
    'vol_ratio':      0.12,
    # Tier-3: Rear confirmation factors
    'up_days_10':     0.06,
    'turnover':       0.05,
    'atr_pct':       -0.05,
}

# ====== V9: Signal Cosine Decay Config ======
SIGNAL_DECAY_ENABLED = True
DECAY_HALF_LIFE = 5       # Day 5: signal strength decays to 50%
DECAY_COSINE_POWER = 0.5

# ====== V9: Score Sigmoid Compression ======
SCORE_SIGMOID = True
SIGMOID_CENTER = 0.0
SIGMOID_STEEPNESS = 3.0   # Higher = more head differentiation

# ====== V4 Baseline Config (preserved) ======
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
SLIPPAGE = 0.001
BEAR_MA_PERIOD = 120
MIN_AVG_AMOUNT = 50_000_000
MIN_SCORE = 0.3

EXIT_CONFIG = {
    'atr_mult_low': 1.2, 'atr_mult_high': 2.0,
    'max_hold_days': 10, 'profit_target': 0.12,
    'ma_period': 20, 'rank_exit': 80, 'rank_drop_exit': 50,
    'stop_loss_hard': -0.05, 'stop_loss_bear': -0.03,
}

POS_STRONG = 5; POS_NEUTRAL = 4; POS_BEAR = 1
SINGLE_PCT = 0.12; SINGLE_BEAR = 0.08; CASH_RESERVE = 0.35

PORTFOLIO_FILE = os.path.join(CACHE_DIR, 'live_portfolio_v9.json')
REPORT_DIR = r'C:\AI\daily_reports'
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ====== Cosine Taper Functions ======
def cosine_taper(t, total, ratio=1.5):
    """Cosine decay curve: from ratio*x down to x/ratio
    Mirrors the paper's cosine decay schedule for FFN width.
    """
    progress = t / total
    cos_val = math.cos(progress * math.pi)
    return 1.0 + (ratio - 1.0) * (1.0 + cos_val) / 2.0


def signal_cosine_decay(days_held, half_life=5):
    """Position signal cosine decay.
    Day 0: weight=1.0, Day 5: weight=0.5, Day 10: weight->0
    """
    if days_held <= 0:
        return 1.0
    t = days_held / half_life
    if t >= 2.0:
        return 0.01
    return max(0.01, math.cos(math.pi * t / 4))


def sigmoid_score(x, center=0.0, steepness=3.0):
    """Sigmoid score compression: amplify head differences, compress tail.
    Like the wedge shape in TLM: wider at front, narrower at back.
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))



# ====== Data Loading (same as V4) ======
CSI300_STOCKS = None

def get_csi300_list():
    cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
    today = pd.Timestamp.now().strftime('%Y%m%d')
    if os.path.exists(cache_file):
        cached = pickle.load(open(cache_file, 'rb'))
        if cached.get('date') == today:
            return cached['stocks']
    stocks = _fallback_list()
    pickle.dump({'date': today, 'stocks': stocks}, open(cache_file, 'wb'))
    return stocks

def _baostock_code(code):
    return f'sh.{code}' if (code.startswith('6') or code.startswith('68')) else f'sz.{code}'

def _fallback_list():
    """CSI300 core constituents (50 representative stocks)"""
    codes = [
        ('600519','Kweichow Moutai'),('000858','Wuliangye'),('601318','Ping An'),
        ('600036','CMB'),('000333','Midea'),('600276','Hengrui'),
        ('002415','Hikvision'),('000001','Ping An Bank'),('002594','BYD'),
        ('601888','CTF'),('600030','CITIC Sec'),('000651','Gree'),
        ('600900','Yangtze Power'),('002475','Luxshare'),('300750','CATL'),
        ('601398','ICBC'),('601166','CIB'),('600887','Yili'),
        ('600809','Shanxi Fenjiu'),('000568','Luzhou Laojiao'),('000002','Vanke'),
        ('601012','Longi'),('600585','Conch'),('601668','CSCEC'),
        ('600028','Sinopec'),('601857','PetroChina'),('601088','China Shenhua'),
        ('600309','Wanhua'),('002352','SF Express'),('300059','East Money'),
        ('002714','Muyuan'),('000725','BOE'),('688981','SMIC'),
        ('601899','Zijin Mining'),('600690','Haier'),('601601','CPIC'),
        ('002304','Yanghe'),('000338','Weichai'),('300015','Aier Eye'),
        ('600031','SANY'),('601225','Shaanxi Coal'),('002142','Ningbo Bank'),
        ('600048','Poly'),('601328','BoCom'),('600016','Minsheng Bank'),
        ('601939','CCB'),('601288','ABC'),('600104','SAIC'),
        ('002271','Oriental Yuhong'),('600050','China Unicom'),
    ]
    return [{'code':c,'name':n} for c,n in codes]

def fetch_online_kline(code, days=150):
    try:
        bs_code = _baostock_code(code)
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days+30)).strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date=start_date, end_date=end_date,
            frequency='d', adjustflag='2')
        if rs.error_code != '0':
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if len(data) < 30:
            return None
        df = pd.DataFrame(data, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open','high','low','close','volume','amount','turn','pctChg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.rename(columns={'turn':'turnover','pctChg':'pct_chg'})
        return df.sort_values('date').reset_index(drop=True)
    except:
        return None

def load_data(stocks):
    all_data = {}
    bench_df = None
    bs.login()
    try:
        end_d = pd.Timestamp.now().strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus('sh.000300', 'date,close',
            start_date='2023-01-01', end_date=end_d, frequency='d', adjustflag='2')
        if rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if len(data_list) >= 60:
                bench_df = pd.DataFrame(data_list, columns=['date','close'])
                bench_df['date'] = pd.to_datetime(bench_df['date'])
                bench_df['close'] = pd.to_numeric(bench_df['close'], errors='coerce')
    except:
        pass

    for i, s in enumerate(stocks):
        code = s['code']
        df = fetch_online_kline(code)
        if df is not None and len(df) >= 60:
            all_data[code] = {'name': s['name'], 'df': df}
        if (i+1) % 10 == 0:
            print(f'  Loaded {i+1}/{len(stocks)}...')
    bs.logout()
    return all_data, bench_df



# ====== Factor Computation (same as V4) ======
def compute_factors(df, date_str):
    d = df[df['date'] <= date_str]
    if len(d) < 60:
        return None
    close = d['close'].values
    high = d['high'].values
    low = d['low'].values
    volume = d['volume'].values
    n = len(d)

    ret_5d = (close[-1]/close[-6]-1)*100 if n >= 6 else 0
    ret_10d = (close[-1]/close[-11]-1)*100 if n >= 11 else 0
    ret_20d = (close[-1]/close[-21]-1)*100 if n >= 21 else 0

    vol_20 = np.mean(volume[-21:-1]) if n >= 21 and np.mean(volume[-21:-1]) > 0 else 1
    vol_ratio = volume[-1]/vol_20 if vol_20 > 0 else 1

    ma_5 = np.mean(close[-6:-1])
    ma_10 = np.mean(close[-11:-1]) if n >= 11 else ma_5
    ma_20 = np.mean(close[-21:-1]) if n >= 21 else ma_10
    ma_60 = np.mean(close[-61:-1]) if n >= 61 else ma_20

    price_vs_ma20 = (close[-1]/ma_20-1)*100 if ma_20 > 0 else 0
    ma_alignment = 1.0 if (ma_5 > ma_10 > ma_20 > ma_60) else (0.5 if ma_5 > ma_20 else 0.0)

    tr_list = []
    for i in range(max(1, n-20), n):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr_list.append(tr)
    atr_20 = np.mean(tr_list) if tr_list else close[-1]*0.02
    atr_pct = (atr_20/close[-1])*100 if close[-1] > 0 else 2.0

    up_days_10 = sum(1 for i in range(max(1, n-10), n) if close[i] > close[i-1])
    avg_amount = np.mean(d['amount'].values[-21:-1]) if n >= 21 else 0
    turnover = d['turnover'].values[-1] if 'turnover' in d.columns else 0.0

    return {
        'ret_5d': ret_5d, 'ret_10d': ret_10d, 'ret_20d': ret_20d,
        'vol_ratio': vol_ratio, 'price_vs_ma20': price_vs_ma20,
        'ma_alignment': ma_alignment, 'atr_pct': atr_pct,
        'up_days_10': up_days_10, 'turnover': turnover,
        'close': close[-1], 'atr_20': atr_20, 'ma_20': ma_20,
        'avg_amount': avg_amount
    }


# ====== V9 Tapered Scoring ======
def score_all_tapered(all_data, date_str):
    """V9 Tapered scoring:
    1. Cosine-tapered factor weights (front > mid > rear)
    2. Sigmoid compression (amplify head differences, compress tail)
    """
    fl = []
    for code, info in all_data.items():
        f = compute_factors(info['df'], date_str)
        if f is None:
            continue
        f['code'] = code
        f['name'] = info['name']
        fl.append(f)

    if len(fl) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(fl)
    df = df[df['avg_amount'] >= MIN_AVG_AMOUNT].copy()
    if len(df) == 0:
        return pd.DataFrame()

    sc = {}
    for factor, weight in FACTOR_WEIGHTS_TAPERED.items():
        if factor not in df.columns:
            continue
        col = df[factor].copy()
        m = col.mean()
        s = col.std()
        z = pd.Series(0.0, index=col.index) if (s == 0 or pd.isna(s)) else ((col - m) / s).clip(-3, 3)
        sc[factor] = z * weight

    df['score'] = sum(sc.values())

    # V9: Sigmoid compression - amplify head, compress tail (wedge shape)
    if SCORE_SIGMOID:
        score_mean = df['score'].mean()
        score_std = df['score'].std()
        if score_std > 0:
            df['score_raw'] = df['score'].copy()
            z_scaled = (df['score'] - score_mean) / score_std
            df['score'] = sigmoid_score(z_scaled, SIGMOID_CENTER, SIGMOID_STEEPNESS)

    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df



# ====== V9 Portfolio Manager (with signal decay) ======
class LivePortfolioV9:
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
            except:
                pass

    def save(self):
        data = {'positions': self.positions, 'trade_log': self.trade_log[-50:]}
        json.dump(data, open(PORTFOLIO_FILE, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2, default=str)

    def check_exits(self, scored, all_data, date_str, is_bear):
        signals = []
        to_remove = []
        for code, pos in self.positions.items():
            row = scored[scored['code'] == code]
            if len(row) == 0:
                signals.append({'code': code, 'name': pos['name'], 'action': 'SELL',
                    'reason': 'data missing', 'price': pos['entry_price']})
                to_remove.append(code)
                continue
            r = row.iloc[0]
            close = r['close']
            ma20 = r['ma_20']
            rank = int(r['rank'])
            atr_pct = r['atr_pct']
            atr = r['atr_20']
            pos['days_held'] = pos.get('days_held', 0) + 1
            profit = (close / pos['entry_price'] - 1)
            highest = max(pos.get('highest', pos['entry_price']), close)
            pos['highest'] = highest
            hard_stop = EXIT_CONFIG['stop_loss_bear'] if is_bear else EXIT_CONFIG['stop_loss_hard']
            if atr_pct > 5:
                atm = EXIT_CONFIG['atr_mult_high']
            elif atr_pct > 2.5:
                atm = (EXIT_CONFIG['atr_mult_low'] + EXIT_CONFIG['atr_mult_high']) / 2
            else:
                atm = EXIT_CONFIG['atr_mult_low']
            # V9: Signal-decayed rank exit threshold
            days = pos['days_held']
            decay = signal_cosine_decay(days, DECAY_HALF_LIFE)
            adjusted_rank_exit = int(EXIT_CONFIG['rank_exit'] / max(decay, 0.3))
            reason = None
            if profit <= hard_stop:
                reason = 'hard stop({:.1f}%)'.format(profit*100)
            elif close < highest - atm * atr:
                reason = 'trailing stop'
            elif close < ma20:
                reason = 'below MA20'
            elif profit >= EXIT_CONFIG['profit_target']:
                reason = 'TP(+{:.1f}%)'.format(profit*100)
            elif pos['days_held'] >= EXIT_CONFIG['max_hold_days']:
                reason = 'expired({}d)'.format(pos['days_held'])
            elif rank > adjusted_rank_exit:
                reason = 'rank exit(#{}, decay {:.2f})'.format(rank, decay)
            elif rank - pos.get('entry_rank', 999) > EXIT_CONFIG['rank_drop_exit']:
                reason = 'rank drop'
            if reason:
                pnl_pct = (close / pos['entry_price'] - 1) * 100
                signals.append({'code': code, 'name': pos['name'], 'action': 'SELL',
                    'price': close, 'reason': reason, 'profit_pct': pnl_pct,
                    'days_held': pos['days_held'], 'entry_price': pos['entry_price']})
                self.trade_log.append({'date': date_str, 'action': 'SELL', 'code': code,
                    'name': pos['name'], 'entry_price': pos['entry_price'],
                    'exit_price': close, 'profit_pct': pnl_pct,
                    'reason': reason, 'days_held': pos['days_held']})
                to_remove.append(code)
        for code in to_remove:
            del self.positions[code]
        return signals

    def generate_buys(self, scored, max_pos, date_str):
        signals = []
        slots = max_pos - len(self.positions)
        if slots <= 0:
            return signals
        best_score = scored['score'].iloc[0] if len(scored) > 0 else -99
        if best_score < MIN_SCORE:
            return signals
        held = set(self.positions.keys())
        for _, r in scored.iterrows():
            if slots <= 0:
                break
            code = r['code']
            if code in held:
                continue
            if r['score'] < MIN_SCORE:
                break
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
            signals.append({
                'code': code, 'name': r['name'], 'action': 'BUY',
                'price': entry_price, 'score': float(r['score']),
                'rank': int(r['rank'])
            })
            slots -= 1
        return signals


# ====== Market State ======
def check_market(bench_df, date_str):
    d = bench_df[bench_df['date'] <= date_str]
    if len(d) < BEAR_MA_PERIOD:
        return 'NEUTRAL'
    ma = np.mean(d['close'].values[-BEAR_MA_PERIOD:])
    close = d['close'].values[-1]
    return 'BEAR' if close < ma else ('STRONG' if close > ma*1.05 else 'NEUTRAL')


# ====== Main ======
def main():
    print('='*60)
    print('  Dingjiashan V9 Tapered - TLM-inspired Signal Optimization')
    print('  Paper: Tapered Language Models (arxiv.org/abs/2606.23670)')
    print('='*60)

    stocks = get_csi300_list()
    print('[STOCKS] {} stocks in universe'.format(len(stocks)))

    print('[DATA] Loading from baostock...')
    all_data, bench_df = load_data(stocks)
    print('[DATA] {} stocks loaded'.format(len(all_data)))

    if bench_df is None:
        print('[WARN] No benchmark data')
        return

    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    ms = check_market(bench_df, today)
    max_pos = POS_STRONG if ms == 'STRONG' else (POS_BEAR if ms == 'BEAR' else POS_NEUTRAL)
    print('[MARKET] {} | Max positions: {}'.format(ms, max_pos))

    print('[SCORE] Computing tapered scores...')
    scored = score_all_tapered(all_data, today)
    print('[SCORE] {} candidates scored'.format(len(scored)))
    if len(scored) > 0:
        print('  Top 5:')
        for i, (_, r) in enumerate(scored.head(5).iterrows()):
            print('    #{} {} {} score={:.4f}'.format(i+1, r['code'], r['name'], r['score']))

    pf = LivePortfolioV9()
    print('[PORTFOLIO] {} current positions'.format(len(pf.positions)))

    # Check exits
    sell_signals = pf.check_exits(scored, all_data, today, ms == 'BEAR')
    for s in sell_signals:
        print('  SELL {} {} @ {:.2f} ({:+.1f}%) [{}]'.format(
            s['code'], s['name'], s['price'], s['profit_pct'], s['reason']))

    # Generate buys
    buy_signals = pf.generate_buys(scored, max_pos, today)
    for s in buy_signals:
        print('  BUY  {} {} @ {:.2f} score={:.4f} rank=#{}'.format(
            s['code'], s['name'], s['price'], s['score'], s['rank']))

    pf.save()

    # Generate report
    front_sum = sum(v for k,v in FACTOR_WEIGHTS_TAPERED.items() if k in ['ret_20d','price_vs_ma20','ma_alignment'])
    mid_sum = sum(v for k,v in FACTOR_WEIGHTS_TAPERED.items() if k in ['ret_5d','ret_10d','vol_ratio'])
    rear_sum = sum(v for k,v in FACTOR_WEIGHTS_TAPERED.items() if k in ['up_days_10','turnover','atr_pct'])

    report = []
    report.append('# Dingjiashan V9 Tapered Daily Signal - {}'.format(today))
    report.append('')
    report.append('> **Market**: {} | **Max Pos**: {}'.format(ms, max_pos))
    report.append('> **Optimizations**: Tapered Factor Weights + Cosine Signal Decay + Sigmoid Compression')
    report.append('> **Paper**: [Tapered Language Models](https://arxiv.org/abs/2606.23670)')
    report.append('')
    report.append('## V9 Tapered Architecture')
    report.append('')
    report.append('| Tier | Factors | Weight Sum | Role |')
    report.append('|------|---------|------------|------|')
    report.append('| Front (core) | ret_20d, price_vs_ma20, ma_alignment | {:.0%} | Trend/Momentum |'.format(front_sum))
    report.append('| Mid (aux) | ret_5d, ret_10d, vol_ratio | {:.0%} | Volume/Price |'.format(mid_sum))
    report.append('| Rear (confirm) | up_days_10, turnover, atr_pct | {:.0%} | Quality/Confirm |'.format(rear_sum))
    report.append('')

    if buy_signals:
        report.append('## BUY Signals')
        report.append('| Code | Name | Price | Score | Rank |')
        report.append('|------|------|-------|-------|------|')
        for s in buy_signals:
            report.append('| {} | {} | {:.2f} | {:.4f} | #{} |'.format(s['code'], s['name'], s['price'], s['score'], s['rank']))

    if sell_signals:
        report.append('')
        report.append('## SELL Signals')
        report.append('| Code | Name | Price | Return | Reason |')
        report.append('|------|------|-------|--------|--------|')
        for s in sell_signals:
            report.append('| {} | {} | {:.2f} | {:+.1f}% | {} |'.format(s['code'], s['name'], s['price'], s['profit_pct'], s['reason']))

    if pf.positions:
        report.append('')
        report.append('## Current Positions')
        report.append('| Code | Name | Entry | Days | Decay | Score |')
        report.append('|------|------|-------|------|-------|-------|')
        for code, pos in pf.positions.items():
            days = pos.get('days_held', 0)
            decay = signal_cosine_decay(days, DECAY_HALF_LIFE) if SIGNAL_DECAY_ENABLED else 1.0
            report.append('| {} | {} | {:.2f} | {}d | {:.2f} | {:.4f} |'.format(code, pos['name'], pos['entry_price'], days, decay, pos.get('entry_score',0)))

    if not buy_signals and not sell_signals:
        report.append('## No Signals Today')

    report.append('')
    report.append('---')
    report.append('*V9 Tapered - Let weights flow where they matter most*')

    report_path = os.path.join(REPORT_DIR, 'report_v9_{}.md'.format(today))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print('')
    print('[REPORT] Saved to {}'.format(report_path))

    # Print V9 summary
    print('')
    print('='*60)
    print('  V9 Tapered Optimization Summary:')
    print('  Factor Weights: Front={:.0%} / Mid={:.0%} / Rear={:.0%}'.format(front_sum, mid_sum, rear_sum))
    print('  Signal Decay: half-life={} days'.format(DECAY_HALF_LIFE))
    print('  Sigmoid Compression: center={}, steepness={}'.format(SIGMOID_CENTER, SIGMOID_STEEPNESS))
    print('  Zero extra cost: same factors & compute, pure redistribution')
    print('='*60)

    return report_path


if __name__ == '__main__':
    main()
