"""
雷霆点睛 Strategy Backtest
=========================
Classic breakout strategy:
- Buy: Volume burst + price breakout + strong close + MA bullish
- Sell: MA5 breakdown / hard stop / time exit / profit take
"""

import os, struct, pickle
import numpy as np
import pandas as pd

TDX_PATH = r'C:\zd_pazq_hy\vipdoc'
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
COMMISSION = 0.0003
SLIPPAGE = 0.001

# 雷霆点睛 Strategy Parameters
VOL_BURST = 1.8
CLOSE_STRENGTH = 0.70
LOOKBACK_HIGH = 15
STOP_LOSS = -0.05
MAX_HOLD = 8
PROFIT_TAKE = 0.12
MIN_AVG_AMOUNT = 30_000_000

os.makedirs(CACHE_DIR, exist_ok=True)

def parse_day_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        raw = f.read()
    n = len(raw) // 32
    if n == 0:
        return None
    dates, opens, highs, lows, closes, amounts, volumes = [], [], [], [], [], [], []
    for i in range(n):
        offset = i * 32
        date, open_p, high, low, close_p, amount, vol, _ = struct.unpack('IIIIIfII', raw[offset:offset+32])
        dates.append(str(date))
        opens.append(open_p/100.0)
        highs.append(high/100.0)
        lows.append(low/100.0)
        closes.append(close_p/100.0)
        amounts.append(amount)
        volumes.append(vol)
    df = pd.DataFrame({
        'date': pd.to_datetime(dates, format='%Y%m%d'),
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'amount': amounts, 'volume': volumes,
    })
    return df.sort_values('date').reset_index(drop=True)

def get_day_path(code):
    if code.startswith('6') or code.startswith('68'):
        return os.path.join(TDX_PATH, 'sh', 'lday', f'sh{code}.day')
    return os.path.join(TDX_PATH, 'sz', 'lday', f'sz{code}.day')

def load_all_data(stocks):
    all_data = {}
    start_dt = pd.Timestamp(START_DATE)
    for i, s in enumerate(stocks):
        code = s['code']
        df = parse_day_file(get_day_path(code))
        if df is not None:
            df = df[df['date'] >= start_dt].copy()
            if len(df) >= 60:
                all_data[code] = {'name': s['name'], 'df': df}
        if (i+1) % 50 == 0:
            print(f'  Loaded {i+1}/{len(stocks)}...')
    print(f'[DATA] {len(all_data)} stocks loaded')
    return all_data

def get_trading_dates(all_data):
    dates = set()
    for info in all_data.values():
        dates.update(info['df']['date'].dt.strftime('%Y-%m-%d').tolist())
    return sorted([d for d in dates if d >= START_DATE])

def check_leitong_signal(df, date_str):
    """Thunderbolt Eye-Dotting buy conditions"""
    d = df[df['date'] <= date_str]
    if len(d) < 60:
        return None

    close = d['close'].values
    high = d['high'].values
    low = d['low'].values
    open_p = d['open'].values
    volume = d['volume'].values
    n = len(d)

    # 1. Volume burst: vol > 1.8x 20-day avg
    vol_20 = np.mean(volume[-21:-1])
    if vol_20 <= 0 or volume[-1] < VOL_BURST * vol_20:
        return None

    # 2. Strong close: close in top 70% of day range
    day_range = high[-1] - low[-1]
    if day_range <= 0:
        return None
    close_pos = (close[-1] - low[-1]) / day_range
    if close_pos < CLOSE_STRENGTH:
        return None

    # 3. Break above 15-day high
    high_15 = max(high[-LOOKBACK_HIGH-1:-1])
    if close[-1] <= high_15:
        return None

    # 4. MA5 > MA10 (short-term momentum)
    ma5 = np.mean(close[-6:-1])
    ma10 = np.mean(close[-11:-1])
    if ma5 <= ma10:
        return None

    # 5. Positive close
    if close[-1] <= open_p[-1]:
        return None

    # Liquidity filter
    avg_amount = np.mean(d['amount'].values[-21:-1])
    if avg_amount < MIN_AVG_AMOUNT:
        return None

    return {
        'close': close[-1],
        'ma5': ma5,
        'vol_ratio': volume[-1] / vol_20,
        'close_strength': close_pos,
    }

def run_leitong(all_data):
    dates = get_trading_dates(all_data)
    print(f'\n=== 雷霆点睛: {dates[0]} ~ {dates[-1]} ({len(dates)} days) ===\n')

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    daily_values = []
    daily_dates = []
    buys = sells = 0

    bench_path = os.path.join(TDX_PATH, 'sh', 'lday', 'sh000300.day')
    bench_df = parse_day_file(bench_path)

    for idx, date_str in enumerate(dates):
        # Scan buy signals
        signals = []
        for code, info in all_data.items():
            sig = check_leitong_signal(info['df'], date_str)
            if sig is not None:
                sig['code'] = code
                sig['name'] = info['name']
                signals.append(sig)
        signals.sort(key=lambda x: x['vol_ratio'], reverse=True)

        # Check exits
        to_sell = []
        for code, pos in list(positions.items()):
            if code not in all_data:
                continue
            df = all_data[code]['df']
            d = df[df['date'] <= date_str]
            if len(d) < 5:
                continue
            close_now = d['close'].iloc[-1]
            ma5_now = d['close'].iloc[-6:-1].mean() if len(d) >= 6 else close_now
            pos['days_held'] += 1
            profit_pct = (close_now / pos['entry_price'] - 1)

            reason = None
            if profit_pct <= STOP_LOSS:
                reason = 'HardStop'
            elif profit_pct >= PROFIT_TAKE:
                reason = 'ProfitTake'
            elif pos['days_held'] >= MAX_HOLD:
                reason = 'TimeExit'
            elif close_now < ma5_now:
                reason = 'BelowMA5'
            if reason:
                to_sell.append((code, reason, close_now))

        for code, reason, price in to_sell:
            pos = positions[code]
            gross = pos['shares'] * price
            cost = gross * (COMMISSION + SLIPPAGE)
            cash += gross - cost
            buy_cost = pos['shares'] * pos['entry_price'] * (1 + COMMISSION + SLIPPAGE)
            pnl = ((gross - cost) / buy_cost - 1) * 100
            trades.append({
                'date': date_str, 'code': code, 'name': pos['name'],
                'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
                'exit_price': price, 'profit_pct': pnl, 'days_held': pos['days_held'],
                'reason': reason,
            })
            del positions[code]
            sells += 1

        # Buy new
        slots = MAX_POSITIONS - len(positions)
        if slots > 0:
            held = set(positions.keys())
            for sig in signals:
                if slots <= 0:
                    break
                if sig['code'] in held:
                    continue
                total_eq = cash
                for c, p in positions.items():
                    if c in all_data:
                        dd = all_data[c]['df']
                        dd = dd[dd['date'] <= date_str]
                        if len(dd) > 0:
                            total_eq += p['shares'] * dd['close'].iloc[-1]
                target = total_eq / MAX_POSITIONS
                if target > cash:
                    target = cash
                price = sig['close']
                shares = int(target / (price * (1 + COMMISSION + SLIPPAGE)) / 100) * 100
                if shares < 100:
                    continue
                cost = shares * price * (1 + COMMISSION + SLIPPAGE)
                if cost > cash:
                    continue
                cash -= cost
                positions[sig['code']] = {
                    'name': sig['name'], 'entry_price': price,
                    'entry_date': date_str, 'shares': shares, 'days_held': 0,
                }
                buys += 1
                slots -= 1

        # MTM
        total = cash
        for code, pos in positions.items():
            if code in all_data:
                d = all_data[code]['df']
                d = d[d['date'] <= date_str]
                if len(d) > 0:
                    total += pos['shares'] * d['close'].iloc[-1]
        daily_values.append(total)
        daily_dates.append(date_str)

        if (idx+1) % 50 == 0 or idx == 0:
            ret = (total / INITIAL_CAPITAL - 1) * 100
            print(f'  [{date_str}] Day {idx+1}/{len(dates)} | Pos={len(positions)} | Val={total:,.0f} | Ret={ret:+.2f}%')

    # Calc metrics
    eq = pd.Series(daily_values, index=pd.to_datetime(daily_dates))
    rets = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    ann_ret = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (252/len(rets)) - 1) * 100
    rf = 0.03
    exc = rets - rf/252
    sharpe = np.sqrt(252) * exc.mean() / exc.std() if exc.std() > 0 else 0
    cummax = eq.cummax()
    max_dd = ((eq - cummax) / cummax * 100).min()

    sell_trades = [t for t in trades]
    wins = [t for t in sell_trades if t['profit_pct'] > 0]
    losses = [t for t in sell_trades if t['profit_pct'] <= 0]
    wr = len(wins)/len(sell_trades)*100 if sell_trades else 0
    avg_w = np.mean([t['profit_pct'] for t in wins]) if wins else 0
    avg_l = np.mean([t['profit_pct'] for t in losses]) if losses else 0

    bench_ret = None
    if bench_df is not None:
        bench_df = bench_df[bench_df['date'] >= START_DATE]
        if len(bench_df) > 0:
            bench_ret = (bench_df['close'].iloc[-1] / bench_df['close'].iloc[0] - 1) * 100

    print(f'\n{"="*60}')
    print(f'  雷霆点睛 RESULTS')
    print(f'{"="*60}')
    print(f'  Total Return:   {total_ret:+.2f}%')
    print(f'  Annual Return:  {ann_ret:+.2f}%')
    print(f'  Sharpe Ratio:   {sharpe:.2f}')
    print(f'  Max Drawdown:   {max_dd:.2f}%')
    print(f'  Volatility:     {rets.std()*np.sqrt(252)*100:.2f}%')
    print(f'  Trades:         {len(sell_trades)} (Buy signals: {buys})')
    print(f'  Win Rate:       {wr:.1f}%')
    print(f'  Avg Win:        {avg_w:+.2f}%')
    print(f'  Avg Loss:       {avg_l:+.2f}%')
    if avg_l != 0:
        print(f'  Profit Factor:  {abs(avg_w/avg_l):.2f}')
    if sell_trades:
        print(f'  Avg Hold Days:  {np.mean([t["days_held"] for t in sell_trades]):.1f}')
    if bench_ret is not None:
        print(f'  CSI 300 Index:  {bench_ret:+.2f}%')
        print(f'  Alpha:          {total_ret - bench_ret:+.2f}%')
    print(f'{"="*60}')

    # Yearly
    yearly = eq.resample('YE').last()
    prev = INITIAL_CAPITAL
    print(f'\n  年度收益:')
    for yr_dt, val in yearly.items():
        yr = yr_dt.year
        ret = (val / prev - 1) * 100
        print(f'    {yr}: {ret:+.2f}%')
        prev = val

    result = {
        'strategy': '雷霆点睛',
        'dates': daily_dates, 'equity': daily_values, 'trades': trades,
        'summary': {
            'total_return': total_ret, 'annual_return': ann_ret,
            'sharpe': sharpe, 'max_drawdown': max_dd,
            'win_rate': wr, 'avg_win': avg_w, 'avg_loss': avg_l,
            'total_trades': len(sell_trades),
            'benchmark_return': bench_ret,
            'alpha': (total_ret - bench_ret) if bench_ret else None,
        }
    }
    pickle.dump(result, open(os.path.join(CACHE_DIR, 'leitong_result.pkl'), 'wb'))
    return result

if __name__ == '__main__':
    cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
    stocks = pickle.load(open(cache_file, 'rb'))['stocks']
    print(f'[LOAD] CSI 300: {len(stocks)} stocks')
    all_data = load_all_data(stocks)
    result = run_leitong(all_data)
