"""
西湖宾馆 V4 - 极简有效风控
===========================
原名: 西湖宾馆 V4
V1问题: 2023熊市-29.92%
V2/V3教训: 过度风控适得其反
V4方案: 仅做仓位管理 + 宽止损,不暂停不冷却
  1. 市场状态仓位: 熊市缩至2只40%仓位 = 正常16%敞口
  2. 宽移动止损: 2.5x ATR (保护大赢家)
  3. 适度止损: -6%
  4. 原版信号的盈亏比(2.10)足以覆盖亏损
"""

import os, struct, pickle
import numpy as np
import pandas as pd

TDX_PATH = r'C:\zd_pazq_hy\vipdoc'
CACHE_DIR = r'C:\AI\.strategy_cache'
START_DATE = '2023-01-01'
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
MAX_POSITIONS_BEAR = 2
BEAR_MA_PERIOD = 120
BEAR_SIZE_MULT = 0.4       # 熊市单只仓位 = 正常的40%
BULL_SIZE_MULT = 1.0
COMMISSION = 0.0003
SLIPPAGE = 0.001

# 核心信号 (保持原版参数)
VOL_BURST = 1.8
CLOSE_STRENGTH = 0.70
LOOKBACK_HIGH = 15

# 极简风控参数
STOP_LOSS = -0.06          # -6% 硬止损
PROFIT_TAKE = 0.15         # +15% 止盈
MAX_HOLD = 10              # 10天到期
ATR_MULT = 2.5             # 宽移动止损
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
        dates.append(str(date)); opens.append(open_p/100.0); highs.append(high/100.0)
        lows.append(low/100.0); closes.append(close_p/100.0)
        amounts.append(amount); volumes.append(vol)
    df = pd.DataFrame({
        'date': pd.to_datetime(dates, format='%Y%m%d'),
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'amount': amounts, 'volume': volumes,
    })
    return df.sort_values('date').reset_index(drop=True)

def get_day_path(code):
    pfx = 'sh' if (code.startswith('6') or code.startswith('68')) else 'sz'
    return os.path.join(TDX_PATH, pfx, 'lday', f'{pfx}{code}.day')

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
    """西湖宾馆核心信号: 放量突破+强势收盘"""
    d = df[df['date'] <= date_str]
    if len(d) < 60:
        return None

    close = d['close'].values; high = d['high'].values
    low = d['low'].values; open_p = d['open'].values
    volume = d['volume'].values; n = len(d)

    # 量能爆发
    vol_20 = np.mean(volume[-21:-1])
    if vol_20 <= 0 or volume[-1] < VOL_BURST * vol_20:
        return None
    # 强势收盘
    day_range = high[-1] - low[-1]
    if day_range <= 0:
        return None
    close_pos = (close[-1] - low[-1]) / day_range
    if close_pos < CLOSE_STRENGTH:
        return None
    # 突破前高
    high_lb = max(high[-LOOKBACK_HIGH-1:-1])
    if close[-1] <= high_lb:
        return None
    # 短期动量
    ma5 = np.mean(close[-6:-1])
    ma10 = np.mean(close[-11:-1])
    if ma5 <= ma10:
        return None
    # 阳线
    if close[-1] <= open_p[-1]:
        return None
    # 流动性
    avg_amount = np.mean(d['amount'].values[-21:-1])
    if avg_amount < MIN_AVG_AMOUNT:
        return None

    # ATR for trailing stop
    tr_list = []
    for i in range(max(1, n-20), n):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else close[-1]*0.02

    return {
        'close': close[-1], 'ma5': ma5, 'ma10': ma10,
        'vol_ratio': volume[-1]/vol_20, 'close_strength': close_pos,
        'atr': atr, 'atr_pct': (atr/close[-1])*100,
    }

class Portfolio:
    def __init__(self):
        self.cash = INITIAL_CAPITAL
        self.positions = {}
        self.trades = []
        self.equity = []
        self.dates = []
        self.is_bear = False
        self.bear_days = 0
        self.bull_days = 0

    def max_pos(self):
        return MAX_POSITIONS_BEAR if self.is_bear else MAX_POSITIONS

    def size_mult(self):
        return BEAR_SIZE_MULT if self.is_bear else BULL_SIZE_MULT

    def check_exit(self, pos, close, ma5, ma10, atr, days_held):
        """V4: 宽止损+宽移动止损+MA5+止盈+时间"""
        profit = (close / pos['entry_price'] - 1)
        highest = pos.get('highest', pos['entry_price'])
        highest = max(highest, close)
        pos['highest'] = highest
        atr_val = pos.get('entry_atr', pos['entry_price']*0.02)

        if profit <= STOP_LOSS:
            return True, 'HardStop'
        if close < highest - ATR_MULT * atr_val:
            return True, 'TrailStop'
        if close < ma5 and profit < -0.01:
            return True, 'BelowMA5'
        if profit >= PROFIT_TAKE:
            return True, 'ProfitTake'
        if days_held >= MAX_HOLD:
            return True, 'TimeExit'
        return False, ''

    def sell(self, code, price, reason, date_str):
        pos = self.positions[code]
        gross = pos['shares'] * price
        cost = gross * (COMMISSION + SLIPPAGE)
        self.cash += gross - cost
        buy_cost = pos['shares'] * pos['entry_price'] * (1 + COMMISSION + SLIPPAGE)
        pnl = ((gross - cost) / buy_cost - 1) * 100
        self.trades.append({
            'date': date_str, 'code': code, 'name': pos['name'],
            'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
            'exit_price': price, 'profit_pct': pnl, 'days_held': pos['days_held'],
            'reason': reason, 'regime': 'BEAR' if self.is_bear else 'BULL',
        })
        del self.positions[code]

    def buy(self, code, name, price, vol_ratio, atr, atr_pct, date_str):
        """V4: 极简仓位管理 - 仅按市场状态调节"""
        total_eq = self.cash
        for c, p in self.positions.items():
            total_eq += p['shares'] * price
        max_p = self.max_pos()
        base_target = (total_eq / max(max_p, 1)) * self.size_mult()

        # 波动率微调
        if atr_pct > 5:
            base_target *= 0.75
        elif atr_pct > 3:
            base_target *= 0.9

        if base_target > self.cash:
            base_target = self.cash
        shares = int(base_target / (price * (1 + COMMISSION + SLIPPAGE)) / 100) * 100
        if shares < 100:
            return False
        cost = shares * price * (1 + COMMISSION + SLIPPAGE)
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[code] = {
            'name': name, 'entry_price': price, 'entry_date': date_str,
            'shares': shares, 'days_held': 0, 'highest': price,
            'entry_atr': atr, 'vol_ratio': vol_ratio,
        }
        return True

    def mark_day(self, date_str, all_data):
        total = self.cash
        for code, pos in self.positions.items():
            if code in all_data:
                d = all_data[code]['df']; d = d[d['date'] <= date_str]
                if len(d) > 0:
                    total += pos['shares'] * d['close'].iloc[-1]
        self.equity.append(total)
        self.dates.append(date_str)
        return total

def run_backtest(all_data):
    dates = get_trading_dates(all_data)
    print(f'\n=== 西湖宾馆 V2 (风控增强): {dates[0]} ~ {dates[-1]} ({len(dates)} days) ===\n')

    # 加载指数用于市场状态判断
    bench_path = os.path.join(TDX_PATH, 'sh', 'lday', 'sh000300.day')
    bench_df = parse_day_file(bench_path)
    bench_ma = None
    if bench_df is not None:
        bench_df = bench_df[bench_df['date'] >= START_DATE].copy()
        bench_ma = bench_df.set_index('date')['close'].rolling(BEAR_MA_PERIOD).mean()

    pf = Portfolio()
    buys = sells = 0

    for idx, date_str in enumerate(dates):
        # ---- 市场状态检测 ----
        if bench_ma is not None:
            dt = pd.Timestamp(date_str)
            if dt in bench_ma.index:
                ma_val = bench_ma.loc[dt]
                idx_close = bench_df[bench_df['date'] == dt]['close']
                if len(idx_close) > 0:
                    pf.is_bear = idx_close.iloc[0] < ma_val
                    if pf.is_bear: pf.bear_days += 1
                    else: pf.bull_days += 1

        # 扫描买入信号
        signals = []
        for code, info in all_data.items():
            sig = check_leitong_signal(info['df'], date_str)
            if sig is not None:
                sig['code'] = code; sig['name'] = info['name']
                signals.append(sig)
        signals.sort(key=lambda x: x['vol_ratio'], reverse=True)

        # 检查退出
        to_sell = []
        for code, pos in list(pf.positions.items()):
            if code not in all_data:
                continue
            df = all_data[code]['df']; d = df[df['date'] <= date_str]
            if len(d) < 5:
                continue
            close_now = d['close'].iloc[-1]
            ma5_now = d['close'].iloc[-6:-1].mean() if len(d) >= 6 else close_now
            ma10_now = d['close'].iloc[-11:-1].mean() if len(d) >= 11 else ma5_now
            pos['days_held'] += 1
            do_exit, reason = pf.check_exit(pos, close_now, ma5_now, ma10_now, None, pos['days_held'])
            if do_exit:
                to_sell.append((code, reason, close_now))

        for code, reason, price in to_sell:
            pf.sell(code, price, reason, date_str)
            sells += 1

        # 买入 (V4: 市场状态仓位管理)
        slots = pf.max_pos() - len(pf.positions)
        if slots > 0:
            held = set(pf.positions.keys())
            for sig in signals:
                if slots <= 0:
                    break
                if sig['code'] in held:
                    continue
                ok = pf.buy(sig['code'], sig['name'], sig['close'],
                           sig['vol_ratio'], sig['atr'], sig['atr_pct'], date_str)
                if ok:
                    buys += 1; slots -= 1

        total = pf.mark_day(date_str, all_data)

        if (idx+1) % 50 == 0 or idx == 0:
            ret = (total / INITIAL_CAPITAL - 1) * 100
            regime = 'BEAR' if pf.is_bear else 'BULL'
            print(f'  [{date_str}] Day {idx+1}/{len(dates)} | {regime} | '
                  f'Pos={len(pf.positions)}/{pf.max_pos()} | Val={total:,.0f} | Ret={ret:+.2f}%')

    # ---- 结果计算 ----
    eq = pd.Series(pf.equity, index=pd.to_datetime(pf.dates))
    rets = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    ann_ret = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (252/len(rets)) - 1) * 100
    rf = 0.03; exc = rets - rf/252
    sharpe = np.sqrt(252) * exc.mean() / exc.std() if exc.std() > 0 else 0
    cummax = eq.cummax()
    max_dd = ((eq - cummax) / cummax * 100).min()

    sell_trades = pf.trades
    wins = [t for t in sell_trades if t['profit_pct'] > 0]
    losses = [t for t in sell_trades if t['profit_pct'] <= 0]
    wr = len(wins)/len(sell_trades)*100 if sell_trades else 0
    avg_w = np.mean([t['profit_pct'] for t in wins]) if wins else 0
    avg_l = np.mean([t['profit_pct'] for t in losses]) if losses else 0

    bench_ret = None
    if bench_df is not None and len(bench_df) > 0:
        bench_ret = (bench_df['close'].iloc[-1] / bench_df['close'].iloc[0] - 1) * 100

    print(f'\n{"="*60}')
    print(f'  西湖宾馆 V4 (极简风控) RESULTS')
    print(f'{"="*60}')
    print(f'  Total Return:   {total_ret:+.2f}%')
    print(f'  Annual Return:  {ann_ret:+.2f}%')
    print(f'  Sharpe Ratio:   {sharpe:.2f}')
    print(f'  Max Drawdown:   {max_dd:.2f}%')
    print(f'  Volatility:     {rets.std()*np.sqrt(252)*100:.2f}%')
    print(f'  Trades:         {len(sell_trades)} (Buy: {buys})')
    print(f'  Win Rate:       {wr:.1f}%')
    print(f'  Avg Win:        {avg_w:+.2f}%')
    print(f'  Avg Loss:       {avg_l:+.2f}%')
    if avg_l != 0:
        print(f'  Profit Factor:  {abs(avg_w/avg_l):.2f}')
    if sell_trades:
        print(f'  Avg Hold Days:  {np.mean([t["days_held"] for t in sell_trades]):.1f}')
    print(f'')
    if (pf.bull_days + pf.bear_days) > 0:
        print(f'  Bull Days:      {pf.bull_days} ({pf.bull_days/(pf.bull_days+pf.bear_days)*100:.0f}%)')
        print(f'  Bear Days:      {pf.bear_days} ({pf.bear_days/(pf.bull_days+pf.bear_days)*100:.0f}%)')
    if bench_ret is not None:
        print(f'  CSI 300 Index:  {bench_ret:+.2f}%')
        print(f'  Alpha:          {total_ret - bench_ret:+.2f}%')
    print(f'{"="*60}')

    # 年度收益
    yearly = eq.resample('YE').last()
    prev = INITIAL_CAPITAL
    print(f'\n  V4 年度收益:')
    for yr_dt, val in yearly.items():
        yr = yr_dt.year
        ret = (val / prev - 1) * 100
        print(f'    {yr}: {ret:+.2f}%')
        prev = val

    # 退出原因分布
    from collections import Counter
    reasons = Counter(t['reason'] for t in sell_trades)
    print(f'\n  退出原因:')
    for r, c in reasons.most_common():
        avg = np.mean([t['profit_pct'] for t in sell_trades if t['reason'] == r])
        print(f'    {r}: {c}次 ({c/len(sell_trades)*100:.1f}%) avg={avg:+.2f}%')

    result = {
        'strategy': '西湖宾馆_V4',
        'dates': pf.dates, 'equity': pf.equity, 'trades': pf.trades,
        'bear_days': pf.bear_days, 'bull_days': pf.bull_days,
        'summary': {
            'total_return': total_ret, 'annual_return': ann_ret,
            'sharpe': sharpe, 'max_drawdown': max_dd,
            'win_rate': wr, 'avg_win': avg_w, 'avg_loss': avg_l,
            'total_trades': len(sell_trades),
            'benchmark_return': bench_ret,
            'alpha': (total_ret - bench_ret) if bench_ret else None,
        }
    }
    pickle.dump(result, open(os.path.join(CACHE_DIR, 'leitong_v4_result.pkl'), 'wb'))
    return result

if __name__ == '__main__':
    cache_file = os.path.join(CACHE_DIR, 'csi300_stocks.pkl')
    stocks = pickle.load(open(cache_file, 'rb'))['stocks']
    print(f'[LOAD] CSI 300: {len(stocks)} stocks')
    all_data = load_all_data(stocks)
    result = run_backtest(all_data)
