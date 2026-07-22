"""
ETF 行业轮动策略 V4.2 — V4.1 增强版
====================================

V4.2 vs V4.1 核心改进:
  1. 双条件闸门：M60 持续下行 + 股价在 M60 之下 → 空仓
     （避免牛市短暂回调被误杀，只在大盘真走弱时避险）
  2. 绝对动量过滤：只选 2月动量 > -1.5% 的 ETF
  3. 得分²加权：高置信度 ETF 获非线性高仓位

用法：
  python etf_sector_rotation_v4.2.py --backtest   # 回测
  python etf_sector_rotation_v4.2.py --live        # 信号+推送
"""

import os, sys, time, json, argparse, requests
import numpy as np
import pandas as pd
import efinance as ef
from datetime import datetime
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(OUTPUT_DIR, 'pushplus_config_v4.2.json')
CACHE_DIR = os.path.join(OUTPUT_DIR, '.cache', 'etf_sector')
os.makedirs(CACHE_DIR, exist_ok=True)
PUSHPLUS_URL = 'http://www.pushplus.plus/send'

# ============================================================
# V4.2 参数
# ============================================================
PARAMS = {
    # 因子参数 (同 V4.1)
    'momentum_bars': 44,         # 2月动量
    'return_weight': 0.60,
    'volume_weight': 0.40,
    'vol_short': 5,
    'vol_long': 22,

    # 选股规则
    'rank_low': 6,               # 放宽到 6-20（动量过滤后再收窄没意义）
    'rank_high': 20,
    'max_same_sector': 1,
    'num_select': 5,

    # V4.2 新增
    'market_ma': 60,             # 市场闸门 MA 窗口 (0=关闭)
    'market_ma_slope': 20,       # MA 斜率窗口：判断趋势方向
    'min_momentum': -0.015,      # 绝对动量最低阈值 (-1.5%)
    'score_power': 2.0,          # 得分加权指数 (2=平方加权)

    # 宏观倾斜（默认关闭，极端时自动激活）
    'macro_tilt': False,           # 是否启用 PMI+LPR 宏观倾斜
    'macro_pmi_contraction': 48,   # PMI 连续 3 月 < 此值 → 深度收缩 → 加仓防御
    'macro_pmi_expansion': 52,     # PMI 连续 3 月 > 此值 → 强劲扩张 → 加仓周期
    'macro_lpr_easing': -0.25,     # LPR 120 日变动 < 此值 → 显著宽松 → 加仓成长

    # 国家队信心指数（基于季度持仓变化）
    'nt_conviction_enabled': True,   # 是否启用国家队信心指数
    'nt_conviction_bull': 0.15,      # 买卖比 > 0.15 → 偏多，全体 +0.10 boost
    'nt_conviction_bear': -0.10,     # 买卖比 < -0.10 → 偏空，全体 -0.05 discount

    # 预测性仓位（基于K线拟合，领先季度数据）
    'nt_predictive_position': True,   # 是否使用预测仓位替代固定折扣
    'nt_pred_pos_cache': 'nt-position-sizer/../.cache/national_team/predicted_position.json',

    # 交易
    'trade_cost': 0.001,
    'buy_day': 3,
    'backtest_start': '20240101',
    'backtest_end': '20260610',
}

# 宏观倾斜-行业映射
MACRO_SECTOR_BOOST = {
    'contraction': ('公用事业', '医药', '消费'),   # 收缩→防御
    'expansion':   ('周期', '制造'),               # 扩张→周期
    'easing':      ('TMT', '新能源'),              # 宽松→成长
    'tightening':  ('金融',),                       # 紧缩→银行
}

# ============================================================
# ETF 池 (V4.1 同款 45 只)
# ============================================================
SECTOR_ETFS = {
    '159993': ('证券ETF鹏华', '金融'),    '512880': ('证券ETF国泰', '金融'),
    '512800': ('银行ETF华宝', '金融'),    '512660': ('军工ETF易方达', '国防军工'),
    '512670': ('国防ETF鹏华', '国防军工'),  '512710': ('军工龙头ETF', '国防军工'),
    '515880': ('通信ETF国泰', 'TMT'),     '512760': ('半导体ETF国泰', 'TMT'),
    '159869': ('游戏ETF华夏', 'TMT'),     '512980': ('传媒ETF广发', 'TMT'),
    '516510': ('云计算ETF易方达', 'TMT'),  '159995': ('芯片ETF华夏', 'TMT'),
    '515050': ('5GETF华夏', 'TMT'),       '159852': ('软件ETF易方达', 'TMT'),
    '560860': ('工业有色ETF万家', '周期'),  '515220': ('煤炭ETF国泰', '周期'),
    '516970': ('建材ETF国泰', '周期'),     '515210': ('钢铁ETF国泰', '周期'),
    '516780': ('稀土ETF华泰', '周期'),     '561330': ('矿业ETF国泰', '周期'),
    '159611': ('电力ETF广发', '公用事业'),  '516160': ('环保ETF易方达', '公用事业'),
    '561170': ('碳中和ETF易方达', '公用事业'), '159886': ('机械ETF富国', '制造'),
    '562500': ('机器人ETF华夏', '制造'),   '159638': ('高端装备ETF嘉实', '制造'),
    '516960': ('工业母机ETF华夏', '制造'),  '159996': ('家电ETF国泰', '消费'),
    '512690': ('酒ETF鹏华', '消费'),       '159843': ('食品饮料ETF招商', '消费'),
    '516130': ('旅游ETF华夏', '消费'),     '159766': ('旅游ETF富国', '消费'),
    '159875': ('新能源ETF嘉实', '新能源'),  '516390': ('新能源汽车ETF', '新能源'),
    '159857': ('光伏ETF天弘', '新能源'),    '561910': ('电池ETF易方达', '新能源'),
    '512010': ('医药ETF华夏', '医药'),     '159647': ('中药ETF鹏华', '医药'),
    '512170': ('医疗ETF华宝', '医药'),     '159883': ('医疗器械ETF永赢', '医药'),
    '516950': ('基建ETF广发', '基建'),     '512200': ('房地产ETF华夏', '基建'),
    '159745': ('建材ETF国泰', '基建'),     '159865': ('养殖ETF国泰', '农业'),
    '159825': ('农业ETF富国', '农业'),
}

# ============================================================
# 数据获取
# ============================================================
def fetch_all_data(use_cache=True):
    """获取 ETF 数据 + 沪深300 基准"""
    codes = list(SECTOR_ETFS.keys())
    cache_path = os.path.join(CACHE_DIR, f'sector_v42_{PARAMS["backtest_start"]}_{PARAMS["backtest_end"]}.parquet')

    if use_cache and os.path.exists(cache_path):
        data = pd.read_parquet(cache_path)
        valid = [c for c in codes if c in data.columns]
        # 沪深300单独取
        hs300 = fetch_hs300()
        return data, hs300, valid

    print(f'[Data] Fetching {len(codes)} ETFs + 沪深300...')
    dfs = {}
    for code in codes:
        name, _ = SECTOR_ETFS[code]
        print(f'  [{code}] {name}...', end=' ', flush=True)
        try:
            df = ef.fund.get_quote_history(code)
            df = df.rename(columns={'日期': 'date', '累计净值': 'cum_nav'})
            df['date'] = pd.to_datetime(df['date'])
            df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce').ffill()
            df = df.sort_values('date').set_index('date')
            df = df.loc[PARAMS['backtest_start']:PARAMS['backtest_end']]
            if len(df) >= 100:
                dfs[code] = df['cum_nav']
                print(f'OK({len(dfs[code])})')
            else:
                print(f'SKIP({len(df)})')
        except Exception as e:
            print(f'ERR:{type(e).__name__}')
        time.sleep(0.6)

    data = pd.DataFrame(dfs).ffill().bfill()
    data.to_parquet(cache_path)
    valid = [c for c in codes if c in data.columns]
    hs300 = fetch_hs300()
    return data, hs300, valid


def fetch_hs300():
    """获取沪深300ETF(510300)作为市场基准"""
    cache_path = os.path.join(CACHE_DIR, 'hs300_benchmark.parquet')
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        df = ef.fund.get_quote_history('510300')
        df = df.rename(columns={'日期': 'date', '累计净值': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
        df = df.sort_values('date').set_index('date')
        df = df.loc[PARAMS['backtest_start']:PARAMS['backtest_end']]
        df.to_parquet(cache_path)
        return df
    except:
        return None


# ============================================================
# 市场闸门
# ============================================================
def market_gate(hs300: pd.DataFrame, date, ma_window=60, slope_window=20):
    """
    双条件闸门：仅当 M60 持续下行 + 股价在 M60 之下 时空仓。
    避免牛市短暂回调被误杀。
    """
    if ma_window <= 0 or hs300 is None or date not in hs300.index:
        return True
    idx = hs300.index.get_loc(date)
    need = max(ma_window, ma_window + slope_window)
    if idx < need:
        return True  # 数据不足，开闸

    close = hs300['close'].iloc[idx]
    ma = hs300['close'].iloc[idx - ma_window:idx + 1].mean()
    # M60 近 N 日方向：当前 MA vs N 日前 MA
    ma_past = hs300['close'].iloc[idx - slope_window - ma_window:idx - slope_window + 1].mean()

    price_below = close < ma
    ma_declining = ma < ma_past  # M60 在下行

    return not (price_below and ma_declining)  # 双条件同时满足才关闸


# ============================================================
# 宏观倾斜 (PMI + LPR)
# ============================================================

_macro_cache = None  # 缓存宏观数据，避免重复拉取


def _load_macro_data():
    """加载 PMI + LPR 数据（带缓存）"""
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache
    try:
        import akshare as ak
        # PMI
        pmi = ak.macro_china_pmi()
        pmi.columns = ['ds', 'pmi_mfg', 'pmi_yoy', 'pmi_nmfg', 'pmi_nmfg_yoy']
        pmi['date'] = pd.to_datetime(
            pmi['ds'].str.replace('年', '-').str.replace('月份', '').str.replace('月', '') + '-01')
        pmi = pmi.dropna(subset=['date']).set_index('date').sort_index()
        pmi['pmi_mfg'] = pd.to_numeric(pmi['pmi_mfg'], errors='coerce')
        pmi = pmi.dropna(subset=['pmi_mfg'])
        pmi['pmi_trend'] = pmi['pmi_mfg'].diff(3)
        # LPR
        lpr = ak.macro_china_lpr()
        lpr = lpr.rename(columns={'TRADE_DATE': 'date', 'LPR1Y': 'lpr_1y'})
        lpr['date'] = pd.to_datetime(lpr['date'])
        lpr = lpr[['date', 'lpr_1y']].dropna().set_index('date').sort_index()
        lpr['lpr_trend'] = lpr['lpr_1y'].diff(120)
        _macro_cache = (pmi, lpr)
        return _macro_cache
    except Exception as e:
        print(f'[Macro] 加载失败: {e}')
        _macro_cache = (None, None)
        return _macro_cache


def _load_nt_conviction():
    """加载国家队信心指数（带缓存，季度有效）"""
    import json
    conv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '.cache', 'national_team', 'conviction.json')
    try:
        if os.path.exists(conv_file):
            with open(conv_file) as f:
                data = json.load(f)
            age = (datetime.now() - datetime.fromisoformat(data.get('computed_at', '2000-01-01'))).days
            if age < 90:
                return data
    except Exception:
        pass
    return None


def macro_regime(date, params):
    """
    判断宏观状态。仅在极端信号时返回非 neutral，日常返回 neutral。

    Returns
    -------
    dict: {'pmi': 'expansion'|'contraction'|'neutral',
           'lpr': 'easing'|'tightening'|'neutral',
           'nt_conviction': 'bullish'|'bearish'|'neutral',
           'active': bool}
    """
    pmi_r = lpr_r = nt_r = 'neutral'
    active = False

    # PMI + LPR 倾斜
    if params.get('macro_tilt'):
        pmi_df, lpr_df = _load_macro_data()
        p = params

        if pmi_df is not None and date in pmi_df.index:
            idx = pmi_df.index.get_loc(date)
            lookback = pmi_df.iloc[max(0, idx - 2):idx + 1]['pmi_mfg']
            if len(lookback) >= 3:
                if lookback.max() < p['macro_pmi_contraction']:
                    pmi_r = 'contraction'; active = True
                elif lookback.min() > p['macro_pmi_expansion']:
                    pmi_r = 'expansion'; active = True

        if lpr_df is not None:
            common = lpr_df.index[lpr_df.index <= date]
            if len(common) > 0:
                trend = lpr_df.loc[common[-1], 'lpr_trend']
                if pd.notna(trend):
                    if trend < p.get('macro_lpr_easing', -0.25):
                        lpr_r = 'easing'; active = True
                    elif trend > 0.25:
                        lpr_r = 'tightening'; active = True

    # 国家队信心指数
    if params.get('nt_conviction_enabled'):
        nt = _load_nt_conviction()
        if nt and nt.get('signal') in ('bullish', 'bearish'):
            nt_r = nt['signal']
            active = True

    return {'pmi': pmi_r, 'lpr': lpr_r, 'nt_conviction': nt_r, 'active': active}


def _load_predicted_position():
    """加载预测性仓位（来自 predictive_engine.py）"""
    import json
    p = PARAMS
    if not p.get('nt_predictive_position'):
        return None
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              p.get('nt_pred_pos_cache', ''))
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            age = (datetime.now() - datetime.fromisoformat(data['date'])).days
            if age < 7:
                return data
    except Exception:
        pass
    return None


# ============================================================
# 因子计算
# ============================================================
def calc_factors(data, codes):
    p = PARAMS
    MOM, VS, VL = p['momentum_bars'], p['vol_short'], p['vol_long']

    mom = pd.DataFrame(index=data.index)
    vol = pd.DataFrame(index=data.index)
    for c in codes:
        mom[c] = data[c] / data[c].shift(MOM) - 1.0
        dr = data[c].pct_change().abs()
        vol[c] = dr.rolling(VS).mean() / dr.rolling(VL).mean().replace(0, np.nan)

    scores = pd.DataFrame(index=mom.index)
    for c in codes:
        scores[c] = (p['return_weight'] * mom[c].rank(pct=True)
                     + p['volume_weight'] * vol[c].rank(pct=True))
    return scores.dropna(), mom, vol


# ============================================================
# V4.2 选股 (带动量过滤 + 得分加权)
# ============================================================
def select_v42(scores_row, mom_row, active_codes, date=None):
    """V4.2 选股：动量过滤 → 排名区间 → 行业约束 → 得分加权（+可选宏观倾斜）"""
    p = PARAMS

    # Step 1: 过滤负动量 ETF (2月收益 < -2% 的排除)
    valid = []
    for c in active_codes:
        if c in scores_row.index and c in mom_row.index:
            if pd.notna(mom_row[c]) and mom_row[c] >= p['min_momentum']:
                valid.append(c)

    if len(valid) < p['num_select']:
        # 动量过滤后不够，放宽阈值
        valid = [c for c in active_codes
                 if c in scores_row.index and c in mom_row.index
                 and pd.notna(mom_row[c])]

    # Step 1.5: 宏观倾斜（仅在极端信号时激活）
    adj_scores = scores_row[valid].copy()
    regime = macro_regime(date or pd.Timestamp.now(), p)
    if regime.get('active'):
        # PMI/LPR 行业倾斜
        for c in valid:
            sector = SECTOR_ETFS[c][1]
            boost = 1.0
            if regime['pmi'] == 'contraction' and sector in MACRO_SECTOR_BOOST['contraction']:
                boost += 0.25
            elif regime['pmi'] == 'expansion' and sector in MACRO_SECTOR_BOOST['expansion']:
                boost += 0.30
            if regime['lpr'] == 'easing' and sector in MACRO_SECTOR_BOOST['easing']:
                boost += 0.25
            elif regime['lpr'] == 'tightening' and sector in MACRO_SECTOR_BOOST['tightening']:
                boost += 0.20
            adj_scores[c] *= boost

        # 国家队信心指数：全局微调
        if regime.get('nt_conviction') == 'bullish':
            adj_scores *= 1.10   # 整体偏乐观，10% boost
        elif regime.get('nt_conviction') == 'bearish':
            adj_scores *= 0.95   # 整体偏谨慎，5% discount

    ranked = adj_scores.sort_values(ascending=False)
    n = len(ranked)

    # Step 2: 排名区间
    if n >= 20:
        lo, hi = p['rank_low'], p['rank_high']
    elif n >= 10:
        lo, hi = 1, min(n, 15)
    else:
        lo, hi = 1, n
    candidates = ranked.iloc[lo - 1:hi]

    # Step 3: 行业约束
    selected = []
    seen_sectors = set()
    for code, sc in candidates.items():
        sector = SECTOR_ETFS[code][1]
        if sector in seen_sectors:
            continue
        selected.append(code)
        seen_sectors.add(sector)
        if len(selected) >= p['num_select']:
            break
    if len(selected) < p['num_select']:
        for code, sc in candidates.items():
            if code not in selected:
                selected.append(code)
                if len(selected) >= p['num_select']:
                    break

    # Step 4: 得分加权 (score^alpha / sum)
    weights = {}
    if selected:
        raw = np.array([max(scores_row.get(c, 0), 0.01) for c in selected])
        powered = raw ** p['score_power']
        total = powered.sum()
        for c, pw in zip(selected, powered):
            weights[c] = pw / total if total > 0 else 1.0 / len(selected)

    return selected, weights


# ============================================================
# 交易日历
# ============================================================
def next_trading_day(date, dates, forward=True):
    if forward:
        for d in dates:
            if d >= date: return d
        return dates[-1]
    else:
        for d in reversed(dates):
            if d <= date: return d
        return dates[0]


def get_trade_dates(score_dates, buy_day=3):
    trade_dates = []
    months = sorted(set((d.year, d.month) for d in score_dates))
    for yr, mo in months:
        buy_target = pd.Timestamp(year=yr, month=mo, day=min(buy_day, 28))
        buy_date = next_trading_day(buy_target, score_dates, forward=True)
        month_dates = [d for d in score_dates if d.year == yr and d.month == mo]
        if not month_dates: continue
        sell_date = max(month_dates)
        if buy_date > sell_date: buy_date = sell_date
        trade_dates.append({'month': f'{yr}-{mo:02d}', 'buy': buy_date, 'sell': sell_date})
    return trade_dates


# ============================================================
# 回测
# ============================================================
def backtest(data, hs300, scores, mom, vol, daily_ret, active_codes):
    p = PARAMS
    td = get_trade_dates(scores.index, p['buy_day'])
    print(f'[Backtest] {len(td)} months, {scores.index[0].date()}~{scores.index[-1].date()}')

    holdings = {}
    nav = 1.0
    nav_hist = pd.Series(1.0, index=scores.index, dtype=float)
    trades_log = []
    in_position = False
    gate_blocks = 0  # 闸门拦截次数

    for i, date in enumerate(scores.index):
        is_buy = any(t['buy'].date() == date.date() for t in td)
        is_sell = any(t['sell'].date() == date.date() for t in td)

        # 市场闸门检查（仅在买入日）
        market_ok = market_gate(hs300, date, p['market_ma'], p['market_ma_slope'])

        # 卖出
        if is_sell and in_position and holdings:
            for c in list(holdings.keys()):
                trades_log.append({'date': date, 'action': 'SELL', 'code': c})
            holdings = {}
            in_position = False

        # 买入（需过闸门）
        if is_buy and not in_position:
            if market_ok:
                sr = scores.loc[date].dropna()
                mr = mom.loc[date].dropna() if date in mom.index else pd.Series()
                selected, weights = select_v42(sr, mr, active_codes, date)
                if selected:
                    for c in selected:
                        trades_log.append({'date': date, 'action': 'BUY', 'code': c})
                    nav *= (1 - p['trade_cost'])

                    # 预测性仓位：将 ETF 权重按仓位比例缩放，其余留现金
                    if p.get('nt_predictive_position'):
                        pred = _load_predicted_position()
                        if pred:
                            pos_scale = pred['predicted_position']
                            # 缩放到目标仓位（如 74%），其余为现金
                            holdings = {c: w * pos_scale for c, w in weights.items()}
                            cash_pct = 1.0 - pos_scale
                        else:
                            holdings = weights
                    else:
                        holdings = weights

                    in_position = True
            else:
                gate_blocks += 1

        # 持仓期收益
        if holdings and i < len(daily_ret):
            day_ret = sum(holdings.get(c, 0) * (daily_ret[c].iloc[i]
                          if not np.isnan(daily_ret[c].iloc[i]) else 0)
                          for c in holdings)
            nav *= (1 + day_ret)
        nav_hist[date] = nav

    # 绩效
    nav_hist = nav_hist.dropna()
    rets = nav_hist.pct_change().dropna()
    total = nav_hist.iloc[-1] - 1
    yrs = len(rets) / 252
    cagr = (1 + total) ** (1 / max(yrs, 0.1)) - 1
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = nav_hist.expanding().max()
    max_dd = float(((nav_hist / peak) - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = float((rets > 0).sum() / len(rets))
    # 仅统计持仓日的胜率
    pos_rets = rets[rets != 0]
    pos_wr = float((pos_rets > 0).sum() / len(pos_rets)) if len(pos_rets) > 0 else 0

    eq_r = daily_ret[active_codes].mean(axis=1).loc[nav_hist.index]
    eq_n = (1.0 + eq_r).cumprod()
    eq_t = eq_n.iloc[-1] - 1
    eq_c = (1 + eq_t) ** (1 / max(yrs, 0.1)) - 1

    return {
        'nav': nav_hist,
        'trades': trades_log,
        'metrics': {
            'total': total, 'cagr': cagr, 'sharpe': sharpe,
            'max_dd': max_dd, 'calmar': calmar, 'win_rate': wr,
            'pos_win_rate': pos_wr,
            'years': yrs, 'trades': len(trades_log),
            'gate_blocks': gate_blocks,
            'bench_cagr': eq_c, 'bench_total': eq_t,
        },
    }


# ============================================================
# 信号 + 推送
# ============================================================
def live_signal(data, hs300, scores, mom, vol, active_codes):
    p = PARAMS
    today = pd.Timestamp.now().normalize()
    dates = scores.index
    sig_date = next_trading_day(today, dates, forward=False)
    if sig_date > today:
        sig_date = dates[-1]

    td = get_trade_dates(dates, p['buy_day'])
    action = 'HOLD'
    next_action = 'N/A'
    for t in td:
        if t['buy'].date() == sig_date.date():
            action = 'BUY'; break
        if t['sell'].date() == sig_date.date():
            action = 'SELL'; break

    # 找下一个事件日
    for t in td:
        if t['buy'] > sig_date:
            next_action = str(t['buy'].date()); break
        if t['sell'] > sig_date:
            next_action = str(t['sell'].date()); break

    market_ok = market_gate(hs300, sig_date, p['market_ma'], p['market_ma_slope'])

    selected, weights = [], {}
    if action == 'BUY' and market_ok:
        sr = scores.loc[sig_date].dropna() if sig_date in scores.index else scores.iloc[-1].dropna()
        mr = mom.loc[sig_date].dropna() if sig_date in mom.index else pd.Series()
        selected, weights = select_v42(sr, mr, active_codes, sig_date)

    ranking_list = []
    latest = scores.iloc[-1].dropna().sort_values(ascending=False)
    for rank, (code, sc) in enumerate(latest.items(), 1):
        if code in SECTOR_ETFS and len(ranking_list) < 15:
            name, sector = SECTOR_ETFS[code]
            ranking_list.append({
                'rank': rank, 'code': code, 'name': name, 'sector': sector,
                'score': float(sc), 'selected': code in selected,
            })

    # 宏观状态
    regime = macro_regime(sig_date, p)
    nt_str = f' NT={regime["nt_conviction"]}' if regime.get('nt_conviction') not in ('neutral', None) else ''

    # 预测仓位
    pred_pos_str = ''
    if p.get('nt_predictive_position'):
        pred = _load_predicted_position()
        if pred:
            pred_pos_str = f' 预测仓位={pred["predicted_position"]:.0%}'

    macro_str = f'PMI={regime["pmi"]} LPR={regime["lpr"]}{nt_str}{pred_pos_str}{" ⚡" if regime.get("active") else ""}'

    return {
        'action': 'HOLD' if not market_ok and action == 'BUY' else action,
        'market_gate': 'OPEN' if market_ok else 'CLOSED (沪深300 < MA60)',
        'macro': macro_str,
        'selected': [{
            'code': c, 'name': SECTOR_ETFS[c][0], 'sector': SECTOR_ETFS[c][1],
            'weight': f'{weights.get(c,0):.1%}',
            'score': float(latest.get(c, 0)),
            'momentum': float(mom.iloc[-1].get(c, 0)) if c in mom.columns else 0,
        } for c in selected],
        'ranking': ranking_list,
        'trade_date': str(sig_date.date()),
        'next_trade': next_action,
    }


# ============================================================
# PushPlus
# ============================================================
def push(token, title, html):
    if not token: return False
    try:
        r = requests.post(PUSHPLUS_URL, json={'token': token, 'title': title, 'content': html, 'template': 'html'}, timeout=10)
        ok = r.json().get('code') == 200
        print(f'[PushPlus] {"✅" if ok else "❌"} {title}')
        return ok
    except Exception as e:
        print(f'[PushPlus] ❌ {e}')
        return False


def html_signal(sig):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    act = sig['action']
    emoji = {'BUY': '🟢 买入', 'SELL': '🔴 卖出', 'HOLD': '🟡 持有'}.get(act, '📊')
    color = {'BUY': '#27ae60', 'SELL': '#e74c3c', 'HOLD': '#f39c12'}.get(act, '#333')

    rows = ''
    for i, etf in enumerate(sig['selected']):
        rows += f"""<tr><td>#{i+1}</td><td><b>{etf['name']}</b></td>
            <td>{etf['code']}</td><td>{etf['sector']}</td>
            <td>{etf['score']:.4f}</td><td>{etf['momentum']:+.2%}</td><td><b>{etf['weight']}</b></td></tr>"""

    rank_rows = ''
    for r in sig['ranking'][:12]:
        mark = '★' if r['selected'] else ''
        c = '#FF8124' if r['selected'] else '#999'
        rank_rows += f"""<tr style="color:{c}"><td>{mark}#{r['rank']}</td>
            <td>{r['name']}</td><td>{r['code']}</td><td>{r['sector']}</td><td>{r['score']:.4f}</td></tr>"""

    return f"""<div style="font-family:'Microsoft YaHei',sans-serif;max-width:620px">
<h2 style="color:{color};border-bottom:2px solid {color};padding-bottom:10px">{emoji}信号 · ETF行业轮动 V4.2</h2>
<p style="color:#666">信号日: {sig['trade_date']} | 生成: {now}<br>
下次调仓: {sig['next_trade']} | 闸门: {sig['market_gate']}<br>宏观: {sig['macro']}</p>
<h3>🎯 {'本期持仓' if act == 'BUY' else '评分排名'} ({len(sig['selected'])}只, 得分加权)</h3>
<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:13px">
<tr style="background:#f5f5f5"><th>#</th><th>ETF</th><th>代码</th><th>行业</th><th>得分</th><th>动量</th><th>仓位</th></tr>
{rows}</table>
<h3 style="margin-top:18px">📊 Top 12 排名</h3>
<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:12px">
<tr style="background:#f5f5f5"><th>排名</th><th>ETF</th><th>代码</th><th>行业</th><th>得分</th></tr>
{rank_rows}</table>
<div style="margin-top:18px;padding:10px;background:#fff3e0;border-left:4px solid #FF8124;font-size:12px">
<b>V4.2 增强:</b> 沪深300 MA60闸门 | 动量> -2%过滤 | 得分²加权<br>
<b>规则:</b> 每月3日买入, 月末卖出 | 排名6-20 | 同行业≤1 | 5只</div>
<p style="color:#999;font-size:11px">⚠️ 仅供学习参考，不构成投资建议<br>ETF行业轮动 V4.2 · {now}</p></div>"""


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='ETF 行业轮动策略 V4.2')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    # 配置
    config = {'pushplus_token': ''}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    else:
        # 复制 V4.1 的 Token
        old_cfg = os.path.join(OUTPUT_DIR, 'pushplus_config_v4.json')
        if os.path.exists(old_cfg):
            with open(old_cfg) as f:
                old = json.load(f)
            config['pushplus_token'] = old.get('pushplus_token', '')
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

    # 数据
    data, hs300, active = fetch_all_data(use_cache=not args.no_cache)
    print(f'[Data] {len(active)} ETFs, HS300: {hs300 is not None}')
    scores, mom, vol = calc_factors(data, active)
    daily_ret = data[active].pct_change()
    print(f'[Factors] {len(scores)} valid days')

    # 实盘
    if args.live:
        sig = live_signal(data, hs300, scores, mom, vol, active)
        print(f"\n{'='*60}\n  V4.2 {'BUY' if sig['action']=='BUY' else 'SELL'} 信号\n{'='*60}")
        print(f"  日期: {sig['trade_date']} | 闸门: {sig['market_gate']}")
        print(f"  宏观: {sig['macro']}")
        print(f"  下次: {sig['next_trade']}")
        for etf in sig['selected']:
            print(f"  {etf['code']} {etf['name']:<14} [{etf['sector']:<5}] "
                  f"得分:{etf['score']:.4f} 动量:{etf['momentum']:+.2%} 仓位:{etf['weight']}")
        print(f"\n  Top 12:")
        for r in sig['ranking']:
            m = '★' if r['selected'] else ' '
            print(f"  {m}#{r['rank']:<3} {r['code']} {r['name']:<14} [{r['sector']:<5}] {r['score']:.4f}")

        token = config.get('pushplus_token', '')
        if token and sig['action'] in ('BUY', 'SELL') and not args.dry_run:
            push(token, f"ETF轮动V4.2 · {sig['action']} · {sig['trade_date']}", html_signal(sig))
        elif not token:
            print('\n[PushPlus] 未配置Token')
        return sig

    # 回测
    print(f"\n{'='*60}\n  V4.2 回测模式\n{'='*60}")
    result = backtest(data, hs300, scores, mom, vol, daily_ret, active)

    m = result['metrics']
    print(f"\n  {'指标':<18} {'V4.1':>10} {'V4.2':>10} {'改进':>10}")
    print(f"  {'-'*48}")
    v41 = {'cagr': 0.3690, 'sharpe': 1.46, 'max_dd': -0.1870, 'win_rate': 0.4798, 'total': 0.9701, 'pos_wr': 0.4798}
    for key, label in [('total', '累计收益'), ('cagr', '年化收益'), ('sharpe', '夏普'), ('max_dd', '最大回撤'), ('pos_win_rate', '持仓胜率')]:
        v = m[key] if key != 'total' else m['total']
        old = v41.get(key, 0)
        delta = v - old
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
        if key in ('max_dd',): arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
        fmt = '.2%' if key != 'sharpe' else '.2f'
        print(f"  {label:<18} {old:{fmt}}  {v:{fmt}}  {delta:+{fmt}} {arrow}")

    print(f"\n  闸门拦截: {m['gate_blocks']} 个月 (空仓避跌)")
    print(f"  累计交易: {m['trades']} 笔")
    print(f"  全天胜率: {m['win_rate']:.1%} (含空仓日)")
    print(f"  等权基准年化: {m['bench_cagr']:.2%}")
    print(f"  超额收益: {m['cagr'] - m['bench_cagr']:.2%}")

    return result


if __name__ == '__main__':
    main()
