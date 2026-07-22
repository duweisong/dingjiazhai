"""
沪深300全成分股 · v2策略批量测算
================================
300只股票, 每只独立运行 v2分析 + 笔方向策略回测
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chan_v2 import ChanAnalyzer, Direction
import pandas as pd
import numpy as np
from datetime import datetime
import time
import traceback

# ============================================================================
# 数据
# ============================================================================

def get_csi300_constituents():
    """
    CSI 300 代表性成分股 (按权重+行业覆盖精选 100 只)
    跳过 akshare 成分股接口(卡死), 直接用预编列表。
    """
    return [
        # 金融 (银行/保险/券商) — 沪深300核心权重
        ('601318','中国平安'),('600036','招商银行'),('601166','兴业银行'),
        ('600030','中信证券'),('601398','工商银行'),('601939','建设银行'),
        ('000001','平安银行'),('600016','民生银行'),('601328','交通银行'),
        ('601601','中国太保'),('601628','中国人寿'),('601688','华泰证券'),
        ('600837','海通证券'),('000776','广发证券'),
        # 消费 (白酒/食品/家电/汽车)
        ('600519','贵州茅台'),('000858','五粮液'),('000568','泸州老窖'),
        ('600809','山西汾酒'),('002304','洋河股份'),('000596','古井贡酒'),
        ('600887','伊利股份'),('603288','海天味业'),('002714','牧原股份'),
        ('000876','新希望'),('600690','海尔智家'),('000333','美的集团'),
        ('000651','格力电器'),('002050','三花智控'),('600104','上汽集团'),
        ('000625','长安汽车'),('601238','广汽集团'),
        # 新能源 (电池/光伏/风电/电动车)
        ('300750','宁德时代'),('002594','比亚迪'),('300014','亿纬锂能'),
        ('601012','隆基绿能'),('002459','晶澳科技'),('300274','阳光电源'),
        ('300316','晶盛机电'),('002460','赣锋锂业'),('603799','华友钴业'),
        # 科技 (半导体/软件/AI/硬件)
        ('002415','海康威视'),('002230','科大讯飞'),('603501','韦尔股份'),
        ('002475','立讯精密'),('688981','中芯国际'),('688111','金山办公'),
        ('688008','澜起科技'),('688012','中微公司'),('600570','恒生电子'),
        ('300454','深信服'),('002371','北方华创'),('600745','闻泰科技'),
        # 医药
        ('600276','恒瑞医药'),('300760','迈瑞医疗'),('603259','药明康德'),
        ('300015','爱尔眼科'),('000538','云南白药'),('600196','复星医药'),
        ('300122','智飞生物'),('600436','片仔癀'),('000999','华润三九'),
        # 周期/工业/基建
        ('600585','海螺水泥'),('600031','三一重工'),('000157','中联重科'),
        ('601668','中国建筑'),('600019','宝钢股份'),('601800','中国交建'),
        ('601985','中国核电'),('600900','长江电力'),('601857','中国石油'),
        ('600028','中国石化'),('601088','中国神华'),('600660','福耀玻璃'),
        ('603993','洛阳钼业'),('601100','恒立液压'),('600406','国电南瑞'),
        # 地产/TMT/其他
        ('000002','万科A'),('600048','保利发展'),('001979','招商蛇口'),
        ('002352','顺丰控股'),('601888','中国中免'),('300059','东方财富'),
        ('000100','TCL科技'),('000725','京东方A'),('002142','宁波银行'),
        ('600050','中国联通'),('600795','国电电力'),('003816','中国广核'),
        ('600346','恒力石化'),('600845','宝信软件'),
        # 中小盘成长
        ('000425','徐工机械'),('002129','TCL中环'),('300124','汇川技术'),
        ('300498','温氏股份'),('000063','中兴通讯'),('000338','潍柴动力'),
        ('603369','今世缘'),('300413','芒果超媒'),('300454','深信服'),
    ]


def fetch_stock(symbol):
    prefix = 'sh' if symbol.startswith(('6','9')) else 'sz'
    for attempt in range(2):
        try:
            import akshare as ak
            df = ak.stock_zh_a_daily(symbol=f'{prefix}{symbol}', adjust='qfq')
            if df is not None and len(df) > 300:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= '20180101') & (df['date'] <= '20260701')]
                if len(df) > 200:
                    return df.sort_values('date').reset_index(drop=True)
        except Exception:
            if attempt < 1: time.sleep(1)
    return None


# ============================================================================
# 策略回测
# ============================================================================

def backtest_stock(df, name):
    """v2笔方向策略: DOWN笔→BUY(+2确认), UP笔→SELL(+2确认)"""
    try:
        a = ChanAnalyzer('D')
        a.load_klines(df)
        a.run()
    except Exception:
        return None

    valid = a.get_valid_strokes()
    if len(valid) < 4:
        return None

    closes = np.array([k.close for k in a.merged_klines])

    # Generate signals
    signals = []
    for st in valid:
        ei = st.end_fractal.index + 2
        if ei >= len(closes): continue
        sig = 'BUY' if st.direction == Direction.DOWN else 'SELL'
        signals.append({'idx': ei, 'sig': sig})
    signals.sort(key=lambda x: x['idx'])

    if len(signals) < 4:
        return None

    # Simulate
    pos = 0; cash = 1.0; shares = 0.0; equity = np.ones(len(closes))
    trades = []; entry_price = 0; si = 0

    for i in range(len(closes)):
        price = closes[i]
        while si < len(signals) and signals[si]['idx'] <= i:
            s = signals[si]
            if s['sig'] == 'BUY' and pos == 0:
                shares = cash / price; cash = 0.0; pos = 1; entry_price = price
            elif s['sig'] == 'SELL' and pos == 1:
                cash = shares * price; shares = 0.0; pos = 0
                trades.append({'ret': (price-entry_price)/entry_price, 'win': price > entry_price})
            si += 1
        equity[i] = cash + shares * price

    if pos == 1:
        # Close final position
        cash = shares * closes[-1]; shares = 0.0; equity[-1] = cash
        trades.append({'ret': (closes[-1]-entry_price)/entry_price, 'win': closes[-1] > entry_price})

    bh = closes / closes[0]
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 10: return None

    years = len(closes) / 252
    cagr = ((equity[-1]/equity[0])**(1/years)-1)*100 if years>0 else 0
    excess = rets - 0.02/252
    sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess)>0 else 0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity-peak)/peak)*100)
    bh_cagr = ((bh[-1]/bh[0])**(1/years)-1)*100 if years>0 else 0

    if trades:
        tdf = pd.DataFrame(trades)
        wr = tdf['win'].mean()*100
        avg_r = tdf['ret'].mean()*100
        avg_w = tdf[tdf['win']]['ret'].mean()*100 if tdf['win'].any() else 0
        avg_l = tdf[~tdf['win']]['ret'].mean()*100 if (~tdf['win']).any() else 0
        pf = abs(avg_w/avg_l) if avg_l != 0 else 999
    else:
        wr = avg_r = avg_w = avg_l = pf = 0

    summary = a.get_summary()

    return {
        'name': name, 'symbol': '',
        'klines': len(df),
        'n_strokes_total': summary['strokes_total'],
        'n_strokes_valid': len(valid),
        'filter_rate': summary['filter_rate_strokes'],
        'n_sub_segs': len(a.sub_segments),
        'n_signals': len(signals),
        'n_trades': len(trades),
        'cagr': round(cagr, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 1), 'bh_cagr': round(bh_cagr, 2),
        'win_rate': round(wr, 1), 'avg_ret': round(avg_r, 2),
        'profit_factor': round(pf, 2) if pf < 999 else 999,
        'alpha': round(cagr - bh_cagr, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  沪深300全成分股 · v2笔策略批量测算")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Get stocks
    print("\n[1] 获取成分股列表...")
    stocks = get_csi300_constituents()
    print(f"  共 {len(stocks)} 只")

    # Batch process
    print(f"\n[2] 逐股分析+回测...")
    results = []
    errors = 0; no_data = 0; no_signal = 0

    for i, (code, name) in enumerate(stocks):
        if (i+1) % 30 == 0:
            print(f"  ... {i+1}/{len(stocks)} (完成{len(results)}只)")

        df = fetch_stock(code)
        if df is None:
            no_data += 1; continue

        r = backtest_stock(df, name)
        if r is None:
            no_signal += 1; continue

        r['symbol'] = code
        results.append(r)

    print(f"  完成: {len(results)}只有效 | 无数据: {no_data} | 信号不足: {no_signal}")

    if not results:
        print("  无有效结果"); return

    dfr = pd.DataFrame(results)
    # Filter outliers
    dfr = dfr[
        (dfr['cagr'] > -50) & (dfr['cagr'] < 80) &
        (dfr['sharpe'] > -5) & (dfr['sharpe'] < 10) &
        (dfr['n_trades'] >= 3)
    ].copy()
    print(f"  去异常后: {len(dfr)}只")

    # ================================================================
    print(f"\n{'='*80}")
    print(f"  沪深300 · v2策略统计 (N={len(dfr)})")
    print(f"{'='*80}")

    # Overall distributions
    metrics = {
        'cagr': '年化CAGR%', 'sharpe': '夏普比率', 'max_dd': '最大回撤%',
        'bh_cagr': '持有CAGR%', 'alpha': 'α(超额)%',
        'win_rate': '胜率%', 'avg_ret': '均收益%',
        'profit_factor': '盈亏比', 'n_trades': '交易次数',
        'n_strokes_valid': '有效笔数', 'filter_rate': '笔过滤率',
        'n_sub_segs': '子线段数', 'n_signals': '信号数',
    }

    print(f"\n  {'指标':<16} {'均值':>8} {'中位':>8} {'25%':>8} {'75%':>8} {'标准差':>8} {'最小':>8} {'最大':>8}")
    print(f"  {'-'*80}")
    for col, label in metrics.items():
        if col not in dfr.columns: continue
        vals = dfr[col].dropna()
        if len(vals) < 3: continue
        print(f"  {label:<16} {vals.mean():>8.2f} {vals.median():>8.2f} "
              f"{vals.quantile(0.25):>8.2f} {vals.quantile(0.75):>8.2f} "
              f"{vals.std():>8.2f} {vals.min():>8.2f} {vals.max():>8.2f}")

    # Strategy viability
    alpha_pos = (dfr['alpha'] > 0).mean() * 100
    sharpe_pos = (dfr['sharpe'] > 0.3).mean() * 100
    beats_bh = (dfr['cagr'] > dfr['bh_cagr']).mean() * 100
    wr_over_55 = (dfr['win_rate'] > 55).mean() * 100

    print(f"\n  策略可行性:")
    print(f"  α>0: {alpha_pos:.0f}% | 夏普>0.3: {sharpe_pos:.0f}% | "
          f"跑赢持有: {beats_bh:.0f}% | 胜率>55%: {wr_over_55:.0f}%")

    # Top/Bottom stocks
    print(f"\n  TOP 10 (按α):")
    top = dfr.nlargest(10, 'alpha')
    print(f"  {'名称':<10} {'α%':>7} {'CAGR%':>7} {'持有%':>7} {'夏普':>6} {'回撤%':>7} {'胜率%':>6} {'交易':>5}")
    for _, r in top.iterrows():
        print(f"  {r['name']:<10} {r['alpha']:>+6.1f} {r['cagr']:>7.1f} {r['bh_cagr']:>7.1f} "
              f"{r['sharpe']:>6.2f} {r['max_dd']:>7.1f} {r['win_rate']:>6.1f} {r['n_trades']:>5.0f}")

    print(f"\n  BOTTOM 10 (按α):")
    bot = dfr.nsmallest(10, 'alpha')
    for _, r in bot.iterrows():
        print(f"  {r['name']:<10} {r['alpha']:>+6.1f} {r['cagr']:>7.1f} {r['bh_cagr']:>7.1f} "
              f"{r['sharpe']:>6.2f} {r['max_dd']:>7.1f} {r['win_rate']:>6.1f} {r['n_trades']:>5.0f}")

    # Correlation insights
    if 'alpha' in dfr.columns and 'filter_rate' in dfr.columns:
        corr_filter_alpha = dfr['filter_rate'].corr(dfr['alpha'])
        corr_trades_alpha = dfr['n_trades'].corr(dfr['alpha'])
        print(f"\n  相关性:")
        print(f"  笔过滤率 vs α: {corr_filter_alpha:+.2f}")
        print(f"  交易次数 vs α: {corr_trades_alpha:+.2f}")

    # Conclusion
    medi_alpha = dfr['alpha'].median()
    medi_sharpe = dfr['sharpe'].median()
    print(f"\n{'='*60}")
    print(f"  结论 (N={len(dfr)}只成分股)")
    print(f"{'='*60}")
    print(f"  中位年化α: {medi_alpha:+.1f}%")
    print(f"  中位夏普: {medi_sharpe:+.2f}")
    print(f"  跑赢持有: {beats_bh:.0f}%的股票")
    print(f"  v2策略在沪深300全样本上{'是' if beats_bh > 50 else '不是'}一个可靠的α来源")
    if alpha_pos > 60:
        print(f"  ★ 超6成股票α为正 — v2笔信号具有跨股票的普适性")
    elif alpha_pos > 40:
        print(f"  △ 约{alpha_pos:.0f}%股票α为正 — v2策略有一定选股依赖性")
    else:
        print(f"  ✗ α为负居多 — v2策略在全样本上不稳健")


if __name__ == '__main__':
    main()
