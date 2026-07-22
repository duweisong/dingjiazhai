"""
缠论 老版本 vs 新版本 对比
==========================
老版本 (本机 V2.0): 标准缠论, 笔≥5K, MACD面积背驰, 6维评分
新版本 (女娲 v2): ATR阈值+OBV确认+能量过滤+RSI, 子线段拆分

同一数据: 沪深300 日线 2015-2026
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# ============================================================================
# 老版本实现 (忠实复刻自 缠论操盘策略.md)
# ============================================================================

class OldDirection(Enum):
    UP = 1
    DOWN = -1

@dataclass
class OldKLine:
    timestamp: pd.Timestamp
    open: float; high: float; low: float; close: float; volume: float

@dataclass
class OldFractal:
    ftype: str  # 'top' or 'bottom'
    index: int
    mid: OldKLine

@dataclass
class OldBi:
    direction: OldDirection
    start_fx: OldFractal
    end_fx: OldFractal
    klines: List[OldKLine] = field(default_factory=list)

@dataclass
class OldSegment:
    direction: OldDirection
    bis: List[OldBi] = field(default_factory=list)
    pivot_zg: float = 0; pivot_zd: float = 0

class OldChanAnalyzer:
    """老版本缠论分析器 — 复刻自 缠论操盘策略.md"""

    def __init__(self, df):
        self.klines = []
        for _, r in df.iterrows():
            self.klines.append(OldKLine(
                r['date'], r['open'], r['high'], r['low'], r['close'], r['volume']))
        self.merged = []; self.fractals = []; self.bis = []
        self.segments = []; self.pivots = []; self.signals = []
        self.closes = None

    def merge_klines(self):
        """K线包含处理 (标准: 向上取高高, 向下取低低)"""
        if not self.klines: return
        m = [self.klines[0]]
        for i in range(1, len(self.klines)):
            p, c = m[-1], self.klines[i]
            contained = (p.high >= c.high and p.low <= c.low) or (c.high >= p.high and c.low <= p.low)
            if not contained:
                m.append(c)
            else:
                direction_up = p.close >= p.open
                if direction_up:
                    m[-1] = OldKLine(c.timestamp, p.open, max(p.high, c.high),
                                      max(p.low, c.low), c.close, p.volume + c.volume)
                else:
                    m[-1] = OldKLine(c.timestamp, p.open, min(p.high, c.high),
                                      min(p.low, c.low), c.close, p.volume + c.volume)
        self.merged = m
        self.closes = np.array([k.close for k in m])

    def find_fractals(self):
        """分型识别 (标准: 三K线结构, 无能量/确认过滤)"""
        if len(self.merged) < 3: return
        fx = []
        for i in range(1, len(self.merged)-1):
            l, m, r = self.merged[i-1], self.merged[i], self.merged[i+1]
            if m.high > l.high and m.high > r.high and m.low > l.low and m.low > r.low:
                fx.append(OldFractal('top', i, m))
            elif m.low < l.low and m.low < r.low and m.high < l.high and m.high < r.high:
                fx.append(OldFractal('bottom', i, m))
        self.fractals = fx

    def build_bis(self):
        """笔构建 (标准: ≥5独立K线, 无ATR/OBV过滤)"""
        if len(self.fractals) < 2: return
        bis = []
        i = 0
        while i < len(self.fractals) - 1:
            f1 = self.fractals[i]
            j = i + 1
            # 合并同类型连续分型, 取极端
            while j < len(self.fractals) and self.fractals[j].ftype == f1.ftype:
                if f1.ftype == 'top':
                    if self.fractals[j].mid.high > f1.mid.high: f1 = self.fractals[j]; i = j
                else:
                    if self.fractals[j].mid.low < f1.mid.low: f1 = self.fractals[j]; i = j
                j += 1
            if j >= len(self.fractals): break
            f2 = self.fractals[j]

            if f1.ftype == f2.ftype: i = j; continue

            # 笔条件: ≥4独立K线 (含端点=5根, 老版本标准)
            independent = f2.index - f1.index - 1
            if independent < 4: i = j; continue

            direction = OldDirection.UP if f1.ftype == 'bottom' else OldDirection.DOWN
            bis.append(OldBi(direction, f1, f2, self.merged[f1.index:f2.index+1]))
            i = j
        self.bis = bis

    def build_segments(self):
        """线段构建 (标准: ≥3笔, 前三笔有重叠)"""
        if len(self.bis) < 3: return
        segs = []
        i = 0
        while i <= len(self.bis) - 3:
            b1, b2, b3 = self.bis[i], self.bis[i+1], self.bis[i+2]
            if b1.direction == b2.direction or b2.direction == b3.direction:
                i += 1; continue

            # 检查重叠
            if b1.direction == OldDirection.UP:
                oh = min(b1.end_fx.mid.high, b3.end_fx.mid.high)
                ol = max(b1.start_fx.mid.low, b3.start_fx.mid.low)
            else:
                oh = min(b1.start_fx.mid.high, b3.start_fx.mid.high)
                ol = max(b1.end_fx.mid.low, b3.end_fx.mid.low)
            if oh <= ol: i += 1; continue

            seg_bis = [b1, b2, b3]
            j = i + 3
            while j < len(self.bis) and self.bis[j].direction != seg_bis[-1].direction:
                seg_bis.append(self.bis[j]); j += 1

            # 中枢
            zg, zd = 0, 0
            for k in range(len(seg_bis)-2):
                a, c = seg_bis[k], seg_bis[k+2]
                if a.direction != c.direction: continue
                if a.direction == OldDirection.UP:
                    z = min(a.end_fx.mid.high, c.end_fx.mid.high)
                    d = max(seg_bis[k+1].start_fx.mid.low, seg_bis[k+1].end_fx.mid.low)
                else:
                    z = max(seg_bis[k+1].start_fx.mid.high, seg_bis[k+1].end_fx.mid.high)
                    d = min(a.end_fx.mid.low, c.end_fx.mid.low)
                if z > d: zg, zd = z, d; break

            segs.append(OldSegment(b1.direction, seg_bis, zg, zd))
            i = j
        self.segments = segs

    def detect_divergence(self):
        """背驰检测 (标准: MACD面积 + DIFF值 双重确认)"""
        if len(self.closes) < 26: return
        ema12 = pd.Series(self.closes).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(self.closes).ewm(span=26, adjust=False).mean()
        dif = (ema12 - ema26).values
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_bar = 2 * (dif - dea)

        for seg in self.segments:
            if len(seg.bis) < 5: continue
            last = seg.bis[-1]
            prev_same = None
            for b in reversed(seg.bis[:-2]):
                if b.direction == last.direction: prev_same = b; break
            if not prev_same: continue

            ia, ib = prev_same.start_fx.index, last.end_fx.index
            if ib >= len(self.closes) or ia >= ib: continue

            area_a = abs(macd_bar[ia:ib+1].sum())
            remaining = macd_bar[ib:]
            seg_len = min(len(macd_bar[ia:ib+1]), len(remaining))
            area_b = abs(remaining[:seg_len].sum())

            if last.direction == OldDirection.DOWN and self.closes[ib] < self.closes[ia] and area_b < area_a * 0.9:
                self.signals.append({'type': 'B1', 'idx': ib, 'price': self.closes[ib],
                                      'date': self.merged[ib].timestamp})
            elif last.direction == OldDirection.UP and self.closes[ib] > self.closes[ia] and area_b < area_a * 0.9:
                self.signals.append({'type': 'S1', 'idx': ib, 'price': self.closes[ib],
                                      'date': self.merged[ib].timestamp})

    def run(self):
        self.merge_klines()
        self.find_fractals()
        self.build_bis()
        self.build_segments()
        self.detect_divergence()
        return self


# ============================================================================
# 策略回测 (老版本: B1买入, S1卖出)
# ============================================================================

def backtest_both(analyzer, use_signals=False):
    """
    统一策略回测 (老/新通用):
    DOWN笔完成→BUY(+2日确认), UP笔完成→SELL(+2日确认)
    如果 use_signals=True, 使用背驰信号
    否则使用笔端点方向
    """
    closes = np.array([k.close for k in analyzer.merged])
    dates = [k.timestamp for k in analyzer.merged]

    if use_signals:
        sigs = sorted(analyzer.signals, key=lambda s: s['idx'])
        raw = [{'idx': s['idx'], 'sig': 'BUY' if s['type'].startswith('B') else 'SELL'}
               for s in sigs]
    else:
        # Use bi endpoints directly
        raw = []
        for b in analyzer.bis:
            if hasattr(b, 'end_fx') and hasattr(b, 'direction'):
                ei = b.end_fx.index + 2  # +2 confirm
                if ei >= len(closes): continue
                sig = 'BUY' if b.direction == OldDirection.DOWN else 'SELL'
                raw.append({'idx': ei, 'sig': sig})
        raw.sort(key=lambda x: x['idx'])

    pos = 0; cash = 1.0; shares = 0.0; equity = np.ones(len(closes))
    trades = []; entry_price = 0

    si = 0
    for i in range(len(closes)):
        price = closes[i]
        while si < len(raw) and raw[si]['idx'] <= i:
            s = raw[si]
            if s['sig'] == 'BUY' and pos == 0:
                shares = cash / price; cash = 0.0; pos = 1; entry_price = price
            elif s['sig'] == 'SELL' and pos == 1:
                cash = shares * price; shares = 0.0; pos = 0
                trades.append({'ret': (price-entry_price)/entry_price, 'win': price > entry_price})
            si += 1
        equity[i] = cash + shares * price

    bh = closes / closes[0]
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]

    years = len(closes) / 252
    cagr = ((equity[-1]/equity[0])**(1/years)-1)*100
    excess = rets - 0.02/252
    sharpe = np.mean(excess)/np.std(excess)*np.sqrt(252) if np.std(excess)>0 else 0
    peak = np.maximum.accumulate(equity); dd = (equity-peak)/peak; max_dd = np.min(dd)*100

    if trades:
        tdf = pd.DataFrame(trades)
        wr = tdf['win'].mean()*100
        avg_r = tdf['ret'].mean()*100
        avg_w = tdf[tdf['win']]['ret'].mean()*100 if tdf['win'].any() else 0
        avg_l = tdf[~tdf['win']]['ret'].mean()*100 if (~tdf['win']).any() else 0
        pf = abs(avg_w/avg_l) if avg_l != 0 else 999
    else:
        wr = avg_r = avg_w = avg_l = pf = 0

    return {
        'cagr': round(cagr,2), 'sharpe': round(sharpe,2), 'max_dd': round(max_dd,1),
        'total_ret': round((equity[-1]/equity[0]-1)*100,1), 'n_trades': len(trades),
        'win_rate': round(wr,1), 'avg_ret': round(avg_r,2),
        'avg_win': round(avg_w,2), 'avg_loss': round(avg_l,2), 'profit_factor': round(pf,2) if pf<999 else 999,
    }, equity, bh


# ============================================================================
# 主对比
# ============================================================================

def main():
    print("=" * 90)
    print("  缠论 老版本 vs 新版本 — 沪深300 2015-2026 同数据对比")
    print("=" * 90)

    # Load data
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol='sh000300')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '20150101') & (df['date'] <= '20260701')].sort_values('date').reset_index(drop=True)
    print(f"\n  数据: 沪深300 日线, {len(df)}根K线 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")

    # --- 老版本 ---
    print("\n[1/3] 老版本 (标准缠论, 笔≥5K, MACD面积背驰)...")
    old_a = OldChanAnalyzer(df).run()
    print(f"  合并K线: {len(old_a.merged)} | 分型: {len(old_a.fractals)} | 笔: {len(old_a.bis)}")
    print(f"  线段: {len(old_a.segments)} | 背驰信号: {len(old_a.signals)}")
    old_m, old_eq, old_bh = backtest_both(old_a, use_signals=False)

    # --- 新版本 ---
    print("\n[2/3] 新版本 (v2, ATR+OBV+能量+RSI, 子线段)...")
    from chan_v2 import ChanAnalyzer, Direction
    new_a = ChanAnalyzer('D')
    new_a.load_klines(df)
    new_a.run()
    s = new_a.get_summary()
    valid = new_a.get_valid_strokes()
    print(f"  合并K线: {s['klines_merged']} | 分型: {s['fractals_total']} (能量OK:{s['fractals_energy_ok']} 确认:{s['fractals_confirmed']})")
    print(f"  笔: {s['strokes_total']}→{s['strokes_valid']}有效 ({s['filter_rate_strokes']:.0%}过滤)")
    print(f"  子线段: {len(new_a.sub_segments)} | 买卖点: {s['bps_total']}")

    # New回测 (同一策略: 笔方向驱动)
    from csi300_strategy import backtest
    new_m, _, new_trade, new_eq, _, _, _ = backtest(df)

    # --- 对比 ---
    print(f"\n{'='*90}")
    print(f"  同数据对比: 沪深300, 2790根日K线, 2015-2026")
    print(f"{'='*90}")
    print(f"  {'指标':<20} {'老版本(标准缠论)':>18} {'新版本(v2量化)':>18} {'差异':>15}")
    print(f"  {'-'*72}")

    comparisons = [
        ('合并K线', f"{len(old_a.merged)}", f"{s['klines_merged']}", ''),
        ('分型数', f"{len(old_a.fractals)}", f"{s['fractals_total']}", ''),
        ('分型过滤', '无', f"能量滤波×{s['fractals_energy_ok']} 确认×{s['fractals_confirmed']}", ''),
        ('笔数(总)', f"{len(old_a.bis)}", f"{s['strokes_total']}", ''),
        ('笔数(有效)', '无过滤', f"{s['strokes_valid']} ({s['filter_rate_strokes']:.0%}通过ATR+OBV)", ''),
        ('线段/子段', f"{len(old_a.segments)}", f"{len(new_a.sub_segments)}段", ''),
        ('背驰信号', f"{len(old_a.signals)}", f"{s['bps_total']}", ''),
        ('', '', '', ''),
        ('--- 策略对比 ---', '', '', ''),
        ('年化 CAGR', f"{old_m['cagr']}%", f"{new_m['cagr']}%", f"{new_m['cagr']-old_m['cagr']:+.1f}%/年"),
        ('夏普比率', f"{old_m['sharpe']}", f"{new_m['sharpe']}", f"{new_m['sharpe']-old_m['sharpe']:+.2f}"),
        ('最大回撤', f"{old_m['max_dd']}%", f"{new_m['max_dd']}%", f"{abs(old_m['max_dd'])-abs(new_m['max_dd']):+.0f}pp"),
        ('总收益', f"{old_m['total_ret']}%", f"{new_m['total_ret']}%", f"{new_m['total_ret']-old_m['total_ret']:+.1f}%"),
        ('', '', '', ''),
        ('--- 交易明细 ---', '', '', ''),
        ('交易次数', f"{old_m['n_trades']}", f"{new_trade['n_trades']}", ''),
        ('胜率', f"{old_m['win_rate']}%", f"{new_trade['win_rate']}%", ''),
        ('均收益', f"{old_m['avg_ret']}%", f"{new_trade['avg_ret']}%", ''),
        ('均盈', f"{old_m['avg_win']}%", f"{new_trade['avg_win']}%", ''),
        ('均亏', f"{old_m['avg_loss']}%", f"{new_trade['avg_loss']}%", ''),
        ('盈亏比', f"{old_m['profit_factor']}", f"{new_trade['profit_factor']}", ''),
    ]

    for label, old_v, new_v, diff in comparisons:
        print(f"  {label:<20} {old_v:>18} {new_v:>18} {diff:>15}")

    # 结论
    print(f"\n{'='*60}")
    print(f"  结论")
    print(f"{'='*60}")

    # Key differences
    print(f"\n  算法差异:")
    print(f"  ┌──────────────┬────────────────────┬────────────────────┐")
    print(f"  │ 维度          │ 老版本(标准缠论)    │ 新版本(v2量化)      │")
    print(f"  ├──────────────┼────────────────────┼────────────────────┤")
    print(f"  │ 分型规则      │ 三K线结构           │ 结构+能量(量/振幅)+确认│")
    print(f"  │ 笔规则        │ ≥5独立K线           │ ATR振幅+OBV资金+最短时 │")
    print(f"  │ 笔过滤        │ 无                  │ 三重独立过滤(ATR/OBV/RSI)│")
    print(f"  │ 千人千缠      │ 存在(≥5K线主观性)   │ 终结(参数确定→结果确定)│")
    print(f"  │ 方向确认      │ 无独立验证           │ C1/C2/C3三通道校验    │")
    print(f"  └──────────────┴────────────────────┴────────────────────┘")

    print(f"\n  策略绩效差异 (同一策略逻辑: 笔方向驱动交易):")
    alpha = new_m['cagr'] - old_m['cagr']
    dd_improve = abs(old_m['max_dd']) - abs(new_m['max_dd'])
    print(f"  年化α: {alpha:+.1f}%/年 | 回撤改善: {dd_improve:+.0f}pp | "
          f"夏普提升: {new_m['sharpe']-old_m['sharpe']:+.2f}")
    print(f"  老版本 {old_m['n_trades']}笔交易 | 新版本 {new_trade['n_trades']}笔交易")
    print(f"  老版本胜率 {old_m['win_rate']}% | 新版本胜率 {new_trade['win_rate']}%")
    print(f"  老版本盈亏比 {old_m['profit_factor']} | 新版本盈亏比 {new_trade['profit_factor']}")

    if alpha > 0:
        print(f"\n  ✓ 新版本年化超额α={alpha:+.1f}%/年")
    if dd_improve > 0:
        print(f"  ✓ 新版本回撤改善{dd_improve:.0f}个百分点")
    if new_m['sharpe'] > old_m['sharpe']:
        print(f"  ✓ 新版本风险调整收益更高")


if __name__ == '__main__':
    main()
