#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO + Keltner Channel 均值回归交易策略 (增强版)
==============================================
基于 Awesome Oscillator (动量) + Keltner Channel (波动率) 的完整交易系统

策略逻辑：
  入场做多：AO 动量上升（当前值 > N根K线前）
  离场平仓：收盘价跌破 KC 下轨 / 触发止损

增强功能：
  - ATR 动态仓位管理（波动大仓位小）
  - 固定止损 + 移动止损
  - 信号过滤防频繁交易 + 冷却期
  - 可视化图表（价格+指标+信号+权益曲线）
  - 数据本地缓存（避免重复下载/限流）
  - 信号告警导出

用法：
  python ao_keltner_strategy.py                        # AES, 5年回测+图表
  python ao_keltner_strategy.py --ticker NVDA --check  # 快速查看最新信号
  python ao_keltner_strategy.py --plot                 # 只显示图表
  python ao_keltner_strategy.py --export               # 导出CSV
  python ao_keltner_strategy.py --demo                 # 演示模式(模拟数据)
"""

import argparse
import os
import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# --- 可选依赖 ---
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# 数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".strategy_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 通达信默认路径
TDX_DEFAULT = r"C:\zd_pazq_hy"


# ============================================================
#  策略参数配置（可调优）
# ============================================================
class Cfg:
    """所有策略参数集中管理"""
    # --- AO (Awesome Oscillator) ---
    AO_SHORT = 5            # 短周期 SMA
    AO_LONG = 34            # 长周期 SMA
    AO_SHIFT = 5            # 比较 N 根K线前的AO值

    # --- Keltner Channel ---
    KC_PERIOD = 20          # EMA 周期
    KC_MULT = 2.0           # ATR 倍数

    # --- 交易管理 ---
    MIN_BARS = 3            # 两笔交易最小间隔K线数
    RISK_PCT = 0.02         # 单笔风险占资金比 (2%)
    STOP_ATR = 2.5          # 固定止损 ATR 倍数
    TRAIL_ATR = 3.0         # 移动止损 ATR 倍数
    MAX_POS = 0.30          # 单票最大仓位 (30%)

    # --- 资金 ---
    CAPITAL = 100_000       # 初始资金（美元）


# ============================================================
#  指标计算
# ============================================================
def calc_ao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Awesome Oscillator
    AO = SMA(中位价, 5) - SMA(中位价, 34)
    返回 ao（值）和 ao_up（是否上升）
    """
    df = df.copy()
    mp = (df["High"] + df["Low"]) / 2
    df["ao"] = mp.rolling(Cfg.AO_SHORT).mean() - mp.rolling(Cfg.AO_LONG).mean()
    df["ao_up"] = df["ao"] > df["ao"].shift(Cfg.AO_SHIFT)
    return df


def calc_kc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keltner Channel
    中轨 = EMA(典型价, 20)
    上下轨 = 中轨 +/- 2 * ATR(20)
    """
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["kc_mid"] = tp.ewm(span=Cfg.KC_PERIOD, adjust=False).mean()

    # ATR
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["atr"] = pd.Series(tr).rolling(Cfg.KC_PERIOD).mean()

    df["kc_low"] = df["kc_mid"] - Cfg.KC_MULT * df["atr"]
    df["kc_hi"] = df["kc_mid"] + Cfg.KC_MULT * df["atr"]
    df["below_kc"] = df["Close"] < df["kc_low"]
    return df


# ============================================================
#  信号生成 + 仓位管理（核心逻辑）
# ============================================================
def generate(df: pd.DataFrame) -> pd.DataFrame:
    """
    遍历K线，逐日生成交易信号和持仓状态

    入场条件：AO上升 AND 距上次交易 >= MIN_BARS
    离场条件（满足任一）：
      1. 收盘价跌破 KC 下轨（原策略）
      2. 跌破固定止损
      3. 跌破移动止损
      4. AO转负 + 价格跌破 KC 中轨（额外保护）

    仓位计算：基于 ATR 的风险管理
      shares = min(risk_amount / stop_distance, max_position / price)
    """
    df = df.copy()
    n = len(df)

    # 初始化输出列
    df["signal"] = 0          # 1=买入, -1=卖出
    df["pos"] = 0             # 持仓股数
    df["entry"] = np.nan      # 入场价
    df["stop"] = np.nan       # 当前止损价
    df["trail"] = np.nan      # 移动止损价
    df["pnl"] = 0.0           # 平仓盈亏
    df["action"] = ""         # 可读的操作描述

    # 状态变量
    held = False              # 是否持仓
    shares = 0                # 持仓股数
    entry = 0.0               # 入场价格
    stop = 0.0                # 固定止损价格
    trail = 0.0               # 移动止损价格
    hi = 0.0                  # 持仓期间最高收盘价
    bars = Cfg.MIN_BARS       # 距上次交易K线数
    cash = Cfg.CAPITAL        # 当前可用资金

    start = max(Cfg.AO_LONG, Cfg.KC_PERIOD) + 1
    for i in range(start, n):
        ao_up = df["ao_up"].iloc[i]
        below = df["below_kc"].iloc[i]
        close = df["Close"].iloc[i]
        atr = df["atr"].iloc[i]
        ao = df["ao"].iloc[i]
        opn = df["Open"].iloc[i]
        idx = df.index[i]

        if np.isnan(atr) or atr == 0:
            continue

        # ========================================
        #  空仓状态：检查入场条件
        # ========================================
        if not held:
            if ao_up and bars >= Cfg.MIN_BARS:
                # 计算仓位
                risk = cash * Cfg.RISK_PCT          # 总风险金额
                dist = Cfg.STOP_ATR * atr            # 止损距离
                if dist <= 0:
                    continue
                raw = risk / dist                    # 按风险计算的股数
                mx = (cash * Cfg.MAX_POS) / close    # 仓位上限股数
                shares = int(min(raw, mx))
                if shares <= 0:
                    continue

                # 执行买入
                entry = opn
                stop = entry - dist
                trail = entry - Cfg.TRAIL_ATR * atr
                hi = close
                held = True
                bars = 0

                df.loc[idx, "signal"] = 1
                df.loc[idx, "pos"] = shares
                df.loc[idx, "entry"] = entry
                df.loc[idx, "stop"] = stop
                df.loc[idx, "trail"] = trail
                df.loc[idx, "action"] = (
                    f"BUY {shares}sh @${entry:.2f} "
                    f"stop=${stop:.2f} risk=${risk:.0f}"
                )
            else:
                reason = ""
                if not ao_up:
                    reason = "AO未上升"
                elif bars < Cfg.MIN_BARS:
                    reason = f"冷却中({bars}/{Cfg.MIN_BARS})"
                df.loc[idx, "action"] = f"WAIT ({reason})" if reason else "WAIT"

        # ========================================
        #  持仓状态：更新止损 + 检查离场
        # ========================================
        else:
            hi = max(hi, close)
            trail = max(trail, hi - Cfg.TRAIL_ATR * atr)

            # 判断离场
            exit_ok = False
            why = ""

            if below:
                exit_ok = True
                why = "KC下限"
            elif close <= stop:
                exit_ok = True
                why = "固定止损"
            elif close <= trail:
                exit_ok = True
                why = "移动止损"
            elif ao < 0 and not ao_up and close < df["kc_mid"].iloc[i]:
                exit_ok = True
                why = "AO转负+破中轨"

            if exit_ok:
                # 执行卖出
                xp = opn
                pnl = (xp - entry) * shares
                pct = (xp / entry - 1) * 100

                df.loc[idx, "signal"] = -1
                df.loc[idx, "pos"] = 0
                df.loc[idx, "pnl"] = pnl
                df.loc[idx, "action"] = (
                    f"SELL {shares}sh @${xp:.2f} "
                    f"PnL=${pnl:+,.0f}({pct:+.1f}%) [{why}]"
                )

                cash += pnl
                held = False
                shares = 0
                bars = 0
            else:
                # 继续持有
                df.loc[idx, "pos"] = shares
                df.loc[idx, "entry"] = entry
                df.loc[idx, "stop"] = max(stop, trail)
                df.loc[idx, "trail"] = trail
                upnl = (close - entry) * shares
                upct = (close / entry - 1) * 100
                df.loc[idx, "action"] = (
                    f"HOLD {shares}sh @${entry:.2f} "
                    f"uPnL=${upnl:+,.0f}({upct:+.1f}%)"
                )

        bars += 1

    return df


# ============================================================
#  绩效统计
# ============================================================
def stats(df: pd.DataFrame) -> dict:
    """统计策略绩效指标"""
    t = df[df["signal"] == -1]
    if len(t) == 0:
        return {}

    w = t[t["pnl"] > 0]
    l = t[t["pnl"] <= 0]
    total = t["pnl"].sum()
    dd = (t["pnl"].cumsum() - t["pnl"].cumsum().cummax()).min()

    if len(l) > 0 and l["pnl"].sum() != 0:
        pf = abs(w["pnl"].sum() / l["pnl"].sum())
    else:
        pf = 99

    return {
        "交易次数": len(t),
        "盈/亏": f"{len(w)}/{len(l)}",
        "胜率": f"{len(w)/len(t)*100:.1f}%",
        "总盈亏": f"${total:+,.0f}",
        "总收益率": f"{total/Cfg.CAPITAL*100:+.1f}%",
        "平均盈利": f"${w['pnl'].mean():+,.0f}" if len(w) > 0 else "N/A",
        "平均亏损": f"${l['pnl'].mean():+,.0f}" if len(l) > 0 else "N/A",
        "盈亏比": f"{pf:.2f}",
        "最大回撤": f"${dd:+,.0f}",
        "最终资金": f"${Cfg.CAPITAL+total:,.0f}",
    }


# ============================================================
#  报表输出
# ============================================================
def report(df: pd.DataFrame, ticker: str, period: str):
    """打印完整交易报表（参数、绩效、信号、持仓建议）"""
    st = stats(df)
    last = df.iloc[-1]
    bar = "=" * 60

    print(f"\n{bar}")
    print(f"  AO + Keltner Channel 均值回归交易策略")
    print(f"  股票: {ticker}  |  周期: {period}")
    print(f"  数据范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(bar)

    # 参数
    print(f"\n  策略参数")
    print(f"  {'-'*40}")
    print(f"  AO: ({Cfg.AO_SHORT},{Cfg.AO_LONG}) shift={Cfg.AO_SHIFT}")
    print(f"  KC: ({Cfg.KC_PERIOD},{Cfg.KC_MULT})")
    print(f"  单笔风险: {Cfg.RISK_PCT*100:.0f}%  |  仓位上限: {Cfg.MAX_POS*100:.0f}%")
    print(f"  初始资金: ${Cfg.CAPITAL:,.0f}")

    # 绩效
    if st:
        print(f"\n  回测绩效")
        print(f"  {'-'*40}")
        for k, v in st.items():
            print(f"  {k:<10}: {v}")

    # 最近交易信号
    sigs = df[df["signal"] != 0].tail(8)
    if len(sigs) > 0:
        print(f"\n  最近 8 笔交易信号")
        print(f"  {'-'*55}")
        for idx, row in sigs.iterrows():
            tag = ">>> BUY  " if row["signal"] == 1 else "<<< SELL "
            print(f"  {idx.date()}  {tag} {row['action']}")

    # 当前持仓 + 操作建议
    print(f"\n  {'='*56}")
    if last["pos"] > 0:
        c = last["Close"]
        e = last["entry"]
        pnl = (c - e) * last["pos"]
        pct = (c / e - 1) * 100
        print(f"  [持仓中] {int(last['pos'])} 股 @ ${e:.2f}")
        print(f"  现价: ${c:.2f}  浮盈: ${pnl:+,.0f} ({pct:+.1f}%)")
        print(f"  止损: ${last['stop']:.2f}  移动止损: ${last['trail']:.2f}")
        print(f"  >>> 操作建议: 继续持有，守止损位 <<<")
    else:
        print(f"  [空仓]")
        ao_s = "AO上升" if last["ao_up"] else "AO下降"
        kc_s = "跌破KC下轨!" if last["below_kc"] else "KC下轨上方"
        print(f"  AO动量: {ao_s}   KC状态: {kc_s}")
        if last["ao_up"] and not last["below_kc"]:
            print(f"  >>> 操作建议: 关注入场！AO动量回升中 <<<")
        elif last["below_kc"]:
            print(f"  >>> 操作建议: 回避！价格在KC下轨之下 <<<")
        else:
            print(f"  >>> 操作建议: 观望，等待AO回升信号 <<<")
    print(f"  {'='*56}")
    print()


# ============================================================
#  通达信 .day 文件读取器
# ============================================================
def read_tdx_day(filepath: str) -> pd.DataFrame:
    """
    读取通达信日线 .day 文件。
    格式: 每条32字节 = date(i) open(i) high(i) low(i) close(i)
                       amount(f) volume(i) reserved(i)
    价格以 int*100 存储。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    records = []
    with open(filepath, "rb") as f:
        while True:
            data = f.read(32)
            if len(data) < 32:
                break
            date, o, h, l, c, amt, vol, _ = struct.unpack("i i i i i f i i", data)
            # 跳过无效日期（可能为0或未来日期）
            if date < 19900101 or date > 20991231:
                continue
            records.append({
                "date": date,
                "open": o / 100.0,
                "high": h / 100.0,
                "low": l / 100.0,
                "close": c / 100.0,
                "amount": amt,
                "volume": vol,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    # 标准化列名
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "amount": "Amount", "volume": "Volume",
    })
    return df


def find_tdx_file(code: str, base_path: str = TDX_DEFAULT) -> str | None:
    """
    根据股票代码查找通达信 .day 文件。
    code: 6位代码 (如 "000001", "600519")
    返回完整路径或 None。
    """
    code = code.zfill(6)
    # 按优先级搜索：sh > sz > bj
    for exchange in ("sh", "sz", "bj"):
        fpath = os.path.join(base_path, "vipdoc", exchange,
                             "lday", f"{exchange}{code}.day")
        if os.path.exists(fpath):
            return fpath
    return None


def list_tdx_stocks(base_path: str = TDX_DEFAULT, limit: int = 20):
    """列出通达信中可用的股票代码"""
    stocks = []
    for exchange in ("sh", "sz", "bj"):
        lday_dir = os.path.join(base_path, "vipdoc", exchange, "lday")
        if not os.path.isdir(lday_dir):
            continue
        for fname in os.listdir(lday_dir):
            if fname.endswith(".day"):
                code = fname.replace(exchange, "").replace(".day", "")
                stocks.append((f"{exchange}{code}", exchange, code))
    return sorted(stocks)[:limit]
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.csv")


def load_cached(ticker: str) -> pd.DataFrame | None:
    """从本地缓存加载数据"""
    p = _cache_path(ticker)
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df
    return None


def save_cache(ticker: str, df: pd.DataFrame):
    df.to_csv(_cache_path(ticker), encoding="utf-8-sig")


def demo_data(n_bars: int = 1500) -> pd.DataFrame:
    """
    生成模拟数据（模仿 AES 走势特征：长期横盘、间歇性暴跌、缓慢修复）。
    当所有在线数据源不可用时作为演示。
    """
    np.random.seed(42)
    dates = pd.date_range(start="2000-01-03", periods=n_bars, freq="B")
    close = 45.0
    trend = -0.002  # 轻微下行趋势（模拟长期阴跌）
    regime = "normal"
    regime_counter = 0
    prices = []

    for i in range(n_bars):
        # 随机切换市场状态
        if regime_counter <= 0:
            r = np.random.random()
            if r < 0.03:
                regime = "crash"
                regime_counter = np.random.randint(30, 80)
            elif r < 0.10:
                regime = "recovery"
                regime_counter = np.random.randint(50, 120)
            else:
                regime = "normal"
                regime_counter = np.random.randint(60, 200)
        regime_counter -= 1

        if regime == "crash":
            shock = np.random.randn() * 3.5 - 1.5
        elif regime == "recovery":
            shock = np.random.randn() * 2.0 + 0.8
        else:
            shock = np.random.randn() * 1.8 + trend

        close = close * (1 + shock / 100)
        close = max(close, 1.5)  # 不归零
        prices.append(close)

    df = pd.DataFrame({
        "Open":  [p * (1 + np.random.randn() * 0.008) for p in prices],
        "High":  [p * (1 + abs(np.random.randn()) * 0.018) for p in prices],
        "Low":   [p * (1 - abs(np.random.randn()) * 0.018) for p in prices],
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n_bars),
    }, index=dates)

    # 确保 OHLC 关系正确
    for i in range(n_bars):
        o, h, l, c = df.iloc[i][["Open", "High", "Low", "Close"]]
        df.iloc[i, df.columns.get_loc("High")] = max(o, c, h)
        df.iloc[i, df.columns.get_loc("Low")] = min(o, c, l)

    return df


def fetch_akshare(code: str, years: int = 5) -> pd.DataFrame | None:
    """通过 akshare (东方财富) 获取A股日线数据"""
    try:
        import akshare as ak
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "最高": "High",
            "最低": "Low", "收盘": "Close", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"  [akshare] {e}")
        return None


def fetch_data(ticker: str, period: str = "5y",
               use_local: bool = False, tdx_path: str = TDX_DEFAULT,
               online: bool = False) -> pd.DataFrame:
    """
    获取数据优先级:
      1. 通达信本地 (--local)
      2. akshare 在线 (--online, A股)
      3. 缓存
      4. Yahoo Finance (美股)
      5. 演示模拟数据
    """
    clean_code = ticker.replace(".SS", "").replace(".SZ", "").replace(".SH", "")
    is_a = clean_code.isdigit() and len(clean_code) <= 6

    # ---- 1. 通达信本地 ----
    if use_local and is_a:
        fpath = find_tdx_file(clean_code, tdx_path)
        if fpath:
            print(f"  [tdx] {clean_code} ...")
            try:
                df = read_tdx_day(fpath)
                if len(df) > 0:
                    print(f"  [tdx] {len(df)} rows")
                    save_cache(ticker, df)
                    return df
            except Exception as e:
                print(f"  [tdx] {e}")

    # ---- 2. akshare 在线 ----
    if online and is_a:
        print(f"  [akshare] {clean_code} ...")
        df = fetch_akshare(clean_code)
        if df is not None and len(df) > 50:
            print(f"  [akshare] {len(df)} rows")
            save_cache(ticker, df)
            return df
        print(f"  [akshare] failed")

    # ---- 3. 缓存 ----
    cached = load_cached(ticker)
    if cached is not None:
        print(f"  [cache] {len(cached)} rows")
        return cached

    # ---- 4. Yahoo ----
    if HAS_YFINANCE:
        try:
            print(f"  [yfinance] {ticker} ...")
            end = datetime.now()
            delta_map = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
            delta = delta_map.get(period, 3650)
            start = end - timedelta(days=delta) if period != "max" else "1990-01-01"
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=15)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                save_cache(ticker, df)
                print(f"  [yfinance] {len(df)} rows")
                return df
        except Exception as e:
            print(f"  [yfinance] {e}")

    # ---- 5. 演示数据 ----
    print(f"  [demo] simulation ...")
    return demo_data(1500)


# ============================================================
#  可视化
# ============================================================
def plot_strategy(df: pd.DataFrame, ticker: str, save: bool = True):
    """绘制策略全景图：价格+KC通道、AO指标、持仓信号、权益曲线"""
    if not HAS_MPL:
        print("  [!] matplotlib not installed, skipping plot")
        print("      pip install matplotlib")
        return

    trades = df[df["signal"] != 0]
    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
    fig.suptitle(f"AO + Keltner Channel Strategy — {ticker}",
                 fontsize=16, fontweight="bold", y=0.98)

    # ====== 图1: 价格 + KC 通道 + 买卖点 ======
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#1a1a2e", linewidth=0.8, label="Close", zorder=1)
    ax1.plot(df.index, df["kc_mid"], color="#888888", linewidth=0.5,
             linestyle="--", label="KC Mid", zorder=1)
    ax1.fill_between(df.index, df["kc_low"], df["kc_hi"],
                     alpha=0.08, color="#3498db", label="KC Band", zorder=0)
    ax1.plot(df.index, df["kc_low"], color="#e74c3c", linewidth=0.5, alpha=0.6, zorder=1)
    ax1.plot(df.index, df["kc_hi"], color="#2ecc71", linewidth=0.5, alpha=0.6, zorder=1)

    # 买卖点标注
    if len(buys) > 0:
        ax1.scatter(buys.index, buys["Close"], marker="^", s=80,
                    color="#00ff88", edgecolors="black", linewidths=0.5,
                    zorder=5, label=f"Buy ({len(buys)})")
    if len(sells) > 0:
        ax1.scatter(sells.index, sells["Close"], marker="v", s=80,
                    color="#ff4757", edgecolors="black", linewidths=0.5,
                    zorder=5, label=f"Sell ({len(sells)})")

    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.2)
    ax1.set_title("Price + Keltner Channel + Trade Signals", fontsize=12, fontweight="bold")

    # ====== 图2: AO 动量震荡指标 ======
    ax2 = axes[1]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["ao"].fillna(0)]
    ax2.bar(df.index, df["ao"], width=1, color=colors, alpha=0.7, zorder=2)
    ax2.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax2.set_ylabel("AO", fontsize=11)
    ax2.set_title("Awesome Oscillator (AO)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    # ====== 图3: 持仓状态 + 止损线 ======
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df["pos"], color="#2ecc71", alpha=0.1, zorder=1)

    # 持仓期间画价格和止损
    in_pos = df["pos"] > 0
    if in_pos.any():
        pos_df = df[in_pos]
        ax3.plot(pos_df.index, pos_df["Close"], color="#1a1a2e",
                 linewidth=0.6, alpha=0.8, label="Price (in position)")
        ax3.plot(pos_df.index, pos_df["stop"], color="#e74c3c",
                 linewidth=0.6, linestyle="--", alpha=0.7, label="Stop Loss")

    ax3.set_ylabel("Price ($)", fontsize=11)
    ax3.set_title("Position Status & Stop Loss", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ====== 图4: 权益曲线 ======
    ax4 = axes[3]
    if len(sells) > 0:
        equity = sells["pnl"].cumsum() + Cfg.CAPITAL
        ax4.plot(sells.index, equity, color="#2ecc71", linewidth=1.2, label="Strategy Equity")

        # 最大回撤区域
        cummax = equity.cummax()
        dd = equity - cummax
        ax4.fill_between(sells.index, equity, cummax,
                         where=(dd < 0), color="#e74c3c", alpha=0.15, label="Drawdown")

        ax4.axhline(y=Cfg.CAPITAL, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        final_val = equity.iloc[-1]
        ret = (final_val / Cfg.CAPITAL - 1) * 100
        ax4.set_title(f"Equity Curve  (Final: ${final_val:,.0f}  |  Return: {ret:+.1f}%)",
                      fontsize=12, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No completed trades", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=14)

    ax4.set_ylabel("Equity ($)", fontsize=11)
    ax4.set_xlabel("Date", fontsize=11)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    # 格式化
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    if save:
        fname = f"c:/AI/{ticker}_strategy_chart.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"\n  Chart saved: {fname}")
    plt.close(fig)


# ============================================================
#  快速信号检查（近200天）
# ============================================================
def quick_check(ticker: str):
    """快速检查最新信号，适合每日盯盘使用"""
    print(f"\n  Checking {ticker} latest signals...")

    end = datetime.now()
    start = end - timedelta(days=200)

    if HAS_YFINANCE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=15)
            if df is None or len(df) == 0:
                df = demo_data(200)[-150:]
            elif isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df = demo_data(200)[-150:]
    else:
        df = demo_data(200)[-150:]

    if df.empty:
        print(f"  No data for {ticker}")
        return

    df = calc_ao(df)
    df = calc_kc(df)
    df = generate(df)

    # 最近6天明细
    recent = df.tail(6)
    hdr = f"  {'Date':<12} {'Close':>8} {'AO':>8} {'AOup':>5} {'<KC':>5}  {'Action'}"
    print(f"\n{hdr}")
    print(f"  {'-'*65}")
    for idx, row in recent.iterrows():
        ao_flag = "Y" if row["ao_up"] else "N"
        kc_flag = "Y" if row["below_kc"] else "N"
        print(f"  {str(idx.date()):<12} {row['Close']:>8.2f} {row['ao']:>8.4f} "
              f"{ao_flag:>5} {kc_flag:>5}  {row['action'][:35]}")

    last = df.iloc[-1]
    print(f"\n  {'='*55}")
    print(f"  Latest: {last.name.date()}  "
          f"Close=${last['Close']:.2f}  AO={last['ao']:.4f}  "
          f"KC_low=${last['kc_low']:.2f}")
    if last["pos"] > 0:
        print(f"  Position: {int(last['pos'])}sh @ ${last['entry']:.2f}  "
              f"stop=${last['stop']:.2f}")
        print(f"  >> HOLD - watch stop loss <<")
    else:
        if last["ao_up"] and not last["below_kc"]:
            print(f"  >> ENTRY signal active! AO rising & price above KC low <<")
        elif last["below_kc"]:
            print(f"  >> AVOID - price below KC lower band <<")
        else:
            print(f"  >> WAIT for AO momentum to turn up <<")
    print(f"  {'='*55}\n")


# ============================================================
#  沪深300 成分股列表（2025Q4 调整后）
# ============================================================
HS300_STOCKS = [
    "000001","000002","000063","000100","000157","000166","000301","000333","000338",
    "000408","000425","000538","000568","000596","000617","000625","000630","000651",
    "000661","000725","000768","000776","000786","000792","000800","000831","000858",
    "000876","000895","000938","000963","000975","000977","000983","002001","002007",
    "002027","002049","002050","002129","002142","002179","002180","002230","002236",
    "002241","002252","002271","002304","002311","002352","002371","002410","002415",
    "002460","002466","002475","002493","002594","002601","002648","002709","002714",
    "002736","002812","002916","002920","002938","300014","300015","300033","300059",
    "300122","300124","300142","300207","300223","300274","300308","300316","300347",
    "300390","300408","300413","300433","300442","300450","300498","300502","300529",
    "300628","300661","300750","300751","300760","300763","300896","300919","300957",
    "300979","300999","600000","600009","600010","600011","600015","600016","600018",
    "600019","600025","600028","600029","600030","600031","600036","600048","600050",
    "600061","600066","600085","600089","600104","600111","600115","600132","600150",
    "600161","600176","600183","600188","600196","600233","600276","600309","600346",
    "600362","600377","600383","600406","600415","600426","600436","600438","600460",
    "600482","600489","600515","600519","600547","600570","600584","600585","600588",
    "600600","600606","600690","600703","600732","600741","600745","600754","600760",
    "600763","600795","600803","600809","600837","600845","600886","600887","600893",
    "600900","600905","600918","600919","600926","600938","600941","600958","600989",
    "601006","601009","601012","601021","601058","601066","601077","601088",
    "601100","601108","601111","601117","601127","601138","601166","601169","601186",
    "601211","601225","601229","601236","601238","601288","601318","601319","601328",
    "601336","601360","601377","601390","601398","601456","601600","601601","601607",
    "601615","601618","601628","601633","601658","601668","601669","601688","601689",
    "601696","601698","601728","601766","601788","601800","601808","601816","601818",
    "601838","601857","601868","601872","601877","601878","601881","601888","601898",
    "601899","601901","601916","601919","601939","601966","601985","601988","601995",
    "601998","603019","603160","603259","603260","603288","603290","603296","603369",
    "603392","603501","603596","603606","603659","603799","603806","603833","603899",
    "603986","603993","605117","605499","688008","688009","688012","688036","688041",
    "688047","688065","688082","688111","688126","688169","688187","688223","688256",
    "688271","688303","688396","688484","688506","688525","688561","688599","688728",
    "688777","688819","688981",
]


# ============================================================
#  批量回测 - 沪深300
# ============================================================
def batch_backtest_hs300(tdx_path: str = TDX_DEFAULT, capital: float = 100_000,
                         risk: float = 0.02) -> pd.DataFrame:
    """
    对沪深300全部成分股执行批量回测。
    返回包含每只股票绩效的 DataFrame。
    """
    results = []
    total = len(HS300_STOCKS)
    skipped = 0

    print(f"\n  {'='*80}")
    print(f"  沪深300 批量回测")
    print(f"  初始资金: ${capital:,.0f}  |  单笔风险: {risk*100:.0f}%")
    print(f"  成分股总数: {total}  |  数据源: 通达信 {tdx_path}")
    print(f"  {'='*80}\n")

    Cfg.CAPITAL = capital
    Cfg.RISK_PCT = risk

    for idx, code in enumerate(HS300_STOCKS):
        # 进度
        pct_done = (idx + 1) / total * 100
        bar_len = 30
        filled = int(bar_len * (idx + 1) / total)
        bar = "=" * filled + "-" * (bar_len - filled)

        # 查找文件
        fpath = find_tdx_file(code, tdx_path)
        if not fpath:
            skipped += 1
            print(f"\r  [{bar}] {pct_done:5.1f}%  {code}  [SKIP: not found]", end="")
            continue

        try:
            df = read_tdx_day(fpath)
            if len(df) < 100:  # 数据太少跳过
                skipped += 1
                continue

            bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
            df = calc_ao(df)
            df = calc_kc(df)
            df = generate(df)
            st = stats(df)

            if st:
                # 提取数值
                trades_n = int(st["交易次数"])
                winrate = float(st["胜率"].replace("%", ""))
                total_pnl_str = st["总盈亏"].replace("$", "").replace(",", "")
                total_pnl = float(total_pnl_str.replace("+", ""))
                ret_pct = float(st["总收益率"].replace("%", "").replace("+", ""))
                pf = float(st["盈亏比"])
                dd_str = st["最大回撤"].replace("$", "").replace(",", "")
                max_dd = float(dd_str.replace("+", ""))
                final_str = st["最终资金"].replace("$", "").replace(",", "")
                final_val = float(final_str)

                results.append({
                    "code": code,
                    "trades": trades_n,
                    "winrate": winrate,
                    "pnl": total_pnl,
                    "return_pct": ret_pct,
                    "profit_factor": pf,
                    "max_dd": max_dd,
                    "final": final_val,
                    "bh_return": bh_ret,
                    "excess": ret_pct - bh_ret,  # 超额收益
                })
            else:
                skipped += 1

        except Exception as e:
            skipped += 1

        # 每10只刷新一次进度
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"\r  [{bar}] {pct_done:5.1f}%  {code}  "
                  f"({len(results)} ok, {skipped} skip)", end="")

    print()  # 换行
    print(f"\n  完成: {len(results)} 只股票有效, {skipped} 只跳过\n")

    # 生成排名
    if not results:
        print("  No results to display.")
        return pd.DataFrame()

    rdf = pd.DataFrame(results).sort_values("return_pct", ascending=False)
    rdf["rank"] = range(1, len(rdf) + 1)

    return rdf


def print_batch_report(rdf: pd.DataFrame):
    """打印批量回测汇总报告"""
    if rdf.empty:
        return

    n = len(rdf)
    profitable = len(rdf[rdf["return_pct"] > 0])
    beat_bh = len(rdf[rdf["excess"] > 0])
    avg_ret = rdf["return_pct"].mean()
    avg_winrate = rdf["winrate"].mean()
    avg_pf = rdf["profit_factor"].mean()
    median_ret = rdf["return_pct"].median()

    # Use ASCII-safe output for Windows GBK terminals
    try:
        _ = "沪深"
        use_cn = True
    except:
        use_cn = False

    sep = "=" * 80
    dash = "-" * 80

    print(f"\n  {sep}")
    if use_cn:
        print(f"  CSI 300 Batch Backtest - Summary Report")
    else:
        print(f"  CSI 300 Batch Backtest - Summary Report")
    print(f"  {sep}")
    print(f"  Valid samples: {n}")
    print(f"  Strategy profitable: {profitable} ({profitable/n*100:.1f}%)")
    print(f"  Beat Buy & Hold:    {beat_bh} ({beat_bh/n*100:.1f}%)")
    print(f"  Avg return:   {avg_ret:+.1f}%")
    print(f"  Median return:{median_ret:+.1f}%")
    print(f"  Avg win rate: {avg_winrate:.1f}%")
    print(f"  Avg PF:       {avg_pf:.2f}")
    print()

    # TOP 20
    print(f"  {dash}")
    print(f"  *** TOP 20 (by Strategy Return) ***")
    print(f"  {dash}")
    hdr = f"  {'Rank':<5} {'Code':<8} {'Trades':>6} {'Win%':>7} {'StratRet':>9} {'B&HRet':>9} {'Excess':>9} {'PF':>7} {'MaxDD':>10}"
    print(hdr)
    print(f"  {dash}")
    for _, row in rdf.head(20).iterrows():
        print(f"  {row['rank']:<5} {row['code']:<8} {int(row['trades']):>6} "
              f"{row['winrate']:>6.1f}% {row['return_pct']:>8.1f}% "
              f"{row['bh_return']:>8.1f}% {row['excess']:>8.1f}% "
              f"{row['profit_factor']:>6.2f} ${row['max_dd']:>9,.0f}")

    # BOTTOM 10
    print(f"\n  {dash}")
    print(f"  *** BOTTOM 10 (Lowest Return) ***")
    print(f"  {dash}")
    print(hdr)
    print(f"  {dash}")
    for _, row in rdf.tail(10).iloc[::-1].iterrows():
        print(f"  {row['rank']:<5} {row['code']:<8} {int(row['trades']):>6} "
              f"{row['winrate']:>6.1f}% {row['return_pct']:>8.1f}% "
              f"{row['bh_return']:>8.1f}% {row['excess']:>8.1f}% "
              f"{row['profit_factor']:>6.2f} ${row['max_dd']:>9,.0f}")

    # 超额收益TOP
    print(f"\n  {dash}")
    print(f"  *** Excess Return TOP 20 (Strategy - Buy&Hold) ***")
    print(f"  {dash}")
    excess_top = rdf.sort_values("excess", ascending=False).head(20)
    exhdr = f"  {'Code':<8} {'StratRet':>9} {'B&HRet':>9} {'Excess':>9} {'Win%':>7} {'PF':>7}"
    print(exhdr)
    print(f"  {dash}")
    for _, row in excess_top.iterrows():
        print(f"  {row['code']:<8} {row['return_pct']:>8.1f}% "
              f"{row['bh_return']:>8.1f}% {row['excess']:>8.1f}% "
              f"{row['winrate']:>6.1f}% {row['profit_factor']:>6.2f}")

    print(f"\n  {sep}\n")
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="AO + Keltner Channel 均值回归交易策略 (增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ao_keltner_strategy.py --local --ticker 000001    上证指数
  python ao_keltner_strategy.py --local --ticker 600519    贵州茅台
  python ao_keltner_strategy.py --local --ticker 000001 --export
  python ao_keltner_strategy.py --tdx-list                  列出可用股票
  python ao_keltner_strategy.py --demo --ticker DEMO        演示模式
        """,
    )
    p.add_argument("--ticker", default="AES",
                   help="股票代码: 美股如 AES/NVDA, A股如 000001/600519 (默认: AES)")
    p.add_argument("--period", default="5y",
                   help="回看周期: 1y/2y/5y/10y/max (默认: 5y)")
    p.add_argument("--capital", type=float, default=100000,
                   help="初始资金 (默认: 100000)")
    p.add_argument("--risk", type=float, default=0.02,
                   help="单笔风险比例 (默认: 0.02=2%%)")
    p.add_argument("--check", action="store_true",
                   help="快速检查最新交易信号")
    p.add_argument("--export", action="store_true",
                   help="导出交易信号和权益曲线到 CSV")
    p.add_argument("--demo", action="store_true",
                   help="使用模拟演示数据（跳过网络下载）")
    p.add_argument("--plot", action="store_true",
                   help="仅生成图表，跳过文字报表")
    p.add_argument("--local", action="store_true",
                   help="使用通达信本地数据")
    p.add_argument("--online", action="store_true",
                   help="使用 akshare 在线获取A股数据")
    p.add_argument("--tdx-path", default=TDX_DEFAULT,
                   help=f"通达信安装路径 (默认: {TDX_DEFAULT})")
    p.add_argument("--tdx-list", action="store_true",
                   help="列出通达信可用股票代码")
    p.add_argument("--batch-hs300", action="store_true",
                   help="批量回测沪深300全部成分股")
    args = p.parse_args()

    # 批量回测沪深300
    if args.batch_hs300:
        rdf = batch_backtest_hs300(
            tdx_path=args.tdx_path,
            capital=args.capital,
            risk=args.risk,
        )
        print_batch_report(rdf)
        if args.export and not rdf.empty:
            rdf.to_csv("c:/AI/HS300_batch_results.csv",
                       index=False, encoding="utf-8-sig")
            print(f"  Full results saved: c:/AI/HS300_batch_results.csv")
        sys.exit(0)

    # 列出可用股票
    if args.tdx_list:
        stocks = list_tdx_stocks(args.tdx_path, limit=100)
        print(f"\n  通达信可用股票 (前100只):")
        print(f"  {'代码':<12} {'交易所':<6}")
        print(f"  {'-'*18}")
        for code, ex, num in stocks:
            print(f"  {code:<12} {ex.upper():<6}")
        print(f"\n  共 {len(stocks)} 只有效股票")
        print(f"  用法: python ao_keltner_strategy.py --local --ticker 000001")
        sys.exit(0)

    Cfg.CAPITAL = args.capital
    Cfg.RISK_PCT = args.risk
    ticker = args.ticker.upper()
    period = args.period

    # 快速模式
    if args.check:
        quick_check(ticker)
        sys.exit(0)

    # ==== 1. 获取数据 ====
    if args.demo:
        print(f"\n  [demo] Generating simulation data for {ticker}...")
        df = demo_data(1500)
        print(f"  [demo] {len(df)} bars | "
              f"Buy&Hold: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.1f}%")
    else:
        df = fetch_data(ticker, period,
                        use_local=args.local, tdx_path=args.tdx_path,
                        online=args.online)

    if df is None or df.empty:
        print(f"  No data available for {ticker}")
        sys.exit(1)

    # ==== 2. 计算指标 + 生成信号 ====
    print(f"  Calculating indicators + generating signals...")
    df = calc_ao(df)
    df = calc_kc(df)
    df = generate(df)

    # ==== 3. 输出报表 ====
    if not args.plot:
        report(df, ticker, period)

    # ==== 4. 生成图表 ====
    if HAS_MPL:
        plot_strategy(df, ticker, save=True)
    elif not args.plot:
        print("  [!] Install matplotlib for charts: pip install matplotlib")

    # ==== 5. 导出 ====
    if args.export:
        out = df[df["signal"] != 0][["signal", "action", "pnl"]].copy()
        out["date"] = out.index.date
        fname = f"c:/AI/{ticker}_signals.csv"
        out.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  Signals exported: {fname} ({len(out)} records)")

        eq = df[df["signal"] == -1]["pnl"].cumsum() + Cfg.CAPITAL
        eq.to_csv(f"c:/AI/{ticker}_equity.csv", encoding="utf-8-sig")
        print(f"  Equity curve exported: c:/AI/{ticker}_equity.csv")
