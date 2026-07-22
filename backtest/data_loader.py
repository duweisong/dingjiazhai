"""
数据加载模块 —— 支持 efinance (东方财富) 和 akshare 双数据源获取A股历史行情

默认使用 efinance (一行代码，含实时行情)，失败时自动回退 akshare。
在 config.py 中设置 DATA_SOURCE = "efinance" | "akshare" 切换。
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from config import (
    STOCK_CODE, STOCK_NAME, START_DATE, END_DATE, MARKET,
    DATA_SOURCE,
    MA_SHORT, MA_LONG, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, BOLL_PERIOD, BOLL_STD, ATR_PERIOD,
)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fetch_via_efinance(
    symbol: str,
    start: str,
    end: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    通过 efinance (东方财富) 获取个股日线历史数据。

    Parameters
    ----------
    symbol : str
        6位股票代码，如 '600519', '002036'
    start : str
        起始日期 "YYYY-MM-DD"
    end : str
        结束日期 "YYYY-MM-DD"
    max_retries : int
        最大重试次数 (默认3次，含指数退避)

    Returns
    -------
    pd.DataFrame with standardized English columns
    """
    import efinance as ef

    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"[Data] 正在从 efinance (东方财富) 下载 {symbol} "
                  f"({start} ~ {end}) ..."
                  + (f" [重试 {attempt+1}/{max_retries}]" if attempt > 0 else ""))

            df = ef.stock.get_quote_history(symbol)

            if df is None or df.empty:
                raise RuntimeError(f"efinance 未返回 {symbol} 的数据，请检查股票代码")

            # efinance 返回中文列名 → 标准化英文化
            col_map = {
                "股票名称": "name", "股票代码": "code",
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

            # 确保日期列存在
            if "date" not in df.columns:
                raise KeyError(f"efinance 返回数据中未找到日期列，实际列: {df.columns.tolist()}")

            # 筛选日期范围
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]

            if df.empty:
                raise RuntimeError(f"efinance {symbol} 在 {start}~{end} 范围内无数据")

            return df

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[Data] efinance 失败: {type(e).__name__}: {str(e)[:80]}")
                print(f"[Data] {wait}s 后重试...")
                time.sleep(wait)

    raise RuntimeError(
        f"efinance 在 {max_retries} 次尝试后仍无法获取 {symbol} 的数据。\n"
        f"  最后一次错误: {type(last_error).__name__}: {last_error}"
    )


def _fetch_via_akshare(
    symbol: str,
    start: str,
    end: str,
    market: str,
) -> pd.DataFrame:
    """
    通过 akshare (sina 源 + eastmoney 回退) 获取个股日线历史数据。

    Returns
    -------
    pd.DataFrame with standardized English columns
    """
    # 构造 sina 格式代码: sz002036 或 sh600000
    if market.upper() == "SZ":
        sina_code = f"sz{symbol}"
    else:
        sina_code = f"sh{symbol}"

    print(f"[Data] 正在从 akshare (sina) 下载 {sina_code} ({start} ~ {end}) ...")
    try:
        df = ak.stock_zh_a_daily(
            symbol=sina_code,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
    except Exception as e:
        print(f"[Data] sina 接口失败: {e}")
        print("[Data] 尝试东方财富接口...")
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
        except Exception as e2:
            raise RuntimeError(
                f"无法获取 {symbol} 的数据。\n"
                f"  sina: {e}\n"
                f"  eastmoney: {e2}"
            )

    if df is None or df.empty:
        raise RuntimeError(f"无法获取 {symbol} 的数据，请检查网络或股票代码")

    # akshare 有时返回中文列名，有时英文 — 统一映射
    if any(c in df.columns for c in ["成交额", "日期", "开盘"]):
        col_map = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount",
            "振幅": "amplitude", "涨跌幅": "pct_chg", "涨跌额": "change",
            "换手率": "turnover",
        }
        df.rename(columns=col_map, inplace=True)

    if "date" not in df.columns:
        raise KeyError(f"akshare 返回数据中未找到日期列，实际列: {df.columns.tolist()}")

    return df


def fetch_stock_data(
    symbol: str = STOCK_CODE,
    start: str = START_DATE,
    end: str = END_DATE,
    market: str = MARKET,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    获取A股日线历史数据，优先使用本地缓存。

    根据 config.DATA_SOURCE 选择数据源：
      - "efinance" — 东方财富接口 (默认，一行代码，含实时行情)
      - "akshare"   — akshare (sina 源 + eastmoney 回退)

    Parameters
    ----------
    symbol : str
        股票代码 (6位)
    start : str
        起始日期 "YYYY-MM-DD"
    end : str
        结束日期 "YYYY-MM-DD"
    market : str
        "SH" or "SZ" (仅 akshare 使用)
    force_refresh : bool
        强制重新下载

    Returns
    -------
    pd.DataFrame with columns:
        date, open, high, low, close, volume, amount, amplitude, pct_chg, turnover
    """
    cache_path = CACHE_DIR / f"{symbol}_{start}_{end}.parquet"

    if cache_path.exists() and not force_refresh:
        print(f"[Data] 从缓存加载 {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    # ---- 数据源 dispatch ----
    if DATA_SOURCE == "efinance":
        try:
            df = _fetch_via_efinance(symbol, start, end)
        except Exception as e:
            print(f"[Data] efinance 失败，自动回退 akshare: {type(e).__name__}")
            df = _fetch_via_akshare(symbol, start, end, market)
    else:
        df = _fetch_via_akshare(symbol, start, end, market)

    # ---- 统一后处理 (两源复用) ----
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 保存缓存
    df.to_parquet(cache_path, index=False)
    print(f"[Data] 数据已缓存至 {cache_path}")

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标，添加至 DataFrame。
    """
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ---- 均线 ----
    df["ma_short"] = close.rolling(MA_SHORT).mean()
    df["ma_long"] = close.rolling(MA_LONG).mean()
    df["ma_60"] = close.rolling(60).mean()          # 季线参考

    # ---- MACD ----
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd_dif"] = ema_fast - ema_slow
    df["macd_dea"] = df["macd_dif"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ---- RSI ----
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ---- 布林带 ----
    df["boll_mid"] = close.rolling(BOLL_PERIOD).mean()
    boll_std = close.rolling(BOLL_PERIOD).std()
    df["boll_upper"] = df["boll_mid"] + BOLL_STD * boll_std
    df["boll_lower"] = df["boll_mid"] - BOLL_STD * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]  # 带宽

    # ---- ATR (平均真实波幅) ----
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # ---- KDJ ----
    low_n = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ---- 成交量均线 ----
    df["vol_ma_5"] = df["volume"].rolling(5).mean()
    df["vol_ma_20"] = df["volume"].rolling(20).mean()

    # ---- 涨幅 ----
    df["ret"] = close.pct_change()
    df["ret_5d"] = close.pct_change(5)              # 5日涨幅
    df["ret_20d"] = close.pct_change(20)            # 20日涨幅

    # ---- 新高/新低标记 ----
    df["hh_20d"] = high.rolling(20).max()
    df["ll_20d"] = low.rolling(20).min()

    return df


def get_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    一站式获取数据：下载 + 计算指标
    """
    raw = fetch_stock_data(force_refresh=force_refresh)
    df = compute_indicators(raw)
    return df


if __name__ == "__main__":
    print(f"数据源: {DATA_SOURCE}")
    df = get_data(force_refresh=True)
    print(f"\n数据概览 ({STOCK_NAME} {STOCK_CODE}):")
    print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  交易天数: {len(df)}")
    print(f"  最新收盘: {df['close'].iloc[-1]:.2f}")
    print(f"  列: {df.columns.tolist()}")
    print(f"\n{df.tail(10).to_string()}")
