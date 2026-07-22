
"""
数据模块 —— 获取A股行情 + 大盘指数数据

复用现有 backtest/data_loader.py 的核心逻辑，
扩展为支持多股票 + 大盘指数的数据获取。
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_index_daily(symbol: str, start: str = "2015-01-01",
                      end: str = "2025-12-31") -> pd.DataFrame:
    """获取大盘指数日线数据 (上证/深证/创业板/沪深300)"""
    cache_path = CACHE_DIR / f"index_{symbol}_{start}_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df

    print(f"[Data] 下载指数 {symbol} {start}~{end}...")
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
    except Exception:
        try:
            df = ak.stock_zh_index_daily(symbol=f"sz{symbol}")
        except Exception as e:
            raise RuntimeError(f"无法获取指数{symbol}: {e}")

    df.rename(columns={"date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 标准化列名
    col_map = {"open": "open", "close": "close", "high": "high",
               "low": "low", "volume": "volume"}
    for old, new in list(col_map.items()):
        if old in df.columns:
            pass

    df.to_parquet(cache_path, index=False)
    return df


def fetch_stock_daily(symbol: str, name: str = "",
                      start: str = "2015-01-01",
                      end: str = "2025-12-31",
                      adjust: str = "qfq") -> pd.DataFrame:
    """获取个股日线数据（前复权）"""
    cache_path = CACHE_DIR / f"stock_{symbol}_{start}_{end}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print(f"[Data] 下载 {symbol} {name} {start}~{end}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"无法获取{symbol}: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"{symbol}无数据")

    # 标准化列名
    rename_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
        "涨跌额": "change", "换手率": "turnover",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["symbol"] = symbol
    df["name"] = name

    df.to_parquet(cache_path, index=False)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # 均线
    for p in [5, 10, 20, 30, 60, 120, 250]:
        df[f"ma_{p}"] = close.rolling(p).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 布林带
    df["boll_mid"] = close.rolling(20).mean()
    boll_std = close.rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2 * boll_std
    df["boll_lower"] = df["boll_mid"] - 2 * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close

    # 成交量均线
    for p in [5, 10, 20]:
        df[f"vol_ma_{p}"] = df["volume"].rolling(p).mean()

    # 涨跌幅
    df["ret_1d"] = close.pct_change()
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # 均线多头排列
    df["ma_bullish"] = (
        (df["ma_5"] > df["ma_20"]) &
        (df["ma_20"] > df["ma_60"])
    ).astype(int)

    # 量比
    df["vol_ratio"] = df["volume"] / df["vol_ma_20"].replace(0, np.nan)

    return df


def get_index_data(code: str = "000001", start: str = "2015-01-01",
                   end: str = "2025-12-31") -> pd.DataFrame:
    """一站式获取指数数据+指标"""
    raw = fetch_index_daily(code, start, end)
    return compute_indicators(raw)


def get_stock_data(symbol: str, name: str = "",
                   start: str = "2015-01-01",
                   end: str = "2025-12-31") -> pd.DataFrame:
    """一站式获取个股数据+指标"""
    raw = fetch_stock_daily(symbol, name, start, end)
    return compute_indicators(raw)


if __name__ == "__main__":
    # 测试下载上证指数
    df = get_index_data("000001", "2024-01-01", "2024-12-31")
    print(f"上证指数: {len(df)}行, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(df.tail(3)[["date", "close", "ma_20", "ma_60", "atr_pct"]].to_string())
