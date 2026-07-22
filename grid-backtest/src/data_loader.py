"""
网格交易回测系统 — 数据加载模块

支持 efinance + akshare 双数据源，含自动回退与缓存。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _ensure_efinance():
    """延迟导入 efinance，缺失时给出明确提示"""
    try:
        import efinance as ef  # noqa: F401
        return ef
    except ImportError:
        raise ImportError(
            "efinance 未安装，请运行: pip install efinance"
        )


def _ensure_akshare():
    """延迟导入 akshare，缺失时给出明确提示"""
    try:
        import akshare as ak  # noqa: F401
        return ak
    except ImportError:
        raise ImportError(
            "akshare 未安装，请运行: pip install akshare"
        )


def _symbol_to_ef_code(symbol: str, market: str) -> str:
    """将 6 位代码 + 市场转为 efinance 格式"""
    if market.upper() == "SH":
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _fetch_via_efinance(symbol: str, market: str,
                        start: str, end: str) -> pd.DataFrame:
    """通过 efinance 获取日线数据"""
    ef = _ensure_efinance()
    code = _symbol_to_ef_code(symbol, market)
    df = ef.stock.get_quote_history(code, beg=start, end=end)
    if df is None or df.empty:
        raise ValueError(f"efinance 返回空数据: {code}")

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["open", "close", "high", "low", "volume", "amount"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _fetch_via_akshare(symbol: str, market: str,
                       start: str, end: str) -> pd.DataFrame:
    """通过 akshare 获取日线数据（备用）"""
    ak = _ensure_akshare()
    raw_start = start.replace("-", "")
    raw_end = end.replace("-", "")

    if market.upper() == "SH":
        full = f"sh{symbol}"
    else:
        full = f"sz{symbol}"

    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=raw_start, end_date=raw_end,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare 返回空数据: {full}")

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _cache_path(symbol: str, market: str, start: str, end: str) -> Path:
    key = f"{symbol}_{market}_{start}_{end}".replace("-", "")
    return CACHE_DIR / f"{key}.parquet"


def get_stock_data(symbol: str, market: str = "SZ",
                   start: str = "2022-01-01", end: str = "2025-12-31",
                   force_refresh: bool = False) -> pd.DataFrame:
    """
    获取单只股票/ETF 的日线数据，优先缓存。

    Args:
        symbol: 6 位代码 (如 "002036")
        market: "SZ" 或 "SH"
        start: 起始日期 "YYYY-MM-DD"
        end: 结束日期 "YYYY-MM-DD"
        force_refresh: 强制重新下载

    Returns:
        DataFrame with columns: date, open, close, high, low, volume, amount
    """
    cache_file = _cache_path(symbol, market, start, end)

    if not force_refresh and cache_file.exists():
        df = pd.read_parquet(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 尝试 efinance → akshare 回退
    df = None
    for fetcher, name in [(_fetch_via_efinance, "efinance"),
                           (_fetch_via_akshare, "akshare")]:
        try:
            df = fetcher(symbol, market, start, end)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  [{name}] 获取失败: {e}")

    if df is None or df.empty:
        raise RuntimeError(
            f"无法获取 {symbol}.{market} 的日线数据，"
            f"efinance 和 akshare 均失败"
        )

    df.to_parquet(cache_file, index=False)
    return df


def get_multi_stock_data(
    symbols: List[Tuple[str, str]],
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> dict:
    """
    批量获取多只股票数据。

    Args:
        symbols: [(code, market), ...] 如 [("002036","SZ"), ("510300","SH")]
        start/end: 日期范围
        force_refresh: 强制刷新

    Returns:
        {code: DataFrame} 字典
    """
    result = {}
    for symbol, market in symbols:
        try:
            df = get_stock_data(symbol, market, start, end, force_refresh)
            result[symbol] = df
        except Exception as e:
            print(f"  [{symbol}.{market}] 跳过: {e}")
    return result


def get_etf_pool(codes: Optional[List[str]] = None,
                 start: str = "2022-01-01", end: str = "2025-12-31",
                 force_refresh: bool = False) -> dict:
    """
    获取 ETF 池数据。

    Args:
        codes: ETF 代码列表，None 则用默认池
        start/end: 日期范围
        force_refresh: 强制刷新

    Returns:
        {code: DataFrame} 字典
    """
    from .config import DEFAULT_ETF_POOL
    pool = codes or DEFAULT_ETF_POOL
    symbols = [(c, "SZ" if c.startswith("1") else "SH") for c in pool]
    return get_multi_stock_data(symbols, start, end, force_refresh)
