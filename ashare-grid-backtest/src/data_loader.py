"""
A股网格交易回测 — 数据加载模块

三数据源，按优先级自动回退:
  1. baostock — 无 SSL 烦恼，原生 socket 协议，最稳定
  2. akshare (curl 补丁) — curl TLS 指纹绕过东方财富 CDN 检测
  3. efinance — 简单场景备选

所有数据统一转为标准列名: date, open, high, low, close, volume, amount
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 确保项目根在路径（curl fix 需要）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# baostock 在中文 locale 下日期解析不稳定 → 提前 import + 英文 locale
if "LANG" not in os.environ:
    os.environ["LANG"] = "en_US.UTF-8"
import baostock as _bs  # noqa: E402 — 必须在 LANG 设置后 import

# ============================================================
# 数据源 1: baostock
# ============================================================

def _bs_login():
    """baostock 登录，失败时重试一次"""
    for attempt in range(2):
        lg = _bs.login()
        if lg.error_code == "0":
            return
        if attempt == 0:
            logger.warning("baostock 登录重试 (%s)", lg.error_msg)
    raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")


def _bs_symbol(symbol: str, market: str) -> str:
    """002036, SZ → sz.002036"""
    return f"{market.lower()}.{symbol}"


def _fetch_via_baostock(
    symbol: str, market: str, start: str, end: str
) -> pd.DataFrame:
    """通过 baostock 获取日线数据（前复权）"""
    _bs_login()
    bs_code = _bs_symbol(symbol, market)

    fields = "date,open,high,low,close,volume,amount"
    rs = _bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        frequency="d",
        adjustflag="2",  # 前复权
    )

    if rs is None:
        raise RuntimeError(
            f"baostock 查询返回空 (日期范围: {start}~{end})，"
            f"请检查日期格式或股票代码"
        )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise ValueError(f"baostock {bs_code} 在 {start}~{end} 无数据")

    df = pd.DataFrame(rows, columns=fields.split(","))
    df.columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    df["date"] = pd.to_datetime(df["date"])

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"baostock {bs_code} 清洗后无有效数据")

    logger.info("baostock: %d 条 (%s~%s)", len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ============================================================
# 数据源 2: akshare (curl 补丁)
# ============================================================

_CURL_PATCHED = False


def _ensure_curl_patch():
    global _CURL_PATCHED
    if _CURL_PATCHED:
        return
    try:
        from akshare_curl_fix import patch_akshare
        patch_akshare()
        _CURL_PATCHED = True
        logger.info("akshare curl 补丁已加载")
    except ImportError:
        logger.warning("akshare_curl_fix 未找到，akshare 可能因 TLS 指纹被拦截")


def _fetch_via_akshare(
    symbol: str, market: str, start: str, end: str
) -> pd.DataFrame:
    """通过 akshare (curl 补丁) 获取日线数据"""
    _ensure_curl_patch()
    import akshare as ak

    raw_start = start.replace("-", "")
    raw_end = end.replace("-", "")

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=raw_start,
        end_date=raw_end,
        adjust="qfq",
    )

    if df is None or df.empty:
        raise ValueError(f"akshare 返回空数据: {symbol}")

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("akshare(curl): %d 条 (%s~%s)", len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ============================================================
# 数据源 3: efinance (备选)
# ============================================================

def _fetch_via_efinance(
    symbol: str, market: str, start: str, end: str
) -> pd.DataFrame:
    """通过 efinance 获取日线数据"""
    import efinance as ef

    suffix = "SH" if market.upper() == "SH" else "SZ"
    code = f"{symbol}.{suffix}"

    df = ef.stock.get_quote_history(code, beg=start, end=end)

    if df is None or df.empty:
        raise ValueError(f"efinance 返回空数据: {code}")

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "换手率": "turnover",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("efinance: %d 条 (%s~%s)", len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ============================================================
# 缓存工具
# ============================================================

def _cache_path(symbol: str, market: str, start: str, end: str) -> Path:
    key = f"{symbol}_{market}_{start}_{end}".replace("-", "")
    return CACHE_DIR / f"{key}.parquet"


def _find_superset_cache(symbol: str, market: str, start: str, end: str) -> Path | None:
    """查找覆盖请求区间的已有缓存"""
    prefix = f"{symbol}_{market}_"
    for f in CACHE_DIR.glob(f"{prefix}*.parquet"):
        try:
            stem = f.stem
            parts = stem[len(prefix):].split("_")
            cached_start = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]}"
            cached_end = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
            if cached_start <= start and cached_end >= end:
                return f
        except (ValueError, IndexError):
            continue
    return None


# ============================================================
# 主入口
# ============================================================

def get_stock_data(
    symbol: str,
    market: str = "SZ",
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    获取 A 股日线数据，优先缓存。

    数据源优先级: 缓存 > baostock > akshare(curl) > efinance

    Args:
        symbol: 6 位代码
        market: "SZ" | "SH"
        start / end: "YYYY-MM-DD"
        force_refresh: 跳过缓存强制重新下载

    Returns:
        DataFrame: date, open, high, low, close, volume, amount
    """
    cache_file = _cache_path(symbol, market, start, end)

    # 1) 精准缓存命中
    if not force_refresh and cache_file.exists():
        df = pd.read_parquet(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("缓存命中: %s (%d 条)", cache_file.name, len(df))
        return df

    # 2) 大区间缓存裁剪
    if not force_refresh:
        superset = _find_superset_cache(symbol, market, start, end)
        if superset is not None:
            df = pd.read_parquet(superset)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            logger.info("缓存裁剪: %s → %d 条", superset.name, len(df))
            return df

    # 3) 远程获取 — 按优先级尝试
    fetchers = [
        ("baostock", lambda: _fetch_via_baostock(symbol, market, start, end)),
        ("akshare(curl)", lambda: _fetch_via_akshare(symbol, market, start, end)),
        ("efinance", lambda: _fetch_via_efinance(symbol, market, start, end)),
    ]

    df = None
    last_error = None
    for name, fetcher in fetchers:
        try:
            logger.info("尝试 %s: %s.%s (%s~%s)", name, symbol, market, start, end)
            df = fetcher()
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_error = e
            logger.warning("%s 失败: %s", name, e)

    if df is None or df.empty:
        raise RuntimeError(
            f"无法获取 {symbol}.{market} 的日线数据。\n"
            f"  已尝试: baostock, akshare(curl), efinance\n"
            f"  最后错误: {last_error}"
        )

    # 保存缓存
    df.to_parquet(cache_file, index=False)
    logger.info("已缓存: %s (%d 条)", cache_file.name, len(df))
    return df
