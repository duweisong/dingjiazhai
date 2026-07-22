"""
缓存数据适配器 — 从已有 .cache/etf_sector/ parquet 文件加载 ETF 数据，
转换为网格引擎所需的 OHLCV 格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ETF 代码 → (名称, 行业) 映射
SECTOR_ETFS: Dict[str, tuple] = {
    '159993': ('证券ETF鹏华', '金融'),
    '512880': ('证券ETF国泰', '金融'),
    '512800': ('银行ETF华宝', '金融'),
    '512660': ('军工ETF易方达', '国防军工'),
    '512670': ('国防ETF鹏华', '国防军工'),
    '512710': ('军工龙头ETF', '国防军工'),
    '515880': ('通信ETF国泰', 'TMT'),
    '512760': ('半导体ETF国泰', 'TMT'),
    '159869': ('游戏ETF华夏', 'TMT'),
    '512980': ('传媒ETF广发', 'TMT'),
    '516510': ('云计算ETF易方达', 'TMT'),
    '159995': ('芯片ETF华夏', 'TMT'),
    '515050': ('5GETF华夏', 'TMT'),
    '159852': ('软件ETF易方达', 'TMT'),
    '560860': ('工业有色ETF万家', '周期'),
    '515220': ('煤炭ETF国泰', '周期'),
    '516970': ('建材ETF国泰', '周期'),
    '515210': ('钢铁ETF国泰', '周期'),
    '516780': ('稀土ETF华泰', '周期'),
    '561330': ('矿业ETF国泰', '周期'),
    '159611': ('电力ETF广发', '公用事业'),
    '516160': ('环保ETF易方达', '公用事业'),
    '561170': ('碳中和ETF易方达', '公用事业'),
    '159886': ('机械ETF富国', '制造'),
    '562500': ('机器人ETF华夏', '制造'),
    '159638': ('高端装备ETF嘉实', '制造'),
    '516960': ('工业母机ETF华夏', '制造'),
    '159996': ('家电ETF国泰', '消费'),
    '512690': ('酒ETF鹏华', '消费'),
    '159843': ('食品饮料ETF招商', '消费'),
    '516130': ('旅游ETF华夏', '消费'),
    '159766': ('旅游ETF富国', '消费'),
    '159875': ('新能源ETF嘉实', '新能源'),
    '516390': ('新能源汽车ETF', '新能源'),
    '159857': ('光伏ETF天弘', '新能源'),
    '561910': ('电池ETF易方达', '新能源'),
    '512010': ('医药ETF华夏', '医药'),
    '159647': ('中药ETF鹏华', '医药'),
    '512170': ('医疗ETF华宝', '医药'),
    '159883': ('医疗器械ETF永赢', '医药'),
    '516950': ('基建ETF广发', '基建'),
    '512200': ('房地产ETF华夏', '基建'),
    '159745': ('建材ETF国泰', '基建'),
    '159865': ('养殖ETF国泰', '农业'),
    '159825': ('农业ETF富国', '农业'),
}

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "etf_sector"


def load_etf_price_data(parquet_name: str = "sector_etfs_20240101_20260610.parquet",
                        ) -> pd.DataFrame:
    """
    从缓存加载 ETF 价格矩阵。

    Returns:
        DataFrame: index=date, columns=etf_code, values=close_price
    """
    path = CACHE_DIR / parquet_name
    if not path.exists():
        raise FileNotFoundError(f"缓存文件不存在: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def price_to_ohlcv(close_series: pd.Series,
                   daily_volatility: float = 0.008) -> pd.DataFrame:
    """
    从收盘价序列还原 OHLCV 数据（用于网格穿越检测）。

    使用合理的 ETF 日内波动假设：
    - ETF 日内振幅通常约 0.8%~1.2%（比个股小）
    - high/low 围绕 open/close 极值扩展
    - open 在前一日 close 附近（含小幅跳空）
    """
    df = close_series.reset_index()
    df.columns = ["date", "close"]

    rng = np.random.default_rng(42)
    n = len(df)

    # 日内振幅（约 0.8% 日波动，对 ETF 合理）
    daily_range = df["close"] * daily_volatility

    # open 在前一日 close 基础上加小跳空（±0.3%）
    gap = rng.normal(0, 0.003, n)
    df["open"] = df["close"].shift(1).fillna(df["close"]) * (1 + gap)

    # high 取 max(open, close) + 半幅波动
    # low 取 min(open, close) - 半幅波动
    o_c_max = df[["open", "close"]].max(axis=1)
    o_c_min = df[["open", "close"]].min(axis=1)
    # 日内额外波动（high高于max, low低于min的部分）
    extra_high = daily_range * rng.uniform(0.2, 0.7, n)
    extra_low = daily_range * rng.uniform(0.2, 0.7, n)
    df["high"] = o_c_max + extra_high
    df["low"] = o_c_min - extra_low

    df["volume"] = np.random.randint(5_000_000, 50_000_000, n)
    df["amount"] = df["close"] * df["volume"] / 100

    return df.dropna()


def get_all_etf_data(parquet_name: str = "sector_etfs_20240101_20260610.parquet",
                     min_days: int = 100,
                     ) -> Dict[str, pd.DataFrame]:
    """
    加载所有 ETF 的 OHLCV 数据。

    Args:
        parquet_name: 缓存文件名
        min_days: 最少交易日要求

    Returns:
        {etf_code: ohlcv_DataFrame}
    """
    prices = load_etf_price_data(parquet_name)
    result = {}

    for code in prices.columns:
        close = prices[code].dropna()
        if len(close) < min_days:
            continue
        try:
            ohlcv = price_to_ohlcv(close)
            # 添加元信息
            info = SECTOR_ETFS.get(code, (code, "未知"))
            ohlcv.attrs["name"] = info[0]
            ohlcv.attrs["sector"] = info[1]
            result[code] = ohlcv
        except Exception:
            continue

    return result


def get_sector_summary() -> pd.DataFrame:
    """获取 ETF 行业分布概览"""
    rows = []
    for code, (name, sector) in SECTOR_ETFS.items():
        rows.append({"code": code, "name": name, "sector": sector})
    return pd.DataFrame(rows)
