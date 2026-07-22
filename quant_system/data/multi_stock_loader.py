"""
Multi-stock data loader.

Batch-loads OHLCV data and computes technical indicators for an entire
stock pool. Uses the unified DataRouter for multi-source fallback,
and backtest.data_loader.compute_indicators() for indicator computation.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time

import pandas as pd
import numpy as np

from ..utils.types import MultiStockData, StockPool
from ..utils.logger import get_logger
from ..config import get_config, GlobalConfig
from .sources import DataRouter, create_default_router, BaostockSource, AkShareSource, EfinanceSource

# Reuse indicator computation from existing backtest
import sys
_BT = Path(__file__).parent.parent.parent / "backtest"
if str(_BT) not in sys.path:
    sys.path.insert(0, str(_BT))
from data_loader import compute_indicators

logger = get_logger(__name__)


class MultiStockLoader:
    """Batch data loader for multiple stocks.

    Fetches OHLCV data, computes technical indicators, and manages
    a Parquet cache per stock.
    """

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        cache_dir: Optional[Path] = None,
        router: Optional[DataRouter] = None,
    ):
        self.config = config or get_config()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.config.data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up data router with fallback chain
        if router is not None:
            self.router = router
        else:
            self.router = create_default_router(
                tushare_token=self.config.data_source.tushare_token
            )

    def load(
        self,
        pool: StockPool,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_stocks: Optional[int] = None,
        refresh: bool = False,
    ) -> MultiStockData:
        """Load OHLCV + indicators for all stocks in the pool.

        Args:
            pool: Stock pool to load.
            start_date: Start date (YYYY-MM-DD), defaults to config.
            end_date: End date (YYYY-MM-DD), defaults to config.
            max_stocks: Limit number of stocks (for quick testing).
            refresh: Force re-download, bypass cache.

        Returns:
            MultiStockData with prices and indicators.
        """
        start_date = start_date or self.config.backtest.start_date
        end_date = end_date or self.config.backtest.end_date

        codes = pool.codes[:max_stocks] if max_stocks else pool.codes
        n = len(codes)
        logger.info(f"Loading data for {n} stocks ({start_date} ~ {end_date})...")

        prices: Dict[str, pd.DataFrame] = {}
        indicators: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for i, code in enumerate(codes):
            try:
                p, ind = self._load_single(code, start_date, end_date, refresh)
                if p is not None and len(p) > 0:
                    prices[code] = p
                    indicators[code] = ind
                else:
                    failed.append(code)
            except Exception as e:
                logger.warning(f"Failed to load {code}: {e}")
                failed.append(code)

            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{n} loaded, {len(failed)} failed")

        # Align all dataframes to a common date index
        if prices:
            common_dates = self._find_common_dates(list(prices.values()))
            for code in prices:
                prices[code] = prices[code].reindex(common_dates)
                indicators[code] = indicators[code].reindex(common_dates)

        logger.info(
            f"Loaded {len(prices)}/{n} stocks successfully, "
            f"{len(failed)} failed, "
            f"{len(common_dates) if prices else 0} common trading days"
        )

        return MultiStockData(
            prices=prices,
            indicators=indicators,
            dates=common_dates if prices else pd.DatetimeIndex([]),
            codes=list(prices.keys()),
        )

    def _load_single(
        self, code: str, start_date: str, end_date: str, refresh: bool
    ) -> tuple:
        """Load data for a single stock via DataRouter with cache."""
        cache_file = self.cache_dir / f"{code}_{start_date}_{end_date}.parquet"

        # Use cache if available and fresh
        if not refresh and cache_file.exists():
            df = pd.read_parquet(cache_file)
            price_cols = ["open", "high", "low", "close", "volume"]
            indicator_cols = [c for c in df.columns if c not in price_cols]
            # Only return indicator cols that exist
            ind = df[[c for c in indicator_cols if c in df.columns]].copy() if indicator_cols else pd.DataFrame(index=df.index)
            return df[price_cols].copy(), ind

        # Fetch via DataRouter (baostock → akshare → efinance)
        result = self.router.fetch_daily(code, start_date, end_date)

        if not result.success:
            logger.warning(f"  {code}: all data sources failed")
            return None, None

        raw = result.data
        logger.debug(f"  {code} ← {result.source} ({len(raw)} rows, {result.latency_ms:.0f}ms)")

        if raw is None or len(raw) < 10:
            return None, None

        # Ensure 'date' column for compute_indicators
        if "date" not in raw.columns and isinstance(raw.index, pd.DatetimeIndex):
            raw = raw.reset_index()

        # Compute indicators
        try:
            df = compute_indicators(raw)
        except Exception as e:
            logger.warning(f"  {code}: indicator computation failed: {e}")
            # Still return raw data without indicators
            df = raw.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        # Cache
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass

        price_cols = ["open", "high", "low", "close", "volume"]
        indicator_cols = [c for c in df.columns if c not in price_cols]
        ind = df[[c for c in indicator_cols if c in df.columns]].copy() if indicator_cols else pd.DataFrame(index=df.index)
        return df[price_cols].copy(), ind

    def _find_common_dates(
        self, dataframes: List[pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """Find the intersection of dates across all DataFrames."""
        if not dataframes:
            return pd.DatetimeIndex([])
        common = dataframes[0].index
        for df in dataframes[1:]:
            common = common.intersection(df.index)
        return pd.DatetimeIndex(common.sort_values())

    def get_price_matrix(
        self, data: MultiStockData, field: str = "close"
    ) -> pd.DataFrame:
        """Extract a (dates x codes) matrix for a given price field.

        Args:
            data: MultiStockData container.
            field: "open" | "high" | "low" | "close" | "volume"

        Returns:
            DataFrame with dates as index and stock codes as columns.
        """
        matrix = {}
        for code, df in data.prices.items():
            if field in df.columns:
                matrix[code] = df[field]
        result = pd.DataFrame(matrix, index=data.dates)
        return result

    def get_return_matrix(self, data: MultiStockData) -> pd.DataFrame:
        """Compute daily returns matrix (dates x codes)."""
        close = self.get_price_matrix(data, "close")
        return close.pct_change().dropna()

    def check_data_quality(self, data: MultiStockData) -> Dict:
        """Run data quality checks and return a report.

        Checks: missing dates, NaN percentage, zero-volume days,
        extreme price jumps, stale prices (suspension detection).
        """
        report = {
            "n_stocks": data.n_stocks,
            "n_dates": data.n_dates,
            "missing_dates_pct": {},
            "nan_pct": {},
            "suspension_days": {},
            "extreme_jumps": {},
        }

        for code in data.codes:
            df = data.prices.get(code)
            if df is None:
                continue

            # NaN percentage
            nan_pct = df["close"].isna().mean()
            report["nan_pct"][code] = round(nan_pct, 4)

            # Suspension detection (consecutive zero returns)
            rets = df["close"].pct_change().fillna(0)
            zero_runs = (rets == 0).astype(int).groupby(
                (rets != 0).cumsum()
            ).cumsum()
            max_suspend = zero_runs.max()
            if max_suspend > 3:
                report["suspension_days"][code] = int(max_suspend)

            # Extreme jumps (>20% in a day)
            jumps = (rets.abs() > 0.20).sum()
            if jumps > 0:
                report["extreme_jumps"][code] = int(jumps)

        # Summary
        high_nan = {k: v for k, v in report["nan_pct"].items() if v > 0.1}
        total_suspended = len(report["suspension_days"])
        total_jumps = sum(report["extreme_jumps"].values())

        report["summary"] = (
            f"Quality check: {data.n_stocks} stocks, {data.n_dates} days. "
            f"High NaN (>10%): {len(high_nan)} stocks. "
            f"Suspensions (>3d): {total_suspended} stocks. "
            f"Extreme jumps (>20%): {total_jumps} events."
        )
        return report
