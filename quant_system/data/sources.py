"""
Unified multi-source data layer.

Provides a single interface to all A-share data sources with automatic
fallback. The router tries sources in priority order and returns the
first successful result.

Sources:
  baostock  — Free, TCP-based, no SSL issues. Daily K-line, fundamentals, indices.
  tushare   — Token-based, comprehensive (fundamentals, factors, indices, funds).
  akshare   — Wide coverage via web scraping (East Money, Sina, etc.). Needs curl fix.
  efinance  — East Money direct API. Fast when reachable.

Architecture:
  DataSource (ABC)
    ├── BaostockSource
    ├── TushareSource
    ├── AkShareSource
    └── EfinanceSource

  DataRouter(fallback_chain) → single fetch_daily() / fetch_stock_list()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ── Abstract Data Source ──────────────────────────────────────

class DataSource(ABC):
    """Abstract interface for an A-share data source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the source is configured and reachable."""
        ...

    @abstractmethod
    def fetch_daily(
        self, code: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data.

        Args:
            code: Stock code with exchange suffix (e.g. 'sz.000001' or '000001.SZ').
            start: Start date 'YYYY-MM-DD'.
            end: End date 'YYYY-MM-DD'.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount.
            Returns None on failure.
        """
        ...

    def fetch_stock_list(self, index: str) -> Optional[List[str]]:
        """Fetch constituent stocks of an index.

        Args:
            index: 'hs300' | 'csi500' | 'all'.

        Returns:
            List of stock codes with exchange suffix.
        """
        return None  # Optional — not all sources support this

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard OHLCV column names."""
        col_map = {
            "trade_date": "date",
            "ts_code": "code",
            "vol": "volume",
            "amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df


# ── Baostock Source ───────────────────────────────────────────

class BaostockSource(DataSource):
    """Baostock data source — free, TCP-based, reliable.

    Pros: No HTTP/SSL issues, free, fast, good historical coverage.
    Cons: No real-time data, limited index constituents API.
    """

    def __init__(self):
        self._logged_in = False

    @property
    def name(self) -> str:
        return "baostock"

    def is_available(self) -> bool:
        try:
            import baostock as bs
            return True
        except ImportError:
            return False

    def _ensure_login(self):
        if self._logged_in:
            return
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        self._logged_in = True

    def _logout(self):
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False

    def _convert_code(self, code: str) -> str:
        """Convert '000001.SZ' → 'sz.000001'."""
        code = code.strip()
        if "." in code:
            parts = code.split(".")
            num, mkt = parts[0].zfill(6), parts[1].lower()
            if mkt in ("sh", "sz"):
                return f"{mkt}.{num}"
        # Guess: 6xxxxx → SH, others → SZ
        num = code.zfill(6)
        if num.startswith(("6", "9")):
            return f"sh.{num}"
        return f"sz.{num}"

    def _convert_code_back(self, code: str) -> str:
        """Convert 'sz.000001' → '000001.SZ'."""
        if "." in code:
            parts = code.split(".")
            return f"{parts[1]}.{parts[0].upper()}"
        return code

    def fetch_daily(
        self, code: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            self._ensure_login()
            import baostock as bs

            bs_code = self._convert_code(code)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="2",  # 前复权
            )

            if rs.error_code != "0":
                logger.warning(f"baostock query failed for {code}: {rs.error_msg}")
                return None

            data = []
            while rs.next():
                data.append(rs.get_row_data())

            if not data:
                return None

            df = pd.DataFrame(data, columns=rs.fields)
            df = self._normalize_columns(df)

            if len(df) < 10:
                return None

            logger.debug(f"baostock: {code} → {len(df)} rows")
            return df

        except Exception as e:
            logger.warning(f"baostock fetch_daily({code}) failed: {e}")
            return None

    def fetch_stock_list(self, index: str = "hs300") -> Optional[List[str]]:
        """Fetch HS300/CSI500 constituents via baostock."""
        try:
            self._ensure_login()
            import baostock as bs

            index_map = {"hs300": "sh.000300", "csi500": "sh.000905"}
            bs_idx = index_map.get(index.lower())
            if not bs_idx:
                return None

            rs = bs.query_stock_industry()
            if rs.error_code != "0":
                return None

            # baostock doesn't directly give index constituents,
            # but we can get all stocks and filter by market cap
            rs_all = bs.query_stock_basic()
            codes = []
            while rs_all.next():
                row = rs_all.get_row_data()
                codes.append(self._convert_code_back(f"{row[1]}.{row[0]}"))

            return codes[:300] if codes else None

        except Exception as e:
            logger.warning(f"baostock stock_list failed: {e}")
            return None


# ── Tushare Source ────────────────────────────────────────────

class TushareSource(DataSource):
    """Tushare data source — token-based, comprehensive.

    Pros: Rich fundamentals, factor data, indices, funds. Clean API.
    Cons: Requires registration + token. Rate-limited on free tier.
    """

    def __init__(self, token: str = ""):
        self.token = token
        self._pro = None

    @property
    def name(self) -> str:
        return "tushare"

    def is_available(self) -> bool:
        if not self.token:
            return False
        try:
            import tushare as ts
            return True
        except ImportError:
            return False

    def _get_pro(self):
        if self._pro is None:
            import tushare as ts
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

    def _convert_code(self, code: str) -> str:
        """Convert '000001.SZ' → '000001.SZ' (tushare uses same format)."""
        code = code.strip()
        if "." not in code:
            code = code.zfill(6)
            if code.startswith(("6", "9")):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"
        return code

    def fetch_daily(
        self, code: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            pro = self._get_pro()
            ts_code = self._convert_code(code)
            start_fmt = start.replace("-", "")
            end_fmt = end.replace("-", "")

            df = pro.daily(
                ts_code=ts_code,
                start_date=start_fmt,
                end_date=end_fmt,
                fields="trade_date,open,high,low,close,vol,amount",
            )

            if df is None or len(df) == 0:
                return None

            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume",
            })
            df = self._normalize_columns(df)
            df = df.sort_values("date")

            logger.debug(f"tushare: {code} → {len(df)} rows")
            return df

        except Exception as e:
            logger.warning(f"tushare fetch_daily({code}) failed: {e}")
            return None

    def fetch_stock_list(self, index: str = "hs300") -> Optional[List[str]]:
        """Fetch HS300/CSI500 constituents."""
        try:
            pro = self._get_pro()

            idx_map = {"hs300": "000300.SH", "csi500": "000905.SH"}
            ts_idx = idx_map.get(index.lower())
            if not ts_idx:
                return None

            df = pro.index_weight(index_code=ts_idx)
            if df is None or len(df) == 0:
                return None

            codes = df["con_code"].tolist()
            return codes

        except Exception as e:
            logger.warning(f"tushare stock_list({index}) failed: {e}")
            return None


# ── AkShare Source ────────────────────────────────────────────

class AkShareSource(DataSource):
    """AkShare data source — web scraping aggregator.

    Pros: Widest coverage, no token needed, many APIs.
    Cons: Can be slow, may break on website changes, SSL issues with East Money.
    """

    @property
    def name(self) -> str:
        return "akshare"

    def is_available(self) -> bool:
        try:
            import akshare as ak
            return True
        except ImportError:
            return False

    def fetch_daily(
        self, code: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak

            # Try curl-patched akshare first, fall back to normal
            try:
                from akshare_curl_fix import patch_akshare
                patch_akshare()
            except Exception:
                pass

            # Convert code format for akshare
            raw = code.split(".")[0].zfill(6)

            df = ak.stock_zh_a_hist(
                symbol=raw,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )

            if df is None or len(df) == 0:
                return None

            # akshare returns Chinese columns — standardize
            col_map = {
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
            }
            df = df.rename(columns=col_map)
            df = self._normalize_columns(df)

            logger.debug(f"akshare: {code} → {len(df)} rows")
            return df

        except Exception as e:
            logger.debug(f"akshare fetch_daily({code}) failed: {e}")
            return None


# ── Efinance Source ───────────────────────────────────────────

class EfinanceSource(DataSource):
    """Efinance data source — East Money direct API.

    Pros: Fast, real-time data, clean API.
    Cons: TLS fingerprint blocking, needs curl workaround.
    """

    @property
    def name(self) -> str:
        return "efinance"

    def is_available(self) -> bool:
        try:
            import efinance as ef
            return True
        except ImportError:
            return False

    def fetch_daily(
        self, code: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            import efinance as ef

            raw = code.split(".")[0].zfill(6)
            df = ef.stock.get_quote_history(raw, beg=start, end=end)

            if df is None or len(df) == 0:
                return None

            col_map = {
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
            }
            for cn, en in col_map.items():
                if cn in df.columns:
                    df = df.rename(columns={cn: en})

            df = self._normalize_columns(df)

            logger.debug(f"efinance: {code} → {len(df)} rows")
            return df

        except Exception as e:
            logger.debug(f"efinance fetch_daily({code}) failed: {e}")
            return None


# ── Data Router ───────────────────────────────────────────────

@dataclass
class FetchResult:
    """Result from a data fetch attempt."""
    source: str
    data: Optional[pd.DataFrame]
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.data is not None and len(self.data) > 0


class DataRouter:
    """Routes data requests through a fallback chain of sources.

    Usage:
        router = DataRouter()
        router.add_source(BaostockSource(), priority=1)
        router.add_source(AkShareSource(), priority=2)
        df = router.fetch_daily('000001.SZ', '2024-01-01', '2024-12-31')
    """

    def __init__(self):
        self._sources: List[Tuple[int, DataSource]] = []

    def add_source(self, source: DataSource, priority: int = 10):
        """Register a data source with priority (lower = tried first)."""
        self._sources.append((priority, source))
        self._sources.sort(key=lambda x: x[0])
        logger.info(f"DataRouter: registered {source.name} (priority={priority})")

    @property
    def sources(self) -> List[DataSource]:
        return [s for _, s in self._sources]

    def fetch_daily(
        self, code: str, start: str, end: str, timeout_per_source: float = 15.0
    ) -> FetchResult:
        """Fetch daily data, trying sources in priority order.

        Returns the first successful result.
        """
        for priority, source in self._sources:
            if not source.is_available():
                continue

            t0 = time.time()
            try:
                df = source.fetch_daily(code, start, end)
                latency = (time.time() - t0) * 1000

                if df is not None and len(df) >= 10:
                    return FetchResult(
                        source=source.name,
                        data=df,
                        latency_ms=latency,
                    )
                else:
                    logger.debug(
                        f"{source.name}: {code} returned {len(df) if df is not None else 0} rows, trying next..."
                    )
            except Exception as e:
                logger.debug(f"{source.name}: {code} error: {e}")

        return FetchResult(
            source="none",
            data=None,
            error="All sources failed",
        )

    def fetch_multiple(
        self,
        codes: List[str],
        start: str,
        end: str,
        max_workers: int = 4,
    ) -> Dict[str, FetchResult]:
        """Fetch data for multiple stocks (sequential to avoid rate limits)."""
        results = {}
        n = len(codes)

        for i, code in enumerate(codes):
            result = self.fetch_daily(code, start, end)
            results[code] = result

            if result.success:
                logger.debug(
                    f"[{i+1}/{n}] {code} → {result.source} "
                    f"({len(result.data)} rows, {result.latency_ms:.0f}ms)"
                )
            else:
                logger.warning(f"[{i+1}/{n}] {code} → FAILED ({result.error})")

            # Rate limiting between requests
            if i < n - 1:
                time.sleep(0.1)

        return results

    def fetch_stock_list(self, index: str = "hs300") -> Optional[List[str]]:
        """Fetch index constituents, trying each source."""
        for _, source in self._sources:
            if not source.is_available():
                continue
            codes = source.fetch_stock_list(index)
            if codes:
                return codes
        return None

    def status(self) -> str:
        """Report which sources are available."""
        lines = ["DataRouter Status:"]
        for priority, source in self._sources:
            status = "READY" if source.is_available() else "OFFLINE"
            lines.append(f"  [{priority}] {source.name}: {status}")
        return "\n".join(lines)


# ── Pre-configured Router ─────────────────────────────────────

def create_default_router(tushare_token: str = "") -> DataRouter:
    """Create a DataRouter with the default source priority chain.

    Priority:
      1. baostock — free, reliable, TCP-based (no SSL issues)
      2. tushare — comprehensive, token-based
      3. akshare — wide coverage, web-scraping
      4. efinance — East Money direct
    """
    router = DataRouter()

    # 1. Baostock first — most reliable in restricted network environments
    router.add_source(BaostockSource(), priority=1)

    # 2. Tushare — premium features
    if tushare_token:
        router.add_source(TushareSource(token=tushare_token), priority=2)

    # 3. AkShare — wide coverage
    router.add_source(AkShareSource(), priority=3)

    # 4. Efinance — last resort
    router.add_source(EfinanceSource(), priority=4)

    return router
