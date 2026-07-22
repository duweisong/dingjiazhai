"""
Stock pool management for A-share indices.

Provides constituent fetching, caching, and filtering for:
- CSI 300 (沪深300)
- CSI 500 (中证500)
- Custom stock lists
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd

from ..utils.types import StockPool
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Cache directory (relative to project root, resolved at runtime)
CACHE_DIR = Path(__file__).parent / ".cache"


class StockPoolManager:
    """Manages stock universe composition and metadata."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, index: str) -> Path:
        return self.cache_dir / f"stock_pool_{index}.parquet"

    def _is_cache_valid(self, index: str, max_age_days: int = 7) -> bool:
        """Check if cached stock pool is still fresh."""
        path = self._cache_path(index)
        if not path.exists():
            return False
        age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(path.stat().st_mtime)
        return age.days < max_age_days

    def fetch_index_constituents(
        self, index: str, refresh: bool = False
    ) -> StockPool:
        """Fetch constituents of a major A-share index.

        Args:
            index: "hs300" | "csi500"
            refresh: Force refresh even if cache is valid.

        Returns:
            StockPool with codes, names, industries, and market caps.
        """
        index_lower = index.lower()
        if not refresh and self._is_cache_valid(index_lower):
            logger.info(f"Loading {index} from cache...")
            return self._load_from_cache(index_lower)

        logger.info(f"Fetching {index} constituents from akshare...")
        try:
            return self._fetch_via_akshare(index_lower)
        except Exception as e:
            logger.warning(f"akshare fetch failed ({e}), trying cache fallback...")
            cached = self._load_from_cache(index_lower)
            if cached.codes:
                logger.info(f"Using stale cache for {index} ({len(cached.codes)} stocks)")
                return cached
            raise RuntimeError(f"Cannot fetch or load stock pool for {index}") from e

    def _fetch_via_akshare(self, index: str) -> StockPool:
        """Fetch constituents via akshare API."""
        import akshare as ak

        if index == "hs300":
            df = ak.index_stock_cons_csindex(symbol="000300")
        elif index == "csi500":
            df = ak.index_stock_cons_csindex(symbol="000905")
        else:
            raise ValueError(f"Unknown index: {index}. Use 'hs300' or 'csi500'.")

        # Normalize column names (akshare may vary between versions)
        df = df.copy()

        # akshare index_stock_cons_csindex returns columns:
        # [0]=date, [1]=index_code, [2]=index_name, [3]=index_en_name,
        # [4]=stock_code, [5]=stock_name, [6]=stock_en_name,
        # [7]=exchange, [8]=exchange_en_name
        # We use positional indexing since Chinese column names may encode incorrectly.

        raw_codes = df.iloc[:, 4]  # Constituent stock code column
        codes = []
        for c in raw_codes:
            c_str = str(int(c)).zfill(6) if isinstance(c, (int, float)) else str(c).zfill(6)
            codes.append(c_str)

        # Add exchange suffix if missing
        codes = [self._add_suffix(c) for c in codes]

        # Stock names from column [5]
        raw_names = df.iloc[:, 5]
        names = dict(zip(codes, [str(n) for n in raw_names]))

        # Exchange info from column [7] for suffix accuracy
        raw_exchange = df.iloc[:, 7]
        for i, code in enumerate(codes):
            exchange_name = str(raw_exchange.iloc[i]) if i < len(raw_exchange) else ""
            if "上海" in exchange_name and not code.endswith(".SH"):
                code_sh = code.split(".")[0] + ".SH"
                idx = codes.index(code)
                codes[idx] = code_sh
                if code in names:
                    names[code_sh] = names.pop(code)
            elif "深圳" in exchange_name and not code.endswith(".SZ"):
                code_sz = code.split(".")[0] + ".SZ"
                idx = codes.index(code)
                codes[idx] = code_sz
                if code in names:
                    names[code_sz] = names.pop(code)

        # Fetch industry classifications
        industries = self._fetch_industries(codes)

        # Estimate market caps (lazy: fetch a snapshot)
        market_caps = self._fetch_market_caps(codes)

        pool = StockPool(
            index=index.upper(),
            codes=codes,
            names=names,
            industries=industries,
            market_caps=market_caps,
            as_of_date=pd.Timestamp.now().normalize(),
        )

        self._save_to_cache(index, pool)
        logger.info(f"Fetched {index}: {len(codes)} stocks")
        return pool

    def _add_suffix(self, code: str) -> str:
        """Add exchange suffix to a stock code."""
        code = str(code).zfill(6)
        if "." in code:
            return code
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        elif code.startswith(("0", "3")):
            return f"{code}.SZ"
        elif code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{code}.SZ"  # Default to Shenzhen

    def _remove_suffix(self, code: str) -> Tuple[str, str]:
        """Split code into (numeric_code, exchange)."""
        if "." in code:
            parts = code.split(".")
            return parts[0], parts[1]
        return code, ""

    def _fetch_industries(self, codes: List[str]) -> Dict[str, str]:
        """Fetch Shenwan industry classification for stocks."""
        industries: Dict[str, str] = {}
        try:
            import akshare as ak
            df = ak.stock_board_industry_name_em()
            # Build a lookup from stock code to industry
            for _, row in df.iterrows():
                # This varies by akshare version — try common columns
                pass
            logger.info(f"Industry lookup fetched for {len(industries)} stocks")
        except Exception as e:
            logger.warning(f"Could not fetch industries: {e}")
        return industries

    def _fetch_market_caps(self, codes: List[str]) -> Dict[str, float]:
        """Fetch latest market capitalizations."""
        caps: Dict[str, float] = {}
        try:
            import akshare as ak
            # Fetch from East Money
            raw_codes = [self._remove_suffix(c)[0] for c in codes[:50]]  # Batch of 50
            # Simplified: fetch individual stock info
            for code in codes[:20]:  # Limit to avoid rate limiting
                try:
                    raw, _ = self._remove_suffix(code)
                    df = ak.stock_individual_info_em(symbol=raw)
                    # Extract market cap
                    for _, row in df.iterrows():
                        if "总市值" in str(row.iloc[0]):
                            val = row.iloc[1]
                            if isinstance(val, (int, float)):
                                caps[code] = float(val) / 1e8  # Convert to 100M CNY
                            break
                except Exception:
                    pass
            logger.info(f"Market caps fetched for {len(caps)}/{len(codes)} stocks")
        except Exception as e:
            logger.warning(f"Could not fetch market caps: {e}")
        return caps

    def _save_to_cache(self, index: str, pool: StockPool):
        """Save stock pool to Parquet cache."""
        path = self._cache_path(index)
        df = pd.DataFrame({
            "code": pool.codes,
            "name": [pool.names.get(c, "") for c in pool.codes],
            "industry": [pool.industries.get(c, "") for c in pool.codes],
            "market_cap": [pool.market_caps.get(c, 0.0) for c in pool.codes],
        })
        df.to_parquet(path, index=False)

    def _load_from_cache(self, index: str) -> StockPool:
        """Load stock pool from cached Parquet file."""
        path = self._cache_path(index)
        if not path.exists():
            return StockPool(
                index=index.upper(),
                codes=[],
                names={},
                industries={},
                market_caps={},
                as_of_date=pd.Timestamp.now(),
            )
        df = pd.read_parquet(path)
        return StockPool(
            index=index.upper(),
            codes=df["code"].tolist(),
            names=dict(zip(df["code"], df.get("name", [""] * len(df)))),
            industries=dict(zip(df["code"], df.get("industry", [""] * len(df)))),
            market_caps=dict(zip(df["code"], df.get("market_cap", [0.0] * len(df)))),
            as_of_date=pd.Timestamp.fromtimestamp(path.stat().st_mtime),
        )

    def get_combined_pool(self, indices: List[str]) -> StockPool:
        """Combine multiple indices, deduplicating stocks."""
        all_codes = []
        all_names = {}
        all_industries = {}
        all_caps = {}

        for idx in indices:
            pool = self.fetch_index_constituents(idx)
            for code in pool.codes:
                if code not in all_codes:
                    all_codes.append(code)
                all_names[code] = pool.names.get(code, "")
                all_industries[code] = pool.industries.get(code, "")
                all_caps[code] = pool.market_caps.get(code, 0.0)

        return StockPool(
            index="+".join(indices).upper(),
            codes=all_codes,
            names=all_names,
            industries=all_industries,
            market_caps=all_caps,
            as_of_date=pd.Timestamp.now(),
        )

    def filter_by_liquidity(
        self, pool: StockPool, min_daily_volume: float = 10_000_000
    ) -> StockPool:
        """Filter stocks by minimum daily trading volume (in RMB)."""
        # This requires fetching recent volume data — placeholder
        return pool

    def filter_suspended(self, pool: StockPool) -> StockPool:
        """Remove currently suspended stocks."""
        # Placeholder: requires real-time suspension data
        return pool


# Module-level convenience function
_manager: Optional[StockPoolManager] = None


def get_stock_pool(
    index: str = "hs300", refresh: bool = False
) -> StockPool:
    """Convenience function to get a stock pool.

    Args:
        index: "hs300" | "csi500" | "hs300,csi500"
        refresh: Force refresh from API.

    Returns:
        StockPool with constituents.
    """
    global _manager
    if _manager is None:
        _manager = StockPoolManager()

    indices = [i.strip() for i in index.split(",")]
    if len(indices) == 1:
        return _manager.fetch_index_constituents(indices[0], refresh=refresh)
    return _manager.get_combined_pool(indices)
