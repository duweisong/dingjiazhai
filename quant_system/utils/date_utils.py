"""
A-share trading calendar utilities.

Handles business day calculations with awareness of Chinese market holidays.
Uses akshare to fetch the official trading calendar.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd


class TradingCalendar:
    """A-share trading calendar with holiday awareness."""

    _instance: Optional["TradingCalendar"] = None
    _trading_days: Optional[pd.DatetimeIndex] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _fetch_calendar(self) -> pd.DatetimeIndex:
        """Fetch A-share trading calendar from akshare."""
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            trading_days = pd.to_datetime(df[df["trade_date"].notna()]["trade_date"])
            trading_days = trading_days.sort_values().reset_index(drop=True)
            return pd.DatetimeIndex(trading_days)
        except Exception:
            # Fallback: use pandas business days (ignores holidays)
            return pd.bdate_range("2010-01-01", pd.Timestamp.now(), freq="C", holidays=[])

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """Get all trading days (lazy-loaded and cached)."""
        if self._trading_days is None:
            self._trading_days = self._fetch_calendar()
        return self._trading_days

    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """Check if a date is a trading day."""
        date = pd.Timestamp(date).normalize()
        return date in self.trading_days

    def next_trading_day(self, date: pd.Timestamp, n: int = 1) -> pd.Timestamp:
        """Get the n-th next trading day."""
        date = pd.Timestamp(date).normalize()
        idx = self.trading_days.searchsorted(date, side="right")
        target_idx = min(idx + n - 1, len(self.trading_days) - 1)
        return self.trading_days[max(0, target_idx)]

    def prev_trading_day(self, date: pd.Timestamp, n: int = 1) -> pd.Timestamp:
        """Get the n-th previous trading day."""
        date = pd.Timestamp(date).normalize()
        idx = self.trading_days.searchsorted(date, side="left")
        target_idx = max(idx - n, 0)
        return self.trading_days[target_idx]

    def trading_days_between(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Get all trading days in [start, end]."""
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        mask = (self.trading_days >= start) & (self.trading_days <= end)
        return self.trading_days[mask]

    def n_trading_days_between(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> int:
        """Count trading days between start and end (inclusive)."""
        return len(self.trading_days_between(start, end))

    def trading_days_per_year(self) -> int:
        """Approximate trading days per year."""
        return 244  # Standard for A-shares

    def last_n_trading_days(self, n: int) -> pd.DatetimeIndex:
        """Get the last N trading days up to today."""
        today = pd.Timestamp.now().normalize()
        idx = self.trading_days.searchsorted(today, side="right")
        start_idx = max(0, idx - n)
        return self.trading_days[start_idx:idx]

    def get_week_trading_days(self, date: pd.Timestamp) -> pd.DatetimeIndex:
        """Get all trading days in the week containing `date`."""
        date = pd.Timestamp(date).normalize()
        monday = date - timedelta(days=date.dayofweek)
        friday = monday + timedelta(days=4)
        return self.trading_days_between(monday, friday)

    def is_weekly_rebalance_day(self, date: pd.Timestamp) -> bool:
        """Check if date is the last trading day of the week (typically Thursday check, Friday execution)."""
        date = pd.Timestamp(date).normalize()
        if not self.is_trading_day(date):
            return False
        # Friday or last trading day of the week
        week_days = self.get_week_trading_days(date)
        return date == week_days[-1]

    def refresh(self):
        """Force refresh the trading calendar cache."""
        self._trading_days = self._fetch_calendar()
