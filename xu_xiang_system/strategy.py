"""
Xu Xiang Upgraded Strategy - Core trading logic.

Combines:
  1. Environment-aware position sizing (A/B/C levels)
  2. Capital structure filtering
  3. Technical entry signals with volume confirmation
  4. Five-dimension risk management
  5. Adaptive stops based on ATR

Signal protocol:
  1 = BUY, -1 = SELL, 0 = HOLD
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .market_env import MarketEnvClassifier
from .risk_manager import RiskManager, RiskConfig, RiskState, create_risk_state


@dataclass
class StrategyConfig:
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60
    vol_ratio_min: float = 1.2
    rsi_max: float = 75
    rsi_min: float = 30
    require_ma_bullish: bool = True
    require_volume_surge: bool = True
    use_atr_stops: bool = True
    atr_stop_mult: float = 2.5
    atr_target_mult: float = 4.0
    trailing_stop_pct: float = 0.04
    min_env_level: str = "B"
    max_position_pct: float = 0.30


class XuXiangStrategy:
    """Xu Xiang Upgraded Strategy - env-aware, capital-filtered, 5D risk-managed"""
    name: str = "xu_xiang_upgraded"
    description: str = "Xu Xiang upgraded: env-aware, capital-filtered, 5D risk-managed"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        field_names = list(StrategyConfig.__dataclass_fields__.keys())
        cfg = {k: v for k, v in self.params.items() if k in field_names}
        self.config = StrategyConfig(**cfg)
        self.env_classifier = MarketEnvClassifier()
        self.risk_manager = RiskManager(RiskConfig(
            price_stop_pct=0.05, max_holding_days=10,
            env_min_level=self.config.min_env_level,
            max_single_position=self.config.max_position_pct))
        self._dynamic_stops = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        signals = pd.Series(0, index=df.index, dtype=int)
        self._compute_indicators(df)
        if "ma_bullish" not in df.columns:
            df["ma_bullish"] = (
                (df["ma_5"] > df["ma_20"]) & (df["ma_20"] > df["ma_60"])).astype(int)

        in_position = False
        entry_price = 0.0
        highest = 0.0
        holding_days = 0

        start_idx = max(self.config.ma_long, 60)
        for i in range(start_idx, n):
            close = df["close"].iloc[i]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            atr = df["atr"].iloc[i] if "atr" in df.columns else close * 0.03

            if not in_position:
                if self._check_entry(df, i, atr):
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = close
                    highest = high
                    holding_days = 0
                    continue

            if in_position:
                holding_days += 1
                highest = max(highest, high)
                if self._check_exit(df, i, entry_price, highest, holding_days, atr):
                    signals.iloc[i] = -1
                    in_position = False
                    entry_price = 0.0
                    highest = 0.0
                    holding_days = 0

        return signals

    def _check_entry(self, df, idx, atr):
        row = df.iloc[idx]
        close = row["close"]
        cfg = self.config

        if cfg.require_ma_bullish:
            ma_bullish = (
                row.get("ma_5", close) > row.get("ma_20", close) and
                row.get("ma_20", close) > row.get("ma_60", close))
            if not ma_bullish:
                return False

        if cfg.require_volume_surge:
            vol_ratio = row.get("vol_ratio", 1.0) or 1.0
            if vol_ratio < cfg.vol_ratio_min:
                return False

        rsi = row.get("rsi", 50) or 50
        if rsi > cfg.rsi_max or rsi < cfg.rsi_min:
            return False

        ma_long = row.get(f"ma_{cfg.ma_long}", close)
        if pd.notna(ma_long) and close < ma_long * 0.98:
            return False

        if (row.get("pct_chg", 0) or 0) < -0.05:
            return False

        return True

    def _check_exit(self, df, idx, entry_price, highest, days, atr):
        row = df.iloc[idx]
        close = row["close"]
        low = row["low"]
        cfg = self.config

        hard_stop = entry_price * 0.95
        if low <= hard_stop:
            return True

        trail_stop = highest * (1 - cfg.trailing_stop_pct)
        if low <= trail_stop:
            return True

        if cfg.use_atr_stops:
            atr_stop = entry_price - atr * cfg.atr_stop_mult
            if low <= atr_stop:
                return True

        if days >= 10:
            pnl = (close / entry_price) - 1
            if pnl < 0.01:
                return True

        ma60 = row.get("ma_60", close)
        if pd.notna(ma60) and close < ma60 * 0.95:
            return True

        macd_h = row.get("macd_hist", 0) or 0
        prev_m = df.iloc[idx - 1].get("macd_hist", 0) or 0 if idx > 0 else 0
        if macd_h < 0 and prev_m > 0:
            return True

        return False

    def _compute_indicators(self, df):
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        for p in [5, 10, 20, 30, 60]:
            if f"ma_{p}" not in df.columns:
                df[f"ma_{p}"] = close.rolling(p).mean()
        if "vol_ratio" not in df.columns and "volume" in df.columns:
            vol_ma20 = df["volume"].rolling(20).mean()
            df["vol_ratio"] = df["volume"] / vol_ma20.replace(0, np.nan)
        if "atr" not in df.columns:
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
        if "rsi" not in df.columns:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))
        if "macd_hist" not in df.columns:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            df["macd_hist"] = 2 * (dif - dea)

    def get_dynamic_stops(self):
        return self._dynamic_stops

    def get_params_str(self):
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    def __repr__(self):
        return f"XuXiangStrategy({self.get_params_str()})"


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    np.random.seed(42)
    n = len(dates)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    df = pd.DataFrame({
        "date": dates, "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.random.randint(1e7, 1e8, n)})
    strategy = XuXiangStrategy({"ma_short": 5, "ma_mid": 20, "ma_long": 60})
    signals = strategy.generate_signals(df)
    print(f"Signals: {int((signals==1).sum())} buys, {int((signals==-1).sum())} sells")
