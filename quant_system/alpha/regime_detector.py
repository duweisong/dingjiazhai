"""
Market regime detection.

Identifies distinct market regimes (bull, bear, sideways, high-vol)
to condition strategy behavior. Different factors work in different
regimes — knowing which regime you're in is half the battle.

Methods:
- HMM (Hidden Markov Model): Unsupervised regime classification
- Trend-based: Simple MA-based bull/bear classification
- Volatility-based: High/low vol regime based on VIX-like metrics
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try importing hmmlearn for HMM-based regime detection
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class RegimeDetector:
    """Detects market regimes for adaptive strategy behavior.

    "Momentum works in trending markets. Mean reversion works in
    ranging markets. The trick is knowing which one you're in."
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self._hmm_model = None
        self._regime_labels = None

    def detect_trend(
        self,
        returns: pd.Series,
        short_window: int = 20,
        long_window: int = 60,
    ) -> pd.Series:
        """Simple trend-based regime classification.

        Returns:
            Series with values: "bull", "bear", "neutral" for each date.
        """
        # Moving averages
        ma_short = returns.rolling(short_window).mean()
        ma_long = returns.rolling(long_window).mean()

        # Price relative to moving averages
        regimes = pd.Series("neutral", index=returns.index)

        # Bull: short MA > long MA and both positive
        regimes[(ma_short > ma_long) & (ma_short > 0)] = "bull"

        # Bear: short MA < long MA and long MA negative
        regimes[(ma_short < ma_long) & (ma_long < 0)] = "bear"

        return regimes

    def detect_volatility_regime(
        self,
        returns: pd.Series,
        window: int = 20,
        high_vol_pct: float = 0.80,
    ) -> pd.Series:
        """Classify periods as high-vol or low-vol based on historical
        percentile of realized volatility.

        Args:
            returns: Daily return series.
            window: Rolling window for volatility calculation.
            high_vol_pct: Percentile threshold for "high vol" regime.

        Returns:
            Series with "high_vol" / "low_vol" labels.
        """
        rolling_vol = returns.rolling(window).std() * np.sqrt(244)

        # Expanding percentile
        expanding_pct = rolling_vol.expanding().rank(pct=True)

        regimes = pd.Series("low_vol", index=returns.index)
        regimes[expanding_pct > high_vol_pct] = "high_vol"

        return regimes

    def detect_hmm(
        self,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """HMM-based regime detection.

        Uses a Gaussian HMM to classify market states based on
        return characteristics and optional additional features.

        Args:
            returns: Daily return series.
            features: Additional features (e.g., volume, volatility).

        Returns:
            Series with integer regime labels (0, 1, ...).
        """
        if not HAS_HMMLEARN:
            logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
            # Fallback to trend-based
            trend = self.detect_trend(returns)
            return trend.map({"bull": 1, "bear": 0, "neutral": 0}).fillna(0)

        # Prepare features
        ret_clean = returns.dropna()
        X = ret_clean.values.reshape(-1, 1)

        if features is not None:
            feat_aligned = features.reindex(ret_clean.index).dropna()
            common_idx = ret_clean.index.intersection(feat_aligned.index)
            ret_aligned = ret_clean[common_idx]
            feat_aligned = feat_aligned.loc[common_idx]
            X = np.column_stack([ret_aligned.values, feat_aligned.values])
            idx = common_idx
        else:
            idx = ret_clean.index

        # Fit HMM
        try:
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
            )
            model.fit(X)
            states = model.predict(X)
            self._hmm_model = model
            self._regime_labels = pd.Series(states, index=idx)

            # Order regimes by mean return (regime 0 = lowest return)
            regime_means = {}
            for s in range(self.n_regimes):
                mask = states == s
                regime_means[s] = float(ret_clean.loc[idx[mask]].mean())
            sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
            mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
            states_remapped = np.array([mapping[s] for s in states])

            return pd.Series(states_remapped, index=idx)

        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}. Using trend-based fallback.")
            trend = self.detect_trend(returns)
            return trend.map({"bull": 1, "bear": 0, "neutral": 0}).fillna(0)

    def combine_regimes(
        self,
        trend_regime: pd.Series,
        vol_regime: pd.Series,
    ) -> pd.Series:
        """Combine trend and volatility regimes into a unified classification.

        Returns:
            Series with labels: "bull_low_vol", "bull_high_vol",
            "bear_low_vol", "bear_high_vol", "neutral".
        """
        combined = pd.Series("neutral", index=trend_regime.index)

        bull_mask = trend_regime == "bull"
        bear_mask = trend_regime == "bear"
        high_vol = vol_regime == "high_vol"
        low_vol = vol_regime == "low_vol"

        combined[bull_mask & low_vol] = "bull_low_vol"
        combined[bull_mask & high_vol] = "bull_high_vol"
        combined[bear_mask & low_vol] = "bear_low_vol"
        combined[bear_mask & high_vol] = "bear_high_vol"

        return combined

    def get_regime_stats(
        self, regimes: pd.Series, returns: pd.Series
    ) -> Dict:
        """Compute performance statistics per regime.

        Returns:
            Dict with per-regime stats (mean return, vol, Sharpe, % of time).
        """
        aligned = pd.DataFrame({"regime": regimes, "return": returns}).dropna()
        stats = {}

        for regime in aligned["regime"].unique():
            mask = aligned["regime"] == regime
            regime_rets = aligned.loc[mask, "return"]
            n = len(regime_rets)
            mean_ret = float(regime_rets.mean())
            vol = float(regime_rets.std())
            sharpe = float(mean_ret / vol * np.sqrt(244)) if vol > 0 else 0.0
            pct_time = n / len(aligned)

            stats[str(regime)] = {
                "n_days": n,
                "pct_of_time": pct_time,
                "mean_daily_return": mean_ret,
                "annual_return": mean_ret * 244,
                "annual_volatility": vol * np.sqrt(244),
                "sharpe": sharpe,
            }

        return stats
