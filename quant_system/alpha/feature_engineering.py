"""
Feature engineering pipeline for alpha signal research.

Implements the standard quant workflow:
1. Standardize (z-score cross-sectionally)
2. Winsorize (clip extreme values)
3. Neutralize (sector + size regression residuals)
4. Smooth (optional EMA smoothing)

All operations are cross-sectional (per date, across stocks) to
avoid look-ahead bias.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Transforms raw factor values into clean, tradeable signals.

    The Citadel approach: "A raw factor is just data. A clean,
    neutralized, properly scaled factor is an edge."

    All operations respect the temporal ordering of data:
    transformations at date T use only information available at T.
    """

    def __init__(
        self,
        winsorize_pct: float = 0.01,
        neutralize: bool = True,
    ):
        self.winsorize_pct = winsorize_pct
        self.neutralize_enabled = neutralize

    def pipeline(
        self,
        factor_values: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.Series] = None,
        steps: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Args:
            factor_values: DataFrame with dates as index, stock codes as columns.
            sectors: Series mapping stock_code -> sector name.
            market_caps: Series mapping stock_code -> market cap.
            steps: List of steps to apply. Default: all.
                   Options: "standardize", "winsorize", "neutralize", "smooth".

        Returns:
            Transformed factor DataFrame with same shape.
        """
        if steps is None:
            steps = ["winsorize", "standardize", "neutralize"]

        result = factor_values.copy()

        for step in steps:
            if step == "winsorize":
                result = self.winsorize(result)
            elif step == "standardize":
                result = self.standardize(result)
            elif step == "neutralize":
                if self.neutralize_enabled:
                    result = self.neutralize(result, sectors, market_caps)
            elif step == "smooth":
                result = self.smooth(result, window=3)

        return result

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score standardization.

        For each date, each stock's value is (value - mean) / std
        across all stocks at that date.
        """
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    def winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize extreme values cross-sectionally.

        Clips values at the lower and upper percentiles for each date,
        treating them as outliers rather than signals.
        """
        lower = df.quantile(self.winsorize_pct, axis=1)
        upper = df.quantile(1 - self.winsorize_pct, axis=1)
        return df.clip(lower=lower, upper=upper, axis=0)

    def neutralize(
        self,
        df: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
        market_caps: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Neutralize factor against sector and size.

        For each date, regress factor values on sector dummies and
        log(market_cap), then take the residuals. This ensures the
        factor captures pure stock-specific effects.

        Args:
            df: Factor values (dates x stocks).
            sectors: Stock -> sector mapping (optional).
            market_caps: Stock -> market cap mapping (optional).

        Returns:
            Neutralized factor values (residuals).
        """
        result = df.copy()
        dates = df.index

        for date in dates:
            row = df.loc[date].dropna()
            stocks = row.index.tolist()

            if len(stocks) < 10:
                continue  # Too few stocks for meaningful regression

            y = row.values.astype(float)

            # Build design matrix
            X_cols = []

            # Sector dummies
            if sectors is not None:
                sector_series = sectors.reindex(stocks).fillna("Unknown")
                sector_dummies = pd.get_dummies(sector_series, drop_first=True)
                # Ensure all dummy rows exist
                sector_dummies = sector_dummies.reindex(stocks).fillna(0)
                if sector_dummies.shape[1] > 0 and sector_dummies.shape[1] < len(stocks) - 2:
                    X_cols.append(sector_dummies.values)

            # Log market cap
            if market_caps is not None:
                caps = market_caps.reindex(stocks).fillna(0)
                log_cap = np.log(np.maximum(caps.values, 1.0))
                X_cols.append(log_cap.reshape(-1, 1))

            if not X_cols:
                continue

            X = np.column_stack(X_cols) if len(X_cols) > 1 else X_cols[0]
            X = np.column_stack([np.ones(len(stocks)), X])  # Add intercept

            # OLS regression
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                residuals = y - y_pred
                result.loc[date, stocks] = residuals
            except np.linalg.LinAlgError:
                continue

        return result

    def smooth(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """Apply EMA smoothing across time for each stock.

        Reduces noise in factor values without introducing look-ahead bias
        (uses only past values at each point).

        Args:
            df: Factor values (dates x stocks).
            window: EMA span in days.

        Returns:
            Smoothed factor values.
        """
        return df.ewm(span=window, adjust=False).mean()

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile ranking (0 to 1).

        Sometimes used instead of z-score for robustness to outliers.
        """
        return df.rank(axis=1, pct=True)

    def check_quality(self, df: pd.DataFrame) -> Dict:
        """Run quality checks on factor values.

        Returns a report with:
        - NaN percentage
        - Coverage (fraction of stocks with non-NaN values)
        - Skewness
        - Kurtosis
        """
        report = {
            "shape": df.shape,
            "nan_pct": float(df.isna().mean().mean()),
            "coverage": float((~df.isna()).mean().mean()),
            "mean": float(df.mean().mean()),
            "std": float(df.std().mean()),
        }

        # Cross-sectional stats
        cs_mean = df.mean(axis=1)
        cs_std = df.std(axis=1)
        report["cs_mean_stability"] = float(cs_mean.std())
        report["cs_std_stability"] = float(cs_std.std())

        return report
