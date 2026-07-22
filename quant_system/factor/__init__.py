"""Factor Model Builder — AQR-style systematic factor investing.

Builds multi-factor models, analyzes factor correlations, and
decomposes returns into factor contributions vs. residual alpha.
"""

from .factor_definitions import FactorBuilder
from .factor_correlation import FactorCorrelationAnalyzer
from .multi_factor_composer import MultiFactorComposer
from .attribution import PerformanceAttribution

__all__ = [
    "FactorBuilder",
    "FactorCorrelationAnalyzer",
    "MultiFactorComposer",
    "PerformanceAttribution",
]
