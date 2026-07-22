"""Alpha Signal Research — Citadel-style systematic signal discovery.

Pipeline: Feature Engineering → Signal Testing (IC/Quantile/Fama-MacBeth)
→ Decay Analysis → Signal Combination → Regime Detection.
"""

from .feature_engineering import FeatureEngineer
from .signal_tester import SignalTester
from .decay_analyzer import DecayAnalyzer
from .signal_combiner import SignalCombiner
from .regime_detector import RegimeDetector

__all__ = [
    "FeatureEngineer",
    "SignalTester",
    "DecayAnalyzer",
    "SignalCombiner",
    "RegimeDetector",
]
