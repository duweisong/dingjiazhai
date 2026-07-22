"""Strategy Architect — Goldman Sachs-style strategy design.

Strategy specification parsing, signal pipeline composition,
and built-in strategy templates.
"""

from .spec_parser import StrategySpecParser, StrategySpec
from .signal_pipeline import SignalPipeline

__all__ = [
    "StrategySpecParser",
    "StrategySpec",
    "SignalPipeline",
]
