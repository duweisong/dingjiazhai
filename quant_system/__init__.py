"""
Quant System — Multi-Role Quantitative Trading Framework

Inspired by the 15-role institutional quant pipeline:
Goldman Sachs · Renaissance · Citadel · Two Sigma · Jane Street · AQR
D.E. Shaw · Bridgewater · Bloomberg · Virtu · Point72 · Man Group
Millennium · Dimensional

Core 6 roles implemented:
  1. Strategy Architect (Goldman Sachs style)
  2. Backtest Engine (Renaissance style)
  3. Risk Manager (Two Sigma style)
  4. Alpha Signal Researcher (Citadel style)
  5. Factor Model Builder (AQR style)
  6. Portfolio Optimizer (Man Group style)
"""

import sys
from pathlib import Path

_PARENT = Path(__file__).parent.parent  # c:\AI
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# The backtest/ module uses flat imports (from config import ...),
# so its directory must also be on sys.path
_BACKTEST_DIR = _PARENT / "backtest"
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

__version__ = "0.1.0"
__author__ = "Quant System"
