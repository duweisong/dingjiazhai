"""
Strategy specification parser and validator.

Parses YAML/JSON strategy definitions into validated StrategySpec
objects that drive the entire quant pipeline.

Goldman Sachs approach: "A strategy without a written spec is not
a strategy — it's a hope. Every parameter must be explicit, every
assumption documented, every edge case considered."
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StrategySpec:
    """Immutable strategy specification."""
    name: str
    version: str = "1.0"
    description: str = ""

    # Universe
    universe_filter: str = "hs300"           # e.g. "hs300 AND market_cap > 100"

    # Factor exposures
    factor_exposures: Dict[str, float] = field(default_factory=dict)

    # Signal composition
    signal_weights: Dict[str, float] = field(default_factory=dict)

    # Regime gating
    regime_gating: Dict[str, str] = field(default_factory=dict)

    # Portfolio constraints
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Risk parameters
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.25
    trailing_stop_pct: float = 0.06
    max_single_position: float = 0.10
    max_sector_exposure: float = 0.30

    # Execution
    rebalance_freq: str = "weekly"           # "daily" | "weekly" | "monthly"
    position_method: str = "signal_strength" # "equal_weight" | "signal_strength" | "optimized"
    min_signal_strength: float = 0.3


class StrategySpecParser:
    """Parses and validates strategy specifications."""

    # Built-in templates
    TEMPLATES = {
        "momentum": {
            "name": "Momentum Strategy",
            "version": "1.0",
            "description": "Cross-sectional momentum strategy across HS300 stocks",
            "universe_filter": "hs300",
            "factor_exposures": {
                "momentum_20d": 0.3,
                "momentum_60d": 0.4,
                "momentum_120d": 0.3,
            },
            "signal_weights": {"momentum_composite": 1.0},
            "regime_gating": {"bear": "off", "bull": "active", "neutral": "active"},
            "constraints": {"max_weight": 0.10, "min_positions": 5, "max_positions": 20},
        },
        "value_quality": {
            "name": "Value + Quality Strategy",
            "version": "1.0",
            "description": "Combined value and quality factor strategy",
            "universe_filter": "hs300",
            "factor_exposures": {
                "value_pe": 0.25,
                "value_pb": 0.25,
                "quality_roe": 0.25,
                "quality_gross_margin": 0.25,
            },
            "signal_weights": {"value_composite": 0.5, "quality_composite": 0.5},
            "regime_gating": {"bear": "passive", "bull": "active", "neutral": "active"},
            "constraints": {"max_weight": 0.10, "min_positions": 8, "max_positions": 30},
        },
        "low_vol": {
            "name": "Low Volatility Strategy",
            "version": "1.0",
            "description": "Low volatility anomaly strategy",
            "universe_filter": "hs300",
            "factor_exposures": {
                "volatility_20d": 0.5,
                "volatility_60d": 0.5,
            },
            "signal_weights": {"vol_composite": 1.0},
            "regime_gating": {"bear": "active", "bull": "active", "neutral": "active"},
            "constraints": {"max_weight": 0.08, "min_positions": 10, "max_positions": 30},
        },
        "multi_factor": {
            "name": "Multi-Factor Strategy",
            "version": "1.0",
            "description": "Diversified multi-factor strategy: momentum + value + quality + low vol",
            "universe_filter": "hs300",
            "factor_exposures": {
                "momentum_60d": 0.30,
                "value_pe": 0.20,
                "value_pb": 0.10,
                "quality_roe": 0.15,
                "volatility_20d": 0.15,
                "size": 0.10,
            },
            "signal_weights": {
                "momentum_composite": 0.35,
                "value_composite": 0.25,
                "quality_composite": 0.20,
                "vol_composite": 0.20,
            },
            "regime_gating": {"bear": "passive", "bull": "active", "neutral": "active"},
            "constraints": {"max_weight": 0.08, "min_positions": 10, "max_positions": 30},
        },
    }

    def load(self, path: str) -> StrategySpec:
        """Load a strategy specification from a YAML or JSON file.

        Args:
            path: Path to .yaml or .json file.

        Returns:
            Validated StrategySpec.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                raw = yaml.safe_load(f)
            elif path.suffix == ".json":
                raw = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {path.suffix}. Use .yaml or .json")

        return self._from_dict(raw)

    def load_template(self, name: str) -> StrategySpec:
        """Load a built-in strategy template.

        Args:
            name: "momentum" | "value_quality" | "low_vol" | "multi_factor".

        Returns:
            Validated StrategySpec.
        """
        if name not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template '{name}'. Available: {available}")

        return self._from_dict(self.TEMPLATES[name])

    def _from_dict(self, raw: Dict) -> StrategySpec:
        """Build StrategySpec from raw dict with defaults."""
        return StrategySpec(
            name=raw.get("name", "Unnamed Strategy"),
            version=raw.get("version", "1.0"),
            description=raw.get("description", ""),
            universe_filter=raw.get("universe_filter", "hs300"),
            factor_exposures=raw.get("factor_exposures", {}),
            signal_weights=raw.get("signal_weights", {}),
            regime_gating=raw.get("regime_gating", {}),
            constraints=raw.get("constraints", {}),
            stop_loss_pct=raw.get("stop_loss_pct", 0.08),
            take_profit_pct=raw.get("take_profit_pct", 0.25),
            trailing_stop_pct=raw.get("trailing_stop_pct", 0.06),
            max_single_position=raw.get("max_single_position", 0.10),
            max_sector_exposure=raw.get("max_sector_exposure", 0.30),
            rebalance_freq=raw.get("rebalance_freq", "weekly"),
            position_method=raw.get("position_method", "signal_strength"),
            min_signal_strength=raw.get("min_signal_strength", 0.3),
        )

    def validate(self, spec: StrategySpec) -> List[str]:
        """Validate a strategy specification.

        Returns:
            List of validation errors (empty = valid).
        """
        errors = []

        # Name
        if not spec.name or spec.name == "Unnamed Strategy":
            errors.append("Strategy must have a name")

        # Factor exposures sum check
        if spec.factor_exposures:
            total = sum(spec.factor_exposures.values())
            if total <= 0:
                errors.append("Factor exposures must sum to a positive value")
            if total > 2.0:
                errors.append(f"Factor exposure sum ({total:.2f}) seems very high")

        # Signal weights sum check
        if spec.signal_weights:
            total = sum(spec.signal_weights.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"Signal weights should sum to 1.0 (got {total:.2f})")

        # Constraints
        if spec.max_single_position > 0.15:
            errors.append(f"max_single_position ({spec.max_single_position:.0%}) is very high")
        if spec.max_single_position < 0.02:
            errors.append(f"max_single_position ({spec.max_single_position:.0%}) may be too restrictive")

        # Risk parameters
        if spec.stop_loss_pct > 0.20:
            errors.append(f"Stop loss ({spec.stop_loss_pct:.0%}) seems too wide")
        if spec.take_profit_pct < spec.stop_loss_pct:
            errors.append("Take profit should be larger than stop loss")

        # Rebalance frequency
        valid_freqs = {"daily", "weekly", "monthly"}
        if spec.rebalance_freq not in valid_freqs:
            errors.append(f"Invalid rebalance_freq '{spec.rebalance_freq}'. Use: {valid_freqs}")

        return errors

    def list_templates(self) -> Dict[str, str]:
        """List available strategy templates with descriptions."""
        return {name: t["description"] for name, t in self.TEMPLATES.items()}

    def export_yaml(self, spec: StrategySpec, path: str):
        """Export a strategy spec to YAML file."""
        data = {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "universe_filter": spec.universe_filter,
            "factor_exposures": spec.factor_exposures,
            "signal_weights": spec.signal_weights,
            "regime_gating": spec.regime_gating,
            "constraints": spec.constraints,
            "stop_loss_pct": spec.stop_loss_pct,
            "take_profit_pct": spec.take_profit_pct,
            "trailing_stop_pct": spec.trailing_stop_pct,
            "max_single_position": spec.max_single_position,
            "max_sector_exposure": spec.max_sector_exposure,
            "rebalance_freq": spec.rebalance_freq,
            "position_method": spec.position_method,
            "min_signal_strength": spec.min_signal_strength,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Exported strategy spec to {path}")
