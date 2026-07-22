"""
Cross-module data registry.

A singleton registry that allows modules to share large data objects
without circular imports. Any module can publish or retrieve data
by name.
"""

from typing import Any, Dict, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataRegistry:
    """Thread-safe singleton registry for shared data objects.

    Modules publish their outputs (stock pools, factor data, signals, etc.)
    and other modules retrieve them by name. This breaks circular import
    chains and allows lazy loading.

    Usage:
        registry = DataRegistry()
        registry.put("stock_pool", my_pool)
        pool = registry.get("stock_pool")
    """

    _instance: Optional["DataRegistry"] = None
    _store: Dict[str, Any]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = {}
        return cls._instance

    def put(self, name: str, obj: Any, overwrite: bool = True):
        """Store an object in the registry."""
        if name in self._store and not overwrite:
            logger.warning(f"Registry key '{name}' already exists. Use overwrite=True to replace.")
            return
        self._store[name] = obj
        logger.debug(f"Registered '{name}' ({type(obj).__name__})")

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve an object from the registry."""
        return self._store.get(name, default)

    def has(self, name: str) -> bool:
        """Check if a key exists in the registry."""
        return name in self._store

    def remove(self, name: str):
        """Remove an object from the registry."""
        self._store.pop(name, None)

    def clear(self):
        """Clear all registered objects."""
        self._store.clear()

    def list_keys(self) -> list:
        """List all registered keys."""
        return list(self._store.keys())

    def summary(self) -> str:
        """Return a summary of all registered objects."""
        lines = ["DataRegistry contents:"]
        for name, obj in self._store.items():
            type_name = type(obj).__name__
            size_hint = ""
            if hasattr(obj, "__len__"):
                size_hint = f" (len={len(obj)})"
            lines.append(f"  {name}: {type_name}{size_hint}")
        return "\n".join(lines)
