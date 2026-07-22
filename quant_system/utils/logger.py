"""
Structured logging for quant_system.

Provides both file and console logging with consistent formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_loggers: dict = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Get or create a structured logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level.
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
    logger.addHandler(console)

    # File handler (if requested)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def setup_root_logger(log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Set up the root quant_system logger.

    Args:
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Root logger.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "output" / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"quant_system_{timestamp}.log"

    return get_logger("quant_system", level=level, log_file=log_file)
