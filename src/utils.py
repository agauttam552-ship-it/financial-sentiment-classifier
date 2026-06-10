"""
Shared utilities: logging setup, path helpers.
"""

import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to console with timestamp."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs():
    """Create all required project directories."""
    for d in ["data", "models", "reports", "notebooks"]:
        Path(d).mkdir(exist_ok=True)
