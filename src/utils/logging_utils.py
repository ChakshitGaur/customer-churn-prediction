"""Project-wide logging setup."""

from __future__ import annotations

import logging

from src.config import LOG_FORMAT


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with the project's standard format.

    Args:
        name: Logger name, usually ``__name__`` of the calling module.
        level: Logging level for this logger.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
