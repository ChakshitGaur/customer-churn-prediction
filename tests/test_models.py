"""Model tests. Phase 0: reproducibility + logging. Phase 4 adds training/metric tests."""

from __future__ import annotations

import logging

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed()
    first = np.random.rand(5)
    set_seed()
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)


def test_get_logger_has_single_handler() -> None:
    logger = get_logger("test_logger")
    get_logger("test_logger")  # calling twice must not duplicate handlers
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
