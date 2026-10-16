"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and hashing so runs are reproducible.

    Args:
        seed: The seed value. Defaults to ``config.SEED``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
