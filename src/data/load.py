"""Load raw data from data/raw (read-only).

The loader returns the data exactly as stored: no type fixes, no row drops.
Cleaning happens in :mod:`src.data.clean` (Phase 2) so the raw view stays auditable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ID_COL, RAW_DATA_FILE, TARGET_COL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw churn CSV without modifying it.

    ``TotalCharges`` is read as text on purpose: the raw file contains blank strings
    for brand-new customers, which cleaning handles explicitly.

    Args:
        path: CSV location. Defaults to ``config.RAW_DATA_FILE``.

    Returns:
        The raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the columns do not match the expected schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")

    df = pd.read_csv(path, dtype={ID_COL: str, "TotalCharges": str})
    validate_schema(df)
    logger.info("Loaded %s rows x %s cols from %s", *df.shape, path.name)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Check that the expected columns are present and the target has known labels.

    Args:
        df: Raw DataFrame.

    Raises:
        ValueError: On missing columns or unexpected target values.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    labels = set(df[TARGET_COL].dropna().unique())
    if not labels <= {"Yes", "No"}:
        raise ValueError(f"Unexpected {TARGET_COL} labels: {sorted(labels)}")
