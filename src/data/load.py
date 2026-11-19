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

_YES_NO = ["Yes", "No"]
_ADDON = ["Yes", "No", "No internet service"]

# The category values the model was built for. The one-hot encoder maps any other value to
# "none of the above", so a new or misspelled value is scored without error but meaninglessly.
EXPECTED_CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": _ADDON,
    "OnlineBackup": _ADDON,
    "DeviceProtection": _ADDON,
    "TechSupport": _ADDON,
    "StreamingTV": _ADDON,
    "StreamingMovies": _ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def unexpected_categories(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Category values outside :data:`EXPECTED_CATEGORIES`, with how many rows have each.

    Values are compared after trimming whitespace (cleaning strips it). Missing values and
    columns that are absent are not reported here; ``validate_schema`` covers missing columns.

    Args:
        df: Raw (or cleaned) DataFrame.

    Returns:
        ``{column: {value: row_count}}`` for columns with at least one unexpected value;
        empty when everything is as expected.
    """
    found: dict[str, dict[str, int]] = {}
    for col, allowed in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        extra = values[~values.isin(allowed)].value_counts()
        if not extra.empty:
            found[col] = {str(v): int(n) for v, n in extra.items()}
    return found


def warn_unexpected_categories(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Log one warning per column that has unexpected category values; never raises.

    Returns:
        The same mapping as :func:`unexpected_categories`.
    """
    found = unexpected_categories(df)
    for col, counts in found.items():
        shown = ", ".join(f"{v!r} ({n} rows)" for v, n in list(counts.items())[:3])
        more = f" and {len(counts) - 3} more" if len(counts) > 3 else ""
        logger.warning(
            "%s has values the model was not built for: %s%s. They are encoded as 'none of the "
            "above', so scores for those rows are unreliable. Expected: %s",
            col,
            shown,
            more,
            ", ".join(EXPECTED_CATEGORIES[col]),
        )
    return found


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

    Unexpected category values are logged as warnings, not errors (see
    :func:`warn_unexpected_categories`).
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")

    df = pd.read_csv(path, dtype={ID_COL: str, "TotalCharges": str})
    validate_schema(df)
    warn_unexpected_categories(df)
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

    # Compare trimmed labels: clean() strips whitespace, so " Yes " is valid raw data.
    labels = {str(v).strip() for v in df[TARGET_COL].dropna().unique()}
    if not labels <= {"Yes", "No"}:
        raise ValueError(f"Unexpected {TARGET_COL} labels: {sorted(labels)}")
