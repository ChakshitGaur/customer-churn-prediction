"""Deterministic cleaning: type fixes, missing values, target encoding, leakage removal.

Every step here is a fixed rule (no statistics learned from the data), so it is safe to
apply before splitting. Anything learned from data (scaling, encoding, imputation by
mean/median) belongs in the Phase 3 pipeline, fit on the training split only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ID_COL, INTERIM_DIR, TARGET_COL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

CLEAN_FILE: Path = INTERIM_DIR / "telco_clean.csv"

NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns only known after (or because of) churn. Telco has none, but the list makes the
# rule explicit and enforceable if the data source changes (e.g. churn reason, exit survey).
LEAKAGE_COLS: list[str] = ["ChurnReason", "ChurnScore", "ChurnDate", "ChurnCategory", "CLTV"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    Steps:
        1. Strip whitespace from all text columns.
        2. ``TotalCharges``: blank strings -> 0.0 (only tenure-0 customers, not yet billed),
           then cast to float.
        3. Encode the target ``Churn``: "Yes" -> 1, "No" -> 0 (skipped when the column is
           absent, e.g. current customers being scored).
        4. Drop any known post-churn leakage columns.
        5. Drop exact duplicate rows (none expected; logged if found).

    Args:
        df: Raw DataFrame from :func:`src.data.load.load_raw`.

    Returns:
        Cleaned DataFrame; the input is not modified.

    Raises:
        ValueError: If blank ``TotalCharges`` appear for customers with tenure > 0,
            which would mean the 0-fill rule no longer holds.
    """
    out = df.copy()

    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].str.strip()

    total = out["TotalCharges"]
    blank = total.isna()
    if pd.api.types.is_string_dtype(total) or total.dtype == object:
        blank |= total.astype("string").str.strip().eq("").fillna(False)
    if (out.loc[blank, "tenure"].astype(float) > 0).any():
        raise ValueError("Blank TotalCharges found for customers with tenure > 0")
    out["TotalCharges"] = pd.to_numeric(total.where(~blank, 0)).astype(float)
    logger.info("TotalCharges: filled %d blank values (tenure 0) with 0.0", int(blank.sum()))

    if TARGET_COL in out.columns:
        target = out[TARGET_COL]
        if not pd.api.types.is_numeric_dtype(target):
            target = target.map({"Yes": 1, "No": 0})
        out[TARGET_COL] = target.astype(int)

    leaked = [c for c in LEAKAGE_COLS if c in out.columns]
    if leaked:
        out = out.drop(columns=leaked)
        logger.warning("Dropped post-churn leakage columns: %s", leaked)

    n_before = len(out)
    out = out.drop_duplicates()
    if len(out) < n_before:
        logger.warning("Dropped %d exact duplicate rows", n_before - len(out))

    if not out[ID_COL].is_unique:
        raise ValueError(f"{ID_COL} is not unique after cleaning")

    out[NUMERIC_COLS] = out[NUMERIC_COLS].astype(float)
    logger.info("Cleaned data: %s rows x %s cols", *out.shape)
    return out.reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: Path = CLEAN_FILE) -> Path:
    """Write cleaned data to ``data/interim``.

    Args:
        df: Cleaned DataFrame.
        path: Output CSV path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)
    return path
