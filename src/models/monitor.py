"""Drift monitoring with the Population Stability Index (PSI).

PSI compares how a feature (or the score) is distributed in a new batch versus the
training data: ``sum((cur - ref) * ln(cur / ref))`` over bins.

Common reading (also used in ``reports/monitoring_plan.md``):
    * PSI < 0.10        stable
    * 0.10 <= PSI < 0.25 moderate shift: investigate
    * PSI >= 0.25       major shift: check data pipeline, consider retraining
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE: float = 0.10
PSI_MAJOR: float = 0.25
EPS: float = 1e-4  # floor for empty bins so the log stays finite

MONITORED_FEATURES: list[str] = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
]


def _shares(counts: pd.Series) -> np.ndarray:
    shares = counts.to_numpy(dtype=float) / max(counts.sum(), 1)
    return np.clip(shares, EPS, None)


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    Numeric columns use quantile bins from the reference; categorical columns use the
    union of observed categories (a category missing on one side gets the floor share).

    Args:
        reference: Feature values from training data.
        current: Feature values from the new batch.
        n_bins: Number of quantile bins for numeric features.

    Returns:
        PSI (0 = identical distributions).
    """
    ref, cur = reference.dropna(), current.dropna()
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > n_bins:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = pd.cut(ref, edges).value_counts(sort=False)
        cur_counts = pd.cut(cur, edges).value_counts(sort=False)
    else:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        ref_counts = ref.astype(str).value_counts().reindex(cats, fill_value=0)
        cur_counts = cur.astype(str).value_counts().reindex(cats, fill_value=0)
    r, c = _shares(ref_counts), _shares(cur_counts)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    """``"stable"``, ``"moderate"`` or ``"major"`` for a PSI value."""
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """PSI and status for each monitored feature present in both frames.

    Args:
        reference: Training data.
        current: New batch.
        features: Columns to check. Defaults to :data:`MONITORED_FEATURES`.

    Returns:
        One row per feature, sorted by PSI (largest drift first).
    """
    features = features if features is not None else MONITORED_FEATURES
    rows = []
    for col in features:
        if col in reference.columns and col in current.columns:
            value = psi(reference[col], current[col])
            rows.append({"feature": col, "psi": round(value, 4), "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
