"""Domain feature engineering for Telco churn.

Every feature is a fixed, row-wise rule (no statistics learned from data), so the
transformer is stateless and cannot leak information between splits. It runs as the
first step of the scikit-learn Pipeline in :mod:`src.features.pipeline`.

Feature groups (adapted to the columns available):
    * Tenure: band, new-customer flag, contract length in months.
    * Usage / services: add-on counts, protection counts, internet type flags.
    * Billing signals: auto-pay, electronic check, paperless, average monthly spend.
    * Trend ratios: current bill vs. historical average (price increase proxy), spend per service.
    * Support interactions: fiber without protection, month-to-month without tech support.
    * Household: partner or dependents.

Not available in this dataset: recency (no last-activity date) and support-ticket history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_COLS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PROTECTION_COLS: list[str] = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

TENURE_BINS: list[float] = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS: list[str] = ["0-6", "7-12", "13-24", "25-48", "49+"]
CONTRACT_MONTHS: dict[str, int] = {"Month-to-month": 1, "One year": 12, "Two year": 24}
AUTO_PAY_METHODS: set[str] = {"Bank transfer (automatic)", "Credit card (automatic)"}

ENGINEERED_NUMERIC: list[str] = [
    "contract_months",
    "num_addons",
    "num_protection",
    "num_services",
    "avg_monthly_spend",
    "charge_trend_ratio",
    "spend_per_service",
]
ENGINEERED_BINARY: list[str] = [
    "is_new_customer",
    "has_internet",
    "is_fiber",
    "auto_pay",
    "electronic_check",
    "has_family",
    "fiber_no_protection",
    "m2m_no_techsupport",
]
ENGINEERED_CATEGORICAL: list[str] = ["tenure_band"]
# Single source of truth for output column order. ``add_features`` and
# ``FeatureBuilder.get_feature_names_out`` must agree, because sklearn's ``set_output``
# renames columns by position.
ENGINEERED_ALL: list[str] = [*ENGINEERED_CATEGORICAL, *ENGINEERED_BINARY, *ENGINEERED_NUMERIC]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered features added.

    Also simplifies redundant levels: "No internet service" / "No phone service" become
    "No" in the service columns, since ``has_internet`` / ``PhoneService`` carry that fact.

    Args:
        df: Cleaned customer data (output of :func:`src.data.clean.clean`).

    Returns:
        DataFrame with original plus engineered columns.
    """
    out = df.copy()

    for col in [*ADDON_COLS, "MultipleLines"]:
        out[col] = out[col].replace({"No internet service": "No", "No phone service": "No"})

    tenure = out["tenure"].astype(float)
    monthly = out["MonthlyCharges"].astype(float)
    total = out["TotalCharges"].astype(float)

    # Tenure
    out["tenure_band"] = pd.cut(tenure, TENURE_BINS, labels=TENURE_LABELS).astype(str)
    out["is_new_customer"] = (tenure <= 6).astype(int)
    out["contract_months"] = out["Contract"].map(CONTRACT_MONTHS).fillna(1).astype(int)

    # Services
    addon_yes = out[ADDON_COLS].eq("Yes")
    out["num_addons"] = addon_yes.sum(axis=1)
    out["num_protection"] = out[PROTECTION_COLS].eq("Yes").sum(axis=1)
    out["has_internet"] = out["InternetService"].ne("No").astype(int)
    out["is_fiber"] = out["InternetService"].eq("Fiber optic").astype(int)
    out["num_services"] = (
        out["num_addons"]
        + out["PhoneService"].eq("Yes").astype(int)
        + out["MultipleLines"].eq("Yes").astype(int)
        + out["has_internet"]
    )

    # Billing
    out["auto_pay"] = out["PaymentMethod"].isin(AUTO_PAY_METHODS).astype(int)
    out["electronic_check"] = out["PaymentMethod"].eq("Electronic check").astype(int)
    out["avg_monthly_spend"] = np.where(tenure > 0, total / tenure.clip(lower=1), monthly)

    # Trend ratios: >1 means today's bill is above the customer's historical average
    out["charge_trend_ratio"] = monthly / np.maximum(out["avg_monthly_spend"], 1e-6)
    out["spend_per_service"] = monthly / out["num_services"].clip(lower=1)

    # Household
    out["has_family"] = (out["Partner"].eq("Yes") | out["Dependents"].eq("Yes")).astype(int)

    # Support interactions
    out["fiber_no_protection"] = (out["is_fiber"].eq(1) & out["num_protection"].eq(0)).astype(int)
    out["m2m_no_techsupport"] = (
        out["Contract"].eq("Month-to-month") & out["TechSupport"].ne("Yes")
    ).astype(int)

    original = [c for c in df.columns if c not in ENGINEERED_ALL]
    return out[original + ENGINEERED_ALL]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless scikit-learn wrapper around :func:`add_features`."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:
        """Record input column names only; no statistics are learned. Returns ``self``."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to ``X``."""
        return add_features(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return output column names (inputs plus engineered features)."""
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        base = [c for c in input_features if c not in ENGINEERED_ALL]
        return np.array(base + ENGINEERED_ALL, dtype=object)
