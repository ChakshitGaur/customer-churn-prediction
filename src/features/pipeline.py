"""scikit-learn preprocessing Pipeline: engineered features -> scaling + one-hot encoding.

Leakage rule: this pipeline is only ever ``fit`` on the training split (directly, or inside
cross-validation folds via :func:`make_model_pipeline`). Val/test are only ``transform``-ed.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COL, TARGET_COL
from src.features.build_features import (
    ADDON_COLS,
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureBuilder,
)

RAW_NUMERIC: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_BINARY: list[str] = ["SeniorCitizen"]  # already 0/1
RAW_CATEGORICAL: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_COLS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered columns that exactly duplicate a raw/one-hot column (|r| >= 0.99 on train, see
# notebooks/02_features). They stay in ``add_features`` (interactions use them) but are not
# model inputs: duplicates split SHAP credit and blur coefficient interpretation.
DUPLICATE_FEATURES: list[str] = [
    "is_new_customer",  # == tenure_band "0-6"
    "has_internet",  # == 1 - InternetService "No"
    "is_fiber",  # == InternetService "Fiber optic"
    "electronic_check",  # == PaymentMethod "Electronic check"
    "avg_monthly_spend",  # r = 0.996 with MonthlyCharges
]

NUMERIC_FEATURES: list[str] = RAW_NUMERIC + [
    c for c in ENGINEERED_NUMERIC if c not in DUPLICATE_FEATURES
]
BINARY_FEATURES: list[str] = RAW_BINARY + [
    c for c in ENGINEERED_BINARY if c not in DUPLICATE_FEATURES
]
CATEGORICAL_FEATURES: list[str] = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target, dropping the customer ID.

    Args:
        df: A cleaned split (train, val, or test).

    Returns:
        ``(X, y)`` where ``X`` excludes ID and target columns.
    """
    return df.drop(columns=[ID_COL, TARGET_COL]), df[TARGET_COL]


def build_preprocessor() -> Pipeline:
    """Create the unfitted preprocessing pipeline.

    Steps:
        1. ``features``: stateless domain features (:class:`FeatureBuilder`).
        2. ``encode``: StandardScaler on numeric columns, one-hot on categoricals
           (``handle_unknown="ignore"`` so unseen categories in val/test don't crash),
           binary flags passed through. Unlisted columns are dropped.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline` that outputs a pandas DataFrame.
    """
    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    pipe = Pipeline([("features", FeatureBuilder()), ("encode", encode)])
    return pipe.set_output(transform="pandas")


def make_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Chain the preprocessor with a model so CV refits preprocessing inside each fold.

    Args:
        estimator: Any scikit-learn compatible classifier.

    Returns:
        ``Pipeline([("prep", preprocessor), ("model", estimator)])``.
    """
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
