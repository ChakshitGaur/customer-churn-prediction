"""Testable logic behind the Streamlit app (no Streamlit imports here).

``app.py`` only lays out widgets; everything that transforms data lives in this module
so it can be unit-tested.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import FIGURES_DIR, ID_COL, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.data.clean import clean
from src.data.load import EXPECTED_COLUMNS
from src.explain.shap_analysis import compute_shap, group_shap, plot_customer_waterfall
from src.models.predict import CALIBRATED_MODEL_FILE, CalibratedModel, build_risk_list

FEATURE_COLUMNS: list[str] = [c for c in EXPECTED_COLUMNS if c not in (ID_COL, TARGET_COL)]

YES_NO: list[str] = ["Yes", "No"]
INTERNET_ADDON: list[str] = ["No", "Yes", "No internet service"]

# Choices for the single-customer form, in the raw Telco format.
FORM_OPTIONS: dict[str, list[Any]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Defaults: a typical high-risk profile, so the demo shows something interesting.
DEFAULT_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
}


def model_available(path: Path = CALIBRATED_MODEL_FILE) -> bool:
    """Whether the pipeline has produced the calibrated model."""
    return path.exists()


def make_customer(inputs: dict[str, Any], customer_id: str = "NEW-CUSTOMER") -> pd.DataFrame:
    """Build a one-row raw customer frame from form inputs.

    ``TotalCharges`` defaults to ``tenure * MonthlyCharges`` when not given, which matches
    the data (ratio median 1.00 in EDA).

    Raises:
        ValueError: If a required field is missing or a numeric value is negative.
    """
    row = dict(inputs)
    row.setdefault(
        "TotalCharges", float(row.get("tenure", 0)) * float(row.get("MonthlyCharges", 0))
    )
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if float(row[col]) < 0:
            raise ValueError(f"{col} cannot be negative")
    return pd.DataFrame([{ID_COL: customer_id, **{c: row[c] for c in FEATURE_COLUMNS}}])


def prepare_batch(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean an uploaded batch in the raw Telco format.

    A ``Churn`` column is allowed (e.g. labelled history) and is dropped, so the
    ranked list never shows the outcome.

    Raises:
        ValueError: On missing columns, an empty file, or duplicate customer IDs.
    """
    if raw.empty:
        raise ValueError("The file has no rows")
    missing = [c for c in [ID_COL, *FEATURE_COLUMNS] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    raw = raw.copy()
    raw[ID_COL] = raw[ID_COL].astype(str)
    if raw[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate {ID_COL} values in the file")
    cleaned = clean(raw[[ID_COL, *FEATURE_COLUMNS]])
    return cleaned


def read_csv_upload(data: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes, keeping IDs and TotalCharges as text like the raw file."""
    return pd.read_csv(io.BytesIO(data), dtype={ID_COL: str, "TotalCharges": str})


def score(model: CalibratedModel, customers: pd.DataFrame) -> pd.DataFrame:
    """Ranked risk list (probability, tier, reasons, actions) for cleaned customers."""
    return build_risk_list(model, customers)


def what_if(model: CalibratedModel, customer: pd.DataFrame, changes: dict[str, Any]) -> float:
    """Calibrated churn probability after changing some attributes of one customer.

    This is the model's answer, not a causal estimate: customers who choose long
    contracts may differ in ways the data doesn't capture.
    """
    changed = customer.assign(**changes)
    X = changed.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
    return float(model.predict_proba(X)[0])


def customer_waterfall(model: CalibratedModel, customer: pd.DataFrame) -> Figure:
    """SHAP waterfall (log-odds, grouped by original feature) for one cleaned customer."""
    X = customer.drop(columns=[ID_COL, TARGET_COL], errors="ignore").reset_index(drop=True)
    sv = compute_shap(model.raw, X)
    base_value = float(np.ravel(sv.base_values)[0])
    return plot_customer_waterfall(group_shap(sv), X, base_value, row=0, max_display=10)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV bytes for a download button."""
    return df.to_csv(index=False).encode("utf-8")


def sample_batch(path: Path = PROCESSED_DIR / "test.csv", n: int = 200) -> pd.DataFrame:
    """Demo batch: the first ``n`` held-out test customers, outcome column removed."""
    df = pd.read_csv(path, dtype={ID_COL: str}).head(n)
    return df.drop(columns=[TARGET_COL], errors="ignore")


def training_reference(path: Path = PROCESSED_DIR / "train.csv") -> pd.DataFrame:
    """Training split, the reference distribution for drift checks."""
    return pd.read_csv(path, dtype={ID_COL: str})


@dataclass
class Insights:
    """Saved pipeline outputs shown on the Insights tab."""

    model_summary: dict[str, Any] = field(default_factory=dict)
    business_summary: dict[str, Any] = field(default_factory=dict)
    tiers: pd.DataFrame | None = None
    strategies: pd.DataFrame | None = None
    shap_importance: pd.DataFrame | None = None
    figures: dict[str, Path] = field(default_factory=dict)


INSIGHT_FIGURES: dict[str, str] = {
    "Targeting: cumulative gain and lift": "eval_gain_lift",
    "What drives churn (SHAP)": "shap_global_importance",
    "Calibration before and after": "business_calibration_before_after",
    "Precision-recall curve": "eval_pr_curve",
}


def load_insights(metrics_dir: Path = METRICS_DIR, figures_dir: Path = FIGURES_DIR) -> Insights:
    """Load whichever saved metrics and figures exist (missing ones are skipped)."""

    def _json(name: str) -> dict[str, Any]:
        p = metrics_dir / name
        return json.loads(p.read_text()) if p.exists() else {}

    def _csv(name: str, **kwargs: Any) -> pd.DataFrame | None:
        p = metrics_dir / name
        return pd.read_csv(p, **kwargs) if p.exists() else None

    figures = {
        title: figures_dir / f"{stem}.png"
        for title, stem in INSIGHT_FIGURES.items()
        if (figures_dir / f"{stem}.png").exists()
    }
    return Insights(
        model_summary=_json("model_summary.json"),
        business_summary=_json("business_summary.json"),
        tiers=_csv("risk_tiers.csv"),
        strategies=_csv("strategy_comparison.csv"),
        shap_importance=_csv("shap_importance.csv"),
        figures=figures,
    )


def headline_metrics(insights: Insights) -> dict[str, str]:
    """Formatted headline numbers for the metric cards (empty if not available)."""
    test = insights.model_summary.get("test", {})
    final, base = test.get("final", {}), test.get("baseline", {})
    strategies = insights.strategies
    out: dict[str, str] = {}
    if final:
        out["Test PR-AUC"] = f"{final['pr_auc']:.3f}"
        if base:
            out["vs baseline"] = f"{final['pr_auc'] - base['pr_auc']:+.3f}"
        out["Top-10% lift"] = f"{final['lift_top10']:.2f}x"
    if strategies is not None:
        model_row = strategies[strategies["strategy"] == "Model targeting"]
        if not model_row.empty:
            out["Profit / 1,000 customers"] = (
                f"{model_row['profit_per_1000_customers'].iloc[0]:,.0f}"
            )
    return out
