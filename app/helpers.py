"""Testable logic behind the Streamlit app (no Streamlit imports here).

``app.py`` only lays out widgets; everything that transforms data lives in this module
so it can be unit-tested.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import (
    FIGURES_DIR,
    ID_COL,
    METRICS_DIR,
    MODEL_MAX_AGE_DAYS,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DATA_FILE,
    TARGET_COL,
)
from src.data.clean import clean
from src.data.load import EXPECTED_COLUMNS
from src.explain.shap_analysis import compute_shap, group_shap, plot_customer_waterfall
from src.models.model_card import file_sha256, library_versions
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


MODEL_CARD_FILE: Path = MODELS_DIR / "model_card.json"
FINAL_MODEL_FILE: Path = MODELS_DIR / "final_model.joblib"
MTIME_TOLERANCE_SECONDS: float = 5.0


@dataclass(frozen=True)
class Notice:
    """A message about the loaded model for the dashboard to show.

    ``level`` is ``"warning"`` when the model may not match its data or environment, and
    ``"info"`` for context that does not need action.
    """

    level: str
    message: str


def load_model_card(path: Path = MODEL_CARD_FILE) -> dict[str, Any] | None:
    """The model card written by the pipeline, or ``None`` if absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=4)
def _sha256_cached(path: str, mtime_ns: int, size: int) -> str:
    """Hash a file, remembering the result until the file changes."""
    return file_sha256(Path(path))


def _current_data_hash(raw_file: Path) -> str | None:
    try:
        stat = raw_file.stat()
    except OSError:
        return None
    return _sha256_cached(str(raw_file), stat.st_mtime_ns, stat.st_size)


def model_notices(
    card_path: Path = MODEL_CARD_FILE,
    raw_file: Path = RAW_DATA_FILE,
    model_file: Path = FINAL_MODEL_FILE,
    now: datetime | None = None,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
) -> list[Notice]:
    """Check that the model on disk still matches the data and environment it was trained in.

    Compares the model card written by the pipeline with the current state: the raw data hash,
    the model file's modification time, the installed library versions, and the model's age.
    A missing raw data file is not an error (a deployed app may not ship the data), so that
    check is skipped.

    Args:
        card_path: Model card JSON.
        raw_file: Current raw data file.
        model_file: Trained model artifact the card describes.
        now: Current time (for tests); defaults to the real time.
        max_age_days: Age after which an info note is shown.

    Returns:
        Warnings first, then info notes; empty when everything matches.
    """
    card = load_model_card(card_path)
    if card is None:
        return [
            Notice(
                "info",
                "No model card found, so this model cannot be checked against its data. "
                "Run `python -m src.pipeline` (or `python -m src.models.model_card`) to create it.",
            )
        ]
    if not (
        isinstance(card.get("data"), dict)
        and "sha256" in card["data"]
        and isinstance(card.get("code"), dict)
        and isinstance(card["code"].get("libraries"), dict)
    ):
        return [
            Notice(
                "info",
                "The model card is incomplete or from an older version, so the model cannot be "
                "checked. Run `python -m src.models.model_card` to rewrite it.",
            )
        ]
    notices: list[Notice] = []

    data_hash = _current_data_hash(raw_file)
    if data_hash is not None and data_hash != card["data"]["sha256"]:
        notices.append(
            Notice(
                "warning",
                "The raw data has changed since this model was trained. "
                "Rerun `python -m src.pipeline` to retrain on the current data.",
            )
        )

    trained_at = card.get("model_trained_at")
    if trained_at and model_file.exists():
        recorded = datetime.fromisoformat(trained_at)
        actual = datetime.fromtimestamp(model_file.stat().st_mtime, UTC)
        if abs((actual - recorded).total_seconds()) > MTIME_TOLERANCE_SECONDS:
            notices.append(
                Notice(
                    "warning",
                    "The model file was replaced after its model card was written, so the card "
                    "may not describe it. Rerun `python -m src.models.model_card`.",
                )
            )

    installed = library_versions(tuple(card["code"]["libraries"]))
    changed = [
        f"{name} {was} → {installed[name]}"
        for name, was in card["code"]["libraries"].items()
        if was is not None and installed[name] != was
    ]
    if changed:
        notices.append(
            Notice(
                "warning",
                "Library versions differ from training, which can change a saved model's "
                f"behaviour: {', '.join(changed)}.",
            )
        )

    if trained_at:
        now = now if now is not None else datetime.now(UTC)
        age_days = (now - datetime.fromisoformat(trained_at)).days
        if age_days > max_age_days:
            notices.append(
                Notice(
                    "info",
                    f"This model was trained {age_days} days ago. Check the monitoring plan's "
                    "retraining triggers.",
                )
            )
    return sorted(notices, key=lambda n: n.level != "warning")


def provenance_caption(card: dict[str, Any] | None) -> str:
    """One line for the sidebar saying when and on what the model was trained."""
    if card is None:
        return "Provenance unknown (no model card)."
    trained = (card.get("model_trained_at") or "unknown date")[:10]
    commit = (card.get("code") or {}).get("commit")
    sha = (card.get("data") or {}).get("sha256")
    parts = [f"Trained {trained}"]
    if sha:
        parts.append(f"data {sha[:8]}")
    if commit:
        parts.append(f"commit {commit[:7]}")
    return " · ".join(parts)


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
