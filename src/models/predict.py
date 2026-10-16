"""Calibrate the final model, score customers, and build the ranked retention list.

Calibration:
    Sigmoid (Platt) calibration is fit on the **validation** split on top of the frozen
    Phase 4 model. The mapping is strictly increasing, so the customer ranking (and
    PR-AUC, lift) is unchanged; only the probability values become trustworthy.
    The Phase 4 threshold is mapped through the same function, so exactly the same
    customers are contacted as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from src.config import ID_COL, MODELS_DIR
from src.explain.shap_analysis import compute_shap, group_shap, top_reasons
from src.features.pipeline import split_xy
from src.models.recommend import recommend_actions
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

CALIBRATED_MODEL_FILE = MODELS_DIR / "calibrated_model.joblib"
RISK_TIERS: tuple[str, str, str] = ("High", "Medium", "Low")


@dataclass
class CalibratedModel:
    """Final model plus its calibrator and decision thresholds."""

    raw: Pipeline  # Phase 4 model (used for SHAP explanations)
    calibrated: CalibratedClassifierCV  # raw model + sigmoid on top
    threshold: float  # contact threshold on the calibrated scale
    raw_threshold: float  # the original Phase 4 threshold
    medium_cutoff: float  # calibrated probability above which risk is above average
    name: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated churn probability for each row."""
        return self.calibrated.predict_proba(X)[:, 1]


def calibrate(
    final: dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series, medium_cutoff: float
) -> CalibratedModel:
    """Fit sigmoid calibration on validation data on top of the frozen final model.

    Args:
        final: The Phase 4 artifact (``pipeline``, ``threshold``, ``name``).
        X_val: Validation features.
        y_val: Validation labels.
        medium_cutoff: Calibrated probability for the Medium tier (use the train base rate).

    Returns:
        A :class:`CalibratedModel`.
    """
    raw: Pipeline = final["pipeline"]
    calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="sigmoid")
    calibrated.fit(X_val, y_val)

    threshold = map_threshold(calibrated, final["threshold"])
    logger.info(
        "Calibrated %s: raw threshold %.2f -> calibrated %.3f",
        final["name"],
        final["threshold"],
        threshold,
    )
    return CalibratedModel(
        raw=raw,
        calibrated=calibrated,
        threshold=threshold,
        raw_threshold=float(final["threshold"]),
        medium_cutoff=float(medium_cutoff),
        name=final["name"],
    )


def map_threshold(calibrated: CalibratedClassifierCV, raw_threshold: float) -> float:
    """Pass the raw threshold through the fitted calibration function.

    The calibrator is strictly increasing, so ``cal(raw) >= cal(t)`` exactly when
    ``raw >= t``: the same customers are selected on every dataset, not just validation.
    """
    (calibrator,) = calibrated.calibrated_classifiers_[0].calibrators
    return float(calibrator.predict(np.array([raw_threshold]))[0])


def risk_tier(prob: np.ndarray, threshold: float, medium_cutoff: float) -> np.ndarray:
    """High = at/above the contact threshold; Medium = above-average risk; Low = the rest."""
    return np.select(
        [prob >= threshold, prob >= medium_cutoff], [RISK_TIERS[0], RISK_TIERS[1]], RISK_TIERS[2]
    )


def build_risk_list(model: CalibratedModel, customers: pd.DataFrame) -> pd.DataFrame:
    """Score customers and return them ranked by churn risk, with reasons and actions.

    Args:
        model: Calibrated final model.
        customers: Cleaned customer rows (ID column plus features; target optional).

    Returns:
        One row per customer, highest risk first, with ``rank``, ``customerID``,
        ``churn_probability``, ``risk_tier``, ``contact``, ``reason_1..3``,
        ``primary_action``, ``secondary_action``, and key account attributes.
    """
    X = customers.drop(columns=[c for c in ("Churn",) if c in customers.columns])
    X = X.drop(columns=[ID_COL])
    prob = model.predict_proba(X)
    reasons = top_reasons(group_shap(compute_shap(model.raw, X)), X, k=3)

    out = pd.DataFrame(
        {
            ID_COL: customers[ID_COL].to_numpy(),
            "churn_probability": np.round(prob, 4),
            "risk_tier": risk_tier(prob, model.threshold, model.medium_cutoff),
            "contact": prob >= model.threshold,
        }
    )
    for k in range(3):
        out[f"reason_{k + 1}"] = [r[k] if len(r) > k else "" for r in reasons]
    actions = [recommend_actions(r, t) for r, t in zip(reasons, out["risk_tier"], strict=True)]
    out["primary_action"] = [a[0] for a in actions]
    out["secondary_action"] = [a[1] for a in actions]
    for col in ("Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"):
        out[col] = customers[col].to_numpy()

    out = out.sort_values("churn_probability", ascending=False, kind="stable")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def save_calibrated(model: CalibratedModel) -> None:
    """Persist the calibrated model to ``models/``."""
    joblib.dump(model, CALIBRATED_MODEL_FILE)


def load_calibrated() -> CalibratedModel:
    """Load the calibrated model saved by :func:`save_calibrated`."""
    return joblib.load(CALIBRATED_MODEL_FILE)


def score_split(model: CalibratedModel, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated scores and labels for a labelled split."""
    X, y = split_xy(df)
    return model.predict_proba(X), y.to_numpy()
