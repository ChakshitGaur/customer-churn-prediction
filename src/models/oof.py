"""Out-of-fold calibrated churn probabilities on the development data (train + validation).

The saved calibrated model is fit on the validation split and scored on the test set, which
must stay untouched for analyses like the profit sensitivity and subgroup checks. Instead,
these analyses use out-of-fold predictions: every customer is scored by a model that never
saw them, with sigmoid calibration fit inside each training fold.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR
from src.data.split import Splits, load_splits
from src.models.repeated_cv import make_tuned_pipeline
from src.models.train import make_cv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def oof_calibrated_scores(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold calibrated churn probabilities.

    Args:
        pipeline: Unfitted model pipeline.
        X: Features.
        y: Binary labels.

    Returns:
        Array of probabilities aligned with ``X``; each row is scored by a model that did
        not see it.
    """
    model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    scores = cross_val_predict(model, X, y, cv=make_cv(), method="predict_proba")
    return np.asarray(scores)[:, 1]


def dev_frame(splits: Splits | None = None) -> pd.DataFrame:
    """Train + validation rows (the test split is never included)."""
    splits = splits if splits is not None else load_splits()
    return pd.concat([splits.train, splits.val], ignore_index=True)


def final_pipeline(y: pd.Series) -> Pipeline:
    """Unfitted pipeline for the selected final model with its tuned params from Phase 4."""
    name = json.loads((METRICS_DIR / "model_summary.json").read_text())["final_model"]
    tuning = json.loads((METRICS_DIR / "tuning_results.json").read_text())
    return make_tuned_pipeline(name, tuning[name]["best_params"], y)
