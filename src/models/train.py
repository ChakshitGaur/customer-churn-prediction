"""Train and cross-validate candidate models on the training split only.

Imbalance is handled with class weights (no resampling):
    * LogisticRegression / RandomForest / LightGBM: ``class_weight="balanced"``-style options.
    * XGBoost: ``scale_pos_weight = negatives / positives`` computed from the training data.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.config import CV_FOLDS, SEED
from src.features.pipeline import make_model_pipeline
from src.models.evaluate import CV_SCORERS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def make_cv(seed: int = SEED) -> StratifiedKFold:
    """Stratified K-fold used everywhere in the project."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)


def candidate_models(y_train: pd.Series) -> dict[str, BaseEstimator]:
    """Unfitted candidate estimators with default-ish, sensible settings.

    Args:
        y_train: Training labels, used only for XGBoost's ``scale_pos_weight``.

    Returns:
        Mapping of model name to estimator.
    """
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    return {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=SEED,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
        ),
    }


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator] | None = None,
    feature_sets: tuple[bool, ...] = (False, True),
) -> pd.DataFrame:
    """Cross-validate every model on every feature set (train split only).

    Preprocessing is inside the pipeline, so it is refit within each fold (no leakage).

    Args:
        X_train: Training features.
        y_train: Training labels.
        models: Estimators to evaluate. Defaults to :func:`candidate_models`.
        feature_sets: ``False`` = raw columns only, ``True`` = raw + engineered.

    Returns:
        One row per (model, feature set) with mean and std of each CV metric,
        sorted by mean PR-AUC (descending).
    """
    models = models if models is not None else candidate_models(y_train)
    cv = make_cv()
    rows = []
    for name, estimator in models.items():
        for engineered in feature_sets:
            pipe = make_model_pipeline(estimator, engineered=engineered)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORERS, n_jobs=1)
            row: dict[str, object] = {
                "model": name,
                "features": "engineered" if engineered else "raw",
            }
            for metric in CV_SCORERS:
                vals = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std())
            rows.append(row)
            logger.info(
                "CV %-14s %-10s PR-AUC %.4f ± %.4f | ROC-AUC %.4f | recall@10%% %.4f",
                name,
                row["features"],
                row["pr_auc_mean"],
                row["pr_auc_std"],
                row["roc_auc_mean"],
                row["recall_top10_mean"],
            )
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
