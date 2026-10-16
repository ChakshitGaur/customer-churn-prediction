"""Hyperparameter tuning (train split, CV) and cost-based threshold selection (val split)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.features.pipeline import build_preprocessor, make_model_pipeline
from src.models.evaluate import CV_SCORERS, best_profit_threshold
from src.models.train import candidate_models, make_cv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

N_ITER: int = 40

# Search spaces. ``prep`` lets the search choose raw vs. engineered features as well.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logreg": {
        "model__C": loguniform(1e-3, 1e2),
        "model__l1_ratio": [0.0, 0.5, 1.0],
        "model__solver": ["saga"],
    },
    "random_forest": {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": [None, 6, 8, 12, 16],
        "model__min_samples_leaf": randint(2, 40),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(0.005, 0.2),
        "model__max_depth": randint(2, 6),
        "model__min_child_weight": randint(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.4, 0.6),
        "model__reg_lambda": loguniform(0.1, 50),
        "model__gamma": uniform(0, 2),
    },
}


def tune_model(
    name: str, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = N_ITER
) -> tuple[Pipeline, dict[str, Any]]:
    """Randomized search over one model's space (and the feature set) on the training split.

    Args:
        name: Key in :data:`SEARCH_SPACES` and :func:`candidate_models`.
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of sampled configurations.

    Returns:
        ``(best_pipeline_refit_on_train, summary)`` where ``summary`` has the best params
        and CV metrics of the best configuration.
    """
    estimator: BaseEstimator = candidate_models(y_train)[name]
    space = {
        **SEARCH_SPACES[name],
        "prep": [build_preprocessor(engineered=True), build_preprocessor(engineered=False)],
    }
    search = RandomizedSearchCV(
        make_model_pipeline(estimator),
        space,
        n_iter=n_iter,
        scoring=CV_SCORERS,
        refit="pr_auc",
        cv=make_cv(),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    i = search.best_index_
    res = search.cv_results_
    params = {
        k: (v if k != "prep" else ("engineered" if "features" in v.named_steps else "raw"))
        for k, v in search.best_params_.items()
    }
    summary = {
        "model": name,
        "best_params": {k: _to_builtin(v) for k, v in params.items()},
        "cv_pr_auc_mean": float(res["mean_test_pr_auc"][i]),
        "cv_pr_auc_std": float(res["std_test_pr_auc"][i]),
        "cv_roc_auc_mean": float(res["mean_test_roc_auc"][i]),
        "cv_recall_top10_mean": float(res["mean_test_recall_top10"][i]),
    }
    logger.info(
        "Tuned %-14s CV PR-AUC %.4f ± %.4f | features=%s",
        name,
        summary["cv_pr_auc_mean"],
        summary["cv_pr_auc_std"],
        params["prep"],
    )
    return search.best_estimator_, summary


def choose_threshold(y_val: pd.Series, val_scores: np.ndarray) -> dict[str, float]:
    """Pick the decision threshold that maximizes expected profit on the validation split.

    Args:
        y_val: Validation labels.
        val_scores: Validation churn probabilities from a model fit on train only.

    Returns:
        Dict with ``threshold`` and the validation ``profit`` it achieves.
    """
    threshold, profit = best_profit_threshold(np.asarray(y_val), val_scores)
    logger.info("Profit-maximizing threshold on val: %.2f (profit %.0f)", threshold, profit)
    return {"threshold": threshold, "val_profit": profit}


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON."""
    return value.item() if isinstance(value, np.generic) else value
