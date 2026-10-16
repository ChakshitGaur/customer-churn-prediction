"""Logistic regression baseline: the model every candidate has to beat."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.features.pipeline import make_model_pipeline


def make_baseline(engineered: bool = False) -> Pipeline:
    """Create the unfitted baseline pipeline.

    Raw features only by default: the baseline should be the simplest reasonable model,
    so any gain from engineered features or non-linear models shows up against it.

    Args:
        engineered: Whether to include engineered features.

    Returns:
        Preprocessing + ``LogisticRegression(class_weight="balanced")`` pipeline.
    """
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    return make_model_pipeline(model, engineered=engineered)
