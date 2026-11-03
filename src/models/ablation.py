"""Sensitivity check for the snapshot-data caveat: how much does the model lean on tenure?

For churned customers, ``tenure`` and ``TotalCharges`` are measured when they left, while
active customers are measured today. A short tenure therefore partly means "already gone".
Without dated data the leak cannot be measured, but it can be bounded: remove the columns and
see how far PR-AUC falls. The fall is an upper bound on the exposure, not a measured leak,
because tenure is also a legitimate risk signal for active customers.

Columns are set to a constant 0, so they carry no information while the preprocessing
pipeline still finds every column it expects. Uses train + validation only; the test set is
never touched.

Run with ``python -m src.models.ablation`` (about a minute).
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ABLATION_REPEATS, CV_FOLDS, METRICS_DIR
from src.data.split import Splits, load_splits
from src.features.pipeline import split_xy
from src.models.baseline import make_baseline
from src.models.repeated_cv import make_tuned_pipeline, repeated_cv_scores
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SCENARIOS: dict[str, list[str]] = {
    "none": [],
    "TotalCharges": ["TotalCharges"],
    "tenure + TotalCharges": ["tenure", "TotalCharges"],
}


def zero_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy of ``X`` with ``columns`` set to 0 so they carry no information.

    Args:
        X: Feature frame.
        columns: Columns to neutralize. Must exist in ``X``.

    Returns:
        A new frame; ``X`` is not modified.

    Raises:
        KeyError: If a column is missing.
    """
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"Columns not in X: {missing}")
    return X.assign(**{c: 0.0 for c in columns})


def feature_ablation(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    scenarios: dict[str, list[str]] | None = None,
    n_splits: int = CV_FOLDS,
    n_repeats: int = ABLATION_REPEATS,
) -> pd.DataFrame:
    """Mean CV metrics per (scenario, model) with the given columns removed.

    Args:
        pipelines: Name -> unfitted pipeline, scored on identical folds in every scenario.
        X: Features.
        y: Binary labels.
        scenarios: Scenario name -> columns to neutralize. Defaults to :data:`SCENARIOS`.
        n_splits: Folds per repeat.
        n_repeats: Repeats.

    Returns:
        One row per (scenario, model) with mean ``pr_auc`` and ``roc_auc``.
    """
    scenarios = SCENARIOS if scenarios is None else scenarios
    rows = []
    for scenario, cols in scenarios.items():
        scores = repeated_cv_scores(
            pipelines,
            zero_columns(X, cols),
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
        means = scores.groupby("model", sort=False)[["pr_auc", "roc_auc"]].mean()
        for model, vals in means.iterrows():
            rows.append({"removed": scenario, "model": model, **vals.to_dict()})
    return pd.DataFrame(rows)


def run_feature_ablation(splits: Splits | None = None) -> pd.DataFrame:
    """Run the sensitivity check on train + validation and save the table.

    Args:
        splits: Splits to use; defaults to those saved by Phase 2.

    Returns:
        The result table (also written to ``reports/metrics/feature_ablation.csv``).
    """
    splits = splits if splits is not None else load_splits()
    dev = pd.concat([splits.train, splits.val], ignore_index=True)
    X, y = split_xy(dev)

    final_name = json.loads((METRICS_DIR / "model_summary.json").read_text())["final_model"]
    tuning = json.loads((METRICS_DIR / "tuning_results.json").read_text())
    pipelines = {
        "baseline": make_baseline(),
        "final": make_tuned_pipeline(final_name, tuning[final_name]["best_params"], y),
    }
    table = feature_ablation(pipelines, X, y)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "feature_ablation.csv", index=False)
    for row in table.itertuples():
        logger.info(
            "removed=%-22s %-8s PR-AUC %.4f | ROC-AUC %.4f",
            row.removed,
            row.model,
            row.pr_auc,
            row.roc_auc,
        )
    return table


if __name__ == "__main__":
    run_feature_ablation()
