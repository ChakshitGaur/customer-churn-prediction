"""Explainable Boosting Machine (EBM): an interpretable alternative to the XGBoost model.

An EBM is a generalized additive model fit by boosting: each feature gets its own learned
shape function and the score is their sum, so every prediction decomposes exactly into
per-feature contributions. No SHAP approximation is needed.

This module only *compares* the EBM against the current final model and baseline using the
repeated-CV scheme from :mod:`src.models.repeated_cv` on train + validation. The test set is
never touched and the shipped model is unchanged: replacing it would be a decision to make
with these results, not something this module does.

Pairwise interaction terms are off (``interactions=0``): the model stays purely additive,
which keeps it fully explainable, and fitting takes seconds instead of minutes.

Requires ``interpret-core`` (see ``requirements.txt``). Run with ``python -m src.models.ebm``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from src.config import CV_REPEATS, METRICS_DIR, SEED
from src.data.split import Splits, load_splits
from src.explain.shap_analysis import original_feature
from src.features.pipeline import make_model_pipeline, split_xy
from src.models.baseline import make_baseline
from src.models.evaluate import save_json
from src.models.repeated_cv import (
    METRICS,
    corrected_paired_test,
    make_tuned_pipeline,
    repeated_cv_scores,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EBM_PARAMS: dict[str, Any] = {
    "interactions": 0,
    "outer_bags": 8,
    "learning_rate": 0.02,
}


def make_ebm_pipeline(seed: int = SEED, engineered: bool = False) -> Pipeline:
    """Unfitted preprocessing + EBM pipeline.

    Args:
        seed: Random seed for the EBM.
        engineered: Whether to include engineered features (raw by default, like the final
            model).

    Returns:
        ``Pipeline([("prep", ...), ("model", ExplainableBoostingClassifier)])``.
    """
    ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1, **EBM_PARAMS)
    return make_model_pipeline(ebm, engineered=engineered)


def importance_by_feature(pipeline: Pipeline) -> pd.DataFrame:
    """Global importance of each original feature in a fitted EBM pipeline.

    Importance is the mean absolute contribution of a term to the log-odds. One-hot columns
    are summed back to their source feature, matching how SHAP importance is reported.

    Args:
        pipeline: Fitted pipeline from :func:`make_ebm_pipeline`.

    Returns:
        Frame with ``feature``, ``importance`` and ``share``, sorted by importance.
    """
    model = pipeline.named_steps["model"]
    terms = pd.DataFrame(
        {
            "feature": [original_feature(n) for n in model.term_names_],
            "importance": np.asarray(model.term_importances(), dtype=float),
        }
    )
    out = terms.groupby("feature", as_index=False)["importance"].sum()
    out["share"] = out["importance"] / out["importance"].sum()
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def compare_scores(scores: pd.DataFrame, n_train: int, n_test: int) -> dict[str, Any]:
    """Per-model means and paired EBM-vs-others tests from repeated-CV scores.

    Args:
        scores: Output of :func:`repeated_cv_scores` including an ``ebm`` model.
        n_train: Training rows per fold.
        n_test: Test rows per fold.

    Returns:
        JSON-ready dict with ``per_model`` means and ``paired`` Nadeau-Bengio tests.
    """
    per_model = {
        name: {m: float(g[m].mean()) for m in METRICS}
        for name, g in scores.groupby("model", sort=False)
    }
    wide = {
        m: scores.pivot_table(index=["repeat", "fold"], columns="model", values=m) for m in METRICS
    }
    paired = {
        f"ebm_minus_{other}": {
            m: corrected_paired_test((wide[m]["ebm"] - wide[m][other]).to_numpy(), n_train, n_test)
            for m in METRICS
        }
        for other in scores["model"].unique()
        if other != "ebm"
    }
    return {"per_model": per_model, "paired": paired}


def run_ebm_comparison(splits: Splits | None = None) -> dict[str, Any]:
    """Compare the EBM with the final model and baseline, and save results and importances.

    Args:
        splits: Splits to use; defaults to those saved by Phase 2. Only train and
            validation are used.

    Returns:
        Summary dict (also written to ``reports/metrics/ebm_comparison.json``); the EBM's
        feature importances go to ``reports/metrics/ebm_importance.csv``.
    """
    splits = splits if splits is not None else load_splits()
    dev = pd.concat([splits.train, splits.val], ignore_index=True)
    X, y = split_xy(dev)

    final_name = json.loads((METRICS_DIR / "model_summary.json").read_text())["final_model"]
    tuning = json.loads((METRICS_DIR / "tuning_results.json").read_text())
    pipelines = {
        "baseline": make_baseline(),
        "final": make_tuned_pipeline(final_name, tuning[final_name]["best_params"], y),
        "ebm": make_ebm_pipeline(),
    }
    scores = repeated_cv_scores(pipelines, X, y)
    n_test = len(dev) // 5
    result = compare_scores(scores, n_train=len(dev) - n_test, n_test=n_test)
    result.update(
        final_model=final_name, ebm_params=EBM_PARAMS, dev_rows=len(dev), repeats=CV_REPEATS
    )

    importance = importance_by_feature(make_ebm_pipeline().fit(X, y))
    shap_path = METRICS_DIR / "shap_importance.csv"
    if shap_path.exists():
        shap_imp = pd.read_csv(shap_path).rename(columns={"share": "shap_share"})
        importance = importance.merge(shap_imp[["feature", "shap_share"]], on="feature", how="left")
        both = importance.dropna(subset=["shap_share"])
        rho = spearmanr(both["share"], both["shap_share"]).statistic
        result["importance_rank_correlation_with_shap"] = float(rho)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(METRICS_DIR / "ebm_importance.csv", index=False)
    save_json(result, METRICS_DIR / "ebm_comparison.json")

    for name, vals in result["per_model"].items():
        logger.info("%-9s PR-AUC %.4f | ROC-AUC %.4f", name, vals["pr_auc"], vals["roc_auc"])
    for key, tests in result["paired"].items():
        t = tests["pr_auc"]
        logger.info(
            "%s PR-AUC %+.4f [95%% CI %+.4f, %+.4f] p=%.3f",
            key,
            t["mean_diff"],
            t["ci_low"],
            t["ci_high"],
            t["p_value"],
        )
    logger.info("Top EBM features: %s", ", ".join(importance["feature"].head(6)))
    return result


if __name__ == "__main__":
    run_ebm_comparison()
