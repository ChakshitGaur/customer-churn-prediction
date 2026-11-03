"""Repeated stratified K-fold comparison of the final model against the baselines.

The held-out test set is 1,057 customers (~280 churners), so its PR-AUC gain over the baseline
has a wide confidence interval. This module re-estimates the gain on the development data
(train + validation) with repeated stratified K-fold, which gives ``CV_REPEATS * CV_FOLDS``
paired scores per model. The test set is never touched.

Models compared (same folds, preprocessing refit inside every fold):
    * ``baseline``: untuned logistic regression on raw features (the project baseline).
    * ``logreg_tuned``: logistic regression with the best tuned params from Phase 4.
    * ``final``: the selected model with its tuned params from Phase 4.

Caveat: the tuned params were chosen with CV on the *train* split, which is part of the
development data used here. Both tuned models share that advantage over the untuned
baseline, so ``final`` vs ``logreg_tuned`` is the fairest comparison of algorithms.
The estimates are therefore slightly optimistic for tuned models vs. an untouched test set.

Run with ``python -m src.models.repeated_cv`` (a few minutes).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, CV_REPEATS, METRICS_DIR, SEED
from src.data.split import Splits, load_splits
from src.features.pipeline import make_model_pipeline, split_xy
from src.models.baseline import make_baseline
from src.models.evaluate import compute_metrics, save_json
from src.models.train import candidate_models
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

METRICS: tuple[str, ...] = ("pr_auc", "roc_auc", "recall_top10", "lift_top10")


def make_tuned_pipeline(name: str, params: dict[str, Any], y: pd.Series) -> Pipeline:
    """Build a model pipeline from saved tuning params (``model__*`` keys and ``prep``).

    Args:
        name: Model name known to :func:`candidate_models`.
        params: Best params as saved in ``tuning_results.json``.
        y: Labels, used only for XGBoost's ``scale_pos_weight``.

    Returns:
        Unfitted pipeline with the params applied.
    """
    params = dict(params)
    engineered = params.pop("prep", "raw") == "engineered"
    pipe = make_model_pipeline(candidate_models(y)[name], engineered=engineered)
    return pipe.set_params(**params)


def repeated_cv_scores(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Score every pipeline on the same repeated stratified folds.

    Args:
        pipelines: Name -> unfitted pipeline. Each is cloned and refit per fold.
        X: Features.
        y: Binary labels.
        n_splits: Folds per repeat.
        n_repeats: Number of repeats.
        seed: Seed for the fold generator.

    Returns:
        Long frame with one row per (model, repeat, fold) and one column per metric.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(i, n_splits)
        for name, pipe in pipelines.items():
            fitted = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            scores = fitted.predict_proba(X.iloc[te])[:, 1]
            metrics = compute_metrics(y.iloc[te], scores)
            rows.append(
                {"model": name, "repeat": repeat, "fold": fold, **{m: metrics[m] for m in METRICS}}
            )
        if (i + 1) % n_splits == 0:
            logger.info("Repeat %d/%d done", repeat + 1, n_repeats)
    return pd.DataFrame(rows)


def corrected_paired_test(
    diffs: np.ndarray, n_train: int, n_test: int, alpha: float = 0.05
) -> dict[str, float]:
    """Nadeau-Bengio corrected resampled t-test on paired per-fold differences.

    Folds from repeated CV overlap, so their scores are positively correlated. The plain
    t-test then understates the variance; the correction inflates it by ``n_test / n_train``.

    Args:
        diffs: Per-fold metric differences (model A minus model B), same folds.
        n_train: Training rows per fold.
        n_test: Test rows per fold.
        alpha: Two-sided significance level for the interval.

    Returns:
        Mean difference, corrected standard error, CI bounds, p-value and the share of
        folds where A beat B.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean = float(diffs.mean())
    var = float(diffs.var(ddof=1))
    se = float(np.sqrt((1.0 / n + n_test / n_train) * var))
    if se == 0.0:
        return {"mean_diff": mean, "se": 0.0, "ci_low": mean, "ci_high": mean, "p_value": 1.0,
                "win_rate": float((diffs > 0).mean())}  # fmt: skip
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return {
        "mean_diff": mean,
        "se": se,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "p_value": float(2 * stats.t.sf(abs(mean / se), df=n - 1)),
        "win_rate": float((diffs > 0).mean()),
    }


def summarize(
    scores: pd.DataFrame, n_train: int, n_test: int, reference: str = "baseline"
) -> dict[str, Any]:
    """Per-model mean/std and paired differences against ``reference`` and tuned LR.

    Args:
        scores: Output of :func:`repeated_cv_scores`.
        n_train: Training rows per fold.
        n_test: Test rows per fold.
        reference: Model name the ``final`` model is compared against first.

    Returns:
        JSON-ready dict with ``per_model`` and ``paired`` sections.
    """
    per_model = {
        name: {m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1))} for m in METRICS}
        for name, g in scores.groupby("model", sort=False)
    }
    wide = {
        m: scores.pivot_table(index=["repeat", "fold"], columns="model", values=m) for m in METRICS
    }
    paired: dict[str, Any] = {}
    for other in (reference, "logreg_tuned"):
        if other not in scores["model"].unique() or other == "final":
            continue
        paired[f"final_minus_{other}"] = {
            m: corrected_paired_test(
                (wide[m]["final"] - wide[m][other]).to_numpy(), n_train, n_test
            )
            for m in METRICS
        }
    return {"n_folds_total": int(len(wide["pr_auc"])), "per_model": per_model, "paired": paired}


def run_repeated_cv(splits: Splits | None = None) -> dict[str, Any]:
    """Run the repeated-CV comparison on train + validation and save the results.

    Args:
        splits: Splits to use; defaults to those saved by Phase 2. Only ``train`` and
            ``val`` are used.

    Returns:
        The summary dict (also written to ``reports/metrics/repeated_cv_summary.json``).
    """
    splits = splits if splits is not None else load_splits()
    dev = pd.concat([splits.train, splits.val], ignore_index=True)
    X, y = split_xy(dev)

    summary_in = json.loads((METRICS_DIR / "model_summary.json").read_text())
    tuning = json.loads((METRICS_DIR / "tuning_results.json").read_text())
    final_name = summary_in["final_model"]
    pipelines = {
        "baseline": make_baseline(),
        "logreg_tuned": make_tuned_pipeline("logreg", tuning["logreg"]["best_params"], y),
        "final": make_tuned_pipeline(final_name, tuning[final_name]["best_params"], y),
    }

    logger.info(
        "Repeated CV: %d x %d folds on %d dev rows (test set untouched)",
        CV_REPEATS,
        CV_FOLDS,
        len(dev),
    )
    scores = repeated_cv_scores(pipelines, X, y)
    n_test = len(dev) // CV_FOLDS
    result = summarize(scores, n_train=len(dev) - n_test, n_test=n_test)
    result["final_model"] = final_name
    result["dev_rows"] = len(dev)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(METRICS_DIR / "repeated_cv_scores.csv", index=False)
    save_json(result, METRICS_DIR / "repeated_cv_summary.json")

    for name, vals in result["per_model"].items():
        logger.info("%-13s PR-AUC %.4f ± %.4f", name, vals["pr_auc"]["mean"], vals["pr_auc"]["std"])
    for key, tests in result["paired"].items():
        t = tests["pr_auc"]
        logger.info(
            "%s PR-AUC %+.4f [95%% CI %+.4f, %+.4f] p=%.3f win=%.0f%%",
            key,
            t["mean_diff"],
            t["ci_low"],
            t["ci_high"],
            t["p_value"],
            100 * t["win_rate"],
        )
    return result


if __name__ == "__main__":
    run_repeated_cv()
