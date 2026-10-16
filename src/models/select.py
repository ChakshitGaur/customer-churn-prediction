"""Phase 4 orchestration: CV comparison -> tuning -> val selection -> threshold -> test once.

Decision rules (all decided before the test set is scored):
    * Candidates are cross-validated on train; the top models are tuned on train (CV).
    * The final model is the tuned model with the best **validation** PR-AUC.
    * Thresholds (final model and baseline) maximize expected profit on **validation**.
    * The test set is scored exactly once, at the end, for both models. Nothing is
      chosen based on test results.
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR
from src.data.split import Splits
from src.features.pipeline import split_xy
from src.models.baseline import make_baseline
from src.models.evaluate import bootstrap_pr_auc_diff, compute_metrics, save_json
from src.models.train import cross_validate_models
from src.models.tune import choose_threshold, tune_model
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

TUNE_CANDIDATES: list[str] = ["logreg", "random_forest", "xgboost"]


def run_modeling(splits: Splits) -> dict[str, Any]:
    """Run all of Phase 4 and save models and metrics.

    Args:
        splits: Train/val/test splits from Phase 2.

    Returns:
        Summary dict (also written to ``reports/metrics/model_summary.json``).
    """
    X_train, y_train = split_xy(splits.train)
    X_val, y_val = split_xy(splits.val)
    X_test, y_test = split_xy(splits.test)

    # 1. Untuned comparison on train CV (both feature sets)
    cv_results = cross_validate_models(X_train, y_train)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(METRICS_DIR / "cv_results.csv", index=False)

    # 2. Tune candidates on train CV; score each once on val
    tuned: dict[str, Any] = {}
    tuning_report: dict[str, Any] = {}
    for name in TUNE_CANDIDATES:
        pipe, summary = tune_model(name, X_train, y_train)
        summary["val"] = compute_metrics(y_val, pipe.predict_proba(X_val)[:, 1])
        tuned[name], tuning_report[name] = pipe, summary
    save_json(tuning_report, METRICS_DIR / "tuning_results.json")

    # 3. Select final model on validation PR-AUC
    best_name = max(tuning_report, key=lambda n: tuning_report[n]["val"]["pr_auc"])
    final_model = tuned[best_name]
    logger.info(
        "Selected %s (val PR-AUC %.4f)", best_name, tuning_report[best_name]["val"]["pr_auc"]
    )

    # 4. Baseline (untuned LR, raw features) fit on train
    baseline = make_baseline().fit(X_train, y_train)

    # 5. Thresholds from validation profit
    final_val = final_model.predict_proba(X_val)[:, 1]
    base_val = baseline.predict_proba(X_val)[:, 1]
    final_thr = choose_threshold(y_val, final_val)
    base_thr = choose_threshold(y_val, base_val)

    # 6. Test set: scored once, for both models
    final_test = final_model.predict_proba(X_test)[:, 1]
    base_test = baseline.predict_proba(X_test)[:, 1]
    test_final = compute_metrics(y_test, final_test, final_thr["threshold"])
    test_base = compute_metrics(y_test, base_test, base_thr["threshold"])
    diff = bootstrap_pr_auc_diff(y_test.to_numpy(), final_test, base_test)
    beats = test_final["pr_auc"] > test_base["pr_auc"]

    summary = {
        "final_model": best_name,
        "final_params": tuning_report[best_name]["best_params"],
        "threshold": final_thr["threshold"],
        "baseline_threshold": base_thr["threshold"],
        "val": {
            "final": compute_metrics(y_val, final_val, final_thr["threshold"]),
            "baseline": compute_metrics(y_val, base_val, base_thr["threshold"]),
        },
        "test": {"final": test_final, "baseline": test_base},
        "test_pr_auc_diff_bootstrap": diff,
        "beats_baseline_on_test": bool(beats),
    }
    save_json(summary, METRICS_DIR / "model_summary.json")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": final_model, "threshold": final_thr["threshold"], "name": best_name},
        MODELS_DIR / "final_model.joblib",
    )
    joblib.dump(
        {"pipeline": baseline, "threshold": base_thr["threshold"], "name": "baseline_logreg"},
        MODELS_DIR / "baseline.joblib",
    )

    logger.info(
        "TEST  final %s PR-AUC %.4f vs baseline %.4f | diff %.4f [95%% CI %.4f, %.4f] | beats=%s",
        best_name,
        test_final["pr_auc"],
        test_base["pr_auc"],
        diff["diff"],
        diff["ci_low"],
        diff["ci_high"],
        beats,
    )
    return summary


def results_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Side-by-side val/test metrics for the final model and the baseline."""
    rows = []
    for split in ("val", "test"):
        for who in ("final", "baseline"):
            rows.append({"split": split, "model": who, **summary[split][who]})
    return pd.DataFrame(rows)
