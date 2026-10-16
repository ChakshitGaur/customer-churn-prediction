"""Phase 5 orchestration: diagnostic plots, segment performance, and SHAP explanations.

Uses the models saved in Phase 4 and the **test** predictions already scored there.
Nothing here changes the model or threshold; it only describes the final result.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from src.config import METRICS_DIR, MODELS_DIR
from src.data.split import Splits
from src.explain.shap_analysis import (
    compute_shap,
    global_importance,
    group_shap,
    plot_beeswarm,
    plot_customer_waterfall,
    plot_global_importance,
    top_reasons,
)
from src.features.build_features import add_features
from src.features.pipeline import split_xy
from src.models.evaluate import (
    calibration_summary,
    gain_table,
    performance_by_segment,
    plot_calibration,
    plot_confusion,
    plot_gain_lift,
    plot_pr_curves,
    plot_roc_curves,
    save_figure,
    save_json,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SEGMENT_COLS: list[str] = [
    "Contract",
    "tenure_band",
    "InternetService",
    "PaymentMethod",
    "SeniorCitizen",
]


def example_customers(y_score: np.ndarray, y_true: np.ndarray, threshold: float) -> dict[str, int]:
    """Pick three illustrative test customers deterministically.

    Returns:
        Row positions for the highest-risk churner, the customer closest to the
        threshold, and the lowest-risk customer.
    """
    churners = np.flatnonzero(y_true == 1)
    return {
        "high_risk": int(churners[np.argmax(y_score[churners])]),
        "borderline": int(np.argmin(np.abs(y_score - threshold))),
        "low_risk": int(np.argmin(y_score)),
    }


def run_evaluation(splits: Splits) -> dict[str, Any]:
    """Run all of Phase 5 and save figures and metric tables.

    Args:
        splits: Train/val/test splits from Phase 2.

    Returns:
        Summary dict (also written to ``reports/metrics/evaluation_summary.json``).
    """
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    baseline = joblib.load(MODELS_DIR / "baseline.joblib")
    X_test, y_test = split_xy(splits.test)
    y = y_test.to_numpy()
    threshold = final["threshold"]

    final_name = f"{final['name']} (final)"
    scores = {
        final_name: final["pipeline"].predict_proba(X_test)[:, 1],
        "logreg baseline": baseline["pipeline"].predict_proba(X_test)[:, 1],
    }
    s_final = scores[final_name]

    # Curves and diagnostics
    save_figure(plot_pr_curves(y, scores), "eval_pr_curve")
    save_figure(plot_roc_curves(y, scores), "eval_roc_curve")
    save_figure(plot_confusion(y, s_final, threshold), "eval_confusion_matrix")
    save_figure(plot_gain_lift(y, scores), "eval_gain_lift")
    save_figure(plot_calibration(y, scores), "eval_calibration")

    gains = gain_table(y, s_final)
    gains.to_csv(METRICS_DIR / "gain_table.csv")

    # Segment performance (engineered tenure_band used for slicing only)
    segment_df = add_features(splits.test)
    segments = performance_by_segment(segment_df, s_final, SEGMENT_COLS)
    segments.to_csv(METRICS_DIR / "segment_performance.csv", index=False)

    # SHAP
    sv = compute_shap(final["pipeline"], X_test)
    grouped = group_shap(sv)
    importance = global_importance(grouped)
    importance.to_csv(METRICS_DIR / "shap_importance.csv", index_label="feature")
    save_figure(plot_global_importance(importance), "shap_global_importance")
    save_figure(plot_beeswarm(sv), "shap_beeswarm")

    examples = example_customers(s_final, y, threshold)
    reasons = top_reasons(grouped, X_test)
    base_value = float(np.ravel(sv.base_values)[0])
    example_info = {}
    for label, row in examples.items():
        title = f"{label.replace('_', ' ')}: churn probability {s_final[row]:.2f}"
        save_figure(
            plot_customer_waterfall(grouped, X_test, base_value, row, title=title),
            f"shap_customer_{label}",
        )
        example_info[label] = {
            "customer_id": str(splits.test.iloc[row]["customerID"]),
            "probability": float(s_final[row]),
            "actual_churn": int(y[row]),
            "top_reasons": reasons[row],
        }

    summary = {
        "threshold": threshold,
        "calibration": {name: calibration_summary(y, s) for name, s in scores.items()},
        "top_decile": gains.loc[1].to_dict(),
        "top3_deciles_capture": float(gains.loc[3, "cum_churners_pct"]),
        "shap_top_features": importance.head(10)["share"].round(4).to_dict(),
        "example_customers": example_info,
    }
    save_json(summary, METRICS_DIR / "evaluation_summary.json")
    logger.info(
        "Evaluation saved: top decile lift %.2f, top 30%% captures %.1f%% of churners",
        gains.loc[1, "lift"],
        100 * summary["top3_deciles_capture"],
    )
    return summary
