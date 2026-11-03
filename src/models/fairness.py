"""Subgroup check: does the model treat demographic groups comparably?

For each group (gender, senior citizen, partner, dependents) this reports how well the model
ranks, whether its probabilities are calibrated, and how often and how accurately it flags
customers for a retention offer. Uses out-of-fold calibrated scores on train + validation
(the test set is untouched) and today's decision threshold.

Groups differ in true churn rate, so the contact rate is *expected* to differ. The checks that
indicate unequal treatment are calibration (predicted vs actual churn within the group) and
recall (the share of the group's churners the model reaches). Differences larger than
:data:`CALIBRATION_FLAG` or :data:`RECALL_FLAG` are flagged for a human to look at; a flag is
a prompt to investigate, not a verdict.

Run with ``python -m src.models.fairness`` (about a minute).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, TARGET_COL
from src.features.pipeline import split_xy
from src.models.evaluate import performance_by_segment
from src.models.oof import dev_frame, final_pipeline, oof_calibrated_scores
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

GROUP_COLS: list[str] = ["gender", "SeniorCitizen", "Partner", "Dependents"]
CALIBRATION_FLAG: float = 0.03  # |mean predicted - actual churn rate| within a group
RECALL_FLAG: float = 0.10  # recall gap versus the overall recall


def subgroup_report(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    group_cols: list[str] = GROUP_COLS,
) -> pd.DataFrame:
    """Per-group ranking, calibration and targeting metrics.

    Args:
        df: Rows aligned with ``scores``; must contain the target and ``group_cols``.
        scores: Calibrated churn probabilities.
        threshold: Customers scored at or above it are contacted.
        group_cols: Columns defining the groups.

    Returns:
        One row per (group column, value) with ``customers``, ``churn_rate``,
        ``mean_score``, ``calibration_gap``, ``contact_rate``, ``precision``, ``recall``,
        ``recall_gap`` (versus overall), AUCs, and a ``flags`` string.
    """
    scores = np.asarray(scores)
    data = df.reset_index(drop=True).assign(_score=scores)
    y_all = data[TARGET_COL].to_numpy()
    overall_recall = float((scores[y_all == 1] >= threshold).mean())

    base = performance_by_segment(df, scores, group_cols)
    rows = []
    for rec in base.to_dict("records"):
        part = data[data[rec["segment"]].astype(str) == rec["value"]]
        y, s = part[TARGET_COL].to_numpy(), part["_score"].to_numpy()
        contacted = s >= threshold
        recall = float(contacted[y == 1].mean()) if (y == 1).any() else float("nan")
        precision = float(y[contacted].mean()) if contacted.any() else float("nan")
        gap = rec["mean_score"] - rec["churn_rate"]
        recall_gap = recall - overall_recall
        flags = []
        if abs(gap) > CALIBRATION_FLAG:
            flags.append("calibration")
        if abs(recall_gap) > RECALL_FLAG:
            flags.append("recall")
        rows.append(
            {
                **rec,
                "calibration_gap": gap,
                "contact_rate": float(contacted.mean()),
                "precision": precision,
                "recall": recall,
                "recall_gap": recall_gap,
                "flags": ", ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def run_fairness() -> pd.DataFrame:
    """Run the subgroup check and save the table.

    Returns:
        The report (also written to ``reports/metrics/subgroup_report.csv``).
    """
    dev = dev_frame()
    X, y = split_xy(dev)
    scores = oof_calibrated_scores(final_pipeline(y), X, y)
    threshold = json.loads((METRICS_DIR / "business_summary.json").read_text())[
        "threshold_calibrated"
    ]
    report = subgroup_report(dev, scores, threshold)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(METRICS_DIR / "subgroup_report.csv", index=False)
    for r in report.itertuples():
        logger.info(
            "%-14s %-4s n=%4d churn %.1f%% contact %.1f%% recall %.1f%% calib gap %+.3f %s",
            r.segment,
            r.value,
            r.customers,
            100 * r.churn_rate,
            100 * r.contact_rate,
            100 * r.recall,
            r.calibration_gap,
            f"FLAG[{r.flags}]" if r.flags else "",
        )
    return report


if __name__ == "__main__":
    run_fairness()
