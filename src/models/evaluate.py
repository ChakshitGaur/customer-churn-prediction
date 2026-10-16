"""Evaluation metrics: ranking quality, top-decile targeting, and business value.

Phase 4 uses the metric functions; Phase 5 adds plots (PR/ROC, lift, calibration).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    OFFER_SUCCESS_RATE,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    SEED,
    TOP_DECILE,
)


def top_fraction_mask(y_score: np.ndarray, fraction: float = TOP_DECILE) -> np.ndarray:
    """Boolean mask of the highest-scored ``fraction`` of rows (at least one row).

    Ties are broken by a stable sort, so results are deterministic.
    """
    n_top = max(1, int(round(len(y_score) * fraction)))
    order = np.argsort(-np.asarray(y_score), kind="stable")
    mask = np.zeros(len(y_score), dtype=bool)
    mask[order[:n_top]] = True
    return mask


def recall_at_top(y_true: np.ndarray, y_score: np.ndarray, fraction: float = TOP_DECILE) -> float:
    """Share of all churners captured in the top ``fraction`` of risk scores."""
    y_true = np.asarray(y_true)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    return float(y_true[top_fraction_mask(y_score, fraction)].sum() / positives)


def lift_at_top(y_true: np.ndarray, y_score: np.ndarray, fraction: float = TOP_DECILE) -> float:
    """Churn rate in the top ``fraction`` divided by the overall churn rate.

    A lift of 2.5 means targeting the top 10% finds churners 2.5x more often than random.
    """
    y_true = np.asarray(y_true)
    base = y_true.mean()
    if base == 0:
        return 0.0
    return float(y_true[top_fraction_mask(y_score, fraction)].mean() / base)


def expected_profit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost: float = RETENTION_OFFER_COST,
    value: float = RETAINED_CUSTOMER_VALUE,
    success_rate: float = OFFER_SUCCESS_RATE,
) -> float:
    """Net value of contacting every customer scored at or above ``threshold``.

    Each contact costs ``cost``. A contacted true churner is saved with probability
    ``success_rate``, worth ``value``. Non-contacted customers cost and earn nothing.

    Returns:
        Expected profit in currency units (can be negative).
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(y_score) >= threshold
    true_pos = int((contacted & (y_true == 1)).sum())
    return float(true_pos * value * success_rate - contacted.sum() * cost)


def best_profit_threshold(
    y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Threshold that maximizes :func:`expected_profit` on the given data.

    Args:
        y_true: True labels (use the validation split, never test).
        y_score: Predicted churn probabilities.
        grid: Thresholds to try. Defaults to 0.01..0.99 in steps of 0.01.

    Returns:
        ``(threshold, profit)``.
    """
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    profits = [expected_profit(y_true, y_score, t) for t in grid]
    best = int(np.argmax(profits))
    return float(grid[best]), float(profits[best])


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """All project metrics for one set of predictions.

    Args:
        y_true: True labels.
        y_score: Predicted churn probabilities.
        threshold: If given, threshold-dependent metrics are included.

    Returns:
        Dict with ``pr_auc`` (primary), ``roc_auc``, ``recall_top10``, ``lift_top10``,
        ``brier``, and, with a threshold, ``precision``, ``recall``, ``f1``,
        ``contact_rate``, ``profit``, ``profit_per_customer``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_top10": recall_at_top(y_true, y_score),
        "lift_top10": lift_at_top(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(y_true.mean()),
    }
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        profit = expected_profit(y_true, y_score, threshold)
        out.update(
            threshold=float(threshold),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            contact_rate=float(y_pred.mean()),
            profit=profit,
            profit_per_customer=profit / len(y_true),
        )
    return out


def bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b) on the same rows.

    Returns:
        Dict with the observed difference, a 95% CI, and the share of resamples where
        ``a`` beats ``b``.
    """
    rng = np.random.default_rng(seed)
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(
            average_precision_score(y_true[idx], score_a[idx])
            - average_precision_score(y_true[idx], score_b[idx])
        )
    diffs_arr = np.array(diffs)
    return {
        "diff": float(
            average_precision_score(y_true, score_a) - average_precision_score(y_true, score_b)
        ),
        "ci_low": float(np.percentile(diffs_arr, 2.5)),
        "ci_high": float(np.percentile(diffs_arr, 97.5)),
        "p_a_better": float((diffs_arr > 0).mean()),
    }


def _recall_top10_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return recall_at_top(y_true, y_score)


# Scorers for cross-validation (PR-AUC is the primary / refit metric).
CV_SCORERS: dict[str, Any] = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall_top10": make_scorer(_recall_top10_score, response_method="predict_proba"),
}


def save_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as pretty JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path
