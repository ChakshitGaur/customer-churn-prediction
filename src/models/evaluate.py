"""Evaluation: ranking metrics, top-decile targeting, business value, and diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (  # noqa: E402
    FIGURES_DIR,
    OFFER_SUCCESS_RATE,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    SEED,
    TARGET_COL,
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


# --- Phase 5: plots, gain table, calibration, segments --------------------------


def _new_ax(figsize: tuple[float, float] = (6, 4.5)) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_pr_curves(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> Figure:
    """Precision-recall curves for several models, with the no-skill line."""
    fig, ax = _new_ax()
    for name, s in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, s)
        ax.plot(recall, precision, label=f"{name} (AP {average_precision_score(y_true, s):.3f})")
    base = float(np.mean(y_true))
    ax.axhline(base, ls="--", c="grey", label=f"no skill ({base:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", ylim=(0, 1.02))
    ax.legend(loc="upper right")
    return fig


def plot_roc_curves(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> Figure:
    """ROC curves for several models, with the diagonal."""
    fig, ax = _new_ax()
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    return fig


def plot_confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Figure:
    """Confusion matrix at ``threshold`` with counts and row percentages."""
    cm = confusion_matrix(y_true, (np.asarray(y_score) >= threshold).astype(int))
    fig, ax = _new_ax((4.8, 4))
    ax.imshow(cm, cmap="Blues")
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.0%})", ha="center", va="center", c=color)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["stay", "churn"],
        yticklabels=["stay", "churn"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Confusion matrix (threshold {threshold:.2f})",
    )
    ax.grid(False)
    return fig


def gain_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: churn rate, lift, and cumulative share of churners captured.

    Decile 1 holds the highest-risk 10% of customers.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score)})
    df = df.sort_values("s", ascending=False, kind="stable").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    g = df.groupby("decile").agg(
        customers=("y", "size"), churners=("y", "sum"), min_score=("s", "min")
    )
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / df["y"].mean()
    g["cum_customers_pct"] = g["customers"].cumsum() / len(df)
    g["cum_churners_pct"] = g["churners"].cumsum() / df["y"].sum()
    return g


def plot_gain_lift(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> Figure:
    """Cumulative gain (left) and per-decile lift (right) for several models."""
    fig, (ax_gain, ax_lift) = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.8 / len(scores)
    for k, (name, s) in enumerate(scores.items()):
        g = gain_table(y_true, s)
        ax_gain.plot(
            np.r_[0, g["cum_customers_pct"]],
            np.r_[0, g["cum_churners_pct"]],
            marker="o",
            label=name,
        )
        offset = (k - (len(scores) - 1) / 2) * width
        ax_lift.bar(g.index + offset, g["lift"], width, label=name)
    ax_gain.plot([0, 1], [0, 1], ls="--", c="grey", label="random")
    ax_gain.set(
        xlabel="Share of customers contacted (highest risk first)",
        ylabel="Share of churners captured",
        title="Cumulative gain",
    )
    ax_gain.legend()
    ax_lift.axhline(1, ls="--", c="grey")
    ax_lift.set(xlabel="Risk decile (1 = highest)", ylabel="Lift", title="Lift by decile")
    ax_lift.set_xticks(range(1, 11))
    ax_lift.legend()
    fig.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, scores: dict[str, np.ndarray], n_bins: int = 10) -> Figure:
    """Reliability diagram: predicted probability vs. observed churn rate."""
    fig, ax = _new_ax()
    for name, s in scores.items():
        frac_pos, mean_pred = calibration_curve(y_true, s, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, s)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration")
    ax.legend(loc="upper left")
    return fig


def calibration_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Mean predicted vs. observed churn rate and expected calibration error (10 bins)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    return {
        "mean_predicted": float(y_score.mean()),
        "observed_rate": float(y_true.mean()),
        "ece": float(np.mean(np.abs(frac_pos - mean_pred))),
    }


def performance_by_segment(
    df: pd.DataFrame, y_score: np.ndarray, segment_cols: list[str], min_size: int = 30
) -> pd.DataFrame:
    """PR-AUC, ROC-AUC, churn rate, and mean score per segment value.

    Args:
        df: Rows aligned with ``y_score``; must contain the target and ``segment_cols``.
        y_score: Predicted churn probabilities.
        segment_cols: Columns to slice by.
        min_size: Segments smaller than this, or with a single class, get NaN AUCs.

    Returns:
        One row per (segment column, value).
    """
    data = df.reset_index(drop=True).assign(_score=np.asarray(y_score))
    rows = []
    for col in segment_cols:
        for value, part in data.groupby(col, observed=True):
            y, s = part[TARGET_COL].to_numpy(), part["_score"].to_numpy()
            ok = len(part) >= min_size and 0 < y.sum() < len(y)
            rows.append(
                {
                    "segment": col,
                    "value": str(value),
                    "customers": len(part),
                    "churn_rate": float(y.mean()),
                    "mean_score": float(s.mean()),
                    "pr_auc": float(average_precision_score(y, s)) if ok else np.nan,
                    "roc_auc": float(roc_auc_score(y, s)) if ok else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: Figure, name: str) -> Path:
    """Save to ``reports/figures/<name>.png`` and close the figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
