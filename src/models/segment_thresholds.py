"""Would a separate contact threshold per customer group make more money?

The model uses one global threshold. Groups differ in churn rate, so the contact rate and
recall differ by group (``python -m src.models.fairness``), which raises the question of
group-specific thresholds. This module answers the *profit* side of that question; it never
changes the production threshold.

For a calibrated model the profit-maximizing rule is the same for every group: contact a
customer when ``p >= cost / (value * success_rate)``. Per-group optima found on the data
therefore mostly fit noise. To test that honestly, the thresholds are chosen on one random
half of the data and judged on the other half, repeated many times; the in-sample gain is
reported next to it to show how optimistic it is.

Uses out-of-fold calibrated scores on train + validation; the test set is untouched.

Run with ``python -m src.models.segment_thresholds`` (about a minute).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    METRICS_DIR,
    OFFER_SUCCESS_RATE,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    SEED,
)
from src.features.pipeline import split_xy
from src.models.fairness import GROUP_COLS
from src.models.oof import dev_frame, final_pipeline, oof_calibrated_scores
from src.models.sensitivity import THRESHOLDS, profit_per_1000
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

N_REPEATS: int = 200  # random half-splits for the out-of-sample estimate
MIN_GROUP_SIZE: int = 30  # a group smaller than this keeps the global threshold


def _total_profit(y: np.ndarray, s: np.ndarray, thresholds: np.ndarray, **econ: float) -> float:
    """Profit (currency units) when each row is contacted at its own threshold."""
    contacted = s >= thresholds
    return float(
        (contacted & (y == 1)).sum() * econ["value"] * econ["rate"] - contacted.sum() * econ["cost"]
    )


def best_threshold(
    y: np.ndarray, s: np.ndarray, grid: np.ndarray = THRESHOLDS, **econ: float
) -> float:
    """Profit-maximizing threshold on ``grid`` (lowest on ties); empty input -> grid median."""
    if len(y) == 0:
        return float(np.median(grid))
    profits = [profit_per_1000(y, s, t, econ["cost"], econ["value"], econ["rate"])[0] for t in grid]
    return float(grid[int(np.argmax(profits))])


def group_thresholds(
    groups: pd.Series, y: np.ndarray, s: np.ndarray, fallback: float, **econ: float
) -> np.ndarray:
    """Per-row threshold: the group's own optimum, or ``fallback`` for tiny groups."""
    out = np.full(len(y), fallback, dtype=float)
    keys = groups.astype(str).to_numpy()
    for key in np.unique(keys):
        mask = keys == key
        if mask.sum() >= MIN_GROUP_SIZE:
            out[mask] = best_threshold(y[mask], s[mask], **econ)
    return out


def compare_thresholds(
    groups: pd.Series,
    y: np.ndarray,
    s: np.ndarray,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
    cost: float = RETENTION_OFFER_COST,
    value: float = RETAINED_CUSTOMER_VALUE,
    rate: float = OFFER_SUCCESS_RATE,
) -> dict[str, float]:
    """Profit per 1,000 customers: per-group thresholds versus one global threshold.

    Args:
        groups: Group label per row (e.g. ``Dependents``).
        y: True labels.
        s: Calibrated churn probabilities.
        n_repeats: Random half-splits for the out-of-sample estimate.
        seed: Random seed.
        cost, value, rate: Offer cost, saved-customer value and offer success rate.

    Returns:
        ``in_sample_gain`` (optimistic: thresholds and profit use the same rows) and the
        ``oos_gain_mean`` / ``oos_gain_p05`` / ``oos_gain_p95`` / ``oos_share_better`` over
        random halves (thresholds chosen on one half, profit measured on the other), all per
        1,000 customers. ``theory_threshold`` is the break-even probability.
    """
    econ = {"cost": cost, "value": value, "rate": rate}
    y, s = np.asarray(y), np.asarray(s)
    n = len(y)
    global_t = best_threshold(y, s, **econ)
    in_sample = _total_profit(y, s, group_thresholds(groups, y, s, global_t, **econ), **econ)
    baseline = _total_profit(y, s, np.full(n, global_t), **econ)

    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        a, b = order[: n // 2], order[n // 2 :]
        t_global = best_threshold(y[a], s[a], **econ)
        t_rows = group_thresholds(groups.iloc[a], y[a], s[a], t_global, **econ)
        # Look up each held-out row's group threshold from the fitted halves.
        lookup = dict(zip(groups.iloc[a].astype(str), t_rows, strict=True))
        t_b = np.array([lookup.get(g, t_global) for g in groups.iloc[b].astype(str)])
        gain = _total_profit(y[b], s[b], t_b, **econ) - _total_profit(
            y[b], s[b], np.full(len(b), t_global), **econ
        )
        gains.append(1000 * gain / len(b))
    gains_arr = np.array(gains)
    return {
        "global_threshold": global_t,
        "theory_threshold": cost / (value * rate),
        "in_sample_gain": 1000 * (in_sample - baseline) / n,
        "oos_gain_mean": float(gains_arr.mean()),
        "oos_gain_p05": float(np.percentile(gains_arr, 5)),
        "oos_gain_p95": float(np.percentile(gains_arr, 95)),
        "oos_share_better": float((gains_arr > 0).mean()),
    }


def run_segment_thresholds() -> pd.DataFrame:
    """Compare per-group and global thresholds for each group column and save the table.

    Returns:
        One row per group column (also written to
        ``reports/metrics/segment_thresholds.csv``).
    """
    dev = dev_frame()
    X, y = split_xy(dev)
    scores = oof_calibrated_scores(final_pipeline(y), X, y)
    rows = []
    for col in GROUP_COLS:
        result = compare_thresholds(dev[col].reset_index(drop=True), y.to_numpy(), scores)
        rows.append({"group": col, **result})
        logger.info(
            "%-14s in-sample gain %+7.0f | out-of-sample %+7.0f [%+.0f, %+.0f] per 1,000, "
            "better in %.0f%% of splits",
            col,
            result["in_sample_gain"],
            result["oos_gain_mean"],
            result["oos_gain_p05"],
            result["oos_gain_p95"],
            100 * result["oos_share_better"],
        )
    table = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "segment_thresholds.csv", index=False)
    (METRICS_DIR / "segment_thresholds.json").write_text(
        json.dumps(table.round(4).to_dict("records"), indent=2)
    )
    return table


if __name__ == "__main__":
    run_segment_thresholds()
