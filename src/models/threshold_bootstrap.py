"""How stable is the decision threshold, and what does a poorly chosen one cost?

The threshold is the probability above which a customer is contacted. It is chosen to maximize
expected profit on the validation split, which holds only ~1,000 customers (~280 churners), so
the choice is noisy. This module resamples validation-sized sets from the out-of-fold
calibrated scores (train + validation; the test set is untouched), re-picks the threshold
each time, and measures:

* the spread of the chosen threshold, and
* the **regret**: profit per 1,000 customers lost on the whole development data by using the
  resampled threshold instead of the best one for that data.

The development data contains the resampled rows, so "the best threshold on all of it" is a
stand-in for the truth, slightly flattering to the resamples; read regret as a lower bound.

Run with ``python -m src.models.threshold_bootstrap`` (about a minute).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    METRICS_DIR,
    OFFER_SUCCESS_RATE,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    SEED,
    TARGET_COL,
    THRESHOLD_BOOTSTRAP_N,
)
from src.features.pipeline import split_xy
from src.models.evaluate import save_json
from src.models.oof import dev_frame, final_pipeline, oof_calibrated_scores
from src.models.sensitivity import THRESHOLDS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def profit_curve(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
    cost: float = RETENTION_OFFER_COST,
    value: float = RETAINED_CUSTOMER_VALUE,
    rate: float = OFFER_SUCCESS_RATE,
) -> np.ndarray:
    """Expected profit per 1,000 customers at every threshold, computed in one pass.

    Args:
        y: True labels.
        scores: Calibrated churn probabilities.
        thresholds: Thresholds to evaluate.
        cost: Cost of one offer.
        value: Value of a saved customer.
        rate: Probability that a contacted churner is saved.

    Returns:
        Array aligned with ``thresholds``; customers scored at or above a threshold are
        contacted.
    """
    y = np.asarray(y)
    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    # positives at or above each sorted position (suffix sums), with a trailing 0
    pos_above = np.concatenate([np.cumsum(y[order][::-1])[::-1], [0]])
    start = np.searchsorted(sorted_scores, thresholds, side="left")
    contacts = len(y) - start
    profit = pos_above[start] * value * rate - contacts * cost
    return 1000.0 * profit / len(y)


def bootstrap_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    sample_size: int,
    n_boot: int = THRESHOLD_BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bootstrap the profit-maximizing threshold and its regret.

    Args:
        y: True labels of the development data.
        scores: Out-of-fold calibrated probabilities.
        sample_size: Rows per resample (the validation split size).
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        Dict with the full-data optimum, the resampled thresholds' distribution, the regret
        distribution, and the full-data profit curve with a 90% resampling band.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    full = profit_curve(y, scores)
    best_full = int(np.argmax(full))
    rng = np.random.default_rng(seed)

    chosen = np.empty(n_boot)
    band = np.empty((n_boot, len(THRESHOLDS)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), sample_size)
        curve = profit_curve(y[idx], scores[idx])
        band[b] = curve
        chosen[b] = THRESHOLDS[int(np.argmax(curve))]
    regret = full[best_full] - full[np.searchsorted(THRESHOLDS, chosen)]

    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {
        "n_boot": n_boot,
        "sample_size": sample_size,
        "full_data_best_threshold": float(THRESHOLDS[best_full]),
        "full_data_best_profit_per_1000": float(full[best_full]),
        "threshold": {
            "mean": float(chosen.mean()),
            "std": float(chosen.std(ddof=1)),
            "p05": q(chosen, 5),
            "median": q(chosen, 50),
            "p95": q(chosen, 95),
        },
        "regret_per_1000": {
            "mean": float(regret.mean()),
            "p95": q(regret, 95),
            "share_within_500": float((regret <= 500).mean()),
            "share_within_1000": float((regret <= 1000).mean()),
        },
        "curve": {
            "thresholds": THRESHOLDS.tolist(),
            "profit_per_1000": full.tolist(),
            "band_p05": np.percentile(band, 5, axis=0).tolist(),
            "band_p95": np.percentile(band, 95, axis=0).tolist(),
        },
    }


def flat_region(curve: dict[str, Any], within: float = 0.95) -> tuple[float, float]:
    """Range of thresholds whose profit is within ``within`` of the best.

    Args:
        curve: The ``curve`` entry of :func:`bootstrap_thresholds`.
        within: Fraction of the best profit a threshold must reach.

    Returns:
        ``(lowest, highest)`` such threshold.
    """
    t = np.asarray(curve["thresholds"])
    p = np.asarray(curve["profit_per_1000"])
    ok = t[p >= within * p.max()]
    return float(ok.min()), float(ok.max())


def run_threshold_bootstrap() -> dict[str, Any]:
    """Run the analysis on train + validation and save the results.

    Returns:
        The result dict (also written to ``reports/metrics/threshold_bootstrap.json``, with
        the curve in ``threshold_profit_curve.csv``).
    """
    from src.data.split import load_splits

    splits = load_splits()
    dev = dev_frame(splits)
    X, y = split_xy(dev)
    scores = oof_calibrated_scores(final_pipeline(y), X, y)
    result = bootstrap_thresholds(dev[TARGET_COL].to_numpy(), scores, sample_size=len(splits.val))
    lo, hi = flat_region(result["curve"])
    result["within_95pct_of_best"] = {"low": lo, "high": hi}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    curve = result.pop("curve")
    pd.DataFrame(curve).to_csv(METRICS_DIR / "threshold_profit_curve.csv", index=False)
    save_json(result, METRICS_DIR / "threshold_bootstrap.json")

    t, r = result["threshold"], result["regret_per_1000"]
    logger.info(
        "Full-data best threshold %.2f (profit %.0f per 1,000)",
        result["full_data_best_threshold"],
        result["full_data_best_profit_per_1000"],
    )
    logger.info(
        "Resampled threshold: median %.2f, 90%% interval [%.2f, %.2f], sd %.3f",
        t["median"],
        t["p05"],
        t["p95"],
        t["std"],
    )
    logger.info(
        "Regret per 1,000: mean %.0f, p95 %.0f, within 500: %.0f%% | 95%%-of-best [%.2f, %.2f]",
        r["mean"],
        r["p95"],
        100 * r["share_within_500"],
        lo,
        hi,
    )
    return result


if __name__ == "__main__":
    run_threshold_bootstrap()
