"""How sensitive is the profit case to the placeholder business assumptions?

Offer cost, customer value and offer success rate are placeholders in ``src/config.py``, and
every profit figure, the decision threshold and the tier sizes depend on them. This module
sweeps a grid of plausible values and, for each combination, finds the profit-maximizing
threshold on out-of-fold calibrated scores (train + validation; the test set is untouched).

Run with ``python -m src.models.sensitivity`` (about a minute).
"""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from src.config import (
    METRICS_DIR,
    SENSITIVITY_COSTS,
    SENSITIVITY_SUCCESS_RATES,
    SENSITIVITY_VALUES,
    TARGET_COL,
)
from src.features.pipeline import split_xy
from src.models.oof import dev_frame, final_pipeline, oof_calibrated_scores
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

THRESHOLDS: np.ndarray = np.round(np.arange(0.01, 1.0, 0.01), 2)


def profit_per_1000(
    y: np.ndarray, scores: np.ndarray, threshold: float, cost: float, value: float, rate: float
) -> tuple[float, float]:
    """Expected profit per 1,000 customers and contact rate for one threshold.

    Args:
        y: True labels.
        scores: Calibrated churn probabilities.
        threshold: Contact customers scored at or above this.
        cost: Cost of one offer.
        value: Value of a saved customer.
        rate: Probability that a contacted churner is saved.

    Returns:
        ``(profit_per_1000, contact_rate)``.
    """
    contacted = scores >= threshold
    saved_value = (contacted & (y == 1)).sum() * value * rate
    profit = saved_value - contacted.sum() * cost
    return float(1000 * profit / len(y)), float(contacted.mean())


def profit_grid(
    y: np.ndarray,
    scores: np.ndarray,
    current_threshold: float,
    costs: tuple[float, ...] = SENSITIVITY_COSTS,
    values: tuple[float, ...] = SENSITIVITY_VALUES,
    rates: tuple[float, ...] = SENSITIVITY_SUCCESS_RATES,
) -> pd.DataFrame:
    """Profit under every combination of cost, value and success rate.

    Args:
        y: True labels.
        scores: Out-of-fold calibrated probabilities.
        current_threshold: The threshold used today, to show what keeping it would cost.
        costs: Offer costs to try.
        values: Saved-customer values to try.
        rates: Offer success rates to try.

    Returns:
        One row per combination with the break-even probability, the best threshold and its
        profit per 1,000 customers, the profit if today's threshold is kept, and the
        "contact everyone" alternative.
    """
    y = np.asarray(y)
    scores = np.asarray(scores)
    base = float(y.mean())
    rows = []
    for cost, value, rate in product(costs, values, rates):
        per_threshold = [profit_per_1000(y, scores, t, cost, value, rate) for t in THRESHOLDS]
        best = int(np.argmax([p for p, _ in per_threshold]))
        best_profit, best_contact = per_threshold[best]
        kept_profit, kept_contact = profit_per_1000(y, scores, current_threshold, cost, value, rate)
        rows.append(
            {
                "cost": cost,
                "value": value,
                "success_rate": rate,
                "break_even_probability": cost / (value * rate),
                "best_threshold": float(THRESHOLDS[best]),
                "best_profit_per_1000": best_profit,
                "best_contact_rate": best_contact,
                "kept_threshold_profit_per_1000": kept_profit,
                "kept_threshold_contact_rate": kept_contact,
                "contact_everyone_per_1000": 1000 * (base * value * rate - cost),
            }
        )
    return pd.DataFrame(rows)


def summarize_grid(grid: pd.DataFrame) -> dict[str, float]:
    """Headline numbers across the grid.

    Args:
        grid: Output of :func:`profit_grid`.

    Returns:
        Share of scenarios where targeting earns a profit, where contacting everyone does,
        and the average profit lost by keeping today's threshold instead of re-tuning it.
    """
    return {
        "scenarios": int(len(grid)),
        "targeting_profitable_share": float((grid["best_profit_per_1000"] > 0).mean()),
        "contact_everyone_profitable_share": float((grid["contact_everyone_per_1000"] > 0).mean()),
        "mean_loss_from_keeping_threshold_per_1000": float(
            (grid["best_profit_per_1000"] - grid["kept_threshold_profit_per_1000"]).mean()
        ),
    }


def run_sensitivity() -> pd.DataFrame:
    """Run the profit sensitivity analysis and save the grid.

    Returns:
        The grid (also written to ``reports/metrics/profit_sensitivity.csv``).
    """
    dev = dev_frame()
    X, y = split_xy(dev)
    scores = oof_calibrated_scores(final_pipeline(y), X, y)
    business = json.loads((METRICS_DIR / "business_summary.json").read_text())
    grid = profit_grid(dev[TARGET_COL].to_numpy(), scores, business["threshold_calibrated"])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(METRICS_DIR / "profit_sensitivity.csv", index=False)
    for key, val in summarize_grid(grid).items():
        logger.info("%s: %s", key, val)
    return grid


if __name__ == "__main__":
    run_sensitivity()
