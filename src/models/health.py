"""Health check for a finished pipeline run: did the outputs come out right?

The unit tests check each piece with toy data. This module checks the *real* run: it reads the
files ``python -m src.pipeline`` wrote and fails if the model, its metrics or its provenance
look wrong. A scheduled CI job runs the pipeline on the real dataset and then this check, so a
regression that only shows on real data (a package update, a platform difference, a change to
the data source) is caught within a week instead of by a surprised user.

Run with ``python -m src.models.health`` after the pipeline. Exit code 0 means healthy, 1 means
at least one problem (each is logged).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    HEALTH_MAX_ECE,
    HEALTH_MAX_HIGH_TIER_GAP,
    HEALTH_MIN_TEST_ROC_AUC,
    HEALTH_REFERENCE_TEST_PR_AUC,
    HEALTH_TEST_PR_AUC_TOLERANCE,
    METRICS_DIR,
    MODELS_DIR,
)
from src.models.business import RISK_LIST_FILE
from src.models.select import EXPLAINABLE_MODELS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load(path: Path, problems: list[str]) -> dict[str, Any] | None:
    """Read a JSON output, recording a problem (not raising) if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        problems.append(f"cannot read {path.name}: {err}. Did the pipeline finish?")
        return None


def check_outputs(
    metrics_dir: Path = METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    risk_list_file: Path = RISK_LIST_FILE,
) -> list[str]:
    """Check the outputs of one pipeline run.

    Args:
        metrics_dir: Directory with ``model_summary.json``, ``business_summary.json`` and
            ``split_summary.json``.
        models_dir: Directory with ``model_card.json``.
        risk_list_file: The ranked customer list the pipeline wrote.

    Returns:
        One plain-language message per problem found; empty when the run is healthy.
    """
    problems: list[str] = []
    summary = _load(metrics_dir / "model_summary.json", problems)
    business = _load(metrics_dir / "business_summary.json", problems)
    splits = _load(metrics_dir / "split_summary.json", problems)
    card = _load(models_dir / "model_card.json", problems)

    if summary is not None:
        name = summary.get("final_model")
        if name not in EXPLAINABLE_MODELS:
            problems.append(f"final model {name!r} is not one the explanations support")
        if summary.get("beats_baseline_on_test") is not True:
            problems.append("the final model does not beat the logistic regression baseline")
        test = summary.get("test", {}).get("final", {})
        pr_auc, roc_auc = test.get("pr_auc"), test.get("roc_auc")
        low = HEALTH_REFERENCE_TEST_PR_AUC - HEALTH_TEST_PR_AUC_TOLERANCE
        high = HEALTH_REFERENCE_TEST_PR_AUC + HEALTH_TEST_PR_AUC_TOLERANCE
        if pr_auc is None or not low <= pr_auc <= high:
            problems.append(
                f"test PR-AUC {pr_auc} is outside the expected {low:.3f} to {high:.3f} "
                f"(reference {HEALTH_REFERENCE_TEST_PR_AUC})"
            )
        if roc_auc is None or roc_auc < HEALTH_MIN_TEST_ROC_AUC:
            problems.append(f"test ROC-AUC {roc_auc} is below {HEALTH_MIN_TEST_ROC_AUC}")

    if business is not None:
        ece = business.get("calibration", {}).get("test_after", {}).get("ece")
        if ece is None or ece > HEALTH_MAX_ECE:
            problems.append(f"calibration error {ece} is above {HEALTH_MAX_ECE}")
        high_tier = business.get("tiers", {}).get("High", {})
        predicted, actual = high_tier.get("mean_probability"), high_tier.get("actual_churn_rate")
        if (
            predicted is None
            or actual is None
            or abs(predicted - actual) > HEALTH_MAX_HIGH_TIER_GAP
        ):
            problems.append(
                f"High tier predicts {predicted} but {actual} actually churned "
                f"(allowed gap {HEALTH_MAX_HIGH_TIER_GAP})"
            )
        if not 0.05 < business.get("threshold_calibrated", 0) < 0.95:
            problems.append("the contact threshold is outside 0.05 to 0.95")

    if card is not None:
        if card.get("data", {}).get("matches_expected_source") is not True:
            problems.append("the raw data does not match the expected source (checksum differs)")
        if summary is not None and card.get("model", {}).get("name") != summary.get("final_model"):
            problems.append("the model card names a different model than the run selected")

    if not risk_list_file.exists():
        problems.append(f"{risk_list_file.name} was not written")
    else:
        risk = pd.read_csv(risk_list_file, dtype={"customerID": str})
        prob = risk["churn_probability"]
        expected = None if splits is None else splits.get("test", {}).get("rows")
        if expected is not None and len(risk) != expected:
            problems.append(f"risk list has {len(risk)} customers, the test split has {expected}")
        if prob.isna().any() or not prob.between(0, 1).all():
            problems.append("risk list has missing or out-of-range probabilities")
        if not prob.is_monotonic_decreasing:
            problems.append("risk list is not sorted from highest to lowest risk")
        if not risk["customerID"].is_unique:
            problems.append("risk list has duplicate customer IDs")
    return problems


def run_health_check() -> int:
    """Run :func:`check_outputs` on the current run and log the result.

    Returns:
        Process exit code: 0 if healthy, 1 if there were problems.
    """
    problems = check_outputs()
    if problems:
        for problem in problems:
            logger.error("UNHEALTHY: %s", problem)
        logger.error("%d problem(s) found", len(problems))
        return 1
    logger.info("Healthy: model, metrics, calibration, provenance and risk list all check out")
    return 0


if __name__ == "__main__":
    sys.exit(run_health_check())
