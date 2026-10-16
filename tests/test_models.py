"""Model tests. Phase 0: reproducibility + logging. Phase 4 adds training/metric tests."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain.shap_analysis import compute_shap, group_shap, original_feature, top_reasons
from src.features.build_features import ADDON_COLS
from src.features.pipeline import make_model_pipeline
from src.models.analysis import example_customers
from src.models.baseline import make_baseline
from src.models.evaluate import (
    best_profit_threshold,
    bootstrap_pr_auc_diff,
    calibration_summary,
    compute_metrics,
    expected_profit,
    gain_table,
    lift_at_top,
    performance_by_segment,
    recall_at_top,
)
from src.models.train import candidate_models, cross_validate_models
from src.models.tune import SEARCH_SPACES
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed()
    first = np.random.rand(5)
    set_seed()
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)


def test_get_logger_has_single_handler() -> None:
    logger = get_logger("test_logger")
    get_logger("test_logger")  # calling twice must not duplicate handlers
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO


# --- evaluate.py --------------------------------------------------------------

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
PERFECT = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95])


def test_recall_and_lift_at_top_on_known_values() -> None:
    # top 10% of 10 rows = 1 row (score 0.95, a churner): captures 1 of 3 churners
    assert recall_at_top(Y, PERFECT, 0.10) == pytest.approx(1 / 3)
    # churn rate in top row = 1.0, base rate = 0.3 -> lift 3.33
    assert lift_at_top(Y, PERFECT, 0.10) == pytest.approx(1 / 0.3)
    assert recall_at_top(Y, PERFECT, 0.30) == pytest.approx(1.0)


def test_expected_profit_formula() -> None:
    # threshold 0.5 contacts 3 rows, all churners: 3 * 500 * 0.3 - 3 * 50 = 300
    assert expected_profit(Y, PERFECT, 0.5, cost=50, value=500, success_rate=0.3) == 300
    # contacting everyone: 3 * 150 - 10 * 50 = -50
    assert expected_profit(Y, PERFECT, 0.0, cost=50, value=500, success_rate=0.3) == -50


def test_best_profit_threshold_avoids_contacting_non_churners() -> None:
    threshold, profit = best_profit_threshold(Y, PERFECT)
    assert 0.2 < threshold <= 0.8
    assert profit == expected_profit(Y, PERFECT, threshold)


def test_compute_metrics_perfect_ranking() -> None:
    m = compute_metrics(Y, PERFECT, threshold=0.5)
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["contact_rate"] == 0.3


def test_bootstrap_diff_detects_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    noise = rng.normal(0, 1, 400)
    res = bootstrap_pr_auc_diff(y, good, noise, n_boot=300)
    assert res["diff"] > 0 and res["ci_low"] > 0 and res["p_a_better"] > 0.95


# --- baseline / train / tune ----------------------------------------------------


def _toy_xy(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config.SEED)
    tenure = rng.integers(0, 72, n).astype(float)
    monthly = rng.uniform(20, 110, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    churn = ((contract == "Month-to-month") & (tenure < 24)).astype(int)
    churn = np.where(rng.random(n) < 0.1, 1 - churn, churn)
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    X = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": tenure,
            "PhoneService": yes_no(),
            "MultipleLines": yes_no(),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            **{c: yes_no() for c in ADDON_COLS},
            "Contract": contract,
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
        }
    )
    return X, pd.Series(churn, name=config.TARGET_COL)


def test_baseline_learns_signal() -> None:
    X, y = _toy_xy()
    proba = make_baseline().fit(X, y).predict_proba(X)[:, 1]
    assert compute_metrics(y, proba)["roc_auc"] > 0.8


def test_candidate_models_and_search_spaces_align() -> None:
    _, y = _toy_xy()
    models = candidate_models(y)
    assert set(models) == {"logreg", "random_forest", "xgboost", "lightgbm"}
    assert set(SEARCH_SPACES) <= set(models)
    neg_pos = (y == 0).sum() / (y == 1).sum()
    assert models["xgboost"].get_params()["scale_pos_weight"] == pytest.approx(neg_pos)


def test_cross_validate_models_returns_sorted_table() -> None:
    X, y = _toy_xy()
    small = {"logreg": candidate_models(y)["logreg"]}
    table = cross_validate_models(X, y, models=small)
    assert len(table) == 2  # raw + engineered
    assert table["pr_auc_mean"].is_monotonic_decreasing


# --- Phase 5: evaluation helpers ---------------------------------------------


def test_gain_table_deciles_and_capture() -> None:
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)  # churners get the 10 highest scores
    g = gain_table(y, scores)
    assert len(g) == 10 and g["customers"].sum() == 100
    assert g.loc[1, "churn_rate"] == 1.0 and g.loc[1, "lift"] == pytest.approx(10.0)
    assert g["cum_churners_pct"].iloc[-1] == pytest.approx(1.0)
    assert g["cum_churners_pct"].is_monotonic_increasing


def test_calibration_summary_flags_inflated_scores() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < p).astype(int)
    honest = calibration_summary(y, p)
    inflated = calibration_summary(y, np.clip(p + 0.2, 0, 1))
    assert honest["ece"] < 0.05 < inflated["ece"]
    assert inflated["mean_predicted"] > inflated["observed_rate"]


def test_performance_by_segment_handles_single_class_segments() -> None:
    df = pd.DataFrame(
        {
            "seg": ["a"] * 40 + ["b"] * 40,
            config.TARGET_COL: [0, 1] * 20 + [0] * 40,  # segment b has no churners
        }
    )
    scores = np.r_[np.tile([0.2, 0.8], 20), np.full(40, 0.1)]
    out = performance_by_segment(df, scores, ["seg"]).set_index("value")
    assert out.loc["a", "roc_auc"] == pytest.approx(1.0)
    assert np.isnan(out.loc["b", "roc_auc"])


def test_example_customers_picks_expected_rows() -> None:
    y = np.array([0, 1, 1, 0])
    s = np.array([0.05, 0.9, 0.4, 0.6])
    ex = example_customers(s, y, threshold=0.58)
    assert ex == {"high_risk": 1, "borderline": 3, "low_risk": 0}


# --- Phase 5: SHAP -------------------------------------------------------------


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        ("num__tenure", "tenure"),
        ("bin__SeniorCitizen", "SeniorCitizen"),
        ("cat__Contract_Month-to-month", "Contract"),
        ("cat__PaymentMethod_Bank transfer (automatic)", "PaymentMethod"),
        ("cat__tenure_band_0-6", "tenure_band"),
    ],
)
def test_original_feature_mapping(col: str, expected: str) -> None:
    assert original_feature(col) == expected


def test_grouped_shap_is_exact_and_reasons_are_readable() -> None:
    from scipy.special import expit
    from xgboost import XGBClassifier

    X, y = _toy_xy()
    pipe = make_model_pipeline(
        XGBClassifier(n_estimators=50, max_depth=3, random_state=config.SEED), engineered=False
    ).fit(X, y)
    sv = compute_shap(pipe, X)
    grouped = group_shap(sv)

    assert "Contract" in grouped.columns and not any("__" in c for c in grouped.columns)
    reconstructed = expit(grouped.sum(axis=1).to_numpy() + np.ravel(sv.base_values))
    np.testing.assert_allclose(reconstructed, pipe.predict_proba(X)[:, 1], atol=1e-5)

    reasons = top_reasons(grouped, X, k=3)
    assert len(reasons) == len(X) and all(len(r) <= 3 for r in reasons)
    riskiest = int(np.argmax(pipe.predict_proba(X)[:, 1]))
    assert any(r.startswith("Contract = ") or r.startswith("tenure = ") for r in reasons[riskiest])
