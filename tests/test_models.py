"""Model-layer tests: metrics, models, SHAP, calibration, actions, business impact."""

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
from src.models.business import strategy_comparison, tier_summary
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
from src.models.predict import CalibratedModel, build_risk_list, calibrate, risk_tier
from src.models.recommend import (
    ACTION_RULES,
    DEFAULT_ACTION,
    MEDIUM_ACTION,
    NO_ACTION,
    matched_actions,
    recommend_actions,
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


# --- Phase 6: recommendations ---------------------------------------------------


def test_matched_actions_dedupes_and_skips_unknown_reasons() -> None:
    reasons = [
        "Contract = Month-to-month",
        "gender = Male",
        "OnlineSecurity = No",
        "TechSupport = No",
    ]
    actions = matched_actions(reasons)
    assert actions[0] == ACTION_RULES[("Contract", "Month-to-month")]
    assert len(actions) == 2  # security and tech support share one bundle action


def test_recommend_actions_by_tier() -> None:
    reasons = ["Contract = Month-to-month", "tenure = 2 months", "InternetService = Fiber optic"]
    primary, secondary = recommend_actions(reasons, "High")
    assert primary == ACTION_RULES[("Contract", "Month-to-month")]
    assert secondary == ACTION_RULES[("tenure", None)]
    assert recommend_actions(reasons, "Medium") == (f"{MEDIUM_ACTION} Contract", "")
    assert recommend_actions(reasons, "Low") == (NO_ACTION, "")
    assert recommend_actions([], "High") == (NO_ACTION, "")
    assert recommend_actions(["gender = Male"], "High") == (DEFAULT_ACTION, "")


# --- Phase 6: calibration, tiers, risk list -------------------------------------


def _calibrated_toy() -> tuple[CalibratedModel, pd.DataFrame, pd.Series]:
    from xgboost import XGBClassifier

    X, y = _toy_xy(600)
    pipe = make_model_pipeline(
        XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=3, random_state=config.SEED),
        engineered=False,
    ).fit(X.iloc[:400], y.iloc[:400])
    final = {"pipeline": pipe, "threshold": 0.6, "name": "toy"}
    model = calibrate(final, X.iloc[400:500], y.iloc[400:500], medium_cutoff=float(y.mean()))
    return model, X.iloc[500:], y.iloc[500:]


def test_calibrated_threshold_selects_same_customers_and_keeps_ranking() -> None:
    model, X, _ = _calibrated_toy()
    raw = model.raw.predict_proba(X)[:, 1]
    cal = model.predict_proba(X)
    np.testing.assert_array_equal(raw >= model.raw_threshold, cal >= model.threshold)
    order_raw = np.argsort(-raw, kind="stable")
    assert np.all(np.diff(cal[order_raw]) <= 1e-12)  # calibrated scores keep the raw order


def test_risk_tier_boundaries() -> None:
    tiers = risk_tier(np.array([0.9, 0.4, 0.3, 0.1]), threshold=0.4, medium_cutoff=0.25)
    assert tiers.tolist() == ["High", "High", "Medium", "Low"]


def test_build_risk_list_is_ranked_and_complete() -> None:
    model, X, y = _calibrated_toy()
    customers = X.assign(**{config.ID_COL: [f"c{i}" for i in range(len(X))]}, Churn=y.to_numpy())
    rl = build_risk_list(model, customers)
    assert len(rl) == len(customers) and rl[config.ID_COL].is_unique
    assert rl["rank"].tolist() == list(range(1, len(rl) + 1))
    assert rl["churn_probability"].is_monotonic_decreasing
    assert set(rl["risk_tier"]) <= {"High", "Medium", "Low"}
    assert (rl.loc[rl["risk_tier"] == "Low", "primary_action"] == NO_ACTION).all()
    assert (rl["contact"] == (rl["risk_tier"] == "High")).all()
    assert "Churn" not in rl.columns  # the actual outcome never leaks into the list


# --- Phase 6: business impact ----------------------------------------------------


def test_strategy_comparison_math() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    contact = np.array([True, True, False, False, False, False, False, False, False, False])
    t = strategy_comparison(y, contact).set_index("strategy")
    value_per_save = config.RETAINED_CUSTOMER_VALUE * config.OFFER_SUCCESS_RATE
    cost = config.RETENTION_OFFER_COST
    assert t.loc["Contact nobody", "profit"] == 0
    assert t.loc["Model targeting", "profit"] == pytest.approx(2 * value_per_save - 2 * cost)
    assert t.loc["Random (same number as model)", "churners_reached"] == pytest.approx(0.4)
    assert t.loc["Contact everyone", "profit"] == pytest.approx(2 * value_per_save - 10 * cost)


def test_tier_summary_shares_add_up() -> None:
    rl = pd.DataFrame(
        {
            "customerID": list("abcdef"),
            "churn_probability": [0.9, 0.8, 0.4, 0.3, 0.1, 0.05],
            "risk_tier": ["High", "High", "Medium", "Medium", "Low", "Low"],
        }
    )
    y_by_id = pd.Series([1, 1, 0, 1, 0, 0], index=list("abcdef"))
    t = tier_summary(rl, y_by_id)
    assert t["share"].sum() == pytest.approx(1.0)
    assert t["share_of_churners"].sum() == pytest.approx(1.0)
    assert t.loc["High", "actual_churn_rate"] == 1.0


def test_download_skips_when_file_exists(tmp_path, monkeypatch) -> None:
    import src.data.download as dl

    existing = tmp_path / "telco.csv"
    existing.write_text("already here")
    monkeypatch.setattr(dl, "RAW_DATA_FILE", existing)
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no network"))
    dl.download()
    assert existing.read_text() == "already here"


# --- Repeated CV --------------------------------------------------------------


def test_corrected_paired_test_is_wider_than_naive() -> None:
    from scipy import stats

    from src.models.repeated_cv import corrected_paired_test

    diffs = np.array([0.02, 0.01, 0.03, -0.01, 0.02, 0.00, 0.01, 0.02, 0.03, 0.01])
    out = corrected_paired_test(diffs, n_train=800, n_test=200)
    naive_se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert out["se"] > naive_se
    assert out["ci_low"] < out["mean_diff"] < out["ci_high"]
    assert out["win_rate"] == pytest.approx(0.8)
    assert out["p_value"] > stats.ttest_1samp(diffs, 0).pvalue


def test_corrected_paired_test_handles_identical_models() -> None:
    from src.models.repeated_cv import corrected_paired_test

    out = corrected_paired_test(np.zeros(10), n_train=800, n_test=200)
    assert out["mean_diff"] == 0.0
    assert out["p_value"] == 1.0


def test_repeated_cv_scores_shape_and_pairing() -> None:
    from src.models.repeated_cv import METRICS, repeated_cv_scores, summarize

    X, y = _toy_xy()
    pipes = {
        "baseline": make_baseline(),
        "final": make_model_pipeline(candidate_models(y)["logreg"], engineered=False),
    }
    scores = repeated_cv_scores(pipes, X, y, n_splits=3, n_repeats=2)
    assert len(scores) == 2 * 3 * 2  # models x folds x repeats
    assert set(METRICS) <= set(scores.columns)
    summary = summarize(scores, n_train=len(X) * 2 // 3, n_test=len(X) // 3)
    assert summary["n_folds_total"] == 6
    assert "final_minus_baseline" in summary["paired"]


def test_make_tuned_pipeline_applies_params() -> None:
    from src.models.repeated_cv import make_tuned_pipeline

    _, y = _toy_xy()
    pipe = make_tuned_pipeline("logreg", {"model__C": 0.5, "prep": "raw"}, y)
    assert pipe.named_steps["model"].C == 0.5
    assert "features" not in pipe.named_steps["prep"].named_steps


# --- Feature ablation ---------------------------------------------------------


def test_zero_columns_zeroes_only_requested_and_copies() -> None:
    from src.models.ablation import zero_columns

    X, _ = _toy_xy()
    out = zero_columns(X, ["tenure"])
    assert (out["tenure"] == 0).all()
    assert X["tenure"].gt(0).any()  # original untouched
    pd.testing.assert_series_equal(out["MonthlyCharges"], X["MonthlyCharges"])
    with pytest.raises(KeyError):
        zero_columns(X, ["not_a_column"])


def test_feature_ablation_removing_signal_hurts() -> None:
    from src.models.ablation import feature_ablation

    X, y = _toy_xy()
    pipes = {"baseline": make_baseline()}
    table = feature_ablation(
        pipes, X, y, scenarios={"none": [], "tenure": ["tenure"]}, n_splits=3, n_repeats=1
    )
    assert set(table["removed"]) == {"none", "tenure"}
    assert len(table) == 2
    pr = table.set_index("removed")["pr_auc"]
    assert pr["tenure"] < pr["none"]


# --- Model card ---------------------------------------------------------------


def _fake_metrics(tmp_path, with_repeated: bool = True):  # type: ignore[no-untyped-def]
    import json

    metrics = tmp_path / "metrics"
    metrics.mkdir()
    m = {"pr_auc": 0.67, "roc_auc": 0.85, "recall_top10": 0.29, "lift_top10": 2.9}
    files = {
        "model_summary.json": {
            "final_model": "xgboost",
            "final_params": {"model__max_depth": 5, "prep": "raw"},
            "threshold": 0.58,
            "val": {"final": m, "baseline": m},
            "test": {"final": m, "baseline": {**m, "pr_auc": 0.64}},
            "test_pr_auc_diff_bootstrap": {"diff": 0.03},
        },
        "split_summary.json": {"train": {"rows": 70, "share": 0.7, "churn_rate": 0.25}},
        "business_summary.json": {
            "calibration": {"method": "sigmoid"},
            "threshold_calibrated": 0.377,
        },
        "tuning_results.json": {"xgboost": {"cv_pr_auc_mean": 0.67}},
    }
    if with_repeated:
        gain = {"mean_diff": 0.01, "ci_low": -0.01, "ci_high": 0.03, "p_value": 0.3}
        files["repeated_cv_summary.json"] = {
            "per_model": {
                "baseline": {"pr_auc": {"mean": 0.656}},
                "final": {"pr_auc": {"mean": 0.666}},
            },
            "paired": {"final_minus_baseline": {"pr_auc": gain}},
        }
    for name, content in files.items():
        (metrics / name).write_text(json.dumps(content))
    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n")
    return metrics, raw


def test_build_model_card_records_provenance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    from src.models.model_card import build_model_card, render_markdown

    metrics, raw = _fake_metrics(tmp_path)
    card = build_model_card(metrics, raw, model_file=tmp_path / "missing.joblib")
    assert card["data"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert card["data"]["matches_expected_source"] is False  # not the real dataset
    assert card["model"]["name"] == "xgboost"
    assert card["model_trained_at"] is None
    assert card["performance"]["repeated_cv"]["final_minus_baseline_pr_auc"]["mean_diff"] == 0.01
    assert card["limitations"]
    text = render_markdown(card)
    assert "# Model card" in text and "DOES NOT match" in text and "0.377" in text


def test_model_card_without_repeated_cv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.models.model_card import build_model_card, render_markdown

    metrics, raw = _fake_metrics(tmp_path, with_repeated=False)
    card = build_model_card(metrics, raw, model_file=tmp_path / "missing.joblib")
    assert card["performance"]["repeated_cv"] is None
    assert "Repeated CV" not in render_markdown(card)


def test_model_card_missing_inputs_raise_clear_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.models.model_card import build_model_card

    with pytest.raises(FileNotFoundError, match="src.pipeline"):
        build_model_card(tmp_path, tmp_path / "raw.csv")
    metrics, _ = _fake_metrics(tmp_path)
    with pytest.raises(FileNotFoundError, match="src.data.download"):
        build_model_card(metrics, tmp_path / "nope.csv")


def test_model_card_model_file_time_is_recorded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.models.model_card import build_model_card

    metrics, raw = _fake_metrics(tmp_path)
    model = tmp_path / "final_model.joblib"
    model.write_bytes(b"x")
    assert build_model_card(metrics, raw, model_file=model)["model_trained_at"] is not None


# --- Explainable Boosting Machine ---------------------------------------------


def test_ebm_pipeline_fits_and_importances_map_to_source_features() -> None:
    from src.models.ebm import importance_by_feature, make_ebm_pipeline

    X, y = _toy_xy()
    pipe = make_ebm_pipeline().fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert compute_metrics(y, proba)["roc_auc"] > 0.8
    imp = importance_by_feature(pipe)
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp["importance"].is_monotonic_decreasing
    assert not imp["feature"].str.contains("__").any()  # one-hot columns mapped back
    assert "Contract" in set(imp["feature"])


def test_ebm_is_purely_additive() -> None:
    from src.models.ebm import make_ebm_pipeline

    X, y = _toy_xy()
    model = make_ebm_pipeline().fit(X, y).named_steps["model"]
    assert not any(" & " in name for name in model.term_names_)  # no interaction terms


def test_compare_scores_pairs_ebm_against_each_other_model() -> None:
    from src.models.ebm import compare_scores, make_ebm_pipeline
    from src.models.repeated_cv import repeated_cv_scores

    X, y = _toy_xy()
    pipes = {"baseline": make_baseline(), "ebm": make_ebm_pipeline()}
    scores = repeated_cv_scores(pipes, X, y, n_splits=3, n_repeats=2)
    out = compare_scores(scores, n_train=len(X) * 2 // 3, n_test=len(X) // 3)
    assert set(out["per_model"]) == {"baseline", "ebm"}
    assert set(out["paired"]) == {"ebm_minus_baseline"}
    assert set(out["paired"]["ebm_minus_baseline"]) >= {"pr_auc", "roc_auc"}


# --- Profit sensitivity -------------------------------------------------------


def test_profit_per_1000_matches_hand_calculation() -> None:
    from src.models.sensitivity import profit_per_1000

    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    profit, contact = profit_per_1000(y, s, 0.5, cost=10.0, value=100.0, rate=0.5)
    # contacts: 3 customers, 2 true churners -> 2 * 100 * 0.5 - 3 * 10 = 70 over 10 customers
    assert profit == pytest.approx(1000 * 70 / 10)
    assert contact == pytest.approx(0.3)


def test_profit_grid_is_monotone_in_success_rate_and_flags_break_even() -> None:
    from src.models.sensitivity import profit_grid, summarize_grid

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    s = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.15, 400), 0.01, 0.99)
    grid = profit_grid(y, s, 0.35, costs=(50.0,), values=(500.0,), rates=(0.1, 0.3, 0.5))
    assert len(grid) == 3
    assert grid["best_profit_per_1000"].is_monotonic_increasing
    assert (grid["best_profit_per_1000"] >= grid["kept_threshold_profit_per_1000"] - 1e-9).all()
    assert grid["break_even_probability"].iloc[0] == pytest.approx(50 / (500 * 0.1))
    out = summarize_grid(grid)
    assert out["scenarios"] == 3 and 0 <= out["targeting_profitable_share"] <= 1
