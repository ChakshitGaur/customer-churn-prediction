"""Edge-case tests: malformed input, degenerate metrics, empty batches and boundary values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import helpers
from src import config
from src.data.clean import clean
from src.data.split import split_data
from src.models.evaluate import (
    best_profit_threshold,
    compute_metrics,
    expected_profit,
    lift_at_top,
    recall_at_top,
    top_fraction_mask,
)
from src.models.monitor import drift_report, psi
from src.models.predict import build_risk_list, risk_tier
from tests.synthetic import make_synthetic_telco
from tests.test_app import _raw_batch
from tests.test_models import _calibrated_toy

# --- cleaning ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["yes", "Maybe", "", None])
def test_clean_rejects_unrecognised_or_missing_churn_labels(bad: str | None) -> None:
    raw = make_synthetic_telco(50)
    raw["Churn"] = raw["Churn"].astype(object)
    raw.loc[0, "Churn"] = bad
    with pytest.raises(ValueError, match="Churn must be"):
        clean(raw)


def test_clean_accepts_numeric_target_and_rejects_non_binary_numbers() -> None:
    raw = make_synthetic_telco(50).assign(Churn=lambda d: (d["Churn"] == "Yes").astype(int))
    assert set(clean(raw)["Churn"]) <= {0, 1}
    raw.loc[0, "Churn"] = 2
    with pytest.raises(ValueError, match="Churn must be"):
        clean(raw)


def test_clean_rejects_blank_total_charges_with_missing_tenure() -> None:
    raw = make_synthetic_telco(50)
    idx = raw.index[raw["tenure"] == 0][0] if (raw["tenure"] == 0).any() else 0
    raw["tenure"] = raw["tenure"].astype(float)
    raw.loc[idx, ["tenure", "TotalCharges"]] = [np.nan, " "]
    with pytest.raises(ValueError, match="missing tenure"):
        clean(raw)


def test_clean_handles_non_breaking_space_and_padding() -> None:
    raw = make_synthetic_telco(50)
    zero = raw["tenure"] == 0
    raw.loc[zero, "TotalCharges"] = " "
    raw["Contract"] = " " + raw["Contract"] + "  "
    out = clean(raw)
    assert (out.loc[zero.to_numpy(), "TotalCharges"] == 0.0).all()
    assert out["Contract"].isin(["Month-to-month", "One year", "Two year"]).all()


def test_clean_empty_frame_and_exact_duplicates() -> None:
    raw = make_synthetic_telco(30)
    assert len(clean(raw.iloc[0:0])) == 0
    assert len(clean(pd.concat([raw, raw.iloc[[0]]]))) == 30  # exact duplicate is dropped


def test_clean_rejects_conflicting_duplicate_ids_and_text_in_numeric_column() -> None:
    raw = make_synthetic_telco(30)
    with pytest.raises(ValueError, match="not unique"):
        clean(
            raw.assign(customerID=raw["customerID"].mask(raw.index == 1, raw.loc[0, "customerID"]))
        )
    bad = raw.copy()
    bad["TotalCharges"] = bad["TotalCharges"].mask(bad.index == 5, "abc")
    with pytest.raises(ValueError):
        clean(bad)


def test_clean_without_target_column_keeps_rows() -> None:
    assert "Churn" not in clean(make_synthetic_telco(20).drop(columns="Churn")).columns


# --- splitting --------------------------------------------------------------------


def test_split_is_a_partition_and_keeps_rates_on_various_sizes() -> None:
    for n in (200, 501, 1000):
        df = clean(make_synthetic_telco(n, seed=n))
        s = split_data(df)
        assert sum(len(x) for _, x in s.items()) == n
        ids = [set(x[config.ID_COL]) for _, x in s.items()]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_fails_loudly_on_tiny_or_misconfigured_input() -> None:
    df = clean(make_synthetic_telco(300))
    with pytest.raises(ValueError):
        split_data(df.head(20))  # too small to keep churn rates comparable
    with pytest.raises(KeyError):
        split_data(df, date_col="no_such_column")


# --- metrics ----------------------------------------------------------------------


def test_top_fraction_mask_boundaries() -> None:
    s = np.linspace(0, 1, 10)
    assert top_fraction_mask(s, 0.0).sum() == 1  # always at least one row
    assert top_fraction_mask(s, 1.0).sum() == 10
    assert top_fraction_mask(s, 5.0).sum() == 10  # over-large fractions are capped
    assert top_fraction_mask(np.array([0.3]), 0.1).sum() == 1
    assert top_fraction_mask(np.array([]), 0.1).sum() == 0


def test_top_fraction_mask_is_deterministic_under_ties() -> None:
    s = np.full(20, 0.5)
    first = top_fraction_mask(s, 0.25)
    assert first.sum() == 5 and (first == top_fraction_mask(s, 0.25)).all()
    assert first[:5].all()  # stable sort: earliest rows win ties


def test_recall_and_lift_degenerate_labels() -> None:
    s = np.linspace(0, 1, 10)
    assert recall_at_top(np.array([]), np.array([])) == 0.0
    assert recall_at_top(np.zeros(10), s) == 0.0
    assert lift_at_top(np.zeros(10), s) == 0.0
    assert lift_at_top(np.ones(10), s) == pytest.approx(1.0)  # everyone churns: no lift


def test_perfect_and_inverted_rankings() -> None:
    y = np.array([0] * 90 + [1] * 10)
    perfect = compute_metrics(y, y.astype(float))
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["recall_top10"] == pytest.approx(1.0)
    assert perfect["lift_top10"] == pytest.approx(10.0)
    worst = compute_metrics(y, 1.0 - y)
    assert worst["roc_auc"] == pytest.approx(0.0) and worst["recall_top10"] == 0.0


def test_constant_scores_give_chance_level_roc() -> None:
    y = np.array([0, 1] * 20)
    m = compute_metrics(y, np.full(40, 0.5))
    assert m["roc_auc"] == pytest.approx(0.5) and m["pr_auc"] == pytest.approx(y.mean())


def test_single_class_labels_do_not_crash_the_metrics() -> None:
    m = compute_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert np.isnan(m["roc_auc"]) and m["recall_top10"] == 0.0


def test_metrics_reject_nan_and_out_of_range_scores() -> None:
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        compute_metrics(y, np.array([0.1, np.nan, 0.2, 0.9]))
    with pytest.raises(ValueError):
        compute_metrics(y, np.array([0.1, 3.0, 0.2, 0.9]))


def test_threshold_extremes() -> None:
    y = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    s = np.linspace(0.05, 0.95, 10)
    low = compute_metrics(y, s, threshold=0.0)
    assert low["contact_rate"] == 1.0 and low["recall"] == 1.0
    high = compute_metrics(y, s, threshold=1.5)
    assert high["contact_rate"] == 0.0 and high["profit"] == 0.0 and high["precision"] == 0.0
    assert expected_profit(y, s, 2.0) == 0.0  # nobody contacted -> no cost, no gain


def test_threshold_is_inclusive() -> None:
    y = np.array([1, 0])
    s = np.array([0.5, 0.2])
    assert compute_metrics(y, s, threshold=0.5)["contact_rate"] == 0.5


def test_best_profit_threshold_edge_inputs() -> None:
    t, p = best_profit_threshold(np.array([]), np.array([]))
    assert 0 < t < 1 and p == 0.0
    with pytest.raises(ValueError, match="at least one threshold"):
        best_profit_threshold(np.array([1, 0]), np.array([0.9, 0.1]), np.array([]))
    # churners only above 0.5, so contacting only them beats contacting everyone
    y = np.array([1] * 20 + [0] * 80)
    s = np.r_[np.full(20, 0.9), np.full(80, 0.1)]
    t, _ = best_profit_threshold(y, s)
    assert 0.1 < t <= 0.9


# --- drift ------------------------------------------------------------------------


def test_psi_degenerate_inputs_are_finite() -> None:
    num = pd.Series(np.arange(100.0))
    for ref, cur in [
        (num, pd.Series([], dtype=float)),
        (pd.Series([], dtype=float), num),
        (pd.Series([np.nan] * 5), pd.Series([np.nan] * 5)),
        (pd.Series([5.0] * 50), pd.Series([5.0] * 50)),
        (num, pd.Series([1e9] * 20)),  # entirely outside the reference range
        (pd.Series([True, False] * 20), pd.Series([True] * 40)),
    ]:
        assert np.isfinite(psi(ref, cur))
    assert psi(pd.Series([5.0] * 50), pd.Series([5.0] * 50)) == pytest.approx(0.0)
    assert psi(num, pd.Series([np.nan] * 5 + [3.0] * 5)) == psi(num, pd.Series([3.0] * 5))


def test_psi_is_independent_of_row_order() -> None:
    rng = np.random.default_rng(config.SEED)
    ref, cur = pd.Series(rng.normal(size=500)), pd.Series(rng.normal(0.3, 1, 500))
    assert psi(ref, cur) == pytest.approx(psi(ref.sample(frac=1, random_state=1), cur[::-1]))


def test_drift_report_with_no_matching_features_is_empty_not_an_error() -> None:
    report = drift_report(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), features=["x"])
    assert report.empty and list(report.columns) == ["feature", "psi", "status"]


# --- single-customer form ---------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_make_customer_rejects_non_finite_numbers(bad: float) -> None:
    for col in ("tenure", "MonthlyCharges"):
        with pytest.raises(ValueError, match="finite"):
            helpers.make_customer({**helpers.DEFAULT_CUSTOMER, col: bad})


def test_make_customer_rejects_text_numbers_and_accepts_zero_tenure() -> None:
    with pytest.raises(ValueError):
        helpers.make_customer({**helpers.DEFAULT_CUSTOMER, "tenure": "abc"})
    row = helpers.make_customer({**helpers.DEFAULT_CUSTOMER, "tenure": 0})
    assert row.loc[0, "TotalCharges"] == 0.0
    assert len(clean(row)) == 1


def test_what_if_rejects_unknown_attributes_instead_of_ignoring_them() -> None:
    model, _, _ = _calibrated_toy()
    customer = clean(helpers.make_customer(helpers.DEFAULT_CUSTOMER))
    with pytest.raises(ValueError, match="Unknown customer attribute"):
        helpers.what_if(model, customer, {"Contrat": "Two year"})  # typo must not be silent


# --- batch upload -----------------------------------------------------------------


@pytest.mark.parametrize("ids", [[None, "a", "b"], ["", "a", "b"], ["  ", "a", "b"]])
def test_prepare_batch_rejects_blank_customer_ids(ids: list[str | None]) -> None:
    with pytest.raises(ValueError, match="blank customerID"):
        helpers.prepare_batch(_raw_batch().assign(customerID=ids))


def test_prepare_batch_detects_duplicates_after_text_conversion() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        helpers.prepare_batch(_raw_batch().assign(customerID=[1, "1", "b"]))


def test_prepare_batch_normalises_senior_citizen_spellings() -> None:
    raw = _raw_batch(4).assign(SeniorCitizen=["Yes", "no", "1", "0.0"])
    assert helpers.prepare_batch(raw)["SeniorCitizen"].tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize(
    ("col", "value", "message"),
    [
        ("tenure", "nan", "non-numeric"),
        ("tenure", "inf", "above"),
        ("tenure", 1e9, "above"),
        ("tenure", -1, "negative"),
        ("MonthlyCharges", "$50", "non-numeric"),
        ("MonthlyCharges", "50,5", "non-numeric"),
        ("MonthlyCharges", "", "blank"),
        ("Contract", "month-to-month", "Did you mean"),
        ("Contract", "", "blank"),
    ],
)
def test_prepare_batch_reports_each_bad_value(col: str, value: object, message: str) -> None:
    raw = _raw_batch(3)
    raw[col] = raw[col].astype(object)
    raw.loc[0, col] = value
    with pytest.raises(ValueError, match=message):
        helpers.prepare_batch(raw)


def test_prepare_batch_reports_all_problems_together() -> None:
    raw = _raw_batch(3).assign(tenure=[-1, 1, 2], Contract=["Nope", "One year", "One year"])
    with pytest.raises(ValueError, match=r"2 problem\(s\)"):
        helpers.prepare_batch(raw)


def test_prepare_batch_ignores_extra_columns_and_tolerates_padding() -> None:
    raw = _raw_batch(3).assign(extra="x", Contract=" Two year ")
    out = helpers.prepare_batch(raw)
    assert "extra" not in out.columns and (out["Contract"] == "Two year").all()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "empty"),
        (b"  \n \n", "empty"),
        (b"a,b\n\xe9,1\n", "UTF-8"),
        (bytes(range(256)) * 4, "UTF-8"),
        (b"customerID;tenure\n1;2\n", "separated"),
        (b"customerID\ttenure\n1\t2\n", "separated"),
    ],
)
def test_read_csv_upload_rejects_unusable_files(data: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        helpers.read_csv_upload(data)


def test_read_csv_upload_accepts_bom_and_header_only() -> None:
    assert (
        helpers.read_csv_upload(b"\xef\xbb\xbfcustomerID,tenure\n0001,2\n").loc[0, "customerID"]
        == "0001"
    )
    assert helpers.read_csv_upload(b"customerID,tenure\n").empty


# --- CSV export safety --------------------------------------------------------------


def test_escape_formulas_covers_all_prefixes_and_spares_numbers() -> None:
    df = pd.DataFrame(
        {"id": ["=1+1", "+x", "-y", "@z", "\tq", "\rr", "ok", None], "n": range(-4, 4)}
    )
    out = helpers.escape_formulas(df)
    assert out["id"].tolist()[:6] == ["'=1+1", "'+x", "'-y", "'@z", "'\tq", "'\rr"]
    assert out["id"].tolist()[6] == "ok" and out["id"].isna().tolist()[7]
    assert out["n"].tolist() == list(range(-4, 4))  # negative numbers stay numbers
    assert df["id"][0] == "=1+1"  # the input is not modified
    assert b"'=1+1" in helpers.to_csv_bytes(df)


def test_to_csv_bytes_round_trips_unicode() -> None:
    df = pd.DataFrame({"id": ["café", "名前"], "n": [1, 2]})
    back = pd.read_csv(pd.io.common.BytesIO(helpers.to_csv_bytes(df)))
    assert back["id"].tolist() == ["café", "名前"]


# --- scoring ----------------------------------------------------------------------


def test_risk_tier_handles_boundaries_and_nan() -> None:
    p = np.array([0.4, 0.25, 0.2499, np.nan])
    assert risk_tier(p, 0.4, 0.25).tolist() == ["High", "Medium", "Low", "Low"]


def test_build_risk_list_rejects_an_empty_batch() -> None:
    model, X, _ = _calibrated_toy()
    c = X.assign(**{config.ID_COL: [f"c{i}" for i in range(len(X))]})
    with pytest.raises(ValueError, match="No customers"):
        build_risk_list(model, c.iloc[0:0])


def test_build_risk_list_single_row_unseen_category_and_missing_numeric() -> None:
    model, X, _ = _calibrated_toy()
    c = X.assign(**{config.ID_COL: [f"c{i}" for i in range(len(X))]})
    one = build_risk_list(model, c.iloc[:1])
    assert len(one) == 1 and one.loc[0, "rank"] == 1
    odd = build_risk_list(model, c.iloc[:3].assign(Contract="Weekly", tenure=np.nan))
    assert odd["churn_probability"].between(0, 1).all() and len(odd) == 3


def test_build_risk_list_is_order_independent_and_handles_duplicate_ids() -> None:
    model, X, _ = _calibrated_toy()
    c = X.assign(**{config.ID_COL: [f"c{i}" for i in range(len(X))]}).iloc[:20]
    a = build_risk_list(model, c).set_index(config.ID_COL)["churn_probability"]
    b = build_risk_list(model, c.sample(frac=1, random_state=1)).set_index(config.ID_COL)[
        "churn_probability"
    ]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
    assert len(build_risk_list(model, c.assign(**{config.ID_COL: "same"}))) == 20


def test_build_risk_list_requires_every_model_column() -> None:
    model, X, _ = _calibrated_toy()
    c = X.assign(**{config.ID_COL: [f"c{i}" for i in range(len(X))]})
    with pytest.raises(ValueError, match="missing"):
        build_risk_list(model, c.drop(columns=["Contract"]).iloc[:3])


# --- per-group thresholds ---------------------------------------------------------


def _scored_groups(n: int = 6000, seed: int = 0, shift_b: float = 0.0):
    """Two groups with calibrated scores; group B can have its true risk shifted off its score."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(rng.choice(["A", "B"], n))
    s = rng.uniform(0.02, 0.98, n)
    true_p = np.clip(s + np.where(groups == "B", shift_b, 0.0), 0.0, 1.0)
    return groups, (rng.random(n) < true_p).astype(int), s


def test_best_threshold_matches_break_even_and_handles_empty_input() -> None:
    from src.models.segment_thresholds import best_threshold

    _, y, s = _scored_groups(20000)
    econ = {"cost": 50.0, "value": 500.0, "rate": 0.3}
    assert best_threshold(y, s, **econ) == pytest.approx(50 / (500 * 0.3), abs=0.06)
    assert 0 < best_threshold(np.array([]), np.array([]), **econ) < 1


def test_group_thresholds_use_fallback_for_tiny_groups() -> None:
    from src.models.segment_thresholds import MIN_GROUP_SIZE, group_thresholds

    groups, y, s = _scored_groups(400)
    groups.iloc[: MIN_GROUP_SIZE - 1] = "tiny"
    t = group_thresholds(groups, y, s, 0.77, cost=50.0, value=500.0, rate=0.3)
    assert (t[: MIN_GROUP_SIZE - 1] == 0.77).all()


def test_per_group_thresholds_do_not_pay_when_scores_are_calibrated() -> None:
    from src.models.segment_thresholds import compare_thresholds

    groups, y, s = _scored_groups(4000)
    r = compare_thresholds(groups, y, s, n_repeats=60)
    assert r["in_sample_gain"] >= 0  # fitting on the same rows can only look better
    assert r["oos_gain_mean"] <= 50  # ...but on unseen rows there is nothing real to gain
    assert r["theory_threshold"] == pytest.approx(1 / 3)


def test_per_group_thresholds_pay_when_one_group_is_miscalibrated() -> None:
    from src.models.segment_thresholds import compare_thresholds

    groups, y, s = _scored_groups(8000, shift_b=0.35)  # B churns far more than its score says
    r = compare_thresholds(groups, y, s, n_repeats=60)
    assert r["oos_gain_mean"] > 500 and r["oos_share_better"] > 0.95


def test_compare_thresholds_is_reproducible() -> None:
    from src.models.segment_thresholds import compare_thresholds

    groups, y, s = _scored_groups(1500)
    assert compare_thresholds(groups, y, s, n_repeats=10) == compare_thresholds(
        groups, y, s, n_repeats=10
    )
