"""Tests for the dashboard helpers (app/helpers.py) and drift monitoring (src/models/monitor.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app import helpers
from src import config
from src.data.clean import clean
from src.models.monitor import PSI_MAJOR, PSI_MODERATE, drift_report, psi, psi_status
from tests.test_models import _calibrated_toy

# --- drift (PSI) ---------------------------------------------------------------


def test_psi_is_zero_for_identical_and_large_for_shifted_numeric() -> None:
    rng = np.random.default_rng(config.SEED)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < PSI_MODERATE
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > PSI_MAJOR


def test_psi_categorical_handles_new_and_missing_categories() -> None:
    ref = pd.Series(["Month-to-month"] * 50 + ["One year"] * 30 + ["Two year"] * 20)
    same = ref.sample(frac=1, random_state=config.SEED)
    assert psi(ref, same) == pytest.approx(0.0, abs=1e-9)
    shifted = pd.Series(["Month-to-month"] * 90 + ["Weekly"] * 10)  # new + missing levels
    assert np.isfinite(psi(ref, shifted)) and psi(ref, shifted) > PSI_MAJOR


@pytest.mark.parametrize(
    ("value", "status"), [(0.0, "stable"), (0.1, "moderate"), (0.24, "moderate"), (0.25, "major")]
)
def test_psi_status_thresholds(value: float, status: str) -> None:
    assert psi_status(value) == status


def test_drift_report_skips_missing_columns_and_sorts() -> None:
    ref = pd.DataFrame({"Contract": ["A"] * 50 + ["B"] * 50, "tenure": np.arange(100.0)})
    cur = pd.DataFrame({"Contract": ["A"] * 95 + ["B"] * 5, "tenure": np.arange(100.0)})
    report = drift_report(ref, cur, features=["Contract", "tenure", "not_there"])
    assert report["feature"].tolist() == ["Contract", "tenure"]
    assert report.loc[0, "status"] == "major" and report.loc[1, "status"] == "stable"


# --- single customer -------------------------------------------------------------


def test_make_customer_fills_total_charges_and_validates() -> None:
    row = helpers.make_customer(helpers.DEFAULT_CUSTOMER)
    assert len(row) == 1 and row.loc[0, config.ID_COL] == "NEW-CUSTOMER"
    expected = helpers.DEFAULT_CUSTOMER["tenure"] * helpers.DEFAULT_CUSTOMER["MonthlyCharges"]
    assert row.loc[0, "TotalCharges"] == pytest.approx(expected)
    assert list(row.columns) == [config.ID_COL, *helpers.FEATURE_COLUMNS]

    with pytest.raises(ValueError, match="Missing fields"):
        helpers.make_customer({"tenure": 3})
    with pytest.raises(ValueError, match="cannot be negative"):
        helpers.make_customer({**helpers.DEFAULT_CUSTOMER, "tenure": -1})


def test_form_options_cover_defaults() -> None:
    for col, choices in helpers.FORM_OPTIONS.items():
        assert helpers.DEFAULT_CUSTOMER[col] in choices, col
    assert set(helpers.FORM_OPTIONS) | {"tenure", "MonthlyCharges", "TotalCharges"} == set(
        helpers.FEATURE_COLUMNS
    )


def test_score_and_what_if_on_toy_model() -> None:
    model, _, _ = _calibrated_toy()
    customer = clean(helpers.make_customer(helpers.DEFAULT_CUSTOMER))
    result = helpers.score(model, customer)
    assert len(result) == 1 and 0 <= result.loc[0, "churn_probability"] <= 1
    assert result.loc[0, "risk_tier"] in {"High", "Medium", "Low"}

    base = float(model.predict_proba(customer.drop(columns=[config.ID_COL]))[0])
    assert helpers.what_if(model, customer, {}) == pytest.approx(base)
    # toy data: churn only for month-to-month customers, so a 2-year contract lowers risk
    assert helpers.what_if(model, customer, {"Contract": "Two year"}) < base


# --- batch -------------------------------------------------------------------------


def _raw_batch(n: int = 3) -> pd.DataFrame:
    row = {config.ID_COL: "x", **helpers.make_customer(helpers.DEFAULT_CUSTOMER).iloc[0]}
    df = pd.DataFrame([row] * n)
    df[config.ID_COL] = [f"c{i}" for i in range(n)]
    return df


def test_prepare_batch_cleans_and_drops_outcome() -> None:
    raw = _raw_batch().assign(Churn=["Yes", "No", "No"])
    raw["TotalCharges"] = raw["TotalCharges"].astype(str)
    raw.loc[0, ["tenure", "TotalCharges"]] = [0, " "]
    out = helpers.prepare_batch(raw)
    assert config.TARGET_COL not in out.columns
    assert out.loc[0, "TotalCharges"] == 0.0 and out["TotalCharges"].dtype == float


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.iloc[0:0], "no rows"),
        (lambda d: d.drop(columns=["Contract"]), "Missing columns"),
        (lambda d: d.assign(customerID="same"), "Duplicate"),
    ],
)
def test_prepare_batch_rejects_bad_files(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        helpers.prepare_batch(mutate(_raw_batch()))


def test_csv_round_trip_keeps_ids_as_text() -> None:
    raw = _raw_batch().assign(customerID=["0001", "0002", "0003"])
    parsed = helpers.read_csv_upload(helpers.to_csv_bytes(raw))
    assert parsed[config.ID_COL].tolist() == ["0001", "0002", "0003"]
    assert len(helpers.prepare_batch(parsed)) == 3


# --- insights ----------------------------------------------------------------------


def test_load_insights_tolerates_missing_files(tmp_path) -> None:
    ins = helpers.load_insights(tmp_path, tmp_path)
    assert ins.model_summary == {} and ins.tiers is None and ins.figures == {}
    assert helpers.headline_metrics(ins) == {}


def test_headline_metrics_formats_saved_results(tmp_path) -> None:
    summary = {
        "test": {
            "final": {"pr_auc": 0.6713, "lift_top10": 2.92},
            "baseline": {"pr_auc": 0.6412},
        }
    }
    (tmp_path / "model_summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"strategy": ["Model targeting"], "profit_per_1000_customers": [12441.0]}).to_csv(
        tmp_path / "strategy_comparison.csv", index=False
    )
    cards = helpers.headline_metrics(helpers.load_insights(tmp_path, tmp_path))
    assert cards == {
        "Test PR-AUC": "0.671",
        "vs baseline": "+0.030",
        "Top-10% lift": "2.92x",
        "Profit / 1,000 customers": "12,441",
    }
