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


# --- model staleness notices --------------------------------------------------


def _card_setup(tmp_path):  # type: ignore[no-untyped-def]
    """A raw file, a model file and a matching model card written the way the pipeline does."""
    from src.models.model_card import build_model_card
    from tests.test_models import _fake_metrics

    metrics, raw = _fake_metrics(tmp_path)
    model = tmp_path / "final_model.joblib"
    model.write_bytes(b"model")
    card = build_model_card(metrics, raw, model_file=model)
    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(card))
    return card_path, raw, model, card


def test_notices_empty_when_everything_matches(tmp_path) -> None:  # type: ignore[no-untyped-def]
    card_path, raw, model, _ = _card_setup(tmp_path)
    assert helpers.model_notices(card_path, raw, model) == []


def test_notice_when_model_card_is_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    notices = helpers.model_notices(tmp_path / "none.json", tmp_path / "raw.csv", tmp_path / "m")
    assert [n.level for n in notices] == ["info"] and "model card" in notices[0].message


def test_warning_when_raw_data_changed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    card_path, raw, model, _ = _card_setup(tmp_path)
    raw.write_text("a,b\n9,9\n")
    notices = helpers.model_notices(card_path, raw, model)
    assert [n.level for n in notices] == ["warning"] and "raw data has changed" in notices[
        0
    ].message


def test_missing_raw_file_skips_the_data_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    card_path, raw, model, _ = _card_setup(tmp_path)
    raw.unlink()
    assert helpers.model_notices(card_path, raw, model) == []


def test_warning_when_model_file_replaced_after_card(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import os

    card_path, raw, model, _ = _card_setup(tmp_path)
    os.utime(model, (1_000_000_000, 1_000_000_000))  # a different modification time
    notices = helpers.model_notices(card_path, raw, model)
    assert [n.level for n in notices] == ["warning"] and "replaced" in notices[0].message


def test_warning_when_library_versions_differ(tmp_path) -> None:  # type: ignore[no-untyped-def]
    card_path, raw, model, card = _card_setup(tmp_path)
    card["code"]["libraries"]["scikit-learn"] = "0.0.1"
    card_path.write_text(json.dumps(card))
    notices = helpers.model_notices(card_path, raw, model)
    assert [n.level for n in notices] == ["warning"]
    assert "scikit-learn 0.0.1" in notices[0].message


def test_info_when_model_is_old_and_warnings_sort_first(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime, timedelta

    card_path, raw, model, _ = _card_setup(tmp_path)
    later = datetime.now(UTC) + timedelta(days=200)
    old = helpers.model_notices(card_path, raw, model, now=later)
    assert [n.level for n in old] == ["info"] and "days ago" in old[0].message
    raw.write_text("a,b\n9,9\n")  # add a warning as well
    both = helpers.model_notices(card_path, raw, model, now=later)
    assert [n.level for n in both] == ["warning", "info"]


def test_provenance_caption(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, _, _, card = _card_setup(tmp_path)
    text = helpers.provenance_caption(card)
    assert text.startswith("Trained 20") and f"data {card['data']['sha256'][:8]}" in text
    assert helpers.provenance_caption(None) == "Provenance unknown (no model card)."
    assert helpers.load_model_card(tmp_path / "missing.json") is None


def test_incomplete_model_card_gives_a_note_not_a_crash(tmp_path) -> None:  # type: ignore[no-untyped-def]
    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps({"model": {"name": "xgboost"}}))
    notices = helpers.model_notices(card_path, tmp_path / "raw.csv", tmp_path / "m")
    assert [n.level for n in notices] == ["info"] and "incomplete" in notices[0].message
    assert helpers.provenance_caption({"model": {}}) == "Trained unknown date"


# --- input validation ---------------------------------------------------------


def _bad(**cols):  # type: ignore[no-untyped-def]
    return _raw_batch().assign(**cols)


def test_form_options_cover_every_model_category_and_match_training_data() -> None:
    from src.features.pipeline import RAW_CATEGORICAL

    assert set(RAW_CATEGORICAL) <= set(helpers.FORM_OPTIONS)
    train = config.PROCESSED_DIR / "train.csv"
    if train.exists():  # needs the pipeline's outputs; otherwise only the coverage check runs
        df = pd.read_csv(train)
        for col in RAW_CATEGORICAL:
            assert set(df[col].unique()) <= {str(v) for v in helpers.FORM_OPTIONS[col]}, col


def test_wrong_case_category_is_rejected_with_a_hint() -> None:
    with pytest.raises(ValueError, match="unrecognised") as exc:
        helpers.prepare_batch(_bad(Contract="MONTH-TO-MONTH"))
    assert "'MONTH-TO-MONTH' -> 'Month-to-month'" in str(exc.value)
    assert "Allowed: Month-to-month, One year, Two year" in str(exc.value)


def test_unknown_category_names_the_column_and_rows() -> None:
    with pytest.raises(ValueError) as exc:
        helpers.prepare_batch(_bad(Contract="Three year"))
    msg = str(exc.value)
    assert "'Contract'" in msg and "'Three year'" in msg and "3 row(s)" in msg and "c0" in msg


def test_blank_categorical_is_rejected() -> None:
    with pytest.raises(ValueError, match="'gender' is blank or missing"):
        helpers.prepare_batch(_bad(gender=[None, "Male", "Female"]))


@pytest.mark.parametrize(
    ("col", "value", "message"),
    [
        ("tenure", -5, "'tenure' is negative"),
        ("tenure", 5000, "'tenure' is above 120"),
        ("MonthlyCharges", float("nan"), "'MonthlyCharges' is blank or missing"),
        ("MonthlyCharges", "29.85 USD", "'MonthlyCharges' has non-numeric values"),
        ("MonthlyCharges", 1e6, "'MonthlyCharges' is above 300"),
        ("TotalCharges", "abc", "'TotalCharges' has non-numeric values"),
    ],
)
def test_bad_numbers_are_rejected(col: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        helpers.prepare_batch(_bad(**{col: value}).astype({col: object}))


def test_blank_total_charges_only_allowed_for_new_customers() -> None:
    ok = _bad(tenure=0, TotalCharges=" ").astype({"TotalCharges": object})
    assert helpers.prepare_batch(ok)["TotalCharges"].eq(0.0).all()
    with pytest.raises(ValueError, match="'TotalCharges' is blank or missing"):
        helpers.prepare_batch(_bad(tenure=12, TotalCharges=" ").astype({"TotalCharges": object}))


def test_senior_citizen_yes_no_text_is_accepted_and_garbage_rejected() -> None:
    out = helpers.prepare_batch(_bad(SeniorCitizen=["Yes", "no", " YES "]))
    assert out["SeniorCitizen"].tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="'SeniorCitizen' must be 0/1"):
        helpers.prepare_batch(_bad(SeniorCitizen=["maybe", 0, 1]))


def test_all_problems_are_reported_together() -> None:
    with pytest.raises(ValueError, match="Found 3 problem") as exc:
        helpers.prepare_batch(_bad(Contract="Three year", tenure=-1, gender=None))
    msg = str(exc.value)
    assert "'Contract'" in msg and "'tenure'" in msg and "'gender'" in msg


def test_whitespace_padding_is_still_fine() -> None:
    out = helpers.prepare_batch(_bad(Partner=" No ", Contract=" Month-to-month"))
    assert out["Contract"].eq("Month-to-month").all()


# --- reading uploads ----------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "file is empty"),
        (b"   \n\n", "file is empty"),
        (bytes(range(256)) * 20, "not a UTF-8 text CSV"),
        (b"customerID;tenure;Contract\nA;1;Month-to-month\n", "semicolon"),
        (b"customerID\ttenure\tContract\nA\t1\tMonth-to-month\n", "tab"),
    ],
)
def test_read_csv_upload_gives_plain_errors(data: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        helpers.read_csv_upload(data)


def test_read_csv_upload_handles_bom_and_crlf() -> None:
    raw = _raw_batch()
    data = ("﻿" + raw.to_csv(index=False, lineterminator="\r\n")).encode("utf-8")
    out = helpers.read_csv_upload(data)
    assert list(out.columns) == list(raw.columns) and len(out) == 3
    assert helpers.prepare_batch(out).shape[0] == 3


# --- dashboard server config --------------------------------------------------


def test_dashboard_config_is_local_only_with_an_upload_cap() -> None:
    import tomllib

    cfg = tomllib.loads((config.PROJECT_ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["address"] == "localhost"
    # Streamlit's default is 200 MB; scoring runs at roughly 2,000 customers per second.
    assert 1 <= cfg["server"]["maxUploadSize"] <= 50


# --- download safety (spreadsheet formula injection) ---------------------------


DANGEROUS_IDS = ["=1+1", "+cmd|' /C calc'!A0", "-2+3", "@SUM(A1)", "\t=1", "\r=1"]


def test_download_neutralizes_formula_like_text() -> None:
    import io

    ids = [*DANGEROUS_IDS, "7590-VHVEG", "ünï-客户", "a=b", "x,y", 'say "hi"']
    df = pd.DataFrame({"customerID": ids, "reason_1": ["=cmd"] * len(ids), "p": [-0.5] * len(ids)})
    back = pd.read_csv(io.BytesIO(helpers.to_csv_bytes(df)), dtype={"customerID": str})
    out = dict(zip(ids, back["customerID"], strict=True))
    for bad in DANGEROUS_IDS:
        assert out[bad] == "'" + bad  # prefixed, original text preserved after the apostrophe
    for safe in ["7590-VHVEG", "ünï-客户", "a=b", "x,y", 'say "hi"']:
        assert out[safe] == safe  # ordinary IDs, including odd characters, are untouched
    assert (back["reason_1"] == "'=cmd").all()
    assert (back["p"] == -0.5).all()  # negative numbers in numeric columns stay numbers


def test_escape_formulas_does_not_modify_its_input_and_skips_numbers() -> None:
    df = pd.DataFrame(
        {"id": ["=x", "ok"], "n": [-1, 2], "flag": [True, False], "none": [None, "-y"]}
    )
    before = df.copy()
    out = helpers.escape_formulas(df)
    pd.testing.assert_frame_equal(df, before)
    assert out["id"].tolist() == ["'=x", "ok"]
    assert out["n"].tolist() == [-1, 2] and out["flag"].tolist() == [True, False]
    assert out["none"].tolist()[1] == "'-y" and pd.isna(out["none"].tolist()[0])


def test_scored_batch_with_formula_ids_downloads_safely() -> None:
    raw = _raw_batch().assign(customerID=["=1+1", "@SUM(A1)", "normal-1"])
    model, _, _ = _calibrated_toy()
    out = helpers.score(model, helpers.prepare_batch(raw))
    text = helpers.to_csv_bytes(out).decode()
    assert "'=1+1" in text and "'@SUM(A1)" in text and "normal-1" in text
    assert not any(line.startswith(("=", "@")) for line in text.splitlines())
