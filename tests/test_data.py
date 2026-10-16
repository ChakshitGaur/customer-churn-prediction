"""Data-layer tests. Phase 0: config sanity. Phase 2 adds cleaning/split/leakage tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.clean import clean
from src.data.load import EXPECTED_COLUMNS, load_raw, validate_schema
from src.data.split import MAX_CHURN_RATE_GAP, Splits, check_leakage, split_data


def test_seed_is_fixed() -> None:
    assert config.SEED == 42


def test_split_ratios_sum_to_one() -> None:
    total = config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE
    assert math.isclose(total, 1.0)


def test_paths_are_inside_project() -> None:
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_data_folders_exist() -> None:
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR):
        assert path.is_dir(), f"missing folder: {path}"


# --- load.py ------------------------------------------------------------------


def _toy_raw() -> pd.DataFrame:
    row = {col: ["x"] for col in EXPECTED_COLUMNS}
    row["Churn"] = ["Yes"]
    return pd.DataFrame(row)


def test_validate_schema_accepts_expected_columns() -> None:
    validate_schema(_toy_raw())


def test_validate_schema_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(_toy_raw().drop(columns=["tenure"]))


def test_validate_schema_rejects_unknown_target_label() -> None:
    df = _toy_raw()
    df["Churn"] = ["Maybe"]
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        validate_schema(df)


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


@pytest.mark.skipif(not config.RAW_DATA_FILE.exists(), reason="raw data not downloaded")
def test_load_raw_real_file_is_unmodified() -> None:
    df = load_raw()
    assert df.shape == (7043, 21)
    assert df[config.ID_COL].is_unique
    assert pd.api.types.is_string_dtype(df["TotalCharges"])  # raw text kept for Phase 2


# --- clean.py -----------------------------------------------------------------


def _raw_rows() -> pd.DataFrame:
    """Small raw-format frame covering the edge cases cleaning must handle."""
    base = {col: ["x", "x", "x"] for col in EXPECTED_COLUMNS}
    base.update(
        customerID=["A", "B", "C"],
        tenure=[0, 5, 10],
        MonthlyCharges=[20.0, 30.0, 40.0],
        TotalCharges=[" ", "150.0", " 400.5 "],
        Churn=["No", "Yes", "No"],
        Contract=[" Month-to-month", "One year", "Two year "],
    )
    return pd.DataFrame(base)


def test_clean_fills_blank_total_charges_with_zero() -> None:
    out = clean(_raw_rows())
    assert out["TotalCharges"].tolist() == [0.0, 150.0, 400.5]


def test_clean_encodes_target_and_strips_text() -> None:
    out = clean(_raw_rows())
    assert out["Churn"].tolist() == [0, 1, 0]
    assert out["Contract"].tolist() == ["Month-to-month", "One year", "Two year"]


def test_clean_does_not_modify_input() -> None:
    raw = _raw_rows()
    snapshot = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_clean_rejects_blank_total_charges_for_tenured_customer() -> None:
    raw = _raw_rows()
    raw.loc[1, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="tenure > 0"):
        clean(raw)


def test_clean_drops_leakage_columns() -> None:
    raw = _raw_rows().assign(ChurnReason=["", "Price", ""])
    assert "ChurnReason" not in clean(raw).columns


# --- split.py -----------------------------------------------------------------


def _clean_frame(n: int = 1000, churn_rate: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    return pd.DataFrame(
        {
            config.ID_COL: [f"id{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, n),
            "Contract": rng.choice(["Month-to-month", "One year"], n),
            config.TARGET_COL: (rng.random(n) < churn_rate).astype(int),
        }
    )


def test_split_sizes_and_disjoint_ids() -> None:
    df = _clean_frame()
    s = split_data(df, date_col=None)
    assert (len(s.train), len(s.val), len(s.test)) == (700, 150, 150)
    ids = [set(part[config.ID_COL]) for _, part in s.items()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_is_stratified() -> None:
    df = _clean_frame()
    overall = df[config.TARGET_COL].mean()
    for _, part in split_data(df, date_col=None).items():
        assert abs(part[config.TARGET_COL].mean() - overall) <= MAX_CHURN_RATE_GAP


def test_split_is_reproducible() -> None:
    df = _clean_frame()
    a, b = split_data(df, date_col=None), split_data(df, date_col=None)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_time_split_puts_newest_rows_in_test() -> None:
    df = _clean_frame().assign(snapshot=pd.date_range("2024-01-01", periods=1000, freq="D"))
    s = split_data(df, date_col="snapshot")
    assert s.train["snapshot"].max() < s.val["snapshot"].min()
    assert s.val["snapshot"].max() < s.test["snapshot"].min()


def test_check_leakage_catches_id_overlap() -> None:
    df = _clean_frame()
    s = split_data(df, date_col=None)
    leaky = Splits(pd.concat([s.train, s.test.head(1)]), s.val, s.test)  # one test row copied
    with pytest.raises(ValueError, match="appear in both"):
        check_leakage(df, leaky)


def test_check_leakage_catches_target_copy() -> None:
    df = _clean_frame().assign(churned_flag=lambda d: d[config.TARGET_COL].map({1: "Y", 0: "N"}))
    with pytest.raises(ValueError, match="copy of the target"):
        split_data(df, date_col=None)
