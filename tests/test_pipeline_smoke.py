"""End-to-end smoke test: the whole pipeline on synthetic data, in an isolated directory.

Runs ``python -m src.pipeline`` as a subprocess with ``CHURN_ROOT`` pointing at a temp
directory, so nothing in the repository is read or written, and with a tiny tuning budget.
It checks that every stage runs and hands its outputs to the next one, which the unit tests
(each stage in isolation) cannot. Skip it for quick runs with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from tests.synthetic import make_synthetic_telco


def test_synthetic_data_matches_raw_schema() -> None:
    from src.data.clean import clean
    from src.data.load import validate_schema

    df = make_synthetic_telco(300)
    validate_schema(df)
    cleaned = clean(df)
    assert cleaned[config.TARGET_COL].isin([0, 1]).all()
    assert 0.1 < cleaned[config.TARGET_COL].mean() < 0.7
    assert (cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"] == 0).all()


@pytest.mark.slow
def test_pipeline_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    make_synthetic_telco(700).to_csv(raw / config.RAW_DATA_FILE.name, index=False)

    env = {**os.environ, "CHURN_ROOT": str(tmp_path), "CHURN_TUNE_ITER": "2"}
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline"],
        cwd=config.PROJECT_ROOT if not os.environ.get("CHURN_ROOT") else Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]

    expected = [
        "data/interim/telco_clean.csv",
        "data/processed/train.csv",
        "models/final_model.joblib",
        "models/baseline.joblib",
        "models/model_card.json",
        "reports/metrics/model_summary.json",
        "reports/metrics/business_summary.json",
        "reports/metrics/feature_list.json",
    ]
    missing = [p for p in expected if not (tmp_path / p).exists()]
    assert not missing, f"pipeline did not produce: {missing}"

    summary = json.loads((tmp_path / "reports/metrics/model_summary.json").read_text())
    assert summary["test"]["final"]["roc_auc"] > 0.6  # the synthetic signal is learnable
    card = json.loads((tmp_path / "models/model_card.json").read_text())
    assert card["model"]["name"] == summary["final_model"]
    assert card["data"]["splits"]["train"]["rows"] > 0
