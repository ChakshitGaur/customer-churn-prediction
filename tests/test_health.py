"""Tests for the pipeline health check (src/models/health.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from src.models.health import check_outputs, run_health_check

Mutation = Callable[[Path], None]


def _write_json(path: Path, mutate: Callable[[dict], None] | None = None) -> None:
    data = json.loads(path.read_text())
    if mutate:
        mutate(data)
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path: Path) -> dict[str, Path]:
    """The files of a healthy pipeline run, shaped like the real ones."""
    metrics, models = tmp_path / "metrics", tmp_path / "models"
    metrics.mkdir()
    models.mkdir()
    (metrics / "model_summary.json").write_text(
        json.dumps(
            {
                "final_model": "xgboost",
                "beats_baseline_on_test": True,
                "test": {"final": {"pr_auc": 0.671, "roc_auc": 0.855}},
            }
        )
    )
    (metrics / "business_summary.json").write_text(
        json.dumps(
            {
                "threshold_calibrated": 0.4,
                "calibration": {"test_after": {"ece": 0.033}},
                "tiers": {"High": {"mean_probability": 0.583, "actual_churn_rate": 0.586}},
            }
        )
    )
    (metrics / "split_summary.json").write_text(json.dumps({"test": {"rows": 4}}))
    (models / "model_card.json").write_text(
        json.dumps({"data": {"matches_expected_source": True}, "model": {"name": "xgboost"}})
    )
    risk = tmp_path / "customer_risk_list.csv"
    pd.DataFrame(
        {"customerID": ["a", "b", "c", "d"], "churn_probability": [0.8, 0.5, 0.2, 0.1]}
    ).to_csv(risk, index=False)
    return {"metrics": metrics, "models": models, "risk": risk}


def _check(run: dict[str, Path]) -> list[str]:
    return check_outputs(run["metrics"], run["models"], run["risk"])


def test_healthy_run_has_no_problems(run: dict[str, Path]) -> None:
    assert _check(run) == []


FAILURES: list[tuple[str, Callable[[dict[str, Path]], None], str]] = [
    (
        "unsupported final model",
        lambda r: _write_json(
            r["metrics"] / "model_summary.json", lambda d: d.update(final_model="random_forest")
        ),
        "not one the explanations support",
    ),
    (
        "does not beat baseline",
        lambda r: _write_json(
            r["metrics"] / "model_summary.json", lambda d: d.update(beats_baseline_on_test=False)
        ),
        "does not beat",
    ),
    (
        "test PR-AUC collapsed",
        lambda r: _write_json(
            r["metrics"] / "model_summary.json", lambda d: d["test"]["final"].update(pr_auc=0.55)
        ),
        "test PR-AUC",
    ),
    (
        "test PR-AUC suspiciously high",
        lambda r: _write_json(
            r["metrics"] / "model_summary.json", lambda d: d["test"]["final"].update(pr_auc=0.78)
        ),
        "test PR-AUC",
    ),
    (
        "ROC-AUC too low",
        lambda r: _write_json(
            r["metrics"] / "model_summary.json", lambda d: d["test"]["final"].update(roc_auc=0.7)
        ),
        "ROC-AUC",
    ),
    (
        "calibration error too high",
        lambda r: _write_json(
            r["metrics"] / "business_summary.json",
            lambda d: d["calibration"]["test_after"].update(ece=0.15),
        ),
        "calibration error",
    ),
    (
        "High tier miscalibrated",
        lambda r: _write_json(
            r["metrics"] / "business_summary.json",
            lambda d: d["tiers"]["High"].update(actual_churn_rate=0.3),
        ),
        "High tier predicts",
    ),
    (
        "threshold out of range",
        lambda r: _write_json(
            r["metrics"] / "business_summary.json", lambda d: d.update(threshold_calibrated=0.99)
        ),
        "contact threshold",
    ),
    (
        "data checksum mismatch",
        lambda r: _write_json(
            r["models"] / "model_card.json",
            lambda d: d["data"].update(matches_expected_source=False),
        ),
        "expected source",
    ),
    (
        "card names a different model",
        lambda r: _write_json(
            r["models"] / "model_card.json", lambda d: d["model"].update(name="logreg")
        ),
        "different model",
    ),
    (
        "risk list missing",
        lambda r: r["risk"].unlink(),
        "was not written",
    ),
    (
        "risk list wrong size",
        lambda r: pd.DataFrame({"customerID": ["a"], "churn_probability": [0.5]}).to_csv(
            r["risk"], index=False
        ),
        "test split has 4",
    ),
    (
        "probabilities out of range",
        lambda r: pd.DataFrame(
            {"customerID": list("abcd"), "churn_probability": [1.4, 0.5, 0.2, 0.1]}
        ).to_csv(r["risk"], index=False),
        "out-of-range",
    ),
    (
        "risk list not sorted",
        lambda r: pd.DataFrame(
            {"customerID": list("abcd"), "churn_probability": [0.1, 0.5, 0.2, 0.8]}
        ).to_csv(r["risk"], index=False),
        "not sorted",
    ),
    (
        "duplicate customers",
        lambda r: pd.DataFrame(
            {"customerID": ["a", "a", "c", "d"], "churn_probability": [0.8, 0.5, 0.2, 0.1]}
        ).to_csv(r["risk"], index=False),
        "duplicate",
    ),
]


@pytest.mark.parametrize(("label", "break_it", "message"), FAILURES, ids=[f[0] for f in FAILURES])
def test_each_kind_of_breakage_is_reported(
    run: dict[str, Path], label: str, break_it: Callable[[dict[str, Path]], None], message: str
) -> None:
    break_it(run)
    problems = _check(run)
    assert any(message in p for p in problems), (label, problems)


def test_missing_pipeline_outputs_are_reported_not_raised(tmp_path: Path) -> None:
    problems = check_outputs(tmp_path / "m", tmp_path / "d", tmp_path / "r.csv")
    assert sum("Did the pipeline finish" in p for p in problems) == 4  # four unreadable files
    assert any("was not written" in p for p in problems)


def test_run_health_check_exit_codes(run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.health as health

    monkeypatch.setattr(health, "METRICS_DIR", run["metrics"])
    monkeypatch.setattr(health, "MODELS_DIR", run["models"])
    monkeypatch.setattr(health, "RISK_LIST_FILE", run["risk"])
    monkeypatch.setattr(
        health,
        "check_outputs",
        lambda: check_outputs(run["metrics"], run["models"], run["risk"]),
    )
    assert run_health_check() == 0
    run["risk"].unlink()
    assert run_health_check() == 1
