"""Smoke tests that run the real Streamlit script (app/app.py) with Streamlit's AppTest.

They use a small toy model and patch the model loader, so they need neither the raw data nor
the pipeline's outputs and run in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import helpers
from tests.test_models import _calibrated_toy

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_dashboard_shows_setup_help_when_no_model_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "model_available", lambda *a, **k: False)
    at = _run()
    assert not at.exception
    assert len(at.error) == 1 and "src.pipeline" in at.error[0].value
    assert not at.tabs  # nothing else is rendered without a model


@pytest.fixture
def dashboard(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    import src.models.predict as predict

    model, _, _ = _calibrated_toy()
    monkeypatch.setattr(helpers, "model_available", lambda *a, **k: True)
    monkeypatch.setattr(predict, "load_calibrated", lambda: model)
    monkeypatch.setattr(helpers, "load_model_card", lambda *a, **k: None)
    monkeypatch.setattr(helpers, "model_notices", lambda *a, **k: [])
    import streamlit as st

    st.cache_resource.clear()  # get_model() is cached per process; start from the toy model
    return _run()


def test_dashboard_loads_with_three_tabs_and_no_errors(dashboard: AppTest) -> None:
    assert not dashboard.exception and not dashboard.error
    assert [t.label for t in dashboard.tabs] == [
        "Single customer",
        "Batch scoring",
        "Model insights",
    ]


@pytest.mark.parametrize(
    "numbers",
    [{}, {"tenure": 0}, {"tenure": 100}, {"monthly": 0.0}, {"monthly": 200.0}],
    ids=["default", "tenure-0", "tenure-max", "charges-0", "charges-max"],
)
def test_single_customer_scores_at_boundary_inputs(dashboard: AppTest, numbers: dict) -> None:
    for box in dashboard.number_input:
        for prefix, value in numbers.items():
            if box.label.lower().startswith(prefix):
                box.set_value(value)
    next(b for b in dashboard.button if b.label == "Score customer").click()
    dashboard.run()
    assert not dashboard.exception and not dashboard.error
    probability = next(m for m in dashboard.metric if m.label == "Churn probability")
    assert 0 <= int(probability.value.rstrip("%")) <= 100


def test_batch_tab_is_idle_until_a_file_is_chosen(dashboard: AppTest) -> None:
    assert not dashboard.exception and not dashboard.error
    assert not [m for m in dashboard.metric if m.label == "Customers"]
