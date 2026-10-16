"""Feature tests: environment, engineered features, and leakage-safe preprocessing."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build_features import (
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureBuilder,
    add_features,
)
from src.features.pipeline import (
    BINARY_FEATURES,
    DUPLICATE_FEATURES,
    build_preprocessor,
    make_model_pipeline,
    split_xy,
)


@pytest.mark.parametrize(
    "module",
    ["pandas", "numpy", "sklearn", "xgboost", "lightgbm", "shap", "matplotlib", "seaborn"],
)
def test_stack_imports(module: str) -> None:
    importlib.import_module(module)


def _customers() -> pd.DataFrame:
    """Three hand-built cleaned customers with known expected features."""
    return pd.DataFrame(
        {
            config.ID_COL: ["new_fiber", "loyal_dsl", "no_internet"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0.0, 60.0, 24.0],
            "PhoneService": ["Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "No", "No internet service"],
            "Contract": ["Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Mailed check"],
            "MonthlyCharges": [90.0, 70.0, 20.0],
            "TotalCharges": [0.0, 4200.0, 480.0],
            config.TARGET_COL: [1, 0, 0],
        }
    )


def test_add_features_known_values() -> None:
    f = add_features(_customers()).set_index(config.ID_COL)
    new, loyal, offline = f.loc["new_fiber"], f.loc["loyal_dsl"], f.loc["no_internet"]

    assert new["tenure_band"] == "0-6" and new["is_new_customer"] == 1
    assert new["fiber_no_protection"] == 1 and new["m2m_no_techsupport"] == 1
    assert new["avg_monthly_spend"] == 90.0  # tenure 0 -> falls back to MonthlyCharges
    assert new["num_addons"] == 1 and new["num_services"] == 3  # phone + internet + TV

    assert loyal["tenure_band"] == "49+" and loyal["contract_months"] == 24
    assert loyal["num_protection"] == 4 and loyal["auto_pay"] == 1
    assert loyal["avg_monthly_spend"] == pytest.approx(70.0)
    assert loyal["charge_trend_ratio"] == pytest.approx(1.0)

    assert offline["has_internet"] == 0 and offline["num_addons"] == 0
    assert offline["OnlineSecurity"] == "No"  # "No internet service" collapsed
    assert offline["has_family"] == 1


def test_add_features_does_not_modify_input_or_use_target() -> None:
    df = _customers()
    snapshot = df.copy()
    with_target = add_features(df)
    without_target = add_features(df.drop(columns=config.TARGET_COL))
    pd.testing.assert_frame_equal(df, snapshot)
    engineered = [*ENGINEERED_NUMERIC, *ENGINEERED_BINARY, *ENGINEERED_CATEGORICAL]
    pd.testing.assert_frame_equal(with_target[engineered], without_target[engineered])


def test_feature_builder_is_stateless() -> None:
    X, _ = split_xy(_customers())
    a = FeatureBuilder().fit(X).transform(X)
    b = FeatureBuilder().fit(X.iloc[:1]).transform(X)  # fitting on other data changes nothing
    pd.testing.assert_frame_equal(a, b)


def test_feature_builder_names_match_actual_columns() -> None:
    X, _ = split_xy(_customers())
    fb = FeatureBuilder().fit(X)
    out = add_features(X)
    assert list(fb.get_feature_names_out()) == list(out.columns)


def test_preprocessor_columns_are_correctly_labelled() -> None:
    """Regression test: set_output renames by position, so labels must match values."""
    X, _ = split_xy(_customers())
    Xt = build_preprocessor().fit_transform(X)
    expected = add_features(X)
    for col in BINARY_FEATURES:
        np.testing.assert_array_equal(Xt[f"bin__{col}"], expected[col], err_msg=col)
    assert Xt["cat__tenure_band_0-6"].tolist() == [1.0, 0.0, 0.0]
    assert Xt["cat__InternetService_No"].tolist() == [0.0, 0.0, 1.0]
    assert Xt["bin__fiber_no_protection"].tolist() == [1, 0, 0]


def test_duplicate_features_are_excluded_from_model_inputs() -> None:
    X, _ = split_xy(_customers())
    cols = build_preprocessor().fit_transform(X).columns
    for dup in DUPLICATE_FEATURES:
        assert not any(c.endswith(f"__{dup}") for c in cols), dup


def test_preprocessor_output_is_numeric_and_complete() -> None:
    X, _ = split_xy(_customers())
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in Xt.dtypes)
    assert not any(config.ID_COL in c or config.TARGET_COL in c for c in Xt.columns)


def test_preprocessor_fit_only_uses_training_statistics() -> None:
    X, _ = split_xy(_customers())
    train, other = X.iloc[:2], X.iloc[2:]
    pre = build_preprocessor().fit(train)
    scaler = pre.named_steps["encode"].named_transformers_["num"]
    tenure_idx = list(scaler.feature_names_in_).index("tenure")
    assert scaler.mean_[tenure_idx] == pytest.approx(train["tenure"].mean())
    with pytest.warns(UserWarning, match="unknown categories"):
        pre.transform(other)  # unseen category "No internet" is encoded, not a crash


def test_preprocessor_handles_unseen_category() -> None:
    X, _ = split_xy(_customers())
    pre = build_preprocessor().fit(X)
    unseen = X.iloc[:1].assign(PaymentMethod="Crypto")
    with pytest.warns(UserWarning, match="unknown categories"):
        row = pre.transform(unseen)
    pay_cols = [c for c in row.columns if "PaymentMethod_" in c]
    assert row[pay_cols].to_numpy().sum() == 0  # all-zeros encoding, no error


def test_make_model_pipeline_fits_end_to_end() -> None:
    from sklearn.linear_model import LogisticRegression

    df = pd.concat([_customers()] * 10, ignore_index=True)
    X, y = split_xy(df)
    model = make_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and np.all((proba >= 0) & (proba <= 1))
