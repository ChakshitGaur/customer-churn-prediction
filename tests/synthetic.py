"""Synthetic data with the Telco schema, for tests that need a full raw CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import EXPECTED_COLUMNS

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_synthetic_telco(n: int = 700, seed: int = 0) -> pd.DataFrame:
    """Raw-format frame (``TotalCharges`` as text, blanks at tenure 0) with a learnable signal.

    Churn depends on contract type, tenure, fiber internet and monthly charges, so models
    trained on it have something to find, as with the real data.

    Args:
        n: Number of customers.
        seed: Random seed.

    Returns:
        DataFrame with exactly the columns in ``EXPECTED_COLUMNS``.
    """
    rng = np.random.default_rng(seed)

    def yes_no() -> np.ndarray:
        return rng.choice(["Yes", "No"], n)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(
        20 + 35 * (internet == "DSL") + 55 * (internet == "Fiber optic") + rng.normal(0, 5, n), 2
    )
    logit = (
        -0.4
        + 1.5 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.03 * tenure
        + 0.8 * (internet == "Fiber optic")
        + 0.01 * (monthly - 60)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    phone = yes_no()
    total = np.round(tenure * monthly, 2).astype(str).astype(object)
    total[tenure == 0] = " "

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", yes_no()),
            "InternetService": internet,
            **{c: np.where(internet == "No", "No internet service", yes_no()) for c in ADDONS},
            "Contract": contract,
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    return df[EXPECTED_COLUMNS]
