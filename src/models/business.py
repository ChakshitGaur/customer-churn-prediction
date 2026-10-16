"""Phase 6 orchestration: calibration, ranked risk list, and business impact.

The ranked list scores the **test** customers: the only labelled customers the model has
never seen, so the list behaves like a real "next month" scoring run and its quality
can be checked against actual outcomes.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    METRICS_DIR,
    MODELS_DIR,
    OFFER_SUCCESS_RATE,
    REPORTS_DIR,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    TARGET_COL,
)
from src.data.split import Splits
from src.features.pipeline import split_xy
from src.models.evaluate import (
    calibration_summary,
    compute_metrics,
    plot_calibration,
    save_figure,
    save_json,
)
from src.models.predict import RISK_TIERS, build_risk_list, calibrate, save_calibrated
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = REPORTS_DIR / "output"  # customer-level files: gitignored
RISK_LIST_FILE = OUTPUT_DIR / "customer_risk_list.csv"


def strategy_comparison(y: np.ndarray, contact: np.ndarray) -> pd.DataFrame:
    """Expected profit of the model's targeting vs. simple alternatives.

    Strategies: contact nobody; contact everyone; contact the same number of customers
    at random (expected value); contact the customers the model flags.

    Returns:
        One row per strategy with contacts, churners reached, expected customers saved,
        cost, profit, and profit per 1,000 customers.
    """
    n, n_contact, base = len(y), int(contact.sum()), float(y.mean())
    rows = {
        "Contact nobody": (0, 0.0),
        "Contact everyone": (n, float(y.sum())),
        "Random (same number as model)": (n_contact, n_contact * base),
        "Model targeting": (n_contact, float(y[contact].sum())),
    }
    out = []
    for name, (contacts, churners_reached) in rows.items():
        saved = churners_reached * OFFER_SUCCESS_RATE
        cost = contacts * RETENTION_OFFER_COST
        profit = saved * RETAINED_CUSTOMER_VALUE - cost
        out.append(
            {
                "strategy": name,
                "contacts": contacts,
                "churners_reached": round(churners_reached, 1),
                "customers_saved": round(saved, 1),
                "cost": round(cost, 0),
                "profit": round(profit, 0),
                "profit_per_1000_customers": round(1000 * profit / n, 0),
            }
        )
    return pd.DataFrame(out)


def tier_summary(risk_list: pd.DataFrame, y_by_id: pd.Series) -> pd.DataFrame:
    """Customers, share, mean predicted probability, and actual churn rate per tier."""
    df = risk_list.assign(actual=risk_list["customerID"].map(y_by_id))
    g = df.groupby("risk_tier").agg(
        customers=("customerID", "size"),
        mean_probability=("churn_probability", "mean"),
        actual_churn_rate=("actual", "mean"),
        churners=("actual", "sum"),
    )
    g["share"] = g["customers"] / len(df)
    g["share_of_churners"] = g["churners"] / df["actual"].sum()
    return g.reindex(list(RISK_TIERS))


def run_business_output(splits: Splits) -> dict[str, Any]:
    """Run all of Phase 6 and save the calibrated model, risk list, and summaries.

    Args:
        splits: Train/val/test splits from Phase 2.

    Returns:
        Summary dict (also written to ``reports/metrics/business_summary.json``).
    """
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    X_val, y_val = split_xy(splits.val)
    X_test, y_test = split_xy(splits.test)
    y = y_test.to_numpy()

    # 1. Calibrate on validation (threshold mapped, so contacts are unchanged)
    model = calibrate(final, X_val, y_val, medium_cutoff=float(splits.train[TARGET_COL].mean()))
    save_calibrated(model)
    raw_test = final["pipeline"].predict_proba(X_test)[:, 1]
    cal_test = model.predict_proba(X_test)
    save_figure(
        plot_calibration(y, {"before (raw)": raw_test, "after (calibrated)": cal_test}),
        "business_calibration_before_after",
    )

    # 2. Ranked risk list for test customers
    risk_list = build_risk_list(model, splits.test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_list.to_csv(RISK_LIST_FILE, index=False)

    # 3. Summaries
    y_by_id = splits.test.set_index("customerID")[TARGET_COL]
    tiers = tier_summary(risk_list, y_by_id)
    tiers.to_csv(METRICS_DIR / "risk_tiers.csv")
    contact = cal_test >= model.threshold
    strategies = strategy_comparison(y, contact)
    strategies.to_csv(METRICS_DIR / "strategy_comparison.csv", index=False)
    contacted = risk_list[risk_list["contact"]]
    actions = {
        "primary": contacted["primary_action"].value_counts().to_dict(),
        "secondary": contacted["secondary_action"].replace("", "(none)").value_counts().to_dict(),
    }
    reasons = (
        pd.concat([contacted[f"reason_{k}"] for k in (1, 2, 3)])
        .str.split(" = ")
        .str[0]
        .replace("", np.nan)
        .dropna()
        .value_counts()
    )

    summary = {
        "calibration": {
            "method": "sigmoid (Platt), fit on validation",
            "test_before": calibration_summary(y, raw_test),
            "test_after": calibration_summary(y, cal_test),
            "brier_before": compute_metrics(y, raw_test)["brier"],
            "brier_after": compute_metrics(y, cal_test)["brier"],
        },
        "threshold_raw": model.raw_threshold,
        "threshold_calibrated": model.threshold,
        "break_even_probability": RETENTION_OFFER_COST
        / (RETAINED_CUSTOMER_VALUE * OFFER_SUCCESS_RATE),
        "medium_cutoff": model.medium_cutoff,
        "test_customers": len(risk_list),
        "tiers": tiers.round(4).to_dict(orient="index"),
        "strategies": strategies.to_dict(orient="records"),
        "contacted_actions": actions,
        "contacted_reason_features": reasons.to_dict(),
    }
    save_json(summary, METRICS_DIR / "business_summary.json")
    logger.info(
        "Risk list: %d customers (%d High) -> %s | model profit %.0f vs random %.0f",
        len(risk_list),
        int((risk_list["risk_tier"] == "High").sum()),
        RISK_LIST_FILE.name,
        strategies.loc[strategies["strategy"] == "Model targeting", "profit"].iloc[0],
        strategies.loc[strategies["strategy"].str.startswith("Random"), "profit"].iloc[0],
    )
    return summary
