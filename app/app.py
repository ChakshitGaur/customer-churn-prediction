"""Streamlit dashboard for the churn model.

Run from the project root:  .venv\\Scripts\\python -m streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` and `app` importable

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import helpers  # noqa: E402
from src.data.clean import clean  # noqa: E402
from src.models.monitor import PSI_MAJOR, PSI_MODERATE, drift_report  # noqa: E402
from src.models.predict import load_calibrated  # noqa: E402

st.set_page_config(page_title="Churn Risk Dashboard", page_icon="📉", layout="wide")

TIER_COLORS = {"High": "#d64545", "Medium": "#e0a030", "Low": "#3a8f5c"}


@st.cache_resource
def get_model():
    """Load the calibrated model once per server process."""
    return load_calibrated()


@st.cache_data
def get_reference() -> pd.DataFrame:
    return helpers.training_reference()


def tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#888")
    return (
        f"<span style='background:{color};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600'>{tier} risk</span>"
    )


# --- Guard: pipeline must have run ---------------------------------------------
if not helpers.model_available():
    st.title("Churn Risk Dashboard")
    st.error(
        "No trained model found in `models/`. Run these from the project root, then reload:\n\n"
        "```\n.venv\\Scripts\\python -m src.data.download\n"
        ".venv\\Scripts\\python -m src.pipeline\n```"
    )
    st.stop()

model = get_model()

# --- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        f"Calibrated **{model.name}** model trained on the IBM Telco churn data. "
        "Scores are monthly churn probabilities."
    )
    st.markdown(
        f"- **High**: probability ≥ {model.threshold:.3f} → contact\n"
        f"- **Medium**: ≥ {model.medium_cutoff:.3f} (above-average risk)\n"
        f"- **Low**: everything else"
    )
    st.caption(
        "Business costs are placeholders in `src/config.py`. "
        "Reasons come from SHAP; actions from `src/models/recommend.py`."
    )

st.title("📉 Churn Risk Dashboard")
tab_single, tab_batch, tab_insights = st.tabs(
    ["Single customer", "Batch scoring", "Model insights"]
)

# --- Tab 1: single customer ------------------------------------------------------
with tab_single:
    st.subheader("Score one customer")
    d = helpers.DEFAULT_CUSTOMER
    opts = helpers.FORM_OPTIONS

    def pick(label: str, col: str, container) -> object:
        return container.selectbox(label, opts[col], index=opts[col].index(d[col]), key=col)

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Account**")
        inputs = {
            "tenure": c1.number_input("Tenure (months)", 0, 100, int(d["tenure"])),
            "MonthlyCharges": c1.number_input(
                "Monthly charges", 0.0, 200.0, float(d["MonthlyCharges"]), step=5.0
            ),
            "Contract": pick("Contract", "Contract", c1),
            "PaymentMethod": pick("Payment method", "PaymentMethod", c1),
            "PaperlessBilling": pick("Paperless billing", "PaperlessBilling", c1),
        }
        c2.markdown("**Services**")
        for label, col in [
            ("Internet service", "InternetService"),
            ("Online security", "OnlineSecurity"),
            ("Tech support", "TechSupport"),
            ("Online backup", "OnlineBackup"),
            ("Device protection", "DeviceProtection"),
            ("Streaming TV", "StreamingTV"),
            ("Streaming movies", "StreamingMovies"),
        ]:
            inputs[col] = pick(label, col, c2)
        c3.markdown("**Customer**")
        for label, col in [
            ("Gender", "gender"),
            ("Senior citizen (1 = yes)", "SeniorCitizen"),
            ("Partner", "Partner"),
            ("Dependents", "Dependents"),
            ("Phone service", "PhoneService"),
            ("Multiple lines", "MultipleLines"),
        ]:
            inputs[col] = pick(label, col, c3)
        submitted = st.form_submit_button("Score customer", type="primary")

    customer = None
    if submitted:
        try:
            customer = clean(helpers.make_customer(inputs))
        except ValueError as err:
            st.error(str(err))
    if customer is not None:
        result = helpers.score(model, customer).iloc[0]
        left, right = st.columns([1, 2])
        with left:
            st.metric("Churn probability", f"{result['churn_probability']:.0%}")
            st.markdown(tier_badge(result["risk_tier"]), unsafe_allow_html=True)
            st.markdown("#### Top reasons")
            reasons = [result[f"reason_{k}"] for k in (1, 2, 3) if result[f"reason_{k}"]]
            st.markdown("\n".join(f"- {r}" for r in reasons) or "_No risk-increasing factors_")
            st.markdown("#### Recommended action")
            st.markdown(f"**{result['primary_action']}**")
            if result["secondary_action"]:
                st.markdown(f"Also: {result['secondary_action']}")
            if inputs["Contract"] == "Month-to-month":
                alt = helpers.what_if(model, customer, {"Contract": "One year"})
                st.markdown("#### What if")
                st.metric(
                    "On a 1-year contract",
                    f"{alt:.0%}",
                    delta=f"{100 * (alt - result['churn_probability']):+.0f} pts",
                    delta_color="inverse",
                )
                st.caption("Model estimate, not a causal effect: test offers with a control group.")
        with right:
            st.markdown("#### Why this score (SHAP, log-odds)")
            fig = helpers.customer_waterfall(model, customer)
            st.pyplot(fig, clear_figure=True)
            plt.close("all")

# --- Tab 2: batch scoring ------------------------------------------------------------
with tab_batch:
    st.subheader("Score a batch of customers")
    st.write(
        "Upload a CSV in the raw Telco format (`customerID` plus the 19 feature columns; "
        "a `Churn` column is allowed and ignored), or try the sample."
    )
    up_col, sample_col = st.columns([3, 1])
    upload = up_col.file_uploader("Customer CSV", type="csv")
    use_sample = sample_col.button("Use sample (200 test customers)")

    raw = None
    if upload is not None:
        raw = helpers.read_csv_upload(upload.getvalue())
    elif use_sample or st.session_state.get("batch_sample"):
        st.session_state["batch_sample"] = True
        raw = helpers.sample_batch()

    batch = None
    if raw is not None:
        try:
            batch = helpers.prepare_batch(raw)
        except ValueError as err:
            st.error(f"Could not read the file: {err}")
    if batch is not None:
        risk = helpers.score(model, batch)

        counts = risk["risk_tier"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers", f"{len(risk):,}")
        m2.metric("High risk (contact)", f"{counts.get('High', 0):,}")
        m3.metric("Medium", f"{counts.get('Medium', 0):,}")
        m4.metric("Expected churners", f"{risk['churn_probability'].sum():,.0f}")

        tiers = st.multiselect("Show tiers", ["High", "Medium", "Low"], default=["High"])
        view = risk[risk["risk_tier"].isin(tiers)]
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        st.download_button(
            "Download full ranked list (CSV)",
            helpers.to_csv_bytes(risk),
            file_name="customer_risk_list.csv",
            mime="text/csv",
        )

        st.markdown("#### Data drift vs. training data")
        st.caption(
            f"PSI < {PSI_MODERATE} stable · {PSI_MODERATE}–{PSI_MAJOR} investigate · "
            f"≥ {PSI_MAJOR} major shift. See reports/monitoring_plan.md."
        )
        drift = drift_report(get_reference(), batch)
        st.dataframe(drift, hide_index=True)
        if (drift["status"] == "major").any():
            st.warning("Major drift detected: scores for this batch may be unreliable.")

# --- Tab 3: insights -------------------------------------------------------------------
with tab_insights:
    insights = helpers.load_insights()
    cards = helpers.headline_metrics(insights)
    if cards:
        for col, (label, value) in zip(st.columns(len(cards)), cards.items(), strict=True):
            col.metric(label, value)

    left, right = st.columns(2)
    if insights.tiers is not None:
        left.markdown("#### Risk tiers (test set)")
        left.dataframe(insights.tiers.round(3), hide_index=True)
    if insights.strategies is not None:
        right.markdown("#### Targeting strategies (test set)")
        right.dataframe(
            insights.strategies[["strategy", "contacts", "customers_saved", "profit"]],
            hide_index=True,
        )

    figs = list(insights.figures.items())
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, figs[i : i + 2], strict=False):
            col.markdown(f"#### {title}")
            col.image(str(path), width="stretch")

    st.caption("Full write-up: reports/final_report.md")
