"""Global and per-customer SHAP explanations for the final tree model.

SHAP values are in log-odds (the model's margin) and add up exactly to the prediction:
``base_value + sum(shap_values) = logit(churn probability)``.

One-hot columns are summed back into their original feature (e.g. the three
``Contract_*`` columns become one "Contract" effect), so explanations read as
"Contract = Month-to-month" rather than "Contract_Two year = 0". Summing is exact
because SHAP values are additive.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.features.build_features import add_features  # noqa: E402
from src.features.pipeline import CATEGORICAL_FEATURES  # noqa: E402

# Units shown next to numeric values in customer-facing reasons.
UNITS: dict[str, str] = {"tenure": " months", "MonthlyCharges": "/month", "TotalCharges": " total"}


def original_feature(col: str) -> str:
    """Map a transformed column name back to its source feature.

    ``num__tenure`` -> ``tenure``; ``cat__Contract_Month-to-month`` -> ``Contract``;
    ``cat__tenure_band_0-6`` -> ``tenure_band`` (longest matching categorical prefix wins).
    """
    prefix, _, rest = col.partition("__")
    if prefix != "cat":
        return rest or col
    matches = [c for c in CATEGORICAL_FEATURES if rest.startswith(f"{c}_")]
    return max(matches, key=len) if matches else rest


def compute_shap(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values for every row of ``X`` on the pipeline's transformed features.

    Args:
        pipeline: Fitted ``Pipeline([("prep", ...), ("model", tree_model)])``.
        X: Raw feature rows (as passed to ``pipeline.predict_proba``).

    Returns:
        A :class:`shap.Explanation` with one column per transformed feature.
    """
    Xt = pipeline.named_steps["prep"].transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    return explainer(Xt)


def group_shap(sv: shap.Explanation) -> pd.DataFrame:
    """Sum SHAP values of transformed columns into their original features.

    Returns:
        DataFrame (rows = customers, columns = original features) of log-odds contributions.
    """
    values = pd.DataFrame(sv.values, columns=sv.feature_names)
    groups = [original_feature(c) for c in values.columns]
    return values.T.groupby(groups, sort=False).sum().T


def global_importance(grouped: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature, plus the share of total importance."""
    imp = grouped.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"mean_abs_shap": imp, "share": imp / imp.sum()})


def _display_values(X: pd.DataFrame) -> pd.DataFrame:
    """Raw (human-readable) values for every original feature, engineered ones included."""
    return add_features(X).reset_index(drop=True)


def _format_value(feature: str, value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and feature in UNITS:
        unit = UNITS[feature]
        if unit == " months" and round(float(value)) == 1:
            unit = " month"
        return f"{value:,.0f}{unit}"
    return str(value)


def top_reasons(grouped: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Top ``k`` risk-increasing reasons per customer, in plain language.

    Only features that push churn risk **up** (positive SHAP) count as reasons.

    Args:
        grouped: Output of :func:`group_shap`, aligned row-by-row with ``X``.
        X: The same raw rows used to compute the SHAP values.
        k: Number of reasons per customer.

    Returns:
        One list per customer, e.g. ``["Contract = Month-to-month", "tenure = 2 months"]``.
    """
    shown = _display_values(X)
    reasons: list[list[str]] = []
    for i in range(len(grouped)):
        row = grouped.iloc[i]
        top = row[row > 0].sort_values(ascending=False).head(k)
        reasons.append([f"{f} = {_format_value(f, shown.at[i, f])}" for f in top.index])
    return reasons


def plot_global_importance(importance: pd.DataFrame, top_n: int = 15) -> Figure:
    """Horizontal bar chart of mean |SHAP| per original feature."""
    top = importance.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1.2))
    ax.barh(top.index, top["mean_abs_shap"])
    for y, (v, share) in enumerate(zip(top["mean_abs_shap"], top["share"], strict=True)):
        ax.annotate(f" {share:.0%}", (v, y), va="center", fontsize=8)
    ax.set(xlabel="Mean |SHAP| (log-odds)", title="Global feature importance (SHAP)")
    return fig


def plot_beeswarm(sv: shap.Explanation, max_display: int = 15) -> Figure:
    """SHAP beeswarm on transformed features (direction and spread of each effect)."""
    names = [n.split("__", 1)[-1] for n in sv.feature_names]
    renamed = shap.Explanation(
        values=sv.values, base_values=sv.base_values, data=sv.data, feature_names=names
    )
    shap.plots.beeswarm(renamed, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 0.4 * max_display + 1.5)
    plt.title("SHAP beeswarm (red = high feature value)")
    return fig


def plot_customer_waterfall(
    grouped: pd.DataFrame,
    X: pd.DataFrame,
    base_value: float,
    row: int,
    max_display: int = 10,
    title: str = "",
) -> Figure:
    """Waterfall of one customer's grouped SHAP values, labelled with their raw values."""
    shown = _display_values(X)
    values = grouped.iloc[row]
    labels = [f"{f} = {_format_value(f, shown.at[row, f])}" for f in values.index]
    exp = shap.Explanation(values=values.to_numpy(), base_values=base_value, feature_names=labels)
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 0.45 * max_display + 1.5)
    if title:
        plt.title(title)
    return fig
