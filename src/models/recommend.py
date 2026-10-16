"""Turn a customer's top churn reasons and risk tier into concrete retention actions.

The rules are a plain lookup table so the business can edit them without touching the
model. High-risk customers get a primary action (their strongest reason) and a secondary
action (their strongest reason that maps to a *different* action), because almost every
high-risk customer shares the same #1 reason (month-to-month contract). Medium-risk
customers get a low-cost nudge; low-risk customers get no action.
"""

from __future__ import annotations

# (feature, value or None for "any value") -> action. Order does not matter; the
# customer's reason order (strongest first) decides which rule applies.
ACTION_RULES: dict[tuple[str, str | None], str] = {
    ("Contract", "Month-to-month"): "Offer a discount to move to a 1- or 2-year contract",
    ("tenure", None): "Enroll in the first-90-days onboarding and check-in program",
    ("TotalCharges", None): "Enroll in the first-90-days onboarding and check-in program",
    ("InternetService", "Fiber optic"): "Run a fiber service-quality check and price review",
    ("OnlineSecurity", "No"): "Offer a free 3-month trial of the security + tech support bundle",
    ("TechSupport", "No"): "Offer a free 3-month trial of the security + tech support bundle",
    ("OnlineBackup", "No"): "Offer a free 3-month trial of the security + tech support bundle",
    ("DeviceProtection", "No"): "Offer a free 3-month trial of the security + tech support bundle",
    ("PaymentMethod", "Electronic check"): "Give a one-time credit for switching to auto-pay",
    ("MonthlyCharges", None): "Review the plan and right-size it to the customer's usage",
    ("PaperlessBilling", "Yes"): "Send a clear bill summary and a satisfaction check-in",
}
DEFAULT_ACTION: str = "Personal retention call from the customer success team"
MEDIUM_ACTION: str = "Low-cost email nudge about"
NO_ACTION: str = "No action (monitor)"


def _parse(reason: str) -> tuple[str, str]:
    feature, _, value = reason.partition(" = ")
    return feature.strip(), value.strip()


def matched_actions(reasons: list[str]) -> list[str]:
    """Distinct actions for the given reasons, strongest reason first.

    Args:
        reasons: Plain-language reasons, strongest first, e.g. ``["Contract = Month-to-month"]``.

    Returns:
        Actions in order of first appearance, without duplicates. Reasons with no rule
        are skipped.
    """
    actions: list[str] = []
    for reason in reasons:
        feature, value = _parse(reason)
        for key in ((feature, value), (feature, None)):
            if key in ACTION_RULES:
                if ACTION_RULES[key] not in actions:
                    actions.append(ACTION_RULES[key])
                break
    return actions


def recommend_actions(reasons: list[str], tier: str) -> tuple[str, str]:
    """Primary and secondary action for one customer.

    Args:
        reasons: Plain-language reasons, strongest first.
        tier: ``"High"``, ``"Medium"`` or ``"Low"``.

    Returns:
        ``(primary, secondary)``; ``secondary`` is ``""`` when there is none.
    """
    if tier == "Low" or not reasons:
        return NO_ACTION, ""
    actions = matched_actions(reasons) or [DEFAULT_ACTION]
    if tier == "Medium":
        topic = _parse(reasons[0])[0]
        return f"{MEDIUM_ACTION} {topic}", ""
    return actions[0], actions[1] if len(actions) > 1 else ""
