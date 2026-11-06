"""Model card: a record of what the model is, what it was trained on, and its known limits.

Built from the artifacts the pipeline already writes (``reports/metrics/*.json``), the raw
data file, and the environment. It adds nothing new to the model; it makes the model
traceable. Two files are written:

* ``models/model_card.json``: machine-readable, saved next to the model artifact (gitignored
  with the rest of ``models/``), so a deployed model carries its own provenance.
* ``reports/model_card.md``: the same content for people, small enough to review in a PR.

Run with ``python -m src.models.model_card`` after the pipeline has run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import (
    CV_FOLDS,
    CV_REPEATS,
    METRICS_DIR,
    MODELS_DIR,
    OFFER_SUCCESS_RATE,
    PROJECT_ROOT,
    RAW_DATA_FILE,
    REPORTS_DIR,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_OFFER_COST,
    SEED,
    TARGET_COL,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.data.download import EXPECTED_SHA256
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "shap",
    "scipy",
)

LIMITATIONS: tuple[str, ...] = (
    "Snapshot data: tenure and TotalCharges are measured at exit for churned customers, so "
    "reported performance is likely an upper estimate (reports/final_report.md, section 6).",
    "The gain over the logistic regression baseline is small: repeated CV puts it near +0.011 "
    "PR-AUC with an interval that includes zero.",
    "Offer cost, customer value and success rate are placeholders in src/config.py; the "
    "threshold and profit figures depend on them.",
    "The model predicts who will churn, not who an offer will save. Test offers against a "
    "randomly held-out control group before trusting the profit estimates.",
    "No behavioral signals (usage, support tickets, payment failures) are in the data.",
)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Current git commit and whether the working tree has uncommitted changes.

    Args:
        root: Repository root.

    Returns:
        ``{"commit": str | None, "dirty": bool | None}``; both ``None`` outside a repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict[str, str | None]:
    """Installed version of each library, or ``None`` if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON metrics file, with a clear error if the pipeline has not produced it."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.pipeline` first.")
    return json.loads(path.read_text())


def build_model_card(
    metrics_dir: Path = METRICS_DIR,
    raw_file: Path = RAW_DATA_FILE,
    model_file: Path | None = None,
) -> dict[str, Any]:
    """Assemble the model card from saved metrics, the raw data file and the environment.

    Args:
        metrics_dir: Directory holding the pipeline's JSON outputs.
        raw_file: Raw data file to fingerprint.
        model_file: Trained model artifact; its modification time is recorded when present.
            Defaults to ``models/final_model.joblib``.

    Returns:
        JSON-ready dict.

    Raises:
        FileNotFoundError: If the pipeline outputs or the raw data file are missing.
    """
    model_file = model_file if model_file is not None else MODELS_DIR / "final_model.joblib"
    summary = _read_json(metrics_dir / "model_summary.json")
    splits = _read_json(metrics_dir / "split_summary.json")
    business = _read_json(metrics_dir / "business_summary.json")
    tuning = _read_json(metrics_dir / "tuning_results.json")
    if not raw_file.exists():
        raise FileNotFoundError(f"{raw_file} not found. Run `python -m src.data.download`.")
    repeated_path = metrics_dir / "repeated_cv_summary.json"
    repeated = _read_json(repeated_path) if repeated_path.exists() else None

    name = summary["final_model"]
    sha = file_sha256(raw_file)
    trained_at = (
        datetime.fromtimestamp(model_file.stat().st_mtime, UTC).isoformat(timespec="seconds")
        if model_file.exists()
        else None
    )
    card: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_trained_at": trained_at,
        "code": {
            **git_state(),
            "python": platform.python_version(),
            "libraries": library_versions(),
        },
        "data": {
            "file": raw_file.name,
            "sha256": sha,
            "matches_expected_source": sha == EXPECTED_SHA256,
            "target": f"{TARGET_COL} = Yes (left within the last month)",
            "split": (
                f"stratified {TRAIN_SIZE:.0%} train / {VAL_SIZE:.0%} val / "
                f"{1 - TRAIN_SIZE - VAL_SIZE:.0%} test (no date column), seed {SEED}"
            ),
            "splits": splits,
        },
        "model": {
            "name": name,
            "calibration": business["calibration"]["method"],
            "params": summary["final_params"],
            "tuning_cv_pr_auc": tuning[name]["cv_pr_auc_mean"],
            "threshold_raw": summary["threshold"],
            "threshold_calibrated": business["threshold_calibrated"],
        },
        "performance": {
            "val": summary["val"],
            "test": summary["test"],
            "test_pr_auc_diff_bootstrap": summary["test_pr_auc_diff_bootstrap"],
            "repeated_cv": _repeated_cv_view(repeated),
        },
        "business_assumptions": {
            "offer_cost": RETENTION_OFFER_COST,
            "customer_value": RETAINED_CUSTOMER_VALUE,
            "offer_success_rate": OFFER_SUCCESS_RATE,
            "placeholders": True,
        },
        "limitations": list(LIMITATIONS),
    }
    return card


def _repeated_cv_view(repeated: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact view of the repeated-CV result (or ``None`` if it has not been run)."""
    if repeated is None:
        return None
    gain = repeated["paired"]["final_minus_baseline"]["pr_auc"]
    return {
        "scheme": f"{CV_REPEATS}x{CV_FOLDS} stratified K-fold on train + validation",
        "pr_auc": {k: v["pr_auc"]["mean"] for k, v in repeated["per_model"].items()},
        "final_minus_baseline_pr_auc": {
            k: gain[k] for k in ("mean_diff", "ci_low", "ci_high", "p_value")
        },
    }


def render_markdown(card: dict[str, Any]) -> str:
    """Human-readable version of the model card.

    Args:
        card: Output of :func:`build_model_card`.

    Returns:
        Markdown text.
    """
    d, m, p, c = card["data"], card["model"], card["performance"], card["code"]
    test, base = p["test"]["final"], p["test"]["baseline"]
    lines = [
        "# Model card",
        "",
        "Generated by `python -m src.models.model_card` from the pipeline's saved outputs.",
        "",
        "## Model",
        f"- **Algorithm:** {m['name']}, {m['calibration']}",
        f"- **Decision threshold:** {m['threshold_calibrated']:.3f} calibrated "
        f"({m['threshold_raw']:.2f} raw), chosen to maximize expected profit on validation",
        f"- **Trained:** {card['model_trained_at'] or 'unknown (no model file found)'}",
        f"- **Hyperparameters:** `{json.dumps(m['params'], sort_keys=True)}`",
        "",
        "## Data",
        f"- **File:** `{d['file']}`, SHA-256 `{d['sha256']}` "
        f"({'matches' if d['matches_expected_source'] else 'DOES NOT match'} the expected source)",
        f"- **Target:** {d['target']}",
        f"- **Split:** {d['split']}",
        "",
        "| Split | Rows | Churn rate |",
        "|---|---|---|",
        *(f"| {name} | {s['rows']:,} | {s['churn_rate']:.1%} |" for name, s in d["splits"].items()),
        "",
        "## Performance",
        "",
        "| Metric (test, scored once) | Final | Baseline |",
        "|---|---|---|",
        f"| PR-AUC | {test['pr_auc']:.3f} | {base['pr_auc']:.3f} |",
        f"| ROC-AUC | {test['roc_auc']:.3f} | {base['roc_auc']:.3f} |",
        f"| Recall in top 10% | {test['recall_top10']:.1%} | {base['recall_top10']:.1%} |",
        f"| Lift in top 10% | {test['lift_top10']:.2f}x | {base['lift_top10']:.2f}x |",
        "",
    ]
    if p["repeated_cv"]:
        g = p["repeated_cv"]["final_minus_baseline_pr_auc"]
        lines += [
            f"**Repeated CV** ({p['repeated_cv']['scheme']}): PR-AUC gain over the baseline "
            f"{g['mean_diff']:+.3f} (95% CI [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}], "
            f"p = {g['p_value']:.2f}).",
            "",
        ]
    b = card["business_assumptions"]
    lines += [
        "## Business assumptions (placeholders)",
        f"- Offer cost {b['offer_cost']:.0f}, customer value {b['customer_value']:.0f}, "
        f"offer success rate {b['offer_success_rate']:.0%}",
        "",
        "## Known limitations",
        *(f"- {item}" for item in card["limitations"]),
        "",
        "## Provenance",
        f"- **Commit:** `{c['commit']}`{' (uncommitted changes present)' if c['dirty'] else ''}",
        f"- **Python:** {c['python']}",
        "- **Libraries:** "
        + ", ".join(f"{k} {v}" for k, v in c["libraries"].items() if v is not None),
        f"- **Generated:** {card['generated_at']}",
        "",
    ]
    return "\n".join(lines)


def save_model_card(card: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``models/model_card.json`` and ``reports/model_card.md``.

    Args:
        card: Output of :func:`build_model_card`.

    Returns:
        ``(json_path, markdown_path)``.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MODELS_DIR / "model_card.json"
    md_path = REPORTS_DIR / "model_card.md"
    json_path.write_text(json.dumps(card, indent=2))
    md_path.write_text(render_markdown(card))
    logger.info("Saved model card to %s and %s", json_path, md_path)
    return json_path, md_path


def run_model_card() -> dict[str, Any]:
    """Build and save the model card from the current pipeline outputs."""
    card = build_model_card()
    save_model_card(card)
    return card


if __name__ == "__main__":
    run_model_card()
