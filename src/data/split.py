"""Train/val/test split (70/15/15) with leakage checks.

Choice of split:
    * If ``config.DATE_COL`` is set, split by time (oldest -> train, newest -> test), because
      a model is always used on customers *after* the ones it was trained on.
    * Otherwise (Telco has no date column) use a stratified random split so each set keeps
      the 26.5% churn rate; with ~1,060 test customers, an unstratified split could drift
      a few points and distort PR-AUC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATE_COL,
    ID_COL,
    METRICS_DIR,
    PROCESSED_DIR,
    SEED,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_CHURN_RATE_GAP: float = 0.02  # max allowed |split rate - overall rate|


@dataclass(frozen=True)
class Splits:
    """Container for the three data splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        """Return ``(name, frame)`` pairs in train/val/test order."""
        return [("train", self.train), ("val", self.val), ("test", self.test)]


def split_data(df: pd.DataFrame, date_col: str | None = DATE_COL, seed: int = SEED) -> Splits:
    """Split cleaned data into train/val/test using config ratios.

    Args:
        df: Cleaned DataFrame with an integer target column.
        date_col: If given, split chronologically on this column; otherwise stratify.
        seed: Random seed for the stratified split.

    Returns:
        A :class:`Splits` instance.
    """
    if date_col is not None:
        ordered = df.sort_values(date_col).reset_index(drop=True)
        n_train = int(len(ordered) * TRAIN_SIZE)
        n_val = int(len(ordered) * VAL_SIZE)
        splits = Splits(
            ordered.iloc[:n_train],
            ordered.iloc[n_train : n_train + n_val],
            ordered.iloc[n_train + n_val :],
        )
        logger.info("Time-based split on %s", date_col)
    else:
        train_val, test = train_test_split(
            df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=seed
        )
        val_share = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        train, val = train_test_split(
            train_val, test_size=val_share, stratify=train_val[TARGET_COL], random_state=seed
        )
        splits = Splits(train, val, test)
        logger.info("Stratified split on %s (seed=%d)", TARGET_COL, seed)

    check_leakage(df, splits)
    return Splits(*(s.reset_index(drop=True) for _, s in splits.items()))


def check_leakage(df: pd.DataFrame, splits: Splits) -> None:
    """Fail loudly if the split leaks rows or distorts the target.

    Checks:
        * No customer ID appears in more than one split.
        * Splits cover every row exactly once.
        * Each split's churn rate is within ``MAX_CHURN_RATE_GAP`` of the overall rate.
        * No column besides the target is a perfect copy of the target.

    Args:
        df: The full cleaned DataFrame.
        splits: The proposed splits.

    Raises:
        ValueError: If any check fails.
    """
    ids = {name: set(s[ID_COL]) for name, s in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = ids[a] & ids[b]
        if overlap:
            raise ValueError(f"{len(overlap)} customer IDs appear in both {a} and {b}")

    if sum(len(s) for _, s in splits.items()) != len(df):
        raise ValueError("Splits do not cover every row exactly once")

    overall = df[TARGET_COL].mean()
    for name, s in splits.items():
        gap = abs(s[TARGET_COL].mean() - overall)
        if gap > MAX_CHURN_RATE_GAP:
            raise ValueError(f"{name} churn rate differs from overall by {gap:.3f}")

    for col in df.columns.drop(TARGET_COL):
        if df[col].nunique() == 2 and (
            pd.factorize(df[col])[0] == df[TARGET_COL].to_numpy()
        ).mean() in (0.0, 1.0):
            raise ValueError(f"Column {col!r} is a copy of the target (leakage)")

    logger.info("Leakage checks passed")


def save_splits(splits: Splits, out_dir: Path = PROCESSED_DIR) -> dict[str, dict[str, float]]:
    """Write splits to ``data/processed`` and a summary JSON to ``reports/metrics``.

    Args:
        splits: The splits to save.
        out_dir: Output directory for the CSVs.

    Returns:
        Summary dict of size and churn rate per split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    total = sum(len(s) for _, s in splits.items())
    for name, s in splits.items():
        s.to_csv(out_dir / f"{name}.csv", index=False)
        summary[name] = {
            "rows": len(s),
            "share": round(len(s) / total, 4),
            "churn_rate": round(float(s[TARGET_COL].mean()), 4),
        }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved splits to %s: %s", out_dir, summary)
    return summary


def load_splits(in_dir: Path = PROCESSED_DIR) -> Splits:
    """Load previously saved splits from ``data/processed``.

    Args:
        in_dir: Directory containing ``train.csv``, ``val.csv``, ``test.csv``.

    Returns:
        A :class:`Splits` instance.
    """
    frames = [
        pd.read_csv(in_dir / f"{n}.csv", dtype={ID_COL: str}) for n in ("train", "val", "test")
    ]
    return Splits(*frames)
