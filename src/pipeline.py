"""End-to-end pipeline entry point: ``python -m src.pipeline``.

Each phase adds its step here, so the whole project always runs with one command.
"""

from __future__ import annotations

import json

from src.config import METRICS_DIR
from src.data.clean import clean, save_clean
from src.data.load import load_raw
from src.data.split import save_splits, split_data
from src.features.pipeline import build_preprocessor, split_xy
from src.models.select import run_modeling
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run() -> None:
    """Run all implemented pipeline stages in order."""
    set_seed()

    logger.info("Stage 1/4: load + clean")
    df = clean(load_raw())
    save_clean(df)

    logger.info("Stage 2/4: split")
    splits = split_data(df)
    save_splits(splits)

    logger.info("Stage 3/4: features (fit on train only)")
    X_train, _ = split_xy(splits.train)
    X_val, _ = split_xy(splits.val)
    pre = build_preprocessor().fit(X_train)
    Xt_train, Xt_val = pre.transform(X_train), pre.transform(X_val)
    if Xt_train.isna().any().any() or Xt_val.isna().any().any():
        raise ValueError("Preprocessed features contain NaN")
    features = list(Xt_train.columns)
    (METRICS_DIR / "feature_list.json").write_text(json.dumps(features, indent=2))
    logger.info("Built %d features: train %s, val %s", len(features), Xt_train.shape, Xt_val.shape)

    logger.info("Stage 4/4: modeling (CV, tuning, val selection, threshold, test once)")
    run_modeling(splits)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    run()
