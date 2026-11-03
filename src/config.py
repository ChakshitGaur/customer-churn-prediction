"""Central configuration: paths, seeds, split ratios, and column names.

Every other module imports from here so nothing hardcodes a path or a seed.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Reproducibility --------------------------------------------------------
SEED: int = 42

# --- Paths ------------------------------------------------------------------
# CHURN_ROOT redirects every data, model and report path (used by the pipeline smoke test and
# useful for containers). Unset, the root is the repository.
PROJECT_ROOT: Path = Path(os.environ.get("CHURN_ROOT") or Path(__file__).resolve().parents[1])

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"  # read-only: never write here
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# Filled in during Phase 1 once the dataset and churn definition are confirmed.
RAW_DATA_FILE: Path = RAW_DIR / "telco_churn.csv"

# --- Data -------------------------------------------------------------------
TARGET_COL: str = "Churn"  # placeholder until confirmed in Phase 1
ID_COL: str = "customerID"  # placeholder until confirmed in Phase 1
DATE_COL: str | None = None  # set if a usable date column exists

# --- Splitting (train / val / test) -----------------------------------------
TRAIN_SIZE: float = 0.70
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15

# --- Modeling ---------------------------------------------------------------
CV_FOLDS: int = 5
# Configurations sampled per model in hyperparameter tuning. CHURN_TUNE_ITER lowers it for tests.
TUNE_ITER: int = int(os.environ.get("CHURN_TUNE_ITER", "40"))
CV_REPEATS: int = 5  # repeats for repeated stratified K-fold (src/models/repeated_cv.py)
ABLATION_REPEATS: int = 3  # repeats per scenario in src/models/ablation.py
PRIMARY_METRIC: str = "average_precision"  # sklearn scorer name for PR-AUC
TOP_DECILE: float = 0.10  # for "recall in top 10% risk" and lift

# --- Business costs (placeholders until provided) ---------------------------
RETENTION_OFFER_COST: float = 50.0  # cost of contacting/offering one customer
RETAINED_CUSTOMER_VALUE: float = 500.0  # value of keeping a would-be churner
OFFER_SUCCESS_RATE: float = 0.30  # share of contacted would-be churners who then stay
# Break-even probability for a perfectly calibrated model:
#   cost / (value * success) = 50 / (500 * 0.30) = 0.33

MODEL_MAX_AGE_DAYS: int = 90  # dashboard shows a note when the model is older than this
THRESHOLD_BOOTSTRAP_N: int = 1000  # resamples in src/models/threshold_bootstrap.py

# Grid for the profit sensitivity analysis (src/models/sensitivity.py).
SENSITIVITY_COSTS: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
SENSITIVITY_VALUES: tuple[float, ...] = (250.0, 500.0, 750.0, 1000.0)
SENSITIVITY_SUCCESS_RATES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
