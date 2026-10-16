"""Central configuration: paths, seeds, split ratios, and column names.

Every other module imports from here so nothing hardcodes a path or a seed.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility --------------------------------------------------------
SEED: int = 42

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

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
PRIMARY_METRIC: str = "average_precision"  # sklearn scorer name for PR-AUC
TOP_DECILE: float = 0.10  # for "recall in top 10% risk" and lift

# --- Business costs (placeholders until provided) ---------------------------
RETENTION_OFFER_COST: float = 50.0  # cost of contacting/offering one customer
RETAINED_CUSTOMER_VALUE: float = 500.0  # value of keeping a would-be churner

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
