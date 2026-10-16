"""Download the raw IBM Telco churn CSV into data/raw (skipped if already present).

Usage: ``python -m src.data.download``
"""

from __future__ import annotations

import hashlib
import urllib.request

from src.config import RAW_DATA_FILE
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SOURCE_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256: str = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def download() -> None:
    """Fetch the dataset and verify its checksum.

    Raises:
        ValueError: If the downloaded file does not match :data:`EXPECTED_SHA256`.
    """
    if RAW_DATA_FILE.exists():
        logger.info("Raw data already present at %s", RAW_DATA_FILE)
        return
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Checksum mismatch: got {digest}")
    RAW_DATA_FILE.write_bytes(content)
    logger.info("Downloaded %d bytes to %s (checksum OK)", len(content), RAW_DATA_FILE)


if __name__ == "__main__":
    download()
