from __future__ import annotations

import os
import warnings

# Allow the known macOS urllib3 LibreSSL noise in all environments.
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL.*",
    category=Warning,
)

# In CI, fail fast on risky warnings while preserving the known urllib3 noise allowlist.
if os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}:
    warnings.filterwarnings("error")
    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3 v2 only supports OpenSSL.*",
        category=Warning,
    )

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)

if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)
