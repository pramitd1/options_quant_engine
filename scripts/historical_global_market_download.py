#!/usr/bin/env python3
"""Compatibility wrapper for historical global market download."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep.historical_global_market_download import main


if __name__ == "__main__":
    main()
