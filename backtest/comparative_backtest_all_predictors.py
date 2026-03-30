#!/usr/bin/env python3
"""Compatibility wrapper for comparative backtest runner.

The implementation lives in scripts/backtest/comparative_backtest_all_predictors.py.
Keep this wrapper so existing automation that calls the root-level script
continues to work.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest.comparative_backtest_all_predictors import main


if __name__ == "__main__":
    raise SystemExit(main())
