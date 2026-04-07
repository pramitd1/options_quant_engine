"""Backward-compatibility entrypoint for dated watchlist evaluation script.

This module delegates to `scripts/watchlist_realized_evaluation.py`.

Cooldown policy:
- Keep this wrapper only as a temporary compatibility bridge.
- Target removal date: 2026-05-07, if no operational dependency remains.
"""

from scripts.watchlist_realized_evaluation import main


if __name__ == "__main__":
    main()
