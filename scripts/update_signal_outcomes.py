#!/usr/bin/env python3
"""
Module: update_signal_outcomes.py

Purpose:
    Implement the update signal outcomes script used for repeatable operational or research tasks.

Role in the System:
    Part of the operational scripting layer that supports repeatable maintenance and research tasks.

Key Outputs:
    CLI side effects, maintenance artifacts, and repeatable batch jobs.

Downstream Usage:
    Consumed by operators and by repeatable development or research workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path(Path(__file__))

from research.signal_evaluation import (
    SIGNAL_DATASET_PATH,
    resolve_research_as_of,
    update_signal_dataset_outcomes,
)


def parse_args():
    """
    Purpose:
        Parse command-line arguments for the current entry point.

    Context:
        Function inside the `update signal outcomes` module. The module sits in the operations layer that exposes reporting, maintenance, and governance entry points.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Parsed argument namespace.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    parser = argparse.ArgumentParser(
        description="Refresh pending realized outcomes in the canonical signal evaluation dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Path to the canonical signals dataset CSV.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional timestamp cutoff for outcome enrichment, e.g. 2026-03-14T15:25:00+05:30",
    )
    return parser.parse_args()


def main():
    """
    Purpose:
        Run the module entry point for command-line or operational execution.

    Context:
        Function inside the `update signal outcomes` module. The module sits in the operations layer that exposes reporting, maintenance, and governance entry points.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Exit status or workflow result returned by the implementation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    args = parse_args()
    resolved_as_of = resolve_research_as_of(args.as_of)
    frame = update_signal_dataset_outcomes(
        dataset_path=args.dataset_path,
        as_of=resolved_as_of,
    )

    pending = int((frame.get("outcome_status") == "PENDING").sum()) if not frame.empty else 0
    partial = int((frame.get("outcome_status") == "PARTIAL").sum()) if not frame.empty else 0
    complete = int((frame.get("outcome_status") == "COMPLETE").sum()) if not frame.empty else 0

    print(f"dataset_path: {args.dataset_path}")
    print(f"as_of: {resolved_as_of.isoformat()}")
    print(f"rows: {len(frame)}")
    print(f"pending: {pending}")
    print(f"partial: {partial}")
    print(f"complete: {complete}")


if __name__ == "__main__":
    main()
