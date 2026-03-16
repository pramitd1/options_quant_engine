"""
Module: reporting.py

Purpose:
    Summarize tuning experiments, promotion state, and shadow-mode activity for operators and reviewers.

Role in the System:
    Part of the tuning layer that converts experiment and promotion artifacts into lightweight diagnostic summaries.

Key Outputs:
    Experiment leaderboards, promotion workflow snapshots, and shadow-mode summaries.

Downstream Usage:
    Consumed by review tooling, governance workflows, and operator-facing status checks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tuning.artifacts import load_jsonl_frame
from tuning.experiments import EXPERIMENT_LEDGER_PATH
from tuning.promotion import PROMOTION_LEDGER_PATH, PROMOTION_STATE_PATH, load_promotion_state
from tuning.shadow import SHADOW_LOG_PATH, summarize_shadow_log


def load_experiment_ledger(path: str | Path = EXPERIMENT_LEDGER_PATH) -> pd.DataFrame:
    """
    Purpose:
        Load the experiment ledger into a DataFrame for reporting.

    Context:
        Reporting tools consume the persisted JSONL experiment ledger rather than rerunning tuning jobs.

    Inputs:
        path (str | Path): Ledger path to read.

    Returns:
        pd.DataFrame: Experiment ledger rows.

    Notes:
        The helper delegates parsing to the shared JSONL loader so reporting remains consistent with other tuning utilities.
    """
    return load_jsonl_frame(path)


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Purpose:
        Load a JSONL artifact into a DataFrame.

    Context:
        Internal helper used by the reporting module to read experiment and promotion ledgers through one path.

    Inputs:
        path (str | Path): JSONL artifact to load.

    Returns:
        pd.DataFrame: Parsed JSONL rows.

    Notes:
        Kept separate mainly to make the public loaders read cleanly.
    """
    return load_jsonl_frame(path)


def load_promotion_ledger(path: str | Path = PROMOTION_LEDGER_PATH) -> pd.DataFrame:
    """
    Purpose:
        Load the promotion ledger into a DataFrame for workflow summaries.

    Context:
        Promotion reviews rely on the persisted event log rather than transient in-memory state.

    Inputs:
        path (str | Path): Promotion ledger path to read.

    Returns:
        pd.DataFrame: Promotion ledger rows.

    Notes:
        The returned frame can be empty when no promotion events have been recorded yet.
    """
    return _load_jsonl(path)


def summarize_experiments(path: str | Path = EXPERIMENT_LEDGER_PATH, top_n: int = 5) -> dict:
    """
    Purpose:
        Build a compact summary of the experiment ledger.

    Context:
        Operators and reviewers usually need the best packs and broad experiment distribution, not the full raw ledger.

    Inputs:
        path (str | Path): Experiment ledger path to summarize.
        top_n (int): Number of top experiments to include in the leaderboard.

    Returns:
        dict: Summary payload containing leaderboards and validation snapshots.

    Notes:
        The summary is intentionally shallow so it can be displayed quickly in dashboards or CLIs.
    """
    ledger = load_experiment_ledger(path)
    if ledger.empty:
        return {
            "experiment_count": 0,
            "top_packs": [],
            "baseline_vs_candidate": [],
            "validation_summary": [],
        }

    ordered = ledger.sort_values("objective_score", ascending=False, kind="stable")
    top_records = ordered.head(top_n)[["parameter_pack_name", "objective_score", "sample_count", "timestamp"]]
    comparison = (
        ledger.groupby("parameter_pack_name", dropna=False)["objective_score"]
        .agg(["count", "mean", "max"])
        .sort_values("mean", ascending=False, kind="stable")
        .reset_index()
    )
    validation_rows = []
    if "robustness_metrics" in ledger.columns:
        for _, row in ledger.iterrows():
            robustness = row.get("robustness_metrics") or {}
            validation = row.get("validation_results") or {}
            validation_rows.append(
                {
                    "parameter_pack_name": row.get("parameter_pack_name"),
                    "objective_score": row.get("objective_score"),
                    "out_of_sample_score": (validation or {}).get("aggregate_out_of_sample_score"),
                    "robustness_score": (robustness or {}).get("robustness_score"),
                    "timestamp": row.get("timestamp"),
                }
            )
    return {
        "experiment_count": int(len(ledger)),
        "top_packs": top_records.to_dict(orient="records"),
        "baseline_vs_candidate": comparison.to_dict(orient="records"),
        "validation_summary": validation_rows,
    }


def summarize_promotion_workflow(
    *,
    state_path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
    shadow_log_path: str | Path = SHADOW_LOG_PATH,
) -> dict:
    """
    Purpose:
        Summarize the current promotion workflow state and recent events.

    Context:
        This is the high-level status view for the promotion workflow, combining state, event history, and shadow-mode activity.

    Inputs:
        state_path (str | Path): Promotion state file.
        ledger_path (str | Path): Promotion ledger file.
        shadow_log_path (str | Path): Shadow-mode log file.

    Returns:
        dict: Summary of the active packs, recent promotion events, and shadow diagnostics.

    Notes:
        The payload is designed for quick operator review rather than deep forensic analysis.
    """
    state = load_promotion_state(state_path)
    promotion_ledger = load_promotion_ledger(ledger_path)
    latest_events = (
        promotion_ledger.sort_values("timestamp", ascending=False, kind="stable").head(10).to_dict(orient="records")
        if not promotion_ledger.empty
        else []
    )
    return {
        "current_state": {
            "baseline_pack": state.get("baseline"),
            "candidate_pack": state.get("candidate"),
            "shadow_pack": state.get("shadow"),
            "live_pack": state.get("live"),
            "previous_live_pack": state.get("previous_live"),
            "manual_approvals": dict(state.get("manual_approvals") or {}),
        },
        "promotion_event_count": int(len(promotion_ledger)),
        "latest_promotion_events": latest_events,
        "shadow_summary": summarize_shadow_log(shadow_log_path),
    }
