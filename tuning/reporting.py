"""
Reporting helpers for experiment ledger inspection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tuning.artifacts import load_jsonl_frame
from tuning.experiments import EXPERIMENT_LEDGER_PATH
from tuning.promotion import PROMOTION_LEDGER_PATH, PROMOTION_STATE_PATH, load_promotion_state
from tuning.shadow import SHADOW_LOG_PATH, summarize_shadow_log


def load_experiment_ledger(path: str | Path = EXPERIMENT_LEDGER_PATH) -> pd.DataFrame:
    return load_jsonl_frame(path)


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    return load_jsonl_frame(path)


def load_promotion_ledger(path: str | Path = PROMOTION_LEDGER_PATH) -> pd.DataFrame:
    return _load_jsonl(path)


def summarize_experiments(path: str | Path = EXPERIMENT_LEDGER_PATH, top_n: int = 5) -> dict:
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
