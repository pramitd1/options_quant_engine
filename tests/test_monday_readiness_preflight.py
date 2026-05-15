from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.monday_readiness_preflight import (
    PREFLIGHT_BLOCKED,
    PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS,
    build_monday_readiness_preflight_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _guarded_staleness(status: str = "GUARDED_ACCUMULATING_FORWARD_LABELS") -> dict:
    return {
        "report_type": "segmented_probability_guarded_candidate_staleness",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "guarded_staleness_status": status,
        "guarded_staleness_reasons": ["guarded_forward_sample_below_minimum"]
        if status == "GUARDED_ACCUMULATING_FORWARD_LABELS"
        else ["guarded_candidate_age_exceeds_watch_window"],
        "guarded_candidate_summary": {
            "guarded_candidate_generated_at": "2026-05-15T03:30:00+00:00",
            "guarded_candidate_age_days": 0.1,
            "candidate_count": 1,
            "quarantined_candidate_count": 2,
        },
        "dataset_currency": {
            "rows_after_guarded_candidate_generated": 0,
            "quality_labeled_rows_after_guarded_candidate_generated": 0,
        },
        "guarded_routing_policy_stability": {
            "policy_stability_status": "INSUFFICIENT_GUARDED_FORWARD_EVIDENCE",
            "latest_guarded_recommended_routing_policy": None,
        },
    }


def _soak(status: str = "SOAK_ACCUMULATING_TRUE_FORWARD_LABELS") -> dict:
    return {
        "report_type": "segmented_probability_shadow_soak",
        "generated_at": "2026-05-15T04:05:00+00:00",
        "soak_status": status,
        "soak_reasons": ["insufficient_true_forward_sample"],
        "forward_sample_progress": {
            "strict_forward_row_count": 12,
            "forward_sample_gap": 88,
        },
        "guarded_forward_sample_progress": {
            "guarded_strict_forward_row_count": 0,
            "forward_sample_gap": 100,
            "new_post_guarded_true_forward_rows_since_previous_soak": 0,
        },
        "guarded_validation_summary": {
            "guarded_shadow_status": "GUARDED_SHADOW_VALIDATION_PASS",
            "validation_mode_used": "holdout_replay",
        },
        "readiness_summary": {
            "readiness_status": "FORWARD_SHADOW_READINESS_BLOCKED",
        },
        "guarded_candidate_staleness_summary": {
            "guarded_staleness_status": "GUARDED_ACCUMULATING_FORWARD_LABELS",
        },
        "recommended_next_actions": [],
    }


def _old_soak_without_guarded_staleness_context() -> dict:
    payload = _soak("SOAK_CANDIDATE_STALENESS_BLOCKED")
    payload.pop("guarded_candidate_staleness_summary", None)
    return payload


def _dataset(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-05-15T09:20:00+05:30",
                "correct_60m": 1,
                "calibration_label_available": True,
                "label_quality_status": "CLEAN",
            }
        ]
    ).to_csv(path, index=False)
    return path


def _option_chain(path: Path, *, source: str = "ICICI", stale: bool = False) -> Path:
    timestamp = "2026-05-15T03:00:00+00:00" if stale else "2026-05-15T03:59:00+00:00"
    rows = []
    for strike in range(22000, 23050, 50):
        rows.append(
            {
                "strikePrice": strike,
                "OPTION_TYP": "CE",
                "lastPrice": 100,
                "bidPrice": 99,
                "askPrice": 101,
                "impliedVolatility": 18,
                "EXPIRY_DT": "2026-05-21",
                "quoteTimestamp": timestamp,
                "source": source,
            }
        )
        rows.append(
            {
                "strikePrice": strike,
                "OPTION_TYP": "PE",
                "lastPrice": 110,
                "bidPrice": 109,
                "askPrice": 111,
                "impliedVolatility": 19,
                "EXPIRY_DT": "2026-05-21",
                "quoteTimestamp": timestamp,
                "source": source,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_preflight_ready_to_collect_forward_labels_without_fetching_provider(tmp_path: Path):
    guarded_path = _write_json(tmp_path / "guarded.json", _guarded_staleness())
    soak_path = _write_json(tmp_path / "soak.json", _soak())
    dataset_path = _dataset(tmp_path / "signals.csv")

    report = build_monday_readiness_preflight_report(
        source="ICICI",
        symbol="NIFTY",
        dataset_path=dataset_path,
        guarded_staleness_path=guarded_path,
        shadow_soak_path=soak_path,
    )

    assert report["preflight_status"] == PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS
    assert report["provider_fetch_attempted"] is False
    assert report["outcome_refresh_attempted"] is False
    assert report["data_source_policy"]["source_sticky"] is True
    assert report["option_chain_health"]["check_status"] == "NOT_CHECKED"
    assert "shadow_soak" in report["next_commands"]


def test_preflight_blocks_on_stale_option_chain_snapshot(tmp_path: Path):
    guarded_path = _write_json(tmp_path / "guarded.json", _guarded_staleness())
    soak_path = _write_json(tmp_path / "soak.json", _soak())
    dataset_path = _dataset(tmp_path / "signals.csv")
    chain_path = _option_chain(tmp_path / "chain.csv", stale=True)

    report = build_monday_readiness_preflight_report(
        source="ICICI",
        symbol="NIFTY",
        dataset_path=dataset_path,
        guarded_staleness_path=guarded_path,
        shadow_soak_path=soak_path,
        option_chain_path=chain_path,
        spot=22500,
        as_of="2026-05-15T04:00:00+00:00",
        max_quote_age_seconds=300,
    )

    assert report["preflight_status"] == PREFLIGHT_BLOCKED
    assert "option_chain_stale" in report["blockers"]
    assert report["option_chain_health"]["source_matches_selected"] is True
    assert report["data_source_policy"]["fallback_provider_attempted"] is False


def test_preflight_warns_on_source_mismatch_without_overriding(tmp_path: Path):
    guarded_path = _write_json(tmp_path / "guarded.json", _guarded_staleness())
    soak_path = _write_json(tmp_path / "soak.json", _soak())
    dataset_path = _dataset(tmp_path / "signals.csv")
    chain_path = _option_chain(tmp_path / "chain.csv", source="NSE")

    report = build_monday_readiness_preflight_report(
        source="ICICI",
        symbol="NIFTY",
        dataset_path=dataset_path,
        guarded_staleness_path=guarded_path,
        shadow_soak_path=soak_path,
        option_chain_path=chain_path,
        spot=22500,
        as_of="2026-05-15T04:00:00+00:00",
        max_quote_age_seconds=300,
    )

    assert report["data_source_policy"]["selected_source"] == "ICICI"
    assert report["option_chain_health"]["source"] == "NSE"
    assert report["option_chain_health"]["source_matches_selected"] is False
    assert "option_chain_source_differs_from_selected_source" in report["warnings"]
    assert report["data_source_policy"]["source_override_attempted"] is False


def test_preflight_treats_old_source_candidate_staleness_as_context_when_guarded_is_accumulating(tmp_path: Path):
    guarded_path = _write_json(tmp_path / "guarded.json", _guarded_staleness())
    soak_path = _write_json(tmp_path / "soak.json", _old_soak_without_guarded_staleness_context())
    dataset_path = _dataset(tmp_path / "signals.csv")

    report = build_monday_readiness_preflight_report(
        source="ICICI",
        symbol="NIFTY",
        dataset_path=dataset_path,
        guarded_staleness_path=guarded_path,
        shadow_soak_path=soak_path,
    )

    assert report["preflight_status"] == PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS
    assert "shadow_soak_missing_guarded_staleness_context" in report["warnings"]
    assert "source_candidate_staleness_context_only_guarded_bundle_non_blocking" in report["warnings"]
    assert report["blockers"] == []
