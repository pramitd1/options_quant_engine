from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_adoption_reconciliation import (
    ADOPTED_MANUALLY,
    ADOPTION_MISMATCH,
    APPROVED_BUT_NOT_ADOPTED,
    ROLLED_BACK_MANUALLY,
    UNKNOWN_ADOPTION_STATE,
    build_threshold_adoption_reconciliation_report,
    write_threshold_adoption_reconciliation_report,
)


def _promotion_package(value: float = 82.0) -> dict:
    return {
        "report_type": "threshold_promotion_review",
        "promotion_review_status": "PROMOTION_REVIEW_READY",
        "runtime_config_changed": False,
        "promotion_candidate": {
            "source_candidate_key": f"composite_signal_score>={value}",
            "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
            "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": value},
            "overrides": {"evaluation_thresholds.selection.composite_signal_score_floor": value},
            "runtime_config_changed": False,
        },
    }


def _ledger_row(value: float, *, action: str = "APPROVED", reviewed_at: str = "2026-02-01T00:00:00Z", note: str = "") -> dict:
    return {
        "reviewed_at": reviewed_at,
        "report_json": "promotion.json",
        "promotion_review_status": "PROMOTION_REVIEW_READY",
        "candidate_key": f"composite_signal_score>={value}",
        "threshold_field": "composite_signal_score",
        "threshold_value": value,
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
        "review_action": action,
        "reviewer": "unit-test",
        "review_note": note,
        "runtime_config_changed": False,
    }


def _write_ledger(path: Path, *rows: dict) -> Path:
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def _runtime_config(value: float) -> dict:
    return {
        "evaluation_thresholds": {
            "selection": {
                "composite_signal_score_floor": value,
            },
        },
    }


def test_adoption_reconciliation_marks_adopted_when_runtime_matches_approval(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))

    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(82.0),
    )

    assert report["adoption_status"] == ADOPTED_MANUALLY
    assert report["runtime_config_changed"] is False
    assert report["comparison"]["matches_candidate"] is True
    assert report["comparison"]["observed_runtime_value"] == 82.0


def test_adoption_reconciliation_marks_approved_but_not_adopted_when_default_remains(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))

    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(75.0),
    )

    assert report["adoption_status"] == APPROVED_BUT_NOT_ADOPTED
    assert report["comparison"]["matches_default"] is True
    assert "code default" in report["adoption_reasons"][0]


def test_adoption_reconciliation_marks_mismatch_when_runtime_uses_other_value(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))

    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(79.0),
    )

    assert report["adoption_status"] == ADOPTION_MISMATCH
    assert report["comparison"]["matches_candidate"] is False
    assert report["comparison"]["matches_default"] is False


def test_adoption_reconciliation_marks_manual_rollback_from_later_ledger_note(tmp_path: Path):
    ledger = _write_ledger(
        tmp_path / "ledger.csv",
        _ledger_row(82.0),
        _ledger_row(
            82.0,
            action="REJECTED",
            reviewed_at="2026-02-02T00:00:00Z",
            note="Rolled back after manual post-promotion review.",
        ),
    )

    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(75.0),
    )

    assert report["adoption_status"] == ROLLED_BACK_MANUALLY
    assert report["rollback_decision"]["review_action"] == "REJECTED"
    assert report["runtime_config_changed"] is False


def test_adoption_reconciliation_unknown_without_approved_decision(tmp_path: Path):
    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=tmp_path / "missing.csv",
        runtime_config=_runtime_config(82.0),
    )

    assert report["adoption_status"] == UNKNOWN_ADOPTION_STATE
    assert "No APPROVED" in report["adoption_reasons"][0]


def test_adoption_reconciliation_writer_outputs_latest_artifacts(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))
    artifact = write_threshold_adoption_reconciliation_report(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(82.0),
        output_dir=tmp_path,
        report_name="unit_threshold_adoption_reconciliation",
    )

    for key in [
        "json_path",
        "markdown_path",
        "comparison_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_comparison_csv_path",
    ]:
        assert Path(artifact[key]).exists()

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["adoption_status"] == ADOPTED_MANUALLY
