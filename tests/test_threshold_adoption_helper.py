from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_adoption_helper import (
    ADOPTION_PLAN_ALREADY_ACTIVE,
    ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK,
    ADOPTION_PLAN_READY,
    ADOPTION_PLAN_SKIPPED,
    build_threshold_adoption_plan,
    write_threshold_adoption_plan,
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


def _ledger_row(value: float, *, action: str = "APPROVED") -> dict:
    return {
        "reviewed_at": "2026-02-01T00:00:00Z",
        "report_json": "promotion.json",
        "promotion_review_status": "PROMOTION_REVIEW_READY",
        "candidate_key": f"composite_signal_score>={value}",
        "threshold_field": "composite_signal_score",
        "threshold_value": value,
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
        "review_action": action,
        "reviewer": "unit-test",
        "review_note": "",
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


def _target_pack(path: Path, *, value: float | None = None) -> Path:
    overrides = {}
    if value is not None:
        overrides["evaluation_thresholds.selection.composite_signal_score_floor"] = value
    path.write_text(
        json.dumps(
            {
                "name": path.stem,
                "version": "1.0.0",
                "description": "Unit test pack.",
                "parent": "baseline_v1",
                "tags": ["candidate"],
                "metadata": {"state": "candidate"},
                "overrides": overrides,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_adoption_plan_dry_run_builds_exact_parameter_pack_patch_without_writing(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))
    target_path = _target_pack(tmp_path / "candidate_v1.json")
    before = target_path.read_text(encoding="utf-8")

    report = build_threshold_adoption_plan(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(75.0),
        target_parameter_pack_path=target_path,
        apply_changes=False,
    )

    assert report["plan_status"] == ADOPTION_PLAN_READY
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["proposed_change"]["config_hint"] == "evaluation_thresholds.selection.composite_signal_score_floor"
    assert report["proposed_change"]["candidate_value"] == 82.0
    assert report["parameter_pack_patch"]["op"] == "add"
    assert "+    \"evaluation_thresholds.selection.composite_signal_score_floor\": 82.0" in report["parameter_pack_diff"]
    assert target_path.read_text(encoding="utf-8") == before


def test_adoption_plan_apply_writes_target_pack_only_when_explicit(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))
    target_path = _target_pack(tmp_path / "candidate_v1.json", value=75.0)

    artifact = write_threshold_adoption_plan(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(75.0),
        target_parameter_pack_path=target_path,
        apply_changes=True,
        output_dir=tmp_path / "reports",
        report_name="unit_threshold_adoption_plan",
    )

    report = artifact["report"]
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert report["plan_status"] == ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK
    assert report["parameter_pack_file_changed"] is True
    assert payload["overrides"]["evaluation_thresholds.selection.composite_signal_score_floor"] == 82.0
    assert payload["metadata"]["threshold_adoption_candidate_value"] == 82.0
    assert Path(artifact["latest_json_path"]).exists()
    assert Path(artifact["latest_markdown_path"]).exists()


def test_adoption_plan_marks_already_active_when_runtime_matches_candidate(tmp_path: Path):
    ledger = _write_ledger(tmp_path / "ledger.csv", _ledger_row(82.0))
    target_path = _target_pack(tmp_path / "candidate_v1.json")

    report = build_threshold_adoption_plan(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=ledger,
        runtime_config=_runtime_config(82.0),
        target_parameter_pack_path=target_path,
    )

    assert report["plan_status"] == ADOPTION_PLAN_ALREADY_ACTIVE
    assert report["parameter_pack_file_changed"] is False
    assert report["adoption_reconciliation"]["adoption_status"] == "ADOPTED_MANUALLY"


def test_adoption_plan_skips_without_approved_ledger(tmp_path: Path):
    target_path = _target_pack(tmp_path / "candidate_v1.json")

    report = build_threshold_adoption_plan(
        promotion_package_report=_promotion_package(82.0),
        ledger_path=tmp_path / "missing.csv",
        runtime_config=_runtime_config(75.0),
        target_parameter_pack_path=target_path,
        apply_changes=True,
    )

    assert report["plan_status"] == ADOPTION_PLAN_SKIPPED
    assert report["parameter_pack_file_changed"] is False
    assert "No APPROVED" in report["adoption_reconciliation"]["adoption_reasons"][0]
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert payload["overrides"] == {}
