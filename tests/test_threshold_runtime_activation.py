from __future__ import annotations

import json
from pathlib import Path

import app.runtime_sinks as runtime_sinks
from app.runtime_sinks import DefaultSignalCaptureSink
from research.signal_evaluation.evaluator import build_signal_evaluation_row
from research.signal_evaluation.threshold_runtime_activation import (
    build_runtime_activation_capture_guard,
    build_threshold_runtime_activation_marker,
    load_threshold_runtime_activation_marker,
    write_threshold_runtime_activation_marker,
)


def test_runtime_activation_marker_normalizes_timestamp_and_stays_signal_only():
    marker = build_threshold_runtime_activation_marker(
        candidate_pack_name="candidate_v1",
        activated_at="2026-05-15T09:40:00+05:30",
        threshold_value=85.0,
    )

    assert marker["candidate_pack_name"] == "candidate_v1"
    assert marker["activated_at"] == "2026-05-15T04:10:00+00:00"
    assert marker["runtime_config_changed"] is False
    assert marker["execution_behavior_changed"] is False


def test_runtime_activation_writer_outputs_latest_artifacts(tmp_path: Path):
    artifact = write_threshold_runtime_activation_marker(
        candidate_pack_name="candidate_v1",
        activated_at="2026-05-15T09:40:00+05:30",
        activation_note="unit test activation",
        threshold_value=85.0,
        output_dir=tmp_path,
    )

    json_path = Path(artifact["activation_marker_json_path"])
    markdown_path = Path(artifact["activation_marker_markdown_path"])
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["activated_at"] == "2026-05-15T04:10:00+00:00"
    assert "unit test activation" in markdown_path.read_text(encoding="utf-8")


def _activation_marker() -> dict:
    return build_threshold_runtime_activation_marker(
        candidate_pack_name="candidate_v1",
        activated_at="2026-05-15T09:40:00+05:30",
        threshold_value=85.0,
    )


def _capture_payload(
    *,
    mode: str = "LIVE",
    parameter_pack_name: str = "candidate_v1",
    timestamp: str = "2026-05-15T09:45:00+05:30",
    trade_status: str = "WATCHLIST",
) -> dict:
    return {
        "mode": mode,
        "source": "ICICI",
        "symbol": "NIFTY",
        "authoritative_parameter_pack": parameter_pack_name,
        "spot_summary": {
            "timestamp": timestamp,
            "spot": 25100.0,
        },
        "trade": {
            "parameter_pack_name": parameter_pack_name,
            "trade_status": trade_status,
            "valuation_time": timestamp,
        },
    }


def test_runtime_activation_capture_guard_blocks_wrong_pack_live_rows_after_activation():
    guard = build_runtime_activation_capture_guard(
        _capture_payload(parameter_pack_name="baseline_v1"),
        marker=_activation_marker(),
    )

    assert guard["guard_active"] is True
    assert guard["capture_allowed"] is False
    assert guard["status"] == "PARAMETER_PACK_MISMATCH"
    assert guard["expected_parameter_pack"] == "candidate_v1"
    assert guard["observed_parameter_pack"] == "baseline_v1"
    assert guard["runtime_config_changed"] is False
    assert guard["execution_behavior_changed"] is False


def test_runtime_activation_capture_guard_allows_candidate_live_rows_after_activation():
    guard = build_runtime_activation_capture_guard(
        _capture_payload(parameter_pack_name="candidate_v1"),
        marker=_activation_marker(),
    )

    assert guard["capture_allowed"] is True
    assert guard["status"] == "PARAMETER_PACK_MATCH"


def test_runtime_activation_capture_guard_allows_non_live_and_pre_activation_rows():
    replay_guard = build_runtime_activation_capture_guard(
        _capture_payload(mode="REPLAY", parameter_pack_name="baseline_v1"),
        marker=_activation_marker(),
    )
    pre_activation_guard = build_runtime_activation_capture_guard(
        _capture_payload(
            parameter_pack_name="baseline_v1",
            timestamp="2026-05-15T09:35:00+05:30",
        ),
        marker=_activation_marker(),
    )

    assert replay_guard["capture_allowed"] is True
    assert replay_guard["status"] == "NON_LIVE_MODE"
    assert pre_activation_guard["capture_allowed"] is True
    assert pre_activation_guard["status"] == "PRE_ACTIVATION_SIGNAL"


def test_runtime_activation_loader_returns_empty_dict_when_missing(tmp_path: Path):
    marker = load_threshold_runtime_activation_marker(tmp_path / "missing_marker.json")

    assert marker == {}


def test_default_signal_capture_sink_persists_guarded_wrong_pack_after_runtime_activation(monkeypatch):
    saved = []
    result_payload = _capture_payload(parameter_pack_name="baseline_v1")

    monkeypatch.setattr(
        runtime_sinks,
        "build_runtime_activation_capture_guard",
        lambda _payload: {
            "guard_active": True,
            "capture_allowed": False,
            "status": "PARAMETER_PACK_MISMATCH",
            "expected_parameter_pack": "candidate_v1",
            "observed_parameter_pack": "baseline_v1",
        },
    )
    monkeypatch.setattr(
        runtime_sinks,
        "save_signal_evaluation",
        lambda *args, **kwargs: saved.append(args),
    )

    DefaultSignalCaptureSink().apply(
        result_payload=result_payload,
        trade=result_payload["trade"],
        capture_signal_evaluation=True,
        signal_capture_policy="ALL_SIGNALS",
    )

    assert len(saved) == 1
    assert result_payload["signal_capture_status"] == "CAPTURED_GUARDED:PARAMETER_PACK_MISMATCH"
    assert result_payload["signal_capture_guarded"] is True
    assert result_payload["signal_capture_guard_reason"] == "PARAMETER_PACK_MISMATCH"
    assert result_payload["runtime_activation_capture_guard"]["observed_parameter_pack"] == "baseline_v1"


def test_default_signal_capture_sink_persists_allowed_rows_after_runtime_activation(monkeypatch):
    saved = []
    result_payload = _capture_payload(parameter_pack_name="candidate_v1")

    monkeypatch.setattr(
        runtime_sinks,
        "build_runtime_activation_capture_guard",
        lambda _payload: {
            "guard_active": True,
            "capture_allowed": True,
            "status": "PARAMETER_PACK_MATCH",
            "expected_parameter_pack": "candidate_v1",
            "observed_parameter_pack": "candidate_v1",
        },
    )
    monkeypatch.setattr(
        runtime_sinks,
        "save_signal_evaluation",
        lambda *args, **kwargs: saved.append(args),
    )

    DefaultSignalCaptureSink().apply(
        result_payload=result_payload,
        trade=result_payload["trade"],
        capture_signal_evaluation=True,
        signal_capture_policy="ALL_SIGNALS",
    )

    assert len(saved) == 1
    assert result_payload["signal_capture_status"] == "CAPTURED"
    assert result_payload["runtime_activation_capture_guard"]["status"] == "PARAMETER_PACK_MATCH"


def test_signal_evaluation_row_records_runtime_activation_guard_metadata():
    result_payload = _capture_payload(parameter_pack_name="baseline_v1")
    guard = build_runtime_activation_capture_guard(result_payload, marker=_activation_marker())
    result_payload["runtime_activation_capture_guard"] = guard
    result_payload["signal_capture_guarded"] = True
    result_payload["signal_capture_guard_reason"] = guard["status"]

    row = build_signal_evaluation_row(result_payload)

    assert row["signal_capture_guarded"] is True
    assert row["signal_capture_guard_reason"] == "PARAMETER_PACK_MISMATCH"
    assert row["runtime_activation_guard_active"] is True
    assert row["runtime_activation_capture_allowed"] is False
    assert row["runtime_activation_guard_status"] == "PARAMETER_PACK_MISMATCH"
    assert row["runtime_activation_expected_parameter_pack"] == "candidate_v1"
    assert row["runtime_activation_observed_parameter_pack"] == "baseline_v1"
