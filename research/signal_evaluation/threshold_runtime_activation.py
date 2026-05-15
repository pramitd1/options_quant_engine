"""Runtime activation marker for manually selected threshold candidate packs.

This marker records when an operator deliberately starts running signal
generation under an approved candidate parameter pack. It is an ops artifact
only: writing it does not modify parameter packs, runtime config, or execution
behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_runtime_activation"
)
THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME = "latest_threshold_runtime_activation.json"
THRESHOLD_RUNTIME_ACTIVATION_MARKDOWN_FILENAME = "latest_threshold_runtime_activation.md"


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _coerce_timestamp(value: Any) -> str:
    if value in (None, ""):
        return _utc_now()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid activation timestamp: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.isoformat()


def _safe_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def build_threshold_runtime_activation_marker(
    *,
    candidate_pack_name: str = "candidate_v1",
    activated_at: Any = None,
    activated_by: str = "operator",
    activation_note: str | None = None,
    config_hint: str = "evaluation_thresholds.selection.composite_signal_score_floor",
    threshold_value: Any = None,
) -> dict[str, Any]:
    """Build a marker describing deliberate runtime candidate-pack activation."""
    marker = {
        "report_type": "threshold_runtime_activation_marker",
        "generated_at": _utc_now(),
        "candidate_pack_name": str(candidate_pack_name or "").strip(),
        "activated_at": _coerce_timestamp(activated_at),
        "activated_by": str(activated_by or "operator").strip(),
        "activation_note": activation_note,
        "config_hint": config_hint,
        "threshold_value": threshold_value,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
    }
    return _sanitize_value(marker)


def load_threshold_runtime_activation_marker(path: str | Path | None = None) -> dict[str, Any]:
    """Load the latest runtime activation marker, returning an empty dict when absent."""
    marker_path = (
        Path(path)
        if path is not None
        else DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR / THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME
    )
    if not marker_path.exists():
        return {}
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_runtime_activation_capture_guard(
    result_payload: dict[str, Any],
    *,
    marker: dict[str, Any] | None = None,
    marker_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return whether a live signal row is allowed under the activation marker.

    The guard is intentionally conservative but non-mutating: it never switches
    the active parameter pack. It only tells capture sinks whether a live row
    should be persisted into research after the operator has marked a candidate
    pack as active.
    """
    payload = result_payload if isinstance(result_payload, dict) else {}
    loaded_marker = (
        marker
        if isinstance(marker, dict)
        else load_threshold_runtime_activation_marker(marker_path)
    )
    if not loaded_marker:
        return {
            "guard_active": False,
            "capture_allowed": True,
            "status": "NO_RUNTIME_ACTIVATION_MARKER",
            "runtime_config_changed": False,
            "parameter_pack_file_changed": False,
            "execution_behavior_changed": False,
        }

    mode = str(payload.get("mode") or "").upper().strip()
    expected_pack = str(loaded_marker.get("candidate_pack_name") or "").strip()
    activated_at = loaded_marker.get("activated_at")
    activation_ts = _safe_timestamp(activated_at)
    trade = payload.get("trade") if isinstance(payload.get("trade"), dict) else {}
    observed_pack = str(
        trade.get("parameter_pack_name")
        or payload.get("parameter_pack_name")
        or payload.get("authoritative_parameter_pack")
        or ""
    ).strip()
    spot_timestamp = (payload.get("spot_summary", {}) or {}).get("timestamp")
    signal_ts = _safe_timestamp(spot_timestamp or trade.get("valuation_time"))

    base = {
        "guard_active": True,
        "expected_parameter_pack": expected_pack or None,
        "observed_parameter_pack": observed_pack or None,
        "activated_at": activation_ts.isoformat() if activation_ts is not None else activated_at,
        "signal_timestamp": signal_ts.isoformat() if signal_ts is not None else None,
        "marker_generated_at": loaded_marker.get("generated_at"),
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
    }

    if mode != "LIVE":
        return {**base, "capture_allowed": True, "status": "NON_LIVE_MODE"}
    if not expected_pack:
        return {**base, "capture_allowed": True, "status": "MARKER_MISSING_CANDIDATE_PACK"}
    if activation_ts is not None and signal_ts is not None and signal_ts < activation_ts:
        return {**base, "capture_allowed": True, "status": "PRE_ACTIVATION_SIGNAL"}
    if observed_pack == expected_pack:
        return {**base, "capture_allowed": True, "status": "PARAMETER_PACK_MATCH"}

    return {**base, "capture_allowed": False, "status": "PARAMETER_PACK_MISMATCH"}


def render_threshold_runtime_activation_markdown(marker: dict[str, Any]) -> str:
    """Render the activation marker as Markdown."""
    lines = [
        "# Threshold Runtime Activation",
        "",
        f"- Generated at: {marker.get('generated_at')}",
        f"- Candidate pack: `{marker.get('candidate_pack_name')}`",
        f"- Activated at: {marker.get('activated_at')}",
        f"- Activated by: {marker.get('activated_by')}",
        f"- Threshold: `{marker.get('config_hint')} = {marker.get('threshold_value')}`",
        f"- Runtime config changed: {marker.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {marker.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {marker.get('execution_behavior_changed')}",
        "",
        "## Note",
        "",
        marker.get("activation_note") or "No activation note recorded.",
        "",
        "*This marker is signal-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*",
    ]
    return "\n".join(lines)


def write_threshold_runtime_activation_marker(
    *,
    candidate_pack_name: str = "candidate_v1",
    activated_at: Any = None,
    activated_by: str = "operator",
    activation_note: str | None = None,
    config_hint: str = "evaluation_thresholds.selection.composite_signal_score_floor",
    threshold_value: Any = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the latest runtime activation marker JSON/Markdown artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    marker = build_threshold_runtime_activation_marker(
        candidate_pack_name=candidate_pack_name,
        activated_at=activated_at,
        activated_by=activated_by,
        activation_note=activation_note,
        config_hint=config_hint,
        threshold_value=threshold_value,
    )
    json_path = output / THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME
    markdown_path = output / THRESHOLD_RUNTIME_ACTIVATION_MARKDOWN_FILENAME
    assert_artifact_schema(marker, "threshold_runtime_activation_marker")
    _atomic_write_text(json_path, json.dumps(marker, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_threshold_runtime_activation_markdown(marker))
    return {
        "activation_marker": marker,
        "activation_marker_json_path": str(json_path),
        "activation_marker_markdown_path": str(markdown_path),
    }
