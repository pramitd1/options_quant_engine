"""Build a candidate regime-parameter artifact from counterfactual evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.regime_cell_counterfactual import (
    DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR,
    LATEST_COUNTERFACTUAL_JSON_FILENAME,
)
from research.signal_evaluation.regime_cell_review import DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIME_PARAMETER_ARTIFACT_DIR = (
    PROJECT_ROOT / "research" / "ml_research" / "regime_parameter_artifacts"
)
DEFAULT_REGIME_PARAMETER_ARTIFACT_PATH = (
    DEFAULT_REGIME_PARAMETER_ARTIFACT_DIR / "latest_regime_parameter_candidate.json"
)
DEFAULT_REGIME_PARAMETER_MARKDOWN_PATH = (
    DEFAULT_REGIME_PARAMETER_ARTIFACT_DIR / "latest_regime_parameter_candidate.md"
)
DEFAULT_COUNTERFACTUAL_REPORT_PATH = (
    DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR / LATEST_COUNTERFACTUAL_JSON_FILENAME
)

ARTIFACT_VERSION = "regime_parameter_candidate_v1"
SUPPORTED_STATUS = "REPLAY_SUPPORTS_SUPPRESSION"


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _text(value: Any, default: str = "UNKNOWN") -> str:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
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


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _match_from_cell_text(cell_text: str) -> dict[str, str]:
    match = {}
    for part in str(cell_text or "").replace("|", "<br>").split("<br>"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            match[key] = value
    return match


def _rule_from_counterfactual_cell(row: dict[str, Any]) -> dict[str, Any]:
    match = _match_from_cell_text(row.get("matched_cell") or "")
    return {
        "rule_id": (
            "suppress_"
            + "_".join(
                str(match.get(field, "UNKNOWN")).lower()
                for field in ("gamma_regime", "volatility_regime", "direction", "macro_risk_bucket")
            )
        ),
        "action": "SUPPRESS_OR_AVOID",
        "match": {
            "gamma_regime": match.get("gamma_regime"),
            "volatility_regime": match.get("volatility_regime"),
            "direction": match.get("direction"),
            "macro_risk_bucket": match.get("macro_risk_bucket"),
        },
        "research_adjustments": {
            "score_adjustment": -4,
            "threshold_adjustment": 3,
            "size_multiplier": 0.50,
            "allow_trade": False,
            "hold_time_hint": "AVOID_OR_FAST_EXIT_ONLY",
        },
        "evidence": {
            "impact_status": row.get("impact_status"),
            "matched_signal_count": _safe_int(row.get("matched_signal_count")),
            "baseline_trade_count": _safe_int(row.get("baseline_trade_count")),
            "suppressed_trade_count": _safe_int(row.get("suppressed_trade_count")),
            "suppressed_label_count": _safe_int(row.get("suppressed_label_count")),
            "suppressed_avg_return_60m_bps": _round(row.get("suppressed_avg_return_60m_bps")),
            "avoided_suppressed_return_60m_bps": _round(row.get("avoided_suppressed_return_60m_bps")),
        },
    }


def build_regime_parameter_artifact(
    counterfactual_report: dict[str, Any],
    *,
    source_counterfactual_path: str | Path | None = None,
    min_suppressed_labels: int = 20,
) -> dict[str, Any]:
    """Build a disabled candidate artifact from supported suppression cells."""
    cell_summary = [
        row for row in (counterfactual_report.get("cell_summary") or []) if isinstance(row, dict)
    ]
    supported = [
        row
        for row in cell_summary
        if row.get("impact_status") == SUPPORTED_STATUS
        and _safe_int(row.get("suppressed_label_count")) >= int(min_suppressed_labels)
    ]
    rejected_or_insufficient = [
        row
        for row in cell_summary
        if row.get("impact_status") != SUPPORTED_STATUS
        or _safe_int(row.get("suppressed_label_count")) < int(min_suppressed_labels)
    ]

    rules = [_rule_from_counterfactual_cell(row) for row in supported]
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _utc_now(),
        "status": "CANDIDATE_RESEARCH_ONLY",
        "live_activation": {
            "enabled": False,
            "requires_counterfactual_review": True,
            "requires_fresh_forward_monitor": True,
            "runtime_config_changed": False,
            "parameter_pack_file_changed": False,
            "execution_behavior_changed": False,
        },
        "source": "regime_cell_counterfactual",
        "source_counterfactual_path": str(source_counterfactual_path) if source_counterfactual_path is not None else None,
        "source_assessment_status": counterfactual_report.get("assessment_status"),
        "source_assessment_basis": counterfactual_report.get("assessment_basis"),
        "guardrails": {
            "included_impact_status": SUPPORTED_STATUS,
            "min_suppressed_labels": int(min_suppressed_labels),
            "excluded_pcr_specific_cells": True,
            "watchlist_promotions_excluded_from_rules": True,
        },
        "rules": rules,
        "rule_count": int(len(rules)),
        "excluded_cell_count": int(len(rejected_or_insufficient)),
        "excluded_status_counts": {
            str(key): int(value)
            for key, value in pd.Series(
                [_text(row.get("impact_status")) for row in rejected_or_insufficient],
                dtype="object",
            )
            .value_counts()
            .to_dict()
            .items()
        },
        "recommended_next_actions": [
            "Review these rules manually before any engine wiring.",
            "Run a fresh-forward monitor after the next market session before activation.",
            "If activated later, load this as an explicit artifact with a visible runtime toggle and audit trail.",
        ],
    }
    return _sanitize(artifact)


def render_regime_parameter_artifact_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Regime Parameter Candidate Artifact",
        "",
        f"- Artifact version: {artifact.get('artifact_version')}",
        f"- Generated at: {artifact.get('generated_at')}",
        f"- Status: {artifact.get('status')}",
        f"- Source assessment: {artifact.get('source_assessment_status')}",
        f"- Source path: {artifact.get('source_counterfactual_path')}",
        f"- Rule count: {artifact.get('rule_count')}",
        f"- Runtime config changed: {(artifact.get('live_activation') or {}).get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {(artifact.get('live_activation') or {}).get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {(artifact.get('live_activation') or {}).get('execution_behavior_changed')}",
        "",
        "## Included Rules",
        "",
        "| Rule | Match | Action | Labels | Avg Suppressed 60m | Avoided Return |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for rule in artifact.get("rules", []):
        match = rule.get("match") or {}
        evidence = rule.get("evidence") or {}
        match_text = "<br>".join(f"{key}={value}" for key, value in match.items() if value is not None)
        lines.append(
            f"| {rule.get('rule_id')} | {match_text} | {rule.get('action')} | "
            f"{evidence.get('suppressed_label_count')} | {evidence.get('suppressed_avg_return_60m_bps')} | "
            f"{evidence.get('avoided_suppressed_return_60m_bps')} |"
        )
    if not artifact.get("rules"):
        lines.append("| none | none | none | 0 |  |  |")

    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            f"- Excluded cell count: {artifact.get('excluded_cell_count')}",
            f"- Excluded status counts: `{artifact.get('excluded_status_counts')}`",
            "- PCR-specific cells remain excluded until coverage improves.",
            "- Watchlist promotion candidates remain excluded from rules.",
            "",
            "## Activation",
            "",
            "This artifact is disabled by design. It should not change live behavior until a separate runtime wiring step, fresh-forward monitor, and manual activation decision are completed.",
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    for action in artifact.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def write_regime_parameter_artifact(
    *,
    counterfactual_path: str | Path = DEFAULT_COUNTERFACTUAL_REPORT_PATH,
    output_path: str | Path = DEFAULT_REGIME_PARAMETER_ARTIFACT_PATH,
    markdown_path: str | Path = DEFAULT_REGIME_PARAMETER_MARKDOWN_PATH,
    documentation_path: str | Path | None = DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR
    / "latest_regime_parameter_candidate.md",
    min_suppressed_labels: int = 20,
) -> dict[str, Any]:
    counterfactual_path = Path(counterfactual_path)
    output_path = Path(output_path)
    markdown_path = Path(markdown_path)
    counterfactual_report = _load_json(counterfactual_path)
    artifact = build_regime_parameter_artifact(
        counterfactual_report,
        source_counterfactual_path=counterfactual_path,
        min_suppressed_labels=min_suppressed_labels,
    )
    markdown = render_regime_parameter_artifact_markdown(artifact)
    _atomic_write_text(output_path, json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n")
    _atomic_write_text(markdown_path, markdown)

    paths = {
        "artifact": artifact,
        "artifact_path": str(output_path),
        "markdown_path": str(markdown_path),
    }
    if documentation_path is not None:
        documentation_path = Path(documentation_path)
        _atomic_write_text(documentation_path, markdown)
        paths["documentation_markdown_path"] = str(documentation_path)
    return paths
