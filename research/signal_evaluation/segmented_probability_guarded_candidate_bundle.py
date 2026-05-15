"""Generate guarded segmented-probability candidate bundles.

This module converts a passing guarded EV experiment into a new research-only
candidate bundle. The bundle quarantines EV-negative routes and records
rank-preservation governance metadata, but it does not change runtime
configuration, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    _candidate_key,
    _load_candidate_bundle,
)
from research.signal_evaluation.segmented_probability_guarded_ev_experiment import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR,
    GUARDED_EV_EXPERIMENT_PASS,
    SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_JSON_FILENAME,
)
from research.signal_evaluation.signal_quality_model_audit import (
    _atomic_write_csv,
    _atomic_write_text,
    _sanitize_value,
    _utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_guarded_candidate_bundle"
)
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR
    / SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_JSON_FILENAME
)

SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_REPORT_FILENAME = (
    "latest_segmented_probability_guarded_candidate_bundle_report.json"
)
SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_MARKDOWN_FILENAME = (
    "latest_segmented_probability_guarded_candidate_bundle.md"
)
SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME = (
    "latest_segmented_probability_guarded_candidate_bundle.json"
)
SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_CANDIDATES_FILENAME = (
    "latest_segmented_probability_guarded_candidate_bundle_candidates.csv"
)

GUARDED_CANDIDATE_BUNDLE_READY = "GUARDED_CANDIDATE_BUNDLE_READY"
GUARDED_CANDIDATE_BUNDLE_WATCH = "GUARDED_CANDIDATE_BUNDLE_WATCH"
GUARDED_CANDIDATE_BUNDLE_BLOCKED = "GUARDED_CANDIDATE_BUNDLE_BLOCKED"

RECOMMENDED_GUARDED_VARIANT = "quarantine_plus_rank_guard"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_rows(candidates: list[dict[str, Any]], *, status: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_key": _candidate_key(candidate),
                "candidate_status": status,
                "candidate_priority": candidate.get("candidate_priority"),
                "candidate_type": candidate.get("candidate_type"),
                "segment_field": candidate.get("segment_field"),
                "segment_value": candidate.get("segment_value"),
                "selected_calibrator": candidate.get("selected_calibrator"),
                "selection_reason": (candidate.get("selection", {}) or {}).get("selection_reason"),
                "holdout_count": (candidate.get("selection", {}) or {}).get("holdout_count"),
                "holdout_brier_improvement": (candidate.get("selection", {}) or {}).get(
                    "holdout_brier_improvement"
                ),
            }
        )
    return rows


def _side_effect_flags_clean(*payloads: dict[str, Any]) -> bool:
    for payload in payloads:
        if payload.get("runtime_config_changed") is not False:
            return False
        if payload.get("parameter_pack_file_changed") is not False:
            return False
        if payload.get("execution_behavior_changed") is not False:
            return False
    return True


def _bundle_status(
    *,
    guarded_ev_status: str,
    recommended_variant: str | None,
    kept_count: int,
    quarantined_count: int,
    side_effect_flags_clean: bool,
    allow_watch: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not side_effect_flags_clean:
        reasons.append("side_effect_flags_not_clean")
    if guarded_ev_status != GUARDED_EV_EXPERIMENT_PASS:
        reasons.append("guarded_ev_experiment_not_passed")
    if recommended_variant != RECOMMENDED_GUARDED_VARIANT:
        reasons.append("recommended_guarded_variant_not_quarantine_plus_rank_guard")
    if kept_count <= 0:
        reasons.append("no_candidates_remaining_after_quarantine")
    if quarantined_count <= 0:
        reasons.append("no_ev_negative_routes_to_quarantine")
    if not reasons:
        return GUARDED_CANDIDATE_BUNDLE_READY, []
    blocking = {
        "side_effect_flags_not_clean",
        "guarded_ev_experiment_not_passed",
        "no_candidates_remaining_after_quarantine",
    }
    if any(reason in blocking for reason in reasons) and not allow_watch:
        return GUARDED_CANDIDATE_BUNDLE_BLOCKED, reasons
    return GUARDED_CANDIDATE_BUNDLE_WATCH, reasons


def _rank_preservation_policy(guarded_ev_experiment: dict[str, Any]) -> dict[str, Any]:
    selection = guarded_ev_experiment.get("selection_summary", {}) or {}
    return {
        "policy_name": "raw_rank_preservation_guard",
        "enabled_for_research_review": True,
        "governance_only": True,
        "runtime_behavior_changed": False,
        "parameter_pack_file_changed": False,
        "data_source_changed": False,
        "top_fraction": guarded_ev_experiment.get("top_fraction"),
        "raw_rank_ceiling_multiplier": guarded_ev_experiment.get("raw_rank_ceiling_multiplier"),
        "recommended_guarded_variant": selection.get("recommended_guarded_variant"),
        "selection_rule": (
            "Calibrated probabilities may be reviewed as advisory estimates, but top-bucket signal selection "
            "must remain inside the configured raw-probability rank ceiling during research validation."
        ),
        "requires_guard_aware_shadow_evaluation": True,
        "standard_candidate_bundle_consumers_ignore_guard_policy": True,
    }


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("guarded_candidate_bundle_status")
    bundle_path = report.get("guarded_candidate_bundle_path")
    if status == GUARDED_CANDIDATE_BUNDLE_READY:
        return [
            f"Use `{bundle_path}` as the next research-only candidate bundle for guard-aware shadow validation.",
            "Run forward shadow and EV shadow with guard-aware ranking before any manual promotion review.",
            "Keep runtime config, parameter packs, data sources, and execution behavior unchanged.",
        ]
    if status == GUARDED_CANDIDATE_BUNDLE_WATCH:
        return [
            "Review the bundle warnings before using this guarded candidate bundle in shadow validation.",
            "Do not treat this artifact as an approval to change runtime probabilities.",
        ]
    return [
        "Do not use this guarded candidate bundle for the next shadow cycle until blocking reasons are resolved.",
        "Return to guarded EV experiment diagnostics and candidate generation inputs.",
    ]


def build_segmented_probability_guarded_candidate_bundle_report(
    source_candidate_bundle: dict[str, Any],
    guarded_ev_experiment: dict[str, Any],
    *,
    source_candidate_bundle_path: str | Path | None = None,
    guarded_ev_experiment_path: str | Path | None = None,
    guarded_candidate_bundle_path: str | Path | None = None,
    allow_watch: bool = False,
) -> dict[str, Any]:
    """Build a guarded candidate bundle and companion governance report."""
    source = source_candidate_bundle if isinstance(source_candidate_bundle, dict) else {}
    experiment = guarded_ev_experiment if isinstance(guarded_ev_experiment, dict) else {}
    source_candidates = [candidate for candidate in source.get("candidates", []) or [] if isinstance(candidate, dict)]
    quarantined_keys = {
        str(key)
        for key in experiment.get("quarantined_candidate_keys", []) or []
        if str(key).strip()
    }
    kept_candidates = [
        copy.deepcopy(candidate)
        for candidate in source_candidates
        if _candidate_key(candidate) not in quarantined_keys
    ]
    removed_candidates = [
        copy.deepcopy(candidate)
        for candidate in source_candidates
        if _candidate_key(candidate) in quarantined_keys
    ]
    selection = experiment.get("selection_summary", {}) or {}
    side_effects_clean = _side_effect_flags_clean(source, experiment)
    status, reasons = _bundle_status(
        guarded_ev_status=str(experiment.get("guarded_ev_status") or ""),
        recommended_variant=selection.get("recommended_guarded_variant"),
        kept_count=len(kept_candidates),
        quarantined_count=len(removed_candidates),
        side_effect_flags_clean=side_effects_clean,
        allow_watch=allow_watch,
    )
    rank_policy = _rank_preservation_policy(experiment)
    bundle = {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "bundle_variant": "guarded_ev_quarantine_plus_rank_guard",
        "generated_at": _utc_now(),
        "source_candidate_bundle_path": str(source_candidate_bundle_path) if source_candidate_bundle_path else None,
        "source_candidate_bundle_generated_at": source.get("generated_at"),
        "guarded_ev_experiment_path": str(guarded_ev_experiment_path) if guarded_ev_experiment_path else None,
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": source.get("probability_field") or experiment.get("probability_field"),
        "label_field": source.get("label_field") or experiment.get("label_field"),
        "calibration_status": status,
        "guarded_candidate_bundle_status": status,
        "guarded_candidate_bundle_reasons": reasons,
        "candidate_count": int(len(kept_candidates)),
        "source_candidate_count": int(len(source_candidates)),
        "quarantined_candidate_count": int(len(removed_candidates)),
        "quarantined_candidate_keys": sorted(_candidate_key(candidate) for candidate in removed_candidates),
        "requested_quarantine_keys": sorted(quarantined_keys),
        "missing_quarantine_keys": sorted(
            key for key in quarantined_keys if key not in {_candidate_key(candidate) for candidate in removed_candidates}
        ),
        "rank_preservation_policy": rank_policy,
        "required_next_validations": [
            "guard_aware_forward_shadow",
            "guard_aware_ev_shadow_evaluation",
            "ev_rejection_attribution",
            "forward_shadow_readiness_gate",
        ],
        "candidates": kept_candidates,
    }
    candidates_table = (
        _candidate_rows(kept_candidates, status="kept")
        + _candidate_rows(removed_candidates, status="quarantined")
    )
    report = {
        "report_type": "segmented_probability_guarded_candidate_bundle",
        "generated_at": _utc_now(),
        "source_candidate_bundle_path": str(source_candidate_bundle_path) if source_candidate_bundle_path else None,
        "guarded_ev_experiment_path": str(guarded_ev_experiment_path) if guarded_ev_experiment_path else None,
        "guarded_candidate_bundle_path": str(guarded_candidate_bundle_path) if guarded_candidate_bundle_path else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_candidate_bundle_status": status,
        "guarded_candidate_bundle_reasons": reasons,
        "guarded_ev_status": str(experiment.get("guarded_ev_status") or ""),
        "recommended_guarded_variant": selection.get("recommended_guarded_variant"),
        "rank_preservation_policy": rank_policy,
        "source_candidate_count": int(len(source_candidates)),
        "kept_candidate_count": int(len(kept_candidates)),
        "quarantined_candidate_count": int(len(removed_candidates)),
        "quarantined_candidate_keys": sorted(_candidate_key(candidate) for candidate in removed_candidates),
        "missing_quarantine_keys": bundle["missing_quarantine_keys"],
        "required_next_validations": bundle["required_next_validations"],
        "guarded_candidate_bundle": bundle,
        "candidate_rows": candidates_table,
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    return _sanitize_value(report)


def render_segmented_probability_guarded_candidate_bundle_markdown(report: dict[str, Any]) -> str:
    """Render guarded candidate-bundle governance as Markdown."""
    lines = [
        "# Segmented Probability Guarded Candidate Bundle",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Status: `{report.get('guarded_candidate_bundle_status')}`",
        f"- Source bundle: {report.get('source_candidate_bundle_path') or 'inline'}",
        f"- Guarded EV experiment: {report.get('guarded_ev_experiment_path') or 'inline'}",
        f"- Guarded bundle: {report.get('guarded_candidate_bundle_path') or 'pending write'}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Candidate Changes",
        "",
        f"- Source candidates: {report.get('source_candidate_count')}",
        f"- Kept candidates: {report.get('kept_candidate_count')}",
        f"- Quarantined candidates: {report.get('quarantined_candidate_count')}",
        "",
        "| Candidate | Status | Type | Segment | Calibrator |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.get("candidate_rows", []) or []:
        segment = f"{row.get('segment_field')}={row.get('segment_value')}"
        lines.append(
            f"| `{row.get('candidate_key')}` | `{row.get('candidate_status')}` | "
            f"`{row.get('candidate_type')}` | `{segment}` | `{row.get('selected_calibrator')}` |"
        )
    policy = report.get("rank_preservation_policy", {}) or {}
    lines.extend(
        [
            "",
            "## Rank Preservation",
            "",
            f"- Governance only: {policy.get('governance_only')}",
            f"- Requires guard-aware shadow evaluation: {policy.get('requires_guard_aware_shadow_evaluation')}",
            f"- Top fraction: {policy.get('top_fraction')}",
            f"- Raw-rank ceiling multiplier: {policy.get('raw_rank_ceiling_multiplier')}",
            "",
            "## Required Next Validations",
            "",
        ]
    )
    for validation in report.get("required_next_validations", []) or []:
        lines.append(f"- `{validation}`")
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("*Research-only artifact. Runtime config, parameter packs, data sources, and execution behavior are unchanged.*")
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "report_json_path": output / f"{stem}_report.json",
        "markdown_path": output / f"{stem}.md",
        "candidate_bundle_json_path": output / f"{stem}.json",
        "candidates_csv_path": output / f"{stem}_candidates.csv",
        "latest_report_json_path": output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_REPORT_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_MARKDOWN_FILENAME,
        "latest_candidate_bundle_json_path": output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_CANDIDATES_FILENAME,
    }


def write_segmented_probability_guarded_candidate_bundle_report(
    source_candidate_bundle: dict[str, Any],
    guarded_ev_experiment: dict[str, Any],
    *,
    source_candidate_bundle_path: str | Path | None = None,
    guarded_ev_experiment_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    allow_watch: bool = False,
) -> dict[str, Any]:
    """Build and write guarded candidate-bundle artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_guarded_candidate_bundle"
    paths = _artifact_paths(output, stem)
    report = build_segmented_probability_guarded_candidate_bundle_report(
        source_candidate_bundle,
        guarded_ev_experiment,
        source_candidate_bundle_path=source_candidate_bundle_path,
        guarded_ev_experiment_path=guarded_ev_experiment_path,
        guarded_candidate_bundle_path=paths["candidate_bundle_json_path"],
        allow_watch=allow_watch,
    )
    assert_artifact_schema(report, "segmented_probability_guarded_candidate_bundle")
    markdown = render_segmented_probability_guarded_candidate_bundle_markdown(report)
    candidate_bundle = report.get("guarded_candidate_bundle", {}) or {}
    candidates = pd.DataFrame(report.get("candidate_rows", []) or [])

    _atomic_write_text(paths["report_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_text(paths["candidate_bundle_json_path"], json.dumps(candidate_bundle, indent=2, sort_keys=True, default=str))
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    if write_latest:
        latest_report = dict(report)
        latest_report["guarded_candidate_bundle_path"] = str(paths["latest_candidate_bundle_json_path"])
        latest_bundle = dict(candidate_bundle)
        latest_bundle["guarded_candidate_bundle_path"] = str(paths["latest_candidate_bundle_json_path"])
        latest_markdown = render_segmented_probability_guarded_candidate_bundle_markdown(latest_report)
        assert_artifact_schema(latest_report, "segmented_probability_guarded_candidate_bundle")
        _atomic_write_text(
            paths["latest_report_json_path"],
            json.dumps(latest_report, indent=2, sort_keys=True, default=str),
        )
        _atomic_write_text(paths["latest_markdown_path"], latest_markdown)
        _atomic_write_text(
            paths["latest_candidate_bundle_json_path"],
            json.dumps(latest_bundle, indent=2, sort_keys=True, default=str),
        )
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
    artifact = {"report": report, "candidate_bundle": candidate_bundle}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_guarded_candidate_bundle_report_from_paths(
    *,
    source_candidate_bundle_path: str | Path | None = None,
    guarded_ev_experiment_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load source artifacts and write a guarded candidate bundle."""
    source_path = Path(source_candidate_bundle_path) if source_candidate_bundle_path is not None else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    experiment_path = Path(guarded_ev_experiment_path) if guarded_ev_experiment_path is not None else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_PATH
    source_bundle = _load_candidate_bundle(source_path)
    guarded_experiment = _read_json(experiment_path)
    if source_candidate_bundle_path is None and guarded_experiment.get("candidate_bundle_path"):
        candidate = Path(str(guarded_experiment.get("candidate_bundle_path")))
        if candidate.exists():
            source_path = candidate
            source_bundle = _load_candidate_bundle(source_path)
    return write_segmented_probability_guarded_candidate_bundle_report(
        source_bundle,
        guarded_experiment,
        source_candidate_bundle_path=source_path,
        guarded_ev_experiment_path=experiment_path,
        **kwargs,
    )
