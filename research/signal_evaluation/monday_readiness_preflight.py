"""Read-only Monday readiness preflight for the options signal engine.

The preflight is an operator-facing status check. It reads existing artifacts
and optional option-chain snapshots, then prints what is blocking Monday's
forward-shadow loop. It does not refresh outcomes, fetch live providers, change
runtime config, switch data sources, edit parameter packs, or execute trades.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import DATA_SOURCE_OPTIONS, DEFAULT_DATA_SOURCE, DEFAULT_SYMBOL
from data.option_chain_validation import validate_option_chain
from research.signal_evaluation.label_quality import select_quality_labeled_rows
from research.signal_evaluation.segmented_probability_guarded_candidate_staleness import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR,
    GUARDED_ACCUMULATING_FORWARD_LABELS,
    GUARDED_ACTIVE_REVIEW,
    GUARDED_STALENESS_NON_BLOCKING,
    SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_JSON_FILENAME,
)
from research.signal_evaluation.segmented_probability_shadow_soak import (
    DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR,
    SEGMENTED_PROBABILITY_SHADOW_SOAK_JSON_FILENAME,
    SOAK_ACCUMULATING_TRUE_FORWARD_LABELS,
    SOAK_HOLDOUT_REPLAY_REVIEWABLE,
    SOAK_READY_FOR_MANUAL_REVIEW,
)
from research.signal_evaluation.signal_quality_model_audit import (
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GUARDED_STALENESS_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR
    / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_JSON_FILENAME
)
DEFAULT_SHADOW_SOAK_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR
    / SEGMENTED_PROBABILITY_SHADOW_SOAK_JSON_FILENAME
)

PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS = "PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS"
PREFLIGHT_READY_FOR_MANUAL_REVIEW = "PREFLIGHT_READY_FOR_MANUAL_REVIEW"
PREFLIGHT_HOLDOUT_REPLAY_REVIEWABLE = "PREFLIGHT_HOLDOUT_REPLAY_REVIEWABLE"
PREFLIGHT_MONITOR = "PREFLIGHT_MONITOR"
PREFLIGHT_BLOCKED = "PREFLIGHT_BLOCKED"


def _load_json_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_dataset(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _dataset_summary(path: str | Path | None) -> dict[str, Any]:
    file_path = Path(path) if path is not None else default_signal_quality_dataset_path()
    summary = {
        "dataset_path": str(file_path),
        "dataset_exists": file_path.exists(),
        "row_count": 0,
        "quality_labeled_row_count": 0,
        "latest_signal_timestamp": None,
        "dataset_readable": False,
    }
    frame = _read_dataset(file_path)
    if frame.empty:
        return summary
    summary["dataset_readable"] = True
    summary["row_count"] = int(len(frame))
    try:
        summary["quality_labeled_row_count"] = int(len(select_quality_labeled_rows(frame)))
    except Exception:
        label_columns = [column for column in ("correct_60m", "calibration_label") if column in frame.columns]
        if label_columns:
            labels = pd.to_numeric(frame[label_columns[0]], errors="coerce")
            summary["quality_labeled_row_count"] = int(labels.notna().sum())
    if "signal_timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True)
        if timestamps.notna().any():
            summary["latest_signal_timestamp"] = timestamps.max().isoformat()
    return _sanitize_value(summary)


def _read_option_chain(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        if file_path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(file_path)
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _option_chain_health(
    *,
    option_chain_path: str | Path | None,
    selected_source: str,
    spot: float | None,
    as_of: Any = None,
    max_quote_age_seconds: float | None = None,
) -> dict[str, Any]:
    if option_chain_path is None:
        return {
            "check_status": "NOT_CHECKED",
            "reason": "no_option_chain_input",
            "option_chain_path": None,
            "is_valid": None,
            "is_stale": None,
            "summary_status": None,
            "trade_blocking_status": None,
            "source": None,
            "source_matches_selected": None,
            "warnings": [],
            "issues": [],
        }

    frame = _read_option_chain(option_chain_path)
    if frame.empty:
        return {
            "check_status": "UNAVAILABLE",
            "reason": "option_chain_file_missing_or_unreadable",
            "option_chain_path": str(option_chain_path),
            "is_valid": False,
            "is_stale": None,
            "summary_status": "WEAK",
            "trade_blocking_status": "BLOCK",
            "source": None,
            "source_matches_selected": None,
            "warnings": [],
            "issues": ["option_chain_file_missing_or_unreadable"],
        }

    validation = validate_option_chain(
        frame,
        spot=spot,
        as_of=as_of,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    provider = validation.get("provider_health", {}) if isinstance(validation, dict) else {}
    provider = provider if isinstance(provider, dict) else {}
    source = str(provider.get("source") or "").upper().strip() or None
    selected = str(selected_source or "").upper().strip()
    source_matches = bool(source == selected) if source else None
    return {
        "check_status": "CHECKED",
        "reason": None,
        "option_chain_path": str(option_chain_path),
        "is_valid": bool(validation.get("is_valid")),
        "is_stale": bool(validation.get("is_stale")),
        "summary_status": provider.get("summary_status"),
        "trade_blocking_status": provider.get("trade_blocking_status"),
        "trade_blocking_reasons": provider.get("trade_blocking_reasons", []) or [],
        "source": source,
        "source_matches_selected": source_matches,
        "market_data_readiness_score": validation.get("market_data_readiness_score"),
        "market_data_readiness_tier": validation.get("market_data_readiness_tier"),
        "quote_freshness_health": provider.get("quote_freshness_health"),
        "quote_spread_health": provider.get("quote_spread_health"),
        "liquidity_coverage_health": provider.get("liquidity_coverage_health"),
        "expiry_health": provider.get("expiry_health"),
        "warnings": validation.get("warnings", []) or [],
        "issues": validation.get("issues", []) or [],
    }


def _guarded_staleness_summary(report: dict[str, Any], *, path: str | Path | None) -> dict[str, Any]:
    candidate = report.get("guarded_candidate_summary", {}) if isinstance(report, dict) else {}
    currency = report.get("dataset_currency", {}) if isinstance(report, dict) else {}
    routing = report.get("guarded_routing_policy_stability", {}) if isinstance(report, dict) else {}
    return {
        "artifact_path": str(path) if path is not None else None,
        "artifact_found": bool(report),
        "guarded_staleness_status": report.get("guarded_staleness_status"),
        "guarded_staleness_reasons": report.get("guarded_staleness_reasons", []) or [],
        "guarded_candidate_generated_at": candidate.get("guarded_candidate_generated_at"),
        "guarded_candidate_age_days": candidate.get("guarded_candidate_age_days"),
        "guarded_candidate_count": candidate.get("candidate_count"),
        "quarantined_candidate_count": candidate.get("quarantined_candidate_count"),
        "post_guarded_quality_labeled_rows": currency.get(
            "quality_labeled_rows_after_guarded_candidate_generated"
        ),
        "rows_after_guarded_candidate": currency.get("rows_after_guarded_candidate_generated"),
        "guarded_routing_policy_status": routing.get("policy_stability_status"),
        "latest_guarded_recommended_routing_policy": routing.get(
            "latest_guarded_recommended_routing_policy"
        ),
    }


def _soak_summary(report: dict[str, Any], *, path: str | Path | None) -> dict[str, Any]:
    progress = report.get("forward_sample_progress", {}) if isinstance(report, dict) else {}
    guarded_progress = report.get("guarded_forward_sample_progress", {}) if isinstance(report, dict) else {}
    guarded = report.get("guarded_validation_summary", {}) if isinstance(report, dict) else {}
    readiness = report.get("readiness_summary", {}) if isinstance(report, dict) else {}
    guarded_staleness = (
        report.get("guarded_candidate_staleness_summary", {}) if isinstance(report, dict) else {}
    )
    return {
        "artifact_path": str(path) if path is not None else None,
        "artifact_found": bool(report),
        "soak_status": report.get("soak_status"),
        "soak_reasons": report.get("soak_reasons", []) or [],
        "strict_forward_row_count": progress.get("strict_forward_row_count"),
        "forward_sample_gap": progress.get("forward_sample_gap"),
        "guarded_strict_forward_row_count": guarded_progress.get("guarded_strict_forward_row_count"),
        "guarded_forward_sample_gap": guarded_progress.get("forward_sample_gap"),
        "new_post_guarded_true_forward_rows_since_previous_soak": guarded_progress.get(
            "new_post_guarded_true_forward_rows_since_previous_soak"
        ),
        "guarded_shadow_status": guarded.get("guarded_shadow_status"),
        "guarded_validation_mode_used": guarded.get("validation_mode_used"),
        "readiness_status": readiness.get("readiness_status"),
        "guarded_staleness_status": guarded_staleness.get("guarded_staleness_status"),
        "recommended_next_actions": report.get("recommended_next_actions", []) or [],
    }


def _commands(*, source: str, symbol: str) -> dict[str, str]:
    source_clean = str(source or DEFAULT_DATA_SOURCE).upper().strip()
    symbol_clean = str(symbol or DEFAULT_SYMBOL).upper().strip()
    return {
        "preflight": (
            ".venv/bin/python scripts/ops/run_monday_readiness_preflight.py "
            f"--source {source_clean} --symbol {symbol_clean}"
        ),
        "streamlit_live": (
            ".venv/bin/python -m streamlit run app/streamlit_app.py"
            f"  # choose LIVE / {symbol_clean} / {source_clean} in the sidebar"
        ),
        "shadow_soak": ".venv/bin/python scripts/ops/run_segmented_probability_shadow_soak.py",
        "guarded_staleness": (
            ".venv/bin/python scripts/ops/run_segmented_probability_guarded_candidate_staleness.py"
        ),
    }


def _classify_preflight(
    *,
    selected_source_supported: bool,
    option_chain: dict[str, Any],
    guarded: dict[str, Any],
    soak: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    if not selected_source_supported:
        blockers.append("selected_source_not_supported")

    if not guarded.get("artifact_found"):
        blockers.append("missing_guarded_staleness_artifact")
    guarded_status = str(guarded.get("guarded_staleness_status") or "").strip()
    if guarded_status and guarded_status not in GUARDED_STALENESS_NON_BLOCKING:
        blockers.append(f"guarded_staleness_not_clean:{guarded_status}")

    if not soak.get("artifact_found"):
        warnings.append("missing_shadow_soak_artifact")
    soak_status = str(soak.get("soak_status") or "").strip()
    if soak.get("artifact_found") and not soak.get("guarded_staleness_status"):
        warnings.append("shadow_soak_missing_guarded_staleness_context")
    if soak_status in {
        "SOAK_GUARDED_VALIDATION_REJECTED",
        "SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED",
        "SOAK_SIDE_EFFECT_BLOCKED",
    }:
        blockers.append(f"soak_blocked:{soak_status}")
    elif soak_status == "SOAK_CANDIDATE_STALENESS_BLOCKED" and guarded_status not in GUARDED_STALENESS_NON_BLOCKING:
        blockers.append(f"soak_blocked:{soak_status}")
    elif soak_status == "SOAK_CANDIDATE_STALENESS_BLOCKED":
        warnings.append("source_candidate_staleness_context_only_guarded_bundle_non_blocking")

    if option_chain.get("check_status") == "CHECKED":
        if not option_chain.get("is_valid"):
            blockers.append("option_chain_invalid")
        if option_chain.get("is_stale"):
            blockers.append("option_chain_stale")
        if str(option_chain.get("trade_blocking_status") or "").upper() == "BLOCK":
            blockers.append("option_chain_trade_blocking_quality")
        if option_chain.get("source_matches_selected") is False:
            warnings.append("option_chain_source_differs_from_selected_source")
        if str(option_chain.get("summary_status") or "").upper() in {"CAUTION", "WEAK"}:
            warnings.append(f"option_chain_quality:{option_chain.get('summary_status')}")
    elif option_chain.get("check_status") == "UNAVAILABLE":
        warnings.append("option_chain_snapshot_unavailable_for_preflight")

    if blockers:
        return PREFLIGHT_BLOCKED, list(dict.fromkeys(blockers)), list(dict.fromkeys(warnings))
    if soak_status == SOAK_READY_FOR_MANUAL_REVIEW:
        return PREFLIGHT_READY_FOR_MANUAL_REVIEW, [], list(dict.fromkeys(warnings))
    if soak_status == SOAK_HOLDOUT_REPLAY_REVIEWABLE:
        return PREFLIGHT_HOLDOUT_REPLAY_REVIEWABLE, [], list(dict.fromkeys(warnings))
    if guarded_status == GUARDED_ACCUMULATING_FORWARD_LABELS or soak_status == SOAK_ACCUMULATING_TRUE_FORWARD_LABELS:
        return PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS, [], list(dict.fromkeys(warnings))
    if guarded_status == GUARDED_ACTIVE_REVIEW:
        return PREFLIGHT_READY_TO_COLLECT_FORWARD_LABELS, [], list(dict.fromkeys(warnings))
    return PREFLIGHT_MONITOR, [], list(dict.fromkeys(warnings))


def _recommended_actions(
    *,
    status: str,
    blockers: list[str],
    warnings: list[str],
    commands: dict[str, str],
    guarded: dict[str, Any],
    soak: dict[str, Any],
) -> list[str]:
    if status == PREFLIGHT_BLOCKED:
        return [
            "Resolve the listed blockers before treating Monday evidence as review-ready.",
            "Do not change runtime config, parameter packs, data sources, or execution behavior from this preflight.",
            f"After remediation, rerun: {commands['preflight']}",
        ]
    if status == PREFLIGHT_READY_FOR_MANUAL_REVIEW:
        return [
            "Open manual review for the guarded segmented-probability evidence; runtime adoption still requires explicit approval.",
            f"Archive the latest soak artifact, then rerun as needed: {commands['shadow_soak']}",
        ]
    gap = soak.get("guarded_forward_sample_gap")
    if gap is None:
        gap = 100 - int(guarded.get("post_guarded_quality_labeled_rows") or 0)
    gap_text = _round_or_none(gap, 0)
    actions = [
        f"Run the live signal engine after market opens using the selected source; current guarded forward sample gap is {gap_text}.",
        f"Refresh the soak after new outcomes can be labeled: {commands['shadow_soak']}",
    ]
    if warnings:
        actions.append("Review warnings, but keep the user-selected data source unless the operator changes it manually.")
    actions.append("This preflight is advisory only and cannot execute trades.")
    return actions


def build_monday_readiness_preflight_report(
    *,
    source: str | None = None,
    symbol: str | None = None,
    dataset_path: str | Path | None = None,
    guarded_staleness_path: str | Path | None = None,
    shadow_soak_path: str | Path | None = None,
    option_chain_path: str | Path | None = None,
    spot: float | None = None,
    as_of: Any = None,
    max_quote_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the read-only Monday preflight report."""
    selected_source = str(source or DEFAULT_DATA_SOURCE).upper().strip()
    selected_symbol = str(symbol or DEFAULT_SYMBOL).upper().strip()
    supported_sources = [str(item).upper().strip() for item in DATA_SOURCE_OPTIONS]
    selected_source_supported = selected_source in supported_sources

    guarded_path = Path(guarded_staleness_path) if guarded_staleness_path is not None else DEFAULT_GUARDED_STALENESS_PATH
    soak_path = Path(shadow_soak_path) if shadow_soak_path is not None else DEFAULT_SHADOW_SOAK_PATH
    guarded_report = _load_json_file(guarded_path)
    soak_report = _load_json_file(soak_path)

    dataset = _dataset_summary(dataset_path or default_signal_quality_dataset_path())
    option_chain = _option_chain_health(
        option_chain_path=option_chain_path,
        selected_source=selected_source,
        spot=spot,
        as_of=as_of,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    guarded = _guarded_staleness_summary(guarded_report, path=guarded_path)
    soak = _soak_summary(soak_report, path=soak_path)
    commands = _commands(source=selected_source, symbol=selected_symbol)
    status, blockers, warnings = _classify_preflight(
        selected_source_supported=selected_source_supported,
        option_chain=option_chain,
        guarded=guarded,
        soak=soak,
    )
    report = {
        "report_type": "monday_readiness_preflight",
        "generated_at": _utc_now(),
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "trades_executed": False,
        "outcome_refresh_attempted": False,
        "provider_fetch_attempted": False,
        "data_source_policy": {
            "selected_source": selected_source,
            "selected_symbol": selected_symbol,
            "supported_sources": supported_sources,
            "selected_source_supported": bool(selected_source_supported),
            "source_sticky": True,
            "source_override_attempted": False,
            "fallback_provider_attempted": False,
        },
        "preflight_status": status,
        "blockers": blockers,
        "warnings": warnings,
        "dataset_summary": dataset,
        "option_chain_health": option_chain,
        "guarded_staleness": guarded,
        "shadow_soak": soak,
        "next_commands": commands,
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(
        status=status,
        blockers=blockers,
        warnings=warnings,
        commands=commands,
        guarded=guarded,
        soak=soak,
    )
    return _sanitize_value(report)
