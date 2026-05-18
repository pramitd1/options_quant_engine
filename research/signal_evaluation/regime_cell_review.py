"""Review reliable regime outcome cells and turn them into research proposals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.regime_outcome_tables import (
    DEFAULT_REGIME_OUTCOME_TABLE_DIR,
    REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME,
    REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIME_CELL_REVIEW_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "regime_cell_review"
)
DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR = (
    PROJECT_ROOT / "documentation" / "research_reports" / "regime_parameterization"
)

LATEST_REVIEW_JSON_FILENAME = "latest_regime_cell_review.json"
LATEST_REVIEW_MARKDOWN_FILENAME = "latest_regime_cell_review.md"
LATEST_REVIEW_CSV_FILENAME = "latest_regime_cell_review_cells.csv"

TARGET_GROUPS = (
    "gamma_regime+volatility_regime+direction",
    "gamma_regime+volatility_regime+direction+macro_risk_bucket",
)

HORIZON_ORDER = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "120m": 120,
    "session_close": 390,
}


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _int(value: Any, default: int = 0) -> int:
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


def _cell_key(row: pd.Series | dict[str, Any]) -> tuple[str, ...]:
    group_name = _text(row.get("group_name"))
    fields = group_name.split("+")
    return tuple(_text(row.get(field)) for field in fields)


def _format_cell(row: dict[str, Any]) -> str:
    fields = str(row.get("group_name") or "").split("+")
    parts = []
    for field in fields:
        value = row.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return "<br>".join(parts)


def _horizon_profile(horizon_group: pd.DataFrame) -> dict[str, Any]:
    if horizon_group.empty:
        return {
            "early_best_avg_return_bps": None,
            "late_best_avg_return_bps": None,
            "best_late_minus_early_bps": None,
            "early_hit_rate": None,
            "late_hit_rate": None,
            "horizon_rows": [],
        }
    working = horizon_group.copy()
    working["_order"] = working["horizon"].map(HORIZON_ORDER).fillna(999)
    working = working.sort_values("_order", kind="mergesort")
    early = working.loc[working["horizon"].isin(["5m", "15m", "30m"])]
    late = working.loc[working["horizon"].isin(["60m", "120m", "session_close"])]
    early_return = pd.to_numeric(early.get("avg_signed_return_bps"), errors="coerce")
    late_return = pd.to_numeric(late.get("avg_signed_return_bps"), errors="coerce")
    early_hit = pd.to_numeric(early.get("hit_rate"), errors="coerce")
    late_hit = pd.to_numeric(late.get("hit_rate"), errors="coerce")
    early_best = _round(early_return.max() if not early_return.empty else None)
    late_best = _round(late_return.max() if not late_return.empty else None)
    return {
        "early_best_avg_return_bps": early_best,
        "late_best_avg_return_bps": late_best,
        "best_late_minus_early_bps": _round(
            late_best - early_best if late_best is not None and early_best is not None else None
        ),
        "early_hit_rate": _round(early_hit.max() if not early_hit.empty else None),
        "late_hit_rate": _round(late_hit.max() if not late_hit.empty else None),
        "horizon_rows": [
            {
                "horizon": item.get("horizon"),
                "label_count": _int(item.get("label_count")),
                "hit_rate": _round(item.get("hit_rate")),
                "avg_signed_return_bps": _round(item.get("avg_signed_return_bps")),
                "sample_quality": item.get("sample_quality"),
            }
            for item in working.drop(columns=["_order"], errors="ignore").to_dict("records")
        ],
    }


def classify_regime_cell(row: dict[str, Any], horizon_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify one reliable cell into a research-only operating proposal."""
    horizon_profile = horizon_profile or {}
    hit = _round(row.get("hit_rate"))
    avg_return = _round(row.get("avg_signed_return_bps"))
    hit_delta = _round(row.get("hit_rate_delta_vs_all"))
    return_delta = _round(row.get("avg_return_delta_vs_all_bps"))
    label_count = _int(row.get("label_count"))
    best_horizon = _text(row.get("best_horizon"))
    late_minus_early = _round(horizon_profile.get("best_late_minus_early_bps"))

    strong_favorable = (
        hit is not None
        and avg_return is not None
        and hit_delta is not None
        and return_delta is not None
        and hit >= 0.55
        and avg_return >= 5.0
        and hit_delta >= 0.08
        and return_delta >= 15.0
    )
    hard_unfavorable = (
        hit is not None
        and avg_return is not None
        and (
            (hit <= 0.45 and avg_return <= 0.0)
            or (hit_delta is not None and return_delta is not None and hit_delta <= -0.05 and return_delta <= -2.0)
            or (best_horizon == "5m" and avg_return <= 0.0)
        )
    )
    low_hit_positive_tail = hit is not None and avg_return is not None and hit < 0.48 and avg_return > 5.0
    long_hold_edge = (
        best_horizon in {"60m", "120m", "session_close"}
        and late_minus_early is not None
        and late_minus_early >= 10.0
        and avg_return is not None
        and avg_return > 0.0
    )

    if hard_unfavorable:
        action_class = "DOWNGRADE_OR_AVOID"
        score_adjustment = -4
        threshold_adjustment = 3
        size_multiplier = 0.50
        allow_trade = False
        hold_time_hint = "AVOID_OR_FAST_EXIT_ONLY"
        rationale = "Reliable cell has poor hit/return evidence versus the global baseline."
    elif low_hit_positive_tail:
        action_class = "REQUIRE_CONFIRMATION"
        score_adjustment = -1
        threshold_adjustment = 2
        size_multiplier = 0.70
        allow_trade = True
        hold_time_hint = "WAIT_FOR_CONFIRMATION_THEN_USE_BEST_HORIZON"
        rationale = "Average return is positive but hit rate is weak, suggesting tail-driven payoff."
    elif strong_favorable and long_hold_edge:
        action_class = "HOLD_TIME_SPECIAL"
        score_adjustment = 2
        threshold_adjustment = -1
        size_multiplier = 0.85
        allow_trade = True
        hold_time_hint = f"ALLOW_{best_horizon.upper()}_IF_SIGNAL_MATURES"
        rationale = "Reliable favorable cell improves materially at the longer hold horizon."
    elif strong_favorable:
        action_class = "FAVORABLE_BOOST"
        score_adjustment = 3
        threshold_adjustment = -1
        size_multiplier = 1.00
        allow_trade = True
        hold_time_hint = f"PREFER_{best_horizon.upper()}"
        rationale = "Reliable cell is above baseline on both hit rate and realized edge."
    else:
        action_class = "REQUIRE_CONFIRMATION"
        score_adjustment = 0
        threshold_adjustment = 1
        size_multiplier = 0.85
        allow_trade = True
        hold_time_hint = "STANDARD_WITH_EXTRA_CONFIRMATION"
        rationale = "Reliable sample exists, but evidence is mixed or only modestly different from baseline."

    confidence = "HIGH" if label_count >= 500 else ("MEDIUM" if label_count >= 100 else "LOW")
    return {
        "action_class": action_class,
        "proposal_confidence": confidence,
        "score_adjustment_research": score_adjustment,
        "threshold_adjustment_research": threshold_adjustment,
        "size_multiplier_research": size_multiplier,
        "allow_trade_research": allow_trade,
        "hold_time_hint": hold_time_hint,
        "rationale": rationale,
    }


def _review_rows(best_horizon: pd.DataFrame, by_horizon: pd.DataFrame) -> list[dict[str, Any]]:
    if best_horizon.empty:
        return []
    selected = best_horizon.loc[
        best_horizon["group_name"].isin(TARGET_GROUPS)
        & best_horizon["sample_quality"].astype(str).str.upper().eq("RELIABLE")
    ].copy()
    if selected.empty:
        return []

    by_horizon = by_horizon.copy()
    rows = []
    for _idx, row in selected.iterrows():
        key = _cell_key(row)
        group_name = _text(row.get("group_name"))
        fields = group_name.split("+")
        horizon_group = by_horizon.loc[by_horizon["group_name"].eq(group_name)].copy()
        for field, value in zip(fields, key):
            if field in horizon_group.columns:
                horizon_group = horizon_group.loc[horizon_group[field].astype(str).eq(str(value))]
        profile = _horizon_profile(horizon_group)
        base = {key: _sanitize(value) for key, value in row.to_dict().items()}
        base.update(classify_regime_cell(base, profile))
        base["cell"] = _format_cell(base)
        base["horizon_profile"] = profile
        rows.append(base)

    action_order = {
        "HOLD_TIME_SPECIAL": 0,
        "FAVORABLE_BOOST": 1,
        "REQUIRE_CONFIRMATION": 2,
        "DOWNGRADE_OR_AVOID": 3,
    }
    return sorted(
        rows,
        key=lambda item: (
            action_order.get(str(item.get("action_class")), 99),
            -_int(item.get("label_count")),
            -float(item.get("avg_return_delta_vs_all_bps") or 0.0),
        ),
    )


def build_regime_cell_review_report(
    *,
    best_horizon: pd.DataFrame,
    by_horizon: pd.DataFrame,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    review_rows = _review_rows(best_horizon, by_horizon)
    action_counts = pd.Series([row.get("action_class") for row in review_rows], dtype="object").value_counts()
    report = {
        "report_type": "regime_cell_review",
        "generated_at": _utc_now(),
        "source_dir": str(source_dir) if source_dir is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "target_groups": list(TARGET_GROUPS),
        "reviewed_cell_count": int(len(review_rows)),
        "action_counts": {str(key): int(value) for key, value in action_counts.to_dict().items()},
        "proposal_status": "RESEARCH_REVIEW_READY",
        "implementation_note": (
            "These classifications are research proposals only. They should become runtime behavior only after "
            "counterfactual replay and fresh-forward monitoring confirm that the modifiers help."
        ),
        "cells": review_rows,
        "recommended_next_actions": [
            "Manually review HOLD_TIME_SPECIAL and DOWNGRADE_OR_AVOID cells first because they change behavior the most.",
            "Translate stable cells into a versioned regime-parameter artifact only after counterfactual replay.",
            "Keep PCR-specific 5-factor cells out of runtime proposals until PCR coverage is materially higher.",
        ],
    }
    return _sanitize(report)


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


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _flat_rows_for_csv(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "cell",
        "group_name",
        "action_class",
        "proposal_confidence",
        "gamma_regime",
        "volatility_regime",
        "direction",
        "macro_risk_bucket",
        "signal_count",
        "label_count",
        "best_horizon",
        "hit_rate",
        "avg_signed_return_bps",
        "hit_rate_delta_vs_all",
        "avg_return_delta_vs_all_bps",
        "score_adjustment_research",
        "threshold_adjustment_research",
        "size_multiplier_research",
        "allow_trade_research",
        "hold_time_hint",
        "rationale",
    ]
    return [{field: row.get(field) for field in fields} for row in cells]


def render_regime_cell_review_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Regime Cell Review",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Source dir: {report.get('source_dir')}",
        f"- Reviewed cells: {report.get('reviewed_cell_count')}",
        f"- Proposal status: {report.get('proposal_status')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Scope",
        "",
        "This review uses only reliable 3-factor and 4-factor cells:",
    ]
    for group in report.get("target_groups", []):
        lines.append(f"- `{group}`")

    lines.extend(["", "PCR-specific 5-factor cells are intentionally excluded until PCR coverage is materially higher.", ""])
    lines.extend(["## Action Counts", "", "| Action | Count |", "| --- | ---: |"])
    for action, count in (report.get("action_counts") or {}).items():
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Cell Proposals",
            "",
            "| Action | Confidence | Cell | Best Horizon | Labels | Hit | Avg Bps | Delta Bps | Research Adjustment | Hold Hint |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("cells", []):
        adjustment = (
            f"score {row.get('score_adjustment_research')}, "
            f"threshold {row.get('threshold_adjustment_research')}, "
            f"size {row.get('size_multiplier_research')}"
        )
        lines.append(
            f"| {row.get('action_class')} | {row.get('proposal_confidence')} | {row.get('cell')} | "
            f"{row.get('best_horizon')} | {row.get('label_count')} | {row.get('hit_rate')} | "
            f"{row.get('avg_signed_return_bps')} | {row.get('avg_return_delta_vs_all_bps')} | "
            f"{adjustment} | {row.get('hold_time_hint')} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(report.get("implementation_note") or "")
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in report.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def write_regime_cell_review_report(
    *,
    best_horizon_csv: str | Path = DEFAULT_REGIME_OUTCOME_TABLE_DIR / REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME,
    by_horizon_csv: str | Path = DEFAULT_REGIME_OUTCOME_TABLE_DIR / REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME,
    output_dir: str | Path = DEFAULT_REGIME_CELL_REVIEW_DIR,
    documentation_dir: str | Path | None = DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    report_name: str = "regime_cell_review",
    write_latest: bool = True,
) -> dict[str, Any]:
    best_horizon_csv = Path(best_horizon_csv)
    by_horizon_csv = Path(by_horizon_csv)
    best_horizon = pd.read_csv(best_horizon_csv) if best_horizon_csv.exists() else pd.DataFrame()
    by_horizon = pd.read_csv(by_horizon_csv) if by_horizon_csv.exists() else pd.DataFrame()
    report = build_regime_cell_review_report(
        best_horizon=best_horizon,
        by_horizon=by_horizon,
        source_dir=best_horizon_csv.parent,
    )

    output_dir = Path(output_dir)
    json_path = output_dir / f"{report_name}.json"
    markdown_path = output_dir / f"{report_name}.md"
    csv_path = output_dir / f"{report_name}_cells.csv"
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_regime_cell_review_markdown(report))
    _atomic_write_csv(pd.DataFrame(_flat_rows_for_csv(report.get("cells", []))), csv_path)

    artifact = {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "csv_path": str(csv_path),
    }
    if write_latest:
        latest_json = output_dir / LATEST_REVIEW_JSON_FILENAME
        latest_markdown = output_dir / LATEST_REVIEW_MARKDOWN_FILENAME
        latest_csv = output_dir / LATEST_REVIEW_CSV_FILENAME
        _atomic_write_text(latest_json, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown, render_regime_cell_review_markdown(report))
        _atomic_write_csv(pd.DataFrame(_flat_rows_for_csv(report.get("cells", []))), latest_csv)
        artifact.update(
            {
                "latest_json_path": str(latest_json),
                "latest_markdown_path": str(latest_markdown),
                "latest_csv_path": str(latest_csv),
            }
        )

    if documentation_dir is not None:
        documentation_dir = Path(documentation_dir)
        docs_latest = documentation_dir / LATEST_REVIEW_MARKDOWN_FILENAME
        _atomic_write_text(docs_latest, render_regime_cell_review_markdown(report))
        artifact["documentation_markdown_path"] = str(docs_latest)

    return artifact
