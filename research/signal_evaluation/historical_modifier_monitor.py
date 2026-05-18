"""Monitor whether live historical modifiers helped or hurt realized signals.

This report is attribution-by-alignment, not a full counterfactual replay. It
uses the modifier direction recorded at signal time and checks whether realized
15m/30m/60m signed outcomes agreed with that modifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.daily_research_report import (
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
)
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary
from utils.timestamp_helpers import coerce_timestamp_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_MODIFIER_MONITOR_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "historical_modifier_monitor"
)

HISTORICAL_MODIFIER_MONITOR_JSON_FILENAME = "latest_historical_modifier_monitor.json"
HISTORICAL_MODIFIER_MONITOR_MARKDOWN_FILENAME = "latest_historical_modifier_monitor.md"
HISTORICAL_MODIFIER_MONITOR_COMPONENT_CSV_FILENAME = "latest_historical_modifier_monitor_components.csv"
HISTORICAL_MODIFIER_MONITOR_REASON_CSV_FILENAME = "latest_historical_modifier_monitor_reasons.csv"
HISTORICAL_MODIFIER_MONITOR_BUCKET_CSV_FILENAME = "latest_historical_modifier_monitor_buckets.csv"
HISTORICAL_MODIFIER_MONITOR_ROW_CSV_FILENAME = "latest_historical_modifier_monitor_rows.csv"

HORIZONS = (
    ("15m", "signed_return_15m_bps", "correct_15m"),
    ("30m", "signed_return_30m_bps", "correct_30m"),
    ("60m", "signed_return_60m_bps", "correct_60m"),
)

NUMERIC_COLUMNS = (
    "historical_context_score_adjustment",
    "historical_context_probability_adjustment",
    "historical_context_trade_strength_threshold_adjustment",
    "historical_context_composite_threshold_adjustment",
    "historical_context_size_multiplier",
    "historical_interaction_count",
    "historical_interaction_score_adjustment",
    "historical_interaction_probability_adjustment",
    "trade_strength",
    "hybrid_move_probability",
    "move_probability",
    "signal_confidence_score",
)


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


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


def default_historical_modifier_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _text_nonempty(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return series.notna() & ~text.str.upper().isin({"", "NAN", "NA", "N/A", "NONE", "NULL", "<NA>"})


def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    raw = apply_quality_label_view(frame if frame is not None else pd.DataFrame(), fallback_to_legacy=True).copy()
    if raw.empty:
        return raw
    if "signal_timestamp" in raw.columns:
        raw["signal_timestamp"] = coerce_timestamp_series(raw["signal_timestamp"])
        raw = raw.sort_values("signal_timestamp", kind="mergesort")
    if "direction" in raw.columns:
        direction = raw["direction"].astype(str).str.upper().str.strip()
        directional = direction.isin({"CALL", "PUT"})
        if directional.any():
            raw = raw.loc[directional].copy()
    for column in set(NUMERIC_COLUMNS) | {item[1] for item in HORIZONS} | {item[2] for item in HORIZONS}:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw.reset_index(drop=True)


def _add_modifier_flags(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    index = working.index
    score = pd.to_numeric(working.get("historical_context_score_adjustment", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    probability = pd.to_numeric(
        working.get("historical_context_probability_adjustment", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    threshold = pd.to_numeric(
        working.get("historical_context_trade_strength_threshold_adjustment", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    composite_threshold = pd.to_numeric(
        working.get("historical_context_composite_threshold_adjustment", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    size_multiplier = pd.to_numeric(
        working.get("historical_context_size_multiplier", pd.Series(1.0, index=index)),
        errors="coerce",
    ).fillna(1.0)
    interaction_count = pd.to_numeric(
        working.get("historical_interaction_count", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    interaction_score = pd.to_numeric(
        working.get("historical_interaction_score_adjustment", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    interaction_probability = pd.to_numeric(
        working.get("historical_interaction_probability_adjustment", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    override = (
        _text_nonempty(working.get("historical_context_direction_override", pd.Series(index=index, dtype="object")))
        if "historical_context_direction_override" in working.columns
        else pd.Series(False, index=index, dtype=bool)
    )

    working["_score_nonzero"] = score.abs() > 1e-9
    working["_probability_nonzero"] = probability.abs() > 1e-9
    working["_threshold_nonzero"] = threshold.abs() > 1e-9
    working["_composite_threshold_nonzero"] = composite_threshold.abs() > 1e-9
    working["_size_reduced"] = size_multiplier < 0.999
    working["_direction_override_used"] = override.astype(bool)
    working["_interaction_nonzero"] = (
        (interaction_count > 0)
        | (interaction_score.abs() > 1e-9)
        | (interaction_probability.abs() > 1e-9)
    )
    working["_modifier_active"] = (
        working["_score_nonzero"]
        | working["_probability_nonzero"]
        | working["_threshold_nonzero"]
        | working["_composite_threshold_nonzero"]
        | working["_size_reduced"]
        | working["_direction_override_used"]
        | working["_interaction_nonzero"]
    )

    net = (
        score
        + (probability * 100.0)
        - threshold
        - composite_threshold
        - ((1.0 - size_multiplier.clip(upper=1.0)) * 10.0)
        + working["_direction_override_used"].astype(float) * 2.0
    )
    working["_historical_modifier_net_score"] = net.round(4)
    working["_historical_modifier_action"] = "NO_MODIFIER"
    working.loc[working["_modifier_active"] & (net > 0.1), "_historical_modifier_action"] = "SUPPORTIVE"
    working.loc[working["_modifier_active"] & (net < -0.1), "_historical_modifier_action"] = "RESTRICTIVE"
    mixed = working["_modifier_active"] & (net.abs() <= 0.1)
    working.loc[mixed, "_historical_modifier_action"] = "MIXED_NEUTRAL"
    return working


def _attribution(action: str, signed_return_bps: Any) -> str:
    ret = _round_or_none(signed_return_bps, 10)
    if ret is None:
        return "UNLABELED"
    if abs(ret) <= 1e-9:
        return "NEUTRAL"
    if action == "SUPPORTIVE":
        return "HELPED" if ret > 0 else "HURT"
    if action == "RESTRICTIVE":
        return "HELPED" if ret < 0 else "HURT"
    if action == "MIXED_NEUTRAL":
        return "MIXED"
    return "NO_MODIFIER"


def _add_attribution_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    action = working.get("_historical_modifier_action", pd.Series("NO_MODIFIER", index=working.index)).astype(str)
    for horizon, return_col, _hit_col in HORIZONS:
        returns = pd.to_numeric(working.get(return_col, pd.Series(index=working.index, dtype=float)), errors="coerce")
        working[f"_historical_attribution_{horizon}"] = [
            _attribution(item_action, item_return)
            for item_action, item_return in zip(action, returns, strict=False)
        ]
    return working


def _metric_bundle(frame: pd.DataFrame, *, horizon: str, return_col: str, hit_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "label_count": 0,
            "hit_rate": None,
            "avg_signed_return_bps": None,
            "helped_count": 0,
            "hurt_count": 0,
            "mixed_count": 0,
            "neutral_count": 0,
            "help_rate": None,
            "hurt_rate": None,
        }
    returns = pd.to_numeric(frame.get(return_col, pd.Series(index=frame.index)), errors="coerce")
    hits = pd.to_numeric(frame.get(hit_col, pd.Series(index=frame.index)), errors="coerce")
    valid = returns.notna()
    if hit_col in frame.columns:
        valid = valid | hits.notna()
    labeled = frame.loc[valid].copy()
    attribution = labeled.get(f"_historical_attribution_{horizon}", pd.Series(index=labeled.index, dtype="object")).astype(str)
    helped = int((attribution == "HELPED").sum())
    hurt = int((attribution == "HURT").sum())
    mixed = int((attribution == "MIXED").sum())
    neutral = int((attribution == "NEUTRAL").sum())
    denom = helped + hurt
    return {
        "row_count": int(len(frame)),
        "label_count": int(len(labeled)),
        "hit_rate": _round_or_none(hits.loc[valid].mean(), 4) if hits.loc[valid].notna().any() else None,
        "avg_signed_return_bps": _round_or_none(returns.loc[valid].mean(), 4) if returns.loc[valid].notna().any() else None,
        "helped_count": helped,
        "hurt_count": hurt,
        "mixed_count": mixed,
        "neutral_count": neutral,
        "help_rate": _round_or_none(helped / denom, 4) if denom else None,
        "hurt_rate": _round_or_none(hurt / denom, 4) if denom else None,
    }


def _build_horizon_summary(frame: pd.DataFrame) -> dict[str, Any]:
    modified = frame.loc[frame.get("_modifier_active", pd.Series(False, index=frame.index)).astype(bool)].copy()
    return {
        horizon: _metric_bundle(modified, horizon=horizon, return_col=return_col, hit_col=hit_col)
        for horizon, return_col, hit_col in HORIZONS
    }


def _component_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    components = {
        "score_adjustment_nonzero": "_score_nonzero",
        "probability_adjustment_nonzero": "_probability_nonzero",
        "direction_override_used": "_direction_override_used",
        "interaction_nonzero": "_interaction_nonzero",
        "size_reduced": "_size_reduced",
        "threshold_changed": "_threshold_nonzero",
    }
    rows: list[dict[str, Any]] = []
    for name, mask_col in components.items():
        if mask_col not in frame.columns:
            continue
        subset = frame.loc[frame[mask_col].astype(bool)].copy()
        metrics = _metric_bundle(subset, horizon="60m", return_col="signed_return_60m_bps", hit_col="correct_60m")
        rows.append({"component": name, **metrics})
    return rows


def _split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    tokens = []
    for token in str(value).replace(",", "|").split("|"):
        cleaned = token.strip()
        if cleaned and cleaned.upper() not in {"NAN", "NONE", "NULL", "NA", "N/A"}:
            tokens.append(cleaned)
    return tokens


def _reason_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records = []
    for idx, row in frame.iterrows():
        for source, column in (
            ("context", "historical_context_reasons"),
            ("interaction", "historical_interaction_reasons"),
        ):
            for reason in _split_tokens(row.get(column)):
                records.append({"_idx": idx, "reason_source": source, "reason": reason})
    if not records:
        return []
    exploded = pd.DataFrame(records)
    rows: list[dict[str, Any]] = []
    for (source, reason), group in exploded.groupby(["reason_source", "reason"], dropna=False):
        subset = frame.loc[group["_idx"].unique()].copy()
        rows.append(
            {
                "reason_source": str(source),
                "reason": str(reason),
                **_metric_bundle(subset, horizon="60m", return_col="signed_return_60m_bps", hit_col="correct_60m"),
            }
        )
    return sorted(rows, key=lambda item: (-int(item["row_count"]), item["reason_source"], item["reason"]))


def _parse_bucket_state(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        if value is None or pd.isna(value):
            return {}
    except Exception:
        pass
    try:
        payload = json.loads(str(value))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bucket_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "historical_interaction_bucket_state" not in frame.columns:
        return []
    records = []
    for idx, value in frame["historical_interaction_bucket_state"].items():
        payload = _parse_bucket_state(value)
        for field in ("expiry_bucket", "pcr_oi_bucket", "india_vix_bucket", "trend_20d_bucket", "weekday", "pcr_basis"):
            field_value = payload.get(field)
            if field_value not in (None, ""):
                records.append({"_idx": idx, "bucket_field": field, "bucket_value": str(field_value)})
    if not records:
        return []
    exploded = pd.DataFrame(records)
    rows: list[dict[str, Any]] = []
    for (field, value), group in exploded.groupby(["bucket_field", "bucket_value"], dropna=False):
        subset = frame.loc[group["_idx"].unique()].copy()
        rows.append(
            {
                "bucket_field": str(field),
                "bucket_value": str(value),
                **_metric_bundle(subset, horizon="60m", return_col="signed_return_60m_bps", hit_col="correct_60m"),
            }
        )
    return sorted(rows, key=lambda item: (item["bucket_field"], -int(item["row_count"]), item["bucket_value"]))


def _recent_modified_rows(frame: pd.DataFrame, *, limit: int = 50) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    columns = [
        "signal_id",
        "signal_timestamp",
        "direction",
        "trade_status",
        "historical_context_score_adjustment",
        "historical_context_probability_adjustment",
        "historical_context_trade_strength_threshold_adjustment",
        "historical_context_size_multiplier",
        "historical_context_direction_override",
        "historical_interaction_count",
        "historical_context_reasons",
        "historical_interaction_reasons",
        "signed_return_15m_bps",
        "signed_return_30m_bps",
        "signed_return_60m_bps",
        "_historical_modifier_action",
        "_historical_modifier_net_score",
        "_historical_attribution_15m",
        "_historical_attribution_30m",
        "_historical_attribution_60m",
    ]
    available = [column for column in columns if column in frame.columns]
    recent = frame.sort_values("signal_timestamp", ascending=False, kind="mergesort") if "signal_timestamp" in frame.columns else frame
    return recent.loc[:, available].head(limit).to_dict(orient="records")


def _monitor_status(summary_60m: dict[str, Any], *, min_label_sample: int) -> str:
    label_count = int(summary_60m.get("label_count") or 0)
    help_rate = summary_60m.get("help_rate")
    avg_return = summary_60m.get("avg_signed_return_bps")
    if label_count <= 0:
        return "NO_EVIDENCE"
    if label_count < int(min_label_sample):
        return "ACCUMULATING_EVIDENCE"
    if help_rate is not None and help_rate >= 0.60 and (avg_return is None or avg_return >= 0):
        return "HISTORICAL_LAYER_HELPFUL"
    if help_rate is not None and help_rate <= 0.40:
        return "HISTORICAL_LAYER_HURTING"
    return "MIXED_EVIDENCE"


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("monitor_status")
    summary_60m = (report.get("horizon_summary") or {}).get("60m", {}) or {}
    actions: list[str] = []
    if status in {"NO_EVIDENCE", "ACCUMULATING_EVIDENCE"}:
        actions.append(
            f"Keep collecting labeled outcomes; modified 60m label count is {summary_60m.get('label_count')}."
        )
    if status == "HISTORICAL_LAYER_HURTING":
        actions.append("Review negative help-rate components before increasing historical modifier weight.")
    if status == "HISTORICAL_LAYER_HELPFUL":
        actions.append("Use reason and bucket tables to identify which historical modifiers deserve stronger weighting.")
    if report.get("direction_override_count", 0) > 0:
        actions.append("Review direction override rows manually before expanding override authority.")
    if not actions:
        actions.append("Continue monitoring; evidence is mixed and not yet strong enough for new parameter changes.")
    return actions


def build_historical_modifier_monitor_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    min_label_sample: int = 30,
    recent_row_limit: int = 50,
) -> dict[str, Any]:
    raw = frame if frame is not None else pd.DataFrame()
    working = _add_attribution_columns(_add_modifier_flags(_prepare_frame(raw)))
    modified = working.loc[working.get("_modifier_active", pd.Series(False, index=working.index)).astype(bool)].copy()
    horizon_summary = _build_horizon_summary(working)
    report = {
        "report_type": "historical_modifier_monitor",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "assessment_basis": "alignment_proxy_not_counterfactual_replay",
        "row_count": int(len(raw)),
        "directional_row_count": int(len(working)),
        "modifier_row_count": int(len(modified)),
        "score_adjustment_nonzero_count": int(working.get("_score_nonzero", pd.Series(dtype=bool)).sum()),
        "probability_adjustment_nonzero_count": int(working.get("_probability_nonzero", pd.Series(dtype=bool)).sum()),
        "direction_override_count": int(working.get("_direction_override_used", pd.Series(dtype=bool)).sum()),
        "interaction_nonzero_count": int(working.get("_interaction_nonzero", pd.Series(dtype=bool)).sum()),
        "size_reduction_count": int(working.get("_size_reduced", pd.Series(dtype=bool)).sum()),
        "label_quality": label_quality_summary(raw),
        "horizon_summary": horizon_summary,
        "component_summary": _component_rows(working),
        "reason_summary": _reason_rows(modified),
        "bucket_summary": _bucket_rows(modified),
        "recent_modified_rows": _recent_modified_rows(modified, limit=recent_row_limit),
    }
    report["monitor_status"] = _monitor_status(
        horizon_summary.get("60m", {}) or {},
        min_label_sample=min_label_sample,
    )
    report["recommended_next_actions"] = _recommended_actions(report)
    return _sanitize_value(report)


def render_historical_modifier_monitor_markdown(report: dict[str, Any]) -> str:
    summary = report.get("horizon_summary", {}) or {}
    lines = [
        "# Historical Modifier Monitor",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Monitor status: `{report.get('monitor_status')}`",
        f"- Assessment basis: `{report.get('assessment_basis')}`",
        f"- Rows: {report.get('row_count')}",
        f"- Directional rows: {report.get('directional_row_count')}",
        f"- Modified rows: {report.get('modifier_row_count')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Outcome Attribution",
        "",
        "| Horizon | Labels | Helped | Hurt | Help Rate | Avg Signed Return (bps) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon, _return_col, _hit_col in HORIZONS:
        row = summary.get(horizon, {}) or {}
        lines.append(
            f"| {horizon} | {row.get('label_count')} | {row.get('helped_count')} | {row.get('hurt_count')} | "
            f"{row.get('help_rate')} | {row.get('avg_signed_return_bps')} |"
        )

    lines.extend(["", "## Modifier Components", "", "| Component | Rows | Labels | Help Rate | Avg 60m Return |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in report.get("component_summary", []) or []:
        lines.append(
            f"| `{row.get('component')}` | {row.get('row_count')} | {row.get('label_count')} | "
            f"{row.get('help_rate')} | {row.get('avg_signed_return_bps')} |"
        )

    lines.extend(["", "## Top Reasons", "", "| Source | Reason | Rows | Help Rate | Avg 60m Return |", "| --- | --- | ---: | ---: | ---: |"])
    for row in (report.get("reason_summary", []) or [])[:12]:
        lines.append(
            f"| `{row.get('reason_source')}` | `{row.get('reason')}` | {row.get('row_count')} | "
            f"{row.get('help_rate')} | {row.get('avg_signed_return_bps')} |"
        )

    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append(
        "*This monitor is research-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*"
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "component_csv_path": output / f"{stem}_components.csv",
        "reason_csv_path": output / f"{stem}_reasons.csv",
        "bucket_csv_path": output / f"{stem}_buckets.csv",
        "row_csv_path": output / f"{stem}_rows.csv",
        "latest_json_path": output / HISTORICAL_MODIFIER_MONITOR_JSON_FILENAME,
        "latest_markdown_path": output / HISTORICAL_MODIFIER_MONITOR_MARKDOWN_FILENAME,
        "latest_component_csv_path": output / HISTORICAL_MODIFIER_MONITOR_COMPONENT_CSV_FILENAME,
        "latest_reason_csv_path": output / HISTORICAL_MODIFIER_MONITOR_REASON_CSV_FILENAME,
        "latest_bucket_csv_path": output / HISTORICAL_MODIFIER_MONITOR_BUCKET_CSV_FILENAME,
        "latest_row_csv_path": output / HISTORICAL_MODIFIER_MONITOR_ROW_CSV_FILENAME,
    }


def write_historical_modifier_monitor_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    min_label_sample: int = 30,
    recent_row_limit: int = 50,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    report = build_historical_modifier_monitor_report(
        frame,
        dataset_path=dataset_path,
        min_label_sample=min_label_sample,
        recent_row_limit=recent_row_limit,
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_HISTORICAL_MODIFIER_MONITOR_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "historical_modifier_monitor"
    paths = _artifact_paths(output, stem)
    markdown = render_historical_modifier_monitor_markdown(report)
    components = pd.DataFrame(report.get("component_summary", []) or [])
    reasons = pd.DataFrame(report.get("reason_summary", []) or [])
    buckets = pd.DataFrame(report.get("bucket_summary", []) or [])
    rows = pd.DataFrame(report.get("recent_modified_rows", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(components, paths["component_csv_path"])
    _atomic_write_csv(reasons, paths["reason_csv_path"])
    _atomic_write_csv(buckets, paths["bucket_csv_path"])
    _atomic_write_csv(rows, paths["row_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(components, paths["latest_component_csv_path"])
        _atomic_write_csv(reasons, paths["latest_reason_csv_path"])
        _atomic_write_csv(buckets, paths["latest_bucket_csv_path"])
        _atomic_write_csv(rows, paths["latest_row_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_historical_modifier_monitor_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    path = Path(dataset_path) if dataset_path is not None else default_historical_modifier_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_historical_modifier_monitor_report(frame, dataset_path=path, **kwargs)
