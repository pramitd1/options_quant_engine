"""Empirical regime outcome tables for signal-evaluation research.

The tables produced here are research-only. They summarize realized outcomes
by market regime combinations so threshold, sizing, and hold-time ideas can be
formed from evidence before any runtime wiring is considered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from research.signal_evaluation.confidence import outcome_confidence_fields
from research.signal_evaluation.daily_research_report import (
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
)
from research.signal_evaluation.label_quality import apply_quality_label_view
from utils.pcr import normalize_pcr_bucket_for_reporting
from utils.timestamp_helpers import coerce_timestamp_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIME_OUTCOME_TABLE_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "regime_outcome_tables"
)

REGIME_OUTCOME_JSON_FILENAME = "latest_regime_outcome_tables.json"
REGIME_OUTCOME_MARKDOWN_FILENAME = "latest_regime_outcome_tables.md"
REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME = "latest_regime_outcome_tables_by_horizon.csv"
REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME = "latest_regime_outcome_tables_best_horizon.csv"

HORIZONS: tuple[tuple[str, str, str], ...] = (
    ("5m", "signed_return_5m_bps", "correct_5m"),
    ("15m", "signed_return_15m_bps", "correct_15m"),
    ("30m", "signed_return_30m_bps", "correct_30m"),
    ("60m", "signed_return_60m_bps", "correct_60m"),
    ("120m", "signed_return_120m_bps", "correct_120m"),
    ("session_close", "signed_return_session_close_bps", "correct_session_close"),
)

DEFAULT_GROUP_SPECS: tuple[tuple[str, ...], ...] = (
    ("gamma_regime",),
    ("volatility_regime",),
    ("macro_risk_bucket",),
    ("gamma_regime", "volatility_regime"),
    ("gamma_regime", "volatility_regime", "direction"),
    ("gamma_regime", "volatility_regime", "direction", "macro_risk_bucket"),
    ("gamma_regime", "volatility_regime", "direction", "macro_risk_bucket", "pcr_bucket"),
)

_MISSING_TEXT = {"", "NA", "N/A", "NAN", "NONE", "NULL", "<NA>"}


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


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


def _normalize_bucket(value: Any, *, default: str = "UNKNOWN") -> str:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text.upper() in _MISSING_TEXT:
        return default
    return text.upper()


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        if value is None or pd.isna(value):
            return {}
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.upper() in _MISSING_TEXT:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_json_key(frame: pd.DataFrame, column: str, key: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[column].map(lambda value: _parse_json_object(value).get(key))


def _coalesce_text(*series: pd.Series) -> pd.Series:
    if not series:
        return pd.Series(dtype="object")
    out = pd.Series([None] * len(series[0]), index=series[0].index, dtype="object")
    missing = pd.Series(True, index=series[0].index, dtype=bool)
    for item in series:
        normalized = item.map(lambda value: _normalize_bucket(value, default=""))
        available = ~normalized.isin({"", "UNKNOWN"})
        out = out.where(~(missing & available), normalized)
        missing = missing & ~available
    return out.fillna("UNKNOWN").map(lambda value: _normalize_bucket(value))


def default_regime_outcome_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    working = apply_quality_label_view(frame if frame is not None else pd.DataFrame(), fallback_to_legacy=True).copy()
    if working.empty:
        return working

    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = coerce_timestamp_series(working["signal_timestamp"])
        working = working.sort_values("signal_timestamp", kind="mergesort")

    if "direction" in working.columns:
        direction = working["direction"].map(lambda value: _normalize_bucket(value))
        working = working.loc[direction.isin({"CALL", "PUT"})].copy()
        working["direction"] = direction.loc[working.index]

    if "volatility_regime" not in working.columns and "vol_regime" in working.columns:
        working["volatility_regime"] = working["vol_regime"]

    for column in ("gamma_regime", "volatility_regime", "macro_regime", "global_risk_state"):
        if column in working.columns:
            working[column] = working[column].map(lambda value: _normalize_bucket(value))

    if "global_risk_state" in working.columns:
        macro_risk = working["global_risk_state"]
    elif "macro_regime" in working.columns:
        macro_risk = working["macro_regime"]
    else:
        macro_risk = pd.Series("UNKNOWN", index=working.index, dtype="object")
    working["macro_risk_bucket"] = macro_risk.map(lambda value: _normalize_bucket(value))

    canonical_pcr = (
        working["pcr_bucket"]
        if "pcr_bucket" in working.columns
        else pd.Series([None] * len(working), index=working.index, dtype="object")
    )
    statistical_pcr = _extract_json_key(working, "statistical_context_bucket_state", "pcr_oi_bucket")
    historical_pcr = _extract_json_key(working, "historical_interaction_bucket_state", "pcr_oi_bucket")
    historical_state = (
        working["historical_pcr_state"]
        if "historical_pcr_state" in working.columns
        else pd.Series([None] * len(working), index=working.index, dtype="object")
    )
    working["pcr_bucket"] = _coalesce_text(canonical_pcr, statistical_pcr, historical_pcr, historical_state).map(
        normalize_pcr_bucket_for_reporting
    )

    for _horizon, return_col, hit_col in HORIZONS:
        if return_col in working.columns:
            working[return_col] = pd.to_numeric(working[return_col], errors="coerce")
        if hit_col in working.columns:
            working[hit_col] = pd.to_numeric(working[hit_col], errors="coerce")

    for column in (
        "runtime_composite_score",
        "composite_signal_score",
        "trade_strength",
        "hybrid_move_probability",
        "tradeability_score",
        "selected_option_ba_spread_pct",
        "selected_option_volume",
        "selected_option_open_interest",
    ):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    return working.reset_index(drop=True)


def _metric_row_for_horizon(
    group: pd.DataFrame,
    *,
    horizon: str,
    return_col: str,
    hit_col: str,
    min_label_sample: int,
    strong_label_sample: int,
) -> dict[str, Any]:
    hit = pd.to_numeric(group.get(hit_col, pd.Series(index=group.index, dtype=float)), errors="coerce")
    returns = pd.to_numeric(group.get(return_col, pd.Series(index=group.index, dtype=float)), errors="coerce")
    valid = hit.notna() | returns.notna()
    label_count = int(valid.sum())
    hit_labeled = hit.loc[valid].dropna()
    return_labeled = returns.loc[valid].dropna()
    confidence = outcome_confidence_fields(
        hit_labeled,
        return_labeled,
        sample_count=label_count,
        min_sample=min_label_sample,
        strong_sample=strong_label_sample,
    )
    return {
        "horizon": horizon,
        "label_count": label_count,
        "hit_rate": _round_or_none(hit_labeled.mean(), 4) if not hit_labeled.empty else None,
        "avg_signed_return_bps": _round_or_none(return_labeled.mean(), 4) if not return_labeled.empty else None,
        "median_signed_return_bps": _round_or_none(return_labeled.median(), 4) if not return_labeled.empty else None,
        **confidence,
    }


def _baseline_by_horizon(
    frame: pd.DataFrame,
    *,
    min_label_sample: int,
    strong_label_sample: int,
) -> dict[str, dict[str, Any]]:
    return {
        horizon: _metric_row_for_horizon(
            frame,
            horizon=horizon,
            return_col=return_col,
            hit_col=hit_col,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
        )
        for horizon, return_col, hit_col in HORIZONS
    }


def _edge_hint(row: dict[str, Any]) -> str:
    quality = str(row.get("sample_quality") or "NO_EVIDENCE")
    hit_rate = row.get("hit_rate")
    avg_return = row.get("avg_signed_return_bps")
    if quality in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}:
        return "COLLECT_MORE_DATA"
    if hit_rate is None or avg_return is None:
        return "COLLECT_MORE_DATA"
    hit = float(hit_rate)
    ret = float(avg_return)
    if hit >= 0.55 and ret >= 5.0:
        return "FAVORABLE"
    if hit <= 0.48 or ret <= -5.0:
        return "UNFAVORABLE"
    return "MIXED_OR_NEUTRAL"


def _build_group_rows(
    frame: pd.DataFrame,
    *,
    fields: tuple[str, ...],
    baseline: dict[str, dict[str, Any]],
    min_label_sample: int,
    strong_label_sample: int,
) -> list[dict[str, Any]]:
    missing = [field for field in fields if field not in frame.columns]
    if missing or frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    groupby_key: str | list[str] = fields[0] if len(fields) == 1 else list(fields)
    for key, group in frame.groupby(groupby_key, dropna=False, observed=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base_row = {
            "group_name": "+".join(fields),
            "group_depth": len(fields),
            "signal_count": int(len(group)),
        }
        for field, value in zip(fields, key_values):
            base_row[field] = _normalize_bucket(value)

        for horizon, return_col, hit_col in HORIZONS:
            metrics = _metric_row_for_horizon(
                group,
                horizon=horizon,
                return_col=return_col,
                hit_col=hit_col,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            baseline_metrics = baseline.get(horizon, {})
            row = dict(base_row)
            row.update(metrics)
            row["baseline_hit_rate"] = baseline_metrics.get("hit_rate")
            row["baseline_avg_signed_return_bps"] = baseline_metrics.get("avg_signed_return_bps")
            row["hit_rate_delta_vs_all"] = _round_or_none(
                (float(metrics["hit_rate"]) - float(baseline_metrics["hit_rate"]))
                if metrics.get("hit_rate") is not None and baseline_metrics.get("hit_rate") is not None
                else None,
                4,
            )
            row["avg_return_delta_vs_all_bps"] = _round_or_none(
                (float(metrics["avg_signed_return_bps"]) - float(baseline_metrics["avg_signed_return_bps"]))
                if metrics.get("avg_signed_return_bps") is not None
                and baseline_metrics.get("avg_signed_return_bps") is not None
                else None,
                4,
            )
            row["edge_hint"] = _edge_hint(row)
            rows.append(row)
    return rows


def _best_horizon_rows(by_horizon: pd.DataFrame) -> list[dict[str, Any]]:
    if by_horizon.empty:
        return []
    rows: list[dict[str, Any]] = []
    key_cols = [
        "group_name",
        "group_depth",
        "gamma_regime",
        "volatility_regime",
        "direction",
        "macro_risk_bucket",
        "pcr_bucket",
    ]
    present_keys = [column for column in key_cols if column in by_horizon.columns]
    for key, group in by_horizon.groupby(present_keys, dropna=False, observed=False):
        key_values = key if isinstance(key, tuple) else (key,)
        candidates = group.copy()
        candidates["_quality_rank"] = candidates["sample_quality"].map(
            {
                "RELIABLE": 3,
                "LOW_CONFIDENCE": 2,
                "INSUFFICIENT_EVIDENCE": 1,
                "NO_EVIDENCE": 0,
            }
        ).fillna(0)
        usable = candidates.loc[candidates["_quality_rank"] >= 1].copy()
        if usable.empty:
            best = candidates.sort_values(["label_count"], ascending=[False]).head(1)
            basis = "largest_available_sample"
        else:
            best = usable.sort_values(
                ["avg_signed_return_bps", "hit_rate", "label_count"],
                ascending=[False, False, False],
                na_position="last",
            ).head(1)
            basis = "max_avg_signed_return_bps"
        if best.empty:
            continue
        item = best.iloc[0].to_dict()
        row: dict[str, Any] = {}
        for field, value in zip(present_keys, key_values):
            if field == "group_depth":
                row[field] = int(value or 0)
            elif field == "group_name":
                row[field] = str(value)
            else:
                row[field] = _normalize_bucket(value)
        row.update(
            {
                "signal_count": int(item.get("signal_count") or 0),
                "best_horizon": item.get("horizon"),
                "best_horizon_basis": basis,
                "label_count": int(item.get("label_count") or 0),
                "sample_quality": item.get("sample_quality"),
                "hit_rate": _round_or_none(item.get("hit_rate"), 4),
                "hit_rate_ci_low": _round_or_none(item.get("hit_rate_ci_low"), 4),
                "hit_rate_ci_high": _round_or_none(item.get("hit_rate_ci_high"), 4),
                "avg_signed_return_bps": _round_or_none(item.get("avg_signed_return_bps"), 4),
                "return_ci_low_bps": _round_or_none(item.get("return_ci_low_bps"), 4),
                "return_ci_high_bps": _round_or_none(item.get("return_ci_high_bps"), 4),
                "hit_rate_delta_vs_all": _round_or_none(item.get("hit_rate_delta_vs_all"), 4),
                "avg_return_delta_vs_all_bps": _round_or_none(item.get("avg_return_delta_vs_all_bps"), 4),
                "edge_hint": item.get("edge_hint"),
            }
        )
        rows.append(row)
    return rows


def _field_coverage(frame: pd.DataFrame, fields: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = int(len(frame))
    for field in fields:
        if field not in frame.columns:
            rows.append(
                {
                    "field": field,
                    "row_count": total,
                    "available_count": 0,
                    "coverage_ratio": 0.0,
                    "top_values": {},
                }
            )
            continue
        normalized = frame[field].map(lambda value: _normalize_bucket(value))
        available = normalized.ne("UNKNOWN")
        counts = normalized.loc[available].value_counts().head(8).to_dict()
        rows.append(
            {
                "field": field,
                "row_count": total,
                "available_count": int(available.sum()),
                "coverage_ratio": _round_or_none((float(available.sum()) / total) if total else 0.0, 4),
                "top_values": {str(key): int(value) for key, value in counts.items()},
            }
        )
    return rows


def _top_rows(rows: list[dict[str, Any]], *, limit: int, favorable: bool = True) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if favorable:
        selected = frame.loc[frame["edge_hint"].isin(["FAVORABLE", "MIXED_OR_NEUTRAL"])].copy()
        ascending = [False, False, False]
    else:
        selected = frame.loc[frame["edge_hint"].isin(["UNFAVORABLE"])].copy()
        ascending = [True, True, False]
    if selected.empty:
        return []
    if {"group_name", "pcr_bucket"}.issubset(selected.columns):
        pcr_unknown_duplicate = selected["group_name"].astype(str).str.contains("pcr_bucket", regex=False) & selected[
            "pcr_bucket"
        ].astype(str).str.upper().eq("UNKNOWN")
        selected = selected.loc[~pcr_unknown_duplicate].copy()
    if selected.empty:
        return []
    sort_cols = ["avg_signed_return_bps", "hit_rate", "label_count"]
    selected = selected.sort_values(sort_cols, ascending=ascending, na_position="last")
    return [_sanitize(row) for row in selected.head(limit).to_dict("records")]


def build_regime_outcome_table_report(
    frame: pd.DataFrame | None,
    *,
    dataset_path: str | Path | None = None,
    group_specs: Iterable[Iterable[str]] = DEFAULT_GROUP_SPECS,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    top_n: int = 15,
) -> dict[str, Any]:
    prepared = _prepare_frame(frame)
    baseline = _baseline_by_horizon(
        prepared,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    specs = [tuple(spec) for spec in group_specs]
    horizon_rows: list[dict[str, Any]] = []
    for fields in specs:
        horizon_rows.extend(
            _build_group_rows(
                prepared,
                fields=fields,
                baseline=baseline,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
        )
    best_rows = _best_horizon_rows(pd.DataFrame(horizon_rows))

    reliable_best = [row for row in best_rows if row.get("sample_quality") in {"RELIABLE", "LOW_CONFIDENCE"}]
    sparse_rows = [row for row in best_rows if row.get("sample_quality") in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}]
    report = {
        "report_type": "regime_outcome_tables",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": int(len(frame) if frame is not None else 0),
        "directional_row_count": int(len(prepared)),
        "group_specs": ["+".join(spec) for spec in specs],
        "horizons": [horizon for horizon, _return_col, _hit_col in HORIZONS],
        "min_label_sample": int(min_label_sample),
        "strong_label_sample": int(strong_label_sample),
        "baseline_by_horizon": baseline,
        "field_coverage": _field_coverage(
            prepared,
            ("gamma_regime", "volatility_regime", "macro_risk_bucket", "direction", "pcr_bucket"),
        ),
        "by_horizon_row_count": int(len(horizon_rows)),
        "best_horizon_row_count": int(len(best_rows)),
        "reliable_or_low_confidence_best_row_count": int(len(reliable_best)),
        "sparse_best_row_count": int(len(sparse_rows)),
        "top_favorable": _top_rows(reliable_best, limit=top_n, favorable=True),
        "top_unfavorable": _top_rows(reliable_best, limit=top_n, favorable=False),
        "recommended_next_actions": [
            "Inspect reliable favorable/unfavorable cells before proposing threshold, size, or hold-time changes.",
            "Treat sparse multi-factor cells as data-collection targets rather than tuning evidence.",
            "Build counterfactual pack/replay checks before wiring any regime table into runtime behavior.",
        ],
        "quality_note": (
            "The 60m horizon uses the quality-approved label view when available. "
            "Other horizons use available horizon labels because per-horizon quality annotations are not yet separate."
        ),
        "by_horizon": horizon_rows,
        "best_horizon": best_rows,
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


def render_regime_outcome_table_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Regime Outcome Tables",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path')}",
        f"- Directional rows: {report.get('directional_row_count')}",
        f"- Best-horizon rows: {report.get('best_horizon_row_count')}",
        f"- Reliable/low-confidence rows: {report.get('reliable_or_low_confidence_best_row_count')}",
        f"- Sparse rows: {report.get('sparse_best_row_count')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Baseline By Horizon",
        "",
        "| Horizon | Labels | Hit Rate | Avg Signed Return Bps | Sample Quality |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for horizon, row in (report.get("baseline_by_horizon") or {}).items():
        lines.append(
            f"| {horizon} | {row.get('label_count')} | {row.get('hit_rate')} | "
            f"{row.get('avg_signed_return_bps')} | {row.get('sample_quality')} |"
        )

    lines.extend(
        [
            "",
            "## Field Coverage",
            "",
            "| Field | Available | Coverage | Top Values |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in report.get("field_coverage", []):
        top_values = ", ".join(f"{key}:{value}" for key, value in (row.get("top_values") or {}).items())
        lines.append(
            f"| {row.get('field')} | {row.get('available_count')}/{row.get('row_count')} | "
            f"{row.get('coverage_ratio')} | {top_values or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Top Favorable Cells",
            "",
            "| Group | Regime | Best Horizon | Labels | Hit Rate | Avg Return Bps | Delta Bps | Quality | Hint |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("top_favorable", []):
        regime = _format_regime_cell(row)
        lines.append(
            f"| {row.get('group_name')} | {regime} | {row.get('best_horizon')} | {row.get('label_count')} | "
            f"{row.get('hit_rate')} | {row.get('avg_signed_return_bps')} | "
            f"{row.get('avg_return_delta_vs_all_bps')} | {row.get('sample_quality')} | {row.get('edge_hint')} |"
        )
    if not report.get("top_favorable"):
        lines.append("| none | none | none | 0 |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Top Unfavorable Cells",
            "",
            "| Group | Regime | Best Horizon | Labels | Hit Rate | Avg Return Bps | Delta Bps | Quality | Hint |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("top_unfavorable", []):
        regime = _format_regime_cell(row)
        lines.append(
            f"| {row.get('group_name')} | {regime} | {row.get('best_horizon')} | {row.get('label_count')} | "
            f"{row.get('hit_rate')} | {row.get('avg_signed_return_bps')} | "
            f"{row.get('avg_return_delta_vs_all_bps')} | {row.get('sample_quality')} | {row.get('edge_hint')} |"
        )
    if not report.get("top_unfavorable"):
        lines.append("| none | none | none | 0 |  |  |  |  |  |")

    lines.extend(["", "## Recommended Next Actions", ""])
    for action in report.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    lines.extend(["", f"Quality note: {report.get('quality_note')}", ""])
    lines.append("*Research-only artifact. It does not alter runtime config, parameter packs, data sources, or execution behavior.*")
    return "\n".join(lines) + "\n"


def _format_regime_cell(row: dict[str, Any]) -> str:
    group_fields = set(str(row.get("group_name") or "").split("+"))
    parts = []
    for key in ("gamma_regime", "volatility_regime", "direction", "macro_risk_bucket", "pcr_bucket"):
        if key not in group_fields:
            continue
        value = row.get(key)
        if value is not None and str(value).upper() not in _MISSING_TEXT:
            parts.append(f"{key}={value}")
    return "<br>".join(parts) if parts else "ALL"


def write_regime_outcome_table_report(
    frame: pd.DataFrame | None,
    *,
    dataset_path: str | Path | None = None,
    output_dir: Path = DEFAULT_REGIME_OUTCOME_TABLE_DIR,
    report_name: str | None = None,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    top_n: int = 15,
    write_latest: bool = True,
) -> dict[str, Any]:
    report = build_regime_outcome_table_report(
        frame,
        dataset_path=dataset_path,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        top_n=top_n,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report_name or "regime_outcome_tables"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    by_horizon_csv_path = output_dir / f"{stem}_by_horizon.csv"
    best_horizon_csv_path = output_dir / f"{stem}_best_horizon.csv"

    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_regime_outcome_table_markdown(report))
    _atomic_write_csv(pd.DataFrame(report.get("by_horizon", [])), by_horizon_csv_path)
    _atomic_write_csv(pd.DataFrame(report.get("best_horizon", [])), best_horizon_csv_path)

    artifact = {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "by_horizon_csv_path": str(by_horizon_csv_path),
        "best_horizon_csv_path": str(best_horizon_csv_path),
    }
    if write_latest:
        latest_json = output_dir / REGIME_OUTCOME_JSON_FILENAME
        latest_markdown = output_dir / REGIME_OUTCOME_MARKDOWN_FILENAME
        latest_by_horizon = output_dir / REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME
        latest_best_horizon = output_dir / REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME
        _atomic_write_text(latest_json, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown, render_regime_outcome_table_markdown(report))
        _atomic_write_csv(pd.DataFrame(report.get("by_horizon", [])), latest_by_horizon)
        _atomic_write_csv(pd.DataFrame(report.get("best_horizon", [])), latest_best_horizon)
        artifact.update(
            {
                "latest_json_path": str(latest_json),
                "latest_markdown_path": str(latest_markdown),
                "latest_by_horizon_csv_path": str(latest_by_horizon),
                "latest_best_horizon_csv_path": str(latest_best_horizon),
            }
        )
    return artifact
