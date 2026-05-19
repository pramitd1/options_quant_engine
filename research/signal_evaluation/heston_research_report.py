"""Research reports for Heston diagnostics captured in signal datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH
from utils.timestamp_helpers import coerce_timestamp_series

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except Exception:  # pragma: no cover - sklearn is optional for report enrichment
    RandomForestClassifier = None
    RandomForestRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HESTON_REPORT_DIR = PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "heston_research"

HESTON_PARAMETER_FIELDS = [
    "heston_kappa",
    "heston_theta",
    "heston_vol_of_vol",
    "heston_rho",
    "heston_v0",
]

HESTON_FEATURE_FIELDS = [
    *HESTON_PARAMETER_FIELDS,
    "heston_calibration_error",
    "heston_bound_hit_count",
    "heston_tte_days",
    "heston_forward_variance_proxy",
    "bs_vs_heston_price_gap",
    "heston_price_gap_rel_pct",
    "bs_vs_heston_greek_gap",
    "greek_model_divergence_score",
]

TARGET_FIELDS = [
    "correct_60m",
    "signed_return_60m_bps",
    "tradeability_score",
    "option_efficiency_score",
    "strike_efficiency_score",
    "volatility_explosion_probability",
]


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if pd.isna(number):
        return default
    return number


def _round(value, digits=4):
    value = _safe_float(value, None)
    return round(value, digits) if value is not None else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if not isinstance(value, (dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _series(frame: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _with_signal_date(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "signal_timestamp" in working.columns:
        ts = coerce_timestamp_series(working["signal_timestamp"])
        working["signal_date"] = ts.dt.date.astype(str)
    else:
        working["signal_date"] = "UNKNOWN"
    return working


def _quality_by_day(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for day, group in frame.groupby("signal_date", dropna=False):
        calibrated = group[_series(group, "heston_calibration_status", "").astype(str) == "CALIBRATED"]
        rows.append(
            {
                "signal_date": str(day),
                "row_count": int(len(group)),
                "calibrated_count": int(len(calibrated)),
                "calibrated_share": _round(len(calibrated) / len(group), 4) if len(group) else None,
                "mean_calibration_error": _round(
                    pd.to_numeric(calibrated.get("heston_calibration_error", pd.Series(dtype=float)), errors="coerce").mean(),
                    6,
                ),
                "median_divergence_score": _round(
                    pd.to_numeric(group.get("greek_model_divergence_score", pd.Series(dtype=float)), errors="coerce").median(),
                    2,
                ),
                "quality_counts": {
                    str(k): int(v)
                    for k, v in group.get("heston_surface_quality", pd.Series(dtype=object)).fillna("UNKNOWN").value_counts().to_dict().items()
                },
            }
        )
    return sorted(rows, key=lambda row: row["signal_date"])


def _quality_flag_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "heston_quality_flags" not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for value in frame["heston_quality_flags"].fillna("").astype(str):
        for token in value.split(","):
            flag = token.strip()
            if not flag:
                continue
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _quality_by_group(frame: pd.DataFrame, group_field: str, *, top_n: int = 25) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if group_field not in frame.columns:
        return []
    working = frame.copy()
    working["_group"] = (
        working[group_field]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN", "nan": "UNKNOWN", "NaN": "UNKNOWN", "None": "UNKNOWN"})
    )
    rows: list[dict[str, Any]] = []
    for group_value, group in working.groupby("_group", dropna=False):
        status = _series(group, "heston_calibration_status", "").fillna("").astype(str).str.upper()
        calibrated = group[status == "CALIBRATED"]
        rejected = group[status == "REJECTED"]
        rows.append(
            {
                "bucket": str(group_value),
                "row_count": int(len(group)),
                "calibrated_count": int(len(calibrated)),
                "calibrated_share": _round(len(calibrated) / len(group), 4) if len(group) else None,
                "rejected_count": int(len(rejected)),
                "mean_calibration_error": _round(
                    pd.to_numeric(calibrated.get("heston_calibration_error", pd.Series(dtype=float)), errors="coerce").mean(),
                    6,
                ),
                "median_divergence_score": _round(
                    pd.to_numeric(group.get("greek_model_divergence_score", pd.Series(dtype=float)), errors="coerce").median(),
                    2,
                ),
                "median_price_gap_pct": _round(
                    pd.to_numeric(group.get("heston_price_gap_rel_pct", pd.Series(dtype=float)), errors="coerce").median(),
                    2,
                ),
                "median_tte_days": _round(
                    pd.to_numeric(group.get("heston_tte_days", pd.Series(dtype=float)), errors="coerce").median(),
                    4,
                ),
                "quality_counts": {
                    str(k): int(v)
                    for k, v in group.get("heston_surface_quality", pd.Series(dtype=object)).fillna("UNKNOWN").value_counts().to_dict().items()
                },
                "flag_counts": _quality_flag_counts(group),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["row_count"]), row["bucket"]))[:top_n]


def _parameter_stability(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": "no_heston_rows", "fields": {}}
    fields = {}
    for field in HESTON_PARAMETER_FIELDS:
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        if values.empty:
            fields[field] = {"count": 0}
            continue
        by_day = (
            frame.assign(_value=pd.to_numeric(frame[field], errors="coerce"))
            .dropna(subset=["_value"])
            .groupby("signal_date")["_value"]
            .median()
            .sort_index()
        )
        fields[field] = {
            "count": int(len(values)),
            "mean": _round(values.mean(), 6),
            "std": _round(values.std(ddof=0), 6),
            "min": _round(values.min(), 6),
            "max": _round(values.max(), 6),
            "latest_daily_median": _round(by_day.iloc[-1], 6) if not by_day.empty else None,
            "median_abs_daily_change": _round(by_day.diff().abs().median(), 6) if len(by_day) >= 2 else None,
        }
    return {"available": bool(fields), "fields": fields}


def _feature_importance(frame: pd.DataFrame, *, min_sample: int) -> list[dict[str, Any]]:
    rows = []
    for feature in HESTON_FEATURE_FIELDS:
        if feature not in frame.columns:
            continue
        x = pd.to_numeric(frame[feature], errors="coerce")
        for target in TARGET_FIELDS:
            if target not in frame.columns:
                continue
            y = pd.to_numeric(frame[target], errors="coerce")
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(valid) < min_sample or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
                continue
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "sample_size": int(len(valid)),
                    "pearson_corr": _round(valid["x"].corr(valid["y"], method="pearson"), 6),
                    "spearman_corr": _round(valid["x"].corr(valid["y"], method="spearman"), 6),
                    "mean_feature": _round(valid["x"].mean(), 6),
                    "mean_target": _round(valid["y"].mean(), 6),
                }
            )
    rows = sorted(rows, key=lambda row: abs(row.get("spearman_corr") or 0.0), reverse=True)
    return rows


def _ml_feature_importance(frame: pd.DataFrame, *, min_sample: int) -> list[dict[str, Any]]:
    if RandomForestClassifier is None or RandomForestRegressor is None:
        return []
    available_features = [field for field in HESTON_FEATURE_FIELDS if field in frame.columns]
    if not available_features:
        return []

    rows: list[dict[str, Any]] = []
    for target in TARGET_FIELDS:
        if target not in frame.columns:
            continue
        model_frame = frame[[*available_features, target]].copy()
        for feature in available_features:
            model_frame[feature] = pd.to_numeric(model_frame[feature], errors="coerce")
        model_frame[target] = pd.to_numeric(model_frame[target], errors="coerce")
        model_frame = model_frame.dropna()
        if len(model_frame) < min_sample or model_frame[target].nunique() < 2:
            continue
        x = model_frame[available_features]
        y = model_frame[target]
        try:
            if set(y.dropna().unique()).issubset({0, 1}):
                model = RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=2,
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=80,
                    min_samples_leaf=2,
                    random_state=42,
                )
            model.fit(x, y)
        except Exception:
            continue
        importances = sorted(
            zip(available_features, model.feature_importances_),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        rows.append(
            {
                "target": target,
                "sample_size": int(len(model_frame)),
                "top_features": [
                    {"feature": feature, "importance": _round(importance, 6)}
                    for feature, importance in importances[:8]
                    if float(importance) > 0
                ],
            }
        )
    return rows


def build_heston_research_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    min_sample: int = 30,
) -> dict[str, Any]:
    """Build a JSON-serializable Heston research report."""

    working = _with_signal_date(pd.DataFrame(frame))
    enabled = working[_series(working, "heston_research_enabled", False).astype(str).str.upper().isin({"TRUE", "1", "YES"})]
    heston_rows = working[
        _series(working, "heston_surface_quality", "")
        .fillna("")
        .astype(str)
        .str.upper()
        .ne("")
    ]
    calibrated = working[_series(working, "heston_calibration_status", "").astype(str) == "CALIBRATED"]
    report = {
        "report_type": "heston_research_report",
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "row_count": int(len(working)),
        "heston_row_count": int(len(heston_rows)),
        "heston_enabled_row_count": int(len(enabled)),
        "heston_calibrated_row_count": int(len(calibrated)),
        "quality_by_day": _quality_by_day(heston_rows),
        "quality_by_tte_bucket": _quality_by_group(heston_rows, "heston_tte_bucket"),
        "quality_by_expiry_context": _quality_by_group(heston_rows, "heston_expiry_context"),
        "quality_by_selected_iv_quality": _quality_by_group(heston_rows, "heston_selected_iv_quality"),
        "quality_by_direction": _quality_by_group(heston_rows, "direction"),
        "quality_by_provider_health": _quality_by_group(heston_rows, "provider_health_status"),
        "quality_by_option_source": _quality_by_group(heston_rows, "option_source"),
        "parameter_stability": _parameter_stability(calibrated),
        "feature_importance": _feature_importance(calibrated, min_sample=min_sample),
        "ml_feature_importance": _ml_feature_importance(calibrated, min_sample=min_sample),
        "surface_quality_counts": {
            str(k): int(v)
            for k, v in heston_rows.get("heston_surface_quality", pd.Series(dtype=object)).fillna("UNKNOWN").value_counts().to_dict().items()
        },
        "quality_flag_counts": _quality_flag_counts(heston_rows),
        "notes": [
            "Heston diagnostics are research-only and do not change trade decisions.",
            "Black-Scholes remains the live Greek engine.",
            "Feature importance uses univariate correlations and is a screening view, not a promotion gate.",
        ],
    }
    return _json_ready(report)


def render_heston_research_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Heston Research Diagnostics",
        "",
        f"- Dataset: `{report.get('dataset_path') or 'unknown'}`",
        f"- Rows: {report.get('row_count')}",
        f"- Heston rows: {report.get('heston_row_count')}",
        f"- Calibrated rows: {report.get('heston_calibrated_row_count')}",
        "",
        "## Surface Quality",
    ]
    quality_counts = report.get("surface_quality_counts") or {}
    if quality_counts:
        for key, value in quality_counts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No Heston quality rows available yet.")

    lines.extend(["", "## Quality Guard Flags"])
    quality_flag_counts = report.get("quality_flag_counts") or {}
    if quality_flag_counts:
        for key, value in quality_flag_counts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No guard flags observed yet.")

    lines.extend(["", "## Calibration Quality By Day"])
    for row in (report.get("quality_by_day") or [])[:30]:
        lines.append(
            f"- {row.get('signal_date')}: calibrated {row.get('calibrated_count')}/"
            f"{row.get('row_count')} | mean error {row.get('mean_calibration_error')} | "
            f"median divergence {row.get('median_divergence_score')}"
        )
    if not report.get("quality_by_day"):
        lines.append("- No daily calibration rows available.")

    def _render_group_section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}"])
        if not rows:
            lines.append("- No grouped Heston rows available.")
            return
        for row in rows[:12]:
            flags = row.get("flag_counts") or {}
            top_flags = ", ".join(f"{key}={value}" for key, value in list(flags.items())[:3]) or "none"
            lines.append(
                f"- {row.get('bucket')}: rows {row.get('row_count')} | calibrated "
                f"{row.get('calibrated_count')} ({row.get('calibrated_share')}) | rejected "
                f"{row.get('rejected_count')} | mean error {row.get('mean_calibration_error')} | "
                f"median gap {row.get('median_price_gap_pct')}% | flags {top_flags}"
            )

    _render_group_section("Quality By TTE Bucket", report.get("quality_by_tte_bucket") or [])
    _render_group_section("Quality By Expiry Context", report.get("quality_by_expiry_context") or [])
    _render_group_section("Quality By Selected IV Quality", report.get("quality_by_selected_iv_quality") or [])
    _render_group_section("Quality By Direction", report.get("quality_by_direction") or [])
    _render_group_section("Quality By Provider Health", report.get("quality_by_provider_health") or [])
    _render_group_section("Quality By Option Source", report.get("quality_by_option_source") or [])

    lines.extend(["", "## Parameter Stability"])
    fields = ((report.get("parameter_stability") or {}).get("fields") or {})
    if fields:
        for field, stats in fields.items():
            lines.append(
                f"- {field}: mean {stats.get('mean')} | std {stats.get('std')} | "
                f"median daily change {stats.get('median_abs_daily_change')}"
            )
    else:
        lines.append("- No calibrated parameter history available.")

    lines.extend(["", "## Feature Importance Screen"])
    for row in (report.get("feature_importance") or [])[:20]:
        lines.append(
            f"- {row.get('feature')} -> {row.get('target')}: "
            f"spearman {row.get('spearman_corr')} (n={row.get('sample_size')})"
        )
    if not report.get("feature_importance"):
        lines.append("- Not enough calibrated rows for feature-importance screening.")

    lines.extend(["", "## ML Feature Importance Screen"])
    for row in (report.get("ml_feature_importance") or [])[:10]:
        top_features = ", ".join(
            f"{item.get('feature')}={item.get('importance')}"
            for item in (row.get("top_features") or [])[:5]
        )
        lines.append(f"- {row.get('target')}: {top_features or 'no non-zero Heston importances'}")
    if not report.get("ml_feature_importance"):
        lines.append("- Not enough calibrated rows for ML feature-importance screening.")

    lines.extend(["", "## Notes"])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_heston_research_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_HESTON_REPORT_DIR,
    report_name: str | None = None,
    min_sample: int = 30,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write Heston research diagnostics as JSON and Markdown."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "heston_research_report"
    report = build_heston_research_report(frame, dataset_path=dataset_path, min_sample=min_sample)
    markdown = render_heston_research_markdown(report)
    json_path = output / f"{stem}.json"
    md_path = output / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_json_path = output / "latest_heston_research_report.json"
    latest_md_path = output / "latest_heston_research_report.md"
    if write_latest:
        latest_json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        latest_md_path.write_text(markdown, encoding="utf-8")
    return {
        "report": report,
        "json_path": str(json_path),
        "md_path": str(md_path),
        "latest_json_path": str(latest_json_path) if write_latest else None,
        "latest_md_path": str(latest_md_path) if write_latest else None,
    }


def default_heston_dataset_path() -> Path:
    return CUMULATIVE_DATASET_PATH
