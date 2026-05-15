"""Signal-quality model audit for calibration, regimes, and ranking features.

This is a research-only diagnostic. It reads signal-evaluation rows and
produces advisory evidence for future model improvements without changing live
signal generation, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.daily_research_report import (
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
)
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary
from utils.timestamp_helpers import coerce_timestamp_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_QUALITY_MODEL_AUDIT_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "signal_quality_model_audit"
)

SIGNAL_QUALITY_MODEL_AUDIT_JSON_FILENAME = "latest_signal_quality_model_audit.json"
SIGNAL_QUALITY_MODEL_AUDIT_MARKDOWN_FILENAME = "latest_signal_quality_model_audit.md"
SIGNAL_QUALITY_MODEL_AUDIT_CALIBRATION_CSV_FILENAME = "latest_signal_quality_model_audit_calibration.csv"
SIGNAL_QUALITY_MODEL_AUDIT_REGIME_CSV_FILENAME = "latest_signal_quality_model_audit_regime.csv"
SIGNAL_QUALITY_MODEL_AUDIT_FEATURE_CSV_FILENAME = "latest_signal_quality_model_audit_features.csv"
SIGNAL_QUALITY_MODEL_AUDIT_RANKING_CSV_FILENAME = "latest_signal_quality_model_audit_ranking.csv"

DEFAULT_PROBABILITY_FIELD = "hybrid_move_probability"
DEFAULT_LABEL_FIELD = "correct_60m"
DEFAULT_RETURN_FIELD = "signed_return_60m_bps"
DEFAULT_REGIME_FIELDS = (
    "macro_regime",
    "gamma_regime",
    "volatility_regime",
    "global_risk_state",
)
DEFAULT_FEATURE_COLUMNS = (
    "hybrid_move_probability",
    "rule_move_probability",
    "ml_move_probability",
    "signal_confidence_score",
    "composite_signal_score",
    "trade_strength",
    "tradeability_score",
    "target_reachability_score",
    "premium_efficiency_score",
    "strike_efficiency_score",
    "option_efficiency_score",
    "global_risk_score",
    "gamma_vol_acceleration_score",
    "dealer_hedging_pressure_score",
    "macro_event_risk_score",
    "data_quality_score",
    "selected_option_ba_spread_pct",
    "selected_option_volume",
    "selected_option_open_interest",
    "selected_option_iv",
)
DEFAULT_RANKING_FEATURES = (
    "composite_signal_score",
    "signal_confidence_score",
    "hybrid_move_probability",
    "trade_strength",
    "tradeability_score",
    "target_reachability_score",
    "premium_efficiency_score",
    "strike_efficiency_score",
    "option_efficiency_score",
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


def default_signal_quality_dataset_path() -> Path:
    """Return the preferred dataset for signal-quality model audits."""
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _prepare_labeled_frame(frame: pd.DataFrame, *, fallback_to_legacy: bool = True) -> pd.DataFrame:
    working = apply_quality_label_view(
        frame if frame is not None else pd.DataFrame(),
        fallback_to_legacy=fallback_to_legacy,
        drop_unapproved=True,
    ).copy()
    if working.empty:
        return working
    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = coerce_timestamp_series(working["signal_timestamp"])
        working = working.sort_values("signal_timestamp", kind="mergesort")
    if "direction" in working.columns:
        direction = working["direction"].astype(str).str.upper().str.strip()
        directional = direction.isin({"CALL", "PUT"})
        if directional.any():
            working = working.loc[directional].copy()
    for column in set(DEFAULT_FEATURE_COLUMNS) | {DEFAULT_LABEL_FIELD, DEFAULT_RETURN_FIELD, "mae_60m_bps"}:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.reset_index(drop=True)


def _probability_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[field], errors="coerce")
    max_value = values.dropna().max() if values.notna().any() else None
    if max_value is not None and float(max_value) > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def _sample_quality(label_count: int, *, min_label_sample: int, strong_label_sample: int) -> str:
    if label_count <= 0:
        return "NO_EVIDENCE"
    if label_count < int(min_label_sample):
        return "INSUFFICIENT_EVIDENCE"
    if label_count < int(strong_label_sample):
        return "MODERATE_EVIDENCE"
    return "STRONG_EVIDENCE"


def _calibration_bins(
    frame: pd.DataFrame,
    *,
    probability_field: str,
    label_field: str,
) -> pd.DataFrame:
    if frame.empty or probability_field not in frame.columns or label_field not in frame.columns:
        return pd.DataFrame(
            columns=[
                "probability_bucket",
                "signal_count",
                "avg_predicted_probability",
                "actual_hit_rate",
                "calibration_gap",
                "abs_calibration_gap",
                "avg_signed_return_60m_bps",
            ]
        )
    working = frame.copy()
    working["_probability"] = _probability_series(working, probability_field)
    working["_label"] = pd.to_numeric(working[label_field], errors="coerce")
    working["_return"] = pd.to_numeric(working.get(DEFAULT_RETURN_FIELD, pd.Series(index=working.index)), errors="coerce")
    working = working.dropna(subset=["_probability", "_label"])
    if working.empty:
        return pd.DataFrame(columns=["probability_bucket", "signal_count"])

    bins = [0.0, 0.35, 0.50, 0.65, 0.80, 1.000001]
    labels = ["0_35", "35_50", "50_65", "65_80", "80_100"]
    working["probability_bucket"] = pd.cut(
        working["_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    grouped = (
        working.groupby("probability_bucket", dropna=False, observed=False)
        .agg(
            signal_count=("_label", "count"),
            avg_predicted_probability=("_probability", "mean"),
            actual_hit_rate=("_label", "mean"),
            avg_signed_return_60m_bps=("_return", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["signal_count"] > 0].copy()
    grouped["probability_bucket"] = grouped["probability_bucket"].astype(str)
    grouped["calibration_gap"] = grouped["avg_predicted_probability"] - grouped["actual_hit_rate"]
    grouped["abs_calibration_gap"] = grouped["calibration_gap"].abs()
    for column in [
        "avg_predicted_probability",
        "actual_hit_rate",
        "calibration_gap",
        "abs_calibration_gap",
        "avg_signed_return_60m_bps",
    ]:
        grouped[column] = grouped[column].round(4)
    return grouped.reset_index(drop=True)


def build_probability_calibration_audit(
    frame: pd.DataFrame,
    *,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
) -> dict[str, Any]:
    """Build Brier/ECE probability calibration diagnostics."""
    working = frame.copy()
    bins = _calibration_bins(working, probability_field=probability_field, label_field=label_field)
    probability = _probability_series(working, probability_field)
    labels = pd.to_numeric(working.get(label_field, pd.Series(index=working.index)), errors="coerce")
    valid = probability.notna() & labels.notna()
    label_count = int(valid.sum())
    if label_count <= 0:
        summary = {
            "probability_field": probability_field,
            "label_field": label_field,
            "label_count": 0,
            "sample_quality": "NO_EVIDENCE",
            "brier_score": None,
            "expected_calibration_error": None,
            "max_calibration_error": None,
            "mean_predicted_probability": None,
            "actual_hit_rate": None,
            "calibration_status": "NO_EVIDENCE",
        }
        return {"summary": summary, "bins": []}

    p = probability.loc[valid]
    y = labels.loc[valid]
    ece = None
    mce = None
    if not bins.empty:
        weights = bins["signal_count"] / max(float(bins["signal_count"].sum()), 1.0)
        ece = float((weights * bins["abs_calibration_gap"]).sum())
        mce = float(bins["abs_calibration_gap"].max())
    brier = float(((p - y) ** 2).mean())
    mean_pred = float(p.mean())
    hit_rate = float(y.mean())
    sample_quality = _sample_quality(label_count, min_label_sample=min_label_sample, strong_label_sample=strong_label_sample)
    if sample_quality in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}:
        calibration_status = sample_quality
    elif ece is not None and ece > 0.15:
        calibration_status = "POOR_CALIBRATION"
    elif ece is not None and ece > 0.08:
        calibration_status = "WATCH_CALIBRATION"
    else:
        calibration_status = "CALIBRATED"
    summary = {
        "probability_field": probability_field,
        "label_field": label_field,
        "label_count": label_count,
        "sample_quality": sample_quality,
        "brier_score": _round_or_none(brier, 6),
        "expected_calibration_error": _round_or_none(ece, 6),
        "max_calibration_error": _round_or_none(mce, 6),
        "mean_predicted_probability": _round_or_none(mean_pred, 4),
        "actual_hit_rate": _round_or_none(hit_rate, 4),
        "calibration_gap": _round_or_none(mean_pred - hit_rate, 4),
        "calibration_status": calibration_status,
    }
    return {"summary": summary, "bins": bins.to_dict(orient="records")}


def build_regime_calibration_audit(
    frame: pd.DataFrame,
    *,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    return_field: str = DEFAULT_RETURN_FIELD,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    min_regime_sample: int = 10,
) -> list[dict[str, Any]]:
    """Build regime-segmented calibration and threshold-bias hints."""
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for regime_field in regime_fields:
        if regime_field not in frame.columns:
            continue
        for regime_value, group in frame.groupby(regime_field, dropna=False):
            probability = _probability_series(group, probability_field)
            labels = pd.to_numeric(group.get(label_field, pd.Series(index=group.index)), errors="coerce")
            returns = pd.to_numeric(group.get(return_field, pd.Series(index=group.index)), errors="coerce")
            valid = probability.notna() & labels.notna()
            label_count = int(valid.sum())
            if label_count <= 0:
                continue
            mean_pred = float(probability.loc[valid].mean())
            hit_rate = float(labels.loc[valid].mean())
            avg_return = float(returns.loc[valid].mean()) if returns.loc[valid].notna().any() else None
            gap = mean_pred - hit_rate
            if label_count < int(min_regime_sample):
                threshold_bias = "INSUFFICIENT_EVIDENCE"
            elif gap > 0.10 or (avg_return is not None and avg_return < 0):
                threshold_bias = "RAISE_OR_DEFLATE_PROBABILITY"
            elif gap < -0.10 and avg_return is not None and avg_return > 0:
                threshold_bias = "LOWER_OR_EXPAND_SELECTIVELY"
            else:
                threshold_bias = "KEEP"
            rows.append(
                {
                    "regime_field": regime_field,
                    "regime_value": str(regime_value),
                    "label_count": label_count,
                    "mean_predicted_probability": _round_or_none(mean_pred, 4),
                    "actual_hit_rate": _round_or_none(hit_rate, 4),
                    "calibration_gap": _round_or_none(gap, 4),
                    "avg_signed_return_60m_bps": _round_or_none(avg_return, 4),
                    "threshold_bias": threshold_bias,
                }
            )
    return sorted(rows, key=lambda item: (item["threshold_bias"] == "KEEP", -int(item["label_count"])))


def build_feature_stability_audit(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> list[dict[str, Any]]:
    """Build missingness, variance, and recent-vs-prior drift diagnostics."""
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = coerce_timestamp_series(working["signal_timestamp"])
        working = working.sort_values("signal_timestamp", kind="mergesort")
    split_at = max(min(int(len(working) * 0.70), len(working) - 1), 1) if len(working) > 1 else len(working)
    prior = working.iloc[:split_at]
    recent = working.iloc[split_at:]
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        missing_rate = float(values.isna().mean())
        non_null = values.dropna()
        unique_count = int(non_null.nunique())
        prior_values = pd.to_numeric(prior.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        recent_values = pd.to_numeric(recent.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        prior_mean = float(prior_values.mean()) if not prior_values.empty else None
        recent_mean = float(recent_values.mean()) if not recent_values.empty else None
        prior_std = float(prior_values.std(ddof=0)) if len(prior_values) > 1 else None
        mean_shift = (recent_mean - prior_mean) if recent_mean is not None and prior_mean is not None else None
        normalized_shift = None
        if mean_shift is not None and prior_std is not None and prior_std > 1e-9:
            normalized_shift = abs(mean_shift) / prior_std
        if missing_rate > 0.50 or unique_count <= 1 or (normalized_shift is not None and normalized_shift > 2.0):
            stability_status = "FAIL"
        elif missing_rate > 0.20 or (normalized_shift is not None and normalized_shift > 1.0):
            stability_status = "WATCH"
        else:
            stability_status = "OK"
        rows.append(
            {
                "feature": column,
                "non_null_count": int(non_null.count()),
                "missing_rate": _round_or_none(missing_rate, 4),
                "unique_count": unique_count,
                "prior_mean": _round_or_none(prior_mean, 4),
                "recent_mean": _round_or_none(recent_mean, 4),
                "mean_shift": _round_or_none(mean_shift, 4),
                "normalized_mean_shift": _round_or_none(normalized_shift, 4),
                "stability_status": stability_status,
            }
        )
    severity = {"FAIL": 0, "WATCH": 1, "OK": 2}
    return sorted(rows, key=lambda item: (severity.get(str(item["stability_status"]), 3), item["feature"]))


def _risk_adjusted_return(frame: pd.DataFrame, return_field: str) -> pd.Series:
    returns = pd.to_numeric(frame.get(return_field, pd.Series(index=frame.index)), errors="coerce")
    mae = pd.to_numeric(frame.get("mae_60m_bps", pd.Series(index=frame.index)), errors="coerce")
    downside_penalty = mae.abs().fillna(0.0) * 0.25
    return returns - downside_penalty


def build_ev_risk_ranking_audit(
    frame: pd.DataFrame,
    *,
    ranking_features: tuple[str, ...] = DEFAULT_RANKING_FEATURES,
    label_field: str = DEFAULT_LABEL_FIELD,
    return_field: str = DEFAULT_RETURN_FIELD,
    min_label_sample: int = 30,
) -> list[dict[str, Any]]:
    """Evaluate which existing features rank realized EV/risk best."""
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    labels = pd.to_numeric(frame.get(label_field, pd.Series(index=frame.index)), errors="coerce")
    returns = pd.to_numeric(frame.get(return_field, pd.Series(index=frame.index)), errors="coerce")
    risk_return = _risk_adjusted_return(frame, return_field)
    for feature in ranking_features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = values.notna() & labels.notna() & returns.notna()
        subset = frame.loc[valid].copy()
        label_count = int(len(subset))
        if label_count <= 0:
            continue
        valid_values = values.loc[valid]
        try:
            high_cut = float(valid_values.quantile(0.75))
            low_cut = float(valid_values.quantile(0.25))
        except Exception:
            continue
        high_mask = valid & (values >= high_cut)
        low_mask = valid & (values <= low_cut)
        high_count = int(high_mask.sum())
        low_count = int(low_mask.sum())
        high_hit = labels.loc[high_mask].mean() if high_count else None
        low_hit = labels.loc[low_mask].mean() if low_count else None
        high_return = returns.loc[high_mask].mean() if high_count else None
        low_return = returns.loc[low_mask].mean() if low_count else None
        high_risk_return = risk_return.loc[high_mask].mean() if high_count else None
        low_risk_return = risk_return.loc[low_mask].mean() if low_count else None
        hit_delta = (high_hit - low_hit) if high_hit is not None and low_hit is not None else None
        return_delta = (
            high_risk_return - low_risk_return
            if high_risk_return is not None and low_risk_return is not None
            else None
        )
        ranking_score = None
        if return_delta is not None and hit_delta is not None:
            ranking_score = float(return_delta) + 50.0 * float(hit_delta)
        if label_count < int(min_label_sample):
            ranking_status = "INSUFFICIENT_EVIDENCE"
        elif ranking_score is not None and ranking_score > 10:
            ranking_status = "HELPFUL"
        elif ranking_score is not None and ranking_score < -10:
            ranking_status = "INVERTED"
        else:
            ranking_status = "WEAK"
        rows.append(
            {
                "feature": feature,
                "label_count": label_count,
                "low_cutoff": _round_or_none(low_cut, 4),
                "high_cutoff": _round_or_none(high_cut, 4),
                "top_quartile_count": high_count,
                "bottom_quartile_count": low_count,
                "top_quartile_hit_rate": _round_or_none(high_hit, 4),
                "bottom_quartile_hit_rate": _round_or_none(low_hit, 4),
                "hit_rate_delta": _round_or_none(hit_delta, 4),
                "top_quartile_avg_return_bps": _round_or_none(high_return, 4),
                "bottom_quartile_avg_return_bps": _round_or_none(low_return, 4),
                "top_quartile_risk_adjusted_return_bps": _round_or_none(high_risk_return, 4),
                "bottom_quartile_risk_adjusted_return_bps": _round_or_none(low_risk_return, 4),
                "risk_adjusted_return_delta_bps": _round_or_none(return_delta, 4),
                "ranking_score": _round_or_none(ranking_score, 4),
                "ranking_status": ranking_status,
            }
        )
    severity = {"HELPFUL": 0, "WEAK": 1, "INVERTED": 2, "INSUFFICIENT_EVIDENCE": 3}
    return sorted(
        rows,
        key=lambda item: (severity.get(str(item["ranking_status"]), 4), -(item.get("ranking_score") or -1e9)),
    )


def _recommended_actions(
    *,
    calibration_summary: dict[str, Any],
    regime_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if calibration_summary.get("calibration_status") in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}:
        actions.append("collect more quality-approved labels before changing model calibration or ranking logic.")
    if calibration_summary.get("calibration_status") in {"POOR_CALIBRATION", "WATCH_CALIBRATION"}:
        actions.append("Fit or refresh probability calibration before using probabilities for sizing or ranking.")
    if any(row.get("threshold_bias") in {"RAISE_OR_DEFLATE_PROBABILITY", "LOWER_OR_EXPAND_SELECTIVELY"} for row in regime_rows):
        actions.append("Evaluate regime-conditioned probability floors for segments with persistent calibration bias.")
    if any(row.get("stability_status") == "FAIL" for row in feature_rows):
        actions.append("Remove or guard failed stability features before training ranking/calibration models.")
    helpful = [row.get("feature") for row in ranking_rows if row.get("ranking_status") == "HELPFUL"]
    inverted = [row.get("feature") for row in ranking_rows if row.get("ranking_status") == "INVERTED"]
    if helpful:
        actions.append(f"Prioritize EV/risk ranking experiments with helpful features: {', '.join(helpful[:5])}.")
    if inverted:
        actions.append(f"Review inverted ranking features before assigning positive score weight: {', '.join(inverted[:5])}.")
    if not actions:
        actions.append("No immediate model-quality action is supported yet; collect more quality-approved labels.")
    return actions


def build_signal_quality_model_audit_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    min_regime_sample: int = 10,
) -> dict[str, Any]:
    """Build a research-only signal-quality model audit report."""
    raw = frame if frame is not None else pd.DataFrame()
    labeled = _prepare_labeled_frame(raw)
    label_summary = label_quality_summary(raw)
    calibration = build_probability_calibration_audit(
        labeled,
        probability_field=probability_field,
        label_field=DEFAULT_LABEL_FIELD,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    regime_rows = build_regime_calibration_audit(
        labeled,
        probability_field=probability_field,
        label_field=DEFAULT_LABEL_FIELD,
        min_regime_sample=min_regime_sample,
    )
    feature_rows = build_feature_stability_audit(raw)
    ranking_rows = build_ev_risk_ranking_audit(
        labeled,
        min_label_sample=min_label_sample,
    )
    report = {
        "report_type": "signal_quality_model_audit",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": int(len(raw)),
        "quality_labeled_row_count": int(len(labeled)),
        "probability_field": probability_field,
        "label_quality": label_summary,
        "calibration_summary": calibration.get("summary", {}),
        "calibration_bins": calibration.get("bins", []),
        "regime_calibration": regime_rows,
        "feature_stability": feature_rows,
        "ranking_feature_audit": ranking_rows,
        "recommended_next_actions": _recommended_actions(
            calibration_summary=calibration.get("summary", {}),
            regime_rows=regime_rows,
            feature_rows=feature_rows,
            ranking_rows=ranking_rows,
        ),
    }
    return _sanitize_value(report)


def render_signal_quality_model_audit_markdown(report: dict[str, Any]) -> str:
    """Render the signal-quality model audit as Markdown."""
    calibration = report.get("calibration_summary", {}) or {}
    lines = [
        "# Signal Quality Model Audit",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Rows: {report.get('row_count')}",
        f"- Quality-labeled rows: {report.get('quality_labeled_row_count')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Probability Calibration",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Probability field | `{calibration.get('probability_field')}` |",
        f"| Label count | {calibration.get('label_count')} |",
        f"| Sample quality | `{calibration.get('sample_quality')}` |",
        f"| Brier score | {calibration.get('brier_score')} |",
        f"| Expected calibration error | {calibration.get('expected_calibration_error')} |",
        f"| Mean predicted probability | {calibration.get('mean_predicted_probability')} |",
        f"| Actual hit rate | {calibration.get('actual_hit_rate')} |",
        f"| Calibration status | `{calibration.get('calibration_status')}` |",
        "",
        "## Recommended Actions",
        "",
    ]
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")

    lines.extend(["", "## Top Ranking Features", "", "| Feature | Status | Score | Hit Delta | Risk-Adjusted Return Delta |", "| --- | --- | ---: | ---: | ---: |"])
    for row in (report.get("ranking_feature_audit", []) or [])[:8]:
        lines.append(
            f"| `{row.get('feature')}` | `{row.get('ranking_status')}` | {row.get('ranking_score')} | "
            f"{row.get('hit_rate_delta')} | {row.get('risk_adjusted_return_delta_bps')} |"
        )

    lines.extend(["", "## Feature Stability Watchlist", "", "| Feature | Status | Missing Rate | Normalized Shift |", "| --- | --- | ---: | ---: |"])
    for row in [item for item in (report.get("feature_stability", []) or []) if item.get("stability_status") != "OK"][:10]:
        lines.append(
            f"| `{row.get('feature')}` | `{row.get('stability_status')}` | {row.get('missing_rate')} | "
            f"{row.get('normalized_mean_shift')} |"
        )
    lines.append("")
    lines.append("*This audit is research-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*")
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "calibration_csv_path": output / f"{stem}_calibration.csv",
        "regime_csv_path": output / f"{stem}_regime.csv",
        "feature_csv_path": output / f"{stem}_features.csv",
        "ranking_csv_path": output / f"{stem}_ranking.csv",
        "latest_json_path": output / SIGNAL_QUALITY_MODEL_AUDIT_JSON_FILENAME,
        "latest_markdown_path": output / SIGNAL_QUALITY_MODEL_AUDIT_MARKDOWN_FILENAME,
        "latest_calibration_csv_path": output / SIGNAL_QUALITY_MODEL_AUDIT_CALIBRATION_CSV_FILENAME,
        "latest_regime_csv_path": output / SIGNAL_QUALITY_MODEL_AUDIT_REGIME_CSV_FILENAME,
        "latest_feature_csv_path": output / SIGNAL_QUALITY_MODEL_AUDIT_FEATURE_CSV_FILENAME,
        "latest_ranking_csv_path": output / SIGNAL_QUALITY_MODEL_AUDIT_RANKING_CSV_FILENAME,
    }


def write_signal_quality_model_audit_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    min_regime_sample: int = 10,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write signal-quality model audit artifacts."""
    report = build_signal_quality_model_audit_report(
        frame,
        dataset_path=dataset_path,
        probability_field=probability_field,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        min_regime_sample=min_regime_sample,
    )
    assert_artifact_schema(report, "signal_quality_model_audit")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SIGNAL_QUALITY_MODEL_AUDIT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "signal_quality_model_audit"
    paths = _artifact_paths(output, stem)
    markdown = render_signal_quality_model_audit_markdown(report)
    calibration = pd.DataFrame(report.get("calibration_bins", []) or [])
    regimes = pd.DataFrame(report.get("regime_calibration", []) or [])
    features = pd.DataFrame(report.get("feature_stability", []) or [])
    ranking = pd.DataFrame(report.get("ranking_feature_audit", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(calibration, paths["calibration_csv_path"])
    _atomic_write_csv(regimes, paths["regime_csv_path"])
    _atomic_write_csv(features, paths["feature_csv_path"])
    _atomic_write_csv(ranking, paths["ranking_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(calibration, paths["latest_calibration_csv_path"])
        _atomic_write_csv(regimes, paths["latest_regime_csv_path"])
        _atomic_write_csv(features, paths["latest_feature_csv_path"])
        _atomic_write_csv(ranking, paths["latest_ranking_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_signal_quality_model_audit_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset from disk and write the model audit report."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_signal_quality_model_audit_report(frame, dataset_path=path, **kwargs)
