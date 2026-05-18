"""Build the compact statistical-market-context artifact consumed by the engine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATISTICAL_MARKET_STRUCTURE_REPORT_PATH = (
    PROJECT_ROOT
    / "research"
    / "ml_research"
    / "statistical_market_structure"
    / "latest_statistical_market_structure_report.json"
)
DEFAULT_STATISTICAL_CONTEXT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "research"
    / "ml_research"
    / "historical_priors"
    / "latest_statistical_market_context_artifact.json"
)

ARTIFACT_VERSION = "statistical_market_context_artifact_v1"

NUMERIC_PRIOR_FEATURES = [
    "india_vix_level",
    "india_vix_change_24h",
    "realized_vol_20d",
    "pcr_oi",
    "pcr_volume",
    "near_atm_pcr_oi",
    "near_atm_pcr_volume",
    "atm_straddle_pct",
    "max_pain_abs_dist_pct",
    "wall_width_pct",
    "ret_20d_bps",
    "usdinr_change_24h",
    "sp500_change_24h",
]

MACRO_SHOCK_PRIOR_FEATURES = [
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "vix_change_24h",
    "india_vix_change_24h",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
]

CATEGORICAL_PRIOR_FEATURES = [
    "trend_20d_bucket",
    "expiry_bucket",
    "india_vix_bucket",
    "pcr_oi_bucket",
    "weekday",
    "macro_major_event",
    "near_call_wall",
    "near_put_wall",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return round(parsed, digits)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _resolve_path(path_ref: str | Path | None, *, report_dir: Path | None = None) -> Path | None:
    if not path_ref:
        return None
    path = Path(path_ref)
    if path.is_absolute():
        return path
    if report_dir is not None and (report_dir / path.name).exists():
        return report_dir / path.name
    return PROJECT_ROOT / path


def _load_table(report: dict[str, Any], key: str, *, report_dir: Path | None = None) -> pd.DataFrame:
    path = _resolve_path((report.get("paths") or {}).get(key), report_dir=report_dir)
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _metric(row: pd.Series, column: str) -> float | None:
    return _round(row.get(column), 4)


def _numeric_priors(table: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if table.empty:
        return {}
    priors: dict[str, list[dict[str, Any]]] = {}
    for feature in NUMERIC_PRIOR_FEATURES:
        feature_rows = table[table.get("feature").astype(str).eq(feature)].copy() if "feature" in table else pd.DataFrame()
        if feature_rows.empty:
            continue
        feature_priors: list[dict[str, Any]] = []
        for _, row in feature_rows.sort_values("feature_min").iterrows():
            prior = {
                "bucket": str(row.get("bucket")),
                "lower": _metric(row, "feature_min"),
                "upper": _metric(row, "feature_max"),
                "n": int(_round(row.get("n"), 0) or 0),
                "mean_return_bps": _metric(row, "fwd_ret_1d_bps_mean"),
                "hit_positive": _metric(row, "fwd_ret_1d_bps_hit_positive"),
                "expected_abs_move_bps": _metric(row, "fwd_abs_ret_1d_bps_mean"),
                "abs_move_delta_vs_base_bps": _metric(row, "fwd_abs_ret_1d_mean_delta_vs_base"),
                "expected_range_bps": _metric(row, "next_day_range_bps_mean"),
                "range_hit_positive": _metric(row, "next_day_range_bps_hit_positive"),
            }
            feature_priors.append(prior)
        priors[feature] = feature_priors
    return priors


def _categorical_priors(table: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    if table.empty:
        return {}
    priors: dict[str, dict[str, dict[str, Any]]] = {}
    for feature in CATEGORICAL_PRIOR_FEATURES:
        feature_rows = table[table.get("feature").astype(str).eq(feature)].copy() if "feature" in table else pd.DataFrame()
        if feature_rows.empty:
            continue
        items: dict[str, dict[str, Any]] = {}
        for _, row in feature_rows.iterrows():
            bucket = str(row.get("bucket"))
            items[bucket] = {
                "n": int(_round(row.get("n"), 0) or 0),
                "mean_return_bps": _metric(row, "fwd_ret_1d_bps_mean"),
                "hit_positive": _metric(row, "fwd_ret_1d_bps_hit_positive"),
                "expected_abs_move_bps": _metric(row, "fwd_abs_ret_1d_bps_mean"),
                "abs_move_delta_vs_base_bps": _metric(row, "fwd_abs_ret_1d_mean_delta_vs_base"),
                "expected_range_bps": _metric(row, "next_day_range_bps_mean"),
            }
        priors[feature] = items
    return priors


def _distribution_baseline(table: pd.DataFrame) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    if table.empty or "feature" not in table:
        return baseline
    by_feature = {str(row["feature"]): row for _, row in table.iterrows()}
    range_row = by_feature.get("range_bps")
    ret_row = by_feature.get("ret_1d_bps")
    abs_row = by_feature.get("fwd_abs_ret_1d_bps")
    if range_row is not None:
        baseline["daily_range_median_bps"] = _metric(range_row, "median")
        baseline["daily_range_p95_bps"] = _metric(range_row, "p95")
        baseline["daily_range_mean_bps"] = _metric(range_row, "mean")
    if ret_row is not None:
        baseline["daily_return_excess_kurtosis"] = _metric(ret_row, "excess_kurtosis")
        baseline["daily_return_skew"] = _metric(ret_row, "skew")
        baseline["daily_return_p05_bps"] = _metric(ret_row, "p05")
        baseline["daily_return_p95_bps"] = _metric(ret_row, "p95")
    if abs_row is not None:
        baseline["fwd_abs_move_1d_mean_bps"] = _metric(abs_row, "mean")
        baseline["fwd_abs_move_1d_median_bps"] = _metric(abs_row, "median")
    return baseline


def _top_target_correlations(table: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if table.empty:
        return []
    rows = []
    for _, row in table.head(limit).iterrows():
        rows.append(
            {
                "feature": row.get("feature"),
                "target": row.get("target"),
                "n": int(_round(row.get("n"), 0) or 0),
                "pearson": _metric(row, "pearson"),
                "spearman": _metric(row, "spearman"),
            }
        )
    return rows


def _macro_shock_priors(table: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    if table.empty or "feature" not in table or "bucket" not in table:
        return {}
    priors: dict[str, dict[str, dict[str, Any]]] = {}
    for feature in MACRO_SHOCK_PRIOR_FEATURES:
        feature_rows = table[table.get("feature").astype(str).eq(feature)].copy()
        if feature_rows.empty:
            continue
        buckets: dict[str, dict[str, Any]] = {}
        for _, row in feature_rows.iterrows():
            bucket = str(row.get("bucket"))
            buckets[bucket] = {
                "n": int(_round(row.get("n"), 0) or 0),
                "threshold": _metric(row, "threshold"),
                "feature_mean": _metric(row, "feature_mean"),
                "mean_return_bps": _metric(row, "fwd_ret_1d_bps_mean"),
                "hit_positive": _metric(row, "fwd_ret_1d_bps_hit_positive"),
                "expected_abs_move_bps": _metric(row, "fwd_abs_ret_1d_bps_mean"),
                "abs_move_delta_vs_base_bps": _metric(row, "fwd_abs_ret_1d_mean_delta_vs_base"),
                "expected_range_bps": _metric(row, "next_day_range_bps_mean"),
            }
        priors[feature] = buckets
    return priors


def _macro_interaction_priors(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if table.empty or "interaction" not in table:
        return {}
    priors: dict[str, dict[str, Any]] = {}
    for _, row in table.iterrows():
        interaction = str(row.get("interaction"))
        if not interaction or interaction == "nan":
            continue
        priors[interaction] = {
            "interaction": interaction,
            "left_feature": row.get("left_feature"),
            "left_bucket": row.get("left_bucket"),
            "right_feature": row.get("right_feature"),
            "right_bucket": row.get("right_bucket"),
            "n": int(_round(row.get("n"), 0) or 0),
            "mean_return_bps": _metric(row, "fwd_ret_1d_bps_mean"),
            "hit_positive": _metric(row, "fwd_ret_1d_bps_hit_positive"),
            "expected_abs_move_bps": _metric(row, "fwd_abs_ret_1d_bps_mean"),
            "abs_move_delta_vs_base_bps": _metric(row, "fwd_abs_ret_1d_mean_delta_vs_base"),
            "expected_range_bps": _metric(row, "next_day_range_bps_mean"),
        }
    return priors


def _macro_pca_context(pca_explained: pd.DataFrame, pca_loadings: pd.DataFrame) -> dict[str, Any]:
    if pca_explained.empty:
        return {}
    row = pca_explained.iloc[0]
    return {
        "component": row.get("component"),
        "explained_variance_ratio": _metric(row, "explained_variance_ratio"),
        "cumulative_variance_ratio": _metric(row, "cumulative_variance_ratio"),
        "top_loadings": pca_loadings[pca_loadings.get("component").astype(str).eq("PC1")]
        .head(8)
        .to_dict(orient="records")
        if not pca_loadings.empty and "component" in pca_loadings
        else [],
    }


def _macro_lead_lag_notes(table: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if table.empty:
        return []
    rows = []
    for _, row in table.head(limit).iterrows():
        rows.append(
            {
                "feature": row.get("feature"),
                "target": row.get("target"),
                "feature_lag_days": int(_round(row.get("feature_lag_days"), 0) or 0),
                "n": int(_round(row.get("n"), 0) or 0),
                "pearson": _metric(row, "pearson"),
                "spearman": _metric(row, "spearman"),
            }
        )
    return rows


def build_statistical_market_context_artifact(
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    report_path = report_path or DEFAULT_STATISTICAL_MARKET_STRUCTURE_REPORT_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_dir = _resolve_path((report.get("paths") or {}).get("run_dir"))

    distribution = _load_table(report, "distribution_summary", report_dir=report_dir)
    numeric = _load_table(report, "numeric_conditional_distributions", report_dir=report_dir)
    categorical = _load_table(report, "categorical_conditional_distributions", report_dir=report_dir)
    target_corr = _load_table(report, "target_correlations", report_dir=report_dir)
    pca_explained = _load_table(report, "pca_explained_variance", report_dir=report_dir)
    pca_loadings = _load_table(report, "pca_top_loadings", report_dir=report_dir)
    macro_target_corr = _load_table(report, "macro_target_correlations", report_dir=report_dir)
    macro_shock = _load_table(report, "macro_shock_distributions", report_dir=report_dir)
    macro_interactions = _load_table(report, "macro_interaction_distributions", report_dir=report_dir)
    macro_pca_explained = _load_table(report, "macro_pca_explained_variance", report_dir=report_dir)
    macro_pca_loadings = _load_table(report, "macro_pca_top_loadings", report_dir=report_dir)
    macro_lead_lag = _load_table(report, "macro_lead_lag_correlations", report_dir=report_dir)

    pc1 = {}
    if not pca_explained.empty:
        row = pca_explained.iloc[0]
        pc1 = {
            "component": row.get("component"),
            "explained_variance_ratio": _metric(row, "explained_variance_ratio"),
            "cumulative_variance_ratio": _metric(row, "cumulative_variance_ratio"),
            "top_loadings": pca_loadings[pca_loadings.get("component").astype(str).eq("PC1")]
            .head(8)
            .to_dict(orient="records")
            if not pca_loadings.empty and "component" in pca_loadings
            else [],
        }

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "source": "statistical_market_structure_study",
        "source_run_id": report.get("run_id"),
        "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "source_report_json": str(report_path),
        "source_report_pdf": (report.get("paths") or {}).get("latest_pdf") or (report.get("paths") or {}).get("pdf"),
        "coverage": report.get("coverage", {}),
        "baseline": _distribution_baseline(distribution),
        "numeric_bucket_priors": _numeric_priors(numeric),
        "categorical_bucket_priors": _categorical_priors(categorical),
        "target_correlations": _top_target_correlations(target_corr),
        "pca": {"pc1": pc1},
        "macro_context": {
            "shock_priors": _macro_shock_priors(macro_shock),
            "interaction_priors": _macro_interaction_priors(macro_interactions),
            "target_correlations": _top_target_correlations(macro_target_corr),
            "lead_lag_notes": _macro_lead_lag_notes(macro_lead_lag),
            "pca": {"pc1": _macro_pca_context(macro_pca_explained, macro_pca_loadings)},
        },
        "application_rules": {
            "min_bucket_n": 50,
            "expanded_abs_move_delta_bps": 20.0,
            "high_abs_move_delta_bps": 50.0,
            "compressed_abs_move_delta_bps": -20.0,
            "directional_mean_edge_bps": 8.0,
            "directional_hit_edge": 0.56,
            "conflict_hit_edge": 0.46,
            "tail_risk_size_cap": 0.80,
            "elevated_risk_size_cap": 0.90,
            "macro_expanded_abs_move_delta_bps": 35.0,
            "macro_high_abs_move_delta_bps": 70.0,
            "macro_directional_mean_edge_bps": 10.0,
            "macro_directional_hit_edge": 0.57,
            "macro_conflict_hit_edge": 0.46,
            "macro_tail_risk_size_cap": 0.85,
            "macro_elevated_risk_size_cap": 0.90,
            "macro_conflict_size_cap": 0.85,
            "macro_max_score_adjustment": 3,
            "macro_max_probability_adjustment": 0.02,
        },
    }
    return artifact


def write_statistical_market_context_artifact(
    *,
    report_path: Path | None = None,
    artifact_path: Path = DEFAULT_STATISTICAL_CONTEXT_ARTIFACT_PATH,
) -> dict[str, Any]:
    artifact = build_statistical_market_context_artifact(report_path=report_path)
    _atomic_write_json(artifact_path, artifact)
    return {"artifact": artifact, "artifact_path": str(artifact_path)}
