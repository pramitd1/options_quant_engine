"""Research-only statistical study of NIFTY, options, macro, and signal data.

This module builds a reproducible market-structure review pack from the daily
historical feature panel and the live signal-evaluation dataset.  It does not
change live engine behavior.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/options_quant_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from research.signal_evaluation.daily_research_report import (
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_INSIGHT_REPORT_PATH = (
    PROJECT_ROOT / "research" / "ml_research" / "historical_insights" / "latest_historical_insight_report.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research" / "ml_research" / "statistical_market_structure"

DISTRIBUTION_COLUMNS = [
    "ret_1d_bps",
    "gap_bps",
    "intraday_bps",
    "range_bps",
    "fwd_ret_1d_bps",
    "fwd_ret_3d_bps",
    "fwd_ret_5d_bps",
    "fwd_abs_ret_1d_bps",
    "next_day_range_bps",
    "india_vix_change_24h",
    "india_vix_level",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
    "pcr_oi",
    "pcr_volume",
    "pcr_chg_oi",
    "near_atm_pcr_oi",
    "near_atm_pcr_volume",
    "atm_straddle_pct",
    "max_pain_dist_pct",
    "max_pain_abs_dist_pct",
    "call_wall_dist_pct",
    "put_wall_dist_pct",
    "wall_width_pct",
]

STRUCTURE_FEATURES = [
    "ret_1d_bps",
    "ret_5d_bps",
    "ret_20d_bps",
    "gap_bps",
    "intraday_bps",
    "range_bps",
    "realized_vol_5d",
    "realized_vol_20d",
    "realized_vol_60d",
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "vix_change_24h",
    "india_vix_change_24h",
    "india_vix_level",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
    "nifty50_realized_vol_5d",
    "nifty50_realized_vol_30d",
    "banknifty_realized_vol_5d",
    "banknifty_realized_vol_30d",
    "front_dte",
    "pcr_oi",
    "pcr_volume",
    "pcr_chg_oi",
    "near_atm_pcr_oi",
    "near_atm_pcr_volume",
    "oi_top5_concentration",
    "atm_straddle_pct",
    "max_pain_dist_pct",
    "max_pain_abs_dist_pct",
    "call_wall_dist_pct",
    "put_wall_dist_pct",
    "wall_width_pct",
]

TARGET_COLUMNS = [
    "fwd_ret_1d_bps",
    "fwd_ret_3d_bps",
    "fwd_ret_5d_bps",
    "fwd_abs_ret_1d_bps",
    "fwd_abs_ret_3d_bps",
    "fwd_abs_ret_5d_bps",
    "next_day_range_bps",
]

CONDITIONAL_NUMERIC_FEATURES = [
    "india_vix_level",
    "india_vix_change_24h",
    "pcr_oi",
    "pcr_volume",
    "near_atm_pcr_oi",
    "atm_straddle_pct",
    "max_pain_abs_dist_pct",
    "wall_width_pct",
    "ret_20d_bps",
    "realized_vol_20d",
    "usdinr_change_24h",
    "sp500_change_24h",
]

CONDITIONAL_CATEGORICAL_FEATURES = [
    "weekday",
    "month",
    "expiry_bucket",
    "india_vix_bucket",
    "pcr_oi_bucket",
    "trend_20d_bucket",
    "is_expiry_weekday_thursday",
    "near_call_wall",
    "near_put_wall",
    "macro_major_event",
]

SIGNAL_OUTCOME_COLUMNS = [
    "signed_return_15m_bps",
    "signed_return_30m_bps",
    "signed_return_60m_bps",
    "primary_outcome_return_bps",
]

SIGNAL_SCORE_COLUMNS = [
    "trade_strength",
    "hybrid_move_probability",
    "move_probability",
    "signal_confidence_score",
    "activation_score",
    "historical_context_score_adjustment",
    "historical_context_probability_adjustment",
    "historical_interaction_score_adjustment",
]

MACRO_FEATURES = [
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "vix_change_24h",
    "india_vix_change_24h",
    "india_vix_level",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
]

MACRO_SHOCK_FEATURES = [
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

MACRO_TARGET_COLUMNS = [
    "fwd_ret_1d_bps",
    "fwd_abs_ret_1d_bps",
    "next_day_range_bps",
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return round(val, digits)


def _safe_numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(index=index, dtype="float64")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
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


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata")


def default_signal_dataset_path() -> Path | None:
    if DEFAULT_CUMULATIVE_DATASET_PATH.exists():
        return DEFAULT_CUMULATIVE_DATASET_PATH
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    return None


def resolve_daily_feature_panel_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    if not DEFAULT_HISTORICAL_INSIGHT_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Missing latest historical insight report. Run scripts/ops/run_historical_insight_mining.py first."
        )
    report = json.loads(DEFAULT_HISTORICAL_INSIGHT_REPORT_PATH.read_text(encoding="utf-8"))
    panel_ref = (report.get("paths") or {}).get("daily_feature_panel")
    if not panel_ref:
        raise FileNotFoundError("Historical insight report does not point to a daily feature panel.")
    panel_path = Path(panel_ref)
    if not panel_path.is_absolute():
        panel_path = PROJECT_ROOT / panel_path
    if not panel_path.exists():
        raise FileNotFoundError(f"Daily feature panel does not exist: {panel_path}")
    return panel_path


def load_daily_feature_panel(path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    panel_path = resolve_daily_feature_panel_path(path)
    panel = pd.read_csv(panel_path, low_memory=False)
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.tz_localize(None)
        panel = panel.sort_values("date", kind="mergesort").reset_index(drop=True)
    return panel, panel_path


def load_signal_dataset(path: Path | None = None) -> tuple[pd.DataFrame, Path | None]:
    dataset_path = path if path is not None else default_signal_dataset_path()
    if dataset_path is None or not dataset_path.exists():
        return pd.DataFrame(), None
    frame = pd.read_csv(dataset_path, low_memory=False)
    for column in ["signal_timestamp", "timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            break
    return frame, dataset_path


def available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def build_distribution_summary(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in available_columns(frame, columns or DISTRIBUTION_COLUMNS):
        values = _safe_numeric(frame[column]).dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        jb_stat = None
        jb_pvalue = None
        if len(values) >= 8 and values.nunique() >= 3:
            try:
                jb = stats.jarque_bera(values)
                jb_stat = _round(jb.statistic, 4)
                jb_pvalue = _round(jb.pvalue, 6)
            except Exception:
                pass
        abs_median = values.abs().median()
        tail_ratio = None
        if abs_median and math.isfinite(float(abs_median)) and float(abs_median) > 0:
            tail_ratio = _round(values.abs().quantile(0.99) / abs_median, 4)
        rows.append(
            {
                "feature": column,
                "n": int(len(values)),
                "missing": int(frame[column].isna().sum()),
                "mean": _round(values.mean(), 4),
                "median": _round(values.median(), 4),
                "std": _round(values.std(), 4),
                "skew": _round(values.skew(), 4),
                "excess_kurtosis": _round(values.kurtosis(), 4),
                "p01": _round(quantiles.loc[0.01], 4),
                "p05": _round(quantiles.loc[0.05], 4),
                "p25": _round(quantiles.loc[0.25], 4),
                "p75": _round(quantiles.loc[0.75], 4),
                "p95": _round(quantiles.loc[0.95], 4),
                "p99": _round(quantiles.loc[0.99], 4),
                "positive_rate": _round((values > 0).mean(), 4),
                "abs_mean": _round(values.abs().mean(), 4),
                "tail_ratio_p99_abs_to_median_abs": tail_ratio,
                "jarque_bera_stat": jb_stat,
                "jarque_bera_pvalue": jb_pvalue,
            }
        )
    return pd.DataFrame(rows)


def _winsorized_numeric_frame(frame: pd.DataFrame, features: list[str], min_count: int = 250) -> pd.DataFrame:
    cols = []
    for feature in available_columns(frame, features):
        series = _safe_numeric(frame[feature])
        if int(series.notna().sum()) >= min_count and series.nunique(dropna=True) >= 3:
            low = series.quantile(0.01)
            high = series.quantile(0.99)
            if pd.notna(low) and pd.notna(high) and low < high:
                series = series.clip(low, high)
            cols.append(series.rename(feature))
    if not cols:
        return pd.DataFrame(index=frame.index)
    return pd.concat(cols, axis=1)


def build_covariance_structure(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    *,
    min_count: int = 250,
) -> dict[str, pd.DataFrame]:
    data = _winsorized_numeric_frame(frame, features or STRUCTURE_FEATURES, min_count=min_count)
    if data.empty:
        empty = pd.DataFrame()
        return {"feature_frame": empty, "pearson": empty, "spearman": empty, "covariance_z": empty}
    pearson = data.corr(method="pearson", min_periods=max(30, min_count // 4))
    spearman = data.corr(method="spearman", min_periods=max(30, min_count // 4))
    standardized = (data - data.mean(skipna=True)) / data.std(skipna=True).replace(0, np.nan)
    covariance_z = standardized.cov(min_periods=max(30, min_count // 4))
    return {
        "feature_frame": data,
        "pearson": pearson,
        "spearman": spearman,
        "covariance_z": covariance_z,
    }


def build_target_correlation_table(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    min_count: int = 250,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in available_columns(frame, features or STRUCTURE_FEATURES):
        x = _safe_numeric(frame[feature])
        for target in available_columns(frame, targets or TARGET_COLUMNS):
            y = _safe_numeric(frame[target])
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < min_count or x.loc[mask].nunique() < 3 or y.loc[mask].nunique() < 3:
                continue
            pearson = x.loc[mask].corr(y.loc[mask], method="pearson")
            spearman = x.loc[mask].corr(y.loc[mask], method="spearman")
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "pearson": _round(pearson, 5),
                    "spearman": _round(spearman, 5),
                    "abs_spearman": _round(abs(float(spearman)), 5) if pd.notna(spearman) else None,
                    "abs_pearson": _round(abs(float(pearson)), 5) if pd.notna(pearson) else None,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = out[["abs_spearman", "abs_pearson"]].max(axis=1)
    return out.sort_values(["rank_score", "n"], ascending=[False, False]).reset_index(drop=True)


def build_lead_lag_table(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    lags: range = range(0, 6),
    min_count: int = 250,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_targets = targets or ["fwd_ret_1d_bps", "next_day_range_bps"]
    for feature in available_columns(frame, features or STRUCTURE_FEATURES):
        base_x = _safe_numeric(frame[feature])
        for target in available_columns(frame, selected_targets):
            y = _safe_numeric(frame[target])
            for lag in lags:
                x = base_x.shift(lag)
                mask = x.notna() & y.notna()
                n = int(mask.sum())
                if n < min_count or x.loc[mask].nunique() < 3 or y.loc[mask].nunique() < 3:
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "feature_lag_days": int(lag),
                        "n": n,
                        "spearman": _round(x.loc[mask].corr(y.loc[mask], method="spearman"), 5),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values(["abs_spearman", "n"], ascending=[False, False]).reset_index(drop=True)


def build_rolling_correlation_summary(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    window: int = 252,
    min_periods: int = 90,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_features = available_columns(
        frame,
        features
        or [
            "india_vix_level",
            "india_vix_change_24h",
            "pcr_oi",
            "pcr_volume",
            "atm_straddle_pct",
            "max_pain_abs_dist_pct",
            "ret_20d_bps",
            "realized_vol_20d",
            "usdinr_change_24h",
            "sp500_change_24h",
        ],
    )
    for feature in selected_features:
        x = _safe_numeric(frame[feature])
        for target in available_columns(frame, targets or ["fwd_ret_1d_bps", "next_day_range_bps"]):
            y = _safe_numeric(frame[target])
            rolling = x.rolling(window=window, min_periods=min_periods).corr(y)
            vals = rolling.dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "window": window,
                    "n_windows": int(len(vals)),
                    "latest": _round(vals.iloc[-1], 5),
                    "median": _round(vals.median(), 5),
                    "p10": _round(vals.quantile(0.10), 5),
                    "p90": _round(vals.quantile(0.90), 5),
                    "sign_flip_rate": _round((np.sign(vals).diff().fillna(0) != 0).mean(), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "feature"]).reset_index(drop=True) if rows else pd.DataFrame()


def build_numeric_conditional_table(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    buckets: int = 5,
    min_count: int = 60,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_targets = available_columns(frame, targets or ["fwd_ret_1d_bps", "fwd_abs_ret_1d_bps", "next_day_range_bps"])
    for feature in available_columns(frame, features or CONDITIONAL_NUMERIC_FEATURES):
        values = _safe_numeric(frame[feature])
        valid = frame.loc[values.notna()].copy()
        if len(valid) < min_count * 2 or values.nunique(dropna=True) < buckets:
            continue
        try:
            valid["_bucket"] = pd.qcut(values.loc[valid.index], buckets, duplicates="drop")
        except ValueError:
            continue
        for bucket, sub in valid.groupby("_bucket", observed=False):
            row: dict[str, Any] = {
                "feature": feature,
                "bucket": str(bucket),
                "n": int(len(sub)),
                "feature_min": _round(_safe_numeric(sub[feature]).min(), 4),
                "feature_max": _round(_safe_numeric(sub[feature]).max(), 4),
            }
            for target in selected_targets:
                vals = _safe_numeric(sub[target]).dropna()
                row[f"{target}_n"] = int(len(vals))
                if len(vals) >= min_count:
                    row[f"{target}_mean"] = _round(vals.mean(), 4)
                    row[f"{target}_median"] = _round(vals.median(), 4)
                    row[f"{target}_hit_positive"] = _round((vals > 0).mean(), 4)
                    row[f"{target}_abs_mean"] = _round(vals.abs().mean(), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_abs = _safe_numeric(frame.get("fwd_abs_ret_1d_bps"), index=frame.index).mean()
    if math.isfinite(float(base_abs)) if pd.notna(base_abs) else False:
        out["fwd_abs_ret_1d_mean_delta_vs_base"] = out.get("fwd_abs_ret_1d_bps_mean", np.nan) - float(base_abs)
    return out.reset_index(drop=True)


def build_categorical_conditional_table(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    min_count: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_targets = available_columns(frame, targets or ["fwd_ret_1d_bps", "fwd_abs_ret_1d_bps", "next_day_range_bps"])
    for feature in available_columns(frame, features or CONDITIONAL_CATEGORICAL_FEATURES):
        for value, sub in frame.groupby(feature, dropna=False):
            if len(sub) < min_count:
                continue
            row: dict[str, Any] = {"feature": feature, "bucket": str(value), "n": int(len(sub))}
            for target in selected_targets:
                vals = _safe_numeric(sub[target]).dropna()
                row[f"{target}_n"] = int(len(vals))
                if len(vals) >= min_count:
                    row[f"{target}_mean"] = _round(vals.mean(), 4)
                    row[f"{target}_median"] = _round(vals.median(), 4)
                    row[f"{target}_hit_positive"] = _round((vals > 0).mean(), 4)
                    row[f"{target}_abs_mean"] = _round(vals.abs().mean(), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_abs = _safe_numeric(frame.get("fwd_abs_ret_1d_bps"), index=frame.index).mean()
    if pd.notna(base_abs) and math.isfinite(float(base_abs)):
        out["fwd_abs_ret_1d_mean_delta_vs_base"] = out.get("fwd_abs_ret_1d_bps_mean", np.nan) - float(base_abs)
    return out.sort_values(["feature", "n"], ascending=[True, False]).reset_index(drop=True)


def build_pca_summary(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    *,
    min_count: int = 500,
    max_components: int = 6,
) -> dict[str, Any]:
    data = _winsorized_numeric_frame(frame, features or STRUCTURE_FEATURES, min_count=min_count)
    if data.shape[1] < 3 or len(data) < min_count:
        return {"status": "INSUFFICIENT_DATA", "feature_count": int(data.shape[1]), "row_count": int(len(data))}
    imputed = SimpleImputer(strategy="median").fit_transform(data)
    scaled = StandardScaler().fit_transform(imputed)
    component_count = min(max_components, data.shape[1], scaled.shape[0])
    pca = PCA(n_components=component_count, random_state=0)
    transformed = pca.fit_transform(scaled)
    explained = [
        {
            "component": f"PC{idx + 1}",
            "explained_variance_ratio": _round(value, 5),
            "cumulative_variance_ratio": _round(np.cumsum(pca.explained_variance_ratio_)[idx], 5),
        }
        for idx, value in enumerate(pca.explained_variance_ratio_)
    ]
    loading_rows = []
    for idx in range(component_count):
        loadings = pd.Series(pca.components_[idx], index=data.columns).sort_values(key=lambda s: s.abs(), ascending=False)
        for feature, value in loadings.head(8).items():
            loading_rows.append({"component": f"PC{idx + 1}", "feature": feature, "loading": _round(value, 5)})
    scores = pd.DataFrame(transformed[:, : min(3, component_count)], columns=[f"PC{i + 1}" for i in range(min(3, component_count))])
    return {
        "status": "OK",
        "row_count": int(len(data)),
        "feature_count": int(data.shape[1]),
        "features": list(data.columns),
        "explained_variance": explained,
        "top_loadings": loading_rows,
        "scores_head": scores.head(10).round(5).to_dict(orient="records"),
    }


def build_signal_dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"status": "MISSING", "row_count": 0}
    timestamp_col = "signal_timestamp" if "signal_timestamp" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    summary: dict[str, Any] = {
        "status": "OK",
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
    }
    if timestamp_col:
        timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
        summary["date_min"] = timestamps.min().isoformat() if timestamps.notna().any() else None
        summary["date_max"] = timestamps.max().isoformat() if timestamps.notna().any() else None
    if "direction" in frame.columns:
        summary["direction_counts"] = frame["direction"].astype(str).str.upper().value_counts(dropna=False).head(10).to_dict()
    outcome_rows: list[dict[str, Any]] = []
    for column in available_columns(frame, SIGNAL_OUTCOME_COLUMNS):
        values = _safe_numeric(frame[column]).dropna()
        if values.empty:
            continue
        outcome_rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": _round(values.mean(), 4),
                "median": _round(values.median(), 4),
                "std": _round(values.std(), 4),
                "p05": _round(values.quantile(0.05), 4),
                "p95": _round(values.quantile(0.95), 4),
                "positive_rate": _round((values > 0).mean(), 4),
            }
        )
    summary["outcome_distributions"] = outcome_rows

    corr_rows = []
    targets = available_columns(frame, SIGNAL_OUTCOME_COLUMNS + ["correct_15m", "correct_30m", "correct_60m"])
    for score_col in available_columns(frame, SIGNAL_SCORE_COLUMNS):
        x = _safe_numeric(frame[score_col])
        for target in targets:
            y = _safe_numeric(frame[target])
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 50 or x.loc[mask].nunique() < 3 or y.loc[mask].nunique() < 2:
                continue
            corr_rows.append(
                {
                    "score": score_col,
                    "target": target,
                    "n": int(mask.sum()),
                    "spearman": _round(x.loc[mask].corr(y.loc[mask], method="spearman"), 5),
                }
            )
    corr_table = pd.DataFrame(corr_rows)
    if not corr_table.empty:
        corr_table["abs_spearman"] = corr_table["spearman"].abs()
        corr_table = corr_table.sort_values(["abs_spearman", "n"], ascending=[False, False]).reset_index(drop=True)
    summary["score_outcome_correlations"] = corr_table.head(30).to_dict(orient="records") if not corr_table.empty else []
    return summary


def _zscore_series(series: pd.Series) -> pd.Series:
    values = _safe_numeric(series)
    std = values.std(skipna=True)
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return (values - values.mean(skipna=True)) / std


def _tercile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    values = _safe_numeric(series)
    try:
        return pd.qcut(values, len(labels), labels=labels, duplicates="drop").astype("object")
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="object")


def build_macro_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    index = out.index
    sp500 = _zscore_series(out["sp500_change_24h"]) if "sp500_change_24h" in out else pd.Series(np.nan, index=index)
    nasdaq = _zscore_series(out["nasdaq_change_24h"]) if "nasdaq_change_24h" in out else pd.Series(np.nan, index=index)
    us_vix = _zscore_series(out["vix_change_24h"]) if "vix_change_24h" in out else pd.Series(np.nan, index=index)
    usdinr = _zscore_series(out["usdinr_change_24h"]) if "usdinr_change_24h" in out else pd.Series(np.nan, index=index)
    rates = _zscore_series(out["us10y_change_bp"]) if "us10y_change_bp" in out else pd.Series(np.nan, index=index)
    oil = _zscore_series(out["oil_change_24h"]) if "oil_change_24h" in out else pd.Series(np.nan, index=index)
    copper = _zscore_series(out["copper_change_24h"]) if "copper_change_24h" in out else pd.Series(np.nan, index=index)
    gold = _zscore_series(out["gold_change_24h"]) if "gold_change_24h" in out else pd.Series(np.nan, index=index)

    out["macro_risk_sentiment_score"] = pd.concat([sp500, nasdaq, -us_vix], axis=1).mean(axis=1, skipna=True)
    out["macro_fx_rates_stress_score"] = pd.concat([usdinr, rates], axis=1).mean(axis=1, skipna=True)
    out["macro_commodity_pressure_score"] = pd.concat([oil, copper, gold], axis=1).mean(axis=1, skipna=True)
    out["macro_risk_bucket"] = _tercile_bucket(
        out["macro_risk_sentiment_score"],
        ["risk_off", "neutral", "risk_on"],
    )
    out["macro_fx_rates_bucket"] = _tercile_bucket(
        out["macro_fx_rates_stress_score"],
        ["easing", "neutral", "stress"],
    )
    out["macro_commodity_bucket"] = _tercile_bucket(
        out["macro_commodity_pressure_score"],
        ["commodity_down", "neutral", "commodity_up"],
    )
    return out


def build_macro_shock_table(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    *,
    min_count: int = 60,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_targets = available_columns(frame, targets or MACRO_TARGET_COLUMNS)
    for feature in available_columns(frame, features or MACRO_SHOCK_FEATURES):
        values = _safe_numeric(frame[feature])
        valid = values.dropna()
        if len(valid) < min_count * 3 or valid.nunique() < 10:
            continue
        low = valid.quantile(0.10)
        high = valid.quantile(0.90)
        buckets = [
            ("bottom_decile", values <= low, low),
            ("middle_80", values.gt(low) & values.lt(high), None),
            ("top_decile", values >= high, high),
        ]
        for bucket, mask, threshold in buckets:
            sub = frame.loc[mask & values.notna()]
            if len(sub) < min_count:
                continue
            row: dict[str, Any] = {
                "feature": feature,
                "bucket": bucket,
                "n": int(len(sub)),
                "threshold": _round(threshold, 4),
                "feature_mean": _round(values.loc[sub.index].mean(), 4),
                "feature_median": _round(values.loc[sub.index].median(), 4),
            }
            for target in selected_targets:
                vals = _safe_numeric(sub[target]).dropna()
                row[f"{target}_n"] = int(len(vals))
                if len(vals) >= min_count:
                    row[f"{target}_mean"] = _round(vals.mean(), 4)
                    row[f"{target}_median"] = _round(vals.median(), 4)
                    row[f"{target}_hit_positive"] = _round((vals > 0).mean(), 4)
                    row[f"{target}_abs_mean"] = _round(vals.abs().mean(), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_abs = _safe_numeric(frame.get("fwd_abs_ret_1d_bps"), index=frame.index).mean()
    if pd.notna(base_abs) and math.isfinite(float(base_abs)):
        out["fwd_abs_ret_1d_mean_delta_vs_base"] = out.get("fwd_abs_ret_1d_bps_mean", np.nan) - float(base_abs)
    return out.sort_values(["feature", "bucket"]).reset_index(drop=True)


def build_macro_interaction_table(
    frame: pd.DataFrame,
    *,
    min_count: int = 50,
) -> pd.DataFrame:
    working = build_macro_factor_panel(frame)
    interaction_pairs = [
        ("macro_risk_bucket", "india_vix_bucket"),
        ("macro_risk_bucket", "trend_20d_bucket"),
        ("macro_fx_rates_bucket", "trend_20d_bucket"),
        ("macro_commodity_bucket", "india_vix_bucket"),
        ("macro_commodity_bucket", "trend_20d_bucket"),
    ]
    rows: list[dict[str, Any]] = []
    for left, right in interaction_pairs:
        if left not in working.columns or right not in working.columns:
            continue
        grouped = working.dropna(subset=[left, right]).groupby([left, right], observed=False, dropna=False)
        for (left_value, right_value), sub in grouped:
            if len(sub) < min_count:
                continue
            row: dict[str, Any] = {
                "left_feature": left,
                "left_bucket": str(left_value),
                "right_feature": right,
                "right_bucket": str(right_value),
                "interaction": f"{left}={left_value}|{right}={right_value}",
                "n": int(len(sub)),
            }
            for target in available_columns(working, MACRO_TARGET_COLUMNS):
                vals = _safe_numeric(sub[target]).dropna()
                row[f"{target}_n"] = int(len(vals))
                if len(vals) >= min_count:
                    row[f"{target}_mean"] = _round(vals.mean(), 4)
                    row[f"{target}_median"] = _round(vals.median(), 4)
                    row[f"{target}_hit_positive"] = _round((vals > 0).mean(), 4)
                    row[f"{target}_abs_mean"] = _round(vals.abs().mean(), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_abs = _safe_numeric(frame.get("fwd_abs_ret_1d_bps"), index=frame.index).mean()
    if pd.notna(base_abs) and math.isfinite(float(base_abs)):
        out["fwd_abs_ret_1d_mean_delta_vs_base"] = out.get("fwd_abs_ret_1d_bps_mean", np.nan) - float(base_abs)
    return out.sort_values(["left_feature", "right_feature", "n"], ascending=[True, True, False]).reset_index(drop=True)


def _top_abs_pairs(correlation: pd.DataFrame, limit: int = 25) -> list[dict[str, Any]]:
    if correlation.empty:
        return []
    rows = []
    cols = list(correlation.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.isna(value):
                continue
            rows.append({"left": left, "right": right, "correlation": _round(value, 5), "abs_correlation": abs(float(value))})
    return sorted(rows, key=lambda item: item["abs_correlation"], reverse=True)[:limit]


def _derive_review_findings(
    *,
    panel: pd.DataFrame,
    distribution: pd.DataFrame,
    target_correlations: pd.DataFrame,
    covariance: dict[str, pd.DataFrame],
    numeric_conditional: pd.DataFrame,
    categorical_conditional: pd.DataFrame,
    pca_summary: dict[str, Any],
    signal_summary: dict[str, Any],
    macro_correlations: pd.DataFrame | None = None,
    macro_shocks: pd.DataFrame | None = None,
    macro_interactions: pd.DataFrame | None = None,
    macro_pca_summary: dict[str, Any] | None = None,
) -> list[str]:
    findings: list[str] = []
    ret_row = distribution.loc[distribution["feature"].eq("ret_1d_bps")].head(1)
    if not ret_row.empty:
        kurt = ret_row["excess_kurtosis"].iloc[0]
        skew = ret_row["skew"].iloc[0]
        findings.append(
            f"NIFTY daily returns are non-normal in this sample: skew={skew}, excess kurtosis={kurt}; tail-aware thresholds are more appropriate than normal z-score thresholds."
        )
    range_row = distribution.loc[distribution["feature"].eq("range_bps")].head(1)
    if not range_row.empty:
        findings.append(
            f"Daily range has median {range_row['median'].iloc[0]} bps and p95 {range_row['p95'].iloc[0]} bps; this is a natural baseline for expected-move and hold-time calibration."
        )
    if not target_correlations.empty:
        top = target_correlations.iloc[0]
        findings.append(
            f"Strongest univariate rank association is {top['feature']} vs {top['target']} at Spearman {top['spearman']} (n={int(top['n'])}); treat this as conditioning evidence, not a standalone signal."
        )
    pairs = _top_abs_pairs(covariance.get("spearman", pd.DataFrame()), limit=3)
    if pairs:
        pair_text = "; ".join(f"{p['left']}~{p['right']}={p['correlation']}" for p in pairs)
        findings.append(f"Covariance structure is clustered rather than independent. Top rank-correlation pairs: {pair_text}.")
    if pca_summary.get("status") == "OK":
        pc1 = (pca_summary.get("explained_variance") or [{}])[0]
        findings.append(
            f"First principal component explains {pc1.get('explained_variance_ratio')} of standardized feature variance, so a compact regime vector is statistically plausible."
        )
    if macro_correlations is not None and not macro_correlations.empty:
        top_macro = macro_correlations.iloc[0]
        findings.append(
            f"Macro/global layer strongest rank association is {top_macro['feature']} vs {top_macro['target']} at Spearman {top_macro['spearman']} (n={int(top_macro['n'])})."
        )
    if macro_pca_summary and macro_pca_summary.get("status") == "OK":
        macro_pc1 = (macro_pca_summary.get("explained_variance") or [{}])[0]
        findings.append(
            f"Macro PCA PC1 explains {macro_pc1.get('explained_variance_ratio')} of macro feature variance, useful for separating global risk sentiment from local option structure."
        )
    cond = pd.DataFrame()
    if not numeric_conditional.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in numeric_conditional:
        cond = numeric_conditional.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
    if not cond.empty:
        best = cond.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).iloc[0]
        findings.append(
            f"Largest next-day movement uplift appears in {best['feature']} bucket {best['bucket']}: +{_round(best['fwd_abs_ret_1d_mean_delta_vs_base'], 2)} bps over baseline abs move."
        )
    cat = pd.DataFrame()
    if not categorical_conditional.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in categorical_conditional:
        cat = categorical_conditional.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
    if not cat.empty:
        best_cat = cat.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).iloc[0]
        findings.append(
            f"Categorical regimes also matter: {best_cat['feature']}={best_cat['bucket']} shows +{_round(best_cat['fwd_abs_ret_1d_mean_delta_vs_base'], 2)} bps next-day abs-move uplift."
        )
    if macro_shocks is not None and not macro_shocks.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in macro_shocks:
        macro_move = macro_shocks.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not macro_move.empty:
            best_macro = macro_move.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).iloc[0]
            findings.append(
                f"Largest macro shock abs-move uplift appears in {best_macro['feature']} {best_macro['bucket']}: +{_round(best_macro['fwd_abs_ret_1d_mean_delta_vs_base'], 2)} bps over baseline."
            )
    if macro_interactions is not None and not macro_interactions.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in macro_interactions:
        macro_int = macro_interactions.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not macro_int.empty:
            best_interaction = macro_int.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).iloc[0]
            findings.append(
                f"Macro interaction with largest abs-move uplift is {best_interaction['interaction']}: +{_round(best_interaction['fwd_abs_ret_1d_mean_delta_vs_base'], 2)} bps over baseline."
            )
    if signal_summary.get("status") == "OK":
        findings.append(
            f"Live signal dataset is available with {signal_summary.get('row_count')} rows, so intraday outcome studies can be layered on top of the historical daily structure."
        )
    if not findings:
        findings.append("No stable findings were detected; data coverage or feature availability should be improved before engine use.")
    return findings


def build_statistical_market_structure_report(
    panel: pd.DataFrame,
    *,
    panel_path: Path | str | None = None,
    signal_frame: pd.DataFrame | None = None,
    signal_dataset_path: Path | str | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    working = panel.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.tz_localize(None)
        working = working.sort_values("date", kind="mergesort").reset_index(drop=True)

    distribution = build_distribution_summary(working)
    covariance = build_covariance_structure(working)
    target_correlations = build_target_correlation_table(working)
    lead_lag = build_lead_lag_table(working)
    rolling = build_rolling_correlation_summary(working)
    numeric_conditional = build_numeric_conditional_table(working)
    categorical_conditional = build_categorical_conditional_table(working)
    pca_summary = build_pca_summary(working)
    signal_summary = build_signal_dataset_summary(signal_frame if signal_frame is not None else pd.DataFrame())
    macro_panel = build_macro_factor_panel(working)
    macro_covariance = build_covariance_structure(macro_panel, features=MACRO_FEATURES, min_count=250)
    macro_correlations = build_target_correlation_table(
        macro_panel,
        features=MACRO_FEATURES,
        targets=MACRO_TARGET_COLUMNS,
        min_count=250,
    )
    macro_lead_lag = build_lead_lag_table(
        macro_panel,
        features=MACRO_FEATURES,
        targets=["fwd_ret_1d_bps", "next_day_range_bps"],
        lags=range(-3, 6),
        min_count=250,
    )
    macro_shocks = build_macro_shock_table(macro_panel)
    macro_interactions = build_macro_interaction_table(macro_panel)
    macro_pca_summary = build_pca_summary(macro_panel, features=MACRO_FEATURES, min_count=500, max_components=5)

    date_min = working["date"].min() if "date" in working.columns else pd.NaT
    date_max = working["date"].max() if "date" in working.columns else pd.NaT
    option_rows = int(_safe_numeric(working.get("pcr_oi"), index=working.index).notna().sum()) if len(working) else 0
    macro_rows = int(_safe_numeric(working.get("india_vix_level"), index=working.index).notna().sum()) if len(working) else 0
    run_id = run_id or _now_ist().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "report_type": "statistical_market_structure_study",
        "run_id": run_id,
        "generated_at": _now_ist().isoformat(),
        "runtime_config_changed": False,
        "execution_behavior_changed": False,
        "research_only": True,
        "coverage": {
            "panel_rows": int(len(working)),
            "date_min": date_min.date().isoformat() if pd.notna(date_min) else None,
            "date_max": date_max.date().isoformat() if pd.notna(date_max) else None,
            "option_feature_rows": option_rows,
            "macro_global_rows": macro_rows,
            "panel_path": str(panel_path) if panel_path is not None else None,
            "signal_dataset_path": str(signal_dataset_path) if signal_dataset_path is not None else None,
            "signal_rows": int(signal_summary.get("row_count", 0) or 0),
        },
        "distribution_highlights": distribution.head(25).to_dict(orient="records"),
        "top_target_correlations": target_correlations.head(30).to_dict(orient="records") if not target_correlations.empty else [],
        "top_covariance_pairs": _top_abs_pairs(covariance.get("spearman", pd.DataFrame()), limit=25),
        "macro_summary": {
            "feature_count": len(available_columns(macro_panel, MACRO_FEATURES)),
            "top_macro_correlations": macro_correlations.head(20).to_dict(orient="records") if not macro_correlations.empty else [],
            "top_macro_covariance_pairs": _top_abs_pairs(macro_covariance.get("spearman", pd.DataFrame()), limit=20),
            "pca_summary": macro_pca_summary,
        },
        "pca_summary": pca_summary,
        "signal_summary": signal_summary,
    }
    report["findings"] = _derive_review_findings(
        panel=working,
        distribution=distribution,
        target_correlations=target_correlations,
        covariance=covariance,
        numeric_conditional=numeric_conditional,
        categorical_conditional=categorical_conditional,
        pca_summary=pca_summary,
        signal_summary=signal_summary,
        macro_correlations=macro_correlations,
        macro_shocks=macro_shocks,
        macro_interactions=macro_interactions,
        macro_pca_summary=macro_pca_summary,
    )
    report["recommended_research_next_steps"] = [
        "Run walk-forward stability checks on the top covariance and conditional-distribution features.",
        "Run a timing-safe macro alignment review before using overseas-market or rates features in live decisions.",
        "Build regime-specific outcome tables from the same panel before any live threshold wiring.",
        "Join matured intraday signal outcomes to daily regime features to study whether live calls behave like historical daily priors.",
        "Track covariance drift weekly; unstable feature clusters should become monitoring inputs rather than hard signal rules.",
    ]
    tables = {
        "distribution_summary": distribution,
        "target_correlations": target_correlations,
        "lead_lag_correlations": lead_lag,
        "rolling_correlation_summary": rolling,
        "numeric_conditional_distributions": numeric_conditional,
        "categorical_conditional_distributions": categorical_conditional,
        "macro_target_correlations": macro_correlations,
        "macro_lead_lag_correlations": macro_lead_lag,
        "macro_shock_distributions": macro_shocks,
        "macro_interaction_distributions": macro_interactions,
        "macro_pearson_correlation_matrix": macro_covariance.get("pearson", pd.DataFrame()),
        "macro_spearman_correlation_matrix": macro_covariance.get("spearman", pd.DataFrame()),
        "macro_covariance_z_matrix": macro_covariance.get("covariance_z", pd.DataFrame()),
        "macro_pca_explained_variance": pd.DataFrame(macro_pca_summary.get("explained_variance", [])),
        "macro_pca_top_loadings": pd.DataFrame(macro_pca_summary.get("top_loadings", [])),
        "pearson_correlation_matrix": covariance.get("pearson", pd.DataFrame()),
        "spearman_correlation_matrix": covariance.get("spearman", pd.DataFrame()),
        "covariance_z_matrix": covariance.get("covariance_z", pd.DataFrame()),
        "pca_explained_variance": pd.DataFrame(pca_summary.get("explained_variance", [])),
        "pca_top_loadings": pd.DataFrame(pca_summary.get("top_loadings", [])),
        "signal_outcome_distributions": pd.DataFrame(signal_summary.get("outcome_distributions", [])),
        "signal_score_outcome_correlations": pd.DataFrame(signal_summary.get("score_outcome_correlations", [])),
    }
    return report, tables


def _wrapped_text(lines: list[str], width: int = 105) -> str:
    import textwrap

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(str(line), width=width, replace_whitespace=False))
    return "\n".join(wrapped)


def _text_page(pdf: PdfPages, title: str, lines: list[str], subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=22, fontweight="bold", color="#18212f", va="top")
    if subtitle:
        ax.text(0.04, 0.89, subtitle, fontsize=11, color="#526070", va="top")
        y = 0.83
    else:
        y = 0.86
    ax.text(
        0.04,
        y,
        _wrapped_text(lines),
        fontsize=10.5,
        color="#253142",
        va="top",
        linespacing=1.45,
        family="DejaVu Sans",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _plot_distribution_page(pdf: PdfPages, panel: pd.DataFrame) -> None:
    ret = _safe_numeric(panel.get("ret_1d_bps"), index=panel.index).dropna()
    range_bps = _safe_numeric(panel.get("range_bps"), index=panel.index).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Distribution Baseline", fontsize=18, fontweight="bold")
    if not ret.empty:
        axes[0, 0].hist(ret, bins=80, color="#2f6fbb", alpha=0.82)
        axes[0, 0].axvline(ret.median(), color="#101820", linewidth=1.5, label="median")
        axes[0, 0].axvline(ret.quantile(0.05), color="#c0392b", linestyle="--", linewidth=1.2, label="5/95 pct")
        axes[0, 0].axvline(ret.quantile(0.95), color="#c0392b", linestyle="--", linewidth=1.2)
        axes[0, 0].set_title("Daily Return (bps)")
        axes[0, 0].legend(fontsize=8)
        stats.probplot(ret.sample(min(len(ret), 3000), random_state=7), dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Daily Return Q-Q Plot")
    if not range_bps.empty:
        axes[1, 0].hist(range_bps, bins=80, color="#1d9a8a", alpha=0.82)
        axes[1, 0].axvline(range_bps.median(), color="#101820", linewidth=1.5, label="median")
        axes[1, 0].axvline(range_bps.quantile(0.95), color="#c0392b", linestyle="--", linewidth=1.2, label="95 pct")
        axes[1, 0].set_title("Daily High-Low Range (bps)")
        axes[1, 0].legend(fontsize=8)
    abs_ret = ret.abs() if not ret.empty else pd.Series(dtype="float64")
    if not abs_ret.empty:
        axes[1, 1].hist(abs_ret, bins=80, color="#e0a32e", alpha=0.84)
        axes[1, 1].axvline(abs_ret.quantile(0.95), color="#c0392b", linestyle="--", linewidth=1.2, label="95 pct")
        axes[1, 1].set_title("Absolute Daily Return (bps)")
        axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _volatility_series_for_display(series: pd.Series) -> tuple[pd.Series, str]:
    values = _safe_numeric(series)
    valid = values.dropna()
    if valid.empty:
        return values, ""
    # Historical realized-vol columns are annualized decimals, while India VIX
    # arrives as a percent-like index level.  Convert decimals to percent for
    # chart readability without mutating the analytical tables.
    if valid.abs().quantile(0.95) <= 2.5:
        return values * 100.0, "%"
    return values, "index"


def _plot_timeseries_page(pdf: PdfPages, panel: pd.DataFrame) -> None:
    if "date" not in panel.columns:
        return
    dates = pd.to_datetime(panel["date"], errors="coerce")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    fig.suptitle("Time Series Context", fontsize=18, fontweight="bold")
    if "close" in panel.columns:
        axes[0].plot(dates, _safe_numeric(panel["close"]), color="#263f6a", linewidth=1.1)
        axes[0].set_title("NIFTY Close")
    vol_lines = []
    if "realized_vol_20d" in panel.columns:
        realized_vol, realized_unit = _volatility_series_for_display(panel["realized_vol_20d"])
        line = axes[1].plot(
            dates,
            realized_vol,
            color="#2f6fbb",
            linewidth=1.05,
            label=f"NIFTY realized vol 20d ({realized_unit})",
            alpha=0.9,
        )
        vol_lines.extend(line)
        axes[1].set_ylabel("Realized vol (%)", color="#2f6fbb")
        axes[1].tick_params(axis="y", labelcolor="#2f6fbb")
    if "india_vix_level" in panel.columns:
        vix_axis = axes[1].twinx()
        line = vix_axis.plot(
            dates,
            _safe_numeric(panel["india_vix_level"]),
            color="#c0392b",
            linewidth=1.0,
            label="India VIX",
            alpha=0.86,
        )
        vol_lines.extend(line)
        vix_axis.set_ylabel("India VIX", color="#c0392b")
        vix_axis.tick_params(axis="y", labelcolor="#c0392b")
        vix_axis.spines["top"].set_visible(False)
    axes[1].set_title("Volatility Regime (Dual Axis)")
    if vol_lines:
        axes[1].legend(vol_lines, [line.get_label() for line in vol_lines], fontsize=8, ncol=2, loc="upper left")
    for col, color, label in [
        ("pcr_oi", "#1d9a8a", "PCR OI"),
        ("atm_straddle_pct", "#e0a32e", "ATM straddle %"),
    ]:
        if col in panel.columns:
            axes[2].plot(dates, _safe_numeric(panel[col]), color=color, linewidth=1.0, label=label, alpha=0.9)
    axes[2].set_title("Option Structure")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_heatmap_page(pdf: PdfPages, matrix: pd.DataFrame, title: str, *, max_features: int = 20) -> None:
    if matrix.empty:
        return
    counts = matrix.abs().sum(axis=1).sort_values(ascending=False).head(max_features).index
    mat = matrix.loc[counts, counts].astype(float)
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Correlation")
    pdf.savefig(fig)
    plt.close(fig)


def _plot_target_correlation_page(pdf: PdfPages, target_correlations: pd.DataFrame) -> None:
    if target_correlations.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Feature Association With Forward Outcomes", fontsize=18, fontweight="bold")
    for ax, target in zip(axes, ["fwd_ret_1d_bps", "next_day_range_bps"], strict=False):
        subset = target_correlations[target_correlations["target"].eq(target)].copy()
        if subset.empty:
            ax.axis("off")
            ax.set_title(target)
            continue
        subset = subset.reindex(subset["spearman"].abs().sort_values(ascending=True).tail(12).index)
        colors = ["#c0392b" if value < 0 else "#2f6fbb" for value in subset["spearman"]]
        ax.barh(subset["feature"], subset["spearman"], color=colors, alpha=0.86)
        ax.axvline(0, color="#101820", linewidth=0.8)
        ax.set_title(target)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_lead_lag_page(pdf: PdfPages, lead_lag: pd.DataFrame) -> None:
    if lead_lag.empty:
        return
    focus = lead_lag[lead_lag["target"].eq("fwd_ret_1d_bps")].copy()
    if focus.empty:
        focus = lead_lag.copy()
    features = focus.groupby("feature")["abs_spearman"].max().sort_values(ascending=False).head(12).index
    pivot = focus[focus["feature"].isin(features)].pivot_table(
        index="feature",
        columns="feature_lag_days",
        values="spearman",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    im = ax.imshow(pivot, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_title("Lag Persistence vs Forward 1D Return", fontsize=18, fontweight="bold")
    ax.set_xlabel("Feature lag in days (0 = same session feature)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman")
    pdf.savefig(fig)
    plt.close(fig)


def _plot_macro_target_correlation_page(pdf: PdfPages, macro_correlations: pd.DataFrame) -> None:
    if macro_correlations.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Global Macro Associations", fontsize=18, fontweight="bold")
    for ax, target in zip(axes, ["fwd_ret_1d_bps", "next_day_range_bps"], strict=False):
        subset = macro_correlations[macro_correlations["target"].eq(target)].copy()
        if subset.empty:
            ax.axis("off")
            ax.set_title(target)
            continue
        subset = subset.reindex(subset["spearman"].abs().sort_values(ascending=True).tail(10).index)
        colors = ["#c0392b" if value < 0 else "#2f6fbb" for value in subset["spearman"]]
        ax.barh(subset["feature"], subset["spearman"], color=colors, alpha=0.86)
        ax.axvline(0, color="#101820", linewidth=0.8)
        ax.set_title(target)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_macro_lead_lag_page(pdf: PdfPages, macro_lead_lag: pd.DataFrame) -> None:
    if macro_lead_lag.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Global Macro Lead-Lag Timing", fontsize=18, fontweight="bold")
    plotted = False
    for ax, target in zip(axes, ["fwd_ret_1d_bps", "next_day_range_bps"], strict=False):
        focus = macro_lead_lag[macro_lead_lag["target"].eq(target)].copy()
        if focus.empty:
            ax.axis("off")
            ax.set_title(target)
            continue
        features = focus.groupby("feature")["abs_spearman"].max().sort_values(ascending=False).head(9).index
        pivot = focus[focus["feature"].isin(features)].pivot_table(
            index="feature",
            columns="feature_lag_days",
            values="spearman",
            aggfunc="mean",
        )
        if pivot.empty:
            ax.axis("off")
            ax.set_title(target)
            continue
        im = ax.imshow(pivot, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
        ax.set_title(target)
        ax.set_xlabel("Feature lag days; negative = diagnostic future alignment")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(col) for col in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.68, label="Spearman")
        plotted = True
    if plotted:
        pdf.savefig(fig)
    plt.close(fig)


def _plot_macro_shock_page(
    pdf: PdfPages,
    macro_shocks: pd.DataFrame,
    macro_interactions: pd.DataFrame,
) -> None:
    if macro_shocks.empty and macro_interactions.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Global Macro Shock And Interaction Buckets", fontsize=18, fontweight="bold")
    plotted = False
    if not macro_shocks.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in macro_shocks:
        subset = macro_shocks.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not subset.empty:
            subset["label"] = subset["feature"] + "\n" + subset["bucket"].astype(str)
            subset = subset.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).head(10)
            axes[0].barh(subset["label"][::-1], subset["fwd_abs_ret_1d_mean_delta_vs_base"][::-1], color="#2f6fbb")
            axes[0].set_title("Macro Shock Buckets by Next-Day Abs-Move Uplift")
            axes[0].grid(axis="x", alpha=0.18)
            plotted = True
    if not macro_interactions.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in macro_interactions:
        subset = macro_interactions.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not subset.empty:
            subset["label"] = subset["left_bucket"].astype(str) + " x " + subset["right_bucket"].astype(str)
            subset = subset.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).head(10)
            axes[1].barh(subset["label"][::-1], subset["fwd_abs_ret_1d_mean_delta_vs_base"][::-1], color="#1d9a8a")
            axes[1].set_title("Macro Interaction Buckets by Next-Day Abs-Move Uplift")
            axes[1].grid(axis="x", alpha=0.18)
            plotted = True
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    if plotted:
        pdf.savefig(fig)
    plt.close(fig)


def _plot_macro_pca_page(
    pdf: PdfPages,
    macro_pca_summary: dict[str, Any],
    macro_corr_matrix: pd.DataFrame,
) -> None:
    if macro_pca_summary.get("status") != "OK" and macro_corr_matrix.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Global Macro Factor Structure", fontsize=18, fontweight="bold")
    explained = pd.DataFrame(macro_pca_summary.get("explained_variance", []))
    loadings = pd.DataFrame(macro_pca_summary.get("top_loadings", []))
    if not explained.empty:
        axes[0].bar(explained["component"], explained["explained_variance_ratio"], color="#2f6fbb", alpha=0.86)
        axes[0].plot(explained["component"], explained["cumulative_variance_ratio"], color="#c0392b", marker="o", label="cumulative")
        axes[0].set_title("Macro PCA Explained Variance")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.18)
    else:
        axes[0].axis("off")
    if not loadings.empty:
        pc1 = loadings[loadings["component"].eq("PC1")].copy()
        pc1 = pc1.reindex(pc1["loading"].abs().sort_values(ascending=True).index)
        colors = ["#c0392b" if value < 0 else "#2f6fbb" for value in pc1["loading"]]
        axes[1].barh(pc1["feature"], pc1["loading"], color=colors, alpha=0.86)
        axes[1].axvline(0, color="#101820", linewidth=0.8)
        axes[1].set_title("Largest Macro PC1 Loadings")
        axes[1].grid(axis="x", alpha=0.18)
    elif not macro_corr_matrix.empty:
        mat = macro_corr_matrix.astype(float)
        im = axes[1].imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[1].set_xticks(range(len(mat.columns)))
        axes[1].set_yticks(range(len(mat.index)))
        axes[1].set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
        axes[1].set_yticklabels(mat.index, fontsize=7)
        axes[1].set_title("Macro Correlation Matrix")
        fig.colorbar(im, ax=axes[1], shrink=0.68, label="Correlation")
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_conditional_page(
    pdf: PdfPages,
    numeric_conditional: pd.DataFrame,
    categorical_conditional: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Conditional Distribution Clues", fontsize=18, fontweight="bold")
    plotted = False
    if not numeric_conditional.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in numeric_conditional:
        subset = numeric_conditional.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not subset.empty:
            subset["label"] = subset["feature"] + "\n" + subset["bucket"].astype(str)
            subset = subset.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).head(10)
            axes[0].barh(subset["label"][::-1], subset["fwd_abs_ret_1d_mean_delta_vs_base"][::-1], color="#1d9a8a")
            axes[0].set_title("Top Numeric Buckets by Next-Day Abs-Move Uplift")
            axes[0].grid(axis="x", alpha=0.18)
            plotted = True
    if not categorical_conditional.empty and "fwd_abs_ret_1d_mean_delta_vs_base" in categorical_conditional:
        subset = categorical_conditional.dropna(subset=["fwd_abs_ret_1d_mean_delta_vs_base"]).copy()
        if not subset.empty:
            subset["label"] = subset["feature"] + "=" + subset["bucket"].astype(str)
            subset = subset.sort_values("fwd_abs_ret_1d_mean_delta_vs_base", ascending=False).head(10)
            axes[1].barh(subset["label"][::-1], subset["fwd_abs_ret_1d_mean_delta_vs_base"][::-1], color="#e0a32e")
            axes[1].set_title("Top Categorical Buckets by Next-Day Abs-Move Uplift")
            axes[1].grid(axis="x", alpha=0.18)
            plotted = True
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    if plotted:
        pdf.savefig(fig)
    plt.close(fig)


def _plot_pca_page(pdf: PdfPages, pca_summary: dict[str, Any]) -> None:
    if pca_summary.get("status") != "OK":
        return
    explained = pd.DataFrame(pca_summary.get("explained_variance", []))
    loadings = pd.DataFrame(pca_summary.get("top_loadings", []))
    if explained.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Principal Component Structure", fontsize=18, fontweight="bold")
    axes[0].bar(explained["component"], explained["explained_variance_ratio"], color="#2f6fbb", alpha=0.86)
    axes[0].plot(explained["component"], explained["cumulative_variance_ratio"], color="#c0392b", marker="o", label="cumulative")
    axes[0].set_ylim(0, min(1.0, max(0.25, float(explained["cumulative_variance_ratio"].max()) + 0.08)))
    axes[0].set_title("Explained Variance")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.18)
    if not loadings.empty:
        pc1 = loadings[loadings["component"].eq("PC1")].copy()
        pc1 = pc1.reindex(pc1["loading"].abs().sort_values(ascending=True).index)
        colors = ["#c0392b" if value < 0 else "#2f6fbb" for value in pc1["loading"]]
        axes[1].barh(pc1["feature"], pc1["loading"], color=colors, alpha=0.86)
        axes[1].axvline(0, color="#101820", linewidth=0.8)
        axes[1].set_title("Largest PC1 Loadings")
        axes[1].grid(axis="x", alpha=0.18)
        axes[1].tick_params(axis="y", labelsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_signal_page(pdf: PdfPages, signal_summary: dict[str, Any]) -> None:
    if signal_summary.get("status") != "OK":
        return
    outcomes = pd.DataFrame(signal_summary.get("outcome_distributions", []))
    corrs = pd.DataFrame(signal_summary.get("score_outcome_correlations", []))
    if outcomes.empty and corrs.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle("Live Signal Dataset Context", fontsize=18, fontweight="bold")
    if not outcomes.empty:
        axes[0].bar(outcomes["metric"], outcomes["positive_rate"], color="#1d9a8a", alpha=0.86)
        axes[0].axhline(0.5, color="#101820", linewidth=0.8, linestyle="--")
        axes[0].set_title("Outcome Positive Rate")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].set_ylim(0, 1)
    else:
        axes[0].axis("off")
    if not corrs.empty:
        subset = corrs.sort_values("abs_spearman", ascending=True).tail(12)
        labels = subset["score"] + "\nvs " + subset["target"]
        colors = ["#c0392b" if value < 0 else "#2f6fbb" for value in subset["spearman"]]
        axes[1].barh(labels, subset["spearman"], color=colors, alpha=0.86)
        axes[1].axvline(0, color="#101820", linewidth=0.8)
        axes[1].set_title("Score vs Outcome Spearman")
        axes[1].tick_params(axis="y", labelsize=8)
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf_report(
    path: Path,
    *,
    report: dict[str, Any],
    panel: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with PdfPages(tmp_path) as pdf:
            coverage = report.get("coverage", {})
            title_lines = [
                f"Run ID: {report.get('run_id')}",
                f"Generated at: {report.get('generated_at')}",
                f"Daily panel: {coverage.get('panel_rows')} rows from {coverage.get('date_min')} to {coverage.get('date_max')}",
                f"Option feature days: {coverage.get('option_feature_rows')}; macro/global rows: {coverage.get('macro_global_rows')}",
                f"Live signal rows: {coverage.get('signal_rows')}",
                "",
                "Executive findings:",
            ]
            title_lines.extend([f"- {finding}" for finding in report.get("findings", [])])
            _text_page(
                pdf,
                "Statistical Market Structure Review",
                title_lines,
                subtitle="Research-only study of NIFTY spot, option structure, macro/global variables, and live signal outcomes.",
            )
            _plot_distribution_page(pdf, panel)
            _plot_timeseries_page(pdf, panel)
            _plot_heatmap_page(pdf, tables.get("spearman_correlation_matrix", pd.DataFrame()), "Spearman Correlation Structure")
            _plot_target_correlation_page(pdf, tables.get("target_correlations", pd.DataFrame()))
            _plot_lead_lag_page(pdf, tables.get("lead_lag_correlations", pd.DataFrame()))
            _plot_macro_target_correlation_page(pdf, tables.get("macro_target_correlations", pd.DataFrame()))
            _plot_macro_lead_lag_page(pdf, tables.get("macro_lead_lag_correlations", pd.DataFrame()))
            _plot_macro_shock_page(
                pdf,
                tables.get("macro_shock_distributions", pd.DataFrame()),
                tables.get("macro_interaction_distributions", pd.DataFrame()),
            )
            _plot_macro_pca_page(
                pdf,
                report.get("macro_summary", {}).get("pca_summary", {}),
                tables.get("macro_spearman_correlation_matrix", pd.DataFrame()),
            )
            _plot_conditional_page(
                pdf,
                tables.get("numeric_conditional_distributions", pd.DataFrame()),
                tables.get("categorical_conditional_distributions", pd.DataFrame()),
            )
            _plot_pca_page(pdf, report.get("pca_summary", {}))
            _plot_signal_page(pdf, report.get("signal_summary", {}))
            _text_page(
                pdf,
                "Research Use",
                [
                    "This report is deliberately descriptive. It should guide hypotheses, walk-forward tests, and monitoring.",
                    "",
                    "Recommended next steps:",
                    *[f"- {item}" for item in report.get("recommended_research_next_steps", [])],
                    "",
                    "Important caveat: correlations and conditional buckets are not causal proof. Any live-engine use should be preceded by stability checks and out-of-sample validation.",
                ],
            )
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def render_markdown_report(report: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    coverage = report.get("coverage", {})
    lines = [
        "# Statistical Market Structure Review",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Daily panel: `{coverage.get('panel_rows')}` rows from `{coverage.get('date_min')}` to `{coverage.get('date_max')}`",
        f"- Option feature days: `{coverage.get('option_feature_rows')}`",
        f"- Macro/global rows: `{coverage.get('macro_global_rows')}`",
        f"- Live signal rows: `{coverage.get('signal_rows')}`",
        "",
        "## Findings",
    ]
    lines.extend([f"- {finding}" for finding in report.get("findings", [])])
    lines.extend(["", "## Top Target Associations"])
    corr = tables.get("target_correlations", pd.DataFrame())
    if corr is not None and not corr.empty:
        show = corr[["feature", "target", "n", "spearman", "pearson"]].head(20)
        lines.append(_frame_to_markdown(show))
    else:
        lines.append("_No target correlations met sample thresholds._")
    lines.extend(["", "## Top Covariance Pairs"])
    pairs = pd.DataFrame(report.get("top_covariance_pairs", []))
    if not pairs.empty:
        lines.append(_frame_to_markdown(pairs[["left", "right", "correlation"]].head(20)))
    else:
        lines.append("_No covariance pairs available._")
    lines.extend(["", "## Global Macro Associations"])
    macro_corr = tables.get("macro_target_correlations", pd.DataFrame())
    if macro_corr is not None and not macro_corr.empty:
        show = macro_corr[["feature", "target", "n", "spearman", "pearson"]].head(20)
        lines.append(_frame_to_markdown(show))
    else:
        lines.append("_No macro correlations met sample thresholds._")
    lines.extend(["", "## Global Macro Shock Buckets"])
    macro_shocks = tables.get("macro_shock_distributions", pd.DataFrame())
    if macro_shocks is not None and not macro_shocks.empty:
        show_cols = [
            "feature",
            "bucket",
            "n",
            "fwd_ret_1d_bps_mean",
            "fwd_abs_ret_1d_bps_mean",
            "next_day_range_bps_mean",
            "fwd_abs_ret_1d_mean_delta_vs_base",
        ]
        show = macro_shocks[[col for col in show_cols if col in macro_shocks.columns]].head(20)
        lines.append(_frame_to_markdown(show))
    else:
        lines.append("_No macro shock buckets available._")
    lines.extend(["", "## Next Research Steps"])
    lines.extend([f"- {item}" for item in report.get("recommended_research_next_steps", [])])
    lines.extend(["", "## Artifact Paths"])
    for key, value in (report.get("paths") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    working = frame.copy()
    columns = list(working.columns)
    lines = ["|" + "|".join(str(column) for column in columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in working.iterrows():
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    return "\n".join(lines)


def write_statistical_market_structure_artifacts(
    *,
    panel_path: Path | None = None,
    signal_dataset_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str | None = None,
) -> dict[str, Any]:
    panel, resolved_panel_path = load_daily_feature_panel(panel_path)
    signal_frame, resolved_signal_path = load_signal_dataset(signal_dataset_path)
    report, tables = build_statistical_market_structure_report(
        panel,
        panel_path=resolved_panel_path,
        signal_frame=signal_frame,
        signal_dataset_path=resolved_signal_path,
    )
    run_id = str(report["run_id"])
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base_name = report_name or "statistical_market_structure_report"

    csv_paths: dict[str, str] = {}
    for key, table in tables.items():
        if table is None or table.empty:
            continue
        csv_path = run_dir / f"{key}.csv"
        _atomic_write_csv(csv_path, table.reset_index() if key.endswith("_matrix") else table)
        csv_paths[key] = _rel(csv_path)

    json_path = run_dir / f"{base_name}.json"
    markdown_path = run_dir / f"{base_name}.md"
    pdf_path = run_dir / f"{base_name}.pdf"
    latest_json_path = output_dir / "latest_statistical_market_structure_report.json"
    latest_markdown_path = output_dir / "latest_statistical_market_structure_report.md"
    latest_pdf_path = output_dir / "latest_statistical_market_structure_report.pdf"

    report["paths"] = {
        "run_dir": _rel(run_dir),
        "json": _rel(json_path),
        "markdown": _rel(markdown_path),
        "pdf": _rel(pdf_path),
        "latest_json": _rel(latest_json_path),
        "latest_markdown": _rel(latest_markdown_path),
        "latest_pdf": _rel(latest_pdf_path),
        **csv_paths,
    }
    _atomic_write_json(json_path, report)
    markdown = render_markdown_report(report, tables)
    _atomic_write_text(markdown_path, markdown)
    write_pdf_report(pdf_path, report=report, panel=panel, tables=tables)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(latest_json_path, report)
    _atomic_write_text(latest_markdown_path, markdown)
    shutil.copyfile(pdf_path, latest_pdf_path)
    return {
        "report": report,
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "pdf_path": str(pdf_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_pdf_path": str(latest_pdf_path),
    }
