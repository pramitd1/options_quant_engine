"""
Module: reporting.py

Purpose:
    Implement reporting utilities for signal evaluation, reporting, or research diagnostics.

Role in the System:
    Part of the research layer that records signal-evaluation datasets and diagnostic reports.

Key Outputs:
    Signal-evaluation datasets, reports, and comparison artifacts.

Downstream Usage:
    Consumed by tuning, governance reviews, and post-trade analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.reports import build_research_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_EVALUATION_REPORTS_DIR = PROJECT_ROOT / "research" / "signal_evaluation" / "reports"

HORIZON_DEFINITIONS = (
    ("5m", "realized_return_5m", "signed_return_5m_bps", "correct_5m"),
    ("15m", "realized_return_15m", "signed_return_15m_bps", "correct_15m"),
    ("30m", "realized_return_30m", "signed_return_30m_bps", "correct_30m"),
    ("60m", "realized_return_60m", "signed_return_60m_bps", "correct_60m"),
    ("120m", None, "signed_return_120m_bps", "correct_120m"),
    ("session_close", None, "signed_return_session_close_bps", "correct_session_close"),
)

SCORE_FIELDS = (
    "direction_score",
    "magnitude_score",
    "timing_score",
    "tradeability_score",
    "composite_signal_score",
)

REGIME_FIELDS = (
    "signal_regime",
    "macro_regime",
    "gamma_regime",
)

COVERAGE_FIELDS = (
    "signal_timestamp",
    "symbol",
    "direction",
    "trade_strength",
    "move_probability",
    "direction_score",
    "magnitude_score",
    "timing_score",
    "tradeability_score",
    "composite_signal_score",
    "correct_60m",
    "signed_return_60m_bps",
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `reporting` module. The module sits in the research layer that evaluates signals, curates datasets, and renders reports.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (float | None): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    """
    Purpose:
        Process round or none for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (Any): Input associated with value.
        digits (int): Input associated with digits.
    
    Returns:
        float | None: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    coerced = _safe_float(value, None)
    if coerced is None or pd.isna(coerced):
        return None
    return round(float(coerced), digits)


def _frame_with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process frame with timestamp for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    enriched = frame.copy()
    if "signal_timestamp" in enriched.columns:
        enriched["signal_timestamp"] = pd.to_datetime(enriched["signal_timestamp"], errors="coerce")
    return enriched


def _evaluation_period(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Purpose:
        Process evaluation period for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or "signal_timestamp" not in frame.columns:
        return {"start": None, "end": None, "trading_days": 0}

    timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return {"start": None, "end": None, "trading_days": 0}

    return {
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "trading_days": int(timestamps.dt.normalize().nunique()),
    }


def _counts_with_optional_hit_rate(frame: pd.DataFrame, field_name: str, top_n: int) -> list[dict[str, Any]]:
    """
    Purpose:
        Process counts with optional hit rate for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        field_name (str): Human-readable name for field.
        top_n (int): Input associated with top n.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or field_name not in frame.columns:
        return []

    working = frame.copy()
    if "correct_60m" in working.columns:
        working["correct_60m"] = pd.to_numeric(working["correct_60m"], errors="coerce")

    rows = []
    for field_value, group in working.dropna(subset=[field_name]).groupby(field_name, dropna=False):
        rows.append(
            {
                field_name: str(field_value),
                "signal_count": int(len(group)),
                "hit_rate_60m": _round_or_none(group.get("correct_60m", pd.Series(dtype=float)).mean(), 4),
                "avg_composite_signal_score": _round_or_none(
                    pd.to_numeric(group.get("composite_signal_score", pd.Series(dtype=float)), errors="coerce").mean(),
                    2,
                ),
            }
        )

    rows = sorted(rows, key=lambda row: row["signal_count"], reverse=True)
    return rows[:top_n]


def _signal_frequency_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Purpose:
        Process signal frequency summary for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or "signal_timestamp" not in frame.columns:
        return {
            "average_signals_per_day": 0.0,
            "median_signals_per_day": 0.0,
            "max_signals_per_day": 0,
            "active_days": 0,
        }

    timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return {
            "average_signals_per_day": 0.0,
            "median_signals_per_day": 0.0,
            "max_signals_per_day": 0,
            "active_days": 0,
        }

    daily_counts = timestamps.dt.normalize().value_counts().sort_index()
    return {
        "average_signals_per_day": _round_or_none(daily_counts.mean(), 4) or 0.0,
        "median_signals_per_day": _round_or_none(daily_counts.median(), 4) or 0.0,
        "max_signals_per_day": int(daily_counts.max()),
        "active_days": int(len(daily_counts)),
    }


def _score_statistics(frame: pd.DataFrame, field_name: str) -> dict[str, Any]:
    """
    Purpose:
        Process score statistics for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        field_name (str): Human-readable name for field.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    series = pd.to_numeric(frame.get(field_name, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p25": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(series.count()),
        "mean": _round_or_none(series.mean(), 4),
        "median": _round_or_none(series.median(), 4),
        "std": _round_or_none(series.std(ddof=0), 4),
        "min": _round_or_none(series.min(), 4),
        "p25": _round_or_none(series.quantile(0.25), 4),
        "p75": _round_or_none(series.quantile(0.75), 4),
        "max": _round_or_none(series.max(), 4),
    }


def _horizon_performance(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Purpose:
        Process horizon performance for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    rows = []
    for label, raw_field, signed_field, correct_field in HORIZON_DEFINITIONS:
        if raw_field is None and signed_field not in frame.columns and correct_field not in frame.columns:
            continue
        if raw_field is not None and raw_field not in frame.columns and signed_field not in frame.columns:
            continue

        raw_series = pd.to_numeric(frame.get(raw_field, pd.Series(dtype=float)), errors="coerce") if raw_field else pd.Series(dtype=float)
        signed_series = pd.to_numeric(frame.get(signed_field, pd.Series(dtype=float)), errors="coerce")
        correct_series = pd.to_numeric(frame.get(correct_field, pd.Series(dtype=float)), errors="coerce")
        sample_count = int(max(raw_series.notna().sum(), signed_series.notna().sum(), correct_series.notna().sum(), 0))

        rows.append(
            {
                "horizon": label,
                "sample_count": sample_count,
                "avg_realized_return": _round_or_none(raw_series.mean(), 6),
                "avg_signed_return_bps": _round_or_none(signed_series.mean(), 4),
                "hit_rate": _round_or_none(correct_series.mean(), 4),
            }
        )
    return rows


def _score_bucket_performance(frame: pd.DataFrame, score_field: str = "composite_signal_score") -> list[dict[str, Any]]:
    """
    Purpose:
        Process score bucket performance for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        score_field (str): Input associated with score field.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or score_field not in frame.columns:
        return []

    working = frame.copy()
    working[score_field] = pd.to_numeric(working[score_field], errors="coerce")
    working["correct_60m"] = pd.to_numeric(working.get("correct_60m", pd.Series(index=working.index)), errors="coerce")
    working["signed_return_60m_bps"] = pd.to_numeric(
        working.get("signed_return_60m_bps", pd.Series(index=working.index)),
        errors="coerce",
    )
    working = working.dropna(subset=[score_field])
    if working.empty:
        return []

    bins = [float("-inf"), 35.0, 50.0, 65.0, 80.0, float("inf")]
    labels = ["0_34", "35_49", "50_64", "65_79", "80_100"]
    working["score_bucket"] = pd.cut(working[score_field], bins=bins, labels=labels)

    rows = []
    for bucket, group in working.groupby("score_bucket", dropna=False, observed=False):
        if pd.isna(bucket):
            continue
        rows.append(
            {
                "score_bucket": str(bucket),
                "signal_count": int(len(group)),
                "hit_rate_60m": _round_or_none(group["correct_60m"].mean(), 4),
                "avg_signed_return_60m_bps": _round_or_none(group["signed_return_60m_bps"].mean(), 4),
                "avg_composite_signal_score": _round_or_none(group[score_field].mean(), 2),
            }
        )
    return rows


def _regime_breakdown(frame: pd.DataFrame, top_n: int) -> dict[str, list[dict[str, Any]]]:
    """
    Purpose:
        Process regime breakdown for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        top_n (int): Input associated with top n.
    
    Returns:
        dict[str, list[dict[str, Any]]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    breakdown = {}
    for field_name in REGIME_FIELDS:
        breakdown[field_name] = _counts_with_optional_hit_rate(frame, field_name, top_n=top_n)
    return breakdown


def _coverage_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Purpose:
        Process coverage summary for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return []

    rows = []
    for field_name in COVERAGE_FIELDS:
        if field_name not in frame.columns:
            continue
        non_null = int(frame[field_name].notna().sum())
        rows.append(
            {
                "field_name": field_name,
                "non_null_count": non_null,
                "coverage_ratio": _round_or_none(non_null / max(len(frame), 1), 4),
                "missing_count": int(len(frame) - non_null),
            }
        )
    return rows


def _outcome_status_counts(frame: pd.DataFrame) -> dict[str, int]:
    """
    Purpose:
        Process outcome status counts for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, int]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or "outcome_status" not in frame.columns:
        return {}
    counts = frame["outcome_status"].fillna("UNKNOWN").value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _sanitize_value(value: Any) -> Any:
    """
    Purpose:
        Sanitize value before it is consumed downstream.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (Any): Input associated with value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _research_tables(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """
    Purpose:
        Process research tables for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, list[dict[str, Any]]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    tables = {}
    for section_name, section_df in build_research_report(frame).items():
        tables[section_name] = _sanitize_value(section_df.to_dict(orient="records"))
    return tables


def build_signal_evaluation_summary(
    frame: pd.DataFrame,
    *,
    production_pack_name: str | None = None,
    dataset_path: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Purpose:
        Build the signal evaluation summary used by downstream components.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        production_pack_name (str | None): Human-readable name for production pack.
        dataset_path (str | None): Input associated with dataset path.
        top_n (int): Input associated with top n.
    
    Returns:
        dict[str, Any]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    enriched = _frame_with_timestamp(frame)
    score_statistics = {
        field_name: _score_statistics(enriched, field_name)
        for field_name in SCORE_FIELDS
    }

    return _sanitize_value(
        {
        "report_type": "signal_evaluation_summary",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "production_pack_name": production_pack_name,
        "dataset_path": dataset_path,
        "evaluation_period": _evaluation_period(enriched),
        "total_signal_count": int(len(enriched)),
        "signals_by_symbol": _counts_with_optional_hit_rate(enriched, "symbol", top_n=top_n),
        "signals_by_direction": _counts_with_optional_hit_rate(enriched, "direction", top_n=top_n),
        "signal_frequency": _signal_frequency_summary(enriched),
        "outcome_status_counts": _outcome_status_counts(enriched),
        "horizon_performance": _horizon_performance(enriched),
        "score_statistics": score_statistics,
        "score_bucket_performance": _score_bucket_performance(enriched),
        "regime_breakdown": _regime_breakdown(enriched, top_n=top_n),
        "data_coverage_summary": _coverage_summary(enriched),
        "research_tables": _research_tables(enriched),
        }
    )


def render_signal_evaluation_markdown(summary: dict[str, Any]) -> str:
    """
    Purpose:
        Render signal evaluation markdown for operator-facing or report output.
    
    Context:
        Public function within the research layer that records evaluation datasets and diagnostic reports. It exposes a reusable workflow step to other parts of the repository.
    
    Inputs:
        summary (dict[str, Any]): Summary payload being rendered or inspected.
    
    Returns:
        str: Side-effect-oriented result returned by the current workflow.
    
    Notes:
        The output is designed to remain serializable so evaluation artifacts can be replayed, audited, and compared across experiments.
    """
    period = summary.get("evaluation_period", {})
    signal_frequency = summary.get("signal_frequency", {})
    lines = [
        "# Current Signal Evaluation Report",
        "",
        f"- Generated at: {summary.get('generated_at')}",
        f"- Production pack: {summary.get('production_pack_name') or 'unknown'}",
        f"- Dataset path: {summary.get('dataset_path') or 'unknown'}",
        f"- Evaluation period: {period.get('start')} -> {period.get('end')}",
        f"- Trading days: {period.get('trading_days')}",
        f"- Total signal count: {summary.get('total_signal_count', 0)}",
        f"- Average signals per day: {signal_frequency.get('average_signals_per_day', 0.0)}",
        "",
        "## Signals By Symbol",
        "",
        "| Symbol | Signal Count | 60m Hit Rate | Avg Composite Score |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in summary.get("signals_by_symbol", []):
        lines.append(
            f"| {row.get('symbol')} | {row.get('signal_count')} | {row.get('hit_rate_60m')} | {row.get('avg_composite_signal_score')} |"
        )
    if not summary.get("signals_by_symbol"):
        lines.append("| (none) | 0 |  |  |")

    lines.extend(
        [
            "",
            "## Horizon Performance",
            "",
            "| Horizon | Sample Count | Avg Realized Return | Avg Signed Return (bps) | Hit Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("horizon_performance", []):
        lines.append(
            f"| {row.get('horizon')} | {row.get('sample_count')} | {row.get('avg_realized_return')} | {row.get('avg_signed_return_bps')} | {row.get('hit_rate')} |"
        )
    if not summary.get("horizon_performance"):
        lines.append("| (none) | 0 |  |  |  |")

    lines.extend(
        [
            "",
            "## Score Statistics",
            "",
            "| Score | Count | Mean | Median | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for score_name, payload in summary.get("score_statistics", {}).items():
        lines.append(
            f"| {score_name} | {payload.get('count')} | {payload.get('mean')} | {payload.get('median')} | {payload.get('std')} | {payload.get('min')} | {payload.get('max')} |"
        )

    lines.extend(
        [
            "",
            "## Score Bucket Performance",
            "",
            "| Score Bucket | Signal Count | 60m Hit Rate | Avg Signed Return 60m (bps) | Avg Composite Score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("score_bucket_performance", []):
        lines.append(
            f"| {row.get('score_bucket')} | {row.get('signal_count')} | {row.get('hit_rate_60m')} | {row.get('avg_signed_return_60m_bps')} | {row.get('avg_composite_signal_score')} |"
        )
    if not summary.get("score_bucket_performance"):
        lines.append("| (none) | 0 |  |  |  |")

    regime_breakdown = summary.get("regime_breakdown", {})
    for regime_name, rows in regime_breakdown.items():
        lines.extend(
            [
                "",
                f"## Regime Breakdown: {regime_name}",
                "",
                "| Regime | Signal Count | 60m Hit Rate | Avg Composite Score |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        if rows:
            for row in rows:
                lines.append(
                    f"| {row.get(regime_name)} | {row.get('signal_count')} | {row.get('hit_rate_60m')} | {row.get('avg_composite_signal_score')} |"
                )
        else:
            lines.append("| (none) | 0 |  |  |")

    return "\n".join(lines).strip() + "\n"


def _write_table_csvs(summary: dict[str, Any], base_dir: Path) -> dict[str, str]:
    """
    Purpose:
        Write table csvs to the appropriate output artifact.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        summary (dict[str, Any]): Input associated with summary.
        base_dir (Path): Input associated with base dir.
    
    Returns:
        dict[str, str]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    csv_paths: dict[str, str] = {}
    table_mappings = {
        "signals_by_symbol": summary.get("signals_by_symbol", []),
        "signals_by_direction": summary.get("signals_by_direction", []),
        "horizon_performance": summary.get("horizon_performance", []),
        "score_bucket_performance": summary.get("score_bucket_performance", []),
        "data_coverage_summary": summary.get("data_coverage_summary", []),
    }
    for name, rows in table_mappings.items():
        csv_path = base_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        csv_paths[name] = str(csv_path)

    for name, rows in summary.get("research_tables", {}).items():
        csv_path = base_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        csv_paths[name] = str(csv_path)

    return csv_paths


def write_signal_evaluation_report(
    frame: pd.DataFrame,
    *,
    production_pack_name: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | Path = SIGNAL_EVALUATION_REPORTS_DIR,
    report_name: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Purpose:
        Write signal evaluation report to the appropriate output artifact.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        production_pack_name (str | None): Human-readable name for production pack.
        dataset_path (str | None): Input associated with dataset path.
        output_dir (str | Path): Input associated with output dir.
        report_name (str | None): Human-readable name for report.
        top_n (int): Input associated with top n.
    
    Returns:
        None: The function communicates through side effects such as terminal output or persisted artifacts.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    summary = build_signal_evaluation_summary(
        frame,
        production_pack_name=production_pack_name,
        dataset_path=dataset_path,
        top_n=top_n,
    )

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_stem = report_name or f"signal_evaluation_{stamp}"
    report_dir = Path(output_dir) / report_stem
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "summary.json"
    markdown_path = report_dir / "summary.md"
    csv_dir = report_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_signal_evaluation_markdown(summary), encoding="utf-8")
    csv_paths = _write_table_csvs(summary, csv_dir)

    return {
        "report_name": report_stem,
        "report_dir": str(report_dir),
        "summary": summary,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "csv_paths": csv_paths,
    }
