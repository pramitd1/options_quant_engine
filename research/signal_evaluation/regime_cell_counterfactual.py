"""Counterfactual replay for proposed regime-cell adjustments.

This module is deliberately research-only. It replays captured signal rows
against the reliable 3-factor/4-factor regime-cell proposals and estimates
whether the proposed suppression and hold-time behavior would have helped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.daily_research_report import DEFAULT_CUMULATIVE_DATASET_PATH
from research.signal_evaluation.regime_cell_review import (
    DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    DEFAULT_REGIME_CELL_REVIEW_DIR,
    LATEST_REVIEW_JSON_FILENAME,
)
from research.signal_evaluation.regime_outcome_tables import _prepare_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "regime_cell_counterfactual"
)

LATEST_COUNTERFACTUAL_JSON_FILENAME = "latest_regime_cell_counterfactual.json"
LATEST_COUNTERFACTUAL_MARKDOWN_FILENAME = "latest_regime_cell_counterfactual.md"
LATEST_COUNTERFACTUAL_CELLS_CSV_FILENAME = "latest_regime_cell_counterfactual_cells.csv"
LATEST_COUNTERFACTUAL_DETAILS_CSV_FILENAME = "latest_regime_cell_counterfactual_details.csv"

DEFAULT_BASE_TRADE_STRENGTH_THRESHOLD = 55.0
DEFAULT_MIN_CELL_LABELS = 20

HORIZON_RETURN_COLUMNS = {
    "5m": "signed_return_5m_bps",
    "15m": "signed_return_15m_bps",
    "30m": "signed_return_30m_bps",
    "60m": "signed_return_60m_bps",
    "120m": "signed_return_120m_bps",
    "session_close": "signed_return_session_close_bps",
}

HORIZON_HIT_COLUMNS = {
    "5m": "correct_5m",
    "15m": "correct_15m",
    "30m": "correct_30m",
    "60m": "correct_60m",
    "120m": "correct_120m",
    "session_close": "correct_session_close",
}

DETAIL_COLUMNS = (
    "signal_id",
    "signal_timestamp",
    "symbol",
    "source",
    "mode",
    "trade_status",
    "direction",
    "gamma_regime",
    "volatility_regime",
    "macro_risk_bucket",
    "trade_strength",
    "matched_action_class",
    "matched_group_name",
    "matched_cell",
    "baseline_trade",
    "counterfactual_selected",
    "promotion_candidate",
    "counterfactual_decision",
    "proposal_horizon",
    "baseline_return_60m_bps",
    "counterfactual_return_bps",
    "counterfactual_delta_bps",
)


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
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


def _load_review(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _cell_key(cell: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    group_name = _text(cell.get("group_name"))
    fields = tuple(group_name.split("+"))
    values = tuple(_text(cell.get(field)) for field in fields)
    return group_name, values


def _proposal_indexes(review: dict[str, Any]) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    three: dict[tuple[str, ...], dict[str, Any]] = {}
    four: dict[tuple[str, ...], dict[str, Any]] = {}
    for cell in review.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        group_name, values = _cell_key(cell)
        if group_name == "gamma_regime+volatility_regime+direction":
            three[values] = cell
        elif group_name == "gamma_regime+volatility_regime+direction+macro_risk_bucket":
            four[values] = cell
    return three, four


def _match_proposal(row: pd.Series, three: dict[tuple[str, ...], dict[str, Any]], four: dict[tuple[str, ...], dict[str, Any]]) -> dict[str, Any] | None:
    four_key = (
        _text(row.get("gamma_regime")),
        _text(row.get("volatility_regime")),
        _text(row.get("direction")),
        _text(row.get("macro_risk_bucket")),
    )
    if four_key in four:
        return four[four_key]
    three_key = (
        _text(row.get("gamma_regime")),
        _text(row.get("volatility_regime")),
        _text(row.get("direction")),
    )
    return three.get(three_key)


def _return_for_horizon(row: pd.Series, horizon: str) -> float | None:
    column = HORIZON_RETURN_COLUMNS.get(horizon)
    return _safe_float(row.get(column), None) if column else None


def _hit_for_horizon(row: pd.Series, horizon: str) -> float | None:
    column = HORIZON_HIT_COLUMNS.get(horizon)
    return _safe_float(row.get(column), None) if column else None


def _metrics(returns: pd.Series, hits: pd.Series | None = None) -> dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    hits = pd.to_numeric(hits, errors="coerce").dropna() if hits is not None else pd.Series(dtype=float)
    wins = returns.loc[returns > 0]
    losses = returns.loc[returns <= 0]
    label_count = int(max(len(returns), len(hits)))
    return {
        "label_count": label_count,
        "hit_rate": _round(hits.mean()) if not hits.empty else None,
        "avg_return_bps": _round(returns.mean()) if not returns.empty else None,
        "median_return_bps": _round(returns.median()) if not returns.empty else None,
        "sum_return_bps": _round(returns.sum()) if not returns.empty else None,
        "positive_return_rate": _round((returns > 0).mean()) if not returns.empty else None,
        "avg_win_bps": _round(wins.mean()) if not wins.empty else None,
        "avg_loss_bps": _round(losses.mean()) if not losses.empty else None,
    }


def _impact_status(cell: dict[str, Any], *, min_cell_labels: int) -> str:
    action = str(cell.get("action_class") or "")
    if action == "DOWNGRADE_OR_AVOID":
        labels = _safe_int(cell.get("suppressed_label_count"))
        avg_suppressed = _safe_float(cell.get("suppressed_avg_return_60m_bps"), None)
        if labels < min_cell_labels:
            return "INSUFFICIENT_REPLAY_LABELS"
        if avg_suppressed is not None and avg_suppressed <= 0:
            return "REPLAY_SUPPORTS_SUPPRESSION"
        return "REPLAY_REJECTS_SUPPRESSION"
    if action == "HOLD_TIME_SPECIAL":
        labels = _safe_int(cell.get("hold_change_label_count"))
        avg_delta = _safe_float(cell.get("hold_change_avg_delta_bps"), None)
        if labels < min_cell_labels:
            return "INSUFFICIENT_REPLAY_LABELS"
        if avg_delta is not None and avg_delta > 0:
            return "REPLAY_SUPPORTS_HOLD_CHANGE"
        return "REPLAY_REJECTS_HOLD_CHANGE"
    if action == "REQUIRE_CONFIRMATION":
        labels = _safe_int(cell.get("suppressed_label_count"))
        avg_suppressed = _safe_float(cell.get("suppressed_avg_return_60m_bps"), None)
        if labels < min_cell_labels:
            return "INSUFFICIENT_REPLAY_LABELS"
        if avg_suppressed is not None and avg_suppressed <= 0:
            return "REPLAY_SUPPORTS_STRICTER_CONFIRMATION"
        return "REPLAY_MIXED_CONFIRMATION"
    return "REPLAY_NEUTRAL"


def _select_counterfactual(
    *,
    baseline_trade: bool,
    action_class: str,
    trade_strength: float | None,
    score_adjustment: float,
    threshold_adjustment: float,
    base_threshold: float,
) -> tuple[bool, bool, str]:
    adjusted_score = None if trade_strength is None else trade_strength + score_adjustment
    adjusted_threshold = base_threshold + threshold_adjustment
    qualifies = adjusted_score is not None and adjusted_score >= adjusted_threshold

    if action_class == "DOWNGRADE_OR_AVOID":
        return False, False, "SUPPRESSED_BY_DOWNGRADE_OR_AVOID"
    if action_class == "REQUIRE_CONFIRMATION":
        selected = bool(baseline_trade and qualifies)
        return selected, False, "RETAINED_WITH_CONFIRMATION" if selected else "SUPPRESSED_BY_CONFIRMATION"
    if action_class in {"HOLD_TIME_SPECIAL", "FAVORABLE_BOOST"}:
        if baseline_trade:
            return True, False, "RETAINED_WITH_PROPOSED_HOLD"
        if qualifies:
            return False, True, "PROMOTION_CANDIDATE_SANDBOX"
        return False, False, "NOT_PROMOTED_SCORE_BELOW_THRESHOLD"
    return baseline_trade, False, "UNCHANGED"


def build_regime_cell_counterfactual_report(
    frame: pd.DataFrame,
    *,
    review: dict[str, Any],
    dataset_path: str | Path | None = None,
    review_path: str | Path | None = None,
    base_trade_strength_threshold: float = DEFAULT_BASE_TRADE_STRENGTH_THRESHOLD,
    min_cell_labels: int = DEFAULT_MIN_CELL_LABELS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prepared = _prepare_frame(frame)
    three, four = _proposal_indexes(review)

    detail_rows: list[dict[str, Any]] = []
    for _idx, row in prepared.iterrows():
        proposal = _match_proposal(row, three, four)
        if not proposal:
            continue
        action_class = _text(proposal.get("action_class"))
        best_horizon = _text(proposal.get("best_horizon"), "60m")
        if best_horizon not in HORIZON_RETURN_COLUMNS:
            best_horizon = "60m"
        baseline_trade = _text(row.get("trade_status")).upper() == "TRADE"
        trade_strength = _safe_float(row.get("trade_strength"), None)
        score_adjustment = _safe_float(proposal.get("score_adjustment_research"), 0.0) or 0.0
        threshold_adjustment = _safe_float(proposal.get("threshold_adjustment_research"), 0.0) or 0.0
        selected, promotion_candidate, decision = _select_counterfactual(
            baseline_trade=baseline_trade,
            action_class=action_class,
            trade_strength=trade_strength,
            score_adjustment=score_adjustment,
            threshold_adjustment=threshold_adjustment,
            base_threshold=float(base_trade_strength_threshold),
        )
        baseline_return = _return_for_horizon(row, "60m")
        baseline_hit = _hit_for_horizon(row, "60m")
        proposal_return = _return_for_horizon(row, best_horizon)
        proposal_hit = _hit_for_horizon(row, best_horizon)
        counterfactual_return = proposal_return if selected else None
        counterfactual_hit = proposal_hit if selected else None
        promotion_return = proposal_return if promotion_candidate else None
        promotion_hit = proposal_hit if promotion_candidate else None
        delta = (
            counterfactual_return - baseline_return
            if selected and baseline_trade and counterfactual_return is not None and baseline_return is not None
            else None
        )
        detail_rows.append(
            {
                "signal_id": row.get("signal_id"),
                "signal_timestamp": row.get("signal_timestamp"),
                "symbol": row.get("symbol"),
                "source": row.get("source"),
                "mode": row.get("mode"),
                "trade_status": row.get("trade_status"),
                "direction": row.get("direction"),
                "gamma_regime": row.get("gamma_regime"),
                "volatility_regime": row.get("volatility_regime"),
                "macro_risk_bucket": row.get("macro_risk_bucket"),
                "trade_strength": trade_strength,
                "matched_action_class": action_class,
                "matched_group_name": proposal.get("group_name"),
                "matched_cell": proposal.get("cell"),
                "baseline_trade": baseline_trade,
                "counterfactual_selected": selected,
                "promotion_candidate": promotion_candidate,
                "counterfactual_decision": decision,
                "proposal_horizon": best_horizon,
                "baseline_return_60m_bps": baseline_return,
                "baseline_hit_60m": baseline_hit,
                "counterfactual_return_bps": counterfactual_return,
                "counterfactual_hit": counterfactual_hit,
                "counterfactual_delta_bps": delta,
                "promotion_sandbox_return_bps": promotion_return,
                "promotion_sandbox_hit": promotion_hit,
            }
        )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        report = {
            "report_type": "regime_cell_counterfactual",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "review_path": str(review_path) if review_path is not None else None,
            "runtime_config_changed": False,
            "parameter_pack_file_changed": False,
            "execution_behavior_changed": False,
            "assessment_status": "NO_MATCHING_PROPOSAL_ROWS",
            "matched_signal_count": 0,
            "cell_summary": [],
        }
        return _sanitize(report), pd.DataFrame(), details

    baseline_trade = details["baseline_trade"].astype(bool)
    selected = details["counterfactual_selected"].astype(bool)
    suppressed = details.loc[baseline_trade & ~selected].copy()
    retained = details.loc[baseline_trade & selected].copy()
    promoted = details.loc[details["promotion_candidate"].astype(bool)].copy()
    baseline_metrics = _metrics(details.loc[baseline_trade, "baseline_return_60m_bps"], details.loc[baseline_trade, "baseline_hit_60m"])
    counterfactual_metrics = _metrics(
        details.loc[selected, "counterfactual_return_bps"],
        details.loc[selected, "counterfactual_hit"],
    )
    suppressed_metrics = _metrics(suppressed["baseline_return_60m_bps"], suppressed["baseline_hit_60m"])
    promotion_metrics = _metrics(promoted["promotion_sandbox_return_bps"], promoted["promotion_sandbox_hit"])

    baseline_sum = _safe_float(baseline_metrics.get("sum_return_bps"), 0.0) or 0.0
    counterfactual_sum = _safe_float(counterfactual_metrics.get("sum_return_bps"), 0.0) or 0.0
    conservative_delta = _round(counterfactual_sum - baseline_sum)
    suppressed_returns = pd.to_numeric(suppressed.get("baseline_return_60m_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    hold_deltas = pd.to_numeric(retained.get("counterfactual_delta_bps", pd.Series(dtype=float)), errors="coerce").dropna()

    cell_rows = []
    group_cols = ["matched_group_name", "matched_cell", "matched_action_class"]
    for key, group in details.groupby(group_cols, dropna=False, observed=False):
        group_name, cell, action = key
        group_baseline_trade = group["baseline_trade"].astype(bool)
        group_selected = group["counterfactual_selected"].astype(bool)
        group_suppressed = group.loc[group_baseline_trade & ~group_selected]
        group_retained = group.loc[group_baseline_trade & group_selected]
        group_promoted = group.loc[group["promotion_candidate"].astype(bool)]
        group_hold_deltas = pd.to_numeric(group_retained.get("counterfactual_delta_bps"), errors="coerce").dropna()
        group_suppressed_returns = pd.to_numeric(group_suppressed.get("baseline_return_60m_bps"), errors="coerce").dropna()
        row = {
            "matched_group_name": group_name,
            "matched_cell": cell,
            "action_class": action,
            "matched_signal_count": int(len(group)),
            "baseline_trade_count": int(group_baseline_trade.sum()),
            "counterfactual_selected_count": int(group_selected.sum()),
            "suppressed_trade_count": int(len(group_suppressed)),
            "suppressed_label_count": int(group_suppressed_returns.count()),
            "suppressed_avg_return_60m_bps": _round(group_suppressed_returns.mean()) if not group_suppressed_returns.empty else None,
            "avoided_suppressed_return_60m_bps": _round(-group_suppressed_returns.sum()) if not group_suppressed_returns.empty else None,
            "hold_change_label_count": int(group_hold_deltas.count()),
            "hold_change_avg_delta_bps": _round(group_hold_deltas.mean()) if not group_hold_deltas.empty else None,
            "hold_change_sum_delta_bps": _round(group_hold_deltas.sum()) if not group_hold_deltas.empty else None,
            "promotion_candidate_count": int(len(group_promoted)),
            "promotion_candidate_avg_return_bps": _round(
                pd.to_numeric(group_promoted.get("promotion_sandbox_return_bps", pd.Series(dtype=float)), errors="coerce").mean()
            )
            if not group_promoted.empty
            else None,
        }
        row["impact_status"] = _impact_status(row, min_cell_labels=int(min_cell_labels))
        cell_rows.append(row)

    cell_summary = pd.DataFrame(cell_rows).sort_values(
        ["impact_status", "matched_signal_count"],
        ascending=[True, False],
        kind="mergesort",
    )
    status_counts = cell_summary["impact_status"].value_counts().to_dict() if not cell_summary.empty else {}
    assessment_status = "REPLAY_REVIEW_READY"
    if conservative_delta is not None and conservative_delta > 0:
        assessment_status = "COUNTERFACTUAL_POSITIVE"
    elif conservative_delta is not None and conservative_delta < 0:
        assessment_status = "COUNTERFACTUAL_NEGATIVE"

    report = {
        "report_type": "regime_cell_counterfactual",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "review_path": str(review_path) if review_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "assessment_basis": "conservative_research_replay_no_watchlist_promotions",
        "assessment_status": assessment_status,
        "base_trade_strength_threshold": float(base_trade_strength_threshold),
        "min_cell_labels": int(min_cell_labels),
        "matched_signal_count": int(len(details)),
        "baseline_trade_count": int(baseline_trade.sum()),
        "counterfactual_selected_count": int(selected.sum()),
        "suppressed_existing_trade_count": int(len(suppressed)),
        "retained_existing_trade_count": int(len(retained)),
        "promotion_candidate_count_sandbox": int(len(promoted)),
        "conservative_total_return_delta_bps": conservative_delta,
        "avoided_suppressed_return_60m_bps": _round(-suppressed_returns.sum()) if not suppressed_returns.empty else None,
        "hold_time_sum_delta_bps": _round(hold_deltas.sum()) if not hold_deltas.empty else None,
        "hold_time_avg_delta_bps": _round(hold_deltas.mean()) if not hold_deltas.empty else None,
        "baseline_trade_metrics_60m": baseline_metrics,
        "counterfactual_selected_metrics_proposed_horizon": counterfactual_metrics,
        "suppressed_existing_trade_metrics_60m": suppressed_metrics,
        "promotion_sandbox_metrics_proposed_horizon": promotion_metrics,
        "impact_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "cell_summary": cell_summary.to_dict("records"),
        "recommended_next_actions": [
            "Promote only cells with REPLAY_SUPPORTS_* status into a versioned artifact candidate.",
            "Manually review cells where the proposal was rejected or mixed before changing live behavior.",
            "Treat promotion candidates as opportunity analysis only; they were not counted in the conservative replay.",
        ],
    }
    return _sanitize(report), cell_summary, details


def _flat_detail_frame(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    columns = [column for column in DETAIL_COLUMNS if column in details.columns]
    return details[columns].copy()


def render_regime_cell_counterfactual_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Regime Cell Counterfactual Replay",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path')}",
        f"- Review path: {report.get('review_path')}",
        f"- Assessment basis: {report.get('assessment_basis')}",
        f"- Assessment status: {report.get('assessment_status')}",
        f"- Matched signals: {report.get('matched_signal_count')}",
        f"- Baseline TRADE rows: {report.get('baseline_trade_count')}",
        f"- Counterfactual selected rows: {report.get('counterfactual_selected_count')}",
        f"- Suppressed existing TRADE rows: {report.get('suppressed_existing_trade_count')}",
        f"- Promotion candidates, sandbox only: {report.get('promotion_candidate_count_sandbox')}",
        f"- Conservative total return delta: {report.get('conservative_total_return_delta_bps')} bps",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Aggregate Metrics",
        "",
        "| Stream | Signals/Labels | Hit Rate | Avg Return Bps | Sum Return Bps |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    aggregate_rows = [
        ("Baseline TRADE 60m", report.get("baseline_trade_count"), report.get("baseline_trade_metrics_60m") or {}),
        (
            "Counterfactual selected proposed horizon",
            report.get("counterfactual_selected_count"),
            report.get("counterfactual_selected_metrics_proposed_horizon") or {},
        ),
        (
            "Suppressed existing TRADE 60m",
            report.get("suppressed_existing_trade_count"),
            report.get("suppressed_existing_trade_metrics_60m") or {},
        ),
        (
            "Promotion sandbox proposed horizon",
            report.get("promotion_candidate_count_sandbox"),
            report.get("promotion_sandbox_metrics_proposed_horizon") or {},
        ),
    ]
    for label, signal_count, metrics in aggregate_rows:
        lines.append(
            f"| {label} | {signal_count}/{metrics.get('label_count')} | {metrics.get('hit_rate')} | "
            f"{metrics.get('avg_return_bps')} | {metrics.get('sum_return_bps')} |"
        )

    lines.extend(["", "## Impact Status Counts", "", "| Status | Count |", "| --- | ---: |"])
    for status, count in (report.get("impact_status_counts") or {}).items():
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Cell Summary",
            "",
            "| Status | Action | Cell | Baseline Trades | Suppressed | Avoided Return | Hold Delta | Promotion Sandbox |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("cell_summary", []):
        lines.append(
            f"| {row.get('impact_status')} | {row.get('action_class')} | {row.get('matched_cell')} | "
            f"{row.get('baseline_trade_count')} | {row.get('suppressed_trade_count')} | "
            f"{row.get('avoided_suppressed_return_60m_bps')} | {row.get('hold_change_sum_delta_bps')} | "
            f"{row.get('promotion_candidate_count')} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "This is a conservative research replay. It suppresses or changes hold horizon for existing TRADE rows, "
        "but it does not count WATCHLIST promotions in the main return delta."
    )
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in report.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def write_regime_cell_counterfactual_report(
    *,
    dataset_path: str | Path = DEFAULT_CUMULATIVE_DATASET_PATH,
    review_path: str | Path = DEFAULT_REGIME_CELL_REVIEW_DIR / LATEST_REVIEW_JSON_FILENAME,
    output_dir: str | Path = DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR,
    documentation_dir: str | Path | None = DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    report_name: str = "regime_cell_counterfactual",
    base_trade_strength_threshold: float = DEFAULT_BASE_TRADE_STRENGTH_THRESHOLD,
    min_cell_labels: int = DEFAULT_MIN_CELL_LABELS,
    write_latest: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    review_path = Path(review_path)
    frame = pd.read_csv(dataset_path, low_memory=False) if dataset_path.exists() else pd.DataFrame()
    review = _load_review(review_path)
    report, cell_summary, details = build_regime_cell_counterfactual_report(
        frame,
        review=review,
        dataset_path=dataset_path,
        review_path=review_path,
        base_trade_strength_threshold=base_trade_strength_threshold,
        min_cell_labels=min_cell_labels,
    )

    output_dir = Path(output_dir)
    json_path = output_dir / f"{report_name}.json"
    markdown_path = output_dir / f"{report_name}.md"
    cells_csv_path = output_dir / f"{report_name}_cells.csv"
    details_csv_path = output_dir / f"{report_name}_details.csv"
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_regime_cell_counterfactual_markdown(report))
    _atomic_write_csv(cell_summary, cells_csv_path)
    _atomic_write_csv(_flat_detail_frame(details), details_csv_path)

    artifact = {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "cells_csv_path": str(cells_csv_path),
        "details_csv_path": str(details_csv_path),
    }
    if write_latest:
        latest_json = output_dir / LATEST_COUNTERFACTUAL_JSON_FILENAME
        latest_markdown = output_dir / LATEST_COUNTERFACTUAL_MARKDOWN_FILENAME
        latest_cells = output_dir / LATEST_COUNTERFACTUAL_CELLS_CSV_FILENAME
        latest_details = output_dir / LATEST_COUNTERFACTUAL_DETAILS_CSV_FILENAME
        _atomic_write_text(latest_json, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown, render_regime_cell_counterfactual_markdown(report))
        _atomic_write_csv(cell_summary, latest_cells)
        _atomic_write_csv(_flat_detail_frame(details), latest_details)
        artifact.update(
            {
                "latest_json_path": str(latest_json),
                "latest_markdown_path": str(latest_markdown),
                "latest_cells_csv_path": str(latest_cells),
                "latest_details_csv_path": str(latest_details),
            }
        )

    if documentation_dir is not None:
        documentation_dir = Path(documentation_dir)
        docs_latest = documentation_dir / LATEST_COUNTERFACTUAL_MARKDOWN_FILENAME
        _atomic_write_text(docs_latest, render_regime_cell_counterfactual_markdown(report))
        artifact["documentation_markdown_path"] = str(docs_latest)

    return artifact
