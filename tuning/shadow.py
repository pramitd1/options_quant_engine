"""
Module: shadow.py

Purpose:
    Implement shadow utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tuning.artifacts import append_jsonl_record, load_jsonl_frame
from utils.numerics import safe_float as _safe_float

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_tuning"
SHADOW_LOG_PATH = TUNING_RESEARCH_DIR / "shadow_mode_log.jsonl"

SHADOW_COMPARISON_FIELDS = [
    "direction",
    "trade_status",
    "trade_strength",
    "signal_quality",
    "signal_regime",
    "execution_regime",
    "macro_regime",
    "global_risk_state",
    "gamma_vol_acceleration_score",
    "dealer_hedging_pressure_score",
    "option_efficiency_score",
    "overnight_hold_allowed",
]


def _resolve_shadow_session_fields(evaluation_timestamp: Any) -> tuple[str | None, str]:
    """Resolve a session date and intraday bucket from the shadow timestamp."""
    try:
        ts = pd.to_datetime(evaluation_timestamp, errors="coerce")
    except Exception:
        ts = pd.NaT

    if pd.isna(ts):
        return None, "UNKNOWN"

    hour_fraction = float(ts.hour) + (float(ts.minute) / 60.0)
    if hour_fraction < 10.5:
        session_bucket = "OPEN"
    elif hour_fraction < 13.0:
        session_bucket = "MIDDAY"
    elif hour_fraction < 15.75:
        session_bucket = "CLOSE"
    else:
        session_bucket = "OFF_HOURS"

    return ts.strftime("%Y-%m-%d"), session_bucket


def _resolve_shadow_alert_policy() -> dict[str, float]:
    """Load policy-limit thresholds used to flag operator alerts."""
    defaults = {
        "decision_disagreement_alert": 0.20,
        "trade_status_disagreement_alert": 0.25,
        "signal_presence_disagreement_alert": 0.15,
        "overnight_disagreement_alert": 0.20,
        "session_alert_min_snapshots": 2,
    }
    try:
        from config.policy_resolver import get_regime_switch_policy

        policy = get_regime_switch_policy()
    except Exception:
        policy = {}

    for key in (
        "decision_disagreement_alert",
        "trade_status_disagreement_alert",
        "signal_presence_disagreement_alert",
        "overnight_disagreement_alert",
    ):
        defaults[key] = float(_safe_float(policy.get(key), defaults[key]))
    defaults["session_alert_min_snapshots"] = max(int(_safe_float(policy.get("session_alert_min_snapshots"), defaults["session_alert_min_snapshots"])), 1)
    return defaults


def _trade_signal_present(trade: dict[str, Any] | None) -> bool:
    """
    Purpose:
        Process trade signal present for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        trade (dict[str, Any] | None): Input associated with trade.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    trade = trade if isinstance(trade, dict) else {}
    return bool(trade) and (
        trade.get("direction") is not None
        or str(trade.get("trade_status", "")).upper().strip() in {"TRADE", "WATCHLIST", "NO_TRADE"}
    )


def build_shadow_signal_summary(result_payload: dict[str, Any], *, pack_name: str, role: str) -> dict[str, Any]:
    """
    Purpose:
        Build the shadow signal summary used by downstream components.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        result_payload (dict[str, Any]): Payload containing result.
        pack_name (str): Human-readable name for pack.
        role (str): Input associated with role.
    
    Returns:
        dict[str, Any]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    trade = (result_payload or {}).get("trade") or {}
    execution_trade = (
        (result_payload or {}).get("execution_trade")
        or (trade.get("execution_trade") if isinstance(trade, dict) else None)
        or trade
    )
    summary = {
        "role": role,
        "pack_name": pack_name,
        "signal_present": _trade_signal_present(execution_trade),
        "direction": execution_trade.get("direction"),
        "trade_status": execution_trade.get("trade_status"),
        "trade_strength": execution_trade.get("trade_strength"),
        "overnight_hold_allowed": execution_trade.get("overnight_hold_allowed"),
    }
    for field in SHADOW_COMPARISON_FIELDS:
        summary[field] = trade.get(field, execution_trade.get(field))
    return summary


def compare_shadow_trade_outputs(
    baseline_payload: dict[str, Any],
    shadow_payload: dict[str, Any],
    *,
    baseline_pack_name: str,
    shadow_pack_name: str,
) -> dict[str, Any]:
    """
    Purpose:
        Compare shadow trade outputs for governance or diagnostic purposes.
    
    Context:
        Public function in the `shadow` module. It forms part of the tuning workflow exposed by this module.
    
    Inputs:
        baseline_payload (dict[str, Any]): Structured mapping for baseline payload.
        shadow_payload (dict[str, Any]): Structured mapping for shadow payload.
        baseline_pack_name (str): Production or baseline parameter pack used as the comparison reference.
        shadow_pack_name (str): Candidate parameter-pack name being evaluated in shadow mode.
    
    Returns:
        dict[str, Any]: Structured mapping returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    baseline_summary = build_shadow_signal_summary(baseline_payload, pack_name=baseline_pack_name, role="baseline")
    shadow_summary = build_shadow_signal_summary(shadow_payload, pack_name=shadow_pack_name, role="shadow")

    baseline_trade = (
        (baseline_payload or {}).get("execution_trade")
        or (baseline_payload or {}).get("trade", {}).get("execution_trade")
        or (baseline_payload or {}).get("trade")
        or {}
    )
    shadow_trade = (
        (shadow_payload or {}).get("execution_trade")
        or (shadow_payload or {}).get("trade", {}).get("execution_trade")
        or (shadow_payload or {}).get("trade")
        or {}
    )
    evaluation_timestamp = (
        (baseline_payload or {}).get("spot_summary", {}).get("timestamp")
        or (shadow_payload or {}).get("spot_summary", {}).get("timestamp")
        or pd.Timestamp.utcnow().isoformat()
    )

    baseline_strength = _safe_float(baseline_trade.get("trade_strength"), 0.0)
    shadow_strength = _safe_float(shadow_trade.get("trade_strength"), 0.0)
    baseline_direction = str(baseline_trade.get("direction") or "")
    shadow_direction = str(shadow_trade.get("direction") or "")
    baseline_status = str(baseline_trade.get("trade_status") or "")
    shadow_status = str(shadow_trade.get("trade_status") or "")
    baseline_overnight = baseline_trade.get("overnight_hold_allowed")
    shadow_overnight = shadow_trade.get("overnight_hold_allowed")
    signal_presence_disagreement = baseline_summary["signal_present"] != shadow_summary["signal_present"]
    decision_disagreement = baseline_direction != shadow_direction
    trade_status_disagreement = baseline_status != shadow_status
    overnight_disagreement = baseline_overnight != shadow_overnight
    session_date, session_bucket = _resolve_shadow_session_fields(evaluation_timestamp)
    validation_status = (
        "DIVERGED"
        if any(
            [
                decision_disagreement,
                trade_status_disagreement,
                overnight_disagreement,
                signal_presence_disagreement,
            ]
        )
        else "ALIGNED"
    )

    return {
        "evaluation_timestamp": evaluation_timestamp,
        "baseline_pack_name": baseline_pack_name,
        "shadow_pack_name": shadow_pack_name,
        "symbol": (baseline_payload or {}).get("symbol") or (shadow_payload or {}).get("symbol"),
        "mode": (baseline_payload or {}).get("mode") or (shadow_payload or {}).get("mode"),
        "source": (baseline_payload or {}).get("source") or (shadow_payload or {}).get("source"),
        "session_date": session_date,
        "session_bucket": session_bucket,
        "baseline_signal_summary": baseline_summary,
        "shadow_signal_summary": shadow_summary,
        "baseline_trade_strength": baseline_trade.get("trade_strength"),
        "shadow_trade_strength": shadow_trade.get("trade_strength"),
        "baseline_trade_status": baseline_trade.get("trade_status"),
        "shadow_trade_status": shadow_trade.get("trade_status"),
        "baseline_direction": baseline_trade.get("direction"),
        "shadow_direction": shadow_trade.get("direction"),
        "delta_trade_strength": round(shadow_strength - baseline_strength, 4),
        "decision_disagreement_flag": decision_disagreement,
        "trade_status_disagreement_flag": trade_status_disagreement,
        "overnight_disagreement_flag": overnight_disagreement,
        "signal_presence_disagreement_flag": signal_presence_disagreement,
        "validation_status": validation_status,
        "promotion_notes": None,
    }


def append_shadow_log(record: dict[str, Any], path: str | Path = SHADOW_LOG_PATH) -> Path:
    """
    Purpose:
        Process append shadow log for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        record (dict[str, Any]): Input associated with record.
        path (str | Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return append_jsonl_record(dict(record or {}), path)


def load_shadow_log(path: str | Path = SHADOW_LOG_PATH) -> pd.DataFrame:
    """
    Purpose:
        Process load shadow log for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return load_jsonl_frame(path)


def summarize_shadow_log(path: str | Path = SHADOW_LOG_PATH) -> dict[str, Any]:
    """
    Purpose:
        Summarize shadow log into a compact diagnostic payload.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        dict[str, Any]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    frame = load_shadow_log(path)
    alert_policy = _resolve_shadow_alert_policy()
    if frame.empty:
        return {
            "shadow_event_count": 0,
            "decision_disagreement_rate": 0.0,
            "trade_status_disagreement_rate": 0.0,
            "average_delta_trade_strength": 0.0,
            "current_shadow_pairs": [],
            "session_validation_summary": [],
            "latest_session_validation": {},
            "policy_alert_count": 0,
            "latest_policy_alert": {},
            "policy_limits": alert_policy,
        }

    work = frame.copy()
    if "evaluation_timestamp" not in work.columns:
        work["evaluation_timestamp"] = None
    if "baseline_pack_name" not in work.columns:
        work["baseline_pack_name"] = None
    if "shadow_pack_name" not in work.columns:
        work["shadow_pack_name"] = None

    ts_source = work["evaluation_timestamp"]
    work["evaluation_dt"] = pd.to_datetime(ts_source, errors="coerce")

    if "session_date" not in work.columns:
        work["session_date"] = None
    if "session_bucket" not in work.columns:
        work["session_bucket"] = None

    resolved_pairs = work["evaluation_timestamp"].apply(_resolve_shadow_session_fields) if "evaluation_timestamp" in work.columns else pd.Series([(None, "UNKNOWN")] * len(work), index=work.index)
    resolved_dates = resolved_pairs.apply(lambda item: item[0])
    resolved_buckets = resolved_pairs.apply(lambda item: item[1])
    work["session_date"] = work["session_date"].fillna(resolved_dates)
    work["session_bucket"] = work["session_bucket"].fillna(resolved_buckets)

    for col in [
        "decision_disagreement_flag",
        "trade_status_disagreement_flag",
        "signal_presence_disagreement_flag",
        "overnight_disagreement_flag",
    ]:
        if col not in work.columns:
            work[col] = False

    if "delta_trade_strength" not in work.columns:
        work["delta_trade_strength"] = 0.0

    session_group = (
        work.groupby(["session_date", "baseline_pack_name", "shadow_pack_name"], dropna=False)
        .agg(
            snapshot_count=("evaluation_timestamp", "count"),
            decision_disagreement_rate=("decision_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            trade_status_disagreement_rate=("trade_status_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            signal_presence_disagreement_rate=("signal_presence_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            overnight_disagreement_rate=("overnight_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            average_delta_trade_strength=("delta_trade_strength", lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).mean()),
            latest_evaluation_dt=("evaluation_dt", "max"),
            session_buckets_observed=("session_bucket", lambda s: sorted({str(v) for v in s.dropna().tolist() if str(v)})),
        )
        .reset_index()
        .sort_values(["latest_evaluation_dt", "session_date"], ascending=[False, False], kind="stable")
    )

    session_rows = []
    for _, row in session_group.iterrows():
        decision_rate = round(_safe_float(row.get("decision_disagreement_rate"), 0.0), 6)
        trade_status_rate = round(_safe_float(row.get("trade_status_disagreement_rate"), 0.0), 6)
        signal_presence_rate = round(_safe_float(row.get("signal_presence_disagreement_rate"), 0.0), 6)
        overnight_rate = round(_safe_float(row.get("overnight_disagreement_rate"), 0.0), 6)
        snapshot_count = int(row.get("snapshot_count") or 0)

        breached_limits: list[str] = []
        if snapshot_count >= int(alert_policy.get("session_alert_min_snapshots", 1)):
            if decision_rate >= float(alert_policy.get("decision_disagreement_alert", 1.0)):
                breached_limits.append("decision_disagreement_rate")
            if trade_status_rate >= float(alert_policy.get("trade_status_disagreement_alert", 1.0)):
                breached_limits.append("trade_status_disagreement_rate")
            if signal_presence_rate >= float(alert_policy.get("signal_presence_disagreement_alert", 1.0)):
                breached_limits.append("signal_presence_disagreement_rate")
            if overnight_rate >= float(alert_policy.get("overnight_disagreement_alert", 1.0)):
                breached_limits.append("overnight_disagreement_rate")

        policy_alert = bool(breached_limits)
        alert_level = "ALERT" if policy_alert else ("WATCH" if max(decision_rate, trade_status_rate, signal_presence_rate, overnight_rate) > 0.0 else "OK")
        validation_status = "ALIGNED" if max(decision_rate, trade_status_rate) <= 0.0 else ("WATCH" if max(decision_rate, trade_status_rate) < 0.35 else "REVIEW")
        session_rows.append(
            {
                "session_date": row.get("session_date"),
                "baseline_pack_name": row.get("baseline_pack_name"),
                "shadow_pack_name": row.get("shadow_pack_name"),
                "snapshot_count": snapshot_count,
                "decision_disagreement_rate": decision_rate,
                "trade_status_disagreement_rate": trade_status_rate,
                "signal_presence_disagreement_rate": signal_presence_rate,
                "overnight_disagreement_rate": overnight_rate,
                "average_delta_trade_strength": round(float(row.get("average_delta_trade_strength") or 0.0), 6),
                "validation_status": validation_status,
                "alert_level": alert_level,
                "policy_alert": policy_alert,
                "breached_limits": breached_limits,
                "session_buckets_observed": row.get("session_buckets_observed") or [],
                "latest_evaluation_timestamp": row.get("latest_evaluation_dt").isoformat() if pd.notna(row.get("latest_evaluation_dt")) else None,
            }
        )

    bucket_group = (
        work.groupby("session_bucket", dropna=False)
        .agg(
            snapshot_count=("evaluation_timestamp", "count"),
            decision_disagreement_rate=("decision_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            trade_status_disagreement_rate=("trade_status_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            signal_presence_disagreement_rate=("signal_presence_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
            overnight_disagreement_rate=("overnight_disagreement_flag", lambda s: s.fillna(False).astype(bool).mean()),
        )
        .reset_index()
    )
    session_bucket_summary = [
        {
            "session_bucket": row.get("session_bucket") or "UNKNOWN",
            "snapshot_count": int(row.get("snapshot_count") or 0),
            "decision_disagreement_rate": round(_safe_float(row.get("decision_disagreement_rate"), 0.0), 6),
            "trade_status_disagreement_rate": round(_safe_float(row.get("trade_status_disagreement_rate"), 0.0), 6),
            "signal_presence_disagreement_rate": round(_safe_float(row.get("signal_presence_disagreement_rate"), 0.0), 6),
            "overnight_disagreement_rate": round(_safe_float(row.get("overnight_disagreement_rate"), 0.0), 6),
        }
        for _, row in bucket_group.iterrows()
    ]

    driver_rows = []
    for label, column in [
        ("decision_disagreement", "decision_disagreement_flag"),
        ("trade_status_disagreement", "trade_status_disagreement_flag"),
        ("signal_presence_disagreement", "signal_presence_disagreement_flag"),
        ("overnight_disagreement", "overnight_disagreement_flag"),
    ]:
        count = int(work[column].fillna(False).astype(bool).sum())
        rate = round(float(work[column].fillna(False).astype(bool).mean()), 6)
        driver_rows.append({"driver": label, "count": count, "rate": rate})
    dominant_drivers = sorted(driver_rows, key=lambda row: (-row["rate"], -row["count"], row["driver"]))

    policy_alert_rows = [row for row in session_rows if row.get("policy_alert")]
    return {
        "shadow_event_count": int(len(work)),
        "decision_disagreement_rate": round(work["decision_disagreement_flag"].astype(bool).mean(), 6),
        "trade_status_disagreement_rate": round(work["trade_status_disagreement_flag"].astype(bool).mean(), 6),
        "average_delta_trade_strength": round(pd.to_numeric(work["delta_trade_strength"], errors="coerce").fillna(0.0).mean(), 6),
        "current_shadow_pairs": (
            work[["baseline_pack_name", "shadow_pack_name"]]
            .drop_duplicates()
            .to_dict(orient="records")
        ),
        "session_validation_summary": session_rows,
        "latest_session_validation": session_rows[0] if session_rows else {},
        "session_bucket_summary": session_bucket_summary,
        "dominant_disagreement_drivers": dominant_drivers,
        "policy_alert_count": int(len(policy_alert_rows)),
        "latest_policy_alert": policy_alert_rows[0] if policy_alert_rows else {},
        "policy_limits": alert_policy,
    }
