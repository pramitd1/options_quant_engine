"""
Live shadow-mode comparison and logging utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tuning.artifacts import append_jsonl_record, load_jsonl_frame

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _trade_signal_present(trade: dict[str, Any] | None) -> bool:
    trade = trade if isinstance(trade, dict) else {}
    return bool(trade) and (
        trade.get("direction") is not None
        or str(trade.get("trade_status", "")).upper().strip() in {"TRADE", "WATCHLIST", "NO_TRADE"}
    )


def build_shadow_signal_summary(result_payload: dict[str, Any], *, pack_name: str, role: str) -> dict[str, Any]:
    trade = (result_payload or {}).get("trade") or {}
    summary = {
        "role": role,
        "pack_name": pack_name,
        "signal_present": _trade_signal_present(trade),
        "direction": trade.get("direction"),
        "trade_status": trade.get("trade_status"),
        "trade_strength": trade.get("trade_strength"),
        "overnight_hold_allowed": trade.get("overnight_hold_allowed"),
    }
    for field in SHADOW_COMPARISON_FIELDS:
        summary[field] = trade.get(field)
    return summary


def compare_shadow_trade_outputs(
    baseline_payload: dict[str, Any],
    shadow_payload: dict[str, Any],
    *,
    baseline_pack_name: str,
    shadow_pack_name: str,
) -> dict[str, Any]:
    baseline_summary = build_shadow_signal_summary(baseline_payload, pack_name=baseline_pack_name, role="baseline")
    shadow_summary = build_shadow_signal_summary(shadow_payload, pack_name=shadow_pack_name, role="shadow")

    baseline_trade = (baseline_payload or {}).get("trade") or {}
    shadow_trade = (shadow_payload or {}).get("trade") or {}
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

    return {
        "evaluation_timestamp": evaluation_timestamp,
        "baseline_pack_name": baseline_pack_name,
        "shadow_pack_name": shadow_pack_name,
        "symbol": (baseline_payload or {}).get("symbol") or (shadow_payload or {}).get("symbol"),
        "mode": (baseline_payload or {}).get("mode") or (shadow_payload or {}).get("mode"),
        "source": (baseline_payload or {}).get("source") or (shadow_payload or {}).get("source"),
        "baseline_signal_summary": baseline_summary,
        "shadow_signal_summary": shadow_summary,
        "baseline_trade_strength": baseline_trade.get("trade_strength"),
        "shadow_trade_strength": shadow_trade.get("trade_strength"),
        "baseline_trade_status": baseline_trade.get("trade_status"),
        "shadow_trade_status": shadow_trade.get("trade_status"),
        "baseline_direction": baseline_trade.get("direction"),
        "shadow_direction": shadow_trade.get("direction"),
        "delta_trade_strength": round(shadow_strength - baseline_strength, 4),
        "decision_disagreement_flag": baseline_direction != shadow_direction,
        "trade_status_disagreement_flag": baseline_status != shadow_status,
        "overnight_disagreement_flag": baseline_overnight != shadow_overnight,
        "signal_presence_disagreement_flag": baseline_summary["signal_present"] != shadow_summary["signal_present"],
        "promotion_notes": None,
    }


def append_shadow_log(record: dict[str, Any], path: str | Path = SHADOW_LOG_PATH) -> Path:
    return append_jsonl_record(dict(record or {}), path)


def load_shadow_log(path: str | Path = SHADOW_LOG_PATH) -> pd.DataFrame:
    return load_jsonl_frame(path)


def summarize_shadow_log(path: str | Path = SHADOW_LOG_PATH) -> dict[str, Any]:
    frame = load_shadow_log(path)
    if frame.empty:
        return {
            "shadow_event_count": 0,
            "decision_disagreement_rate": 0.0,
            "trade_status_disagreement_rate": 0.0,
            "average_delta_trade_strength": 0.0,
            "current_shadow_pairs": [],
        }

    return {
        "shadow_event_count": int(len(frame)),
        "decision_disagreement_rate": round(frame["decision_disagreement_flag"].astype(bool).mean(), 6),
        "trade_status_disagreement_rate": round(frame["trade_status_disagreement_flag"].astype(bool).mean(), 6),
        "average_delta_trade_strength": round(pd.to_numeric(frame["delta_trade_strength"], errors="coerce").fillna(0.0).mean(), 6),
        "current_shadow_pairs": (
            frame[["baseline_pack_name", "shadow_pack_name"]]
            .drop_duplicates()
            .to_dict(orient="records")
        ),
    }
