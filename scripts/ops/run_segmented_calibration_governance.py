#!/usr/bin/env python3
"""Govern baseline-vs-segmented runtime score calibration comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine_runner import run_engine_snapshot
from backtest.replay_regression import _find_chain_snapshots, _find_spot_snapshots, _nearest_spot_snapshot
from config.signal_policy import TRADE_RUNTIME_THRESHOLDS
from data.replay_loader import resolve_nearest_replay_snapshot_paths
from engine.signal_engine import (
    _canonical_vol_regime,
    _compute_runtime_composite_score,
    _normalize_gamma_vol_score,
    _resolve_regime_thresholds,
)
from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset
from research.signal_evaluation.reporting import write_signal_evaluation_report
from strategy import score_calibration as score_calibration_mod
from utils.regime_normalization import canonical_gamma_regime


MODEL_RELATIVE_PATH = "models_store/runtime_score_calibrator.json"
OUTPUT_ROOT = ROOT / "research" / "reviews" / "segmented_calibration_governance"
REPLAY_COLUMNS = [
    "scenario",
    "signal_id",
    "signal_timestamp",
    "symbol",
    "direction",
    "ok",
    "reason",
    "error",
    "trade_status",
    "runtime_composite_score",
    "no_trade_reason_code",
    "score_calibration_segment_key",
    "weak_data_circuit_breaker_triggered",
    "weak_data_circuit_breaker_trigger_count",
    "weak_data_circuit_breaker_reasons",
    "weak_data_circuit_breaker_shadow_triggered",
    "weak_data_circuit_breaker_shadow_trigger_count",
    "weak_data_circuit_breaker_shadow_reasons",
    "correct_60m",
    "signed_return_60m_bps",
    "saved_spot_snapshot_path",
    "saved_chain_snapshot_path",
]
PROXY_COLUMNS = [
    "signal_id",
    "direction",
    "trade_strength",
    "correct_60m",
    "signed_return_60m_bps",
    "realized_return_60m",
    "proxy_raw_runtime_composite_score",
    "proxy_calibrated_runtime_composite_score",
    "proxy_effective_min_trade_strength",
    "proxy_effective_min_composite_score",
    "proxy_selected",
    "proxy_selected_segment_key",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    calibrator_path: Path
    description: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _reset_runtime_calibrator_state() -> None:
    score_calibration_mod._global_calibrator = None
    score_calibration_mod._calibration_autoload_attempted = False
    score_calibration_mod._loaded_calibrator_path = None
    score_calibration_mod._loaded_calibrator_mtime = None


def _export_head_calibrator(output_dir: Path) -> Path:
    output_path = output_dir / "baseline_runtime_score_calibrator_head.json"
    proc = subprocess.run(
        ["git", "show", f"HEAD:{MODEL_RELATIVE_PATH}"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Unable to export baseline calibrator from git HEAD: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _eligible_replay_rows(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        frame["saved_spot_snapshot_path"].notna()
        & frame["saved_chain_snapshot_path"].notna()
    ].copy()
    eligible = eligible.loc[
        eligible["saved_spot_snapshot_path"].astype(str).str.len().gt(0)
        & eligible["saved_chain_snapshot_path"].astype(str).str.len().gt(0)
    ].copy()

    def _exists(path_text: str) -> bool:
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        return path.exists()

    eligible = eligible.loc[
        eligible["saved_spot_snapshot_path"].map(_exists)
        & eligible["saved_chain_snapshot_path"].map(_exists)
    ].copy()
    if "correct_60m" in eligible.columns:
        eligible = eligible.loc[pd.to_numeric(eligible["correct_60m"], errors="coerce").notna()].copy()
    if "signal_timestamp" in eligible.columns:
        eligible["signal_timestamp"] = pd.to_datetime(eligible["signal_timestamp"], errors="coerce", format="mixed")
        eligible = eligible.sort_values("signal_timestamp")
    return eligible.reset_index(drop=True)


def _archived_replay_rows(limit: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    replay_dir = ROOT / "debug_samples"
    for source in ["ICICI", "NSE", "ZERODHA"]:
        chain_paths = _find_chain_snapshots("NIFTY", source, str(replay_dir / "option_chain_snapshots"))
        spot_paths = _find_spot_snapshots("NIFTY", str(replay_dir / "spot_snapshots"))
        for chain_path in chain_paths:
            spot_path = _nearest_spot_snapshot(chain_path, spot_paths)
            if spot_path is None:
                continue
            rows.append(
                {
                    "signal_id": f"archive::{source}::{chain_path.stem}",
                    "signal_timestamp": None,
                    "symbol": "NIFTY",
                    "source": source,
                    "direction": None,
                    "correct_60m": None,
                    "signed_return_60m_bps": None,
                    "saved_spot_snapshot_path": str(spot_path),
                    "saved_chain_snapshot_path": str(chain_path),
                }
            )
            if len(rows) >= max(int(limit), 1):
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _backfill_outcome_ready_replay_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    required_cols = ["signal_timestamp", "symbol", "source", "correct_60m", "signed_return_60m_bps"]
    for col in required_cols:
        if col not in frame.columns:
            return pd.DataFrame()

    working = frame.copy()
    working["signal_timestamp"] = pd.to_datetime(working["signal_timestamp"], errors="coerce", format="mixed")
    working["correct_60m_num"] = pd.to_numeric(working["correct_60m"], errors="coerce")
    working["signed_return_60m_bps_num"] = pd.to_numeric(working["signed_return_60m_bps"], errors="coerce")
    working = working.loc[
        working["signal_timestamp"].notna()
        & working["correct_60m_num"].notna()
        & working["signed_return_60m_bps_num"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in working.sort_values("signal_timestamp", ascending=False).iterrows():
        symbol = str(row.get("symbol") or "NIFTY").upper().strip()
        source = str(row.get("source") or "ICICI").upper().strip()
        nearest = resolve_nearest_replay_snapshot_paths(
            symbol,
            target_timestamp=row.get("signal_timestamp"),
            replay_dir=str(ROOT / "debug_samples"),
            source_label=source,
            max_spot_delta_seconds=8 * 3600.0,
            max_chain_delta_seconds=8 * 3600.0,
        )
        spot_path = nearest.get("spot_path")
        chain_path = nearest.get("chain_path")
        if not spot_path or not chain_path:
            continue
        rows.append(
            {
                "signal_id": row.get("signal_id") or f"backfill::{source}::{row.get('signal_timestamp')}",
                "signal_timestamp": row.get("signal_timestamp"),
                "symbol": symbol,
                "source": source,
                "direction": row.get("direction"),
                "gamma_regime": row.get("gamma_regime"),
                "vol_regime": row.get("vol_regime"),
                "volatility_regime": row.get("volatility_regime"),
                "trade_strength": row.get("trade_strength"),
                "correct_60m": row.get("correct_60m_num"),
                "signed_return_60m_bps": row.get("signed_return_60m_bps_num"),
                "saved_spot_snapshot_path": str(spot_path),
                "saved_chain_snapshot_path": str(chain_path),
                "snapshot_backfill_spot_delta_seconds": nearest.get("spot_delta_seconds"),
                "snapshot_backfill_chain_delta_seconds": nearest.get("chain_delta_seconds"),
            }
        )
        if len(rows) >= max(int(limit), 1):
            break
    return pd.DataFrame(rows)


def _normalize_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def _trade_payload(result: dict[str, Any]) -> dict[str, Any]:
    trade = result.get("execution_trade")
    if isinstance(trade, dict):
        return trade
    trade = result.get("trade")
    return trade if isinstance(trade, dict) else {}


def _delta(segmented_value: Any, baseline_value: Any) -> Any:
    if segmented_value is None or baseline_value is None:
        return None
    return round(float(segmented_value) - float(baseline_value), 6)


def _calibration_context_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "direction": row.get("direction"),
        "gamma_regime": canonical_gamma_regime(row.get("gamma_regime")),
        "vol_regime": _canonical_vol_regime(row.get("volatility_regime") or row.get("vol_regime")),
    }


def _effective_thresholds(row: pd.Series) -> dict[str, int]:
    thresholds = _resolve_regime_thresholds(
        runtime_thresholds=TRADE_RUNTIME_THRESHOLDS,
        base_min_trade_strength=int(TRADE_RUNTIME_THRESHOLDS.get("min_trade_strength", 62)),
        base_min_composite_score=int(TRADE_RUNTIME_THRESHOLDS.get("min_composite_score", 58)),
        market_state={
            "gamma_regime": row.get("gamma_regime"),
            "vol_regime": row.get("volatility_regime") or row.get("vol_regime"),
            "spot_vs_flip": row.get("spot_vs_flip"),
        },
    )
    return {
        "effective_min_trade_strength": int(thresholds.get("effective_min_trade_strength", TRADE_RUNTIME_THRESHOLDS.get("min_trade_strength", 62))),
        "effective_min_composite_score": int(thresholds.get("effective_min_composite_score", TRADE_RUNTIME_THRESHOLDS.get("min_composite_score", 58))),
    }


def _raw_runtime_composite_score(row: pd.Series) -> int:
    gamma_vol_score = row.get("gamma_vol_acceleration_score_normalized")
    if gamma_vol_score is None or pd.isna(gamma_vol_score):
        gamma_vol_score = _normalize_gamma_vol_score(
            row.get("gamma_vol_acceleration_score"),
            int(float(TRADE_RUNTIME_THRESHOLDS.get("gamma_vol_normalization_scale", 100))),
            int(float(TRADE_RUNTIME_THRESHOLDS.get("gamma_vol_winsor_lower", 12))),
            int(float(TRADE_RUNTIME_THRESHOLDS.get("gamma_vol_winsor_upper", 88))),
        )

    return int(
        _compute_runtime_composite_score(
            trade_strength=row.get("trade_strength"),
            hybrid_move_probability=row.get("hybrid_move_probability"),
            move_probability_score_cap=TRADE_RUNTIME_THRESHOLDS.get("move_probability_score_cap"),
            confirmation_status=row.get("confirmation_status"),
            data_quality_status=row.get("data_quality_status"),
            gamma_vol_acceleration_score_normalized=gamma_vol_score,
            weight_trade_strength=TRADE_RUNTIME_THRESHOLDS.get("composite_weight_trade_strength", 0.50),
            weight_move_probability=TRADE_RUNTIME_THRESHOLDS.get("composite_weight_move_probability", 0.20),
            weight_confirmation=TRADE_RUNTIME_THRESHOLDS.get("composite_weight_confirmation", 0.15),
            weight_data_quality=TRADE_RUNTIME_THRESHOLDS.get("composite_weight_data_quality", 0.10),
            weight_gamma_stability=TRADE_RUNTIME_THRESHOLDS.get("composite_weight_gamma_stability", 0.05),
        )
    )


def _run_replay_scenario(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_calibrator_path = TRADE_RUNTIME_THRESHOLDS.get("runtime_score_calibrator_path")
    TRADE_RUNTIME_THRESHOLDS["runtime_score_calibrator_path"] = str(scenario.calibrator_path)
    _reset_runtime_calibrator_state()
    try:
        for _, row in frame.iterrows():
            result = run_engine_snapshot(
                symbol=str(row.get("symbol") or "NIFTY"),
                mode="REPLAY",
                source=str(row.get("source") or "NSE"),
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=50,
                max_capital=100000.0,
                replay_spot=_normalize_path(str(row.get("saved_spot_snapshot_path"))),
                replay_chain=_normalize_path(str(row.get("saved_chain_snapshot_path"))),
                capture_signal_evaluation=False,
                global_market_snapshot={
                    "data_available": False,
                    "neutral_fallback": True,
                    "warnings": ["governed_segmented_calibration_replay_neutral_global_market"],
                    "issues": [],
                },
            )
            trade = _trade_payload(result)
            full_trade = result.get("trade") if isinstance(result.get("trade"), dict) else {}

            def _trade_field(key: str) -> Any:
                value = trade.get(key)
                if value is None:
                    value = full_trade.get(key)
                return value

            breaker = _trade_field("weak_data_circuit_breaker")
            shadow_breaker = _trade_field("weak_data_circuit_breaker_shadow")
            rows.append(
                {
                    "scenario": scenario.name,
                    "signal_id": row.get("signal_id"),
                    "signal_timestamp": row.get("signal_timestamp"),
                    "symbol": row.get("symbol"),
                    "direction": row.get("direction"),
                    "ok": bool(result.get("ok")),
                    "reason": result.get("reason"),
                    "error": result.get("error"),
                    "trade_status": _trade_field("trade_status"),
                    "runtime_composite_score": _trade_field("runtime_composite_score"),
                    "no_trade_reason_code": _trade_field("no_trade_reason_code"),
                    "score_calibration_segment_key": _trade_field("score_calibration_segment_key"),
                    "weak_data_circuit_breaker_triggered": (breaker or {}).get("triggered") if isinstance(breaker, dict) else None,
                    "weak_data_circuit_breaker_trigger_count": (breaker or {}).get("trigger_count") if isinstance(breaker, dict) else None,
                    "weak_data_circuit_breaker_reasons": "|".join((breaker or {}).get("trigger_reasons") or []) if isinstance(breaker, dict) else None,
                    "weak_data_circuit_breaker_shadow_triggered": (shadow_breaker or {}).get("triggered") if isinstance(shadow_breaker, dict) else None,
                    "weak_data_circuit_breaker_shadow_trigger_count": (shadow_breaker or {}).get("trigger_count") if isinstance(shadow_breaker, dict) else None,
                    "weak_data_circuit_breaker_shadow_reasons": "|".join((shadow_breaker or {}).get("trigger_reasons") or []) if isinstance(shadow_breaker, dict) else None,
                    "correct_60m": row.get("correct_60m"),
                    "signed_return_60m_bps": row.get("signed_return_60m_bps"),
                    "saved_spot_snapshot_path": row.get("saved_spot_snapshot_path"),
                    "saved_chain_snapshot_path": row.get("saved_chain_snapshot_path"),
                }
            )
    finally:
        TRADE_RUNTIME_THRESHOLDS["runtime_score_calibrator_path"] = previous_calibrator_path
        _reset_runtime_calibrator_state()
    return pd.DataFrame(rows, columns=REPLAY_COLUMNS)


def _summarize_replay(frame: pd.DataFrame) -> dict[str, Any]:
    def _reason_counts(series: pd.Series) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in series.dropna().astype(str):
            for reason in [r.strip() for r in value.split("|") if str(r).strip()]:
                key = reason.upper()
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _blocker_precedence_counts(summary_frame: pd.DataFrame) -> dict[str, int]:
        ordered = ["TRADE", "PROVIDER_HEALTH", "WEAK_DATA_CIRCUIT_BREAKER", "NO_SIGNAL_OR_THRESHOLD", "OTHER_BLOCKER"]
        if summary_frame.empty:
            return {key: 0 for key in ordered}

        trade_status = summary_frame.get("trade_status", pd.Series(index=summary_frame.index, dtype="object")).astype(str).str.upper()
        reason_code = summary_frame.get("no_trade_reason_code", pd.Series(index=summary_frame.index, dtype="object")).astype(str).str.upper()
        shadow_reasons = summary_frame.get("weak_data_circuit_breaker_shadow_reasons", pd.Series(index=summary_frame.index, dtype="object")).astype(str).str.upper()
        breaker_triggered = pd.Series(summary_frame.get("weak_data_circuit_breaker_triggered"), dtype="boolean").fillna(False).astype(bool)

        labels = pd.Series("OTHER_BLOCKER", index=summary_frame.index, dtype="object")
        labels.loc[trade_status.eq("TRADE")] = "TRADE"

        provider_mask = (
            trade_status.ne("TRADE")
            & (
                reason_code.str.contains("PROVIDER_HEALTH", na=False)
                | shadow_reasons.str.contains("PROVIDER_HEALTH", na=False)
            )
        )
        labels.loc[provider_mask] = "PROVIDER_HEALTH"

        weak_data_mask = trade_status.ne("TRADE") & labels.ne("PROVIDER_HEALTH") & breaker_triggered
        labels.loc[weak_data_mask] = "WEAK_DATA_CIRCUIT_BREAKER"

        no_signal_mask = (
            trade_status.ne("TRADE")
            & labels.eq("OTHER_BLOCKER")
            & (
                trade_status.eq("NO_SIGNAL")
                | reason_code.str.contains("NO_SIGNAL", na=False)
                | reason_code.str.contains("RUNTIME_COMPOSITE", na=False)
                | reason_code.str.contains("BELOW_FLOOR", na=False)
                | reason_code.str.contains("THRESHOLD", na=False)
            )
        )
        labels.loc[no_signal_mask] = "NO_SIGNAL_OR_THRESHOLD"

        counts = labels.value_counts().to_dict()
        return {key: int(counts.get(key, 0)) for key in ordered}

    def _reason_code_counts(summary_frame: pd.DataFrame) -> dict[str, int]:
        reason_code = summary_frame.get("no_trade_reason_code", pd.Series(index=summary_frame.index, dtype="object"))
        clean = reason_code.astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
        return {str(k): int(v) for k, v in clean.value_counts().to_dict().items()}

    if frame.empty:
        return {
            "replayed_rows": 0,
            "ok_rows": 0,
            "trade_count": 0,
            "watchlist_count": 0,
            "trade_hit_rate_60m": None,
            "trade_avg_signed_return_60m_bps": None,
            "signal_hit_rate_60m": None,
            "signal_avg_signed_return_60m_bps": None,
            "breaker_evaluated_rows": 0,
            "breaker_triggered_rows": 0,
            "breaker_shadow_triggered_rows": 0,
            "breaker_trigger_reason_counts": {},
            "breaker_shadow_reason_counts": {},
            "blocker_precedence_counts": {
                "TRADE": 0,
                "PROVIDER_HEALTH": 0,
                "WEAK_DATA_CIRCUIT_BREAKER": 0,
                "NO_SIGNAL_OR_THRESHOLD": 0,
                "OTHER_BLOCKER": 0,
            },
            "no_trade_reason_code_counts": {},
        }
    signal_hit = pd.to_numeric(frame.get("correct_60m"), errors="coerce")
    signal_ret = pd.to_numeric(frame.get("signed_return_60m_bps"), errors="coerce")
    trade_rows = frame.loc[frame["trade_status"].astype(str).str.upper().eq("TRADE")].copy()
    hit = pd.to_numeric(trade_rows.get("correct_60m"), errors="coerce")
    ret = pd.to_numeric(trade_rows.get("signed_return_60m_bps"), errors="coerce")
    breaker_triggered = pd.Series(frame.get("weak_data_circuit_breaker_triggered"), dtype="boolean").fillna(False).astype(bool)
    shadow_triggered = pd.Series(frame.get("weak_data_circuit_breaker_shadow_triggered"), dtype="boolean").fillna(False).astype(bool)
    breaker_count = pd.to_numeric(frame.get("weak_data_circuit_breaker_trigger_count"), errors="coerce")
    shadow_count = pd.to_numeric(frame.get("weak_data_circuit_breaker_shadow_trigger_count"), errors="coerce")
    evaluated_rows = int((breaker_count.notna() | shadow_count.notna() | breaker_triggered | shadow_triggered).sum())
    return {
        "replayed_rows": int(len(frame)),
        "ok_rows": int(frame["ok"].astype(bool).sum()),
        "trade_count": int(len(trade_rows)),
        "watchlist_count": int(frame["trade_status"].astype(str).str.upper().eq("WATCHLIST").sum()),
        "trade_hit_rate_60m": None if hit.dropna().empty else round(float(hit.mean()), 6),
        "trade_avg_signed_return_60m_bps": None if ret.dropna().empty else round(float(ret.mean()), 6),
        "signal_hit_rate_60m": None if signal_hit.dropna().empty else round(float(signal_hit.mean()), 6),
        "signal_avg_signed_return_60m_bps": None if signal_ret.dropna().empty else round(float(signal_ret.mean()), 6),
        "breaker_evaluated_rows": evaluated_rows,
        "breaker_triggered_rows": int(breaker_triggered.sum()),
        "breaker_shadow_triggered_rows": int(shadow_triggered.sum()),
        "breaker_trigger_reason_counts": _reason_counts(frame.get("weak_data_circuit_breaker_reasons", pd.Series(dtype="object"))),
        "breaker_shadow_reason_counts": _reason_counts(frame.get("weak_data_circuit_breaker_shadow_reasons", pd.Series(dtype="object"))),
        "blocker_precedence_counts": _blocker_precedence_counts(frame),
        "no_trade_reason_code_counts": _reason_code_counts(frame),
    }


def _proxy_frame(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _reset_runtime_calibrator_state()
    for _, row in frame.iterrows():
        direction = str(row.get("direction") or "").upper().strip()
        if direction not in {"CALL", "PUT"}:
            continue
        raw_score = _raw_runtime_composite_score(row)
        thresholds = _effective_thresholds(row)
        context = _calibration_context_from_row(row)
        calibrated_score = score_calibration_mod.apply_score_calibration(
            raw_composite_score=raw_score,
            calibrator_path=str(scenario.calibrator_path),
            calibration_context=context,
        )
        metadata = score_calibration_mod.get_calibrator_runtime_metadata(
            str(scenario.calibrator_path),
            calibration_context=context,
        )
        selected = bool(
            float(row.get("trade_strength") or 0.0) >= thresholds["effective_min_trade_strength"]
            and int(calibrated_score) >= thresholds["effective_min_composite_score"]
        )
        out = row.to_dict()
        out.update(
            {
                "proxy_raw_runtime_composite_score": int(raw_score),
                "proxy_calibrated_runtime_composite_score": int(calibrated_score),
                "proxy_effective_min_trade_strength": int(thresholds["effective_min_trade_strength"]),
                "proxy_effective_min_composite_score": int(thresholds["effective_min_composite_score"]),
                "proxy_selected": selected,
                "proxy_selected_segment_key": metadata.get("selected_segment_key"),
            }
        )
        rows.append(out)
    _reset_runtime_calibrator_state()
    return pd.DataFrame(rows, columns=PROXY_COLUMNS)


def _summarize_selected(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "selected_count": 0,
            "hit_rate_60m": None,
            "avg_signed_return_60m_bps": None,
            "avg_realized_return_60m": None,
            "call_share": None,
            "put_share": None,
        }
    hit = pd.to_numeric(frame.get("correct_60m"), errors="coerce")
    signed = pd.to_numeric(frame.get("signed_return_60m_bps"), errors="coerce")
    realized = pd.to_numeric(frame.get("realized_return_60m"), errors="coerce")
    direction_mix = frame["direction"].astype(str).str.upper().value_counts(normalize=True)
    return {
        "selected_count": int(len(frame)),
        "hit_rate_60m": None if hit.dropna().empty else round(float(hit.mean()), 6),
        "avg_signed_return_60m_bps": None if signed.dropna().empty else round(float(signed.mean()), 6),
        "avg_realized_return_60m": None if realized.dropna().empty else round(float(realized.mean()), 6),
        "call_share": None if "CALL" not in direction_mix else round(float(direction_mix["CALL"]), 6),
        "put_share": None if "PUT" not in direction_mix else round(float(direction_mix["PUT"]), 6),
    }


def _replay_candidates_with_priority(
    dataset: pd.DataFrame,
    proxy_compare: pd.DataFrame,
    replay_limit: int,
    replay_priority_mode: str = "balanced",
    weak_data_min_score: int = 1,
) -> tuple[pd.DataFrame, str]:
    def _append_rows(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        if extra.empty:
            return base
        if base.empty:
            return extra.copy()
        merged_rows = base.to_dict(orient="records")
        merged_rows.extend(extra.to_dict(orient="records"))
        return pd.DataFrame(merged_rows)

    def _pick_diverse_rows(bucket: pd.DataFrame, count: int) -> pd.DataFrame:
        if count <= 0 or bucket.empty:
            return bucket.head(0)
        ordered = bucket.sort_values(["replay_weak_data_score", "replay_score_abs_delta", "signal_timestamp"], ascending=[False, False, False])
        diverse = ordered.drop_duplicates(subset=["replay_diversity_key"], keep="first")
        picked = diverse.head(count)
        if len(picked) >= count:
            return picked
        remainder = ordered.loc[~ordered.index.isin(picked.index)].head(count - len(picked))
        return pd.concat([picked, remainder], ignore_index=False)

    linked = _eligible_replay_rows(dataset)
    linked = linked.assign(replay_candidate_source="dataset_linked_snapshot_rows")

    target_pool_size = max(int(replay_limit) * 8, int(replay_limit), 1)
    candidates = linked.copy()
    if len(candidates) < target_pool_size:
        need = target_pool_size - len(candidates)
        backfilled = _backfill_outcome_ready_replay_rows(dataset, need)
        if not backfilled.empty:
            backfilled = backfilled.assign(replay_candidate_source="dataset_backfilled_snapshot_rows")
            candidates = _append_rows(candidates, backfilled)

    # If dataset-linked/backfilled rows collapse to too few distinct snapshot states,
    # supplement with archived states so exact replay compares on more diverse market contexts.
    snapshot_unique_count = 0
    if not candidates.empty and {"saved_spot_snapshot_path", "saved_chain_snapshot_path"}.issubset(candidates.columns):
        snapshot_unique_count = int(
            candidates[["saved_spot_snapshot_path", "saved_chain_snapshot_path"]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )
    if snapshot_unique_count < max(int(replay_limit), 1):
        archived = _archived_replay_rows(max(int(replay_limit) * 3, int(replay_limit), 1))
        if not archived.empty:
            archived = archived.assign(replay_candidate_source="archived_snapshot_pairs")
            candidates = _append_rows(candidates, archived)

    if candidates.empty:
        archived = _archived_replay_rows(max(int(replay_limit), 1))
        if archived.empty:
            return archived, "archived_snapshot_pairs"
        return archived.assign(replay_candidate_source="archived_snapshot_pairs"), "archived_snapshot_pairs"

    candidates = candidates.drop_duplicates(subset=["signal_id"], keep="first").copy()

    proxy_cols = [
        "signal_id",
        "baseline_calibrated_runtime_composite_score",
        "segmented_calibrated_runtime_composite_score",
        "baseline_proxy_selected",
        "segmented_proxy_selected",
    ]
    candidates = candidates.merge(proxy_compare[proxy_cols], on="signal_id", how="left")
    candidates["baseline_proxy_selected"] = pd.Series(candidates["baseline_proxy_selected"], dtype="boolean").fillna(False).astype(bool)
    candidates["segmented_proxy_selected"] = pd.Series(candidates["segmented_proxy_selected"], dtype="boolean").fillna(False).astype(bool)

    candidates["replay_priority"] = "neutral"
    disagreement_mask = candidates["baseline_proxy_selected"] != candidates["segmented_proxy_selected"]
    either_selected_mask = candidates["baseline_proxy_selected"] | candidates["segmented_proxy_selected"]
    candidates.loc[either_selected_mask, "replay_priority"] = "either_selected"
    candidates.loc[disagreement_mask, "replay_priority"] = "selection_disagreement"

    baseline_score = pd.to_numeric(candidates["baseline_calibrated_runtime_composite_score"], errors="coerce")
    segmented_score = pd.to_numeric(candidates["segmented_calibrated_runtime_composite_score"], errors="coerce")
    candidates["replay_score_abs_delta"] = (segmented_score - baseline_score).abs().fillna(0.0)
    candidates["replay_weak_data_score"] = 0.0
    weak_data_min_score = max(int(weak_data_min_score), 1)

    data_quality = candidates.get("data_quality_status", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper()
    confirmation = candidates.get("confirmation_status", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper()
    provider_health = candidates.get("provider_health_status", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper()
    candidates.loc[data_quality.isin({"WEAK", "CAUTION", "FRAGILE", "PARTIAL"}), "replay_weak_data_score"] += 1.0
    candidates.loc[confirmation.isin({"WEAK", "CAUTION", "PARTIAL"}), "replay_weak_data_score"] += 1.0
    candidates.loc[provider_health.isin({"FRAGILE", "DEGRADED", "PARTIAL"}), "replay_weak_data_score"] += 1.0

    if str(replay_priority_mode).strip().lower() in {"weak-data-heavy", "weak_data_heavy"}:
        # In stress mode, prefer weak-data candidate states before generic neutral rows.
        candidates.loc[
            (candidates["replay_priority"].eq("neutral")) & (candidates["replay_weak_data_score"] >= weak_data_min_score),
            "replay_priority",
        ] = "weak_data_focus"

    priority_rank = {
        "selection_disagreement": 0,
        "either_selected": 1,
        "weak_data_focus": 2,
        "neutral": 3,
    }
    candidates["replay_priority_rank"] = candidates["replay_priority"].map(priority_rank).fillna(4).astype(int)
    candidates["signal_timestamp"] = pd.to_datetime(candidates["signal_timestamp"], errors="coerce", format="mixed")

    direction = candidates.get("direction", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    gamma = candidates.get("gamma_regime", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    vol_raw = candidates.get("volatility_regime", pd.Series(index=candidates.index, dtype="object")).fillna(candidates.get("vol_regime"))
    vol = vol_raw.astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    source = candidates.get("source", pd.Series(index=candidates.index, dtype="object")).astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    candidates["replay_diversity_key"] = direction + "|" + gamma + "|" + vol + "|" + source

    if {"saved_spot_snapshot_path", "saved_chain_snapshot_path"}.issubset(candidates.columns):
        candidates["replay_snapshot_pair_key"] = (
            candidates["saved_spot_snapshot_path"].astype(str) + "||" + candidates["saved_chain_snapshot_path"].astype(str)
        )
        candidates = candidates.sort_values(
            ["replay_priority_rank", "replay_score_abs_delta", "signal_timestamp"],
            ascending=[True, False, False],
        )
        candidates = candidates.drop_duplicates(subset=["replay_snapshot_pair_key"], keep="first")

    limit = max(int(replay_limit), 1)
    if str(replay_priority_mode).strip().lower() in {"weak-data-heavy", "weak_data_heavy"}:
        quota_disagreement = max(1, int(round(limit * 0.35)))
        quota_either = max(1, int(round(limit * 0.20)))
        quota_weak_data_focus = max(1, int(round(limit * 0.35)))
        quota_neutral = max(1, limit - quota_disagreement - quota_either - quota_weak_data_focus)
        quotas = {
            0: quota_disagreement,
            1: quota_either,
            2: quota_weak_data_focus,
            3: quota_neutral,
        }
        rank_order = [0, 1, 2, 3]
    else:
        quota_disagreement = max(1, int(round(limit * 0.50)))
        quota_either = max(1, int(round(limit * 0.30)))
        quota_neutral = max(1, limit - quota_disagreement - quota_either)
        quotas = {
            0: quota_disagreement,
            1: quota_either,
            2: quota_neutral,
        }
        rank_order = [0, 1, 2]

    selected_chunks: list[pd.DataFrame] = []
    for rank in rank_order:
        bucket = candidates.loc[candidates["replay_priority_rank"].eq(rank)].copy()
        selected_chunks.append(_pick_diverse_rows(bucket, quotas[rank]))

    selected = pd.concat(selected_chunks, ignore_index=False)
    selected = selected.loc[~selected.index.duplicated(keep="first")]

    if len(selected) < limit:
        fill = candidates.loc[~candidates.index.isin(selected.index)].sort_values(
            ["replay_priority_rank", "replay_score_abs_delta", "signal_timestamp"],
            ascending=[True, False, False],
        )
        selected = pd.concat([selected, fill.head(limit - len(selected))], ignore_index=False)

    # Preserve outcome-linked replay evidence by enforcing a minimum share of
    # rows with realized labels when such rows are available.
    selected = selected.head(limit).copy()
    selected["_has_outcome"] = pd.to_numeric(selected.get("correct_60m"), errors="coerce").notna()
    min_outcome_rows = min(limit, max(10, int(round(limit * 0.40))))
    selected_outcome_count = int(selected["_has_outcome"].sum())
    if selected_outcome_count < min_outcome_rows:
        need = min_outcome_rows - selected_outcome_count
        outcome_candidates = candidates.loc[
            ~candidates.index.isin(selected.index)
            & pd.to_numeric(candidates.get("correct_60m"), errors="coerce").notna()
        ].sort_values(["replay_priority_rank", "replay_score_abs_delta", "signal_timestamp"], ascending=[True, False, False])
        replacements = outcome_candidates.head(need)
        if not replacements.empty:
            non_outcome = selected.loc[~selected["_has_outcome"]].sort_values(
                ["replay_priority_rank", "replay_score_abs_delta", "signal_timestamp"],
                ascending=[False, True, True],
            )
            drop_count = min(len(non_outcome), len(replacements))
            if drop_count > 0:
                selected = selected.drop(index=non_outcome.head(drop_count).index)
                selected = pd.concat([selected, replacements.head(drop_count)], ignore_index=False)

    selected = selected.drop(columns=["_has_outcome"], errors="ignore")

    selected = selected.sort_values(["replay_priority_rank", "replay_score_abs_delta", "signal_timestamp"], ascending=[True, False, False])
    replay_source = "prioritized_dataset_replay_rows"
    if str(replay_priority_mode).strip().lower() in {"weak-data-heavy", "weak_data_heavy"}:
        replay_source = "prioritized_weak_data_heavy_replay_rows"
    return selected.head(limit).reset_index(drop=True), replay_source


def _replay_sample_composition(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "total_rows": 0,
            "priority_counts": {},
            "source_label_counts": {},
            "direction_counts": {},
            "gamma_regime_counts": {},
            "vol_regime_counts": {},
        }

    def _counts(series: pd.Series) -> dict[str, int]:
        values = series.astype(str).str.upper().replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
        return {str(k): int(v) for k, v in values.value_counts().to_dict().items()}

    vol_source = frame.get("volatility_regime")
    if vol_source is None:
        vol_source = frame.get("vol_regime", pd.Series(index=frame.index, dtype="object"))
    else:
        vol_source = vol_source.fillna(frame.get("vol_regime"))

    return {
        "total_rows": int(len(frame)),
        "outcome_labeled_rows": int(pd.to_numeric(frame.get("correct_60m"), errors="coerce").notna().sum()),
        "candidate_source_counts": _counts(frame.get("replay_candidate_source", pd.Series(index=frame.index, dtype="object"))),
        "source_label_counts": _counts(frame.get("source", pd.Series(index=frame.index, dtype="object"))),
        "priority_counts": _counts(frame.get("replay_priority", pd.Series(index=frame.index, dtype="object"))),
        "direction_counts": _counts(frame.get("direction", pd.Series(index=frame.index, dtype="object"))),
        "gamma_regime_counts": _counts(frame.get("gamma_regime", pd.Series(index=frame.index, dtype="object"))),
        "vol_regime_counts": _counts(vol_source),
    }


def _recent_blocker_precedence_trends(
    *,
    output_root: Path,
    lookback_runs: int,
    current_run_dir: Path,
) -> list[dict[str, Any]]:
    if lookback_runs <= 0:
        return []

    if not output_root.exists():
        return []

    runs = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith("run_") and path != current_run_dir
    ]
    runs = sorted(runs, key=lambda p: p.name, reverse=True)

    records: list[dict[str, Any]] = []
    for run_dir in runs:
        summary_path = run_dir / "segmented_calibration_governance_summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        replay = payload.get("replay") if isinstance(payload.get("replay"), dict) else {}
        scenario_summaries = replay.get("scenario_summaries") if isinstance(replay.get("scenario_summaries"), dict) else {}
        for scenario_name, summary in scenario_summaries.items():
            if not isinstance(summary, dict):
                continue
            records.append(
                {
                    "run_id": run_dir.name,
                    "generated_at": payload.get("generated_at"),
                    "replay_source": replay.get("replay_source"),
                    "scenario": scenario_name,
                    "replayed_rows": summary.get("replayed_rows"),
                    "trade_count": summary.get("trade_count"),
                    "watchlist_count": summary.get("watchlist_count"),
                    "breaker_triggered_rows": summary.get("breaker_triggered_rows"),
                    "breaker_shadow_triggered_rows": summary.get("breaker_shadow_triggered_rows"),
                    "blocker_precedence_counts": summary.get("blocker_precedence_counts") or {},
                }
            )
            if len(records) >= lookback_runs * 2:
                break
        if len(records) >= lookback_runs * 2:
            break

    return records


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Segmented Calibration Governance Comparison",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Baseline calibrator: {payload['artifacts']['baseline_calibrator_path']}",
        f"- Segmented calibrator: {payload['artifacts']['segmented_calibrator_path']}",
        f"- Replay eligible rows: {payload['replay']['eligible_rows']}",
        f"- Proxy evaluated rows: {payload['proxy']['evaluated_rows']}",
        "",
        "## Exact Replay Summary",
        "",
        "| Scenario | Replayed | Trades | Watchlists | Signal 60m Hit Rate | Signal Avg Signed Return 60m (bps) | Trade 60m Hit Rate | Trade Avg Signed Return 60m (bps) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario_name, summary in payload["replay"]["scenario_summaries"].items():
        lines.append(
            f"| {scenario_name} | {summary.get('replayed_rows')} | {summary.get('trade_count')} | {summary.get('watchlist_count')} | {summary.get('signal_hit_rate_60m')} | {summary.get('signal_avg_signed_return_60m_bps')} | {summary.get('trade_hit_rate_60m')} | {summary.get('trade_avg_signed_return_60m_bps')} |"
        )
    lines.extend([
        "",
        "## Replay Sample Composition",
        "",
        f"- Total rows: {payload['replay'].get('sample_composition', {}).get('total_rows')}",
        f"- Outcome-labeled rows: {payload['replay'].get('sample_composition', {}).get('outcome_labeled_rows')}",
        f"- Candidate source mix: {payload['replay'].get('sample_composition', {}).get('candidate_source_counts')}",
        f"- Market source mix: {payload['replay'].get('sample_composition', {}).get('source_label_counts')}",
        f"- Priority mix: {payload['replay'].get('sample_composition', {}).get('priority_counts')}",
        f"- Direction mix: {payload['replay'].get('sample_composition', {}).get('direction_counts')}",
        f"- Gamma regime mix: {payload['replay'].get('sample_composition', {}).get('gamma_regime_counts')}",
        f"- Vol regime mix: {payload['replay'].get('sample_composition', {}).get('vol_regime_counts')}",
        "",
        "## Replay Guardrails",
        "",
    ])
    for scenario_name, summary in payload["replay"]["scenario_summaries"].items():
        lines.append(
            f"- {scenario_name}: breaker_evaluated={summary.get('breaker_evaluated_rows')}, breaker_triggered={summary.get('breaker_triggered_rows')}, shadow_triggered={summary.get('breaker_shadow_triggered_rows')}"
        )
        lines.append(f"  trigger_reasons={summary.get('breaker_trigger_reason_counts')}")
        lines.append(f"  shadow_reasons={summary.get('breaker_shadow_reason_counts')}")
        lines.append(f"  blocker_precedence={summary.get('blocker_precedence_counts')}")
        lines.append(f"  no_trade_reason_codes={summary.get('no_trade_reason_code_counts')}")
    lines.extend([
        "",
        "## Archive Proxy Summary",
        "",
        "| Scenario | Selected Count | 60m Hit Rate | Avg Signed Return 60m (bps) | Avg Realized Return 60m | CALL Share | PUT Share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for scenario_name, summary in payload["proxy"]["scenario_summaries"].items():
        lines.append(
            f"| {scenario_name} | {summary.get('selected_count')} | {summary.get('hit_rate_60m')} | {summary.get('avg_signed_return_60m_bps')} | {summary.get('avg_realized_return_60m')} | {summary.get('call_share')} | {summary.get('put_share')} |"
        )
    lines.extend([
        "",
        "## Deltas",
        "",
        f"- Replay trade-count delta: {payload['replay']['delta'].get('trade_count_delta')}",
        f"- Replay signal 60m hit-rate delta: {payload['replay']['delta'].get('signal_hit_rate_60m_delta')}",
        f"- Replay signal avg signed return 60m delta (bps): {payload['replay']['delta'].get('signal_avg_signed_return_60m_bps_delta')}",
        f"- Replay trade 60m hit-rate delta: {payload['replay']['delta'].get('trade_hit_rate_60m_delta')}",
        f"- Replay trade avg signed return 60m delta (bps): {payload['replay']['delta'].get('trade_avg_signed_return_60m_bps_delta')}",
        f"- Proxy selected-count delta: {payload['proxy']['delta'].get('selected_count_delta')}",
        f"- Proxy 60m hit-rate delta: {payload['proxy']['delta'].get('hit_rate_60m_delta')}",
        f"- Proxy avg signed return 60m delta (bps): {payload['proxy']['delta'].get('avg_signed_return_60m_bps_delta')}",
    ])

    trend_rows = payload.get("trend", {}).get("recent_blocker_precedence", [])
    if trend_rows:
        lines.extend([
            "",
            "## Recent Blocker Trend",
            "",
            "| Run | Scenario | Source | Trades | Watchlists | Provider Health | Weak Data Breaker | No Signal/Threshold | Other Blocker |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in trend_rows:
            counts = row.get("blocker_precedence_counts") or {}
            lines.append(
                f"| {row.get('run_id')} | {row.get('scenario')} | {row.get('replay_source')} | {row.get('trade_count')} | {row.get('watchlist_count')} | {counts.get('PROVIDER_HEALTH', 0)} | {counts.get('WEAK_DATA_CIRCUIT_BREAKER', 0)} | {counts.get('NO_SIGNAL_OR_THRESHOLD', 0)} | {counts.get('OTHER_BLOCKER', 0)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run governed segmented-calibration comparisons.")
    parser.add_argument("--dataset-path", default=str(CUMULATIVE_DATASET_PATH))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--replay-limit", type=int, default=25)
    parser.add_argument(
        "--replay-priority-mode",
        choices=["balanced", "weak-data-heavy"],
        default="balanced",
        help="Candidate selection emphasis for governed exact replay.",
    )
    parser.add_argument(
        "--weak-data-min-score",
        type=int,
        default=2,
        help="Minimum weak-data score required for weak-data focus prioritization in weak-data-heavy mode.",
    )
    parser.add_argument(
        "--trend-lookback-runs",
        type=int,
        default=5,
        help="Number of prior governance runs to include in blocker precedence trend diagnostics.",
    )
    args = parser.parse_args()

    run_dir = Path(args.output_root) / f"run_{_run_id_now()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_calibrator = _export_head_calibrator(run_dir)
    segmented_calibrator = ROOT / MODEL_RELATIVE_PATH
    if not segmented_calibrator.exists():
        raise FileNotFoundError(f"Segmented calibrator artifact missing: {segmented_calibrator}")

    baseline = Scenario("baseline_global", baseline_calibrator, "HEAD single-calibrator artifact")
    segmented = Scenario("segmented_runtime", segmented_calibrator, "Current segmented runtime artifact")

    dataset = load_signals_dataset(args.dataset_path)

    baseline_proxy = _proxy_frame(dataset, baseline)
    segmented_proxy = _proxy_frame(dataset, segmented)
    proxy_compare = baseline_proxy[
        [
            "signal_id",
            "proxy_raw_runtime_composite_score",
            "proxy_calibrated_runtime_composite_score",
            "proxy_effective_min_trade_strength",
            "proxy_effective_min_composite_score",
            "proxy_selected",
            "proxy_selected_segment_key",
        ]
    ].rename(
        columns={
            "proxy_calibrated_runtime_composite_score": "baseline_calibrated_runtime_composite_score",
            "proxy_selected": "baseline_proxy_selected",
            "proxy_selected_segment_key": "baseline_selected_segment_key",
        }
    ).merge(
        segmented_proxy[
            [
                "signal_id",
                "proxy_calibrated_runtime_composite_score",
                "proxy_selected",
                "proxy_selected_segment_key",
            ]
        ].rename(
            columns={
                "proxy_calibrated_runtime_composite_score": "segmented_calibrated_runtime_composite_score",
                "proxy_selected": "segmented_proxy_selected",
                "proxy_selected_segment_key": "segmented_selected_segment_key",
            }
        ),
        on="signal_id",
        how="inner",
    )

    replay_rows, replay_source = _replay_candidates_with_priority(
        dataset,
        proxy_compare,
        max(int(args.replay_limit), 1),
        replay_priority_mode=args.replay_priority_mode,
        weak_data_min_score=int(args.weak_data_min_score),
    )

    baseline_replay = _run_replay_scenario(replay_rows, baseline)
    segmented_replay = _run_replay_scenario(replay_rows, segmented)
    replay_compare = baseline_replay.merge(
        segmented_replay,
        on=[
            "signal_id",
            "signal_timestamp",
            "symbol",
            "direction",
            "correct_60m",
            "signed_return_60m_bps",
            "saved_spot_snapshot_path",
            "saved_chain_snapshot_path",
        ],
        how="outer",
        suffixes=("_baseline", "_segmented"),
    )
    replay_compare_path = run_dir / "replay_comparison.csv"
    replay_compare.to_csv(replay_compare_path, index=False)

    proxy_compare_path = run_dir / "proxy_selection_comparison.csv"
    proxy_compare.to_csv(proxy_compare_path, index=False)

    baseline_selected = baseline_proxy.loc[baseline_proxy["proxy_selected"]].copy()
    segmented_selected = segmented_proxy.loc[segmented_proxy["proxy_selected"]].copy()
    reports_dir = run_dir / "cumulative_reports"
    baseline_report = write_signal_evaluation_report(
        baseline_selected,
        production_pack_name="baseline_global_calibrator_proxy",
        dataset_path=f"{args.dataset_path} [proxy-selected]",
        output_dir=reports_dir,
        report_name="baseline_global_calibrator_proxy",
    )
    segmented_report = write_signal_evaluation_report(
        segmented_selected,
        production_pack_name="segmented_calibrator_proxy",
        dataset_path=f"{args.dataset_path} [proxy-selected]",
        output_dir=reports_dir,
        report_name="segmented_calibrator_proxy",
    )

    baseline_replay_summary = _summarize_replay(baseline_replay)
    segmented_replay_summary = _summarize_replay(segmented_replay)
    baseline_proxy_summary = _summarize_selected(baseline_selected)
    segmented_proxy_summary = _summarize_selected(segmented_selected)

    payload = {
        "generated_at": _utc_now(),
        "artifacts": {
            "baseline_calibrator_path": str(baseline_calibrator),
            "segmented_calibrator_path": str(segmented_calibrator),
            "replay_comparison_csv": str(replay_compare_path),
            "proxy_selection_comparison_csv": str(proxy_compare_path),
            "baseline_report_dir": baseline_report["report_dir"],
            "segmented_report_dir": segmented_report["report_dir"],
        },
        "replay": {
            "replay_source": replay_source,
            "eligible_rows": int(len(replay_rows)),
            "sample_composition": _replay_sample_composition(replay_rows),
            "scenario_summaries": {
                baseline.name: baseline_replay_summary,
                segmented.name: segmented_replay_summary,
            },
            "delta": {
                "trade_count_delta": int(segmented_replay_summary["trade_count"] - baseline_replay_summary["trade_count"]),
                "signal_hit_rate_60m_delta": _delta(segmented_replay_summary.get("signal_hit_rate_60m"), baseline_replay_summary.get("signal_hit_rate_60m")),
                "signal_avg_signed_return_60m_bps_delta": _delta(segmented_replay_summary.get("signal_avg_signed_return_60m_bps"), baseline_replay_summary.get("signal_avg_signed_return_60m_bps")),
                "trade_hit_rate_60m_delta": _delta(segmented_replay_summary.get("trade_hit_rate_60m"), baseline_replay_summary.get("trade_hit_rate_60m")),
                "trade_avg_signed_return_60m_bps_delta": _delta(segmented_replay_summary.get("trade_avg_signed_return_60m_bps"), baseline_replay_summary.get("trade_avg_signed_return_60m_bps")),
            },
        },
        "proxy": {
            "evaluated_rows": int(len(baseline_proxy)),
            "scenario_summaries": {
                baseline.name: baseline_proxy_summary,
                segmented.name: segmented_proxy_summary,
            },
            "delta": {
                "selected_count_delta": int(segmented_proxy_summary["selected_count"] - baseline_proxy_summary["selected_count"]),
                "hit_rate_60m_delta": _delta(segmented_proxy_summary.get("hit_rate_60m"), baseline_proxy_summary.get("hit_rate_60m")),
                "avg_signed_return_60m_bps_delta": _delta(segmented_proxy_summary.get("avg_signed_return_60m_bps"), baseline_proxy_summary.get("avg_signed_return_60m_bps")),
                "avg_realized_return_60m_delta": _delta(segmented_proxy_summary.get("avg_realized_return_60m"), baseline_proxy_summary.get("avg_realized_return_60m")),
            },
        },
        "trend": {
            "lookback_runs": int(args.trend_lookback_runs),
            "recent_blocker_precedence": _recent_blocker_precedence_trends(
                output_root=Path(args.output_root),
                lookback_runs=max(int(args.trend_lookback_runs), 0),
                current_run_dir=run_dir,
            ),
        },
    }

    summary_json = run_dir / "segmented_calibration_governance_summary.json"
    summary_md = run_dir / "segmented_calibration_governance_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "replay_eligible_rows": int(len(replay_rows)),
                "proxy_evaluated_rows": int(len(baseline_proxy)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())