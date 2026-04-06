#!/usr/bin/env python3
"""Governed replay uplift comparison for direction head OFF vs ON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine_runner import run_engine_snapshot
from config.policy_resolver import temporary_parameter_pack
from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset
from scripts.ops.run_segmented_calibration_governance import (
    _archived_replay_rows,
    _backfill_outcome_ready_replay_rows,
    _eligible_replay_rows,
)


OUTPUT_ROOT = ROOT / "research" / "reviews" / "direction_head_uplift_governance"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _weak_data_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=frame.index, dtype="int64")
    data_quality = frame.get("data_quality_status", pd.Series(index=frame.index, dtype="object")).astype(str).str.upper()
    provider = frame.get("provider_health_status", pd.Series(index=frame.index, dtype="object")).astype(str).str.upper()
    confirmation = frame.get("confirmation_status", pd.Series(index=frame.index, dtype="object")).astype(str).str.upper()
    score += data_quality.isin({"WEAK", "CAUTION", "FRAGILE", "PARTIAL"}).astype(int)
    score += provider.isin({"WEAK", "CAUTION", "FRAGILE", "DEGRADED", "PARTIAL"}).astype(int)
    score += confirmation.isin({"WEAK", "CAUTION", "MIXED", "PARTIAL"}).astype(int)
    return score


def _select_replay_rows(
    dataset: pd.DataFrame,
    replay_limit: int,
    replay_priority_mode: str = "balanced",
    weak_data_min_score: int = 1,
) -> tuple[pd.DataFrame, str]:
    rows = _eligible_replay_rows(dataset)
    source_label = "dataset_linked_snapshot_rows"

    if len(rows) < replay_limit:
        need = replay_limit - len(rows)
        backfilled = _backfill_outcome_ready_replay_rows(dataset, need)
        if not backfilled.empty:
            rows = pd.concat([rows, backfilled], ignore_index=True)
            source_label = "dataset_linked_plus_backfilled"

    if rows.empty:
        archived = _archived_replay_rows(replay_limit)
        if archived.empty:
            return archived, "none"
        return archived, "archived_snapshot_pairs"

    mode = str(replay_priority_mode or "balanced").strip().lower()
    if mode in {"weak-data-heavy", "weak_data_heavy"}:
        rows = rows.copy()
        rows["_weak_data_score"] = _weak_data_score(rows)
        rows = rows.sort_values(["_weak_data_score", "signal_timestamp"], ascending=[False, False], na_position="last")
        rows = rows.loc[
            (rows["_weak_data_score"] >= max(int(weak_data_min_score), 1))
            | rows["_weak_data_score"].eq(rows["_weak_data_score"].max())
        ].copy()
        source_label = "weak_data_heavy_priority_rows"

    rows = rows.drop_duplicates(subset=["signal_id"], keep="first").head(replay_limit).reset_index(drop=True)
    rows = rows.drop(columns=["_weak_data_score"], errors="ignore")
    return rows, source_label


def _run_scenario(
    frame: pd.DataFrame,
    scenario_name: str,
    head_enabled: int,
    direction_calibrator_path: str | None,
    min_trade_strength_override: int | None,
    min_composite_score_override: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    overrides: dict[str, Any] = {
        "trade_strength.runtime_thresholds.enable_probabilistic_direction_head": int(head_enabled),
        "trade_strength.runtime_thresholds.direction_head_min_confidence": 0.0,
        "trade_strength.runtime_thresholds.direction_head_override_min_confidence": 0.0,
        "trade_strength.runtime_thresholds.direction_head_allow_vote_override": 1,
    }
    if direction_calibrator_path:
        overrides["trade_strength.runtime_thresholds.direction_probability_calibrator_path"] = str(direction_calibrator_path)
    if min_trade_strength_override is not None:
        overrides["trade_strength.runtime_thresholds.min_trade_strength"] = int(min_trade_strength_override)
    if min_composite_score_override is not None:
        overrides["trade_strength.runtime_thresholds.min_composite_score"] = int(min_composite_score_override)

    with temporary_parameter_pack("baseline_v1", overrides=overrides):
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
                    "warnings": ["direction_head_uplift_governance_neutral_global_market"],
                    "issues": [],
                },
            )

            trade = _trade_payload(result)
            full_trade = result.get("trade") if isinstance(result.get("trade"), dict) else {}

            def _trade_field(key: str) -> Any:
                value = trade.get(key) if isinstance(trade, dict) else None
                if value is None:
                    value = full_trade.get(key) if isinstance(full_trade, dict) else None
                return value

            rows.append(
                {
                    "scenario": scenario_name,
                    "signal_id": row.get("signal_id"),
                    "signal_timestamp": row.get("signal_timestamp"),
                    "symbol": row.get("symbol"),
                    "source": row.get("source"),
                    "gamma_regime": row.get("gamma_regime"),
                    "volatility_regime": row.get("volatility_regime") or row.get("vol_regime"),
                    "correct_60m": row.get("correct_60m"),
                    "signed_return_60m_bps": row.get("signed_return_60m_bps"),
                    "ok": bool(result.get("ok")),
                    "reason": result.get("reason"),
                    "error": result.get("error"),
                    "trade_status": _trade_field("trade_status"),
                    "direction": _trade_field("direction"),
                    "direction_source": _trade_field("direction_source"),
                    "direction_vote_shadow": _trade_field("direction_vote_shadow"),
                    "direction_head_probability_up": _trade_field("direction_head_probability_up"),
                    "direction_head_confidence": _trade_field("direction_head_confidence"),
                    "direction_head_microstructure_friction_score": _trade_field("direction_head_microstructure_friction_score"),
                    "direction_head_used_for_final": _trade_field("direction_head_used_for_final"),
                    "runtime_composite_score": _trade_field("runtime_composite_score"),
                }
            )

    return pd.DataFrame(rows)


def _summarize_trade_slice(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "trade_count": 0,
            "trade_hit_rate_60m": None,
            "trade_avg_signed_return_60m_bps": None,
            "avg_head_confidence": None,
            "avg_head_microstructure_friction": None,
        }

    trade_rows = frame.loc[frame["trade_status"].astype(str).str.upper().eq("TRADE")].copy()
    hit = pd.to_numeric(trade_rows["correct_60m"] if "correct_60m" in trade_rows.columns else pd.Series(dtype=float), errors="coerce")
    ret = pd.to_numeric(trade_rows["signed_return_60m_bps"] if "signed_return_60m_bps" in trade_rows.columns else pd.Series(dtype=float), errors="coerce")
    conf = pd.to_numeric(trade_rows["direction_head_confidence"] if "direction_head_confidence" in trade_rows.columns else pd.Series(dtype=float), errors="coerce")
    friction = pd.to_numeric(
        trade_rows["direction_head_microstructure_friction_score"]
        if "direction_head_microstructure_friction_score" in trade_rows.columns
        else pd.Series(dtype=float),
        errors="coerce",
    )

    return {
        "rows": int(len(frame)),
        "trade_count": int(len(trade_rows)),
        "trade_hit_rate_60m": None if hit.dropna().empty else round(float(hit.mean()), 6),
        "trade_avg_signed_return_60m_bps": None if ret.dropna().empty else round(float(ret.mean()), 6),
        "avg_head_confidence": None if conf.dropna().empty else round(float(conf.mean()), 6),
        "avg_head_microstructure_friction": None if friction.dropna().empty else round(float(friction.mean()), 6),
    }


def _direction_slice(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["direction"] = rows["direction"].astype(str).str.upper()
    rows = rows.loc[rows["direction"].isin(["CALL", "PUT"])].copy()
    rows["signed_return_60m_bps"] = pd.to_numeric(
        rows["signed_return_60m_bps"] if "signed_return_60m_bps" in rows.columns else pd.Series(dtype=float),
        errors="coerce",
    )
    rows = rows.loc[rows["signed_return_60m_bps"].notna()].copy()
    if rows.empty:
        return rows

    rows["directional_return_bps"] = rows["signed_return_60m_bps"]
    put_mask = rows["direction"].eq("PUT")
    rows.loc[put_mask, "directional_return_bps"] = -rows.loc[put_mask, "signed_return_60m_bps"]
    rows["direction_correct_60m"] = (rows["directional_return_bps"] > 0).astype(float)
    return rows


def _summarize_direction_slice(frame: pd.DataFrame) -> dict[str, Any]:
    rows = _direction_slice(frame)
    if rows.empty:
        return {
            "direction_count": 0,
            "direction_accuracy_60m": None,
            "avg_directional_return_60m_bps": None,
        }
    return {
        "direction_count": int(len(rows)),
        "direction_accuracy_60m": round(float(rows["direction_correct_60m"].mean()), 6),
        "avg_directional_return_60m_bps": round(float(rows["directional_return_bps"].mean()), 6),
    }


def _group_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = _direction_slice(frame)
    if rows.empty:
        return pd.DataFrame(columns=[group_col, "direction_count", "direction_accuracy_60m", "avg_directional_return_60m_bps"])

    grouped = (
        rows.groupby(group_col, dropna=False)
        .agg(
            direction_count=("signal_id", "size"),
            direction_accuracy_60m=("direction_correct_60m", "mean"),
            avg_directional_return_60m_bps=("directional_return_bps", "mean"),
        )
        .reset_index()
    )
    grouped["direction_accuracy_60m"] = pd.to_numeric(grouped["direction_accuracy_60m"], errors="coerce").round(6)
    grouped["avg_directional_return_60m_bps"] = pd.to_numeric(grouped["avg_directional_return_60m_bps"], errors="coerce").round(6)
    return grouped


def _delta_table(off_df: pd.DataFrame, on_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    off_group = _group_summary(off_df, group_col).add_suffix("_off")
    on_group = _group_summary(on_df, group_col).add_suffix("_on")

    merged = off_group.merge(
        on_group,
        left_on=f"{group_col}_off",
        right_on=f"{group_col}_on",
        how="outer",
    )

    merged[group_col] = merged[f"{group_col}_off"].combine_first(merged[f"{group_col}_on"])
    merged["direction_count_delta"] = pd.to_numeric(merged["direction_count_on"], errors="coerce").fillna(0) - pd.to_numeric(merged["direction_count_off"], errors="coerce").fillna(0)
    merged["direction_accuracy_60m_delta"] = pd.to_numeric(merged["direction_accuracy_60m_on"], errors="coerce") - pd.to_numeric(merged["direction_accuracy_60m_off"], errors="coerce")
    merged["avg_directional_return_60m_bps_delta"] = pd.to_numeric(merged["avg_directional_return_60m_bps_on"], errors="coerce") - pd.to_numeric(merged["avg_directional_return_60m_bps_off"], errors="coerce")

    cols = [
        group_col,
        "direction_count_off",
        "direction_count_on",
        "direction_count_delta",
        "direction_accuracy_60m_off",
        "direction_accuracy_60m_on",
        "direction_accuracy_60m_delta",
        "avg_directional_return_60m_bps_off",
        "avg_directional_return_60m_bps_on",
        "avg_directional_return_60m_bps_delta",
    ]
    return merged[cols].sort_values(group_col, na_position="last").reset_index(drop=True)


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Direction Head Uplift Governance",
        "",
        f"- run_id: {payload.get('run_id')}",
        f"- generated_at_utc: {payload.get('generated_at_utc')}",
        f"- replay_selection_source: {payload.get('replay_selection_source')}",
        f"- replay_rows: {payload.get('replay_rows')}",
        "",
        "## Overall",
        "",
        f"- head_off: {json.dumps(payload.get('overall', {}).get('head_off', {}), sort_keys=True)}",
        f"- head_on: {json.dumps(payload.get('overall', {}).get('head_on', {}), sort_keys=True)}",
        f"- delta: {json.dumps(payload.get('overall', {}).get('delta', {}), sort_keys=True)}",
        "",
        "## Group Deltas",
        "",
        f"- scenario_diff: {json.dumps(payload.get('scenario_diff', {}), sort_keys=True)}",
        "",
        f"- by_source_csv: {payload.get('artifacts', {}).get('delta_by_source_csv')}",
        f"- by_gamma_regime_csv: {payload.get('artifacts', {}).get('delta_by_gamma_regime_csv')}",
        f"- by_volatility_regime_csv: {payload.get('artifacts', {}).get('delta_by_volatility_regime_csv')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run governed replay uplift for direction head")
    parser.add_argument("--dataset", type=Path, default=CUMULATIVE_DATASET_PATH, help="Input cumulative dataset path")
    parser.add_argument("--replay-limit", type=int, default=200, help="Number of replay rows")
    parser.add_argument(
        "--replay-priority-mode",
        choices=["balanced", "weak-data-heavy"],
        default="balanced",
        help="Replay selection mode. weak-data-heavy prioritizes fragile microstructure contexts.",
    )
    parser.add_argument("--weak-data-min-score", type=int, default=1, help="Minimum weak-data score for weak-data-heavy mode")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Output root directory")
    parser.add_argument(
        "--direction-calibrator-path",
        type=str,
        default=str(ROOT / "models_store" / "direction_probability_calibrator.json"),
        help="Direction-head calibrator path used for head-on scenario",
    )
    parser.add_argument("--override-min-trade-strength", type=int, default=None, help="Optional replay-only override for min trade strength")
    parser.add_argument("--override-min-composite-score", type=int, default=None, help="Optional replay-only override for min composite score")
    parser.add_argument(
        "--allow-zero-direction-changes",
        action="store_true",
        help="Bypass fail-fast guardrail when ON/OFF scenarios produce zero direction changes.",
    )
    args = parser.parse_args()

    dataset = load_signals_dataset(args.dataset)
    if dataset.empty:
        raise RuntimeError("Dataset is empty; cannot run governed uplift replay")

    replay_rows, replay_source = _select_replay_rows(
        dataset,
        max(1, int(args.replay_limit)),
        replay_priority_mode=args.replay_priority_mode,
        weak_data_min_score=int(args.weak_data_min_score),
    )
    if replay_rows.empty:
        raise RuntimeError("No replay rows available")

    run_id = _run_id()
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    head_off = _run_scenario(
        replay_rows,
        "direction_head_off",
        0,
        None,
        args.override_min_trade_strength,
        args.override_min_composite_score,
    )
    head_on = _run_scenario(
        replay_rows,
        "direction_head_on",
        1,
        args.direction_calibrator_path,
        args.override_min_trade_strength,
        args.override_min_composite_score,
    )

    off_csv = run_dir / "scenario_direction_head_off.csv"
    on_csv = run_dir / "scenario_direction_head_on.csv"
    head_off.to_csv(off_csv, index=False)
    head_on.to_csv(on_csv, index=False)

    delta_source = _delta_table(head_off, head_on, "source")
    delta_gamma = _delta_table(head_off, head_on, "gamma_regime")
    delta_vol = _delta_table(head_off, head_on, "volatility_regime")

    comparison = head_off.merge(
        head_on,
        on="signal_id",
        suffixes=("_off", "_on"),
        how="inner",
    )
    direction_changed = int((comparison.get("direction_off") != comparison.get("direction_on")).sum()) if not comparison.empty else 0
    source_changed = int((comparison.get("direction_source_off") != comparison.get("direction_source_on")).sum()) if not comparison.empty else 0
    head_used_rows = (
        int(
            pd.to_numeric(
                head_on["direction_head_used_for_final"]
                if "direction_head_used_for_final" in head_on.columns
                else pd.Series(dtype=float),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        if not head_on.empty
        else 0
    )

    if int(len(comparison)) > 0 and int(direction_changed) == 0 and not bool(args.allow_zero_direction_changes):
        raise RuntimeError(
            "Guardrail triggered: direction_changed_rows == 0 across matched rows. "
            "Use --allow-zero-direction-changes only for smoke/testing runs."
        )

    delta_source_csv = run_dir / "delta_by_source.csv"
    delta_gamma_csv = run_dir / "delta_by_gamma_regime.csv"
    delta_vol_csv = run_dir / "delta_by_volatility_regime.csv"
    delta_source.to_csv(delta_source_csv, index=False)
    delta_gamma.to_csv(delta_gamma_csv, index=False)
    delta_vol.to_csv(delta_vol_csv, index=False)

    overall_off = _summarize_trade_slice(head_off)
    overall_on = _summarize_trade_slice(head_on)
    direction_off = _summarize_direction_slice(head_off)
    direction_on = _summarize_direction_slice(head_on)

    payload = {
        "run_id": run_id,
        "generated_at_utc": _utc_now(),
        "dataset_path": str(args.dataset),
        "replay_rows": int(len(replay_rows)),
        "replay_selection_source": replay_source,
        "runtime_overrides": {
            "min_trade_strength": args.override_min_trade_strength,
            "min_composite_score": args.override_min_composite_score,
        },
        "scenario_diff": {
            "matched_rows": int(len(comparison)),
            "direction_changed_rows": direction_changed,
            "direction_source_changed_rows": source_changed,
            "head_used_for_final_rows": head_used_rows,
        },
        "overall": {
            "head_off": overall_off,
            "head_on": overall_on,
            "delta": {
                "trade_count": int(overall_on.get("trade_count", 0)) - int(overall_off.get("trade_count", 0)),
                "trade_hit_rate_60m": None
                if overall_on.get("trade_hit_rate_60m") is None or overall_off.get("trade_hit_rate_60m") is None
                else round(float(overall_on["trade_hit_rate_60m"]) - float(overall_off["trade_hit_rate_60m"]), 6),
                "trade_avg_signed_return_60m_bps": None
                if overall_on.get("trade_avg_signed_return_60m_bps") is None or overall_off.get("trade_avg_signed_return_60m_bps") is None
                else round(float(overall_on["trade_avg_signed_return_60m_bps"]) - float(overall_off["trade_avg_signed_return_60m_bps"]), 6),
            },
            "direction_head_off": direction_off,
            "direction_head_on": direction_on,
            "direction_delta": {
                "direction_count": int(direction_on.get("direction_count", 0)) - int(direction_off.get("direction_count", 0)),
                "direction_accuracy_60m": None
                if direction_on.get("direction_accuracy_60m") is None or direction_off.get("direction_accuracy_60m") is None
                else round(float(direction_on["direction_accuracy_60m"]) - float(direction_off["direction_accuracy_60m"]), 6),
                "avg_directional_return_60m_bps": None
                if direction_on.get("avg_directional_return_60m_bps") is None or direction_off.get("avg_directional_return_60m_bps") is None
                else round(float(direction_on["avg_directional_return_60m_bps"]) - float(direction_off["avg_directional_return_60m_bps"]), 6),
            },
        },
        "artifacts": {
            "scenario_head_off_csv": str(off_csv),
            "scenario_head_on_csv": str(on_csv),
            "delta_by_source_csv": str(delta_source_csv),
            "delta_by_gamma_regime_csv": str(delta_gamma_csv),
            "delta_by_volatility_regime_csv": str(delta_vol_csv),
        },
    }

    json_path = run_dir / "summary.json"
    md_path = run_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_summary(md_path, payload)

    print(json.dumps({"run_id": run_id, "summary_json": str(json_path), "summary_md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
