#!/usr/bin/env python3
"""Run direction-head uplift governance across multiple replay settings and build one promotion report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_PATH = ROOT / "scripts" / "ops" / "run_direction_head_uplift_governance.py"
OUTPUT_ROOT = ROOT / "research" / "reviews" / "direction_head_uplift_governance"
REPORT_ROOT = ROOT / "research" / "reviews" / "direction_head_promotion_matrix"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _scenario_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "strict_balanced",
            "args": ["--replay-limit", "120"],
        },
        {
            "name": "mild_override_balanced",
            "args": [
                "--replay-limit",
                "120",
                "--override-min-trade-strength",
                "52",
                "--override-min-composite-score",
                "48",
            ],
        },
        {
            "name": "strict_weak_data_heavy",
            "args": [
                "--replay-limit",
                "120",
                "--replay-priority-mode",
                "weak-data-heavy",
            ],
        },
        {
            "name": "mild_override_weak_data_heavy",
            "args": [
                "--replay-limit",
                "120",
                "--replay-priority-mode",
                "weak-data-heavy",
                "--override-min-trade-strength",
                "52",
                "--override-min-composite-score",
                "48",
            ],
        },
    ]


def _run_scenario(py_exec: str, base_args: list[str], scenario: dict[str, Any]) -> dict[str, Any]:
    cmd = [py_exec, str(SCRIPT_PATH)] + base_args + list(scenario["args"])
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Scenario '{scenario['name']}' failed with code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )

    payload = json.loads(proc.stdout)
    summary_path = Path(payload["summary_json"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    overall = summary.get("overall", {})
    scenario_diff = summary.get("scenario_diff", {})
    trade_off = overall.get("head_off", {})
    trade_on = overall.get("head_on", {})
    dir_off = overall.get("direction_head_off", {})
    dir_on = overall.get("direction_head_on", {})
    trade_delta = overall.get("delta", {})
    dir_delta = overall.get("direction_delta", {})

    return {
        "scenario": scenario["name"],
        "run_id": summary.get("run_id"),
        "summary_json": str(summary_path),
        "summary_md": str(payload.get("summary_md")),
        "replay_rows": summary.get("replay_rows"),
        "trade_count_off": trade_off.get("trade_count"),
        "trade_count_on": trade_on.get("trade_count"),
        "trade_count_delta": trade_delta.get("trade_count"),
        "trade_hit_rate_60m_off": trade_off.get("trade_hit_rate_60m"),
        "trade_hit_rate_60m_on": trade_on.get("trade_hit_rate_60m"),
        "trade_hit_rate_60m_delta": trade_delta.get("trade_hit_rate_60m"),
        "direction_count_off": dir_off.get("direction_count"),
        "direction_count_on": dir_on.get("direction_count"),
        "direction_count_delta": dir_delta.get("direction_count"),
        "direction_accuracy_60m_off": dir_off.get("direction_accuracy_60m"),
        "direction_accuracy_60m_on": dir_on.get("direction_accuracy_60m"),
        "direction_accuracy_60m_delta": dir_delta.get("direction_accuracy_60m"),
        "avg_directional_return_60m_bps_off": dir_off.get("avg_directional_return_60m_bps"),
        "avg_directional_return_60m_bps_on": dir_on.get("avg_directional_return_60m_bps"),
        "avg_directional_return_60m_bps_delta": dir_delta.get("avg_directional_return_60m_bps"),
        "direction_changed_rows": scenario_diff.get("direction_changed_rows"),
        "direction_source_changed_rows": scenario_diff.get("direction_source_changed_rows"),
        "head_used_for_final_rows": scenario_diff.get("head_used_for_final_rows"),
        "matched_rows": scenario_diff.get("matched_rows"),
    }


def _evaluate_scenario_gate(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    matched_rows = int(_safe_float(row.get("matched_rows"), 0.0) or 0.0)
    changed_rows = int(_safe_float(row.get("direction_changed_rows"), 0.0) or 0.0)
    changed_share = (float(changed_rows) / float(max(matched_rows, 1))) if matched_rows > 0 else 0.0
    acc_delta = _safe_float(row.get("direction_accuracy_60m_delta"), 0.0) or 0.0
    return_delta = _safe_float(row.get("avg_directional_return_60m_bps_delta"), 0.0) or 0.0
    trade_count_off = int(_safe_float(row.get("trade_count_off"), 0.0) or 0.0)
    trade_count_on = int(_safe_float(row.get("trade_count_on"), 0.0) or 0.0)

    hard_fail_reasons: list[str] = []
    pass_checks: list[str] = []
    caution_reasons: list[str] = []

    if matched_rows < int(thresholds["min_matched_rows_block"]):
        hard_fail_reasons.append("matched_rows_below_block_floor")
    if changed_rows <= 0:
        hard_fail_reasons.append("no_direction_changes")
    if acc_delta < float(thresholds["min_direction_accuracy_delta_block"]):
        hard_fail_reasons.append("direction_accuracy_delta_below_block_floor")
    if return_delta < float(thresholds["min_direction_return_delta_bps_block"]):
        hard_fail_reasons.append("direction_return_delta_below_block_floor")

    if changed_share >= float(thresholds["min_direction_changed_share_pass"]):
        pass_checks.append("direction_changed_share")
    else:
        caution_reasons.append("direction_changed_share_below_pass_floor")

    if acc_delta >= float(thresholds["min_direction_accuracy_delta_pass"]):
        pass_checks.append("direction_accuracy_delta")
    else:
        caution_reasons.append("direction_accuracy_delta_below_pass_floor")

    if return_delta >= float(thresholds["min_direction_return_delta_bps_pass"]):
        pass_checks.append("direction_return_delta_bps")
    else:
        caution_reasons.append("direction_return_delta_bps_below_pass_floor")

    min_trade_evidence_rows = int(thresholds["min_trade_evidence_rows_pass"])
    if min_trade_evidence_rows > 0:
        trade_evidence = (trade_count_off + trade_count_on) >= min_trade_evidence_rows
        if trade_evidence:
            pass_checks.append("trade_evidence")
        else:
            caution_reasons.append("insufficient_trade_evidence")
    else:
        pass_checks.append("trade_evidence_not_required")

    if hard_fail_reasons:
        status = "BLOCK"
    elif len(caution_reasons) == 0:
        status = "PASS"
    else:
        status = "CAUTION"

    return {
        "status": status,
        "metrics": {
            "matched_rows": matched_rows,
            "direction_changed_rows": changed_rows,
            "direction_changed_share": round(changed_share, 6),
            "direction_accuracy_60m_delta": round(float(acc_delta), 6),
            "avg_directional_return_60m_bps_delta": round(float(return_delta), 6),
            "trade_count_off": trade_count_off,
            "trade_count_on": trade_count_on,
            "trade_evidence_rows": int(trade_count_off + trade_count_on),
        },
        "pass_checks": pass_checks,
        "caution_reasons": caution_reasons,
        "hard_fail_reasons": hard_fail_reasons,
    }


def _evaluate_matrix_gate(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    scenario_results: dict[str, Any] = {}
    status_counts = {"PASS": 0, "CAUTION": 0, "BLOCK": 0}

    for row in rows:
        result = _evaluate_scenario_gate(row, thresholds)
        scenario_name = str(row.get("scenario"))
        scenario_results[scenario_name] = result
        status_counts[result["status"]] = int(status_counts.get(result["status"], 0) + 1)

    if status_counts["BLOCK"] > 0:
        overall_status = "BLOCK"
    elif status_counts["CAUTION"] > 0:
        overall_status = "CAUTION"
    else:
        overall_status = "PASS"

    return {
        "overall_status": overall_status,
        "status_counts": status_counts,
        "thresholds": thresholds,
        "scenario_results": scenario_results,
    }


def _render_markdown(run_id: str, rows: list[dict[str, Any]], artifacts_json: Path, gate: dict[str, Any]) -> str:
    lines = [
        "# Direction Head Promotion Matrix",
        "",
        f"- run_id: {run_id}",
        f"- generated_at_utc: {_utc_now()}",
        f"- artifacts_json: {artifacts_json}",
        f"- promotion_gate_status: {gate.get('overall_status')}",
        "",
        "## Promotion Gate",
        "",
        f"- status_counts: {gate.get('status_counts')}",
        "",
        "| Scenario | Gate Status | Hard Fails | Cautions |",
        "| --- | --- | --- | --- |",
    ]

    for row in rows:
        scenario_name = str(row.get("scenario"))
        scenario_gate = gate.get("scenario_results", {}).get(scenario_name, {})
        lines.append(
            "| {scenario} | {status} | {hard} | {caution} |".format(
                scenario=scenario_name,
                status=scenario_gate.get("status", "UNKNOWN"),
                hard=", ".join(scenario_gate.get("hard_fail_reasons", []) or []) or "-",
                caution=", ".join(scenario_gate.get("caution_reasons", []) or []) or "-",
            )
        )

    lines.extend([
        "",
        "## Compact Uplift Table",
        "",
        "| Scenario | Replay Rows | Trade Delta | Direction Delta | Dir Acc Delta | Dir Return Delta (bps) | Direction Changed Rows | Head Used Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for row in rows:
        lines.append(
            "| {scenario} | {replay_rows} | {trade_count_delta} | {direction_count_delta} | {direction_accuracy_60m_delta} | {avg_directional_return_60m_bps_delta} | {direction_changed_rows} | {head_used_for_final_rows} |".format(
                scenario=row.get("scenario"),
                replay_rows=row.get("replay_rows"),
                trade_count_delta=row.get("trade_count_delta"),
                direction_count_delta=row.get("direction_count_delta"),
                direction_accuracy_60m_delta=row.get("direction_accuracy_60m_delta"),
                avg_directional_return_60m_bps_delta=row.get("avg_directional_return_60m_bps_delta"),
                direction_changed_rows=row.get("direction_changed_rows"),
                head_used_for_final_rows=row.get("head_used_for_final_rows"),
            )
        )

    lines.extend([
        "",
        "## Scenario Artifacts",
        "",
    ])
    for row in rows:
        lines.append(f"- {row.get('scenario')}: {row.get('summary_json')}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direction-head promotion matrix")
    parser.add_argument("--python-executable", type=str, default=sys.executable, help="Python executable for child runs")
    parser.add_argument("--direction-calibrator-path", type=str, default=str(ROOT / "models_store" / "direction_probability_calibrator.json"))
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--min-matched-rows-block", type=int, default=60, help="Block if matched rows are below this floor")
    parser.add_argument("--min-direction-changed-share-pass", type=float, default=0.10, help="Pass floor for direction_changed_rows / matched_rows")
    parser.add_argument("--min-direction-accuracy-delta-pass", type=float, default=0.02, help="Pass floor for direction accuracy delta")
    parser.add_argument("--min-direction-return-delta-bps-pass", type=float, default=2.0, help="Pass floor for directional return delta (bps)")
    parser.add_argument(
        "--min-trade-evidence-rows-pass",
        type=int,
        default=0,
        help="Pass floor for trade evidence rows (off+on). Set to 0 for signal-quality-first governance.",
    )
    parser.add_argument("--min-direction-accuracy-delta-block", type=float, default=-0.02, help="Block floor for direction accuracy delta")
    parser.add_argument("--min-direction-return-delta-bps-block", type=float, default=-5.0, help="Block floor for directional return delta (bps)")
    parser.add_argument("--fail-on-block", action="store_true", help="Exit non-zero when overall gate status is BLOCK")
    args = parser.parse_args()

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing uplift governance script: {SCRIPT_PATH}")

    run_id = _run_id()
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_args = ["--direction-calibrator-path", args.direction_calibrator_path]

    rows: list[dict[str, Any]] = []
    for scenario in _scenario_definitions():
        rows.append(_run_scenario(args.python_executable, base_args, scenario))

    gate_thresholds = {
        "min_matched_rows_block": int(args.min_matched_rows_block),
        "min_direction_changed_share_pass": float(args.min_direction_changed_share_pass),
        "min_direction_accuracy_delta_pass": float(args.min_direction_accuracy_delta_pass),
        "min_direction_return_delta_bps_pass": float(args.min_direction_return_delta_bps_pass),
        "min_trade_evidence_rows_pass": int(args.min_trade_evidence_rows_pass),
        "min_direction_accuracy_delta_block": float(args.min_direction_accuracy_delta_block),
        "min_direction_return_delta_bps_block": float(args.min_direction_return_delta_bps_block),
    }
    gate = _evaluate_matrix_gate(rows, gate_thresholds)

    json_path = run_dir / "direction_head_promotion_matrix.json"
    md_path = run_dir / "direction_head_promotion_matrix.md"
    matrix_payload = {
        "run_id": run_id,
        "generated_at_utc": _utc_now(),
        "rows": rows,
        "promotion_gate": gate,
    }
    json_path.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(run_id, rows, json_path, gate), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "matrix_json": str(json_path), "matrix_md": str(md_path)}, indent=2))
    if bool(args.fail_on_block) and str(gate.get("overall_status")) == "BLOCK":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
