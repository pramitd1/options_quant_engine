from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from research.signal_evaluation.threshold_signal_rollout_monitor import (
    CANDIDATE_SIGNAL_ROLLOUT_BLOCKED,
    CANDIDATE_SIGNAL_ROLLOUT_HEALTHY,
    CANDIDATE_SIGNAL_ROLLOUT_WATCH,
    DEFAULT_CONFIG_HINT,
    build_threshold_signal_rollout_monitor_report,
    write_threshold_signal_rollout_monitor_report,
)


def _frame(*, post: bool = True, include_pack: bool = True, pack_name: str = "baseline_v1", side_effect: bool = False) -> pd.DataFrame:
    base = pd.Timestamp("2026-02-02T09:20:00+05:30") if post else pd.Timestamp("2026-01-20T09:20:00+05:30")
    rows = []
    for idx, composite in enumerate([80.0, 90.0, 70.0], start=1):
        row = {
            "signal_id": f"sig-{idx}",
            "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
            "source": "unit",
            "requested_option_source": "nse",
            "option_source": "nse",
            "spot_source": "yfinance",
            "market_data_source_consistency": "MATCHED",
            "market_data_provenance_status": "OK",
            "market_data_trade_blocking_status": "CLEAR",
            "market_data_timestamp_status": "FRESH",
            "trade_strength": 70.0,
            "composite_signal_score": composite,
            "tradeability_score": 70.0,
            "hybrid_move_probability": 0.70,
            "option_efficiency_score": 50.0,
            "global_risk_score": 20.0,
            "overnight_hold_allowed": True,
            "correct_60m": 1.0 if composite >= 80 else 0.0,
            "signed_return_60m_bps": 20.0 if composite >= 80 else -10.0,
        }
        if include_pack:
            row["parameter_pack_name"] = pack_name
        if side_effect and idx == 1:
            row["order_id"] = "broker-order-1"
        rows.append(row)
    return pd.DataFrame(rows)


def _adoption_report() -> dict:
    return {
        "adoption_status": "ADOPTED_MANUALLY",
        "comparison": {
            "config_hint": DEFAULT_CONFIG_HINT,
            "candidate_value": 85.0,
            "observed_runtime_value": 85.0,
        },
        "approval_decision": {
            "reviewed_at": "2026-02-01T00:00:00Z",
        },
    }


def _build(frame: pd.DataFrame, **kwargs):
    with temporary_parameter_pack("baseline_v1"):
        return build_threshold_signal_rollout_monitor_report(
            frame,
            baseline_pack_name="baseline_v1",
            candidate_pack_name="baseline_v1",
            candidate_overrides={DEFAULT_CONFIG_HINT: 85.0},
            adoption_reconciliation_report=_adoption_report(),
            adoption_start_at="2026-02-01T00:00:00Z",
            **kwargs,
        )


def test_signal_rollout_monitor_marks_healthy_for_traceable_candidate_signals():
    report = _build(_frame())

    assert report["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_HEALTHY
    assert report["runtime_config_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["candidate_runtime_value"] == 85.0
    assert report["policy_delta"] == {"composite_signal_score_floor": {"baseline": 75.0, "candidate": 85.0}}
    assert report["rollout_comparison"]["candidate_only_count"] == 0
    assert report["post_adoption_traceability"]["candidate_pack_signal_count"] == 3
    assert report["candidate_label_readiness"]["post_promotion_monitor_ready"] is True


def test_signal_rollout_monitor_uses_runtime_activation_marker_for_traceability():
    frame = pd.concat(
        [
            _frame(pack_name="old_live_pack").assign(
                signal_id=["pre-1", "pre-2", "pre-3"],
                signal_timestamp=[
                    "2026-02-02T09:20:00+05:30",
                    "2026-02-02T09:25:00+05:30",
                    "2026-02-02T09:30:00+05:30",
                ]
            ),
            _frame(pack_name="baseline_v1").assign(
                signal_id=["active-1", "active-2", "active-3"],
                signal_timestamp=[
                    "2026-02-03T09:20:00+05:30",
                    "2026-02-03T09:25:00+05:30",
                    "2026-02-03T09:30:00+05:30",
                ],
            ),
        ],
        ignore_index=True,
    )

    report = _build(
        frame,
        runtime_activation_marker={
            "candidate_pack_name": "baseline_v1",
            "activated_at": "2026-02-03T09:20:00+05:30",
        },
    )

    assert report["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_HEALTHY
    assert report["runtime_activation_timestamp"] == "2026-02-03T03:50:00+00:00"
    assert report["post_adoption_traceability"]["candidate_pack_signal_count"] == 3
    assert report["post_adoption_traceability"]["non_candidate_pack_signal_count"] == 0


def test_signal_rollout_monitor_watches_before_post_adoption_signals_exist():
    report = _build(_frame(post=False, include_pack=False))

    assert report["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_WATCH
    assert report["post_adoption_traceability"]["traceability_status"] == "NO_POST_ADOPTION_SIGNALS_YET"


def test_signal_rollout_monitor_blocks_post_adoption_rows_without_pack_traceability():
    report = _build(_frame(include_pack=False))

    assert report["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_BLOCKED
    assert "parameter_pack_name is not recorded" in " ".join(report["rollout_reasons"])


def test_signal_rollout_monitor_blocks_order_side_effect_fields():
    report = _build(_frame(side_effect=True))

    assert report["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_BLOCKED
    assert report["execution_side_effects"]["orders_submitted"] is True
    assert "Order/execution side-effect" in " ".join(report["rollout_reasons"])


def test_signal_rollout_monitor_writer_outputs_latest_artifacts(tmp_path: Path):
    with temporary_parameter_pack("baseline_v1"):
        artifact = write_threshold_signal_rollout_monitor_report(
            _frame(),
            baseline_pack_name="baseline_v1",
            candidate_pack_name="baseline_v1",
            candidate_overrides={DEFAULT_CONFIG_HINT: 85.0},
            adoption_reconciliation_report=_adoption_report(),
            adoption_start_at="2026-02-01T00:00:00Z",
            output_dir=tmp_path,
            report_name="unit_threshold_signal_rollout_monitor",
        )

    for key in [
        "json_path",
        "markdown_path",
        "comparison_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_comparison_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["rollout_status"] == CANDIDATE_SIGNAL_ROLLOUT_HEALTHY
