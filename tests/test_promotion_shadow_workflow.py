from __future__ import annotations

import tuning.shadow as shadow_utils

from tuning.promotion import (
    evaluate_promotion,
    get_active_live_pack,
    get_active_shadow_pack,
    get_promotion_runtime_context,
    load_promotion_state,
    move_candidate_to_shadow,
    promote_candidate,
    record_manual_approval,
    rollback_live_pack,
    update_pack_state,
)
from tuning.reporting import load_promotion_ledger, summarize_promotion_workflow
from tuning.shadow import append_shadow_log, compare_shadow_trade_outputs, summarize_shadow_log


def _baseline_result():
    return {
        "parameter_pack_name": "baseline_v1",
        "sample_count": 60,
        "objective_score": 0.40,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.20, "drawdown_proxy": 45.0},
            "safeguards": {"minimum_sample_ok": True, "stability_gap": 0.03},
        },
        "validation_results": {
            "aggregate_out_of_sample_score": 0.18,
            "aggregate_out_of_sample_metrics": {"drawdown_proxy": 40.0},
        },
        "robustness_metrics": {"robustness_score": 0.62},
    }


def _candidate_result():
    return {
        "parameter_pack_name": "candidate_v1",
        "sample_count": 60,
        "objective_score": 0.47,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.18, "drawdown_proxy": 46.0},
            "safeguards": {"minimum_sample_ok": True, "stability_gap": 0.03},
        },
        "validation_results": {
            "aggregate_out_of_sample_score": 0.21,
            "aggregate_out_of_sample_metrics": {"drawdown_proxy": 42.0},
        },
        "robustness_metrics": {"robustness_score": 0.71},
        "comparison_summary": {
            "regime_comparison": {
                "gamma_regime_bucket": [
                    {"regime_label": "SHORT_GAMMA", "direction_hit_rate_delta": 0.03},
                    {"regime_label": "LONG_GAMMA", "direction_hit_rate_delta": -0.02},
                ]
            }
        },
    }


def test_promotion_state_model_supports_shadow_and_live(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    state = load_promotion_state(state_path)

    assert state["baseline"] == "baseline_v1"
    assert state["live"] == "baseline_v1"
    assert "shadow_assignment" in state
    assert get_active_live_pack(state_path) == "baseline_v1"
    assert get_active_shadow_pack(state_path) is None


def test_pack_state_transitions_and_manual_approval_are_auditable(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"

    update_pack_state(
        state_name="candidate",
        pack_name="experimental_v1",
        reason="candidate_created",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )
    move_candidate_to_shadow(
        "experimental_v1",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )
    record_manual_approval(
        pack_name="experimental_v1",
        approved=True,
        reviewer="pm",
        notes="approved for shadow",
        path=state_path,
        ledger_path=ledger_path,
    )

    context = get_promotion_runtime_context(state_path)
    assert context["candidate_pack"] == "experimental_v1"
    assert context["shadow_pack"] == "experimental_v1"
    assert context["manual_approvals"]["experimental_v1"]["approved"] is True

    ledger = load_promotion_ledger(ledger_path)
    assert len(ledger) == 3


def test_promotion_criteria_require_manual_approval_when_enabled():
    decision = evaluate_promotion(
        baseline_result=_baseline_result(),
        candidate_result=_candidate_result(),
        require_manual_approval=True,
        manual_approval={"approved": False},
    )
    assert decision.approved is False
    assert decision.reason == "manual_approval_required"


def test_promotion_blocks_when_important_regime_collapses_beyond_guardrail():
    candidate = _candidate_result()
    candidate["comparison_summary"] = {
        "regime_comparison": {
            "gamma_regime_bucket": [
                {
                    "regime_label": "SHORT_GAMMA",
                    "sample_count": 25,
                    "direction_hit_rate_delta": -0.16,
                },
                {
                    "regime_label": "LONG_GAMMA",
                    "sample_count": 4,
                    "direction_hit_rate_delta": -0.30,
                },
            ]
        }
    }

    decision = evaluate_promotion(
        baseline_result=_baseline_result(),
        candidate_result=candidate,
        important_regime_max_collapse=-0.08,
        minimum_important_regime_sample_count=10,
        maximum_important_regime_failures=0,
        important_regime_allowlist={"gamma_regime_bucket": ["SHORT_GAMMA"]},
    )

    assert decision.approved is False
    assert decision.reason == "candidate_regime_collapse_exceeds_limit"
    assert decision.diagnostics["important_regime_failures"] == 1


def test_promote_and_rollback_live_pack_are_safe_and_logged(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"

    update_pack_state(
        state_name="candidate",
        pack_name="experimental_v1",
        reason="candidate_created",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )
    record_manual_approval(
        pack_name="experimental_v1",
        approved=True,
        reviewer="pm",
        notes="approved for live",
        path=state_path,
        ledger_path=ledger_path,
    )
    promote_candidate(
        "experimental_v1",
        approved_by="pm",
        path=state_path,
        ledger_path=ledger_path,
    )
    state_after_promote = load_promotion_state(state_path)
    assert state_after_promote["live"] == "experimental_v1"
    assert state_after_promote["previous_live"] == "baseline_v1"

    rollback_live_pack(
        reviewer="pm",
        path=state_path,
        ledger_path=ledger_path,
    )
    state_after_rollback = load_promotion_state(state_path)
    assert state_after_rollback["live"] == "baseline_v1"

    ledger = load_promotion_ledger(ledger_path)
    assert "rollback_executed" in set(ledger["event_type"])


def test_shadow_mode_comparison_and_logging_are_side_by_side(tmp_path):
    baseline_payload = {
        "symbol": "NIFTY",
        "mode": "LIVE",
        "source": "NSE",
        "spot_summary": {"timestamp": "2026-03-14T09:20:00+05:30"},
        "trade": {
            "direction": "CALL",
            "trade_status": "TRADE",
            "trade_strength": 68,
            "overnight_hold_allowed": True,
            "signal_regime": "DIRECTIONAL_BIAS",
        },
    }
    shadow_payload = {
        "symbol": "NIFTY",
        "mode": "LIVE",
        "source": "NSE",
        "spot_summary": {"timestamp": "2026-03-14T09:20:00+05:30"},
        "trade": {
            "direction": "PUT",
            "trade_status": "WATCHLIST",
            "trade_strength": 61,
            "overnight_hold_allowed": False,
            "signal_regime": "CONFLICTED",
        },
    }

    comparison = compare_shadow_trade_outputs(
        baseline_payload,
        shadow_payload,
        baseline_pack_name="baseline_v1",
        shadow_pack_name="candidate_v1",
    )
    assert comparison["decision_disagreement_flag"] is True
    assert comparison["trade_status_disagreement_flag"] is True
    assert comparison["delta_trade_strength"] == -7.0

    shadow_log_path = tmp_path / "shadow_mode_log.jsonl"
    append_shadow_log(comparison, shadow_log_path)
    summary = summarize_shadow_log(shadow_log_path)
    assert summary["shadow_event_count"] == 1
    assert summary["decision_disagreement_rate"] == 1.0
    assert comparison["session_date"] == "2026-03-14"
    assert comparison["validation_status"] == "DIVERGED"
    assert summary["latest_session_validation"]["session_date"] == "2026-03-14"


def test_shadow_summary_includes_session_by_session_rollup(tmp_path):
    shadow_log_path = tmp_path / "shadow_mode_log.jsonl"
    append_shadow_log(
        {
            "evaluation_timestamp": "2026-03-14T09:20:00+05:30",
            "baseline_pack_name": "baseline_v1",
            "shadow_pack_name": "macro_overlay_v1",
            "decision_disagreement_flag": False,
            "trade_status_disagreement_flag": True,
            "signal_presence_disagreement_flag": False,
            "delta_trade_strength": 2.0,
        },
        shadow_log_path,
    )
    append_shadow_log(
        {
            "evaluation_timestamp": "2026-03-14T13:45:00+05:30",
            "baseline_pack_name": "baseline_v1",
            "shadow_pack_name": "macro_overlay_v1",
            "decision_disagreement_flag": True,
            "trade_status_disagreement_flag": True,
            "signal_presence_disagreement_flag": True,
            "delta_trade_strength": -1.0,
        },
        shadow_log_path,
    )

    summary = summarize_shadow_log(shadow_log_path)

    assert summary["shadow_event_count"] == 2
    assert summary["session_validation_summary"]
    latest = summary["latest_session_validation"]
    assert latest["session_date"] == "2026-03-14"
    assert latest["snapshot_count"] == 2
    assert latest["decision_disagreement_rate"] == 0.5
    assert summary["dominant_disagreement_drivers"][0]["driver"] == "trade_status_disagreement"
    assert "OPEN" in {row["session_bucket"] for row in summary["session_bucket_summary"]}


def test_shadow_summary_flags_policy_limit_breaches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shadow_utils,
        "_resolve_shadow_alert_policy",
        lambda: {
            "decision_disagreement_alert": 0.20,
            "trade_status_disagreement_alert": 0.25,
            "signal_presence_disagreement_alert": 0.15,
            "overnight_disagreement_alert": 0.20,
            "session_alert_min_snapshots": 1,
        },
    )

    shadow_log_path = tmp_path / "shadow_mode_log.jsonl"
    append_shadow_log(
        {
            "evaluation_timestamp": "2026-03-14T09:20:00+05:30",
            "baseline_pack_name": "baseline_v1",
            "shadow_pack_name": "macro_overlay_v1",
            "decision_disagreement_flag": True,
            "trade_status_disagreement_flag": True,
            "signal_presence_disagreement_flag": False,
            "overnight_disagreement_flag": False,
            "delta_trade_strength": -5.0,
        },
        shadow_log_path,
    )

    summary = summarize_shadow_log(shadow_log_path)
    latest = summary["latest_session_validation"]

    assert latest["policy_alert"] is True
    assert latest["alert_level"] == "ALERT"
    assert "decision_disagreement_rate" in latest["breached_limits"]
    assert summary["policy_alert_count"] == 1


def test_reporting_surfaces_current_live_shadow_and_events(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"
    shadow_log_path = tmp_path / "shadow_mode_log.jsonl"

    move_candidate_to_shadow(
        "experimental_v1",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )
    append_shadow_log(
        {
            "evaluation_timestamp": "2026-03-14T09:20:00+05:30",
            "baseline_pack_name": "baseline_v1",
            "shadow_pack_name": "experimental_v1",
            "decision_disagreement_flag": False,
            "trade_status_disagreement_flag": True,
            "delta_trade_strength": 2.0,
        },
        shadow_log_path,
    )
    summary = summarize_promotion_workflow(
        state_path=state_path,
        ledger_path=ledger_path,
        shadow_log_path=shadow_log_path,
    )

    assert summary["current_state"]["shadow_pack"] == "experimental_v1"
    assert summary["promotion_event_count"] >= 1
    assert summary["shadow_summary"]["shadow_event_count"] == 1
