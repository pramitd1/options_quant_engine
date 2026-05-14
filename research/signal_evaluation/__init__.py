"""
Module: __init__.py

Purpose:
    Implement init utilities for signal evaluation, reporting, or research diagnostics.

Role in the System:
    Part of the research layer that records signal-evaluation datasets and diagnostic reports.

Key Outputs:
    Signal-evaluation datasets, reports, and comparison artifacts.

Downstream Usage:
    Consumed by tuning, governance reviews, and post-trade analysis.
"""

from research.signal_evaluation.dataset import (
    CUMULATIVE_DATASET_PATH,
    SIGNAL_DATASET_PATH,
    SIGNAL_DATASET_COLUMNS,
    ensure_signals_dataset_exists,
    load_cumulative_dataset,
    load_signals_dataset,
    sync_live_to_cumulative,
    upsert_signal_rows,
    write_signals_dataset,
)
from research.signal_evaluation.drift_monitor import (
    append_signal_drift_trend_history,
    build_signal_drift_trend_dashboard,
    build_signal_drift_trend_row,
    build_signal_drift_report,
    render_signal_drift_trend_markdown,
    render_signal_drift_markdown,
    write_signal_drift_trend_dashboard,
    write_signal_drift_report,
)
from research.signal_evaluation.drift_alerts import (
    append_signal_drift_review,
    build_signal_drift_alert_summary,
    build_signal_drift_review_row,
    classify_drift_ops_status,
    render_signal_drift_alert_markdown,
    run_signal_drift_alert_workflow,
    write_signal_drift_alert_summary,
)
from research.signal_evaluation.drift_notifications import (
    append_signal_drift_delivery_row,
    build_signal_drift_delivery_row,
    build_signal_drift_notification_payload,
    send_signal_drift_alert_notification,
    send_signal_drift_webhook,
    should_notify_signal_drift_alert,
)
from research.signal_evaluation.evaluator import (
    apply_label_quality_fields,
    build_signal_evaluation_row,
    build_regime_fingerprint,
    evaluate_signal_outcomes,
    save_signal_evaluation,
    update_signal_dataset_outcomes,
)
from research.signal_evaluation.label_quality import (
    apply_quality_label_view,
    label_quality_summary,
    quality_label_mask,
    select_quality_labeled_rows,
)
from research.signal_evaluation.market_data import fetch_realized_spot_path
from research.signal_evaluation.market_data import resolve_research_as_of
from research.signal_evaluation.legacy_backfill import (
    apply_repair_proposals_to_dataset,
    audit_unresolved_signal_contract_matches,
    backfill_signal_contract_fields,
    partition_repair_proposals,
    propose_repairs_for_unresolved_signal_contract_matches,
)
from research.signal_evaluation.policy import (
    CAPTURE_POLICY_ACTIONABLE,
    CAPTURE_POLICY_ALL,
    CAPTURE_POLICY_TRADE_ONLY,
    normalize_capture_policy,
    should_capture_signal,
)
from research.signal_evaluation.reporting import (
    SIGNAL_EVALUATION_REPORTS_DIR,
    build_signal_evaluation_summary,
    render_signal_evaluation_markdown,
    write_signal_evaluation_report,
)
from research.signal_evaluation.reports import (
    average_realized_return_by_horizon,
    average_score_by_signal_quality,
    build_research_report,
    hit_rate_by_macro_regime,
    hit_rate_by_trade_strength,
    move_probability_calibration,
    regime_fingerprint_performance,
    signal_count_by_regime,
)

__all__ = [
    "CAPTURE_POLICY_ACTIONABLE",
    "CAPTURE_POLICY_ALL",
    "CAPTURE_POLICY_TRADE_ONLY",
    "CUMULATIVE_DATASET_PATH",
    "SIGNAL_DATASET_PATH",
    "SIGNAL_DATASET_COLUMNS",
    "ensure_signals_dataset_exists",
    "apply_label_quality_fields",
    "apply_quality_label_view",
    "append_signal_drift_trend_history",
    "append_signal_drift_review",
    "append_signal_drift_delivery_row",
    "build_signal_drift_alert_summary",
    "build_signal_drift_delivery_row",
    "build_signal_drift_notification_payload",
    "build_signal_drift_report",
    "build_signal_drift_trend_dashboard",
    "build_signal_drift_trend_row",
    "build_signal_drift_review_row",
    "build_signal_evaluation_row",
    "build_regime_fingerprint",
    "build_research_report",
    "average_realized_return_by_horizon",
    "average_score_by_signal_quality",
    "evaluate_signal_outcomes",
    "fetch_realized_spot_path",
    "hit_rate_by_macro_regime",
    "hit_rate_by_trade_strength",
    "label_quality_summary",
    "load_cumulative_dataset",
    "load_signals_dataset",
    "move_probability_calibration",
    "normalize_capture_policy",
    "regime_fingerprint_performance",
    "resolve_research_as_of",
    "save_signal_evaluation",
    "quality_label_mask",
    "select_quality_labeled_rows",
    "signal_count_by_regime",
    "SIGNAL_EVALUATION_REPORTS_DIR",
    "should_capture_signal",
    "update_signal_dataset_outcomes",
    "sync_live_to_cumulative",
    "upsert_signal_rows",
    "write_signals_dataset",
    "build_signal_evaluation_summary",
    "render_signal_evaluation_markdown",
    "render_signal_drift_markdown",
    "render_signal_drift_alert_markdown",
    "render_signal_drift_trend_markdown",
    "run_signal_drift_alert_workflow",
    "send_signal_drift_alert_notification",
    "send_signal_drift_webhook",
    "should_notify_signal_drift_alert",
    "write_signal_evaluation_report",
    "write_signal_drift_alert_summary",
    "write_signal_drift_report",
    "write_signal_drift_trend_dashboard",
    "classify_drift_ops_status",
    "backfill_signal_contract_fields",
    "audit_unresolved_signal_contract_matches",
    "propose_repairs_for_unresolved_signal_contract_matches",
    "apply_repair_proposals_to_dataset",
    "partition_repair_proposals",
]
