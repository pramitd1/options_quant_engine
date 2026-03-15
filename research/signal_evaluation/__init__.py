"""
Signal evaluation and calibration dataset helpers.
"""

from research.signal_evaluation.dataset import (
    SIGNAL_DATASET_PATH,
    SIGNAL_DATASET_COLUMNS,
    ensure_signals_dataset_exists,
    load_signals_dataset,
    upsert_signal_rows,
    write_signals_dataset,
)
from research.signal_evaluation.evaluator import (
    build_signal_evaluation_row,
    build_regime_fingerprint,
    evaluate_signal_outcomes,
    save_signal_evaluation,
    update_signal_dataset_outcomes,
)
from research.signal_evaluation.market_data import fetch_realized_spot_path
from research.signal_evaluation.market_data import resolve_research_as_of
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
    "SIGNAL_DATASET_PATH",
    "SIGNAL_DATASET_COLUMNS",
    "ensure_signals_dataset_exists",
    "build_signal_evaluation_row",
    "build_regime_fingerprint",
    "build_research_report",
    "average_realized_return_by_horizon",
    "average_score_by_signal_quality",
    "evaluate_signal_outcomes",
    "fetch_realized_spot_path",
    "hit_rate_by_macro_regime",
    "hit_rate_by_trade_strength",
    "load_signals_dataset",
    "move_probability_calibration",
    "normalize_capture_policy",
    "regime_fingerprint_performance",
    "resolve_research_as_of",
    "save_signal_evaluation",
    "signal_count_by_regime",
    "SIGNAL_EVALUATION_REPORTS_DIR",
    "should_capture_signal",
    "update_signal_dataset_outcomes",
    "upsert_signal_rows",
    "write_signals_dataset",
    "build_signal_evaluation_summary",
    "render_signal_evaluation_markdown",
    "write_signal_evaluation_report",
]
