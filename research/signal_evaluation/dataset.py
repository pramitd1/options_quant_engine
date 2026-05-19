"""
Module: dataset.py

Purpose:
    Implement dataset utilities for signal evaluation, reporting, or research diagnostics.

Role in the System:
    Part of the research layer that records signal-evaluation datasets and diagnostic reports.

Key Outputs:
    Signal-evaluation datasets, reports, and comparison artifacts.

Downstream Usage:
    Consumed by tuning, governance reviews, and post-trade analysis.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3
import threading

import pandas as pd

from config.settings import BASE_DIR

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


SIGNAL_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "signals_dataset.csv"
CUMULATIVE_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"

SIGNAL_DATASET_COLUMNS = [
    "signal_id",
    "signal_timestamp",
    "source",
    "requested_option_source",
    "option_source",
    "spot_source",
    "market_data_source_consistency",
    "market_data_provenance_status",
    "market_data_trade_blocking_status",
    "market_data_timestamp_status",
    "market_data_timestamp_delta_seconds",
    "market_data_provenance_reasons",
    "market_data_provenance_warnings",
    "market_data_provenance_issues",
    "mode",
    "parameter_pack_name",
    "signal_capture_guarded",
    "signal_capture_guard_reason",
    "runtime_activation_guard_active",
    "runtime_activation_capture_allowed",
    "runtime_activation_guard_status",
    "runtime_activation_expected_parameter_pack",
    "runtime_activation_observed_parameter_pack",
    "runtime_activation_activated_at",
    "runtime_activation_signal_timestamp",
    "runtime_activation_marker_generated_at",
    "symbol",
    "ticker",
    "selected_expiry",
    "direction",
    "option_type",
    "strike",
    "entry_price",
    "option_entry_premium",
    "option_target_premium",
    "option_stop_loss_premium",
    "underlying_profit_booking_level",
    "underlying_profit_booking_lower",
    "underlying_profit_booking_upper",
    "underlying_stop_loss_level",
    "underlying_stop_loss_lower",
    "underlying_stop_loss_upper",
    "underlying_exit_plan_confidence",
    "underlying_exit_plan_basis",
    "underlying_exit_plan_reasons",
    "underlying_exit_plan_json",
    "option_premium_pct_of_spot",
    "target_premium_return_pct",
    "stop_loss_premium_return_pct",
    "selected_option_last_price",
    "selected_option_bid_price",
    "selected_option_ask_price",
    "selected_option_mid_price",
    "selected_option_volume",
    "selected_option_open_interest",
    "selected_option_iv",
    "selected_option_iv_is_proxy",
    "selected_option_iv_proxy_source",
    "selected_option_delta",
    "selected_option_delta_is_proxy",
    "selected_option_delta_proxy_source",
    "selected_option_gamma",
    "selected_option_theta",
    "selected_option_vega",
    "selected_option_vanna",
    "selected_option_charm",
    "heston_research_enabled",
    "heston_calibration_status",
    "heston_calibration_reason",
    "heston_calibration_sample_size",
    "heston_kappa",
    "heston_theta",
    "heston_vol_of_vol",
    "heston_rho",
    "heston_v0",
    "heston_calibration_error",
    "heston_surface_quality",
    "heston_quality_flags",
    "heston_bound_hit_count",
    "heston_tte_days",
    "heston_tte_bucket",
    "heston_expiry_context",
    "heston_short_tte_guard",
    "heston_selected_iv_quality",
    "heston_skew_state",
    "heston_forward_variance_proxy",
    "heston_model_price",
    "heston_model_delta",
    "heston_model_gamma",
    "heston_model_iv_proxy",
    "bs_model_price_for_heston",
    "bs_vs_heston_price_gap",
    "heston_price_gap_rel_pct",
    "bs_vs_heston_greek_gap",
    "greek_model_divergence_score",
    "heston_diagnostics_json",
    "selected_option_capital_per_lot",
    "selected_option_ba_spread_ratio",
    "selected_option_ba_spread_pct",
    "selected_option_score",
    "target",
    "stop_loss",
    "recommended_hold_minutes",
    "max_hold_minutes",
    "exit_urgency",
    "spot_at_signal",
    "day_open",
    "day_high",
    "day_low",
    "prev_close",
    "lookback_avg_range_pct",
    "trade_strength",
    "runtime_composite_score",
    "runtime_composite_observation_tier",
    "runtime_composite_observation_threshold",
    "runtime_composite_soft_override_threshold",
    "runtime_composite_soft_override_applied",
    "runtime_composite_soft_override_mode",
    "runtime_composite_soft_override_blockers",
    "runtime_composite_soft_override_reason",
    "runtime_composite_soft_override_constraints",
    "runtime_composite_soft_override_original_status",
    "runtime_composite_soft_override_original_reason_code",
    "runtime_composite_soft_override_original_message",
    "runtime_composite_soft_override_diagnostics",
    "effective_min_trade_strength_threshold",
    "effective_min_composite_score_threshold",
    "signal_quality",
    "signal_regime",
    "execution_regime",
    "regime_fingerprint",
    "regime_fingerprint_id",
    "trade_status",
    "direction_source",
    "final_flow_signal",
    "gamma_regime",
    "spot_vs_flip",
    "macro_regime",
    "event_intelligence_enabled",
    "event_bullish_score",
    "event_bearish_score",
    "event_vol_expansion_score",
    "event_vol_compression_score",
    "event_uncertainty_score",
    "event_gap_risk_score",
    "event_catalyst_alignment_score",
    "event_contradictory_penalty",
    "event_cluster_score",
    "event_decayed_signal",
    "event_relevance_score",
    "event_count",
    "event_routed_count",
    "event_overlay_probability_multiplier",
    "event_overlay_size_multiplier",
    "event_overlay_score_adjustment",
    "event_overlay_suppress_signal",
    "event_overlay_reasons",
    "event_explanations",
    "global_risk_state",
    "global_risk_score",
    "oil_shock_score",
    "commodity_risk_score",
    "volatility_shock_score",
    "volatility_explosion_probability",
    "overnight_gap_risk_score",
    "volatility_expansion_risk_score",
    "overnight_hold_allowed",
    "overnight_hold_reason",
    "overnight_risk_penalty",
    "global_risk_adjustment_score",
    "gamma_vol_acceleration_score",
    "squeeze_risk_state",
    "directional_convexity_state",
    "upside_squeeze_risk",
    "downside_airpocket_risk",
    "overnight_convexity_risk",
    "gamma_vol_adjustment_score",
    "dealer_hedging_pressure_score",
    "dealer_flow_state",
    "upside_hedging_pressure",
    "downside_hedging_pressure",
    "pinning_pressure_score",
    "dealer_pressure_adjustment_score",
    "expected_move_points",
    "expected_move_pct",
    "open_interest_pcr",
    "volume_pcr",
    "volume_pcr_atm",
    "volume_pcr_regime",
    "pcr_value",
    "pcr_basis",
    "pcr_bucket",
    "pcr_data_source",
    "pcr_snapshot_age_seconds",
    "target_reachability_score",
    "premium_efficiency_score",
    "strike_efficiency_score",
    "option_efficiency_score",
    "option_efficiency_adjustment_score",
    "consistency_check_status",
    "consistency_check_issue_count",
    "consistency_check_critical_issue_count",
    "consistency_check_escalated",
    "consistency_check_findings",
    "dealer_position",
    "dealer_hedging_bias",
    "dealer_hedging_flow",
    "market_delta_exposure",
    "market_gamma_exposure",
    "market_theta_exposure",
    "market_vega_exposure",
    "market_vanna_exposure",
    "market_charm_exposure",
    "volatility_regime",
    "liquidity_vacuum_state",
    "historical_context_version",
    "historical_context_mode",
    "historical_prior_artifact_version",
    "historical_prior_artifact_source_run_id",
    "historical_context_applied",
    "historical_volatility_bucket",
    "historical_expected_range_bps",
    "historical_expected_abs_move_bps",
    "historical_range_multiplier",
    "historical_global_prior_direction",
    "historical_global_prior_score",
    "historical_global_prior_evidence",
    "historical_pcr_state",
    "historical_pcr_interpretation",
    "historical_max_pain_state",
    "historical_max_pain_interpretation",
    "historical_wall_state",
    "historical_wall_interpretation",
    "historical_context_score_adjustment_preview",
    "historical_context_probability_adjustment_preview",
    "historical_context_score_adjustment",
    "historical_context_probability_adjustment",
    "historical_context_trade_strength_threshold_adjustment",
    "historical_context_composite_threshold_adjustment",
    "historical_context_size_multiplier",
    "historical_context_size_applied",
    "historical_context_direction_override",
    "historical_context_reasons",
    "historical_interaction_count",
    "historical_interaction_score_adjustment",
    "historical_interaction_probability_adjustment",
    "historical_interaction_reasons",
    "historical_interaction_bucket_state",
    "statistical_market_context_version",
    "statistical_market_context_artifact_version",
    "statistical_market_context_source_run_id",
    "statistical_market_context_applied",
    "statistical_vol_stress_score",
    "statistical_expected_range_prior",
    "statistical_expected_range_bps",
    "statistical_expected_abs_move_bps",
    "statistical_abs_move_delta_vs_base_bps",
    "statistical_directional_followthrough_prior",
    "statistical_directional_basis",
    "statistical_regime_confidence",
    "statistical_hold_time_hint",
    "statistical_context_score_adjustment",
    "statistical_context_probability_adjustment",
    "statistical_context_trade_strength_threshold_adjustment",
    "statistical_context_composite_threshold_adjustment",
    "statistical_context_size_multiplier",
    "statistical_context_reasons",
    "statistical_context_bucket_state",
    "statistical_macro_context_applied",
    "statistical_macro_range_prior",
    "statistical_macro_range_basis",
    "statistical_macro_directional_prior",
    "statistical_macro_directional_basis",
    "statistical_macro_factor_buckets",
    "statistical_macro_shock_state",
    "statistical_macro_score_adjustment",
    "statistical_macro_probability_adjustment",
    "statistical_macro_trade_strength_threshold_adjustment",
    "statistical_macro_composite_threshold_adjustment",
    "statistical_macro_size_multiplier",
    "statistical_macro_reasons",
    "historical_context_notes",
    "historical_context_json",
    "confirmation_status",
    "macro_event_risk_score",
    "data_quality_score",
    "data_quality_status",
    "option_chain_validation_status",
    "option_chain_is_valid",
    "option_chain_is_stale",
    "option_chain_issue_count",
    "option_chain_warning_count",
    "analytics_usable",
    "execution_suggestion_usable",
    "tradable_data_status",
    "tradable_data_score",
    "provider_health_status",
    "provider_health_row",
    "provider_health_pricing",
    "provider_health_pairing",
    "provider_health_iv",
    "provider_health_duplicate",
    "move_probability",
    "rule_move_probability",
    "hybrid_move_probability",
    "ml_move_probability",
    "large_move_probability",
    "macro_news_volatility_shock_score",
    "gamma_flip_distance_pct",
    "vacuum_strength",
    "hedging_flow_ratio",
    "smart_money_flow_score",
    "atm_iv_percentile",
    "vanna_regime",
    "charm_regime",
    "india_vix_level",
    "india_vix_change_24h",
    "atm_iv_scaled",
    "weekday",
    "saved_spot_snapshot_path",
    "saved_chain_snapshot_path",
    "option_premium_path_status",
    "option_premium_path_snapshot_count",
    "option_premium_path_max_lag_seconds",
    "option_premium_path_reasons",
    "option_premium_path_last_updated_at",
    "option_premium_5m",
    "option_premium_15m",
    "option_premium_30m",
    "option_premium_60m",
    "option_premium_120m",
    "option_premium_return_5m_pct",
    "option_premium_return_15m_pct",
    "option_premium_return_30m_pct",
    "option_premium_return_60m_pct",
    "option_premium_return_120m_pct",
    "option_premium_return_5m_bps",
    "option_premium_return_15m_bps",
    "option_premium_return_30m_bps",
    "option_premium_return_60m_bps",
    "option_premium_return_120m_bps",
    "option_premium_pnl_per_lot_5m",
    "option_premium_pnl_per_lot_15m",
    "option_premium_pnl_per_lot_30m",
    "option_premium_pnl_per_lot_60m",
    "option_premium_pnl_per_lot_120m",
    "created_at",
    "updated_at",
    "outcome_last_updated_at",
    "outcome_status",
    "label_quality_status",
    "label_quality_score",
    "label_quality_reasons",
    "calibration_label",
    "calibration_label_horizon",
    "calibration_label_available",
    "primary_outcome_horizon",
    "primary_outcome_return_bps",
    "observed_minutes",
    "evaluation_window_minutes",
    "spot_5m",
    "spot_15m",
    "spot_30m",
    "spot_60m",
    "spot_close_same_day",
    "spot_next_open",
    "spot_next_close",
    "spot_120m",
    "spot_session_close",
    "spot_1d",
    "spot_2d",
    "spot_3d",
    "spot_5d",
    "spot_at_expiry",
    "realized_return_5m",
    "realized_return_15m",
    "realized_return_30m",
    "realized_return_60m",
    "realized_return_120m",
    "signed_return_5m_bps",
    "signed_return_15m_bps",
    "signed_return_30m_bps",
    "signed_return_60m_bps",
    "signed_return_120m_bps",
    "signed_return_session_close_bps",
    "return_1d_bps",
    "return_2d_bps",
    "return_3d_bps",
    "return_5d_bps",
    "return_at_expiry_bps",
    "mfe_points",
    "mae_points",
    "correct_5m",
    "correct_15m",
    "correct_30m",
    "correct_60m",
    "correct_120m",
    "correct_session_close",
    "correct_1d",
    "correct_2d",
    "correct_3d",
    "correct_5d",
    "correct_at_expiry",
    "mfe_60m_bps",
    "mae_60m_bps",
    "mfe_120m_bps",
    "mae_120m_bps",
    "realized_range_60m_bps",
    "realized_range_120m_bps",
    "eod_mfe_bps",
    "eod_mae_bps",
    "target_hit",
    "target_hit_date",
    "stop_loss_hit",
    "stop_loss_hit_date",
    "target_stop_same_bar_ambiguous",
    "target_sl_delta_used",
    "target_sl_delta_source",
    "direction_score",
    "magnitude_score",
    "timing_score",
    "tradeability_score",
    "tradeability_tier",
    "best_outcome_horizon",
    "best_outcome_bps",
    "peak_to_close_decay_bps",
    "exit_efficiency_score",
    "horizon_edge_label",
    "exit_quality_label",
    "composite_signal_score",
    "directional_consistency_score",
    "signal_confidence_score",
    "signal_confidence_level",
    "signal_confidence_calibration_status",
    "signal_confidence_calibration_sample_size",
    "signal_confidence_calibration_regime_match",
    "signal_confidence_calibration_guardrail_status",
    "signal_confidence_recalibration_guards",
    "signal_calibration_bucket",
    "probability_calibration_bucket",
    "ml_rank_score",
    "ml_confidence_score",
    "ml_rank_bucket",
    "ml_confidence_bucket",
    "ml_agreement_with_engine",
    "notes",
]

# Features that are inherently event-driven and may exhibit zero variance
# during calm periods.  Downstream ML and analysis should check variance
# before including these as predictors to avoid degenerate splits.
EVENT_DRIVEN_FEATURES = [
    "volatility_shock_score",
    "macro_event_risk_score",
]

_SIGNAL_ID_CACHE: dict[Path, dict[str, object]] = {}
_DATASET_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_DATASET_THREAD_LOCKS_GUARD = threading.Lock()
_SQLITE_BUSY_TIMEOUT_MS = 30_000


def _ensure_parent_dir(path: Path) -> None:
    """
    Purpose:
        Ensure the parent directory exists before writing artifacts.

    Context:
        Function inside the `dataset` module. The module sits in the research layer that evaluates signals, curates datasets, and renders reports.

    Inputs:
        path (Path): Filesystem path used by the workflow.

    Returns:
        None: Filesystem side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _canonical_path(path: Path) -> Path:
    """Return a stable absolute path for lock/cache keys."""
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path).expanduser().absolute()


def _dataset_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _thread_lock_for_path(path: Path) -> threading.RLock:
    key = _canonical_path(path)
    with _DATASET_THREAD_LOCKS_GUARD:
        lock = _DATASET_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DATASET_THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _dataset_write_lock(path: Path):
    """Serialize CSV/SQLite dataset mutations across threads and processes."""
    dataset_path = Path(path)
    _ensure_parent_dir(dataset_path)
    lock_path = _dataset_lock_path(dataset_path)
    thread_lock = _thread_lock_for_path(lock_path)
    with thread_lock:
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    _ensure_parent_dir(path)
    connection = sqlite3.connect(path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _ensure_parent_dir(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _dataset_store_path(path: Path) -> Path:
    """
    Purpose:
        Process dataset store path for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return path.with_suffix(".sqlite")


def _read_sqlite_dataset(path: Path) -> pd.DataFrame:
    """
    Purpose:
        Load the canonical signal-evaluation dataset from SQLite storage.

    Context:
        Function inside the `dataset` module. The module sits in the research layer that evaluates signals, curates datasets, and renders reports.

    Inputs:
        path (Path): Filesystem path used by the workflow.

    Returns:
        pd.DataFrame: Normalized dataset frame loaded from SQLite.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    if not path.exists():
        return _empty_dataset_frame()
    with closing(_connect_sqlite(path)) as connection:
        frame = pd.read_sql_query("SELECT * FROM signals", connection)
    return _normalize_dataset_frame(frame)


def _write_sqlite_dataset(frame: pd.DataFrame, path: Path) -> None:
    """
    Purpose:
        Write sqlite dataset to the appropriate output artifact.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        path (Path): Input associated with path.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    _ensure_parent_dir(path)
    with closing(_connect_sqlite(path)) as connection:
        normalized = _normalize_dataset_frame(frame)
        with connection:
            normalized.to_sql("signals", connection, if_exists="replace", index=False)
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)")


def _append_sqlite_rows(frame: pd.DataFrame, path: Path) -> None:
    """
    Purpose:
        Process append sqlite rows for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        path (Path): Input associated with path.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return
    _ensure_parent_dir(path)
    with closing(_connect_sqlite(path)) as connection:
        with connection:
            if not _sqlite_has_table(connection, "signals"):
                normalized = _normalize_dataset_frame(frame)
                normalized.to_sql("signals", connection, if_exists="replace", index=False)
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)")
                return
            _sqlite_ensure_columns(connection, "signals", SIGNAL_DATASET_COLUMNS)
            _normalize_dataset_frame(frame).to_sql("signals", connection, if_exists="append", index=False)


def _sqlite_has_table(connection: sqlite3.Connection, table_name: str) -> bool:
    """
    Purpose:
        Process sqlite has table for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        connection (sqlite3.Connection): Input associated with connection.
        table_name (str): Human-readable name for table.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
    return connection.execute(query, (table_name,)).fetchone() is not None


def _sqlite_table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    """Return the ordered column names for an existing SQLite table."""
    rows = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [str(row[1]) for row in rows]


def _sqlite_ensure_columns(connection: sqlite3.Connection, table_name: str, expected_columns: list[str]) -> None:
    """Add missing columns to an existing SQLite table for forward-compat appends.

    SQLite uses dynamic typing, so TEXT affinity is acceptable for schema
    evolution; pandas inserts for numeric fields continue to work.
    """
    existing = set(_sqlite_table_columns(connection, table_name))
    for column in expected_columns:
        if column in existing:
            continue
        safe_col = column.replace('"', '""')
        connection.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_col}" TEXT')


def _empty_dataset_frame() -> pd.DataFrame:
    """
    Purpose:
        Process empty dataset frame for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return pd.DataFrame(columns=SIGNAL_DATASET_COLUMNS)


def _normalize_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Normalize dataset frame into the repository-standard form.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame is None:
        return _empty_dataset_frame()

    # Reindex once to preserve canonical column order without repeatedly
    # inserting missing columns, which fragments the frame internals.
    normalized = frame.copy().reindex(columns=SIGNAL_DATASET_COLUMNS, fill_value=pd.NA)
    return normalized


def _current_file_signature(path: Path) -> tuple[int, int] | None:
    """
    Purpose:
        Process current file signature for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
    
    Returns:
        tuple[int, int] | None: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _update_signal_id_cache(path: Path, signal_ids: Iterable[object]) -> None:
    """
    Purpose:
        Update signal identifier cache using the supplied inputs.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
        signal_ids (Iterable[object]): Input associated with signal ids.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    _SIGNAL_ID_CACHE[path] = {
        "signature": _current_file_signature(path),
        "signal_ids": {str(signal_id) for signal_id in signal_ids if pd.notna(signal_id)},
    }


def _load_existing_signal_ids(path: Path) -> set[str]:
    """
    Purpose:
        Process load existing signal ids for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
    
    Returns:
        set[str]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not path.exists():
        sqlite_path = _dataset_store_path(path)
        if not sqlite_path.exists():
            return set()

    signature = _current_file_signature(path)
    cached = _SIGNAL_ID_CACHE.get(path)
    if cached and cached.get("signature") == signature:
        return set(cached.get("signal_ids", set()))

    sqlite_path = _dataset_store_path(path)
    if sqlite_path.exists():
        try:
            with closing(_connect_sqlite(sqlite_path)) as connection:
                frame = pd.read_sql_query("SELECT signal_id FROM signals", connection)
        except Exception:
            frame = load_signals_dataset(path)[["signal_id"]]
    else:
        try:
            frame = pd.read_csv(path, usecols=["signal_id"])
        except ValueError:
            frame = load_signals_dataset(path)[["signal_id"]]

    signal_ids = {str(signal_id) for signal_id in frame["signal_id"].dropna()}
    _SIGNAL_ID_CACHE[path] = {
        "signature": signature,
        "signal_ids": signal_ids,
    }
    return set(signal_ids)


def _dataset_has_canonical_header(path: Path) -> bool:
    """
    Purpose:
        Process dataset has canonical header for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not path.exists():
        return False
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    return header == SIGNAL_DATASET_COLUMNS


def _dedupe_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Remove duplicate entries from signal frame.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return frame.copy()

    deduped = frame.copy()
    sort_key = "__sort_key__"
    deduped[sort_key] = deduped["updated_at"].fillna(deduped["created_at"]).fillna("")
    deduped = deduped.sort_values(sort_key, kind="stable")
    deduped = deduped.drop_duplicates(subset=["signal_id"], keep="last")
    deduped = deduped.drop(columns=[sort_key]).reset_index(drop=True)
    return deduped


def _append_rows_to_dataset(frame: pd.DataFrame, path: Path, existing_signal_ids: set[str] | None = None) -> None:
    """
    Purpose:
        Process append rows to dataset for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        path (Path): Input associated with path.
        existing_signal_ids (set[str] | None): Input associated with existing signal ids.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return

    with _dataset_write_lock(path):
        existing = _load_signals_dataset_unlocked(path)
        combined = (
            _normalize_dataset_frame(frame)
            if existing.empty
            else _normalize_dataset_frame(
                pd.concat(
                    [
                        existing.dropna(axis=1, how="all"),
                        frame.dropna(axis=1, how="all"),
                    ],
                    ignore_index=True,
                    sort=False,
                )
            )
        )
        combined = _dedupe_signal_frame(combined)
        _write_signals_dataset_unlocked(combined, path)


def _load_signals_dataset_unlocked(dataset_path: Path) -> pd.DataFrame:
    sqlite_path = _dataset_store_path(dataset_path)
    if sqlite_path.exists():
        try:
            return _read_sqlite_dataset(sqlite_path)
        except Exception:
            pass

    if dataset_path.exists():
        frame = pd.read_csv(dataset_path)
        return _normalize_dataset_frame(frame)

    return _empty_dataset_frame()


def _write_signals_dataset_unlocked(frame: pd.DataFrame, dataset_path: Path) -> Path:
    _ensure_parent_dir(dataset_path)
    normalized = _normalize_dataset_frame(frame)
    _write_sqlite_dataset(normalized, _dataset_store_path(dataset_path))
    _atomic_write_csv(normalized, dataset_path)
    _update_signal_id_cache(dataset_path, normalized.get("signal_id", pd.Series(dtype="object")))
    return dataset_path


def _ensure_signals_dataset_exists_unlocked(dataset_path: Path) -> Path:
    sqlite_path = _dataset_store_path(dataset_path)
    if not dataset_path.exists() and not sqlite_path.exists():
        _write_signals_dataset_unlocked(_empty_dataset_frame(), dataset_path)
    elif sqlite_path.exists() and not dataset_path.exists():
        _atomic_write_csv(_read_sqlite_dataset(sqlite_path), dataset_path)
    elif dataset_path.exists() and not sqlite_path.exists():
        _write_sqlite_dataset(_normalize_dataset_frame(pd.read_csv(dataset_path)), sqlite_path)
    return dataset_path


def load_signals_dataset(path: str | Path = SIGNAL_DATASET_PATH) -> pd.DataFrame:
    """
    Purpose:
        Process load signals dataset for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    dataset_path = Path(path)
    with _dataset_write_lock(dataset_path):
        _ensure_signals_dataset_exists_unlocked(dataset_path)
        return _load_signals_dataset_unlocked(dataset_path)


def write_signals_dataset(frame: pd.DataFrame, path: str | Path = SIGNAL_DATASET_PATH) -> Path:
    """
    Purpose:
        Write signals dataset to the appropriate output artifact.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        path (str | Path): Input associated with path.
    
    Returns:
        None: The function communicates through side effects such as terminal output or persisted artifacts.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    dataset_path = Path(path)
    with _dataset_write_lock(dataset_path):
        return _write_signals_dataset_unlocked(frame, dataset_path)


def ensure_signals_dataset_exists(path: str | Path = SIGNAL_DATASET_PATH) -> Path:
    """
    Purpose:
        Ensure signals dataset exists exists and is ready for use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    dataset_path = Path(path)
    with _dataset_write_lock(dataset_path):
        return _ensure_signals_dataset_exists_unlocked(dataset_path)


def upsert_signal_rows(
    rows,
    path: str | Path = SIGNAL_DATASET_PATH,
    *,
    return_frame: bool = True,
) -> pd.DataFrame | None:
    """
    Purpose:
        Process upsert signal rows for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        rows (Any): Input associated with rows.
        path (str | Path): Input associated with path.
        return_frame (bool): Boolean flag associated with return_frame.
    
    Returns:
        pd.DataFrame | None: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    dataset_path = Path(path)
    incoming = pd.DataFrame(rows or [])
    sync_to_cumulative = False
    cumulative_rows = None

    with _dataset_write_lock(dataset_path):
        _ensure_signals_dataset_exists_unlocked(dataset_path)

        if incoming.empty:
            return _load_signals_dataset_unlocked(dataset_path) if return_frame else None

        incoming = _normalize_dataset_frame(incoming)
        if incoming["signal_id"].isna().any():
            raise ValueError("All signal rows must include a non-null signal_id")
        incoming = _dedupe_signal_frame(incoming)

        existing = _load_signals_dataset_unlocked(dataset_path)
        if existing.empty:
            combined = incoming.copy()
        else:
            combined = pd.concat(
                [
                    existing.dropna(axis=1, how="all"),
                    incoming.dropna(axis=1, how="all"),
                ],
                ignore_index=True,
                sort=False,
            )
            combined = _normalize_dataset_frame(combined)

        combined = _dedupe_signal_frame(combined)
        _write_signals_dataset_unlocked(combined, dataset_path)
        result = combined if return_frame else None

        sync_to_cumulative = Path(path) == SIGNAL_DATASET_PATH
        cumulative_rows = incoming.copy() if sync_to_cumulative else None

    # Auto-sync outside the live dataset lock to avoid lock-order coupling.
    if sync_to_cumulative and cumulative_rows is not None:
        _sync_to_cumulative(cumulative_rows)

    return result


# ---------------------------------------------------------------------------
# Cumulative (archival) dataset helpers
# ---------------------------------------------------------------------------

def load_cumulative_dataset(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the cumulative signals dataset (CSV + SQLite fallback)."""
    cumul_path = Path(path) if path else CUMULATIVE_DATASET_PATH
    return load_signals_dataset(cumul_path)


def _apply_ml_inference_to_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ML inference to populate ml_rank_score and ml_confidence_score for signals that need them.
    
    This ensures that every signal generates has ML scores, fixing the infrastructure gap where
    live signals were captured without running through the research evaluation layer.
    
    Returns the dataframe with ML scores populated (in-place modification).
    """
    try:
        from research.ml_models.ml_config import ML_RESEARCH_ENABLED
        if not ML_RESEARCH_ENABLED:
            return df
        
        # Check which signals need ML inference
        needs_inference = df["ml_rank_score"].isna() | df["ml_confidence_score"].isna()
        if not needs_inference.any():
            return df
        
        # Apply inference batch to those signals
        from research.ml_models.ml_inference import infer_batch
        try:
            inferred = infer_batch(df[needs_inference])
            if inferred is not None:
                df.loc[needs_inference, "ml_rank_score"] = inferred["ml_rank_score"]
                df.loc[needs_inference, "ml_confidence_score"] = inferred["ml_confidence_score"]
                df.loc[needs_inference, "ml_rank_bucket"] = inferred["ml_rank_bucket"]
                df.loc[needs_inference, "ml_confidence_bucket"] = inferred["ml_confidence_bucket"]
                df.loc[needs_inference, "ml_agreement_with_engine"] = inferred["ml_agreement_with_engine"]
        except Exception:
            # If batch inference fails, don't break signal capture
            pass
    except Exception:
        # If ML inference infrastructure is broken, don't break signal capture
        pass
    
    return df


def _sync_to_cumulative(incoming: pd.DataFrame) -> None:
    """Append *incoming* rows to the cumulative dataset, deduplicating by signal_id.

    This is called automatically by ``upsert_signal_rows`` whenever the live
    dataset is updated so that every captured signal is also persisted in the
    long-lived cumulative store.
    
    ML inference is applied before syncing to ensure all signals have scores.
    """
    cumul_csv = CUMULATIVE_DATASET_PATH

    if incoming.empty:
        return

    incoming = _normalize_dataset_frame(incoming)
    
    # ── NEW: Apply ML inference to signals before syncing ──
    incoming = _apply_ml_inference_to_signals(incoming)

    with _dataset_write_lock(cumul_csv):
        existing = _load_signals_dataset_unlocked(cumul_csv)
        existing_ids = (
            {str(signal_id) for signal_id in existing["signal_id"].dropna()}
            if not existing.empty and "signal_id" in existing.columns
            else set()
        )

        new_rows = incoming[~incoming["signal_id"].astype(str).isin(existing_ids)] if existing_ids else incoming
        if new_rows.empty:
            return

        if existing.empty:
            combined = new_rows.copy()
        else:
            combined = pd.concat(
                [
                    existing.dropna(axis=1, how="all"),
                    new_rows.dropna(axis=1, how="all"),
                ],
                ignore_index=True,
                sort=False,
            )
            combined = _normalize_dataset_frame(combined)

        combined = _dedupe_signal_frame(combined)
        _write_signals_dataset_unlocked(combined, cumul_csv)


def sync_live_to_cumulative() -> int:
    """Bulk-sync the live signal dataset into the cumulative archive.

    Loads every signal from the live ``SIGNAL_DATASET_PATH``, filters out
    rows already present in the cumulative store, and appends the new ones.
    Called once at engine startup to guard against data loss if a prior
    session wrote to the live dataset without the auto-archival hook.

    Returns the number of newly archived rows.
    """
    live_path = SIGNAL_DATASET_PATH
    if not live_path.exists() and not _dataset_store_path(live_path).exists():
        return 0

    live = load_signals_dataset(live_path)
    if live.empty:
        return 0

    before_count = 0
    cumul_csv = CUMULATIVE_DATASET_PATH
    if cumul_csv.exists() or _dataset_store_path(cumul_csv).exists():
        try:
            before_count = len(load_cumulative_dataset())
        except Exception:
            before_count = 0

    _sync_to_cumulative(live)

    after_count = 0
    if cumul_csv.exists() or _dataset_store_path(cumul_csv).exists():
        try:
            after_count = len(load_cumulative_dataset())
        except Exception:
            after_count = before_count

    return after_count - before_count
