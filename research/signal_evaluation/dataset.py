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
from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

from config.settings import BASE_DIR


SIGNAL_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "signals_dataset.csv"
CUMULATIVE_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"

SIGNAL_DATASET_COLUMNS = [
    "signal_id",
    "signal_timestamp",
    "source",
    "mode",
    "symbol",
    "ticker",
    "selected_expiry",
    "direction",
    "option_type",
    "strike",
    "entry_price",
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
    "target_reachability_score",
    "premium_efficiency_score",
    "strike_efficiency_score",
    "option_efficiency_score",
    "option_efficiency_adjustment_score",
    "dealer_position",
    "dealer_hedging_bias",
    "volatility_regime",
    "liquidity_vacuum_state",
    "confirmation_status",
    "macro_event_risk_score",
    "data_quality_score",
    "data_quality_status",
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
    "saved_spot_snapshot_path",
    "saved_chain_snapshot_path",
    "created_at",
    "updated_at",
    "outcome_last_updated_at",
    "outcome_status",
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
    "mfe_points",
    "mae_points",
    "correct_5m",
    "correct_15m",
    "correct_30m",
    "correct_60m",
    "correct_120m",
    "correct_session_close",
    "mfe_60m_bps",
    "mae_60m_bps",
    "mfe_120m_bps",
    "mae_120m_bps",
    "realized_range_60m_bps",
    "realized_range_120m_bps",
    "direction_score",
    "magnitude_score",
    "timing_score",
    "tradeability_score",
    "composite_signal_score",
    "directional_consistency_score",
    "signal_confidence_score",
    "signal_confidence_level",
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
    with closing(sqlite3.connect(path)) as connection:
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
    with closing(sqlite3.connect(path)) as connection:
        normalized = _normalize_dataset_frame(frame)
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
    with closing(sqlite3.connect(path)) as connection:
        if not _sqlite_has_table(connection, "signals"):
            normalized = _normalize_dataset_frame(frame)
            normalized.to_sql("signals", connection, if_exists="replace", index=False)
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)")
            return
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
            with closing(sqlite3.connect(sqlite_path)) as connection:
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

    if not path.exists() or path.stat().st_size == 0:
        write_signals_dataset(frame, path)
        return

    frame.to_csv(path, mode="a", header=False, index=False)
    _append_sqlite_rows(frame, _dataset_store_path(path))
    existing_ids = set(existing_signal_ids or ())
    existing_ids.update(str(signal_id) for signal_id in frame["signal_id"].dropna())
    _update_signal_id_cache(path, existing_ids)


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
    sqlite_path = _dataset_store_path(dataset_path)
    if not dataset_path.exists():
        write_signals_dataset(_empty_dataset_frame(), dataset_path)
        return _empty_dataset_frame()

    if sqlite_path.exists():
        try:
            return _read_sqlite_dataset(sqlite_path)
        except Exception:
            pass

    frame = pd.read_csv(dataset_path)
    return _normalize_dataset_frame(frame)


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
    _ensure_parent_dir(dataset_path)
    normalized = _normalize_dataset_frame(frame)
    normalized.to_csv(dataset_path, index=False)
    _write_sqlite_dataset(normalized, _dataset_store_path(dataset_path))
    _update_signal_id_cache(dataset_path, normalized.get("signal_id", pd.Series(dtype="object")))
    return dataset_path


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
    if not dataset_path.exists():
        write_signals_dataset(_empty_dataset_frame(), dataset_path)
    return dataset_path


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
    dataset_path = ensure_signals_dataset_exists(path)
    incoming = pd.DataFrame(rows or [])

    if incoming.empty:
        return load_signals_dataset(dataset_path) if return_frame else None

    incoming = _normalize_dataset_frame(incoming)
    if incoming["signal_id"].isna().any():
        raise ValueError("All signal rows must include a non-null signal_id")
    incoming = _dedupe_signal_frame(incoming)

    if not _dataset_has_canonical_header(dataset_path):
        existing = load_signals_dataset(dataset_path)
        write_signals_dataset(existing, dataset_path)

    existing_signal_ids = _load_existing_signal_ids(dataset_path)
    incoming_signal_ids = {str(signal_id) for signal_id in incoming["signal_id"].dropna()}

    if not existing_signal_ids.intersection(incoming_signal_ids):
        _append_rows_to_dataset(incoming, dataset_path, existing_signal_ids=existing_signal_ids)

        # Auto-sync to cumulative dataset when writing to the live dataset.
        if Path(path) == SIGNAL_DATASET_PATH:
            _sync_to_cumulative(incoming)

        return load_signals_dataset(dataset_path) if return_frame else None

    existing = load_signals_dataset(dataset_path)
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
    write_signals_dataset(combined, dataset_path)

    # Auto-sync to cumulative dataset when writing to the live dataset.
    if Path(path) == SIGNAL_DATASET_PATH:
        _sync_to_cumulative(incoming)

    return combined if return_frame else None


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
    cumul_sqlite = _dataset_store_path(cumul_csv)

    if incoming.empty:
        return

    incoming = _normalize_dataset_frame(incoming)
    
    # ── NEW: Apply ML inference to signals before syncing ──
    incoming = _apply_ml_inference_to_signals(incoming)

    # Load existing cumulative signal_ids to avoid duplicates.
    existing_ids: set[str] = set()
    if cumul_sqlite.exists():
        try:
            existing_ids = _load_existing_signal_ids(cumul_csv)
        except Exception:
            existing_ids = set()
    elif cumul_csv.exists():
        try:
            existing_ids = _load_existing_signal_ids(cumul_csv)
        except Exception:
            existing_ids = set()

    # Filter to only truly new rows.
    if "signal_id" in incoming.columns and existing_ids:
        new_rows = incoming[~incoming["signal_id"].isin(existing_ids)]
    else:
        new_rows = incoming

    if new_rows.empty:
        return

    # Append new rows to cumulative CSV.
    header = not cumul_csv.exists() or cumul_csv.stat().st_size == 0
    new_rows.to_csv(cumul_csv, mode="a", index=False, header=header)

    # Append new rows to cumulative SQLite.
    _append_sqlite_rows(new_rows, cumul_sqlite)


def sync_live_to_cumulative() -> int:
    """Bulk-sync the live signal dataset into the cumulative archive.

    Loads every signal from the live ``SIGNAL_DATASET_PATH``, filters out
    rows already present in the cumulative store, and appends the new ones.
    Called once at engine startup to guard against data loss if a prior
    session wrote to the live dataset without the auto-archival hook.

    Returns the number of newly archived rows.
    """
    live_path = SIGNAL_DATASET_PATH
    if not live_path.exists():
        return 0

    live = load_signals_dataset(live_path)
    if live.empty:
        return 0

    before_count = 0
    cumul_csv = CUMULATIVE_DATASET_PATH
    if cumul_csv.exists():
        try:
            before_count = len(load_cumulative_dataset())
        except Exception:
            before_count = 0

    _sync_to_cumulative(live)

    after_count = 0
    if cumul_csv.exists():
        try:
            after_count = len(load_cumulative_dataset())
        except Exception:
            after_count = before_count

    return after_count - before_count
