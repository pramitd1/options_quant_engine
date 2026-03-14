"""
Canonical signal evaluation dataset management.

The dataset is a deduplicated CSV keyed by `signal_id`.
Each signal corresponds to exactly one canonical row, which may be
incrementally enriched over time as realized outcomes become available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import BASE_DIR


SIGNAL_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "signals_dataset.csv"

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
    "signed_return_15m_bps",
    "signed_return_30m_bps",
    "signed_return_60m_bps",
    "signed_return_120m_bps",
    "signed_return_session_close_bps",
    "mfe_points",
    "mae_points",
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
    "signal_calibration_bucket",
    "probability_calibration_bucket",
    "notes",
]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _empty_dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_DATASET_COLUMNS)


def _normalize_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy() if frame is not None else _empty_dataset_frame()

    for column in SIGNAL_DATASET_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[SIGNAL_DATASET_COLUMNS]
    return normalized


def load_signals_dataset(path: str | Path = SIGNAL_DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        write_signals_dataset(_empty_dataset_frame(), dataset_path)
        return _empty_dataset_frame()

    frame = pd.read_csv(dataset_path)
    return _normalize_dataset_frame(frame)


def write_signals_dataset(frame: pd.DataFrame, path: str | Path = SIGNAL_DATASET_PATH) -> Path:
    dataset_path = Path(path)
    _ensure_parent_dir(dataset_path)
    normalized = _normalize_dataset_frame(frame)
    normalized.to_csv(dataset_path, index=False)
    return dataset_path


def ensure_signals_dataset_exists(path: str | Path = SIGNAL_DATASET_PATH) -> Path:
    dataset_path = Path(path)
    if not dataset_path.exists():
        write_signals_dataset(_empty_dataset_frame(), dataset_path)
    return dataset_path


def upsert_signal_rows(rows, path: str | Path = SIGNAL_DATASET_PATH) -> pd.DataFrame:
    existing = load_signals_dataset(path)
    incoming = pd.DataFrame(rows or [])

    if incoming.empty:
        write_signals_dataset(existing, path)
        return existing

    incoming = _normalize_dataset_frame(incoming)
    if incoming["signal_id"].isna().any():
        raise ValueError("All signal rows must include a non-null signal_id")

    if existing.empty:
        combined = incoming.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)

    sort_key = "__sort_key__"
    combined[sort_key] = combined["updated_at"].fillna(combined["created_at"]).fillna("")
    combined = combined.sort_values(sort_key, kind="stable")
    combined = combined.drop_duplicates(subset=["signal_id"], keep="last")
    combined = combined.drop(columns=[sort_key]).reset_index(drop=True)

    write_signals_dataset(combined, path)
    return combined
