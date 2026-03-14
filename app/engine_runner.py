import os
from typing import Dict, Optional

import pandas as pd

from data.spot_downloader import get_spot_snapshot, save_spot_snapshot
from data.data_source_router import DataSourceRouter
from data.replay_loader import (
    latest_replay_snapshot_paths,
    load_option_chain_snapshot,
    load_spot_snapshot,
    save_option_chain_snapshot,
)
from data.option_chain_validation import validate_option_chain
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from macro.macro_news_aggregator import build_macro_news_state
from news.service import build_default_headline_service
from engine.trading_engine import generate_trade
from engine.runtime_metadata import TRADER_VIEW_KEYS
from research.signal_evaluation import (
    CAPTURE_POLICY_ALL,
    SIGNAL_DATASET_PATH,
    normalize_capture_policy,
    save_signal_evaluation,
    should_capture_signal,
)


def _set_runtime_credentials(source: str, credentials: Optional[Dict[str, str]] = None) -> None:
    credentials = credentials or {}
    source = source.upper().strip()

    if source == "ZERODHA":
        mapping = {
            "api_key": "ZERODHA_API_KEY",
            "api_secret": "ZERODHA_API_SECRET",
            "access_token": "ZERODHA_ACCESS_TOKEN",
        }
    elif source == "ICICI":
        mapping = {
            "api_key": "ICICI_BREEZE_API_KEY",
            "secret_key": "ICICI_BREEZE_SECRET_KEY",
            "session_token": "ICICI_BREEZE_SESSION_TOKEN",
        }
    else:
        mapping = {}

    for field_name, env_name in mapping.items():
        value = (credentials.get(field_name) or "").strip()
        if value:
            os.environ[env_name] = value


def _jsonable_headline_state(headline_state) -> Dict[str, object]:
    return {
        "provider_name": headline_state.provider_name,
        "fetched_at": headline_state.fetched_at,
        "latest_headline_at": headline_state.latest_headline_at,
        "is_stale": headline_state.is_stale,
        "data_available": headline_state.data_available,
        "neutral_fallback": headline_state.neutral_fallback,
        "stale_after_minutes": headline_state.stale_after_minutes,
        "issues": headline_state.issues,
        "warnings": headline_state.warnings,
        "provider_metadata": headline_state.provider_metadata,
        "records": [record.to_dict() for record in headline_state.records],
    }


def _trade_view_rows(trade: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for key in TRADER_VIEW_KEYS:
        if key in trade:
            rows.append({"field": key, "value": trade.get(key)})
    return pd.DataFrame(rows)


def _prepare_option_chain_frame(option_chain: pd.DataFrame) -> pd.DataFrame:
    if option_chain is None or option_chain.empty:
        return pd.DataFrame()

    frame = option_chain.copy()
    for col in ["strikePrice", "lastPrice", "openInterest", "changeinOI", "impliedVolatility", "GAMMA", "DELTA", "VEGA", "THETA", "VANNA", "CHARM"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def run_engine_snapshot(
    *,
    symbol: str,
    mode: str,
    source: str,
    apply_budget_constraint: bool,
    requested_lots: int,
    lot_size: int,
    max_capital: float,
    provider_credentials: Optional[Dict[str, str]] = None,
    replay_spot: Optional[str] = None,
    replay_chain: Optional[str] = None,
    replay_dir: str = "debug_samples",
    save_live_snapshots: bool = False,
    capture_signal_evaluation: bool = True,
    signal_capture_policy: str = CAPTURE_POLICY_ALL,
) -> Dict[str, object]:
    mode = mode.upper().strip()
    source = source.upper().strip()
    replay_mode = mode == "REPLAY"
    signal_capture_policy = normalize_capture_policy(signal_capture_policy)
    headline_service = build_default_headline_service()
    data_router = None

    try:
        if replay_mode:
            if not replay_spot or not replay_chain:
                discovered_spot, discovered_chain = latest_replay_snapshot_paths(
                    symbol,
                    replay_dir=replay_dir,
                )
                replay_spot = replay_spot or discovered_spot
                replay_chain = replay_chain or discovered_chain

            if not replay_spot or not replay_chain:
                return {
                    "ok": False,
                    "error": "Replay mode requires both a spot snapshot and an option-chain snapshot.",
                    "mode": mode,
                    "source": source,
                }

            spot_snapshot = load_spot_snapshot(replay_spot)
            option_chain = load_option_chain_snapshot(replay_chain)
            replay_paths = {
                "spot": replay_spot,
                "chain": replay_chain,
            }
        else:
            _set_runtime_credentials(source, provider_credentials)
            data_router = DataSourceRouter(source)
            spot_snapshot = get_spot_snapshot(symbol)
            option_chain = data_router.get_option_chain(symbol)
            replay_paths = None

        spot_validation = spot_snapshot.get("validation", {})
        spot = float(spot_snapshot["spot"])
        day_open = spot_snapshot.get("day_open")
        day_high = spot_snapshot.get("day_high")
        day_low = spot_snapshot.get("day_low")
        prev_close = spot_snapshot.get("prev_close")
        spot_timestamp = spot_snapshot.get("timestamp")
        lookback_avg_range_pct = spot_snapshot.get("lookback_avg_range_pct")

        macro_event_state = evaluate_scheduled_event_risk(
            symbol=symbol,
            as_of=spot_timestamp,
        )

        headline_state = headline_service.fetch(
            symbol=symbol,
            as_of=spot_timestamp,
            replay_mode=replay_mode,
        )
        macro_news_state = build_macro_news_state(
            event_state=macro_event_state,
            headline_state=headline_state,
            as_of=spot_timestamp,
        ).to_dict()

        resolved_expiry = resolve_selected_expiry(option_chain)
        option_chain = filter_option_chain_by_expiry(option_chain, resolved_expiry)
        option_chain_validation = validate_option_chain(option_chain)
        option_chain_frame = _prepare_option_chain_frame(option_chain)

        if save_live_snapshots and not replay_mode:
            saved_paths = {}
            try:
                saved_paths["spot"] = save_spot_snapshot(spot_snapshot)
            except Exception:
                saved_paths["spot"] = None
            try:
                saved_paths["chain"] = save_option_chain_snapshot(
                    option_chain,
                    symbol=symbol,
                    source=source,
                )
            except Exception:
                saved_paths["chain"] = None
        else:
            saved_paths = None

        trade = None
        if spot_validation.get("is_valid", False) and option_chain_validation.get("is_valid", False):
            trade = generate_trade(
                symbol=symbol,
                spot=spot,
                option_chain=option_chain,
                previous_chain=None,
                day_high=day_high,
                day_low=day_low,
                day_open=day_open,
                prev_close=prev_close,
                lookback_avg_range_pct=lookback_avg_range_pct,
                spot_validation=spot_validation,
                option_chain_validation=option_chain_validation,
                apply_budget_constraint=apply_budget_constraint,
                requested_lots=requested_lots,
                lot_size=lot_size,
                max_capital=max_capital,
                macro_event_state=macro_event_state,
                macro_news_state=macro_news_state,
                valuation_time=spot_timestamp,
            )
            if trade:
                trade["selected_expiry"] = option_chain_validation.get("selected_expiry")

        result_payload = {
            "ok": True,
            "mode": mode,
            "source": source,
            "symbol": symbol,
            "replay_paths": replay_paths,
            "saved_paths": saved_paths,
            "spot_snapshot": spot_snapshot,
            "spot_validation": spot_validation,
            "spot_summary": {
                "spot": spot,
                "day_open": day_open,
                "day_high": day_high,
                "day_low": day_low,
                "prev_close": prev_close,
                "timestamp": spot_timestamp,
                "lookback_avg_range_pct": lookback_avg_range_pct,
            },
            "macro_event_state": macro_event_state,
            "headline_state": _jsonable_headline_state(headline_state),
            "macro_news_state": macro_news_state,
            "option_chain_validation": option_chain_validation,
            "option_chain_rows": len(option_chain) if option_chain is not None else 0,
            "option_chain_frame": option_chain_frame,
            "option_chain_preview": option_chain.head(20).to_dict(orient="records"),
            "trade": trade,
            "trader_view_rows": _trade_view_rows(trade) if trade else pd.DataFrame(columns=["field", "value"]),
            "ranked_strikes": pd.DataFrame(trade.get("ranked_strike_candidates", [])) if trade else pd.DataFrame(),
            "headline_records": pd.DataFrame(_jsonable_headline_state(headline_state)["records"]),
            "signal_dataset_path": str(SIGNAL_DATASET_PATH),
            "signal_capture_status": "SKIPPED",
            "signal_capture_policy": signal_capture_policy,
        }

        if capture_signal_evaluation and should_capture_signal(trade, signal_capture_policy):
            try:
                save_signal_evaluation(result_payload)
                result_payload["signal_capture_status"] = "CAPTURED"
            except Exception as exc:
                result_payload["signal_capture_status"] = f"FAILED:{type(exc).__name__}"
                result_payload["signal_capture_error"] = str(exc)
        elif capture_signal_evaluation and trade:
            result_payload["signal_capture_status"] = f"SKIPPED_POLICY:{signal_capture_policy}"

        return result_payload
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "mode": mode,
            "source": source,
            "symbol": symbol,
        }
    finally:
        if data_router is not None:
            data_router.close()
