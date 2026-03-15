import os
from contextlib import nullcontext
from typing import Dict, Optional

import pandas as pd

from app.runtime_sinks import (
    DefaultShadowEvaluationSink,
    DefaultSignalCaptureSink,
    ShadowEvaluationSink,
    SignalCaptureSink,
)
from data.spot_downloader import get_spot_snapshot, save_spot_snapshot
from data.data_source_router import DataSourceRouter
from data.replay_loader import (
    latest_replay_snapshot_paths,
    load_option_chain_snapshot,
    load_spot_snapshot,
    save_option_chain_snapshot,
)
from data.global_market_snapshot import build_global_market_snapshot
from data.option_chain_validation import validate_option_chain
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from macro.macro_news_aggregator import build_macro_news_state
from news.service import build_default_headline_service
from engine.signal_engine import generate_trade
from engine.runtime_metadata import TRADER_VIEW_KEYS
from risk import build_global_risk_state
from research.signal_evaluation import (
    CAPTURE_POLICY_ALL,
    SIGNAL_DATASET_PATH,
    normalize_capture_policy,
)
from tuning.runtime import get_active_parameter_pack, temporary_parameter_pack
from tuning.promotion import get_promotion_runtime_context


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


def _resolve_runtime_pack_selection(
    *,
    authoritative_pack_name: Optional[str],
    shadow_pack_name: Optional[str],
    use_promotion_state: bool,
) -> tuple[Optional[str], Optional[str]]:
    if not use_promotion_state:
        return authoritative_pack_name, shadow_pack_name

    promotion_context = get_promotion_runtime_context()
    return (
        authoritative_pack_name or promotion_context.get("live_pack"),
        shadow_pack_name or promotion_context.get("shadow_pack"),
    )


def _load_market_inputs(
    *,
    replay_mode: bool,
    symbol: str,
    source: str,
    provider_credentials: Optional[Dict[str, str]],
    replay_spot: Optional[str],
    replay_chain: Optional[str],
    replay_dir: str,
    data_router: Optional[DataSourceRouter],
) -> tuple[dict, pd.DataFrame, dict | None, Optional[DataSourceRouter]]:
    managed_data_router = None

    if replay_mode:
        if not replay_spot or not replay_chain:
            discovered_spot, discovered_chain = latest_replay_snapshot_paths(
                symbol,
                replay_dir=replay_dir,
            )
            replay_spot = replay_spot or discovered_spot
            replay_chain = replay_chain or discovered_chain

        if not replay_spot or not replay_chain:
            raise ValueError("Replay mode requires both a spot snapshot and an option-chain snapshot.")

        spot_snapshot = load_spot_snapshot(replay_spot)
        option_chain = load_option_chain_snapshot(replay_chain)
        replay_paths = {
            "spot": replay_spot,
            "chain": replay_chain,
        }
        return spot_snapshot, option_chain, replay_paths, managed_data_router

    _set_runtime_credentials(source, provider_credentials)
    if data_router is None:
        data_router = DataSourceRouter(source)
        managed_data_router = data_router
    spot_snapshot = get_spot_snapshot(symbol)
    option_chain = data_router.get_option_chain(symbol)
    return spot_snapshot, option_chain, None, managed_data_router


def _persist_snapshot_artifacts(
    *,
    save_live_snapshots: bool,
    replay_mode: bool,
    spot_snapshot: dict,
    option_chain: pd.DataFrame,
    symbol: str,
    source: str,
) -> dict | None:
    if not save_live_snapshots or replay_mode:
        return None

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
    return saved_paths


def _build_result_payload(
    *,
    mode: str,
    source: str,
    symbol: str,
    replay_paths: dict | None,
    saved_paths: dict | None,
    spot_snapshot: dict,
    spot_validation: dict,
    spot: float,
    day_open,
    day_high,
    day_low,
    prev_close,
    spot_timestamp,
    lookback_avg_range_pct,
    macro_event_state: dict,
    headline_state,
    macro_news_state: dict,
    global_market_snapshot: dict,
    global_risk_state: dict,
    option_chain_validation: dict,
    option_chain: pd.DataFrame,
    option_chain_frame: pd.DataFrame,
    trade: dict | None,
    authoritative_pack_name: str,
    signal_capture_policy: str,
) -> Dict[str, object]:
    headline_state_payload = _jsonable_headline_state(headline_state)
    return {
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
        "headline_state": headline_state_payload,
        "macro_news_state": macro_news_state,
        "global_market_snapshot": global_market_snapshot,
        "global_risk_state": global_risk_state,
        "option_chain_validation": option_chain_validation,
        "option_chain_rows": len(option_chain) if option_chain is not None else 0,
        "option_chain_frame": option_chain_frame,
        "option_chain_preview": option_chain.head(20).to_dict(orient="records"),
        "trade": trade,
        "authoritative_parameter_pack": authoritative_pack_name,
        "shadow_mode_active": False,
        "trader_view_rows": _trade_view_rows(trade) if trade else pd.DataFrame(columns=["field", "value"]),
        "ranked_strikes": pd.DataFrame(trade.get("ranked_strike_candidates", [])) if trade else pd.DataFrame(),
        "headline_records": pd.DataFrame(headline_state_payload["records"]),
        "signal_dataset_path": str(SIGNAL_DATASET_PATH),
        "signal_capture_status": "SKIPPED",
        "signal_capture_policy": signal_capture_policy,
    }


def _maybe_attach_shadow_evaluation(
    *,
    result_payload: Dict[str, object],
    shadow_pack_name: Optional[str],
    symbol: str,
    mode: str,
    source: str,
    spot: float,
    option_chain: pd.DataFrame,
    previous_chain: Optional[pd.DataFrame],
    day_high,
    day_low,
    day_open,
    prev_close,
    lookback_avg_range_pct,
    spot_validation: dict,
    option_chain_validation: dict,
    apply_budget_constraint: bool,
    requested_lots: int,
    lot_size: int,
    max_capital: float,
    macro_event_state: dict,
    headline_state,
    global_market_snapshot: dict,
    holding_profile: str,
    spot_timestamp,
    baseline_pack_name: str,
    enable_shadow_logging: bool,
    shadow_evaluation_sink: ShadowEvaluationSink,
) -> None:
    shadow_evaluation_sink.apply(
        result_payload=result_payload,
        shadow_pack_name=shadow_pack_name,
        symbol=symbol,
        mode=mode,
        source=source,
        spot=spot,
        option_chain=option_chain,
        previous_chain=previous_chain,
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
        headline_state=headline_state,
        global_market_snapshot=global_market_snapshot,
        holding_profile=holding_profile,
        spot_timestamp=spot_timestamp,
        baseline_pack_name=baseline_pack_name,
        enable_shadow_logging=enable_shadow_logging,
        evaluate_snapshot_for_pack=_evaluate_snapshot_for_pack,
    )


def _apply_signal_capture(
    *,
    result_payload: Dict[str, object],
    trade: dict | None,
    capture_signal_evaluation: bool,
    signal_capture_policy: str,
    signal_capture_sink: SignalCaptureSink,
) -> None:
    signal_capture_sink.apply(
        result_payload=result_payload,
        trade=trade,
        capture_signal_evaluation=capture_signal_evaluation,
        signal_capture_policy=signal_capture_policy,
    )


def _evaluate_snapshot_for_pack(
    *,
    parameter_pack_name: Optional[str],
    symbol: str,
    spot: float,
    option_chain: pd.DataFrame,
    previous_chain: Optional[pd.DataFrame],
    day_high,
    day_low,
    day_open,
    prev_close,
    lookback_avg_range_pct,
    spot_validation: dict,
    option_chain_validation: dict,
    apply_budget_constraint: bool,
    requested_lots: int,
    lot_size: int,
    max_capital: float,
    macro_event_state: dict,
    headline_state,
    global_market_snapshot: dict,
    holding_profile: str,
    spot_timestamp,
):
    context_manager = temporary_parameter_pack(parameter_pack_name) if parameter_pack_name else nullcontext()
    with context_manager:
        active_pack_name = get_active_parameter_pack()["name"]
        macro_news_state = build_macro_news_state(
            event_state=macro_event_state,
            headline_state=headline_state,
            as_of=spot_timestamp,
        ).to_dict()
        global_risk_state = build_global_risk_state(
            macro_event_state=macro_event_state,
            macro_news_state=macro_news_state,
            global_market_snapshot=global_market_snapshot,
            holding_profile=holding_profile,
            as_of=spot_timestamp,
        )

        trade = None
        if spot_validation.get("is_valid", False) and option_chain_validation.get("is_valid", False):
            trade = generate_trade(
                symbol=symbol,
                spot=spot,
                option_chain=option_chain,
                previous_chain=previous_chain,
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
                global_risk_state=global_risk_state,
                holding_profile=holding_profile,
                valuation_time=spot_timestamp,
            )
            if trade:
                trade["selected_expiry"] = option_chain_validation.get("selected_expiry")
                trade["parameter_pack_name"] = active_pack_name

        return {
            "parameter_pack_name": active_pack_name,
            "macro_news_state": macro_news_state,
            "global_risk_state": global_risk_state,
            "trade": trade,
        }


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
    previous_chain: Optional[pd.DataFrame] = None,
    holding_profile: str = "AUTO",
    headline_service=None,
    data_router: Optional[DataSourceRouter] = None,
    authoritative_pack_name: Optional[str] = None,
    shadow_pack_name: Optional[str] = None,
    enable_shadow_logging: bool = True,
    use_promotion_state: bool = False,
    signal_capture_sink: Optional[SignalCaptureSink] = None,
    shadow_evaluation_sink: Optional[ShadowEvaluationSink] = None,
) -> Dict[str, object]:
    mode = mode.upper().strip()
    source = source.upper().strip()
    replay_mode = mode == "REPLAY"
    signal_capture_policy = normalize_capture_policy(signal_capture_policy)
    headline_service = headline_service or build_default_headline_service()
    signal_capture_sink = signal_capture_sink or DefaultSignalCaptureSink()
    shadow_evaluation_sink = shadow_evaluation_sink or DefaultShadowEvaluationSink()
    managed_data_router = None

    try:
        authoritative_pack_name, shadow_pack_name = _resolve_runtime_pack_selection(
            authoritative_pack_name=authoritative_pack_name,
            shadow_pack_name=shadow_pack_name,
            use_promotion_state=use_promotion_state,
        )
        spot_snapshot, option_chain, replay_paths, managed_data_router = _load_market_inputs(
            replay_mode=replay_mode,
            symbol=symbol,
            source=source,
            provider_credentials=provider_credentials,
            replay_spot=replay_spot,
            replay_chain=replay_chain,
            replay_dir=replay_dir,
            data_router=data_router,
        )

        spot_validation = spot_snapshot.get("validation", {})
        spot = float(spot_snapshot["spot"])
        day_open = spot_snapshot.get("day_open")
        day_high = spot_snapshot.get("day_high")
        day_low = spot_snapshot.get("day_low")
        prev_close = spot_snapshot.get("prev_close")
        spot_timestamp = spot_snapshot.get("timestamp")
        lookback_avg_range_pct = spot_snapshot.get("lookback_avg_range_pct")

        macro_event_state = evaluate_scheduled_event_risk(symbol=symbol, as_of=spot_timestamp)

        headline_state = headline_service.fetch(
            symbol=symbol,
            as_of=spot_timestamp,
            replay_mode=replay_mode,
        )
        global_market_snapshot = build_global_market_snapshot(
            symbol,
            as_of=spot_timestamp,
        )

        resolved_expiry = resolve_selected_expiry(option_chain)
        option_chain = filter_option_chain_by_expiry(option_chain, resolved_expiry)
        option_chain_validation = validate_option_chain(option_chain)
        option_chain_frame = _prepare_option_chain_frame(option_chain)

        saved_paths = _persist_snapshot_artifacts(
            save_live_snapshots=save_live_snapshots,
            replay_mode=replay_mode,
            spot_snapshot=spot_snapshot,
            option_chain=option_chain,
            symbol=symbol,
            source=source,
        )

        authoritative_eval = _evaluate_snapshot_for_pack(
            parameter_pack_name=authoritative_pack_name,
            symbol=symbol,
            spot=spot,
            option_chain=option_chain,
            previous_chain=previous_chain,
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
            headline_state=headline_state,
            global_market_snapshot=global_market_snapshot,
            holding_profile=holding_profile,
            spot_timestamp=spot_timestamp,
        )
        macro_news_state = authoritative_eval["macro_news_state"]
        global_risk_state = authoritative_eval["global_risk_state"]
        trade = authoritative_eval["trade"]

        result_payload = _build_result_payload(
            mode=mode,
            source=source,
            symbol=symbol,
            replay_paths=replay_paths,
            saved_paths=saved_paths,
            spot_snapshot=spot_snapshot,
            spot_validation=spot_validation,
            spot=spot,
            day_open=day_open,
            day_high=day_high,
            day_low=day_low,
            prev_close=prev_close,
            spot_timestamp=spot_timestamp,
            lookback_avg_range_pct=lookback_avg_range_pct,
            macro_event_state=macro_event_state,
            headline_state=headline_state,
            macro_news_state=macro_news_state,
            global_market_snapshot=global_market_snapshot,
            global_risk_state=global_risk_state,
            option_chain_validation=option_chain_validation,
            option_chain=option_chain,
            option_chain_frame=option_chain_frame,
            trade=trade,
            authoritative_pack_name=authoritative_eval["parameter_pack_name"],
            signal_capture_policy=signal_capture_policy,
        )

        _maybe_attach_shadow_evaluation(
            result_payload=result_payload,
            shadow_pack_name=shadow_pack_name,
            symbol=symbol,
            mode=mode,
            source=source,
            spot=spot,
            option_chain=option_chain,
            previous_chain=previous_chain,
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
            headline_state=headline_state,
            global_market_snapshot=global_market_snapshot,
            holding_profile=holding_profile,
            spot_timestamp=spot_timestamp,
            baseline_pack_name=authoritative_eval["parameter_pack_name"],
            enable_shadow_logging=enable_shadow_logging,
            shadow_evaluation_sink=shadow_evaluation_sink,
        )

        _apply_signal_capture(
            result_payload=result_payload,
            trade=trade,
            capture_signal_evaluation=capture_signal_evaluation,
            signal_capture_policy=signal_capture_policy,
            signal_capture_sink=signal_capture_sink,
        )

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
        if managed_data_router is not None:
            managed_data_router.close()
