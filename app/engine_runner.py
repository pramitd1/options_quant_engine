"""
Module: engine_runner.py

Purpose:
    Implement engine runner logic used by operator-facing runtime workflows.

Role in the System:
    Part of the application layer that orchestrates runtime loops, sinks, and operator-facing workflows.

Key Outputs:
    Runtime side effects, operator-facing payloads, and orchestration callbacks.

Downstream Usage:
    Consumed by operators, replay tooling, and research capture workflows.
"""
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
from config.settings import STOP_LOSS_PERCENT, TARGET_PROFIT_PERCENT
from data.spot_downloader import get_spot_snapshot, save_spot_snapshot, validate_spot_snapshot
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
from engine.runtime_metadata import TRADER_VIEW_KEYS, attach_trade_views, split_trade_payload
from risk import build_global_risk_state
from research.signal_evaluation import (
    CAPTURE_POLICY_ALL,
    SIGNAL_DATASET_PATH,
    normalize_capture_policy,
)
from tuning.runtime import get_active_parameter_pack, temporary_parameter_pack
from tuning.promotion import get_promotion_runtime_context


def _set_runtime_credentials(source: str, credentials: Optional[Dict[str, str]] = None) -> None:
    """
    Purpose:
        Process set runtime credentials for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        source (str): Data-source label associated with the current snapshot.
        credentials (Optional[Dict[str, str]]): Input associated with credentials.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process jsonable headline state for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        headline_state (Any): Structured state payload for headline.
    
    Returns:
        Dict[str, object]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process trade view rows for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        trade (Dict[str, object]): Input associated with trade.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    trader_view = (trade or {}).get("execution_trade") if isinstance(trade, dict) else None
    trader_view = trader_view if isinstance(trader_view, dict) else trade

    rows = []
    for key in TRADER_VIEW_KEYS:
        if isinstance(trader_view, dict) and key in trader_view:
            rows.append({"field": key, "value": trader_view.get(key)})
    return pd.DataFrame(rows)


def _prepare_option_chain_frame(option_chain: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process prepare option chain frame for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        option_chain (pd.DataFrame): Input associated with option chain.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if option_chain is None or option_chain.empty:
        return pd.DataFrame()

    frame = option_chain.copy()
    for col in ["strikePrice", "lastPrice", "openInterest", "changeinOI", "impliedVolatility", "GAMMA", "DELTA", "VEGA", "THETA", "VANNA", "CHARM"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _is_replay_like_mode(mode: str) -> bool:
    """
    Purpose:
        Classify modes that should behave like historical or replay analysis.

    Context:
        Internal application-layer helper used to keep live-only side effects,
        such as provider freshness expectations, separate from replay and
        backtest orchestration.

    Inputs:
        mode (str): Runtime mode label supplied to the engine runner.

    Returns:
        bool: `True` when the mode should use replay-safe auxiliary data paths.

    Notes:
        Backtests intentionally reuse the replay-safe context so they can flow
        through the same orchestration stages without forcing live headline or
        market-data fetch semantics.
    """

    return str(mode or "").upper().strip() in {"REPLAY", "BACKTEST"}


def _ensure_spot_snapshot_validation(*, spot_snapshot: dict, replay_mode: bool) -> dict:
    """
    Purpose:
        Ensure a spot snapshot includes the validation payload expected by the
        rest of the runtime pipeline.

    Context:
        Internal helper used by both live-loaded and preloaded snapshot paths so
        downstream signal assembly can rely on a stable spot contract.

    Inputs:
        spot_snapshot (dict): Spot snapshot supplied by the loader or backtest
            adapter.
        replay_mode (bool): Whether the snapshot should be validated under
            replay-safe staleness rules.

    Returns:
        dict: Spot snapshot with a populated `validation` payload.

    Notes:
        Historical backtests often synthesize spot summaries from option-chain
        snapshots, so validation needs to be attached here rather than assumed
        to already exist.
    """

    snapshot = dict(spot_snapshot or {})
    validation = snapshot.get("validation")
    if not isinstance(validation, dict):
        snapshot["validation"] = validate_spot_snapshot(snapshot, replay_mode=replay_mode)
    return snapshot


def _extract_spot_context(*, spot_snapshot: dict, replay_mode: bool) -> dict:
    """
    Purpose:
        Normalize the spot snapshot into the scalar fields used across runtime
        orchestration.

    Context:
        Internal helper that keeps the runner's later stages focused on policy
        evaluation rather than repeatedly unpacking the spot payload.

    Inputs:
        spot_snapshot (dict): Spot snapshot supplied by the data-loading or
            backtest adapter stage.
        replay_mode (bool): Whether spot validation should use replay-safe
            staleness rules.

    Returns:
        dict: Normalized spot context containing the validated snapshot plus the
        scalar fields consumed by the signal engine.

    Notes:
        The returned dictionary deliberately mirrors the names later passed into
        `_evaluate_snapshot_for_pack` so live and preloaded paths can share the
        same downstream code.
    """

    snapshot = _ensure_spot_snapshot_validation(spot_snapshot=spot_snapshot, replay_mode=replay_mode)
    return {
        "spot_snapshot": snapshot,
        "spot_validation": snapshot.get("validation", {}),
        "spot": float(snapshot["spot"]),
        "day_open": snapshot.get("day_open"),
        "day_high": snapshot.get("day_high"),
        "day_low": snapshot.get("day_low"),
        "prev_close": snapshot.get("prev_close"),
        "spot_timestamp": snapshot.get("timestamp"),
        "lookback_avg_range_pct": snapshot.get("lookback_avg_range_pct"),
    }


def _neutral_global_market_snapshot(symbol: str, *, as_of, warning: str) -> dict:
    """
    Purpose:
        Build a neutral global-market snapshot for historical paths that do not
        have point-in-time cross-asset data available.

    Context:
        Internal helper used primarily by backtests so they can exercise the
        same global-risk assembly path as live runtime without silently pulling
        present-day market data into historical evaluations.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        as_of (Any): Timestamp associated with the current snapshot.
        warning (str): Reason recorded in the neutral snapshot metadata.

    Returns:
        dict: Neutral global-market snapshot shaped like the live data payload.

    Notes:
        This is a parity-preserving fallback rather than a market-data model. It
        keeps the orchestration path consistent while making the absence of
        historical cross-asset context explicit.
    """

    return {
        "symbol": str(symbol or "").upper().strip(),
        "provider": "BACKTEST_NEUTRAL",
        "as_of": as_of,
        "data_available": False,
        "neutral_fallback": True,
        "issues": [],
        "warnings": [warning],
        "stale": False,
        "lookback_days": None,
        "market_inputs": {},
    }


def _prepare_snapshot_context(
    *,
    symbol: str,
    mode: str,
    source: str,
    spot_snapshot: dict,
    option_chain: pd.DataFrame,
    headline_service,
    holding_profile: str,
    macro_event_state: Optional[dict] = None,
    headline_state=None,
    global_market_snapshot: Optional[dict] = None,
) -> dict:
    """
    Purpose:
        Assemble the non-strategy context shared by all engine evaluations for a
        given snapshot.

    Context:
        Internal application-layer stage that transforms raw market inputs into
        the validated snapshot context consumed by the signal engine, risk
        overlays, and runtime sinks.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        mode (str): Runtime mode label such as `LIVE`, `REPLAY`, or
            `BACKTEST`.
        source (str): Market-data source label associated with the snapshot.
        spot_snapshot (dict): Spot snapshot for the current evaluation.
        option_chain (pd.DataFrame): Option-chain snapshot for the current
            evaluation.
        headline_service (Any): Headline service used when a headline state has
            not already been provided.
        holding_profile (str): Holding intent used by overnight-sensitive
            overlays downstream.
        macro_event_state (Optional[dict]): Precomputed macro-event state, when
            available.
        headline_state (Any): Precomputed headline state, when available.
        global_market_snapshot (Optional[dict]): Precomputed global-market
            snapshot, when available.

    Returns:
        dict: Prepared snapshot context used by the parameter-pack evaluation
        stage.

    Notes:
        This stage is the key refactor seam for runtime/backtest parity. Live
        and historical callers now share the same expiry filtering, validation,
        macro/news assembly, and global-risk inputs once raw market data has
        been loaded.
    """

    replay_mode = _is_replay_like_mode(mode)
    spot_context = _extract_spot_context(spot_snapshot=spot_snapshot, replay_mode=replay_mode)
    spot_timestamp = spot_context["spot_timestamp"]

    prepared_headline_state = headline_state
    if prepared_headline_state is None:
        prepared_headline_state = headline_service.fetch(
            symbol=symbol,
            as_of=spot_timestamp,
            replay_mode=replay_mode,
        )

    prepared_macro_event_state = (
        dict(macro_event_state)
        if isinstance(macro_event_state, dict)
        else evaluate_scheduled_event_risk(symbol=symbol, as_of=spot_timestamp)
    )

    prepared_global_market_snapshot = global_market_snapshot
    if prepared_global_market_snapshot is None:
        prepared_global_market_snapshot = build_global_market_snapshot(
            symbol,
            as_of=spot_timestamp,
        )

    resolved_expiry = resolve_selected_expiry(option_chain)
    filtered_option_chain = filter_option_chain_by_expiry(option_chain, resolved_expiry)
    option_chain_validation = validate_option_chain(filtered_option_chain)
    option_chain_frame = _prepare_option_chain_frame(filtered_option_chain)

    return {
        "mode": mode,
        "source": source,
        "symbol": symbol,
        "holding_profile": holding_profile,
        **spot_context,
        "macro_event_state": prepared_macro_event_state,
        "headline_state": prepared_headline_state,
        "global_market_snapshot": prepared_global_market_snapshot,
        "resolved_expiry": resolved_expiry,
        "option_chain": filtered_option_chain,
        "option_chain_validation": option_chain_validation,
        "option_chain_frame": option_chain_frame,
    }


def _resolve_runtime_pack_selection(
    *,
    authoritative_pack_name: Optional[str],
    shadow_pack_name: Optional[str],
    use_promotion_state: bool,
) -> tuple[Optional[str], Optional[str]]:
    """
    Purpose:
        Resolve runtime pack selection needed by downstream logic.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        authoritative_pack_name (Optional[str]): Human-readable name for authoritative pack.
        shadow_pack_name (Optional[str]): Human-readable name for shadow pack.
        use_promotion_state (bool): Structured state payload for use promotion.
    
    Returns:
        tuple[Optional[str], Optional[str]]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process load market inputs for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        replay_mode (bool): Boolean flag associated with replay_mode.
        symbol (str): Underlying symbol or index identifier.
        source (str): Data-source label associated with the current snapshot.
        provider_credentials (Optional[Dict[str, str]]): Input associated with provider credentials.
        replay_spot (Optional[str]): Input associated with replay spot.
        replay_chain (Optional[str]): Input associated with replay chain.
        replay_dir (str): Input associated with replay dir.
        data_router (Optional[DataSourceRouter]): Input associated with data router.
    
    Returns:
        tuple[dict, pd.DataFrame, dict | None, Optional[DataSourceRouter]]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process persist snapshot artifacts for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        save_live_snapshots (bool): Boolean flag associated with save_live_snapshots.
        replay_mode (bool): Boolean flag associated with replay_mode.
        spot_snapshot (dict): Input associated with spot snapshot.
        option_chain (pd.DataFrame): Input associated with option chain.
        symbol (str): Underlying symbol or index identifier.
        source (str): Data-source label associated with the current snapshot.
    
    Returns:
        dict | None: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Build the result payload used by downstream components.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        mode (str): Execution mode label such as live, replay, or backtest.
        source (str): Data-source label associated with the current snapshot.
        symbol (str): Underlying symbol or index identifier.
        replay_paths (dict | None): Input associated with replay paths.
        saved_paths (dict | None): Input associated with saved paths.
        spot_snapshot (dict): Input associated with spot snapshot.
        spot_validation (dict): Input associated with spot validation.
        spot (float): Input associated with spot.
        day_open (Any): Input associated with day open.
        day_high (Any): Input associated with day high.
        day_low (Any): Input associated with day low.
        prev_close (Any): Input associated with prev close.
        spot_timestamp (Any): Timestamp associated with spot.
        lookback_avg_range_pct (Any): Input associated with lookback avg range percentage.
        macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
        headline_state (Any): Structured state payload for headline.
        macro_news_state (dict): Headline-driven macro state produced by the news layer.
        global_market_snapshot (dict): Cross-asset market snapshot used by the global-risk overlay.
        global_risk_state (dict): Structured state payload for global risk.
        option_chain_validation (dict): Input associated with option chain validation.
        option_chain (pd.DataFrame): Input associated with option chain.
        option_chain_frame (pd.DataFrame): Input associated with option chain frame.
        trade (dict | None): Input associated with trade.
        authoritative_pack_name (str): Human-readable name for authoritative pack.
        signal_capture_policy (str): Input associated with signal capture policy.
    
    Returns:
        Dict[str, object]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    trade = attach_trade_views(trade)
    execution_trade, trade_audit = split_trade_payload(trade)
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
        "execution_trade": execution_trade,
        "trade_audit": trade_audit,
        "authoritative_parameter_pack": authoritative_pack_name,
        "shadow_mode_active": False,
        "trader_view_rows": _trade_view_rows(trade) if trade else pd.DataFrame(columns=["field", "value"]),
        "ranked_strikes": pd.DataFrame((trade_audit or {}).get("ranked_strike_candidates", [])) if trade else pd.DataFrame(),
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
    backtest_mode: bool = False,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
) -> None:
    """
    Purpose:
        Process maybe attach shadow evaluation for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result_payload (Dict[str, object]): Payload containing result.
        shadow_pack_name (Optional[str]): Human-readable name for shadow pack.
        symbol (str): Underlying symbol or index identifier.
        mode (str): Execution mode label such as live, replay, or backtest.
        source (str): Data-source label associated with the current snapshot.
        spot (float): Input associated with spot.
        option_chain (pd.DataFrame): Input associated with option chain.
        previous_chain (Optional[pd.DataFrame]): Input associated with previous chain.
        day_high (Any): Input associated with day high.
        day_low (Any): Input associated with day low.
        day_open (Any): Input associated with day open.
        prev_close (Any): Input associated with prev close.
        lookback_avg_range_pct (Any): Input associated with lookback avg range percentage.
        spot_validation (dict): Input associated with spot validation.
        option_chain_validation (dict): Input associated with option chain validation.
        apply_budget_constraint (bool): Boolean flag associated with apply_budget_constraint.
        requested_lots (int): Input associated with requested lots.
        lot_size (int): Input associated with lot size.
        max_capital (float): Input associated with max capital.
        macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
        headline_state (Any): Structured state payload for headline.
        global_market_snapshot (dict): Cross-asset market snapshot used by the global-risk overlay.
        holding_profile (str): Holding intent that determines whether overnight rules should be considered.
        spot_timestamp (Any): Timestamp associated with spot.
        baseline_pack_name (str): Human-readable name for baseline pack.
        enable_shadow_logging (bool): Boolean flag associated with enable_shadow_logging.
        shadow_evaluation_sink (ShadowEvaluationSink): Input associated with shadow evaluation sink.
        backtest_mode (bool): Whether the underlying evaluation should use backtest-specific signal thresholds.
        target_profit_percent (float): Exit-model target used during both baseline and shadow evaluation.
        stop_loss_percent (float): Exit-model stop-loss percentage used during both baseline and shadow evaluation.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
        backtest_mode=backtest_mode,
        target_profit_percent=target_profit_percent,
        stop_loss_percent=stop_loss_percent,
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
    """
    Purpose:
        Process apply signal capture for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result_payload (Dict[str, object]): Payload containing result.
        trade (dict | None): Input associated with trade.
        capture_signal_evaluation (bool): Boolean flag associated with capture_signal_evaluation.
        signal_capture_policy (str): Input associated with signal capture policy.
        signal_capture_sink (SignalCaptureSink): Input associated with signal capture sink.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    backtest_mode: bool = False,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
):
    """
    Purpose:
        Process evaluate snapshot for pack for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        parameter_pack_name (Optional[str]): Human-readable name for parameter pack.
        symbol (str): Underlying symbol or index identifier.
        spot (float): Input associated with spot.
        option_chain (pd.DataFrame): Input associated with option chain.
        previous_chain (Optional[pd.DataFrame]): Input associated with previous chain.
        day_high (Any): Input associated with day high.
        day_low (Any): Input associated with day low.
        day_open (Any): Input associated with day open.
        prev_close (Any): Input associated with prev close.
        lookback_avg_range_pct (Any): Input associated with lookback avg range percentage.
        spot_validation (dict): Input associated with spot validation.
        option_chain_validation (dict): Input associated with option chain validation.
        apply_budget_constraint (bool): Boolean flag associated with apply_budget_constraint.
        requested_lots (int): Input associated with requested lots.
        lot_size (int): Input associated with lot size.
        max_capital (float): Input associated with max capital.
        macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
        headline_state (Any): Structured state payload for headline.
        global_market_snapshot (dict): Cross-asset market snapshot used by the global-risk overlay.
        holding_profile (str): Holding intent that determines whether overnight rules should be considered.
        spot_timestamp (Any): Timestamp associated with spot.
        backtest_mode (bool): Whether the signal engine should apply backtest-specific runtime thresholds.
        target_profit_percent (float): Exit-model target passed through to the signal engine.
        stop_loss_percent (float): Exit-model stop-loss percentage passed through to the signal engine.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
                backtest_mode=backtest_mode,
                macro_event_state=macro_event_state,
                macro_news_state=macro_news_state,
                global_risk_state=global_risk_state,
                holding_profile=holding_profile,
                valuation_time=spot_timestamp,
                target_profit_percent=target_profit_percent,
                stop_loss_percent=stop_loss_percent,
            )
            if trade:
                trade = attach_trade_views(trade)
                trade["selected_expiry"] = option_chain_validation.get("selected_expiry")
                trade["parameter_pack_name"] = active_pack_name
                trade = attach_trade_views(trade)

        return {
            "parameter_pack_name": active_pack_name,
            "macro_news_state": macro_news_state,
            "global_risk_state": global_risk_state,
            "trade": trade,
        }


def run_preloaded_engine_snapshot(
    *,
    symbol: str,
    mode: str,
    source: str,
    spot_snapshot: dict,
    option_chain: pd.DataFrame,
    apply_budget_constraint: bool,
    requested_lots: int,
    lot_size: int,
    max_capital: float,
    capture_signal_evaluation: bool = True,
    signal_capture_policy: str = CAPTURE_POLICY_ALL,
    previous_chain: Optional[pd.DataFrame] = None,
    holding_profile: str = "AUTO",
    headline_service=None,
    authoritative_pack_name: Optional[str] = None,
    shadow_pack_name: Optional[str] = None,
    enable_shadow_logging: bool = True,
    use_promotion_state: bool = False,
    signal_capture_sink: Optional[SignalCaptureSink] = None,
    shadow_evaluation_sink: Optional[ShadowEvaluationSink] = None,
    replay_paths: dict | None = None,
    saved_paths: dict | None = None,
    macro_event_state: Optional[dict] = None,
    headline_state=None,
    global_market_snapshot: Optional[dict] = None,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
) -> Dict[str, object]:
    """
    Purpose:
        Evaluate one fully loaded market snapshot through the full runtime
        orchestration path.
    
    Context:
        Public application-layer entry point used by live runtime, replay
        tooling, and backtests once spot and option-chain inputs are already in
        memory.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        mode (str): Execution mode label such as live, replay, or backtest.
        source (str): Data-source label associated with the current snapshot.
        spot_snapshot (dict): Loaded or synthesized spot snapshot for the
            current evaluation.
        option_chain (pd.DataFrame): Loaded option-chain snapshot for the
            current evaluation.
        apply_budget_constraint (bool): Boolean flag associated with apply_budget_constraint.
        requested_lots (int): Input associated with requested lots.
        lot_size (int): Input associated with lot size.
        max_capital (float): Input associated with max capital.
        capture_signal_evaluation (bool): Boolean flag associated with capture_signal_evaluation.
        signal_capture_policy (str): Input associated with signal capture policy.
        previous_chain (Optional[pd.DataFrame]): Input associated with previous chain.
        holding_profile (str): Holding intent that determines whether overnight rules should be considered.
        headline_service (Any): Input associated with headline service.
        authoritative_pack_name (Optional[str]): Human-readable name for authoritative pack.
        shadow_pack_name (Optional[str]): Human-readable name for shadow pack.
        enable_shadow_logging (bool): Boolean flag associated with enable_shadow_logging.
        use_promotion_state (bool): Structured state payload for use promotion.
        signal_capture_sink (Optional[SignalCaptureSink]): Input associated with signal capture sink.
        shadow_evaluation_sink (Optional[ShadowEvaluationSink]): Input associated with shadow evaluation sink.
        replay_paths (dict | None): Optional replay artifact paths to expose in
            the result payload.
        saved_paths (dict | None): Optional persisted snapshot paths to expose in
            the result payload.
        macro_event_state (Optional[dict]): Precomputed scheduled-event state, if
            already available.
        headline_state (Any): Precomputed headline state, if already available.
        global_market_snapshot (Optional[dict]): Precomputed global-market
            snapshot, if already available.
        target_profit_percent (float): Exit-model target percentage passed into
            trade construction.
        stop_loss_percent (float): Exit-model stop-loss percentage passed into
            trade construction.
    
    Returns:
        Dict[str, object]: Result returned by the helper.
    
    Notes:
        This function is the parity-preserving orchestration seam shared by live
        runtime and historical workflows. The only difference between callers is
        how they source the market inputs and auxiliary macro/global snapshots.
    """
    mode = mode.upper().strip()
    source = source.upper().strip()
    signal_capture_policy = normalize_capture_policy(signal_capture_policy)
    headline_service = headline_service or build_default_headline_service()
    signal_capture_sink = signal_capture_sink or DefaultSignalCaptureSink()
    shadow_evaluation_sink = shadow_evaluation_sink or DefaultShadowEvaluationSink()
    backtest_mode = mode == "BACKTEST"

    try:
        authoritative_pack_name, shadow_pack_name = _resolve_runtime_pack_selection(
            authoritative_pack_name=authoritative_pack_name,
            shadow_pack_name=shadow_pack_name,
            use_promotion_state=use_promotion_state,
        )

        snapshot_context = _prepare_snapshot_context(
            symbol=symbol,
            mode=mode,
            source=source,
            spot_snapshot=spot_snapshot,
            option_chain=option_chain,
            headline_service=headline_service,
            holding_profile=holding_profile,
            macro_event_state=macro_event_state,
            headline_state=headline_state,
            global_market_snapshot=global_market_snapshot,
        )

        authoritative_eval = _evaluate_snapshot_for_pack(
            parameter_pack_name=authoritative_pack_name,
            symbol=symbol,
            spot=snapshot_context["spot"],
            option_chain=snapshot_context["option_chain"],
            previous_chain=previous_chain,
            day_high=snapshot_context["day_high"],
            day_low=snapshot_context["day_low"],
            day_open=snapshot_context["day_open"],
            prev_close=snapshot_context["prev_close"],
            lookback_avg_range_pct=snapshot_context["lookback_avg_range_pct"],
            spot_validation=snapshot_context["spot_validation"],
            option_chain_validation=snapshot_context["option_chain_validation"],
            apply_budget_constraint=apply_budget_constraint,
            requested_lots=requested_lots,
            lot_size=lot_size,
            max_capital=max_capital,
            macro_event_state=snapshot_context["macro_event_state"],
            headline_state=snapshot_context["headline_state"],
            global_market_snapshot=snapshot_context["global_market_snapshot"],
            holding_profile=holding_profile,
            spot_timestamp=snapshot_context["spot_timestamp"],
            backtest_mode=backtest_mode,
            target_profit_percent=target_profit_percent,
            stop_loss_percent=stop_loss_percent,
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
            spot_snapshot=snapshot_context["spot_snapshot"],
            spot_validation=snapshot_context["spot_validation"],
            spot=snapshot_context["spot"],
            day_open=snapshot_context["day_open"],
            day_high=snapshot_context["day_high"],
            day_low=snapshot_context["day_low"],
            prev_close=snapshot_context["prev_close"],
            spot_timestamp=snapshot_context["spot_timestamp"],
            lookback_avg_range_pct=snapshot_context["lookback_avg_range_pct"],
            macro_event_state=snapshot_context["macro_event_state"],
            headline_state=snapshot_context["headline_state"],
            macro_news_state=macro_news_state,
            global_market_snapshot=snapshot_context["global_market_snapshot"],
            global_risk_state=global_risk_state,
            option_chain_validation=snapshot_context["option_chain_validation"],
            option_chain=snapshot_context["option_chain"],
            option_chain_frame=snapshot_context["option_chain_frame"],
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
            spot=snapshot_context["spot"],
            option_chain=snapshot_context["option_chain"],
            previous_chain=previous_chain,
            day_high=snapshot_context["day_high"],
            day_low=snapshot_context["day_low"],
            day_open=snapshot_context["day_open"],
            prev_close=snapshot_context["prev_close"],
            lookback_avg_range_pct=snapshot_context["lookback_avg_range_pct"],
            spot_validation=snapshot_context["spot_validation"],
            option_chain_validation=snapshot_context["option_chain_validation"],
            apply_budget_constraint=apply_budget_constraint,
            requested_lots=requested_lots,
            lot_size=lot_size,
            max_capital=max_capital,
            macro_event_state=snapshot_context["macro_event_state"],
            headline_state=snapshot_context["headline_state"],
            global_market_snapshot=snapshot_context["global_market_snapshot"],
            holding_profile=holding_profile,
            spot_timestamp=snapshot_context["spot_timestamp"],
            baseline_pack_name=authoritative_eval["parameter_pack_name"],
            enable_shadow_logging=enable_shadow_logging,
            shadow_evaluation_sink=shadow_evaluation_sink,
            backtest_mode=backtest_mode,
            target_profit_percent=target_profit_percent,
            stop_loss_percent=stop_loss_percent,
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
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
) -> Dict[str, object]:
    """
    Purpose:
        Load market inputs for one snapshot and route them through the shared
        runtime orchestration path.

    Context:
        Public application-layer entry point for live and replay usage. This
        function now acts primarily as the market-data loading wrapper around
        `run_preloaded_engine_snapshot`.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        mode (str): Execution mode label such as live or replay.
        source (str): Data-source label associated with the current snapshot.
        apply_budget_constraint (bool): Whether capital-budget rules should be
            enforced during trade construction.
        requested_lots (int): Requested lot count before any optimizer or caps
            are applied.
        lot_size (int): Contract lot size used by trade construction.
        max_capital (float): Maximum capital budget allowed for the trade.
        provider_credentials (Optional[Dict[str, str]]): Provider credentials
            used when loading live market data.
        replay_spot (Optional[str]): Optional replay spot snapshot path.
        replay_chain (Optional[str]): Optional replay option-chain snapshot path.
        replay_dir (str): Directory searched when replay paths are not supplied.
        save_live_snapshots (bool): Whether live snapshots should be persisted
            for later replay.
        capture_signal_evaluation (bool): Whether the result should be offered
            to the signal-evaluation sink.
        signal_capture_policy (str): Capture policy that determines which
            signals are persisted for research.
        previous_chain (Optional[pd.DataFrame]): Previous option-chain snapshot
            used for flow and open-interest delta features.
        holding_profile (str): Holding intent that determines whether overnight
            overlays should apply.
        headline_service (Any): Optional headline service override.
        data_router (Optional[DataSourceRouter]): Optional data router override
            used by tests or embedded runtimes.
        authoritative_pack_name (Optional[str]): Explicit authoritative
            parameter-pack name.
        shadow_pack_name (Optional[str]): Explicit shadow parameter-pack name.
        enable_shadow_logging (bool): Whether shadow comparisons should be
            persisted.
        use_promotion_state (bool): Whether promotion state should select the
            authoritative and shadow packs.
        signal_capture_sink (Optional[SignalCaptureSink]): Optional signal
            capture sink override.
        shadow_evaluation_sink (Optional[ShadowEvaluationSink]): Optional shadow
            evaluation sink override.
        target_profit_percent (float): Exit-model target percentage passed into
            trade construction.
        stop_loss_percent (float): Exit-model stop-loss percentage passed into
            trade construction.

    Returns:
        Dict[str, object]: Result returned by the helper.

    Notes:
        Keeping market-data loading separate from snapshot evaluation makes it
        straightforward for replay and backtest workflows to reuse the same
        macro/news/global-risk orchestration once their inputs are prepared.
    """

    mode = mode.upper().strip()
    source = source.upper().strip()
    replay_mode = mode == "REPLAY"
    managed_data_router = None

    try:
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

        saved_paths = _persist_snapshot_artifacts(
            save_live_snapshots=save_live_snapshots,
            replay_mode=replay_mode,
            spot_snapshot=spot_snapshot,
            option_chain=option_chain,
            symbol=symbol,
            source=source,
        )

        return run_preloaded_engine_snapshot(
            symbol=symbol,
            mode=mode,
            source=source,
            spot_snapshot=spot_snapshot,
            option_chain=option_chain,
            apply_budget_constraint=apply_budget_constraint,
            requested_lots=requested_lots,
            lot_size=lot_size,
            max_capital=max_capital,
            capture_signal_evaluation=capture_signal_evaluation,
            signal_capture_policy=signal_capture_policy,
            previous_chain=previous_chain,
            holding_profile=holding_profile,
            headline_service=headline_service,
            authoritative_pack_name=authoritative_pack_name,
            shadow_pack_name=shadow_pack_name,
            enable_shadow_logging=enable_shadow_logging,
            use_promotion_state=use_promotion_state,
            signal_capture_sink=signal_capture_sink,
            shadow_evaluation_sink=shadow_evaluation_sink,
            replay_paths=replay_paths,
            saved_paths=saved_paths,
            target_profit_percent=target_profit_percent,
            stop_loss_percent=stop_loss_percent,
        )
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
