"""
Module: runtime_sinks.py

Purpose:
    Define runtime sink contracts and default implementations for signal capture and shadow evaluation.

Role in the System:
    Part of the application layer that bridges engine snapshots into research logging and candidate-pack governance.

Key Outputs:
    Sink callbacks that persist signal-evaluation rows, shadow comparisons, and runtime status flags.

Downstream Usage:
    Used by the engine runner after each snapshot to update evaluation datasets and shadow logs.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import pandas as pd

from research.signal_evaluation import save_signal_evaluation, should_capture_signal
from tuning.shadow import append_shadow_log, compare_shadow_trade_outputs


class SignalCaptureSink(Protocol):
    """
    Purpose:
        Protocol for runtime sinks that decide whether and how to persist signal-evaluation records.
    
    Context:
        Used within the `runtime sinks` module. The class helps the runtime bridge engine snapshots into operator-visible sinks or research artifacts.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        Concrete implementations are expected to preserve this behavioral contract so runtime orchestration remains swappable.
    """
    def apply(
        self,
        *,
        result_payload: Dict[str, object],
        trade: dict | None,
        capture_signal_evaluation: bool,
        signal_capture_policy: str,
    ) -> None:
        """
        Purpose:
            Persist or skip the current signal snapshot according to the configured capture policy.
        
        Context:
            Method on `SignalCaptureSink` that forms part of the runtime sink contract implemented by this module.
        
        Inputs:
            result_payload (Dict[str, object]): Mutable snapshot-level payload that accumulates diagnostics for the current engine evaluation.
            trade (dict | None): Trade payload produced by the signal engine, or `None` when no executable trade qualified.
            capture_signal_evaluation (bool): Whether this snapshot should be considered for the signal-evaluation dataset.
            signal_capture_policy (str): Capture policy that decides which signal states are eligible for persistence.
        
        Returns:
            None: Sink implementations persist signal-evaluation rows or update capture status in place.
        
        Notes:
            The contract is intentionally side-effect oriented so runtime orchestration can swap sink implementations without changing engine logic.
        """
        ...


class ShadowEvaluationSink(Protocol):
    """
    Purpose:
        Protocol for runtime sinks that rerun a snapshot under a shadow parameter pack and compare the result with the baseline pack.
    
    Context:
        Used within the `runtime sinks` module. The class helps the runtime bridge engine snapshots into operator-visible sinks or research artifacts.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        Concrete implementations are expected to preserve this behavioral contract so runtime orchestration remains swappable.
    """
    def apply(
        self,
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
        backtest_mode: bool,
        target_profit_percent: float,
        stop_loss_percent: float,
        evaluate_snapshot_for_pack: Callable[..., dict],
    ) -> None:
        """
        Purpose:
            Run the current snapshot under a shadow parameter pack and compare the result with the baseline pack.
        
        Context:
            Method on `ShadowEvaluationSink` that forms part of the runtime sink contract implemented by this module.
        
        Inputs:
            result_payload (Dict[str, object]): Mutable snapshot-level payload that accumulates diagnostics for the current engine evaluation.
            shadow_pack_name (Optional[str]): Candidate parameter-pack name being evaluated in shadow mode.
            symbol (str): Underlying symbol or index identifier for the snapshot under evaluation.
            mode (str): Execution mode label such as live, replay, or backtest.
            source (str): Market-data source label associated with the snapshot.
            spot (float): Current underlying spot price.
            option_chain (pd.DataFrame): Option-chain snapshot passed through to the shadow evaluation.
            previous_chain (Optional[pd.DataFrame]): Previous option-chain snapshot used for change-sensitive features during shadow evaluation.
            day_high (Any): Session high used as part of the intraday context passed into the engine.
            day_low (Any): Session low used as part of the intraday context passed into the engine.
            day_open (Any): Session open used as part of the intraday context passed into the engine.
            prev_close (Any): Previous session close used as a contextual anchor for the engine.
            lookback_avg_range_pct (Any): Historical average range percentage used to normalize intraday movement.
            spot_validation (dict): Validation summary for the spot snapshot.
            option_chain_validation (dict): Validation summary for the option-chain snapshot.
            apply_budget_constraint (bool): Whether the engine should enforce capital-budget rules during trade construction.
            requested_lots (int): Requested lot count before any optimizer or capital cap adjusts position size.
            lot_size (int): Lot size used when converting premium into capital requirements.
            max_capital (float): Maximum capital budget available for the evaluation.
            macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
            headline_state (Any): Headline-risk state produced by the news ingestion layer.
            global_market_snapshot (dict): Cross-asset market snapshot used by the global-risk overlay.
            holding_profile (str): Holding intent that determines whether overnight rules should apply.
            spot_timestamp (Any): Timestamp associated with the spot snapshot.
            baseline_pack_name (str): Baseline or production parameter-pack name used as the shadow comparison reference.
            enable_shadow_logging (bool): Whether the shadow comparison should be appended to the shadow log.
            backtest_mode (bool): Whether the evaluation should use backtest-specific signal thresholds.
            target_profit_percent (float): Exit-model target percentage applied during both baseline and shadow evaluation.
            stop_loss_percent (float): Exit-model stop-loss percentage applied during both baseline and shadow evaluation.
            evaluate_snapshot_for_pack (Callable[..., dict]): Callback that reruns the current snapshot under a specified parameter pack.
        
        Returns:
            None: Sink implementations compare baseline and shadow-pack outputs through side effects.
        
        Notes:
            The contract is intentionally side-effect oriented so runtime orchestration can swap sink implementations without changing engine logic.
        """
        ...


class DefaultSignalCaptureSink:
    """
    Purpose:
        Default runtime sink that writes qualifying signal snapshots into the signal-evaluation dataset.
    
    Context:
        Used within the `runtime sinks` module. The class helps the runtime bridge engine snapshots into operator-visible sinks or research artifacts.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        The class groups behavior and state that need to stay explicit for maintainability and auditability.
    """
    def apply(
        self,
        *,
        result_payload: Dict[str, object],
        trade: dict | None,
        capture_signal_evaluation: bool,
        signal_capture_policy: str,
    ) -> None:
        """
        Purpose:
            Persist or skip the current signal snapshot according to the configured capture policy.
        
        Context:
            Method on `DefaultSignalCaptureSink` that forms part of the runtime sink contract implemented by this module.
        
        Inputs:
            result_payload (Dict[str, object]): Mutable snapshot-level payload that accumulates diagnostics for the current engine evaluation.
            trade (dict | None): Trade payload produced by the signal engine, or `None` when no executable trade qualified.
            capture_signal_evaluation (bool): Whether this snapshot should be considered for the signal-evaluation dataset.
            signal_capture_policy (str): Capture policy that decides which signal states are eligible for persistence.
        
        Returns:
            None: The sink updates capture status and optionally persists a signal-evaluation row.
        
        Notes:
            The contract is intentionally side-effect oriented so runtime orchestration can swap sink implementations without changing engine logic.
        """
        if capture_signal_evaluation and should_capture_signal(trade, signal_capture_policy):
            try:
                save_signal_evaluation(
                    result_payload,
                    as_of=(result_payload.get("spot_summary", {}) or {}).get("timestamp"),
                    return_frame=False,
                )
                result_payload["signal_capture_status"] = "CAPTURED"
            except Exception as exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"CRITICAL: Signal capture failed - research data loss: {exc}")
                result_payload["signal_capture_status"] = f"FAILED:{type(exc).__name__}"
                result_payload["signal_capture_error"] = str(exc)
                result_payload["signal_capture_failed"] = True  # NEW: explicit flag
        elif capture_signal_evaluation and trade:
            result_payload["signal_capture_status"] = f"SKIPPED_POLICY:{signal_capture_policy}"


class DefaultShadowEvaluationSink:
    """
    Purpose:
        Default runtime sink that evaluates a shadow parameter pack and optionally appends the comparison log.
    
    Context:
        Used within the `runtime sinks` module. The class helps the runtime bridge engine snapshots into operator-visible sinks or research artifacts.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        The class groups behavior and state that need to stay explicit for maintainability and auditability.
    """
    def apply(
        self,
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
        backtest_mode: bool,
        target_profit_percent: float,
        stop_loss_percent: float,
        evaluate_snapshot_for_pack: Callable[..., dict],
    ) -> None:
        """
        Purpose:
            Run the current snapshot under a shadow parameter pack and optionally append the comparison log.
        
        Context:
            Method on `DefaultShadowEvaluationSink` that forms part of the runtime sink contract implemented by this module.
        
        Inputs:
            result_payload (Dict[str, object]): Mutable snapshot-level payload that accumulates diagnostics for the current engine evaluation.
            shadow_pack_name (Optional[str]): Candidate parameter-pack name being evaluated in shadow mode.
            symbol (str): Underlying symbol or index identifier for the snapshot under evaluation.
            mode (str): Execution mode label such as live, replay, or backtest.
            source (str): Market-data source label associated with the snapshot.
            spot (float): Current underlying spot price.
            option_chain (pd.DataFrame): Option-chain snapshot passed through to the shadow evaluation.
            previous_chain (Optional[pd.DataFrame]): Previous option-chain snapshot used for change-sensitive features during shadow evaluation.
            day_high (Any): Session high used as part of the intraday context passed into the engine.
            day_low (Any): Session low used as part of the intraday context passed into the engine.
            day_open (Any): Session open used as part of the intraday context passed into the engine.
            prev_close (Any): Previous session close used as a contextual anchor for the engine.
            lookback_avg_range_pct (Any): Historical average range percentage used to normalize intraday movement.
            spot_validation (dict): Validation summary for the spot snapshot.
            option_chain_validation (dict): Validation summary for the option-chain snapshot.
            apply_budget_constraint (bool): Whether the engine should enforce capital-budget rules during trade construction.
            requested_lots (int): Requested lot count before any optimizer or capital cap adjusts position size.
            lot_size (int): Lot size used when converting premium into capital requirements.
            max_capital (float): Maximum capital budget available for the evaluation.
            macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
            headline_state (Any): Headline-risk state produced by the news ingestion layer.
            global_market_snapshot (dict): Cross-asset market snapshot used by the global-risk overlay.
            holding_profile (str): Holding intent that determines whether overnight rules should apply.
            spot_timestamp (Any): Timestamp associated with the spot snapshot.
            baseline_pack_name (str): Baseline or production parameter-pack name used as the shadow comparison reference.
            enable_shadow_logging (bool): Whether the shadow comparison should be appended to the shadow log.
            backtest_mode (bool): Whether the evaluation should use backtest-specific signal thresholds.
            target_profit_percent (float): Exit-model target percentage applied during both baseline and shadow evaluation.
            stop_loss_percent (float): Exit-model stop-loss percentage applied during both baseline and shadow evaluation.
            evaluate_snapshot_for_pack (Callable[..., dict]): Callback that reruns the current snapshot under a specified parameter pack.
        
        Returns:
            None: The sink updates shadow-comparison payload fields and optionally appends a shadow log.
        
        Notes:
            The contract is intentionally side-effect oriented so runtime orchestration can swap sink implementations without changing engine logic.
        """
        if not shadow_pack_name:
            return

        result_payload["shadow_mode_active"] = True
        result_payload["shadow_pack_name"] = shadow_pack_name
        result_payload["shadow_evaluation"] = None
        result_payload["shadow_comparison"] = None
        result_payload["shadow_log_status"] = "SKIPPED"

        try:
            shadow_eval = evaluate_snapshot_for_pack(
                parameter_pack_name=shadow_pack_name,
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
                backtest_mode=backtest_mode,
                target_profit_percent=target_profit_percent,
                stop_loss_percent=stop_loss_percent,
            )
        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Shadow evaluation failed for pack '{shadow_pack_name}': {exc}")
            result_payload["shadow_evaluation_failed"] = True
            result_payload["shadow_evaluation_error"] = str(exc)
            return

        try:
            shadow_payload = {
                "symbol": symbol,
                "mode": mode,
                "source": source,
                "spot_summary": result_payload["spot_summary"],
                "trade": shadow_eval["trade"],
                "execution_trade": (shadow_eval["trade"] or {}).get("execution_trade") if shadow_eval["trade"] else None,
                "trade_audit": (shadow_eval["trade"] or {}).get("trade_audit") if shadow_eval["trade"] else None,
                "macro_news_state": shadow_eval["macro_news_state"],
                "global_risk_state": shadow_eval["global_risk_state"],
            }
            shadow_comparison = compare_shadow_trade_outputs(
                result_payload,
                shadow_payload,
                baseline_pack_name=baseline_pack_name,
                shadow_pack_name=shadow_eval["parameter_pack_name"],
            )
        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Shadow comparison failed for pack '{shadow_pack_name}': {exc}")
            result_payload["shadow_comparison_failed"] = True
            result_payload["shadow_comparison_error"] = str(exc)
            return

        result_payload["shadow_mode_active"] = True
        result_payload["shadow_pack_name"] = shadow_eval["parameter_pack_name"]
        result_payload["shadow_evaluation"] = shadow_payload
        result_payload["shadow_comparison"] = shadow_comparison
        result_payload["shadow_log_status"] = "SKIPPED"
        if enable_shadow_logging:
            try:
                append_shadow_log(shadow_comparison)
                result_payload["shadow_log_status"] = "CAPTURED"
            except Exception as exc:
                result_payload["shadow_log_status"] = f"FAILED:{type(exc).__name__}"
                result_payload["shadow_log_error"] = str(exc)
