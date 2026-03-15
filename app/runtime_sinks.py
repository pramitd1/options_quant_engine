from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import pandas as pd

from research.signal_evaluation import save_signal_evaluation, should_capture_signal
from tuning.shadow import append_shadow_log, compare_shadow_trade_outputs


class SignalCaptureSink(Protocol):
    def apply(
        self,
        *,
        result_payload: Dict[str, object],
        trade: dict | None,
        capture_signal_evaluation: bool,
        signal_capture_policy: str,
    ) -> None:
        ...


class ShadowEvaluationSink(Protocol):
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
        evaluate_snapshot_for_pack: Callable[..., dict],
    ) -> None:
        ...


class DefaultSignalCaptureSink:
    def apply(
        self,
        *,
        result_payload: Dict[str, object],
        trade: dict | None,
        capture_signal_evaluation: bool,
        signal_capture_policy: str,
    ) -> None:
        if capture_signal_evaluation and should_capture_signal(trade, signal_capture_policy):
            try:
                save_signal_evaluation(
                    result_payload,
                    as_of=(result_payload.get("spot_summary", {}) or {}).get("timestamp"),
                    return_frame=False,
                )
                result_payload["signal_capture_status"] = "CAPTURED"
            except Exception as exc:
                result_payload["signal_capture_status"] = f"FAILED:{type(exc).__name__}"
                result_payload["signal_capture_error"] = str(exc)
        elif capture_signal_evaluation and trade:
            result_payload["signal_capture_status"] = f"SKIPPED_POLICY:{signal_capture_policy}"


class DefaultShadowEvaluationSink:
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
        evaluate_snapshot_for_pack: Callable[..., dict],
    ) -> None:
        if not shadow_pack_name:
            return

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
        )
        shadow_payload = {
            "symbol": symbol,
            "mode": mode,
            "source": source,
            "spot_summary": result_payload["spot_summary"],
            "trade": shadow_eval["trade"],
            "macro_news_state": shadow_eval["macro_news_state"],
            "global_risk_state": shadow_eval["global_risk_state"],
        }
        shadow_comparison = compare_shadow_trade_outputs(
            result_payload,
            shadow_payload,
            baseline_pack_name=baseline_pack_name,
            shadow_pack_name=shadow_eval["parameter_pack_name"],
        )
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
