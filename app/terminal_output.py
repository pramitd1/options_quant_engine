"""
Module: terminal_output.py

Purpose:
    Render engine results to the terminal with configurable verbosity.

Role in the System:
    Part of the application layer. Separates terminal display concerns from
    trading logic so the engine payload can be shown at COMPACT, STANDARD, or
    FULL_DEBUG detail levels.

Key Outputs:
    Formatted terminal output for each verbosity mode.

Downstream Usage:
    Called by main.py after each engine snapshot.
"""

from __future__ import annotations

import math
import json
import logging
import os
import re
from datetime import date
from pathlib import Path

from analytics.greeks_engine import compute_option_greeks, _parse_expiry_years
from analytics.signal_confidence import compute_signal_confidence
from engine.runtime_metadata import TRADER_VIEW_KEYS
from utils.consistency_checks import collect_trade_consistency_findings
from utils.regime_normalization import canonical_gamma_regime


def _resolve_next_expiry_from_candidates(expiry_candidates, current_expiry_date):
    """Find the next expiry after *current_expiry_date* from the API candidates list.

    Falls back to ``None`` if no candidate is later than *current_expiry_date*.
    """
    import pandas as _pd

    for raw in expiry_candidates:
        try:
            candidate_date = _pd.Timestamp(raw).date()
            if candidate_date > current_expiry_date:
                return candidate_date
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Overnight hold assessment — presentation-layer resolver
# ---------------------------------------------------------------------------

def resolve_overnight_hold_assessment(trade: dict) -> dict:
    """Classify overnight hold risk from the consolidated engine payload.

    This is a **read-only presentation helper**.  It does NOT modify the trade
    payload or change any trading logic.  It reads the already-consolidated
    overnight fields (``overnight_hold_allowed``, ``overnight_risk_penalty``,
    ``overnight_gap_risk_score``, etc.) and maps them to a human-readable
    decision structure for terminal / Streamlit display.

    Returns a dict with:
        overnight_hold_suggested  – "YES" | "HOLD_WITH_CAUTION" | "NO"
        overnight_hold_confidence – "HIGH" | "MODERATE" | "LOW"
        overnight_hold_reason     – str
        overnight_gap_risk_score  – float
        overnight_risk_penalty    – float
        overnight_constraints     – list[str]
        overnight_risk_summary    – str  (one-liner for COMPACT mode)
    """
    if not trade:
        return {
            "overnight_hold_suggested": "NO",
            "overnight_hold_confidence": "LOW",
            "overnight_hold_reason": "No trade payload",
            "overnight_gap_risk_score": 0,
            "overnight_risk_penalty": 0,
            "overnight_constraints": [],
            "overnight_risk_summary": "No trade signal — overnight hold not applicable",
        }

    hold_allowed = trade.get("overnight_hold_allowed", True)
    penalty = trade.get("overnight_risk_penalty", 0) or 0
    gap_score = trade.get("overnight_gap_risk_score", 0) or 0
    reason = trade.get("overnight_hold_reason") or ""
    trade_block = trade.get("overnight_trade_block", False)

    # ── Expiry-day guard: never suggest overnight hold on expiry day ─────
    is_expiry_day = False
    next_expiry = None
    selected_expiry = trade.get("selected_expiry")
    expiry_candidates = trade.get("expiry_candidates") or []
    if selected_expiry is not None:
        try:
            import pandas as _pd
            expiry_date = _pd.Timestamp(selected_expiry).date()
            if expiry_date == date.today():
                is_expiry_day = True
                next_expiry = _resolve_next_expiry_from_candidates(
                    expiry_candidates, expiry_date,
                )
        except Exception:
            pass

    # Collect per-layer constraint details
    constraints: list[str] = []
    if trade.get("overnight_convexity_risk"):
        constraints.append(f"Convexity risk: {trade['overnight_convexity_risk']}")
    if trade.get("overnight_convexity_penalty", 0):
        constraints.append(f"Convexity penalty: {trade['overnight_convexity_penalty']}")
    if trade.get("overnight_hedging_risk"):
        constraints.append(f"Hedging risk: {trade['overnight_hedging_risk']}")
    if trade.get("overnight_dealer_pressure_penalty", 0):
        constraints.append(f"Dealer pressure penalty: {trade['overnight_dealer_pressure_penalty']}")
    if trade.get("overnight_option_efficiency_penalty", 0):
        constraints.append(f"Option efficiency penalty: {trade['overnight_option_efficiency_penalty']}")

    # --- Classification logic ---
    if is_expiry_day:
        # Current expiry expires today — assess against next expiry instead
        if penalty >= 20 or trade_block or not hold_allowed:
            suggested = "NO"
            reason = reason or "High overnight risk even for next-expiry contracts"
        elif penalty >= 8 or gap_score >= 60:
            suggested = "HOLD_WITH_CAUTION"
            reason = reason or "Current expiry expires today — roll to next expiry for overnight"
        else:
            suggested = "HOLD_WITH_CAUTION"
            reason = reason or "Current expiry expires today — use next expiry for overnight hold"
    elif not hold_allowed or trade_block or penalty >= 20:
        suggested = "NO"
    elif penalty >= 8 or gap_score >= 60:
        suggested = "HOLD_WITH_CAUTION"
    else:
        suggested = "YES"

    # Confidence is based on how clear-cut the decision is
    if is_expiry_day and suggested == "NO":
        confidence = "HIGH"
    elif is_expiry_day:
        confidence = "MODERATE"  # next-expiry assessment has uncertainty
    elif suggested == "NO" and (penalty >= 25 or trade_block):
        confidence = "HIGH"
    elif suggested == "YES" and penalty <= 2 and gap_score <= 20:
        confidence = "HIGH"
    elif suggested == "HOLD_WITH_CAUTION":
        confidence = "MODERATE"
    else:
        confidence = "MODERATE"

    # Human-readable one-liner
    next_exp_str = next_expiry.strftime("%d-%b-%Y") if next_expiry else "next week"
    if is_expiry_day and suggested == "NO":
        summary = f"DO NOT HOLD — risk too high even for next expiry ({next_exp_str})"
    elif is_expiry_day:
        summary = f"EXPIRY DAY — roll to next expiry ({next_exp_str}) for overnight hold (penalty {penalty})"
    elif suggested == "NO":
        summary = f"DO NOT HOLD — {reason or 'overnight risk too high'} (penalty {penalty})"
    elif suggested == "HOLD_WITH_CAUTION":
        summary = f"HOLD WITH CAUTION — gap risk {gap_score}, penalty {penalty}"
    else:
        summary = f"OVERNIGHT HOLD OK — low risk (penalty {penalty})"

    result = {
        "overnight_hold_suggested": suggested,
        "overnight_hold_confidence": confidence,
        "overnight_hold_reason": reason,
        "overnight_gap_risk_score": gap_score,
        "overnight_risk_penalty": penalty,
        "overnight_constraints": constraints,
        "overnight_risk_summary": summary,
    }
    if is_expiry_day and next_expiry:
        result["next_expiry"] = next_expiry.strftime("%d-%b-%Y")
        result["is_expiry_day"] = True

        # Build an actionable roll suggestion when overnight hold is viable
        if suggested != "NO":
            strike = trade.get("strike")
            option_type = trade.get("option_type")
            direction = trade.get("direction")
            if strike and option_type:
                result["roll_strike"] = strike
                result["roll_option_type"] = option_type
                result["roll_direction"] = direction

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, max_items=8):
    """Format a value for terminal display."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        if len(value) <= max_items:
            return value
        return f"{value[:max_items]} ... (+{len(value) - max_items} more)"
    if isinstance(value, dict):
        items = list(value.items())
        preview = {k: v for k, v in items[:max_items]}
        if len(items) <= max_items:
            return preview
        return f"{preview} ... (+{len(items) - max_items} more)"
    return value


def _format_probability_display(probability):
    """Format probabilities without collapsing small nonzero values to `0%`."""
    if not isinstance(probability, (int, float)):
        return "-"
    if isinstance(probability, float) and (math.isnan(probability) or math.isinf(probability)):
        return "-"

    prob = float(probability)
    if prob <= 0:
        return "0%"
    if prob < 0.01:
        return "<1%"
    if prob < 0.10:
        return f"{prob:.1%}"
    return f"{prob:.0%}"


def _summarize_confidence_guards(guards):
    """Return a compact explanation when display confidence is being capped."""
    if not isinstance(guards, list) or not guards:
        return None

    normalized = {str(item).strip().lower() for item in guards if str(item).strip()}
    notes = []

    if "provider_health_weak" in normalized:
        notes.append("capped by weak provider health")
    elif "provider_health_caution" in normalized:
        notes.append("capped by caution provider health")

    if "status_watchlist_or_blocked" in normalized:
        notes.append("capped by blocked/watchlist status")
    if "data_quality_weak" in normalized:
        notes.append("capped by weak data quality")
    elif "data_quality_caution" in normalized:
        notes.append("capped by caution data quality")
    if "explicit_no_trade_reason" in normalized:
        notes.append("capped by explicit no-trade reason")
    if "direction_unresolved" in normalized:
        notes.append("capped by unresolved direction")
    if "confirmation_conflict_or_no_direction" in normalized:
        notes.append("capped by confirmation conflict")

    if not notes:
        return None

    deduped_notes = []
    seen = set()
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        deduped_notes.append(note)
    return "; ".join(deduped_notes)


def _resolve_move_sigma_points(trade):
    """Resolve a 1-sigma move estimate in points for breakout odds."""
    if not isinstance(trade, dict):
        return None

    expected_move = trade.get("expected_move_points")
    if isinstance(expected_move, (int, float)) and expected_move > 0:
        return float(expected_move)

    spot = trade.get("spot")
    atm_iv = trade.get("atm_iv")
    if not isinstance(spot, (int, float)) or spot <= 0:
        return None
    if not isinstance(atm_iv, (int, float)) or atm_iv <= 0:
        return None

    tte_years = _parse_expiry_years(trade.get("selected_expiry"))
    if tte_years is None:
        dte = trade.get("days_to_expiry")
        if isinstance(dte, (int, float)) and dte > 0:
            tte_years = float(dte) / 365.0

    if not isinstance(tte_years, (int, float)) or tte_years <= 0:
        return None

    iv_decimal = float(atm_iv) / 100.0
    sigma_points = float(spot) * iv_decimal * math.sqrt(float(tte_years))
    return sigma_points if sigma_points > 0 else None


def _compute_breakout_probability_rows(trade, thresholds=(100, 150, 200)):
    """Compute directional breakout probabilities for point thresholds.

    Returns:
        (rows, has_direction) where each row is (threshold_pts, up_prob, down_prob).
        When has_direction is False the split is symmetric (50/50); the caller
        should display a single "either direction" probability rather than
        separate up/down columns.
    """
    sigma_points = _resolve_move_sigma_points(trade)
    if not isinstance(sigma_points, (int, float)) or sigma_points <= 0:
        return [], False

    direction = str((trade or {}).get("direction") or "").upper().strip()
    has_direction = bool(direction and ("CALL" in direction or "PUT" in direction))
    is_call = "CALL" in direction

    if has_direction:
        # Derive the directional split from aligned flow/structural signals.
        # hybrid_move_probability measures omnidirectional move likelihood —
        # P(|ΔS| > k) — not P(move in the signaled direction), so it is NOT
        # used here to avoid a semantic mismatch.
        flow_signal = str((trade or {}).get("final_flow_signal") or "").upper()
        dealer_bias = str((trade or {}).get("dealer_hedging_bias") or "").upper()
        spot_vs_flip = str((trade or {}).get("spot_vs_flip") or "").upper()
        macro_regime = str((trade or {}).get("macro_regime") or "").upper()

        signal_bonus = 0.0
        if is_call:
            if "BULLISH" in flow_signal:
                signal_bonus += 0.07
            elif "BEARISH" in flow_signal:
                signal_bonus -= 0.05  # contra-flow reduces confidence
            if "UPSIDE" in dealer_bias:
                signal_bonus += 0.04
            if spot_vs_flip == "ABOVE_FLIP":
                signal_bonus += 0.03
            if macro_regime == "RISK_ON":
                signal_bonus += 0.03
        else:
            if "BEARISH" in flow_signal:
                signal_bonus += 0.07
            elif "BULLISH" in flow_signal:
                signal_bonus -= 0.05  # contra-flow reduces confidence
            if "DOWNSIDE" in dealer_bias:
                signal_bonus += 0.04
            if spot_vs_flip == "BELOW_FLIP":
                signal_bonus += 0.03
            if "RISK_OFF" in macro_regime:
                signal_bonus += 0.03

        # P(move in signal direction): base 0.50 + signal alignment bonus,
        # bounded conservatively so we never overstate directional certainty.
        signal_share = min(max(0.50 + signal_bonus, 0.35), 0.70)
        up_share = signal_share if is_call else 1.0 - signal_share
        down_share = 1.0 - up_share
    else:
        up_share = 0.5
        down_share = 0.5

    rows = []
    sqrt_two = math.sqrt(2.0)
    for threshold in thresholds:
        try:
            k = float(threshold)
        except (TypeError, ValueError):
            continue
        if k <= 0:
            continue

        z = k / sigma_points
        two_sided_tail = math.erfc(z / sqrt_two)
        two_sided_tail = max(0.0, min(1.0, two_sided_tail))

        up_prob = two_sided_tail * up_share
        down_prob = two_sided_tail * down_share
        rows.append((int(round(k)), up_prob, down_prob))

    return rows, has_direction


def _print_section(title, fields):
    """Print a titled key-value section."""
    print(f"\n{title}")
    print("---------------------------")
    for key, value in fields.items():
        if value is None:
            continue
        print(f"{key:26}: {_fmt(value)}")


def _collect_compact_consistency_checks(trade, *, call_oi=None, put_oi=None):
    """Return compact-mode consistency warnings assembled from displayed fields."""
    if not isinstance(trade, dict):
        return []

    findings = [
        str(item.get("message"))
        for item in collect_trade_consistency_findings(trade)
        if isinstance(item, dict) and item.get("message")
    ]

    oi_rows = list(call_oi or []) + list(put_oi or [])
    if oi_rows:
        try:
            all_flat = all(abs(float(row[2] or 0.0)) < 1e-9 for row in oi_rows if len(row) > 2)
        except Exception:
            all_flat = False

        # Tuple shape from _resolve_top_oi_levels:
        # (level, oi, chg_oi, inference, uses_snapshot_proxy, confidence, reason_code, ...)
        all_proxy_only = all(len(row) > 6 and str(row[6]).upper().strip() == "PROXY_ONLY" for row in oi_rows)
        all_oi_flat = all(len(row) > 3 and str(row[3]).upper().strip() == "OI_FLAT" for row in oi_rows)
        all_snapshot_proxy = all(len(row) > 4 and bool(row[4]) for row in oi_rows)

        if all_flat and (all_snapshot_proxy or (all_proxy_only and all_oi_flat)):
            findings.append(
                "all top OI deltas are flat proxy signals; directional OI inference confidence is limited"
            )

    return findings


def _build_trade_for_oi_inference(*, trade, result, spot_summary):
    """Build a trade context enriched with baseline frames for OI inference."""
    if not isinstance(trade, dict):
        return None

    enriched = dict(trade)
    if isinstance(spot_summary, dict):
        enriched["prev_close"] = spot_summary.get("prev_close")

    if isinstance(result, dict):
        enriched["previous_chain_frame"] = result.get("previous_chain_frame")
        enriched["premium_baseline_chain_frames"] = result.get("premium_baseline_chain_frames")
        enriched["premium_baseline_labels"] = result.get("premium_baseline_labels")
        enriched["premium_baseline_chain_frame"] = result.get("premium_baseline_chain_frame")
        enriched["zerodha_oi_baseline_chain_frame"] = result.get("zerodha_oi_baseline_chain_frame")

    return enriched


def _get_top_oi_levels_cached(*, result, trade_for_oi, option_chain_frame, top_n=5):
    """Resolve top OI levels once per snapshot and reuse across renderer sections."""
    if not isinstance(result, dict):
        return _resolve_top_oi_levels(trade_for_oi, option_chain_frame, top_n=top_n)

    cache = result.setdefault("_top_oi_levels_cache", {})
    cache_key = str(top_n)
    if cache_key in cache:
        return cache[cache_key]

    levels = _resolve_top_oi_levels(trade_for_oi, option_chain_frame, top_n=top_n)
    cache[cache_key] = levels
    return levels


def _resolve_top_liquidity_walls(trade, *, top_n=3, formatted=False, option_chain_frame=None, precomputed_oi_levels=None):
    """Resolve strongest top-N resistance/support walls from trade payload."""
    if not isinstance(trade, dict):
        return [], []

    spot = trade.get("spot")
    clusters = trade.get("gamma_clusters") or []
    liquidity_levels = trade.get("liquidity_levels") or []
    support = trade.get("support_wall")
    resistance = trade.get("resistance_wall")
    dlm = trade.get("dealer_liquidity_map") or {}

    strength_scores = {}

    def _add_strength(level, points):
        try:
            level_f = float(level)
        except (TypeError, ValueError):
            return
        strength_scores[level_f] = strength_scores.get(level_f, 0.0) + float(points)

    # Liquidity levels are already ordered by heatmap strength (highest first).
    liq_len = len(liquidity_levels)
    for idx, level in enumerate(liquidity_levels):
        _add_strength(level, max(liq_len - idx, 1) * 1.0)

    # Gamma clusters are structural magnets; earlier items are stronger.
    clu_len = len(clusters)
    for idx, level in enumerate(clusters):
        if isinstance(level, dict):
            level = level.get("strike")
        _add_strength(level, max(clu_len - idx, 1) * 0.8)

    # Direct walls get strong priors.
    _add_strength(support, 3.0)
    _add_strength(resistance, 3.0)
    _add_strength(dlm.get("next_support"), 2.5)
    _add_strength(dlm.get("next_resistance"), 2.5)

    wall_candidates = list(strength_scores.keys())

    support_walls = []  # (level, strength)
    resistance_walls = []  # (level, strength)
    spot_f = None
    try:
        if spot is not None:
            spot_f = float(spot)
    except (TypeError, ValueError):
        spot_f = None

    if spot_f is not None:
        support_walls = [(lvl, strength_scores.get(lvl, 0.0)) for lvl in wall_candidates if lvl <= spot_f]
        resistance_walls = [(lvl, strength_scores.get(lvl, 0.0)) for lvl in wall_candidates if lvl >= spot_f]

    # Rank by structural strength first, then by proximity to spot.
    if spot_f is not None:
        top_supports = [
            lvl for lvl, _score in sorted(
                support_walls,
                key=lambda item: (-item[1], abs(spot_f - item[0]), -item[0]),
            )[:top_n]
        ]
        top_resistances = [
            lvl for lvl, _score in sorted(
                resistance_walls,
                key=lambda item: (-item[1], abs(spot_f - item[0]), item[0]),
            )[:top_n]
        ]
    else:
        top_supports = []
        top_resistances = []

    # Symmetry fallback: always try to provide top_n supports and top_n resistances.
    # Priority: (1) OI strikes when chain is available, (2) nearest known strikes,
    # (3) synthetic nearest strikes using inferred strike step.
    if spot_f is not None and top_n > 0:
        top_supports = [float(x) for x in top_supports]
        top_resistances = [float(x) for x in top_resistances]

        def _append_unique(levels, candidate, *, side):
            try:
                c = float(candidate)
            except (TypeError, ValueError):
                return
            if side == "support" and c > spot_f:
                return
            if side == "resistance" and c < spot_f:
                return
            if c not in levels:
                levels.append(c)

        # 1) Fill from top OI strikes when option-chain frame is available.
        if len(top_supports) < top_n or len(top_resistances) < top_n:
            if precomputed_oi_levels is not None:
                oi_calls, oi_puts = precomputed_oi_levels
            else:
                oi_calls, oi_puts = _resolve_top_oi_levels(trade, option_chain_frame, top_n=max(2 * top_n, 6))
            for level, _oi, _chg_oi, _inf, _uses_snapshot_proxy, _confidence, _reason_code, _horizon_signature, _debug_payload in (oi_calls + oi_puts):
                if level <= spot_f:
                    _append_unique(top_supports, level, side="support")
                if level >= spot_f:
                    _append_unique(top_resistances, level, side="resistance")

        # 2) Fill from nearest known wall candidates on each side.
        if len(top_supports) < top_n or len(top_resistances) < top_n:
            _support_candidates = sorted(
                [lvl for lvl in wall_candidates if lvl <= spot_f and lvl not in top_supports],
                key=lambda lvl: abs(spot_f - lvl),
            )
            _resistance_candidates = sorted(
                [lvl for lvl in wall_candidates if lvl >= spot_f and lvl not in top_resistances],
                key=lambda lvl: abs(spot_f - lvl),
            )
            for lvl in _support_candidates:
                if len(top_supports) >= top_n:
                    break
                _append_unique(top_supports, lvl, side="support")
            for lvl in _resistance_candidates:
                if len(top_resistances) >= top_n:
                    break
                _append_unique(top_resistances, lvl, side="resistance")

        # 3) Synthetic nearest strikes if one side is still short.
        if len(top_supports) < top_n or len(top_resistances) < top_n:
            _all_levels = sorted(set(float(x) for x in wall_candidates))
            _diffs = [
                _all_levels[i + 1] - _all_levels[i]
                for i in range(len(_all_levels) - 1)
                if (_all_levels[i + 1] - _all_levels[i]) > 0
            ]
            strike_step = min(_diffs) if _diffs else 50.0
            if not isinstance(strike_step, (int, float)) or strike_step <= 0:
                strike_step = 50.0

            def _next_support_seed():
                seed = math.floor(spot_f / strike_step) * strike_step
                if seed > spot_f:
                    seed -= strike_step
                return float(seed)

            def _next_resistance_seed():
                seed = math.ceil(spot_f / strike_step) * strike_step
                if seed < spot_f:
                    seed += strike_step
                return float(seed)

            sup_seed = _next_support_seed()
            res_seed = _next_resistance_seed()
            sup_iter = 0
            res_iter = 0
            while len(top_supports) < top_n and sup_iter < 20:
                _append_unique(top_supports, sup_seed - (sup_iter * strike_step), side="support")
                sup_iter += 1
            while len(top_resistances) < top_n and res_iter < 20:
                _append_unique(top_resistances, res_seed + (res_iter * strike_step), side="resistance")
                res_iter += 1

        top_supports = top_supports[:top_n]
        top_resistances = top_resistances[:top_n]

    # Fallback when spot/context is missing.
    if not top_supports and support is not None:
        try:
            top_supports = [float(support)]
        except (TypeError, ValueError):
            top_supports = []
    if not top_resistances and resistance is not None:
        try:
            top_resistances = [float(resistance)]
        except (TypeError, ValueError):
            top_resistances = []

    if not formatted:
        return top_supports, top_resistances

    # Re-sort for display: nearest first (most immediately relevant to the trader).
    # Strength still drives which top-N are chosen; this only affects display order.
    if spot_f is not None:
        top_supports = sorted(top_supports, reverse=True)    # highest level = nearest support below spot
        top_resistances = sorted(top_resistances)            # lowest level  = nearest resistance above spot

    support_display = [
        _format_proximity(level, spot) if spot is not None else str(level)
        for level in top_supports
    ]
    resistance_display = [
        _format_proximity(level, spot) if spot is not None else str(level)
        for level in top_resistances
    ]
    return support_display, resistance_display


def _format_open_interest_value(value):
    """Format open-interest values into compact human-readable units."""
    try:
        oi = float(value)
    except (TypeError, ValueError):
        return "-"
    if oi >= 1_000_000:
        return f"{oi / 1_000_000:.2f}M"
    if oi >= 1_000:
        return f"{oi / 1_000:.1f}K"
    return f"{oi:.0f}"


def _format_oi_change_value(value, *, uses_snapshot_proxy: bool = False):
    """Format change-in-OI values with sign and compact units."""
    try:
        oi_change = float(value)
    except (TypeError, ValueError):
        return "-"

    sign = "+" if oi_change > 0 else ("-" if oi_change < 0 else "")
    magnitude = abs(oi_change)
    if magnitude >= 1_000_000:
        rendered = f"{sign}{magnitude / 1_000_000:.2f}M"
        return f"{rendered}*" if uses_snapshot_proxy else rendered
    if magnitude >= 1_000:
        rendered = f"{sign}{magnitude / 1_000:.1f}K"
        return f"{rendered}*" if uses_snapshot_proxy else rendered
    rendered = f"{sign}{magnitude:.0f}"
    return f"{rendered}*" if uses_snapshot_proxy else rendered


def _format_expected_move_display(*, spot, straddle_points, straddle_pct, expected_move_up, expected_move_down, model_pct):
    """Render expected move using internally consistent straddle/model figures."""
    parts = []

    try:
        spot_f = float(spot) if spot is not None else None
    except (TypeError, ValueError):
        spot_f = None

    try:
        straddle_points_f = float(straddle_points) if straddle_points is not None else None
    except (TypeError, ValueError):
        straddle_points_f = None

    try:
        straddle_pct_f = float(straddle_pct) if straddle_pct is not None else None
    except (TypeError, ValueError):
        straddle_pct_f = None

    try:
        model_pct_f = float(model_pct) if model_pct is not None else None
    except (TypeError, ValueError):
        model_pct_f = None

    if straddle_points_f is not None:
        straddle_part = f"straddle ±{straddle_points_f:.0f} pts"
        if expected_move_down is not None and expected_move_up is not None:
            try:
                straddle_part += f"  [{float(expected_move_down):.0f} - {float(expected_move_up):.0f}]"
            except (TypeError, ValueError):
                pass
        if straddle_pct_f is not None:
            straddle_part += f"  ({straddle_pct_f:.2f}%)"
        parts.append(straddle_part)

    if model_pct_f is not None:
        if spot_f is not None:
            model_points = spot_f * model_pct_f / 100.0
            model_down = spot_f - model_points
            model_up = spot_f + model_points
            model_part = f"model ±{model_points:.0f} pts  [{model_down:.0f} - {model_up:.0f}]  ({model_pct_f:.2f}%)"
        else:
            model_part = f"model {model_pct_f:.2f}%"
        parts.append(model_part)

    if not parts:
        return None

    if len(parts) == 2 and straddle_pct_f is not None and model_pct_f is not None:
        if abs(straddle_pct_f - model_pct_f) < 0.30:
            return parts[0]

    return " | ".join(parts)


def _resolve_top_oi_levels(trade, option_chain_frame, *, top_n=5):
    """Resolve top-N CE/PE strikes with OI, OI change, and side inference.

    Inference priority:
    1) Use option premium delta against rolling 5m / 3m / 1m baselines.
    2) Fallback to previous snapshot premium delta when rolling baselines are unavailable.
    3) Fallback to underlying move vs prev_close when premium history is unavailable.
    """
    if option_chain_frame is None or not hasattr(option_chain_frame, "copy"):
        return [], []

    try:
        import pandas as pd
    except Exception:
        return [], []

    if not isinstance(option_chain_frame, pd.DataFrame) or option_chain_frame.empty:
        return [], []

    df = option_chain_frame.copy()

    def _pick_col(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    strike_col = _pick_col(["strike", "STRIKE_PR", "strikePrice", "strike_price"])
    oi_col = _pick_col(["open_interest", "OPEN_INT", "openInterest", "oi"])
    type_col = _pick_col(["option_type", "OPTION_TYP", "type", "instrument_type"])
    oi_change_col = _pick_col(["CHG_IN_OI", "changeinOI", "change_in_oi"])
    ltp_col = _pick_col(["last_price", "lastPrice", "ltp", "LAST_PRICE", "close", "close_price"])
    expiry_col = _pick_col(["selected_expiry", "EXPIRY_DT", "expiry", "expiry_date"])

    if not strike_col or not oi_col or not type_col:
        return [], []

    selected_expiry = trade.get("selected_expiry") if isinstance(trade, dict) else None
    if selected_expiry and expiry_col:
        try:
            sel = pd.Timestamp(selected_expiry).date()
            exp_dt = pd.to_datetime(df[expiry_col], errors="coerce").dt.date
            filtered = df[exp_dt == sel]
            if not filtered.empty:
                df = filtered
        except Exception:
            pass

    selected_cols = [strike_col, oi_col, type_col]
    if oi_change_col:
        selected_cols.append(oi_change_col)
    if ltp_col:
        selected_cols.append(ltp_col)
    slim = df[selected_cols].copy()
    slim[strike_col] = pd.to_numeric(slim[strike_col], errors="coerce")
    slim[oi_col] = pd.to_numeric(slim[oi_col], errors="coerce")
    if oi_change_col:
        slim[oi_change_col] = pd.to_numeric(slim[oi_change_col], errors="coerce").fillna(0.0)
    if ltp_col:
        slim[ltp_col] = pd.to_numeric(slim[ltp_col], errors="coerce")
    slim[type_col] = slim[type_col].astype(str).str.upper().str.strip()
    slim = slim.dropna(subset=[strike_col, oi_col])
    if slim.empty:
        return [], []

    source_name = ""
    if "source" in df.columns:
        try:
            source_name = str(df["source"].dropna().iloc[0]).upper().strip()
        except Exception:
            source_name = ""

    def _build_chain_lookup_maps(chain_frame):
        if chain_frame is None or not hasattr(chain_frame, "copy"):
            return None, None

        prev_df = chain_frame.copy()

        def _pick_prev_col(candidates):
            for col in candidates:
                if col in prev_df.columns:
                    return col
            return None

        prev_strike_col = _pick_prev_col(["strike", "STRIKE_PR", "strikePrice", "strike_price"])
        prev_oi_col = _pick_prev_col(["open_interest", "OPEN_INT", "openInterest", "oi"])
        prev_type_col = _pick_prev_col(["option_type", "OPTION_TYP", "type", "instrument_type"])
        prev_ltp_col = _pick_prev_col(["last_price", "lastPrice", "ltp", "LAST_PRICE", "close", "close_price"])
        prev_expiry_col = _pick_prev_col(["selected_expiry", "EXPIRY_DT", "expiry", "expiry_date"])

        if not prev_strike_col or not prev_oi_col or not prev_type_col:
            return None, None

        prev_selected_cols = [prev_strike_col, prev_oi_col, prev_type_col]
        if prev_ltp_col:
            prev_selected_cols.append(prev_ltp_col)
        if prev_expiry_col:
            prev_selected_cols.append(prev_expiry_col)

        prev_slim = prev_df[prev_selected_cols].copy()
        prev_slim[prev_strike_col] = pd.to_numeric(prev_slim[prev_strike_col], errors="coerce")
        prev_slim[prev_oi_col] = pd.to_numeric(prev_slim[prev_oi_col], errors="coerce")
        if prev_ltp_col:
            prev_slim[prev_ltp_col] = pd.to_numeric(prev_slim[prev_ltp_col], errors="coerce")
        prev_slim[prev_type_col] = prev_slim[prev_type_col].astype(str).str.upper().str.strip()
        prev_slim = prev_slim.dropna(subset=[prev_strike_col, prev_oi_col])
        if prev_slim.empty:
            return None, None

        prev_group_cols = [prev_strike_col, prev_type_col]
        if prev_expiry_col:
            prev_group_cols.append(prev_expiry_col)
        prev_agg_map = {prev_oi_col: "max"}
        if prev_ltp_col:
            prev_agg_map[prev_ltp_col] = "max"
        prev_slim = prev_slim.groupby(prev_group_cols, as_index=False).agg(prev_agg_map)

        if prev_expiry_col:
            prev_oi_map = {
                (float(r[prev_strike_col]), str(r[prev_type_col]), str(r[prev_expiry_col])): float(r[prev_oi_col])
                for _idx, r in prev_slim.iterrows()
            }
            prev_premium_map = {
                (float(r[prev_strike_col]), str(r[prev_type_col]), str(r[prev_expiry_col])): float(r[prev_ltp_col])
                for _idx, r in prev_slim.iterrows()
            } if prev_ltp_col else None
        else:
            prev_oi_map = {
                (float(r[prev_strike_col]), str(r[prev_type_col]), None): float(r[prev_oi_col])
                for _idx, r in prev_slim.iterrows()
            }
            prev_premium_map = {
                (float(r[prev_strike_col]), str(r[prev_type_col]), None): float(r[prev_ltp_col])
                for _idx, r in prev_slim.iterrows()
            } if prev_ltp_col else None

        return prev_oi_map, prev_premium_map

    premium_baseline_frames = {}
    if isinstance(trade, dict):
        raw_frames = trade.get("premium_baseline_chain_frames") or {}
        if isinstance(raw_frames, dict):
            for horizon_name, frame in raw_frames.items():
                if frame is not None:
                    premium_baseline_frames[str(horizon_name)] = frame
        legacy_premium_frame = trade.get("premium_baseline_chain_frame")
        if legacy_premium_frame is not None and "5m" not in premium_baseline_frames:
            premium_baseline_frames["5m"] = legacy_premium_frame

    if isinstance(option_chain_frame, pd.DataFrame):
        raw_frames = option_chain_frame.attrs.get("premium_baseline_chain_frames") or {}
        if isinstance(raw_frames, dict):
            for horizon_name, frame in raw_frames.items():
                if frame is not None and str(horizon_name) not in premium_baseline_frames:
                    premium_baseline_frames[str(horizon_name)] = frame
        legacy_premium_frame = option_chain_frame.attrs.get("premium_baseline_chain_frame")
        if legacy_premium_frame is not None and "5m" not in premium_baseline_frames:
            premium_baseline_frames["5m"] = legacy_premium_frame

    fallback_previous_chain = trade.get("previous_chain_frame") if isinstance(trade, dict) else None
    if fallback_previous_chain is None and isinstance(option_chain_frame, pd.DataFrame):
        fallback_previous_chain = option_chain_frame.attrs.get("previous_chain_frame")

    previous_oi_by_key = None

    native_oi_change_missing = False
    if source_name == "ZERODHA":
        if oi_change_col:
            try:
                native_oi_change = pd.to_numeric(slim[oi_change_col], errors="coerce").fillna(0.0)
                native_oi_change_missing = bool(native_oi_change.empty or (native_oi_change.abs() <= 1e-9).all())
            except Exception:
                native_oi_change_missing = True
        else:
            native_oi_change_missing = True

    if source_name == "ZERODHA" and native_oi_change_missing:
        zerodha_baseline = trade.get("zerodha_oi_baseline_chain_frame") if isinstance(trade, dict) else None
        if zerodha_baseline is None and isinstance(option_chain_frame, pd.DataFrame):
            zerodha_baseline = option_chain_frame.attrs.get("zerodha_oi_baseline_chain_frame")
        if zerodha_baseline is not None:
            previous_oi_by_key, _unused = _build_chain_lookup_maps(zerodha_baseline)
        elif fallback_previous_chain is not None:
            previous_oi_by_key, _unused = _build_chain_lookup_maps(fallback_previous_chain)

    premium_lookup_maps = {}
    for horizon_name in ("1m", "3m", "5m"):
        premium_frame = premium_baseline_frames.get(horizon_name)
        _unused_oi_map, premium_lookup_map = _build_chain_lookup_maps(premium_frame)
        if premium_lookup_map:
            premium_lookup_maps[horizon_name] = premium_lookup_map

    fallback_previous_premium_map = None
    if fallback_previous_chain is not None:
        _unused_oi_map, fallback_previous_premium_map = _build_chain_lookup_maps(fallback_previous_chain)

    agg_map = {oi_col: "max"}
    if oi_change_col:
        agg_map[oi_change_col] = "sum"
    if ltp_col:
        agg_map[ltp_col] = "max"
    slim = slim.groupby([strike_col, type_col], as_index=False).agg(agg_map)

    expiry_key_value = None
    if selected_expiry is not None:
        try:
            expiry_key_value = str(pd.Timestamp(selected_expiry).date())
        except Exception:
            expiry_key_value = str(selected_expiry)

    def _lookup_baseline_value(lookup_map, strike_value, option_value):
        if lookup_map is None:
            return None
        lookup_key = (float(strike_value), str(option_value), expiry_key_value)
        fallback_key = (float(strike_value), str(option_value), None)
        value = lookup_map.get(lookup_key)
        if value is None:
            value = lookup_map.get(fallback_key)
        return value

    def _derive_premium_change(row, lookup_map):
        if ltp_col is None or lookup_map is None:
            return None
        prev_premium = _lookup_baseline_value(lookup_map, row[strike_col], row[type_col])
        curr_premium = row.get(ltp_col)
        try:
            if prev_premium is None or curr_premium is None or pd.isna(curr_premium):
                return None
            return float(curr_premium) - float(prev_premium)
        except Exception:
            return None

    for horizon_name in ("1m", "3m", "5m"):
        slim[f"_premium_change_{horizon_name}"] = slim.apply(
            lambda row, lookup_map=premium_lookup_maps.get(horizon_name): _derive_premium_change(row, lookup_map),
            axis=1,
        )
    slim["_premium_change_prev"] = slim.apply(
        lambda row: _derive_premium_change(row, fallback_previous_premium_map),
        axis=1,
    )

    use_snapshot_oi_change = source_name == "ZERODHA" and native_oi_change_missing and previous_oi_by_key is not None

    if use_snapshot_oi_change:
        def _derive_snapshot_oi_change(row):
            prev_oi = _lookup_baseline_value(previous_oi_by_key, row[strike_col], row[type_col])
            if prev_oi is None:
                return None, False
            return float(row[oi_col]) - float(prev_oi), True

        derived_change = slim.apply(_derive_snapshot_oi_change, axis=1)
        derived_change_values = derived_change.apply(lambda item: item[0])
        derived_change_flags = derived_change.apply(lambda item: item[1])
        if oi_change_col:
            slim[oi_change_col] = derived_change_values
        else:
            oi_change_col = "_derived_changeinOI"
            slim[oi_change_col] = derived_change_values
        slim["_uses_snapshot_oi_proxy"] = derived_change_flags
    elif source_name == "ZERODHA":
        slim["_uses_snapshot_oi_proxy"] = False

    spot = trade.get("spot") if isinstance(trade, dict) else None
    try:
        spot_f = float(spot) if spot is not None else None
    except (TypeError, ValueError):
        spot_f = None

    prev_close = trade.get("prev_close") if isinstance(trade, dict) else None
    try:
        prev_close_f = float(prev_close) if prev_close is not None else None
    except (TypeError, ValueError):
        prev_close_f = None

    def _underlying_premium_proxy(option_type):
        if prev_close_f is None or spot_f is None:
            return False, False
        price_bias_up = spot_f > prev_close_f
        price_bias_down = spot_f < prev_close_f
        opt = str(option_type or "").upper()
        if opt in {"CE", "CALL", "C"}:
            return price_bias_up, price_bias_down
        return price_bias_down, price_bias_up

    def _primary_premium_change(row):
        for horizon_name in ("5m", "3m", "1m"):
            value = row.get(f"_premium_change_{horizon_name}")
            if value is not None and pd.notna(value):
                return value
        value = row.get("_premium_change_prev")
        if value is not None and pd.notna(value):
            return value
        return None

    def _infer_side(option_type, oi_change, premium_change):
        try:
            oi_chg_f = float(oi_change) if oi_change is not None else None
        except (TypeError, ValueError):
            oi_chg_f = None

        try:
            premium_chg_f = float(premium_change) if premium_change is not None else None
        except (TypeError, ValueError):
            premium_chg_f = None

        if oi_chg_f is None:
            return "OI_ONLY"

        premium_up_proxy = False
        premium_down_proxy = False

        if premium_chg_f is not None:
            eps = 1e-9
            premium_up_proxy = premium_chg_f > eps
            premium_down_proxy = premium_chg_f < -eps
        else:
            premium_up_proxy, premium_down_proxy = _underlying_premium_proxy(option_type)
            if not premium_up_proxy and not premium_down_proxy:
                return "OI_ONLY"

        if oi_chg_f > 0:
            if premium_up_proxy:
                return "BUY_BUILDUP"
            if premium_down_proxy:
                return "WRITE_BUILDUP"
            return "OPEN_BUILDUP"

        if oi_chg_f < 0:
            if premium_up_proxy:
                return "SHORT_COVERING"
            if premium_down_proxy:
                return "LONG_UNWIND"
            return "OI_UNWIND"

        return "OI_FLAT"

    def _premium_sign(raw_value):
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            value = None
        if value is None:
            return "0"
        if value > 1e-9:
            return "+"
        if value < -1e-9:
            return "-"
        return "0"

    def _inference_confidence(*, option_type, oi_value, oi_change, premium_changes, current_premium, uses_snapshot_oi_proxy):
        try:
            oi_f = float(oi_value) if oi_value is not None else None
        except (TypeError, ValueError):
            oi_f = None

        try:
            oi_chg_f = float(oi_change) if oi_change is not None else None
        except (TypeError, ValueError):
            oi_chg_f = None

        if oi_chg_f is None:
            return 0.0, "OI_ONLY"

        try:
            current_premium_f = float(current_premium) if current_premium is not None else None
        except (TypeError, ValueError):
            current_premium_f = None

        weights = {"1m": 0.22, "3m": 0.33, "5m": 0.45}
        horizon_components = {}
        weighted_direction = 0.0
        total_horizon_weight = 0.0
        available_horizons = 0

        premium_scale_den = max(abs(current_premium_f or 0.0), 10.0)
        for horizon_name, weight in weights.items():
            raw_value = premium_changes.get(horizon_name)
            try:
                premium_chg_f = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                premium_chg_f = None
            if premium_chg_f is None:
                horizon_components[horizon_name] = 0.0
                continue

            available_horizons += 1
            total_horizon_weight += weight
            normalized_strength = min(abs(premium_chg_f) / premium_scale_den, 1.0)
            signed_component = weight * normalized_strength * (1.0 if premium_chg_f > 0 else -1.0 if premium_chg_f < 0 else 0.0)
            horizon_components[horizon_name] = signed_component
            weighted_direction += signed_component

        fallback_prev_component = 0.0
        try:
            fallback_prev_change = float(premium_changes.get("prev")) if premium_changes.get("prev") is not None else None
        except (TypeError, ValueError):
            fallback_prev_change = None
        if available_horizons == 0 and fallback_prev_change is not None:
            fallback_prev_component = 0.28 * min(abs(fallback_prev_change) / premium_scale_den, 1.0)
            weighted_direction = fallback_prev_component * (1.0 if fallback_prev_change > 0 else -1.0 if fallback_prev_change < 0 else 0.0)

        oi_scale_den = max((abs(oi_f) * 0.08) if oi_f is not None else 0.0, 1000.0)
        oi_strength = min(abs(oi_chg_f) / oi_scale_den, 1.0)

        underlying_up, underlying_down = _underlying_premium_proxy(option_type)
        dominant_premium_up = weighted_direction > 1e-9
        dominant_premium_down = weighted_direction < -1e-9
        has_premium_signal = available_horizons > 0 or fallback_prev_change is not None

        agreement_boost = 0.0
        if has_premium_signal and ((dominant_premium_up and underlying_up) or (dominant_premium_down and underlying_down)):
            agreement_boost = 0.08
        elif has_premium_signal and ((dominant_premium_up and underlying_down) or (dominant_premium_down and underlying_up)):
            agreement_boost = -0.08

        premium_strength_score = min(abs(weighted_direction), 1.0)
        if available_horizons >= 2 and abs(weighted_direction) >= 0.25:
            reason_code = "PREMIUM_STRONG_AGREE" if agreement_boost > 0 else "PREMIUM_STRONG_CONFLICT" if agreement_boost < 0 else "PREMIUM_STRONG"
        elif has_premium_signal:
            reason_code = "PREMIUM_WEAK_AGREE" if agreement_boost > 0 else "PREMIUM_WEAK_CONFLICT" if agreement_boost < 0 else "PREMIUM_WEAK"
        else:
            reason_code = "PROXY_ONLY"

        snapshot_penalty = -0.05 if uses_snapshot_oi_proxy else 0.0
        base = 0.58 if available_horizons > 0 else 0.50 if fallback_prev_change is not None else 0.40
        confidence = base + (0.22 * oi_strength) + (0.24 * premium_strength_score) + agreement_boost + snapshot_penalty
        confidence = max(0.0, min(0.99, confidence))
        return round(confidence, 2), reason_code

    def _top_for(opt_labels):
        side = slim[slim[type_col].isin(opt_labels)].copy()
        if side.empty:
            return []
        if spot_f is not None:
            side["_dist"] = (side[strike_col] - spot_f).abs()
        else:
            side["_dist"] = 0.0
        side = side.sort_values(by=[oi_col, "_dist", strike_col], ascending=[False, True, True]).head(top_n)
        resolved_rows = []
        for _idx, r in side.iterrows():
            premium_changes = {
                "1m": r.get("_premium_change_1m"),
                "3m": r.get("_premium_change_3m"),
                "5m": r.get("_premium_change_5m"),
                "prev": r.get("_premium_change_prev"),
            }
            horizon_signature = (
                f"1m:{_premium_sign(premium_changes.get('1m'))} "
                f"3m:{_premium_sign(premium_changes.get('3m'))} "
                f"5m:{_premium_sign(premium_changes.get('5m'))}"
            )
            primary_premium_change = _primary_premium_change(r)
            inference = _infer_side(
                r[type_col],
                r.get(oi_change_col) if oi_change_col else None,
                primary_premium_change,
            )
            confidence_score, reason_code = _inference_confidence(
                option_type=r[type_col],
                oi_value=r.get(oi_col),
                oi_change=r.get(oi_change_col) if oi_change_col else None,
                premium_changes=premium_changes,
                current_premium=r.get(ltp_col) if ltp_col else None,
                uses_snapshot_oi_proxy=bool(r.get("_uses_snapshot_oi_proxy", False)),
            )
            resolved_rows.append(
                (
                    float(r[strike_col]),
                    float(r[oi_col]),
                    float(r[oi_change_col]) if oi_change_col and pd.notna(r.get(oi_change_col)) else None,
                    inference,
                    bool(r.get("_uses_snapshot_oi_proxy", False)),
                    confidence_score,
                    reason_code,
                    horizon_signature,
                    {
                        "premium_change_1m": premium_changes.get("1m"),
                        "premium_change_3m": premium_changes.get("3m"),
                        "premium_change_5m": premium_changes.get("5m"),
                        "premium_change_prev": premium_changes.get("prev"),
                        "primary_premium_change": primary_premium_change,
                    },
                )
            )
        return resolved_rows

    top_calls = _top_for({"CE", "CALL", "C"})
    top_puts = _top_for({"PE", "PUT", "P"})
    return top_calls, top_puts


def _resolve_top_oi_strikes(trade, option_chain_frame, *, top_n=5, formatted=True):
    """Resolve top-N CE/PE strikes by raw open interest, preference nearest on ties."""
    top_calls, top_puts = _resolve_top_oi_levels(trade, option_chain_frame, top_n=top_n)
    if not formatted:
        return [lvl for lvl, _oi, _chg, _inf, _proxy, _conf, _reason, _sig, _payload in top_calls], [lvl for lvl, _oi, _chg, _inf, _proxy, _conf, _reason, _sig, _payload in top_puts]

    spot = trade.get("spot") if isinstance(trade, dict) else None
    try:
        spot_f = float(spot) if spot is not None else None
    except (TypeError, ValueError):
        spot_f = None

    def _format(rows):
        out = []
        for lvl, oi, _chg_oi, inf, _uses_snapshot_proxy, confidence, reason_code, horizon_signature, _payload in rows:
            oi_str = _format_open_interest_value(oi)
            prox = _format_proximity(lvl, spot_f if spot_f is not None else spot)
            out.append(f"{prox}  (OI {oi_str}; {inf}; conf {confidence:.2f}; {reason_code}; {horizon_signature})")
        return out

    return _format(top_calls), _format(top_puts)


def _render_market_summary_levels_table(*, spot, resistances, supports, call_oi, put_oi, sort_mode="GROUPED"):
    """Render support/resistance and high-OI strikes as separate compact tables."""
    structural_rows = []
    oi_rows = []

    def _dist(level):
        try:
            lvl = float(level)
            s = float(spot)
        except (TypeError, ValueError):
            return "-", "-"
        d = lvl - s
        p = (d / s * 100.0) if s else 0.0
        return f"{d:+.1f}", f"{p:+.2f}%"

    for idx, lvl in enumerate(sorted([float(x) for x in resistances], reverse=False), start=1):
        d_pts, d_pct = _dist(lvl)
        structural_rows.append(("resistance", idx, f"{lvl:.0f}", d_pts, d_pct, "-", 0.0))

    for idx, lvl in enumerate(sorted([float(x) for x in supports], reverse=True), start=1):
        d_pts, d_pct = _dist(lvl)
        structural_rows.append(("support", idx, f"{lvl:.0f}", d_pts, d_pct, "-", 0.0))

    for idx, (lvl, oi, chg_oi, inference, uses_snapshot_proxy, confidence_score, reason_code, horizon_signature, _payload) in enumerate(call_oi, start=1):
        d_pts, d_pct = _dist(lvl)
        oi_rows.append(
            (
                "CALL",
                idx,
                f"{lvl:.0f}",
                d_pts,
                d_pct,
                _format_open_interest_value(oi),
                _format_oi_change_value(chg_oi, uses_snapshot_proxy=uses_snapshot_proxy),
                str(inference),
                float(confidence_score),
                str(reason_code),
                str(horizon_signature),
                bool(uses_snapshot_proxy),
                float(oi),
            )
        )

    for idx, (lvl, oi, chg_oi, inference, uses_snapshot_proxy, confidence_score, reason_code, horizon_signature, _payload) in enumerate(put_oi, start=1):
        d_pts, d_pct = _dist(lvl)
        oi_rows.append(
            (
                "PUT",
                idx,
                f"{lvl:.0f}",
                d_pts,
                d_pct,
                _format_open_interest_value(oi),
                _format_oi_change_value(chg_oi, uses_snapshot_proxy=uses_snapshot_proxy),
                str(inference),
                float(confidence_score),
                str(reason_code),
                str(horizon_signature),
                bool(uses_snapshot_proxy),
                float(oi),
            )
        )

    if not structural_rows and not oi_rows:
        return

    mode = str(sort_mode or "GROUPED").upper().strip()
    if mode == "NEAREST":
        try:
            spot_f = float(spot)
        except (TypeError, ValueError):
            spot_f = None
        if spot_f is not None:
            structural_rows = sorted(
                structural_rows,
                key=lambda row: abs(float(row[2]) - spot_f),
            )
            oi_rows = sorted(
                oi_rows,
                key=lambda row: (
                    abs(float(row[2]) - spot_f),
                    -float(row[11]),
                ),
            )

    if structural_rows:
        print("\n  SUPPORT / RESISTANCE")
        print("  kind       rank strike   dist_pts dist_%")
        for kind, rank, strike, dist_pts, dist_pct, _oi_str, _oi_num in structural_rows:
            print(f"  {kind:<10} {rank:>4} {strike:>7} {dist_pts:>9} {dist_pct:>8}")

    if oi_rows:
        print("\n  HIGHEST OI STRIKES")
        chg_oi_width = max(len("chg_oi"), *(len(str(row[6])) for row in oi_rows))
        reason_width = max(len("reason"), *(len(str(row[9])) for row in oi_rows))
        hz_width = max(len("horizons"), *(len(str(row[10])) for row in oi_rows))
        print(f"  kind       rank strike   dist_pts dist_%    oi {'chg_oi':>{chg_oi_width}} inference      conf {'reason':<{reason_width}} {'horizons':<{hz_width}}")
        used_snapshot_proxy = any(bool(row[11]) for row in oi_rows)
        for kind, rank, strike, dist_pts, dist_pct, oi_str, chg_oi_str, inference, confidence_score, reason_code, horizon_signature, _uses_snapshot_proxy, _oi_num in oi_rows:
            print(
                f"  {kind:<10} {rank:>4} {strike:>7} {dist_pts:>9} {dist_pct:>8} "
                f"{oi_str:>7} {chg_oi_str:>{chg_oi_width}} {inference:<13} {confidence_score:>5.2f} {reason_code:<{reason_width}} {horizon_signature:<{hz_width}}"
            )
        if used_snapshot_proxy:
            print("  note: * indicates Zerodha snapshot OI delta proxy (5m rolling baseline when available, prior snapshot otherwise); inference confidence decomposes 1m/3m/5m premium baselines with previous-snapshot and underlying-proxy fallback")
        else:
            print("  note: inference confidence decomposes 1m/3m/5m premium baselines, then falls back to previous-snapshot premium delta and underlying move vs prev_close proxy")


def _persist_oi_inference_artifact(*, result, trade, spot_summary, call_oi, put_oi, signal_capture_policy=None, capture_enabled=True):
    """Append multi-horizon OI inference rows to research artifacts for calibration backtests."""
    if not isinstance(result, dict) or not capture_enabled:
        return

    try:
        from research.signal_evaluation import should_capture_signal
    except Exception:
        should_capture_signal = None

    if callable(should_capture_signal) and not should_capture_signal(trade, signal_capture_policy):
        return

    if not call_oi and not put_oi:
        return

    symbol = str(result.get("symbol") or "")
    source = str(result.get("source") or "")
    mode = str(result.get("mode") or "")
    timestamp = spot_summary.get("timestamp") if isinstance(spot_summary, dict) else None
    spot = spot_summary.get("spot") if isinstance(spot_summary, dict) else None
    prev_close = spot_summary.get("prev_close") if isinstance(spot_summary, dict) else None

    as_of_date = None
    if timestamp is not None:
        try:
            import pandas as _pd
            as_of_date = _pd.Timestamp(timestamp).date()
        except Exception:
            as_of_date = None
    if as_of_date is None:
        as_of_date = date.today()

    artifact_dir = Path("research/artifacts/oi_inference") / str(as_of_date)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"multi_horizon_inference_pid{os.getpid()}.jsonl"
    artifact_path = artifact_dir / file_name

    rows = []
    for kind, rows_data in (("CALL", call_oi), ("PUT", put_oi)):
        for rank, payload in enumerate(rows_data, start=1):
            lvl, oi, chg_oi, inference, uses_snapshot_proxy, confidence_score, reason_code, horizon_signature, debug_payload = payload
            rows.append(
                {
                    "kind": kind,
                    "rank": rank,
                    "strike": lvl,
                    "oi": oi,
                    "chg_oi": chg_oi,
                    "inference": inference,
                    "confidence_score": confidence_score,
                    "reason_code": reason_code,
                    "horizon_signature": horizon_signature,
                    "uses_snapshot_oi_proxy": bool(uses_snapshot_proxy),
                    "debug": debug_payload,
                }
            )

    record = {
        "schema_version": 2,
        "timestamp": timestamp,
        "signal_capture_policy": signal_capture_policy,
        "symbol": symbol,
        "source": source,
        "mode": mode,
        "spot": spot,
        "prev_close": prev_close,
        "rows": rows,
    }

    try:
        with artifact_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logging.getLogger(__name__).warning("OI inference artifact persistence failed: %s", exc)
        return


def _first_present(mapping, keys, default=None):
    """Return the first present/non-null key from *mapping* among *keys*."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _resolve_regime_extras(trade):
    """Resolve optional regime-level extras from heterogeneous payload keys."""
    rr_value = _first_present(trade, ("rr_value", "risk_reversal", "risk_reversal_value"))
    rr_momentum = _first_present(trade, ("rr_momentum", "risk_reversal_momentum"))
    oi_velocity_score = _first_present(trade, ("oi_velocity_score", "velocity_score"))
    oi_velocity_regime = _first_present(trade, ("oi_velocity_regime", "velocity_regime"))
    return {
        "risk_reversal": rr_value,
        "rr_momentum": rr_momentum,
        "oi_velocity_score": oi_velocity_score,
        "oi_velocity_regime": oi_velocity_regime,
    }


def _get_regime_impact_note(trade):
    """Generate human-readable explanation of regime constraints on trade strength."""
    if not isinstance(trade, dict):
        return None

    spot_vs_flip = str(trade.get("spot_vs_flip") or "").upper()
    gamma_regime = canonical_gamma_regime(trade.get("gamma_regime"))
    vol_regime = str(trade.get("vol_surface_regime") or "").upper()

    notes = []

    # At-flip impact
    if spot_vs_flip == "AT_FLIP":
        gamma_impact = "uncertain" if gamma_regime == "NEGATIVE_GAMMA" else "mean-reversion support"
        notes.append(f"At-flip: dealer gamma {gamma_impact}; microstructure noise")
    elif spot_vs_flip == "BELOW_FLIP" and gamma_regime == "NEGATIVE_GAMMA":
        notes.append("Below-flip with negative gamma: pinning risk increases")

    # Volume regime impact
    if vol_regime == "HIGH_VOL" and spot_vs_flip == "AT_FLIP":
        notes.append("High vol + at-flip: edge uncertain; short holding time recommended")

    return " | ".join(notes) if notes else None


def _format_proximity(level, spot):
    """Format a market level with distance and direction to spot."""
    if level is None or spot is None:
        return str(level) if level else "-"
    try:
        level_f = float(level)
        spot_f = float(spot)
        distance_pts = level_f - spot_f
        distance_pct = (distance_pts / spot_f * 100.0) if spot_f != 0 else 0
        
        direction = "📍" if abs(distance_pts) < 1 else ("⬇️" if distance_pts < 0 else "⬆️")
        return f"{level_f:.0f} {direction} {distance_pts:+.1f}pts / {distance_pct:+.2f}%"
    except (TypeError, ValueError):
        return str(level)


def _format_spread(spread_pct):
    """Format bid-ask spread, handling NaN gracefully."""
    if spread_pct is None or (isinstance(spread_pct, float) and spread_pct != spread_pct):  # NaN check
        return "N/A (no live quotes; using mid)"
    try:
        spread_f = float(spread_pct)
        return f"{spread_f:.2f}%"
    except (TypeError, ValueError):
        return str(spread_pct)


def _get_flow_signal_icon(flow_signal):
    """Map flow signal labels to compact visual markers."""
    signal = str(flow_signal or "").upper()
    if not signal:
        return ""
    if "SMART_BUY" in signal:
        return "💰"
    if "SMART_SELL" in signal:
        return "💸"
    if "BULLISH" in signal or signal.endswith("_BUY"):
        return "🟢"
    if "BEARISH" in signal or signal.endswith("_SELL"):
        return "🔴"
    if "NEUTRAL" in signal:
        return "⚪"
    if "MIXED" in signal or "CHOP" in signal:
        return "🟡"
    return ""


def _annotate_trigger_with_distance(trigger, spot):
    """Append spot-relative distance to trigger text when it contains a price level."""
    if not trigger or spot in (None, ""):
        return trigger
    trigger_text = str(trigger)
    if "pts" in trigger_text or "%" in trigger_text:
        return trigger_text
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        return trigger_text
    if spot_f == 0:
        return trigger_text

    match = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?)", trigger_text)
    if not match:
        return trigger_text

    try:
        level_f = float(match.group(1))
    except (TypeError, ValueError):
        return trigger_text

    distance_pts = level_f - spot_f
    distance_pct = distance_pts / spot_f * 100.0
    return f"{trigger_text} [{distance_pts:+.1f}pts / {distance_pct:+.2f}%]"


def _split_potential_triggers(triggers):
    """Group potential triggers into state blockers and price-based triggers."""
    state_blockers = []
    price_triggers = []

    for trigger in triggers or []:
        text = str(trigger or "").strip()
        if not text:
            continue

        normalized = text.lower()
        has_price_context = any([
            "support" in normalized,
            "resistance" in normalized,
            "break below" in normalized,
            "break above" in normalized,
            "move below" in normalized,
            "move above" in normalized,
            "hold above" in normalized,
            "hold below" in normalized,
            bool(re.search(r"\[[^\]]*(pts|%)", text)),
        ])

        if has_price_context:
            price_triggers.append(text)
        else:
            state_blockers.append(text)

    return state_blockers, price_triggers


def _infer_expected_direction(trade):
    """Infer an expected directional bias from available trade metadata."""
    if not isinstance(trade, dict):
        return None

    direction = str(trade.get("direction") or "").upper().strip()
    if "CALL" in direction or direction == "CALL":
        return "UP"
    if "PUT" in direction or direction == "PUT":
        return "DOWN"

    flow_signal = str(trade.get("final_flow_signal") or "").upper().strip()
    if any(token in flow_signal for token in ("BULLISH", "BUY")):
        return "UP"
    if any(token in flow_signal for token in ("BEARISH", "SELL")):
        return "DOWN"

    dealer_bias = str(trade.get("dealer_hedging_bias") or "").upper().strip()
    if "UPSIDE" in dealer_bias:
        return "UP"
    if "DOWNSIDE" in dealer_bias:
        return "DOWN"

    return None


def _rewrite_gamma_flip_trigger(trigger, trade):
    """Render gamma-flip triggers with directional wording when context is available."""
    text = str(trigger or "").strip()
    if not text or "gamma flip" not in text.lower() or not isinstance(trade, dict):
        return text

    if "hold above gamma flip" in text.lower() or "reject below gamma flip" in text.lower() or "break back below gamma flip" in text.lower():
        return text

    spot = trade.get("spot")
    gamma_flip = trade.get("gamma_flip")
    try:
        spot_f = float(spot)
        gamma_flip_f = float(gamma_flip)
    except (TypeError, ValueError):
        return text

    expected_direction = _infer_expected_direction(trade)
    flip_str = f"{gamma_flip_f:.2f}"

    if expected_direction == "UP":
        if spot_f >= gamma_flip_f:
            return f"hold above gamma flip {flip_str} with confirmation"
        return f"reject below gamma flip {flip_str} and reclaim above with confirmation"

    if expected_direction == "DOWN" and spot_f >= gamma_flip_f:
        return f"break back below gamma flip {flip_str} with confirmation"

    return text


def _format_trigger_for_display(trigger, trade):
    """Apply trigger wording cleanup and distance annotations for compact output."""
    rewritten = _rewrite_gamma_flip_trigger(trigger, trade)
    spot = trade.get("spot") if isinstance(trade, dict) else None
    return _annotate_trigger_with_distance(rewritten, spot)


def _describe_effective_strength_gate(trade):
    """Build a human-readable threshold summary including confidence adjustment."""
    if not isinstance(trade, dict):
        return None

    effective = trade.get("min_trade_strength_threshold")
    if effective in (None, "", "N/A"):
        return None
    try:
        effective = int(float(effective))
    except (TypeError, ValueError):
        return None

    from config.signal_policy import get_trade_runtime_thresholds
    thresholds = get_trade_runtime_thresholds()
    relief = int(float(thresholds.get("high_confidence_strength_relief", 5)))
    surcharge = int(float(thresholds.get("low_confidence_strength_surcharge", 8)))

    dq = str(trade.get("data_quality_status") or "").upper()
    conf = str(trade.get("confirmation_status") or trade.get("confirmation") or "").upper()
    high_conf = dq == "GOOD" and conf in {"STRONG_CONFIRMATION", "CONFIRMED"}
    low_conf = dq == "WEAK" or conf in {"CONFLICT", "NO_DIRECTION"}

    confidence_note = "none"
    base_est = effective
    if high_conf:
        base_est = effective + relief
        confidence_note = f"confidence_relief(-{relief})"
    elif low_conf:
        base_est = max(0, effective - surcharge)
        confidence_note = f"confidence_surcharge(+{surcharge})"

    regime_adj = trade.get("regime_threshold_adjustments")
    if isinstance(regime_adj, list) and regime_adj:
        regime_note = ", ".join(str(x) for x in regime_adj)
    else:
        regime_note = "none"

    return f"{effective} (base~{base_est}; conf:{confidence_note}; regime:{regime_note})"


def _humanize_requirement_token(token):
    """Convert requirement codes into compact trader-readable labels."""
    raw = str(token or "").strip()
    if not raw:
        return None

    mapping = {
        "missing_directional_consensus": "directional consensus missing",
        "confirmation_filter_not_met": "confirmation filter not met",
        "missing_flow_confirmation": "flow confirmation missing",
        "pinning_structure_dampens_signal": "pinning structure dampens edge",
        "insufficient_trade_strength": "trade strength below threshold",
        "move_probability_not_high_enough": "move probability below conviction floor",
        "option_efficiency_unavailable": "option efficiency unavailable",
        "direction_confirmation_conflict": "direction and confirmation conflict",
    }
    lowered = raw.lower()
    if lowered in mapping:
        return mapping[lowered]
    return lowered.replace("_", " ")


def _build_directionality_diagnostics(trade, *, mode="standard"):
    """Build a concise directionality verdict with evidence and blockers."""
    if not isinstance(trade, dict):
        return {
            "verdict": "UNAVAILABLE",
            "verdict_reason": "trade payload unavailable",
            "direction": "-",
            "direction_source": "-",
            "confirmation": "-",
            "activation": None,
            "maturity": None,
            "flow_alignment": "UNKNOWN",
            "structure_context": "UNKNOWN",
            "evidence": [],
            "blockers": [],
        }

    direction = str(trade.get("direction") or "").upper().strip()
    confirmation = str(
        trade.get("confirmation_status")
        or trade.get("confirmation")
        or ""
    ).upper().strip()
    direction_source = str(trade.get("direction_source") or "UNKNOWN").strip()
    decision = str(trade.get("decision_classification") or "").upper().strip()

    activation = trade.get("setup_activation_score")
    maturity = trade.get("setup_maturity_score")
    directional_resolution_needed = bool(trade.get("directional_resolution_needed"))

    # Tightened trust gate: direction must clear minimum setup activation and
    # maturity floors, not just directional alignment.
    try:
        from config.signal_policy import get_activation_score_policy_config

        _acfg = get_activation_score_policy_config()
        _dead_floor = int(getattr(_acfg, "dead_inactive_threshold", 35))
    except Exception:
        _dead_floor = 35

    activation_floor = max(50, _dead_floor + 15)
    maturity_floor = 65
    if str(mode).lower() == "compact":
        # Compact mode is intentionally stricter to reduce false confidence.
        activation_floor += 5
        maturity_floor += 5

    try:
        activation_value = float(activation) if activation is not None else None
    except (TypeError, ValueError):
        activation_value = None
    try:
        maturity_value = float(maturity) if maturity is not None else None
    except (TypeError, ValueError):
        maturity_value = None

    activation_ok = activation_value is not None and activation_value >= activation_floor
    maturity_ok = maturity_value is not None and maturity_value >= maturity_floor

    flow_signal = str(trade.get("final_flow_signal") or "").upper().strip()
    smart_flow = str(trade.get("smart_money_flow") or "").upper().strip()
    if direction == "CALL":
        aligned = sum(1 for x in (flow_signal, smart_flow) if "BULLISH" in x)
        conflicted = sum(1 for x in (flow_signal, smart_flow) if "BEARISH" in x)
    elif direction == "PUT":
        aligned = sum(1 for x in (flow_signal, smart_flow) if "BEARISH" in x)
        conflicted = sum(1 for x in (flow_signal, smart_flow) if "BULLISH" in x)
    else:
        aligned = 0
        conflicted = 0

    if direction not in {"CALL", "PUT"}:
        flow_alignment = "NO_DIRECTION"
    elif aligned >= 2 and conflicted == 0:
        flow_alignment = "STRONG"
    elif aligned >= 1 and conflicted == 0:
        flow_alignment = "PARTIAL"
    elif conflicted > 0:
        flow_alignment = "CONFLICTED"
    else:
        flow_alignment = "WEAK"

    spot_vs_flip = str(trade.get("spot_vs_flip") or "UNKNOWN").upper().strip()
    dealer_bias = str(trade.get("dealer_hedging_bias") or "UNKNOWN").upper().strip()
    if "PINNING" in dealer_bias or spot_vs_flip == "AT_FLIP":
        structure_context = "CHOP_RISK"
    elif spot_vs_flip in {"ABOVE_FLIP", "BELOW_FLIP"} and "ACCELERATION" in str(trade.get("dealer_flow_state") or "").upper():
        structure_context = "TREND_SUPPORTIVE"
    else:
        structure_context = "MIXED"

    blockers = []
    for item in (trade.get("missing_signal_requirements") or []):
        human = _humanize_requirement_token(item)
        if human:
            blockers.append(human)
    no_trade_reason = str(trade.get("no_trade_reason") or "").strip()
    if no_trade_reason:
        blockers.append(no_trade_reason)

    evidence = []
    if direction in {"CALL", "PUT"}:
        evidence.append(f"direction selected: {direction}")
    if confirmation:
        evidence.append(f"confirmation: {confirmation}")
    if flow_alignment in {"STRONG", "PARTIAL"}:
        evidence.append(f"flow alignment: {flow_alignment.lower()}")
    if structure_context == "TREND_SUPPORTIVE":
        evidence.append("structure supports continuation")

    trusted = (
        direction in {"CALL", "PUT"}
        and confirmation in {"CONFIRMED", "STRONG_CONFIRMATION"}
        and activation_ok
        and maturity_ok
        and not directional_resolution_needed
        and flow_alignment in {"STRONG", "PARTIAL"}
        and decision not in {"DEAD_INACTIVE", "DIRECTIONALLY_AMBIGUOUS"}
    )

    if trusted:
        verdict = "TRUSTED"
        verdict_reason = "direction, confirmation, and flow are aligned"
    else:
        verdict = "UNRESOLVED"
        if direction not in {"CALL", "PUT"}:
            verdict_reason = "directional consensus not formed"
        elif confirmation in {"NO_DIRECTION", "CONFLICT"}:
            verdict_reason = f"confirmation is {confirmation.lower()}"
        elif flow_alignment == "CONFLICTED":
            verdict_reason = "flow and smart-money signals conflict"
        elif structure_context == "CHOP_RISK":
            verdict_reason = "at-flip/pinning structure increases chop risk"
        else:
            if not activation_ok or not maturity_ok:
                verdict_reason = "activation/maturity floors not met"
            else:
                verdict_reason = "execution prerequisites are incomplete"

    return {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "direction": direction if direction else "NONE",
        "direction_source": direction_source,
        "confirmation": confirmation if confirmation else "UNKNOWN",
        "activation": activation,
        "maturity": maturity,
        "flow_alignment": flow_alignment,
        "structure_context": structure_context,
        "activation_floor": activation_floor,
        "maturity_floor": maturity_floor,
        "activation_ok": activation_ok,
        "maturity_ok": maturity_ok,
        "decision_classification": decision,
        "directional_resolution_needed": directional_resolution_needed,
        "final_flow_signal": flow_signal,
        "smart_money_flow": smart_flow,
        "no_trade_reason_code": str(trade.get("no_trade_reason_code") or "").strip() or None,
        "missing_signal_requirements": list(trade.get("missing_signal_requirements") or []),
        "setup_upgrade_conditions": list(trade.get("setup_upgrade_conditions") or []),
        "likely_next_trigger": trade.get("likely_next_trigger"),
        "evidence": evidence[:4],
        "blockers": list(dict.fromkeys(blockers))[:5],
    }


def _render_directionality_diagnostics(trade, *, mode="standard"):
    """Render direction-trust diagnostics for every snapshot."""
    mode_key = str(mode or "standard").lower()
    diag = _build_directionality_diagnostics(trade, mode=mode_key)

    base_fields = {
        "direction_verdict": f"{diag['verdict']} ({diag['verdict_reason']})",
        "direction": diag.get("direction"),
        "direction_source": diag.get("direction_source"),
        "confirmation": diag.get("confirmation"),
        "activation_score": diag.get("activation"),
        "activation_floor": diag.get("activation_floor"),
        "maturity_score": diag.get("maturity"),
        "maturity_floor": diag.get("maturity_floor"),
    }

    if mode_key != "compact":
        base_fields["flow_alignment"] = diag.get("flow_alignment")
        base_fields["structure_context"] = diag.get("structure_context")

    if mode_key == "full_debug":
        base_fields["activation_gate_pass"] = diag.get("activation_ok")
        base_fields["maturity_gate_pass"] = diag.get("maturity_ok")
        base_fields["decision_classification"] = diag.get("decision_classification")
        base_fields["directional_resolution_needed"] = diag.get("directional_resolution_needed")
        base_fields["final_flow_signal"] = diag.get("final_flow_signal")
        base_fields["smart_money_flow"] = diag.get("smart_money_flow")
        base_fields["no_trade_reason_code"] = diag.get("no_trade_reason_code")

    _print_section("DIRECTIONALITY DIAGNOSTICS", base_fields)

    evidence = diag.get("evidence") or []
    blockers = diag.get("blockers") or []

    if mode_key == "compact":
        evidence = evidence[:2]
        blockers = blockers[:2]
    elif mode_key == "standard":
        evidence = evidence[:3]
        blockers = blockers[:3]

    if evidence:
        print("  trust_evidence:")
        for item in evidence:
            print(f"    • {item}")
    if blockers and diag.get("verdict") != "TRUSTED":
        print("  unresolved_reasons:")
        for item in blockers:
            print(f"    • {item}")

    if mode_key == "full_debug":
        missing = diag.get("missing_signal_requirements") or []
        upgrades = diag.get("setup_upgrade_conditions") or []
        next_trigger = diag.get("likely_next_trigger")
        if missing:
            print("  missing_signal_requirements_raw:")
            for item in missing:
                print(f"    • {item}")
        if upgrades:
            print("  setup_upgrade_conditions_raw:")
            for item in upgrades:
                print(f"    • {item}")
        if next_trigger:
            print(f"  likely_next_trigger_raw: {next_trigger}")


def _print_validation(title, validation):
    """Print a validation block with preferred key ordering."""
    print(f"\n{title}")
    print("---------------------------")
    preferred = [
        "validation_mode", "is_valid", "live_trading_valid",
        "replay_analysis_valid", "is_stale", "age_minutes",
        "issues", "warnings",
    ]
    printed = set()
    for key in preferred:
        if key in validation:
            print(f"{key:26}: {validation[key]}")
            printed.add(key)
    for key, value in validation.items():
        if key not in printed:
            print(f"{key:26}: {value}")


def _build_trade_triggers(trade):
    """Return a short, human-readable list of execution triggers."""
    if not isinstance(trade, dict):
        return []

    triggers = []
    direction = str(trade.get("direction") or "").upper().strip()
    flow_signal = str(trade.get("final_flow_signal") or "").upper().strip()
    spot_vs_flip = str(trade.get("spot_vs_flip") or "").upper().strip()
    dealer_bias = str(trade.get("dealer_hedging_bias") or "").upper().strip()
    confirmation = str(trade.get("confirmation_status") or "").upper().strip()
    macro_regime = str(trade.get("macro_regime") or "").upper().strip()
    move_prob = trade.get("hybrid_move_probability")

    if flow_signal == "BEARISH_FLOW" and direction == "PUT":
        triggers.append("Bearish flow aligned with PUT direction")
    elif flow_signal == "BULLISH_FLOW" and direction == "CALL":
        triggers.append("Bullish flow aligned with CALL direction")

    if spot_vs_flip == "BELOW_FLIP" and direction == "PUT":
        triggers.append("Spot trading below gamma flip supports downside setup")
    elif spot_vs_flip == "ABOVE_FLIP" and direction == "CALL":
        triggers.append("Spot trading above gamma flip supports upside setup")
    elif spot_vs_flip == "AT_FLIP":
        triggers.append("Spot holding near gamma flip keeps move sensitivity elevated")

    if dealer_bias == "DOWNSIDE_PINNING" and direction == "PUT":
        triggers.append("Dealer hedging bias favors downside pressure")
    elif dealer_bias == "UPSIDE_PINNING" and direction == "CALL":
        triggers.append("Dealer hedging bias favors upside pressure")
    elif dealer_bias in {"UPSIDE_HEDGING_ACCELERATION", "DOWNSIDE_HEDGING_ACCELERATION"}:
        triggers.append("Dealer hedging acceleration is aligned with the move")

    if confirmation == "STRONG_CONFIRMATION":
        triggers.append("Confirmation filter is fully aligned")
    elif confirmation == "CONFIRMED":
        triggers.append("Confirmation filter supports execution")

    if isinstance(move_prob, (int, float)):
        if move_prob >= 0.65:
            triggers.append(f"Move probability elevated at {move_prob:.0%}")
        elif move_prob >= 0.55:
            triggers.append(f"Move probability supportive at {move_prob:.0%}")

    if macro_regime == "RISK_OFF" and direction == "PUT":
        triggers.append("Risk-off macro regime supports PUT exposure")
    elif macro_regime == "RISK_ON" and direction == "CALL":
        triggers.append("Risk-on macro regime supports CALL exposure")

    seen = set()
    deduped = []
    for trigger in triggers:
        if trigger not in seen:
            deduped.append(trigger)
            seen.add(trigger)
    return deduped[:5]


def _dedupe_potential_triggers(triggers):
    """Deduplicate near-identical trigger lines while preserving order."""
    deduped = []
    seen = set()

    for trigger in triggers:
        text = str(trigger or "").strip()
        if not text:
            continue

        normalized = text.lower()
        normalized = normalized.replace("decisive move below", "break below")
        normalized = normalized.replace("decisive move above", "break above")
        normalized = normalized.replace("clean move away from", "move away from")
        normalized = " ".join(normalized.split())

        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(text)

    return deduped


# ---------------------------------------------------------------------------
# Ranked strikes rendering
# ---------------------------------------------------------------------------

_RANKED_CORE_COLS = [
    ("option_type", "type", ">4"),
    ("strike", "strike", ">8"),
    ("last_price", "ltp", ">10"),
    ("iv", "iv", ">8"),
    ("delta", "delta", ">7"),
    ("volume", "volume", ">10"),
    ("open_interest", "oi", ">10"),
    ("payoff_efficiency_score", "eff_scr", ">7"),
    ("score", "rank_pts", ">8"),
]

_RANKED_EXTENDED_COLS = [
    ("distance_from_spot_pts", "dist_pts", ">9"),
    ("distance_from_spot_pct", "dist_%", ">7"),
    ("ba_spread_pct", "sprd_%", ">7"),
    ("ba_spread_score", "sprd_sc", ">7"),
    ("premium_efficiency_score", "prem_eff", ">8"),
    ("strike_efficiency_score", "strk_eff", ">8"),
    ("enhanced_strike_score", "enh_scr", ">7"),
    ("tradable_intraday", "trd_day", ">7"),
]


def _render_ranked_strikes(candidates, expiry=None, *, extended=False,
                           direction=None):
    """Print the ranked strike candidates table."""
    if not candidates:
        if direction is None:
            print("\nRANKED STRIKES")
            print("---------------------------")
            print("  No ranked strikes — direction not yet determined.")
        return
    title = f"RANKED STRIKES ({expiry})" if expiry else "RANKED STRIKES"
    print(f"\n{title}")
    print("---------------------------")

    cols = list(_RANKED_CORE_COLS)
    if extended:
        cols += [c for c in _RANKED_EXTENDED_COLS if any(c[0] in row for row in candidates)]

    header = " ".join(f"{hdr:{fmt}}" for _, hdr, fmt in cols)
    print(header)

    for row in candidates[:5]:
        parts = []
        for key, _, fmt in cols:
            val = row.get(key, "-")
            if val in (None, ""):
                val = "-"
            elif isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    val = "-"
                else:
                    val = round(val, 2)
            if isinstance(val, bool):
                val = "Y" if val else "N"
            if key == "iv" and row.get("iv_is_proxy") and val != "-":
                val = f"{val}*"
            if key == "delta" and row.get("delta_is_proxy") and val != "-":
                val = f"{val}*"
            parts.append(f"{str(val):{fmt}}")
        print(" ".join(parts))

    if any(bool(row.get("iv_is_proxy")) for row in candidates[:5]):
        print("  * iv uses proxy fallback (neighbor interpolation or atm/moneyness model)")
    if any(bool(row.get("delta_is_proxy")) for row in candidates[:5]):
        print("  * delta uses proxy fallback (neighbor interpolation or moneyness model)")


# ---------------------------------------------------------------------------
# Strike efficiency rendering
# ---------------------------------------------------------------------------

def _render_strike_efficiency(trade, ranked):
    """Print the STRIKE EFFICIENCY block for the selected trade strike."""
    if not ranked:
        return
    selected_strike = trade.get("strike")
    # Find the candidate matching the trade strike
    best = None
    for c in ranked:
        if c.get("strike") == selected_strike:
            best = c
            break
    if best is None:
        best = ranked[0] if ranked else None
    if best is None:
        return

    eff = best.get("payoff_efficiency_score")
    if eff is None:
        return

    premium = best.get("last_price", 0)

    # Prefer the canonical expected move from option-efficiency layer so this
    # display matches engine scoring and research capture exactly.
    expected_move = trade.get("expected_move_points")
    try:
        expected_move = float(expected_move) if expected_move is not None else None
    except (TypeError, ValueError):
        expected_move = None

    # Fallback only if canonical value is unavailable.
    if expected_move is None:
        iv_val = trade.get("atm_iv") or best.get("iv") or 0.15
        # atm_iv and ranked-candidate iv are in percentage points (e.g. 15.8);
        # convert to decimal for Black-Scholes expected-move calculation.
        if isinstance(iv_val, (int, float)) and iv_val > 1.5:
            iv_val = iv_val / 100.0
        spot = trade.get("spot") or trade.get("entry_price") or 0
        dte = trade.get("days_to_expiry") or 1
        expected_move = float(spot) * float(iv_val) * math.sqrt(max(float(dte), 0.1) / 365.0) if spot else 0.0

    eff_ratio = f"{expected_move / premium:.2f}" if premium and premium > 0 else "-"

    strike_label = f"{best['strike']} {best.get('option_type', '')}"
    delta_val = best.get("delta")
    delta_str = f"{delta_val:.4f}" if delta_val is not None and delta_val == delta_val else "-"

    print(f"\nSTRIKE EFFICIENCY")
    print("---------------------------")
    print(f"{'best_strike':26}: {strike_label}")
    print(f"{'delta':26}: {delta_str}")
    print(f"{'efficiency_score':26}: {eff}")
    print(f"{'expected_move':26}: {round(expected_move, 2)} pts")
    print(f"{'premium_paid':26}: {premium}")
    print(f"{'efficiency_ratio':26}: {eff_ratio}")

    # Break-even daily spot move (pts/day to neutralise theta)
    try:
        _spot = float(trade.get("spot") or 0)
        _tte = _parse_expiry_years(trade.get("selected_expiry"))
        if _tte is None:
            _dte = float(trade.get("days_to_expiry") or 1)
            _tte = max(_dte, 0.5) / 365.0
        _otype = "CE" if str(best.get("option_type", "")).upper() in ("CE", "CALL") else "PE"
        _iv_be = best.get("iv")
        _strike_be = best.get("strike")
        if _iv_be and float(_iv_be) > 0 and _strike_be and _spot > 0:
            _greeks_be = compute_option_greeks(
                spot=_spot,
                strike=float(_strike_be),
                time_to_expiry_years=_tte,
                volatility_pct=float(_iv_be),
                option_type=_otype,
            )
            _theta_be = _greeks_be.get("THETA") if _greeks_be else None
            _delta_abs_be = abs(float(delta_val)) if delta_val is not None else None
            if _theta_be and _delta_abs_be and _delta_abs_be > 1e-4:
                _be_pts = abs(float(_theta_be)) / _delta_abs_be
                print(f"{'break_even_pts/day':26}: {round(_be_pts, 2)}")
    except Exception:
        pass

    # Sub-component breakdown (compact)
    components = []
    for key, label in [
        ("pe_premium_eff", "prem_eff"),
        ("pe_delta_align", "delta_align"),
        ("pe_liquidity", "liq"),
        ("pe_dist_target", "dist_tgt"),
        ("pe_iv_eff", "iv_eff"),
    ]:
        val = best.get(key)
        if val is not None:
            components.append(f"{label}={val}")
    if components:
        print(f"{'breakdown':26}: {' | '.join(components)}")


# ---------------------------------------------------------------------------
# Dealer gamma levels rendering
# ---------------------------------------------------------------------------

def _render_dealer_gamma_levels(trade):
    """Print a compact DEALER GAMMA LEVELS block."""
    gamma_flip = trade.get("gamma_flip")
    clusters = trade.get("gamma_clusters") or []
    spot = trade.get("spot")
    dealer_flow_state = trade.get("dealer_flow_state")
    intraday_gamma_state = trade.get("intraday_gamma_state")
    vanna_regime = trade.get("vanna_regime")
    charm_regime = trade.get("charm_regime")
    gex = trade.get("gamma_exposure_greeks")

    # Nothing to show if all data is missing
    has_pressure = any([dealer_flow_state, intraday_gamma_state, vanna_regime, charm_regime, gex])
    if gamma_flip is None and not clusters and not has_pressure:
        return

    print(f"\nDEALER GAMMA LEVELS")
    print("---------------------------")

    if gamma_flip is not None:
        flip_display = _format_proximity(gamma_flip, spot) if spot else str(gamma_flip)
    else:
        flip_display = "UNAVAILABLE"
    print(f"{'gamma_flip':26}: {flip_display}")

    # Top 3 gamma magnet levels
    magnets = []
    for c in clusters:
        level = c if isinstance(c, (int, float)) else (c.get("strike") if isinstance(c, dict) else None)
        if level is not None and level not in magnets:
            magnets.append(level)
        if len(magnets) >= 3:
            break
    if magnets:
        print(f"\n  Magnets")
        for m in magnets:
            m_display = _format_proximity(m, spot) if spot else str(m)
            print(f"    {m_display}")

    # Dealer pressure sub-block: flow posture, intraday gamma dynamics, second-order Greek regimes.
    # These are not shown in REGIME SUMMARY and provide unique structural context for gamma trading.
    pressure_items = []
    if dealer_flow_state:
        pressure_items.append(("flow_state", dealer_flow_state))
    if gex is not None and gex != 0.0:
        gex_label = "NET_LONG_GAMMA" if gex > 0 else "NET_SHORT_GAMMA"
        pressure_items.append(("net_gex", f"{gex:+.4f}  ({gex_label})"))
    if intraday_gamma_state and intraday_gamma_state not in ("NEUTRAL", "NO_SHIFT"):
        pressure_items.append(("gamma_shift", intraday_gamma_state))
    # Gamma flip drift: direction and magnitude vs previous snapshot
    _flip_drift = trade.get("gamma_flip_drift") or {}
    if _flip_drift.get("drift_direction") and _flip_drift["drift_direction"] != "STABLE":
        _drift_pts = _flip_drift.get("drift", 0)
        pressure_items.append(("flip_drift", f"{_flip_drift['drift_direction']}  ({_drift_pts:+.0f} pts)"))
    if vanna_regime and vanna_regime not in ("UNKNOWN", "NEUTRAL_VANNA"):
        pressure_items.append(("vanna", vanna_regime))
    if charm_regime and charm_regime not in ("UNKNOWN", "NEUTRAL_CHARM"):
        pressure_items.append(("charm", charm_regime))
    if pressure_items:
        print(f"\n  Dealer Pressure")
        for label, value in pressure_items:
            print(f"    {label:<22}: {value}")


# ---------------------------------------------------------------------------
# Overnight hold assessment rendering
# ---------------------------------------------------------------------------

_SUGGESTED_ICONS = {"YES": "✅", "HOLD_WITH_CAUTION": "⚠️", "NO": "🛑"}


def _render_overnight_assessment(trade, *, verbose=False):
    """Print the OVERNIGHT HOLD ASSESSMENT block.

    *verbose* controls whether per-layer constraints are shown (STANDARD+).
    """
    assessment = resolve_overnight_hold_assessment(trade)
    suggested = assessment["overnight_hold_suggested"]
    icon = _SUGGESTED_ICONS.get(suggested, "")

    print(f"\nOVERNIGHT HOLD ASSESSMENT")
    print("---------------------------")
    print(f"{'hold_suggested':26}: {icon} {suggested}")
    print(f"{'confidence':26}: {assessment['overnight_hold_confidence']}")
    print(f"{'gap_risk_score':26}: {assessment['overnight_gap_risk_score']}")
    print(f"{'risk_penalty':26}: {assessment['overnight_risk_penalty']}")
    if assessment["overnight_hold_reason"]:
        print(f"{'reason':26}: {assessment['overnight_hold_reason']}")
    print(f"{'summary':26}: {assessment['overnight_risk_summary']}")

    # Show trade direction/strike/expiry context if available
    if trade:
        direction = trade.get("direction")
        strike = trade.get("strike")
        option_type = trade.get("option_type")
        expiry = trade.get("selected_expiry")
        if direction or strike:
            print(f"{'trade_context':26}: {direction or '-'} {strike or '-'} {option_type or '-'} (exp: {expiry or '-'})")

    # Show next-expiry guidance on expiry day
    if assessment.get("is_expiry_day") and assessment.get("next_expiry"):
        print(f"{'next_expiry':26}: {assessment['next_expiry']}")
        roll_strike = assessment.get("roll_strike")
        if roll_strike:
            roll_ot = assessment.get("roll_option_type", "")
            roll_dir = assessment.get("roll_direction", "")
            print(f"{'suggested_roll':26}: {roll_dir} {roll_strike} {roll_ot} — {assessment['next_expiry']} expiry")

    if verbose and assessment["overnight_constraints"]:
        print(f"{'constraints':26}:")
        for c in assessment["overnight_constraints"]:
            print(f"  • {c}")


# ---------------------------------------------------------------------------
# Signal confidence rendering
# ---------------------------------------------------------------------------

_CONFIDENCE_ICONS = {
    "VERY_HIGH": "🟢",
    "HIGH": "🔵",
    "MODERATE": "🟡",
    "LOW": "🟠",
    "UNRELIABLE": "🔴",
}


def _render_provider_health_compact_detail(trade):
    """Print compact provider-health diagnostics for degraded provider health."""
    if not isinstance(trade, dict):
        return

    provider_health = trade.get("provider_health")
    if not isinstance(provider_health, dict):
        return

    summary = str(provider_health.get("summary_status") or "").upper().strip()
    if summary not in {"CAUTION", "WEAK"}:
        return

    ocv = trade.get("option_chain_validation") if isinstance(trade.get("option_chain_validation"), dict) else {}
    source = provider_health.get("source") or "-"

    print("\n  Provider Health Detail")
    print(f"    source    : {source}")
    print(
        "    checks    : "
        f"row={provider_health.get('row_health', '-')}, "
        f"pricing={provider_health.get('pricing_health', '-')}, "
        f"pairing={provider_health.get('pairing_health', '-')}, "
        f"iv={provider_health.get('iv_health', '-')}, "
        f"duplicates={provider_health.get('duplicate_health', '-')}"
    )

    priced_rows = ocv.get("priced_rows")
    row_count = ocv.get("row_count")
    priced_ratio = ocv.get("priced_ratio")
    quoted_rows = ocv.get("quoted_rows")
    quoted_ratio = ocv.get("quoted_ratio")
    effective_rows = ocv.get("effective_priced_rows")
    effective_ratio = ocv.get("effective_priced_ratio")
    pricing_basis = provider_health.get("pricing_basis")
    quote_coverage_mode = provider_health.get("quote_coverage_mode")
    trade_price_health = provider_health.get("trade_price_health")
    quote_health = provider_health.get("quote_health")
    row_escalated = bool(provider_health.get("row_health_escalation_applied"))
    bid_present_rows = ocv.get("bid_present_rows")
    ask_present_rows = ocv.get("ask_present_rows")
    one_sided_quote_rows = ocv.get("one_sided_quote_rows")

    if trade_price_health or quote_health is not None:
        print(
            "    price sub : "
            f"trade={trade_price_health or '-'}, "
            f"quote={quote_health if quote_health is not None else 'N/A'}"
        )
    core_market = provider_health.get("core_marketability_health")
    core_pairing = provider_health.get("core_pairing_health")
    core_iv = provider_health.get("core_iv_health")
    core_quote_integrity = provider_health.get("core_quote_integrity_health")
    if any(v is not None for v in (core_market, core_pairing, core_iv, core_quote_integrity)):
        print(
            "    core      : "
            f"marketability={core_market or '-'}, "
            f"pairing={core_pairing or '-'}, "
            f"iv={core_iv or '-'}, "
            f"quote_integrity={core_quote_integrity or 'N/A'}"
        )
    block_status = provider_health.get("trade_blocking_status")
    if block_status:
        print(f"    block st  : {block_status}")
    block_reasons = provider_health.get("trade_blocking_reasons")
    if isinstance(block_reasons, list) and block_reasons:
        print(f"    block why : {', '.join(str(r) for r in block_reasons)}")
    non_critical_weak = provider_health.get("non_critical_weak_components")
    if isinstance(non_critical_weak, list) and non_critical_weak:
        print(f"    advisory  : weak non-critical -> {', '.join(str(r) for r in non_critical_weak)}")
    if pricing_basis:
        print(f"    basis     : {pricing_basis}")
    if quote_coverage_mode:
        print(f"    quote md  : {quote_coverage_mode}")
    if row_escalated:
        print("    note      : THIN row health escalated to CAUTION by policy")

    if priced_rows is not None and row_count is not None:
        print(f"    traded    : {priced_rows}/{row_count} rows")
    if quoted_rows is not None and row_count is not None and quote_health is not None:
        print(f"    quoted    : {quoted_rows}/{row_count} rows")
    if row_count is not None and isinstance(bid_present_rows, (int, float)) and isinstance(ask_present_rows, (int, float)):
        print(f"    bid/ask   : {int(bid_present_rows)}/{int(ask_present_rows)} rows")
    if isinstance(one_sided_quote_rows, (int, float)) and one_sided_quote_rows > 0:
        print(f"    one-sided : {int(one_sided_quote_rows)} rows")
    if effective_rows is not None and row_count is not None:
        print(f"    effective : {effective_rows}/{row_count} rows")

    print("    threshold : pricing GOOD >= 0.55, CAUTION >= 0.35")
    if isinstance(effective_ratio, (int, float)):
        print(f"    ratio     : {float(effective_ratio):.4f}")
    elif isinstance(priced_ratio, (int, float)):
        print(f"    ratio     : {float(priced_ratio):.4f}")
    if isinstance(quoted_ratio, (int, float)):
        print(f"    quote rt  : {float(quoted_ratio):.4f}")


def _render_signal_confidence(trade, *, show_components=True):
    """Print the SIGNAL CONFIDENCE block."""
    result = compute_signal_confidence(trade)
    icon = _CONFIDENCE_ICONS.get(result["confidence_level"], "")

    print(f"\nSIGNAL CONFIDENCE")
    print("---------------------------")
    print(f"{'confidence_score':26}: {icon} {result['confidence_score']}")
    print(f"{'confidence_level':26}: {result['confidence_level']}")

    if show_components:
        print(f"{'signal_strength':26}: {result['signal_strength_component']}")
        print(f"{'confirmation':26}: {result['confirmation_component']}")
        print(f"{'market_stability':26}: {result['market_stability_component']}")
        print(f"{'data_integrity':26}: {result['data_integrity_component']}")
        print(f"{'option_efficiency':26}: {result['option_efficiency_component']}")


# ---------------------------------------------------------------------------
# COMPACT mode — fast-execution trading dashboard
# ---------------------------------------------------------------------------

def render_compact(*, result, trade, spot_summary, macro_event_state,
                   global_risk_state, execution_trade, option_chain_frame=None,
                   market_levels_sort_mode="GROUPED"):
    """Render structured compact output following a logical trader workflow.

    Section order:
        1. MARKET SUMMARY
        2. REGIME SUMMARY
        3. TRADE DECISION
        4. TRADING SUGGESTION
        5. RANKED STRIKES
        6. RISK SUMMARY
    """
    display = execution_trade or trade
    trade_status = str((trade or {}).get("trade_status") or "").upper()
    has_trade = bool(trade and trade.get("direction") and trade_status == "TRADE")

    # ── 1. MARKET SUMMARY ────────────────────────────────────────────────
    spot = spot_summary.get("spot")
    # Build expected move display with straddle points, range, and optional
    # model delta when model-vs-straddle percentages diverge materially.
    _straddle = trade.get("atm_straddle_price") if trade else None
    _exp_pct = trade.get("expected_move_pct") if trade else None
    _exp_pct_model = trade.get("expected_move_pct_model") if trade else None
    _exp_up = trade.get("expected_move_up") if trade else None
    _exp_down = trade.get("expected_move_down") if trade else None
    _expected_move_str = _format_expected_move_display(
        spot=spot,
        straddle_points=_straddle,
        straddle_pct=_exp_pct,
        expected_move_up=_exp_up,
        expected_move_down=_exp_down,
        model_pct=_exp_pct_model,
    )
    # Max pain display with proximity zone annotation.
    _max_pain = trade.get("max_pain") if trade else None
    _max_pain_zone = trade.get("max_pain_zone") if trade else None
    _max_pain_str = None
    if _max_pain is not None:
        try:
            _max_pain_num = float(_max_pain)
            _max_pain_str = str(int(round(_max_pain_num))) if abs(_max_pain_num - round(_max_pain_num)) < 1e-6 else f"{_max_pain_num:.2f}"
        except Exception:
            _max_pain_str = str(_max_pain)
        if _max_pain_zone and _max_pain_zone not in ("UNAVAILABLE",):
            _zone_label = {"BELOW_SPOT": "below spot", "ABOVE_SPOT": "above spot", "AT_SPOT": "at spot"}.get(_max_pain_zone, "")
            if _zone_label:
                _max_pain_dist = trade.get("max_pain_dist")
                _dist_str = f"  ({abs(_max_pain_dist):.0f} pts {_zone_label})" if _max_pain_dist is not None else f"  ({_zone_label})"
                _max_pain_str += _dist_str

    _trade_with_prev_close = _build_trade_for_oi_inference(
        trade=trade,
        result=result,
        spot_summary=spot_summary,
    )
    _top_call_oi_levels, _top_put_oi_levels = _get_top_oi_levels_cached(
        result=result,
        trade_for_oi=_trade_with_prev_close,
        option_chain_frame=option_chain_frame,
        top_n=5,
    ) if trade else ([], [])

    _top_support_levels, _top_resistance_levels = _resolve_top_liquidity_walls(
        trade,
        top_n=3,
        formatted=False,
        option_chain_frame=option_chain_frame,
        precomputed_oi_levels=(_top_call_oi_levels, _top_put_oi_levels),
    ) if trade else ([], [])

    _print_section("MARKET SUMMARY", {
        "spot": spot,
        "day_range": (
            f"{spot_summary.get('day_low')} – {spot_summary.get('day_high')}"
            if spot_summary.get("day_low") is not None else None
        ),
        "prev_close": spot_summary.get("prev_close"),
        "max_pain": _max_pain_str,
        "expected_move": _expected_move_str,
        "atm_iv": trade.get("atm_iv") if trade else None,
        "macro_event_risk": macro_event_state.get("macro_event_risk_score"),
        "event_lockdown": macro_event_state.get("event_lockdown_flag"),
    })
    _render_market_summary_levels_table(
        spot=spot,
        resistances=_top_resistance_levels,
        supports=_top_support_levels,
        call_oi=_top_call_oi_levels,
        put_oi=_top_put_oi_levels,
        sort_mode=market_levels_sort_mode,
    )

    # ── 2. REGIME SUMMARY ────────────────────────────────────────────────
    if trade:
        regime_extras = _resolve_regime_extras(trade)

        # Add visual icons to gamma and flow regimes
        gamma_regime = canonical_gamma_regime(trade.get("gamma_regime"))
        gamma_icon = "🟢" if gamma_regime == "POSITIVE_GAMMA" else ("🔴" if gamma_regime == "NEGATIVE_GAMMA" else "")

        flow_signal = trade.get("final_flow_signal")
        flow_icon = _get_flow_signal_icon(flow_signal)

        vol_regime = trade.get("volatility_regime") or trade.get("vol_surface_regime")
        _vol_upper = str(vol_regime or "").upper()
        if _vol_upper in {"VOL_EXPANSION", "HIGH_VOL", "SHOCK_VOL", "VOLATILE"}:
            vol_icon = "📈"
        elif _vol_upper in {"VOL_CONTRACTION", "LOW_VOL", "COMPRESSED_VOL"}:
            vol_icon = "📉"
        else:
            vol_icon = ""

        # Volume PCR display: "1.42 (PUT_DOMINANT)"
        _vol_pcr_atm = trade.get("volume_pcr_atm")
        _vol_pcr_regime = trade.get("volume_pcr_regime")
        _vol_pcr_str = None
        if _vol_pcr_atm is not None:
            _vol_pcr_str = f"{_vol_pcr_atm:.2f}"
            if _vol_pcr_regime:
                _vol_pcr_str += f"  ({_vol_pcr_regime})"
        _print_section("REGIME SUMMARY", {
            "gamma_regime": f"{gamma_icon} {gamma_regime}".strip(),
            "spot_vs_flip": trade.get("spot_vs_flip"),
            "flow_signal": f"{flow_icon} {flow_signal}".strip(),
            "vol_regime": f"{vol_icon} {vol_regime}".strip(),
            "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
            "volume_pcr": _vol_pcr_str,
            "risk_reversal": regime_extras.get("risk_reversal"),
            "rr_momentum": regime_extras.get("rr_momentum"),
            "oi_velocity_score": regime_extras.get("oi_velocity_score"),
            "oi_velocity_regime": regime_extras.get("oi_velocity_regime"),
            "macro_regime": trade.get("macro_regime"),
            "global_risk": global_risk_state.get("global_risk_state"),
        })

    # ── 3. TRADE DECISION ────────────────────────────────────────────────
    if trade:
        confidence = compute_signal_confidence(trade)
        conf_icon = _CONFIDENCE_ICONS.get(confidence["confidence_level"], "")
        prob = trade.get("hybrid_move_probability")
        prob_str = _format_probability_display(prob)
        confidence_note = _summarize_confidence_guards(
            confidence.get("confidence_recalibration_guards")
        )
        
        decision_classification = trade.get("decision_classification")
        decision_dict = {
            "decision": decision_classification,
            "trade_strength": display.get("trade_strength") if display else None,
            "signal_quality": display.get("signal_quality") if display else None,
            "confirmation": trade.get("confirmation_status"),
            "move_probability": prob_str,
            "confidence": f"{conf_icon} {confidence['confidence_score']} ({confidence['confidence_level']})",
            "data_quality": trade.get("data_quality_status"),
        }
        
        _print_section("TRADE DECISION", decision_dict)
        if confidence_note:
            print(f"{'confidence_note':26}: {confidence_note}")

        _render_directionality_diagnostics(trade, mode="compact")
        
        # For blocked trades, add regime constraint explanation
        if "BLOCKED" in str(decision_classification):
            regime_note = _get_regime_impact_note(trade)
            if regime_note:
                print(f"  ⚠️  {regime_note}")
    else:
        print("\n  ▸ NO TRADE SIGNAL")

    # ── 3b. DEALER GAMMA LEVELS ──────────────────────────────────────────
    if trade:
        _render_dealer_gamma_levels(trade)

    # ── 4. TRADING SUGGESTION ────────────────────────────────────────────
    print(f"\nTRADING SUGGESTION")
    print("---------------------------")

    if has_trade:
        d = display or trade
        entry = d.get("entry_price")
        target = d.get("target")
        stop = d.get("stop_loss")

        rr = "-"
        try:
            reward = abs(float(target) - float(entry))
            risk = abs(float(entry) - float(stop))
            if risk > 0:
                rr = f"{reward / risk:.1f}"
        except (TypeError, ValueError):
            pass

        print(f"{'instrument':26}: {d.get('symbol')}")
        print(f"{'strike':26}: {d.get('strike')}")
        print(f"{'option_type':26}: {d.get('option_type')}")
        print(f"{'direction':26}: {d.get('direction')}")
        print(f"{'entry_price':26}: {entry}")
        print(f"{'target':26}: {target}")
        print(f"{'stop_loss':26}: {stop}")
        print(f"{'reward_to_risk':26}: {rr}")
        print(f"{'number_of_lots':26}: {d.get('number_of_lots')}")
        macro_lots = d.get("macro_suggested_lots")
        if macro_lots is not None and macro_lots != d.get("number_of_lots"):
            print(f"{'macro_adjusted_lots':26}: {macro_lots}")
        print(f"{'capital_required':26}: {d.get('capital_required')}")
        print(f"{'expiry':26}: {d.get('selected_expiry')}")
        print(f"{'execution_regime':26}: {d.get('execution_regime')}")
        _override_active = bool(d.get("provider_health_override_active") or trade.get("provider_health_override_active"))
        if _override_active:
            print(
                f"{'provider_override_mode':26}: "
                f"{d.get('provider_health_override_mode') or trade.get('provider_health_override_mode')}"
            )
            print(
                f"{'provider_override_reason':26}: "
                f"{d.get('provider_health_override_reason') or trade.get('provider_health_override_reason')}"
            )

        triggers = _build_trade_triggers(trade)
        if triggers:
            print(f"\n  Trade triggers:")
            for trigger in triggers:
                print(f"    • {trigger}")
    else:
        _reason_code = str((trade or {}).get("no_trade_reason_code") or "").upper()
        _reason_text = str((trade or {}).get("no_trade_reason") or "").strip()
        _blocked_by = {
            str(item).strip().lower()
            for item in ((trade or {}).get("blocked_by") or [])
            if item is not None
        }
        _confirmation = str(
            (trade or {}).get("confirmation_status")
            or (trade or {}).get("confirmation")
            or ""
        ).upper()
        _direction = (trade or {}).get("direction")

        if "provider_health" in _blocked_by or _reason_code.startswith("PROVIDER_HEALTH_"):
            print("  No trade yet. Provider health is blocking execution.")
        elif _confirmation in {"NO_DIRECTION", "CONFLICT"} or not _direction:
            print("  No trade yet. Waiting for directional confirmation.")
        elif _reason_text:
            _headline = _reason_text if _reason_text.endswith(".") else f"{_reason_text}."
            print(f"  No trade yet. {_headline}")
        else:
            print("  No trade yet. Conditions are not fully aligned.")

        # ── Threshold status ─────────────────────────────────────────────
        if trade:
            _raw_min_strength = trade.get("min_trade_strength_threshold")
            if _raw_min_strength in (None, "", "N/A"):
                from config.signal_policy import get_trade_runtime_thresholds
                _thresholds = get_trade_runtime_thresholds()
                _raw_min_strength = _thresholds.get("min_trade_strength", 60)
            try:
                _min_strength = int(float(_raw_min_strength))
            except (TypeError, ValueError):
                _min_strength = 60
            _raw_strength = trade.get("trade_strength")
            _has_strength = _raw_strength not in (None, "", "N/A")
            _cur_strength = int(_raw_strength) if _has_strength else 0
            _confirmation = str(
                trade.get("confirmation_status")
                or trade.get("confirmation")
                or ""
            ).upper()
            _no_direction = _confirmation == "NO_DIRECTION" or not trade.get("direction")

            print(f"\n  Trade Strength Threshold")
            _gap = _min_strength - _cur_strength
            _bar_filled = max(0, min(20, round(20 * _cur_strength / max(_min_strength, 1))))
            _bar = "█" * _bar_filled + "░" * (20 - _bar_filled)
            _gap_str = f"  ({_gap:+d} to threshold)" if _gap > 0 else "  ✓ threshold met"
            if _no_direction:
                _gap_str += " (direction pending)"
            print(f"    current  : {_cur_strength}")
            print(f"    required : {_min_strength}{_gap_str}")
            _gate_desc = _describe_effective_strength_gate(trade)
            if _gate_desc:
                print(f"    effective: {_gate_desc}")
            print(f"    progress : [{_bar}] {_cur_strength}/{_min_strength}")

        # ── Best strike candidate ────────────────────────────────────────
        if trade:
            _ranked = trade.get("ranked_strike_candidates") or []
            if _ranked:
                _best = _ranked[0]
                _b_strike = _best.get("strike", "-")
                _b_type = _best.get("option_type", "-")
                _b_score = _best.get("score")
                _b_ltp = _best.get("last_price", "-")
                _b_delta = _best.get("delta")
                _b_iv = _best.get("iv")
                _b_dir = trade.get("direction") or "-"
                print(f"\n  Best Strike Candidate  [{_b_dir}]")
                print(f"    strike   : {_b_strike} {_b_type}")
                print(f"    ltp      : {_b_ltp}")
                if _b_score is not None:
                    print(f"    score    : {round(float(_b_score), 2)}")
                if _b_delta is not None:
                    _delta_txt = f"{round(float(_b_delta), 4)}"
                    if _best.get("delta_is_proxy"):
                        _delta_txt += "*"
                    print(f"    delta    : {_delta_txt}")
                if _best.get("delta_is_proxy"):
                    print(f"    delta_src: {_best.get('delta_proxy_source') or 'PROXY'}")
                if _b_iv is not None:
                    _iv_txt = f"{round(float(_b_iv), 2)}"
                    if _best.get("iv_is_proxy"):
                        _iv_txt += "*"
                    print(f"    iv       : {_iv_txt}")
                if _best.get("iv_is_proxy"):
                    print(f"    iv_src   : {_best.get('iv_proxy_source') or 'PROXY'}")
                _b_spread = _best.get("ba_spread_pct")
                _b_spread_sc = _best.get("ba_spread_score")
                if _b_spread is not None:
                    spread_display = _format_spread(_b_spread)
                    print(f"    sprd_%   : {spread_display}")
                if _b_spread_sc is not None:
                    print(f"    sprd_sc  : {round(float(_b_spread_sc), 2)}")
                # Break-even daily spot move = |theta| / |delta|
                try:
                    _spot_for_be = float(trade.get("spot") or 0)
                    _dte = float(trade.get("days_to_expiry") or 1)
                    _tte = _parse_expiry_years(trade.get("selected_expiry"))
                    if _tte is None:
                        _tte = max(_dte, 0.5) / 365.0
                    _otype = "CE" if str(_b_type).upper() in ("CE", "CALL") else "PE"
                    if _b_iv and float(_b_iv) > 0 and _b_strike and _spot_for_be > 0:
                        _greeks = compute_option_greeks(
                            spot=_spot_for_be,
                            strike=float(_b_strike),
                            time_to_expiry_years=_tte,
                            volatility_pct=float(_b_iv),
                            option_type=_otype,
                        )
                        _theta = _greeks.get("THETA") if _greeks else None
                        _delta_abs = abs(float(_b_delta)) if _b_delta else None
                        if _theta and _delta_abs and _delta_abs > 1e-4:
                            _be = abs(float(_theta)) / _delta_abs
                            print(f"    be_pts/d : {round(_be, 2)}  (min daily move to cover theta)")
                except Exception:
                    pass

        # ── Potential trigger conditions ─────────────────────────────────
        triggers = []
        spot = trade.get("spot")
        if trade:
            _confirmation = str(
                trade.get("confirmation_status")
                or trade.get("confirmation")
                or ""
            ).upper()
            _direction_pending = _confirmation == "NO_DIRECTION" or not trade.get("direction")
            if _direction_pending:
                triggers.append(_format_trigger_for_display("Direction confirmation pending", trade))

            ntr = trade.get("no_trade_reason")
            if ntr:
                triggers.append(_format_trigger_for_display(ntr, trade))
            for cond in (trade.get("setup_upgrade_conditions") or []):
                triggers.append(_format_trigger_for_display(cond, trade))
            lt = trade.get("likely_next_trigger")
            if lt:
                triggers.append(_format_trigger_for_display(lt, trade))
            triggers = _dedupe_potential_triggers(triggers)
        if triggers:
            state_blockers, price_triggers = _split_potential_triggers(triggers)
            print(f"\n  Potential triggers:")
            if state_blockers:
                print("    State blockers:")
                for blocker in state_blockers:
                    print(f"      • {blocker}")
            if price_triggers:
                print("    Price triggers:")
                for price_trigger in price_triggers:
                    print(f"      • {price_trigger}")

        _status = str((trade or {}).get("trade_status") or "").upper()
        _show_provider_detail = _status in {
            "WATCHLIST",
            "NO_TRADE",
            "BLOCKED_SETUP",
            "DATA_INVALID",
            "GLOBAL_RISK_BLOCKED",
            "EVENT_LOCKDOWN",
        }
        if _show_provider_detail:
            _render_provider_health_compact_detail(trade)

    # ── 5. RANKED STRIKES ────────────────────────────────────────────────
    ranked = trade.get("ranked_strike_candidates") if trade else None
    _render_ranked_strikes(
        ranked,
        expiry=trade.get("selected_expiry") if trade else None,
        extended=False,
        direction=trade.get("direction") if trade else None,
    )

    if not trade:
        return

    # ── 5b. STRIKE EFFICIENCY ────────────────────────────────────────────
    if has_trade:
        _render_strike_efficiency(trade, ranked)

    # ── 6. RISK SUMMARY ─────────────────────────────────────────────────
    event_lock = macro_event_state.get("event_lockdown_flag")
    watchlist = trade.get("watchlist_flag")

    oe_status = trade.get("option_efficiency_status")
    oe_reason = trade.get("option_efficiency_reason")
    if oe_status and "UNAVAILABLE" in str(oe_status) and oe_reason:
        oe_display = f"{oe_status} ({oe_reason})"
    else:
        oe_display = oe_status

    # global_risk state label is already shown in REGIME SUMMARY; avoid duplication here.
    # Annotate the score when the RISK_OFF state is driven by macro regime rather than
    # the numeric score alone (score < 50 but state = RISK_OFF indicates macro override).
    _grs_score = global_risk_state.get("global_risk_score")
    _grs_state = global_risk_state.get("global_risk_state")
    if (
        _grs_score is not None
        and isinstance(_grs_score, (int, float))
        and _grs_state == "RISK_OFF"
        and _grs_score < 50
    ):
        _grs_display = f"{_grs_score} (macro-driven)"
    else:
        _grs_display = _grs_score

    risk_fields = {
        "global_risk_state_score": _grs_display,
        "event_lockdown": event_lock if event_lock else None,
        "watchlist": watchlist if watchlist else None,
        "macro_news_status": trade.get("macro_news_status"),
        "option_efficiency": oe_display,
    }
    if trade.get("provider_health_override_active"):
        _override_constraints = trade.get("provider_health_override_constraints") or []
        _override_constraints_text = ", ".join(str(item) for item in _override_constraints if item)
        _override_mode = trade.get("provider_health_override_mode") or "DEGRADED_PROVIDER_TRADE"
        if _override_constraints_text:
            risk_fields["degrade_mode"] = f"ACTIVE ({_override_mode}) [{_override_constraints_text}]"
        else:
            risk_fields["degrade_mode"] = f"ACTIVE ({_override_mode})"
    no_trade = trade.get("no_trade_reason")
    if no_trade:
        risk_fields["no_trade_reason"] = no_trade
    _print_section("RISK SUMMARY", risk_fields)

    # ── 6b. CONSISTENCY CHECK ───────────────────────────────────────────
    consistency_findings = _collect_compact_consistency_checks(
        trade,
        call_oi=_top_call_oi_levels,
        put_oi=_top_put_oi_levels,
    )
    _print_section(
        "CONSISTENCY CHECK",
        {
            "status": "WARN" if consistency_findings else "PASS",
            "issues": len(consistency_findings),
        },
    )
    if consistency_findings:
        for finding in consistency_findings:
            print(f"  • {finding}")

    # ── 7. BREAKOUT PROBABILITY (POINTS) ───────────────────────────────
    breakout_rows, breakout_has_direction = _compute_breakout_probability_rows(trade)
    if breakout_rows:
        print("\nBREAKOUT PROBABILITY (POINTS)")
        print("---------------------------")
        for threshold, up_prob, down_prob in breakout_rows:
            if breakout_has_direction:
                print(f"{f'+/- {threshold} pts':26}: up {up_prob:.0%} | down {down_prob:.0%}")
            else:
                print(f"{f'+/- {threshold} pts':26}: {up_prob + down_prob:.0%} either direction")

# ---------------------------------------------------------------------------
# STANDARD mode — compact + scoring/confirmation diagnostics
# ---------------------------------------------------------------------------

def render_standard(*, result, trade, spot_summary, spot_validation,
                    option_chain_validation, macro_event_state,
                    macro_news_state, global_risk_state, global_market_snapshot,
                    execution_trade, headline_state, option_chain_frame=None):
    """Render standard output: compact sections plus scoring and confirmation."""
    # Validations
    _print_validation("SPOT VALIDATION", spot_validation)

    _print_section("SPOT SNAPSHOT", {
        "spot": spot_summary.get("spot"),
        "day_open": spot_summary.get("day_open"),
        "day_high": spot_summary.get("day_high"),
        "day_low": spot_summary.get("day_low"),
        "prev_close": spot_summary.get("prev_close"),
        "timestamp": spot_summary.get("timestamp"),
    })

    _print_section("MACRO EVENT RISK", {
        "macro_event_risk_score": macro_event_state.get("macro_event_risk_score"),
        "event_window_status": macro_event_state.get("event_window_status"),
        "event_lockdown_flag": macro_event_state.get("event_lockdown_flag"),
        "minutes_to_next_event": macro_event_state.get("minutes_to_next_event"),
        "next_event_name": macro_event_state.get("next_event_name"),
    })

    _print_section("MACRO / NEWS REGIME", {
        "macro_regime": macro_news_state.get("macro_regime"),
        "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
        "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
        "news_confidence_score": macro_news_state.get("news_confidence_score"),
    })

    _print_section("GLOBAL RISK STATE", {
        "global_risk_state": global_risk_state.get("global_risk_state"),
        "global_risk_state_score": trade.get("global_risk_state_score") if trade else global_risk_state.get("global_risk_score"),
        "global_risk_overlay_score": trade.get("global_risk_overlay_score") if trade else None,
        "overnight_hold_allowed": global_risk_state.get("overnight_hold_allowed"),
    })

    _print_validation("OPTION CHAIN VALIDATION", option_chain_validation)
    print(f"\n{'option_chain_rows':26}: {result.get('option_chain_rows')}")

    if isinstance(trade, dict) and isinstance(result, dict) and "previous_chain_frame" in result:
        trade.setdefault("previous_chain_frame", result.get("previous_chain_frame"))
    if isinstance(trade, dict) and isinstance(result, dict) and "premium_baseline_chain_frames" in result:
        trade.setdefault("premium_baseline_chain_frames", result.get("premium_baseline_chain_frames"))
    if isinstance(trade, dict) and isinstance(result, dict) and "premium_baseline_labels" in result:
        trade.setdefault("premium_baseline_labels", result.get("premium_baseline_labels"))
    if isinstance(trade, dict) and isinstance(result, dict) and "premium_baseline_chain_frame" in result:
        trade.setdefault("premium_baseline_chain_frame", result.get("premium_baseline_chain_frame"))
    if isinstance(trade, dict) and isinstance(result, dict) and "zerodha_oi_baseline_chain_frame" in result:
        trade.setdefault("zerodha_oi_baseline_chain_frame", result.get("zerodha_oi_baseline_chain_frame"))

    if trade:
        _print_section("RUNTIME CONTEXT", {
            "parameter_pack_name": trade.get("parameter_pack_name"),
            "auto_pack_suggested": trade.get("auto_pack_suggested"),
            "effective_strength_gate": _describe_effective_strength_gate(trade),
        })

    if not trade:
        print("\n  ▸ NO TRADE SIGNAL")
        _print_section("ENGINE STATUS", {"message": "No trade payload returned"})
        return

    # Trader view
    display = execution_trade or trade
    print("\nTRADER VIEW")
    print("---------------------------")
    for key in TRADER_VIEW_KEYS:
        if key in display:
            print(f"{key:26}: {display[key]}")

    _exp_pct_straddle = trade.get("expected_move_pct")
    _exp_pct_model = trade.get("expected_move_pct_model")
    _exp_pct_div = None
    _top_call_oi, _top_put_oi = _resolve_top_oi_strikes(trade, option_chain_frame, top_n=5, formatted=True)
    if isinstance(_exp_pct_straddle, (int, float)) and isinstance(_exp_pct_model, (int, float)):
        _exp_pct_div = round(float(_exp_pct_model) - float(_exp_pct_straddle), 3)

    # Signal summary
    _print_section("QUANT TRADE SIGNAL", {
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "strike": trade.get("strike"),
        "option_type": trade.get("option_type"),
        "entry_price": trade.get("entry_price"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "decision_classification": trade.get("decision_classification"),
        "setup_quality": trade.get("setup_quality"),
        "hybrid_move_probability": trade.get("hybrid_move_probability"),
        "gamma_regime": trade.get("gamma_regime"),
        "spot_vs_flip": trade.get("spot_vs_flip"),
        "top_resistance_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[1],
        "top_support_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[0],
        "top_call_oi_strikes": _top_call_oi,
        "top_put_oi_strikes": _top_put_oi,
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "gamma_flip_drift": trade.get("gamma_flip_drift"),
        "max_pain": trade.get("max_pain"),
        "max_pain_dist": trade.get("max_pain_dist"),
        "max_pain_zone": trade.get("max_pain_zone"),
        "atm_straddle_price": trade.get("atm_straddle_price"),
        "expected_move_pct_straddle": _exp_pct_straddle,
        "expected_move_pct_model": _exp_pct_model,
        "expected_move_pct_divergence": _exp_pct_div,
        "volume_pcr": trade.get("volume_pcr"),
        "volume_pcr_atm": trade.get("volume_pcr_atm"),
        "volume_pcr_regime": trade.get("volume_pcr_regime"),
        "macro_regime": trade.get("macro_regime"),
        "global_risk_state": trade.get("global_risk_state"),
        "option_efficiency_score": trade.get("option_efficiency_score"),
        "premium_efficiency_score": trade.get("premium_efficiency_score"),
        "strike_efficiency_score": trade.get("strike_efficiency_score"),
        "capital_required": trade.get("capital_required"),
        "parameter_pack_name": trade.get("parameter_pack_name"),
        "effective_strength_gate": _describe_effective_strength_gate(trade),
        "slippage_bps": trade.get("slippage_bps"),
        "risk_reversal": _first_present(trade, ("rr_value", "risk_reversal", "risk_reversal_value")),
        "rr_momentum": _first_present(trade, ("rr_momentum", "risk_reversal_momentum")),
        "oi_velocity_score": _first_present(trade, ("oi_velocity_score", "velocity_score")),
        "oi_velocity_regime": _first_present(trade, ("oi_velocity_regime", "velocity_regime")),
    })

    # Explainability
    _print_section("EXPLAINABILITY", {
        "decision_classification": trade.get("decision_classification"),
        "no_trade_reason_code": trade.get("no_trade_reason_code"),
        "no_trade_reason": trade.get("no_trade_reason"),
        "missing_signal_requirements": trade.get("missing_signal_requirements"),
        "setup_upgrade_conditions": trade.get("setup_upgrade_conditions"),
        "likely_next_trigger": trade.get("likely_next_trigger"),
        "watchlist_flag": trade.get("watchlist_flag"),
        "watchlist_reason": trade.get("watchlist_reason"),
        "option_efficiency_status": trade.get("option_efficiency_status"),
        "option_efficiency_reason": trade.get("option_efficiency_reason"),
        "global_risk_status": trade.get("global_risk_status"),
        "global_risk_reason": trade.get("global_risk_reason"),
        "macro_news_status": trade.get("macro_news_status"),
        "macro_news_reason": trade.get("macro_news_reason"),
    })

    _render_directionality_diagnostics(trade, mode="standard")

    # Scoring breakdown
    scoring = trade.get("scoring_breakdown")
    if scoring:
        _print_section("SCORING BREAKDOWN", {
            k: _fmt(v) for k, v in scoring.items()
        })

    # Confirmation summary
    _print_section("CONFIRMATION SUMMARY", {
        "confirmation_status": trade.get("confirmation_status"),
        "confirmation_veto": trade.get("confirmation_veto"),
        "confirmation_reasons": trade.get("confirmation_reasons"),
    })

    # Signal confidence (full breakdown)
    _render_signal_confidence(trade, show_components=True)

    # Overnight hold assessment (with constraints)
    _render_overnight_assessment(trade, verbose=True)

    # Ranked strikes (extended columns)
    _render_ranked_strikes(
        trade.get("ranked_strike_candidates"),
        expiry=trade.get("selected_expiry"),
        extended=True,
        direction=trade.get("direction") if trade else None,
    )


# ---------------------------------------------------------------------------
# FULL_DEBUG mode — everything the engine produces
# ---------------------------------------------------------------------------

_DEALER_DASHBOARD_KEYS = [
    ("Spot Price", "spot"),
    ("Gamma Exposure", "gamma_exposure"),
    ("Market Gamma", "market_gamma"),
    ("Delta Exposure", "delta_exposure"),
    ("Greek Gamma Exp", "gamma_exposure_greeks"),
    ("Theta Exposure", "theta_exposure"),
    ("Vega Exposure", "vega_exposure"),
    ("Rho Exposure", "rho_exposure"),
    ("Vanna Exposure", "vanna_exposure"),
    ("Charm Exposure", "charm_exposure"),
    ("Greeks Data Warning", "greeks_data_warning"),
    ("Missing Greek Columns", "missing_greek_columns"),
    ("Gamma Flip Level", "gamma_flip"),
    ("Gamma Flip Drift", "gamma_flip_drift"),
    ("Spot vs Flip", "spot_vs_flip"),
    ("Gamma Regime", "gamma_regime"),
    ("Vanna Regime", "vanna_regime"),
    ("Charm Regime", "charm_regime"),
    ("Gamma Clusters", "gamma_clusters"),
    ("Dealer Inventory", "dealer_position"),
    ("Dealer Inv Basis", "dealer_inventory_basis"),
    ("Call OI Change", "call_oi_change"),
    ("Put OI Change", "put_oi_change"),
    ("Net OI Bias", "net_oi_change_bias"),
    ("Dealer Hedging Flow", "dealer_hedging_flow"),
    ("Dealer Hedging Bias", "dealer_hedging_bias"),
    ("Intraday Gamma State", "intraday_gamma_state"),
    ("Volatility Regime", "volatility_regime"),
    ("Vol Surface Regime", "vol_surface_regime"),
    ("ATM IV", "atm_iv"),
    ("ATM Straddle", "atm_straddle_price"),
    ("Expected Move %", "expected_move_pct"),
    ("Max Pain", "max_pain"),
    ("Max Pain Dist", "max_pain_dist"),
    ("Max Pain Zone", "max_pain_zone"),
    ("Volume PCR", "volume_pcr"),
    ("Volume PCR ATM", "volume_pcr_atm"),
    ("Volume PCR Regime", "volume_pcr_regime"),
    ("Flow Signal", "flow_signal"),
    ("Smart Money Flow", "smart_money_flow"),
    ("Final Flow Signal", "final_flow_signal"),
    ("Signal Regime", "signal_regime"),
    ("Execution Regime", "execution_regime"),
    ("Macro Event Risk", "macro_event_risk_score"),
    ("Event Window", "event_window_status"),
    ("Event Lockdown", "event_lockdown_flag"),
    ("Min To Next Event", "minutes_to_next_event"),
    ("Next Event", "next_event_name"),
    ("Macro Regime", "macro_regime"),
    ("Macro Sentiment", "macro_sentiment_score"),
    ("Vol Shock Score", "macro_news_volatility_shock_score"),
    ("India VIX Level", "india_vix_level"),
    ("India VIX Change 24h", "india_vix_change_24h"),
    ("News Confidence", "news_confidence_score"),
    ("Headline Velocity", "headline_velocity"),
    ("Macro Adj Score", "macro_adjustment_score"),
    ("Macro Size Mult", "macro_position_size_multiplier"),
    ("Macro Lots Hook", "macro_suggested_lots"),
    ("Gamma Event", "gamma_event"),
    ("Top Resistance Walls", "top_resistance_walls"),
    ("Top Support Walls", "top_support_walls"),
    ("Top Call OI Strikes", "top_call_oi_strikes"),
    ("Top Put OI Strikes", "top_put_oi_strikes"),
    ("Liquidity Levels", "liquidity_levels"),
    ("Liquidity Voids", "liquidity_voids"),
    ("Liquidity Void Signal", "liquidity_void_signal"),
    ("Liquidity Vacuum Zones", "liquidity_vacuum_zones"),
    ("Liquidity Vacuum State", "liquidity_vacuum_state"),
    ("Provider Health", "provider_health"),
]

_DIAGNOSTIC_KEYS = [
    "gamma_clusters",
    "liquidity_levels",
    "liquidity_voids",
    "liquidity_vacuum_zones",
    "dealer_liquidity_map",
    "gamma_flip_drift",
    "move_probability_components",
    "spot_validation",
    "option_chain_validation",
    "provider_health",
    "data_quality_reasons",
    "macro_adjustment_reasons",
    "confirmation_status",
    "confirmation_veto",
    "confirmation_reasons",
    "confirmation_breakdown",
    "global_risk_state_reasons",
    "global_risk_overlay_reasons",
    "global_risk_diagnostics",
    "global_risk_features",
    "gamma_vol_reasons",
    "gamma_vol_diagnostics",
    "gamma_vol_features",
    "dealer_pressure_reasons",
    "dealer_pressure_diagnostics",
    "dealer_pressure_features",
    "option_efficiency_reasons",
    "option_efficiency_diagnostics",
    "option_efficiency_features",
    "no_trade_reason_details",
    "blocked_by",
    "live_calibration_gate",
    "live_directional_gate",
    "greeks_data_warning",
    "missing_greek_columns",
    "missing_confirmations",
    "signal_promotion_requirements",
    "setup_upgrade_path",
    "watchlist_trigger_levels",
    "directional_resolution_needed",
    "explainability",
    "scoring_breakdown",
    "parameter_pack_name",
    "min_trade_strength_threshold",
    "regime_threshold_adjustments",
    "rr_value",
    "rr_momentum",
    "risk_reversal",
    "oi_velocity_score",
    "oi_velocity_regime",
    "velocity_score",
    "slippage_bps",
]


def _render_dealer_dashboard(summary):
    """Print the full dealer positioning dashboard."""
    print("\nDEALER POSITIONING DASHBOARD")
    print("--------------------------------------------------")
    for label, key in _DEALER_DASHBOARD_KEYS:
        print(f"{label:22}: {_fmt(summary.get(key))}")
    dealer_map = summary.get("dealer_liquidity_map")
    if dealer_map:
        print(f"{'Dealer Liquidity Map':22}: {_fmt(dealer_map)}")
    print(f"{'Large Move Probability':22}: {_fmt(summary.get('large_move_probability'))}")
    print(f"{'ML Move Probability':22}: {_fmt(summary.get('ml_move_probability'))}")
    print("--------------------------------------------------")
    scoring_breakdown = summary.get("scoring_breakdown")
    if scoring_breakdown:
        print("SCORING BREAKDOWN")
        print("--------------------------------------------------")
        for key, value in scoring_breakdown.items():
            print(f"{key:22}: {_fmt(value)}")
        print("--------------------------------------------------")


def _render_diagnostics(trade):
    """Print every diagnostic key available on the trade payload."""
    diagnostics = {}
    for key in _DIAGNOSTIC_KEYS:
        if key in trade:
            diagnostics[key] = _fmt(trade[key])
    if diagnostics:
        _print_section("DIAGNOSTICS", diagnostics)


def _render_full_signal_summary(trade, option_chain_frame=None):
    """Print the exhaustive quant trade signal block."""
    _exp_pct_straddle = trade.get("expected_move_pct")
    _exp_pct_model = trade.get("expected_move_pct_model")
    _exp_pct_div = None
    if isinstance(_exp_pct_straddle, (int, float)) and isinstance(_exp_pct_model, (int, float)):
        _exp_pct_div = round(float(_exp_pct_model) - float(_exp_pct_straddle), 3)

    compact = {
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "direction_source": trade.get("direction_source"),
        "selected_expiry": trade.get("selected_expiry"),
        "strike": trade.get("strike"),
        "option_type": trade.get("option_type"),
        "entry_price": trade.get("entry_price"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "decision_classification": trade.get("decision_classification"),
        "setup_state": trade.get("setup_state"),
        "setup_quality": trade.get("setup_quality"),
        "watchlist_flag": trade.get("watchlist_flag"),
        "watchlist_reason": trade.get("watchlist_reason"),
        "hybrid_move_probability": trade.get("hybrid_move_probability"),
        "flow_signal": trade.get("final_flow_signal"),
        "gamma_regime": trade.get("gamma_regime"),
        "spot_vs_flip": trade.get("spot_vs_flip"),
        "top_resistance_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[1],
        "top_support_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[0],
        "dealer_position": trade.get("dealer_position"),
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "macro_event_risk_score": trade.get("macro_event_risk_score"),
        "event_window_status": trade.get("event_window_status"),
        "event_lockdown_flag": trade.get("event_lockdown_flag"),
        "minutes_to_next_event": trade.get("minutes_to_next_event"),
        "next_event_name": trade.get("next_event_name"),
        "macro_regime": trade.get("macro_regime"),
        "macro_sentiment_score": trade.get("macro_sentiment_score"),
        "macro_news_volatility_shock_score": trade.get("macro_news_volatility_shock_score"),
        "news_confidence_score": trade.get("news_confidence_score"),
        "macro_adjustment_score": trade.get("macro_adjustment_score"),
        "macro_position_size_multiplier": trade.get("macro_position_size_multiplier"),
        "macro_suggested_lots": trade.get("macro_suggested_lots"),
        "global_risk_state": trade.get("global_risk_state"),
        "global_risk_state_score": trade.get("global_risk_state_score"),
        "global_risk_overlay_score": trade.get("global_risk_overlay_score", trade.get("global_risk_score")),
        "gamma_vol_acceleration_score": trade.get("gamma_vol_acceleration_score"),
        "squeeze_risk_state": trade.get("squeeze_risk_state"),
        "directional_convexity_state": trade.get("directional_convexity_state"),
        "dealer_hedging_pressure_score": trade.get("dealer_hedging_pressure_score"),
        "dealer_flow_state": trade.get("dealer_flow_state"),
        "expected_move_points": trade.get("expected_move_points"),
        "atm_straddle_price": trade.get("atm_straddle_price"),
        "expected_move_pct_straddle": _exp_pct_straddle,
        "expected_move_pct_model": _exp_pct_model,
        "expected_move_pct_divergence": _exp_pct_div,
        "expected_move_quality": trade.get("expected_move_quality"),
        "max_pain": trade.get("max_pain"),
        "max_pain_dist": trade.get("max_pain_dist"),
        "max_pain_zone": trade.get("max_pain_zone"),
        "volume_pcr": trade.get("volume_pcr"),
        "volume_pcr_atm": trade.get("volume_pcr_atm"),
        "volume_pcr_regime": trade.get("volume_pcr_regime"),
        "gamma_flip_drift": trade.get("gamma_flip_drift"),
        "target_reachability_score": trade.get("target_reachability_score"),
        "premium_efficiency_score": trade.get("premium_efficiency_score"),
        "strike_efficiency_score": trade.get("strike_efficiency_score"),
        "option_efficiency_score": trade.get("option_efficiency_score"),
        "overnight_gap_risk_score": trade.get("overnight_gap_risk_score"),
        "overnight_hold_allowed": trade.get("overnight_hold_allowed"),
        "overnight_hold_reason": trade.get("overnight_hold_reason"),
        "overnight_risk_penalty": trade.get("overnight_risk_penalty"),
        "atm_iv": trade.get("atm_iv"),
        "vol_surface_regime": trade.get("vol_surface_regime"),
        "capital_required": trade.get("capital_required"),
        "data_quality_score": trade.get("data_quality_score"),
        "data_quality_status": trade.get("data_quality_status"),
        "message": trade.get("message"),
        "parameter_pack_name": trade.get("parameter_pack_name"),
        "effective_strength_gate": _describe_effective_strength_gate(trade),
        "slippage_bps": trade.get("slippage_bps"),
        "risk_reversal": _first_present(trade, ("rr_value", "risk_reversal", "risk_reversal_value")),
        "rr_momentum": _first_present(trade, ("rr_momentum", "risk_reversal_momentum")),
        "oi_velocity_score": _first_present(trade, ("oi_velocity_score", "velocity_score")),
        "oi_velocity_regime": _first_present(trade, ("oi_velocity_regime", "velocity_regime")),
    }
    _print_section("QUANT TRADE SIGNAL", compact)

    explainability = {
        "decision_classification": trade.get("decision_classification"),
        "no_trade_reason_code": trade.get("no_trade_reason_code"),
        "no_trade_reason": trade.get("no_trade_reason"),
        "missing_signal_requirements": trade.get("missing_signal_requirements"),
        "setup_upgrade_conditions": trade.get("setup_upgrade_conditions"),
        "likely_next_trigger": trade.get("likely_next_trigger"),
        "watchlist_flag": trade.get("watchlist_flag"),
        "watchlist_reason": trade.get("watchlist_reason"),
        "option_efficiency_status": trade.get("option_efficiency_status"),
        "option_efficiency_reason": trade.get("option_efficiency_reason"),
        "global_risk_status": trade.get("global_risk_status"),
        "global_risk_reason": trade.get("global_risk_reason"),
        "macro_news_status": trade.get("macro_news_status"),
        "macro_news_reason": trade.get("macro_news_reason"),
    }
    _print_section("EXPLAINABILITY", explainability)


def render_full_debug(*, result, trade, spot_summary, spot_validation,
                      option_chain_validation, macro_event_state,
                      macro_news_state, global_risk_state,
                      global_market_snapshot, headline_state,
                      execution_trade, option_chain_frame=None):
    """Render full debug output — all sections, all diagnostics."""
    _print_validation("SPOT VALIDATION", spot_validation)

    _print_section("SPOT SNAPSHOT", {
        "spot": spot_summary.get("spot"),
        "day_open": spot_summary.get("day_open"),
        "day_high": spot_summary.get("day_high"),
        "day_low": spot_summary.get("day_low"),
        "prev_close": spot_summary.get("prev_close"),
        "timestamp": spot_summary.get("timestamp"),
        "lookback_avg_range_pct": spot_summary.get("lookback_avg_range_pct"),
    })

    _print_section("MACRO EVENT RISK", {
        "macro_event_risk_score": macro_event_state.get("macro_event_risk_score"),
        "event_window_status": macro_event_state.get("event_window_status"),
        "event_lockdown_flag": macro_event_state.get("event_lockdown_flag"),
        "minutes_to_next_event": macro_event_state.get("minutes_to_next_event"),
        "next_event_name": macro_event_state.get("next_event_name"),
    })

    _print_section("MACRO / NEWS REGIME", {
        "macro_regime": macro_news_state.get("macro_regime"),
        "macro_regime_reasons": macro_news_state.get("macro_regime_reasons"),
        "macro_event_risk_score": macro_news_state.get("macro_event_risk_score"),
        "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
        "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
        "event_lockdown_flag": macro_news_state.get("event_lockdown_flag"),
        "news_confidence_score": macro_news_state.get("news_confidence_score"),
        "headline_velocity": macro_news_state.get("headline_velocity"),
        "headline_count": macro_news_state.get("headline_count"),
        "classified_headline_count": macro_news_state.get("classified_headline_count"),
        "next_event_name": macro_news_state.get("next_event_name"),
        "neutral_fallback": macro_news_state.get("neutral_fallback"),
    })

    _print_section("GLOBAL RISK STATE", {
        "global_risk_state": global_risk_state.get("global_risk_state"),
        "global_risk_state_score": trade.get("global_risk_state_score") if trade else global_risk_state.get("global_risk_score"),
        "global_risk_overlay_score": trade.get("global_risk_overlay_score") if trade else None,
        "overnight_gap_risk_score": global_risk_state.get("overnight_gap_risk_score"),
        "volatility_expansion_risk_score": global_risk_state.get("volatility_expansion_risk_score"),
        "overnight_hold_allowed": global_risk_state.get("overnight_hold_allowed"),
        "overnight_hold_reason": global_risk_state.get("overnight_hold_reason"),
        "overnight_risk_penalty": global_risk_state.get("overnight_risk_penalty"),
        "global_risk_adjustment_score": (
            trade.get("global_risk_adjustment_score")
            if isinstance(trade, dict) and trade.get("global_risk_adjustment_score") is not None
            else global_risk_state.get("global_risk_adjustment_score")
        ),
        "global_risk_base_adjustment_score": (
            ((trade.get("scoring_breakdown") or {}).get("global_risk_base_adjustment_score"))
            if isinstance(trade, dict)
            else None
        ),
        "global_risk_feature_adjustment_score": (
            ((trade.get("scoring_breakdown") or {}).get("global_risk_feature_adjustment_score"))
            if isinstance(trade, dict)
            else None
        ),
        "global_risk_state_reasons": trade.get("global_risk_state_reasons") if trade else global_risk_state.get("global_risk_reasons"),
        "global_risk_overlay_reasons": trade.get("global_risk_overlay_reasons") if trade else None,
    })

    market_inputs = global_market_snapshot.get("market_inputs", {})
    _print_section("GLOBAL MARKET SNAPSHOT", {
        "provider": global_market_snapshot.get("provider"),
        "data_available": global_market_snapshot.get("data_available"),
        "stale": global_market_snapshot.get("stale"),
        "oil_change_24h": market_inputs.get("oil_change_24h"),
        "US VIX Change 24h": market_inputs.get("vix_change_24h"),
        "India VIX Level": market_inputs.get("india_vix_level"),
        "India VIX Change 24h": market_inputs.get("india_vix_change_24h"),
        "sp500_change_24h": market_inputs.get("sp500_change_24h"),
        "us10y_change_bp": market_inputs.get("us10y_change_bp"),
        "usdinr_change_24h": market_inputs.get("usdinr_change_24h"),
        "warnings": global_market_snapshot.get("warnings"),
    })

    # Macro/news provider details
    macro_news_details = {
        "headline_provider": headline_state.get("provider_name"),
        "headline_data_available": headline_state.get("data_available"),
        "headline_is_stale": headline_state.get("is_stale"),
        "headline_warnings": headline_state.get("warnings"),
        "headline_issues": headline_state.get("issues"),
        "provider_metadata": headline_state.get("provider_metadata"),
        "classification_preview": macro_news_state.get("classification_preview"),
    }
    if any(macro_news_details.get(k) for k in (
        "headline_warnings", "headline_issues", "provider_metadata",
        "classification_preview",
    )):
        _print_section("MACRO / NEWS DETAILS", {
            k: _fmt(v) for k, v in macro_news_details.items()
        })

    _print_validation("OPTION CHAIN VALIDATION", option_chain_validation)
    print(f"\n{'option_chain_rows':26}: {result.get('option_chain_rows')}")

    if not trade:
        print("\n  ▸ NO TRADE SIGNAL")
        _print_section("ENGINE STATUS", {"message": "No trade payload returned"})
        return

    # Trader view
    display = execution_trade or trade
    print("\nTRADER VIEW")
    print("---------------------------")
    for key in TRADER_VIEW_KEYS:
        if key in display:
            print(f"{key:26}: {display[key]}")

    _render_directionality_diagnostics(trade, mode="full_debug")

    # Dealer dashboard
    dashboard = dict(trade)
    _top_call_oi, _top_put_oi = _resolve_top_oi_strikes(trade, option_chain_frame, top_n=5, formatted=True)
    dashboard.update({
        "spot": spot_summary.get("spot"),
        "spot_timestamp": spot_summary.get("timestamp"),
        "day_open": spot_summary.get("day_open"),
        "day_high": spot_summary.get("day_high"),
        "day_low": spot_summary.get("day_low"),
        "prev_close": spot_summary.get("prev_close"),
        "lookback_avg_range_pct": spot_summary.get("lookback_avg_range_pct"),
        "spot_validation": spot_validation,
        "option_chain_validation": option_chain_validation,
        "macro_regime": macro_news_state.get("macro_regime"),
        "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
        "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
        "news_confidence_score": macro_news_state.get("news_confidence_score"),
        "headline_velocity": macro_news_state.get("headline_velocity"),
        "top_support_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[0],
        "top_resistance_walls": _resolve_top_liquidity_walls(trade, top_n=3, formatted=True, option_chain_frame=option_chain_frame)[1],
        "top_call_oi_strikes": _top_call_oi,
        "top_put_oi_strikes": _top_put_oi,
    })
    _render_dealer_dashboard(dashboard)

    # Full signal summary + explainability
    _render_full_signal_summary(trade, option_chain_frame=option_chain_frame)

    # Full diagnostics
    _render_diagnostics(trade)

    # Signal confidence (full breakdown)
    _render_signal_confidence(trade, show_components=True)

    # Overnight hold assessment (verbose with constraints)
    _render_overnight_assessment(trade, verbose=True)

    # Ranked strikes (extended)
    _render_ranked_strikes(
        trade.get("ranked_strike_candidates"),
        expiry=trade.get("selected_expiry"),
        extended=True,
        direction=trade.get("direction") if trade else None,
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_RENDERERS = {
    "COMPACT": render_compact,
    "STANDARD": render_standard,
    "FULL_DEBUG": render_full_debug,
}


def render_snapshot(mode, *, result, spot_summary, spot_validation,
                    option_chain_validation, macro_event_state,
                    macro_news_state, global_risk_state,
                    global_market_snapshot, headline_state,
                    trade, execution_trade, market_levels_sort_mode="GROUPED",
                    signal_capture_policy=None, capture_oi_inference_artifacts=True):
    """Dispatch to the appropriate renderer based on *mode*.

    Falls back to STANDARD if the mode is unrecognised.
    """
    # ── Visual separator between refreshes ───────────────────────────────
    from datetime import datetime as _dt
    now_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 80}")
    print(f"  ENGINE SNAPSHOT — {now_str}")
    print(f"{'=' * 80}")

    effective = mode.upper().strip() if mode else "STANDARD"
    renderer = _RENDERERS.get(effective, render_standard)

    # Build the common kwargs shared by all renderers.
    kwargs = dict(
        result=result,
        trade=trade,
        spot_summary=spot_summary,
        execution_trade=execution_trade,
        option_chain_frame=result.get("option_chain_frame") if isinstance(result, dict) else None,
    )

    # Capture OI inference artifacts for all output modes (policy-gated).
    if isinstance(result, dict):
        _trade_for_oi = _build_trade_for_oi_inference(
            trade=trade,
            result=result,
            spot_summary=spot_summary,
        )
        _oi_calls, _oi_puts = _get_top_oi_levels_cached(
            result=result,
            trade_for_oi=_trade_for_oi,
            option_chain_frame=kwargs.get("option_chain_frame"),
            top_n=5,
        ) if trade else ([], [])
        _persist_oi_inference_artifact(
            result=result,
            trade=trade,
            spot_summary=spot_summary,
            call_oi=_oi_calls,
            put_oi=_oi_puts,
            signal_capture_policy=signal_capture_policy,
            capture_enabled=bool(capture_oi_inference_artifacts),
        )

    if renderer is render_compact:
        kwargs["macro_event_state"] = macro_event_state
        kwargs["global_risk_state"] = global_risk_state
        kwargs["market_levels_sort_mode"] = market_levels_sort_mode
    else:
        kwargs["spot_validation"] = spot_validation
        kwargs["option_chain_validation"] = option_chain_validation
        kwargs["macro_event_state"] = macro_event_state
        kwargs["macro_news_state"] = macro_news_state
        kwargs["global_risk_state"] = global_risk_state
        kwargs["global_market_snapshot"] = global_market_snapshot
        kwargs["headline_state"] = headline_state

    renderer(**kwargs)
