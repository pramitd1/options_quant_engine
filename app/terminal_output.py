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
from datetime import date

from analytics.signal_confidence import compute_signal_confidence
from engine.runtime_metadata import TRADER_VIEW_KEYS


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


def _print_section(title, fields):
    """Print a titled key-value section."""
    print(f"\n{title}")
    print("---------------------------")
    for key, value in fields.items():
        if value is None:
            continue
        print(f"{key:26}: {_fmt(value)}")


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
    ("score", "score", ">6"),
]

_RANKED_EXTENDED_COLS = [
    ("distance_from_spot_pts", "dist_pts", ">9"),
    ("distance_from_spot_pct", "dist_%", ">7"),
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
            if isinstance(val, float):
                val = round(val, 2)
            if isinstance(val, bool):
                val = "Y" if val else "N"
            parts.append(f"{str(val):{fmt}}")
        print(" ".join(parts))


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
    iv_val = trade.get("atm_iv") or best.get("iv") or 0.15
    # atm_iv and ranked-candidate iv are in percentage points (e.g. 15.8);
    # convert to decimal for Black-Scholes expected-move calculation.
    if isinstance(iv_val, (int, float)) and iv_val > 1.5:
        iv_val = iv_val / 100.0
    spot = trade.get("spot") or trade.get("entry_price") or 0
    dte = trade.get("days_to_expiry") or 1
    expected_move = float(spot) * float(iv_val) * math.sqrt(max(float(dte), 0.1) / 365.0) if spot else 0
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
    support = trade.get("support_wall")
    resistance = trade.get("resistance_wall")
    dlm = trade.get("dealer_liquidity_map") or {}

    # Fall back to dealer_liquidity_map walls if trade-level walls are absent
    if support is None:
        support = dlm.get("next_support")
    if resistance is None:
        resistance = dlm.get("next_resistance")

    # Nothing to show if all data is missing
    if gamma_flip is None and not clusters and support is None and resistance is None:
        return

    print(f"\nDEALER GAMMA LEVELS")
    print("---------------------------")

    if gamma_flip is not None:
        print(f"{'gamma_flip':26}: {gamma_flip}")

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
            print(f"    {m}")

    if support is not None or resistance is not None:
        print(f"\n  Liquidity Walls")
        if resistance is not None:
            print(f"    {resistance}  Resistance")
        if support is not None:
            print(f"    {support}  Support")


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
                   global_risk_state, execution_trade):
    """Render structured compact output following a logical trader workflow.

    Section order:
        1. MARKET SUMMARY
        2. REGIME SUMMARY
        3. TRADE DECISION
        4. TRADING SUGGESTION
        5. RANKED STRIKES
        6. RISK SUMMARY
        7. OVERNIGHT HOLD
    """
    display = execution_trade or trade
    has_trade = bool(trade and trade.get("direction"))

    # ── 1. MARKET SUMMARY ────────────────────────────────────────────────
    spot = spot_summary.get("spot")
    _print_section("MARKET SUMMARY", {
        "spot": spot,
        "day_range": (
            f"{spot_summary.get('day_low')} – {spot_summary.get('day_high')}"
            if spot_summary.get("day_low") is not None else None
        ),
        "prev_close": spot_summary.get("prev_close"),
        "gamma_flip": trade.get("gamma_flip") if trade else None,
        "support_wall": trade.get("support_wall") if trade else None,
        "resistance_wall": trade.get("resistance_wall") if trade else None,
        "atm_iv": trade.get("atm_iv") if trade else None,
        "macro_event_risk": macro_event_state.get("macro_event_risk_score"),
        "event_lockdown": macro_event_state.get("event_lockdown_flag"),
    })

    # ── 2. REGIME SUMMARY ────────────────────────────────────────────────
    if trade:
        _print_section("REGIME SUMMARY", {
            "gamma_regime": trade.get("gamma_regime"),
            "spot_vs_flip": trade.get("spot_vs_flip"),
            "flow_signal": trade.get("final_flow_signal"),
            "vol_regime": trade.get("vol_surface_regime"),
            "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
            "macro_regime": trade.get("macro_regime"),
            "global_risk": global_risk_state.get("global_risk_state"),
        })

    # ── 3. TRADE DECISION ────────────────────────────────────────────────
    if trade:
        confidence = compute_signal_confidence(trade)
        conf_icon = _CONFIDENCE_ICONS.get(confidence["confidence_level"], "")
        prob = trade.get("hybrid_move_probability")
        prob_str = f"{prob:.0%}" if isinstance(prob, (int, float)) else "-"

        _print_section("TRADE DECISION", {
            "decision": trade.get("decision_classification"),
            "trade_strength": display.get("trade_strength") if display else None,
            "signal_quality": display.get("signal_quality") if display else None,
            "confirmation": trade.get("confirmation_status"),
            "move_probability": prob_str,
            "confidence": f"{conf_icon} {confidence['confidence_score']} ({confidence['confidence_level']})",
            "data_quality": trade.get("data_quality_status"),
        })
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
    else:
        print("  No trade yet. Waiting for confirmation.")

        # ── Threshold status ─────────────────────────────────────────────
        if trade:
            from config.signal_policy import get_trade_runtime_thresholds
            _thresholds = get_trade_runtime_thresholds()
            _min_strength = int(_thresholds.get("min_trade_strength", 60))
            _cur_strength = int(trade.get("trade_strength") or 0)
            _confirmation = str(
                trade.get("confirmation_status")
                or trade.get("confirmation")
                or ""
            ).upper()
            _no_direction = _confirmation == "NO_DIRECTION" or not trade.get("direction")

            print(f"\n  Trade Strength Threshold")
            if _no_direction:
                print("    current  : N/A (no direction yet)")
                print(f"    required : {_min_strength}")
                print(f"    progress : [{'░' * 20}] N/A/{_min_strength}")
            else:
                _gap = _min_strength - _cur_strength
                _bar_filled = max(0, min(20, round(20 * _cur_strength / max(_min_strength, 1))))
                _bar = "█" * _bar_filled + "░" * (20 - _bar_filled)
                _gap_str = f"  ({_gap:+d} to threshold)" if _gap > 0 else "  ✓ threshold met"
                print(f"    current  : {_cur_strength}")
                print(f"    required : {_min_strength}{_gap_str}")
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
                    print(f"    delta    : {round(float(_b_delta), 4)}")
                if _b_iv is not None:
                    print(f"    iv       : {round(float(_b_iv), 2)}")

        # ── Potential trigger conditions ─────────────────────────────────
        triggers = []
        if trade:
            ntr = trade.get("no_trade_reason")
            if ntr:
                triggers.append(ntr)
            for cond in (trade.get("setup_upgrade_conditions") or []):
                triggers.append(cond)
            lt = trade.get("likely_next_trigger")
            if lt:
                triggers.append(lt)
        if triggers:
            print(f"\n  Potential triggers:")
            for t in triggers:
                print(f"    • {t}")

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

    risk_fields = {
        "global_risk": global_risk_state.get("global_risk_state"),
        "global_risk_score": global_risk_state.get("global_risk_score"),
        "event_lockdown": event_lock if event_lock else None,
        "watchlist": watchlist if watchlist else None,
        "macro_news_status": trade.get("macro_news_status"),
        "option_efficiency": oe_display,
    }
    no_trade = trade.get("no_trade_reason")
    if no_trade:
        risk_fields["no_trade_reason"] = no_trade
    _print_section("RISK SUMMARY", risk_fields)

    # ── 7. OVERNIGHT HOLD ────────────────────────────────────────────────
    _render_overnight_assessment(trade, verbose=False)


# ---------------------------------------------------------------------------
# STANDARD mode — compact + scoring/confirmation diagnostics
# ---------------------------------------------------------------------------

def render_standard(*, result, trade, spot_summary, spot_validation,
                    option_chain_validation, macro_event_state,
                    macro_news_state, global_risk_state, global_market_snapshot,
                    execution_trade, headline_state):
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
        "global_risk_score": global_risk_state.get("global_risk_score"),
        "overnight_hold_allowed": global_risk_state.get("overnight_hold_allowed"),
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
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "macro_regime": trade.get("macro_regime"),
        "global_risk_state": trade.get("global_risk_state"),
        "option_efficiency_score": trade.get("option_efficiency_score"),
        "premium_efficiency_score": trade.get("premium_efficiency_score"),
        "strike_efficiency_score": trade.get("strike_efficiency_score"),
        "capital_required": trade.get("capital_required"),
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
    ("Gamma Flip Level", "gamma_flip"),
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
    ("Support Wall", "support_wall"),
    ("Resistance Wall", "resistance_wall"),
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
    "global_risk_reasons",
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
    "missing_confirmations",
    "signal_promotion_requirements",
    "setup_upgrade_path",
    "watchlist_trigger_levels",
    "directional_resolution_needed",
    "explainability",
    "scoring_breakdown",
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


def _render_full_signal_summary(trade):
    """Print the exhaustive quant trade signal block."""
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
        "global_risk_score": trade.get("global_risk_score"),
        "gamma_vol_acceleration_score": trade.get("gamma_vol_acceleration_score"),
        "squeeze_risk_state": trade.get("squeeze_risk_state"),
        "directional_convexity_state": trade.get("directional_convexity_state"),
        "dealer_hedging_pressure_score": trade.get("dealer_hedging_pressure_score"),
        "dealer_flow_state": trade.get("dealer_flow_state"),
        "expected_move_points": trade.get("expected_move_points"),
        "expected_move_pct": trade.get("expected_move_pct"),
        "expected_move_quality": trade.get("expected_move_quality"),
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
                      execution_trade):
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
        "global_risk_score": global_risk_state.get("global_risk_score"),
        "overnight_gap_risk_score": global_risk_state.get("overnight_gap_risk_score"),
        "volatility_expansion_risk_score": global_risk_state.get("volatility_expansion_risk_score"),
        "overnight_hold_allowed": global_risk_state.get("overnight_hold_allowed"),
        "overnight_hold_reason": global_risk_state.get("overnight_hold_reason"),
        "overnight_risk_penalty": global_risk_state.get("overnight_risk_penalty"),
        "global_risk_adjustment_score": global_risk_state.get("global_risk_adjustment_score"),
        "global_risk_reasons": global_risk_state.get("global_risk_reasons"),
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

    # Dealer dashboard
    dashboard = dict(trade)
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
    })
    _render_dealer_dashboard(dashboard)

    # Full signal summary + explainability
    _render_full_signal_summary(trade)

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
                    trade, execution_trade):
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
    )

    if renderer is render_compact:
        kwargs["macro_event_state"] = macro_event_state
        kwargs["global_risk_state"] = global_risk_state
    else:
        kwargs["spot_validation"] = spot_validation
        kwargs["option_chain_validation"] = option_chain_validation
        kwargs["macro_event_state"] = macro_event_state
        kwargs["macro_news_state"] = macro_news_state
        kwargs["global_risk_state"] = global_risk_state
        kwargs["global_market_snapshot"] = global_market_snapshot
        kwargs["headline_state"] = headline_state

    renderer(**kwargs)
