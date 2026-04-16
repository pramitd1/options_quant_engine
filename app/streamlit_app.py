"""
Module: streamlit_app.py

Purpose:
    Provide the Streamlit operator interface for live runs, replay analysis, and research inspection.

Role in the System:
    Part of the application layer that exposes runtime controls and diagnostics through a browser-based workstation.

Key Outputs:
    Interactive controls, rendered signal diagnostics, replay tools, and research tables.

Downstream Usage:
    Used by operators and researchers who need a richer interface than the CLI runtime.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.settings import (
    DATA_SOURCE_OPTIONS,
    DEFAULT_DATA_SOURCE,
    DEFAULT_SYMBOL,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
)
from app.engine_runner import run_engine_snapshot
from app.html_utils import esc
from app.state import (
    persist_control_state,
    query_param_bool,
    seed_control_state,
)
from app.styles import OQE_GLOBAL_CSS
from data.replay_loader import list_replay_chain_snapshots
from research.signal_evaluation import (
    SIGNAL_DATASET_PATH,
    build_research_report,
    load_signals_dataset,
)


st.set_page_config(
    page_title="Options Quant Engine",
    page_icon="OQ",
    layout="wide",
)

st.markdown(OQE_GLOBAL_CSS, unsafe_allow_html=True)


_LOG = logging.getLogger(__name__)
_WARN_ONCE_KEYS: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    _LOG.warning(message, *args)


def _signed_option_type(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.upper().str.strip()
    signed = normalized.map({"CE": 1.0, "PE": -1.0})
    if signed.isna().any():
        unknown = sorted(set(normalized[signed.isna()].tolist()))
        raise ValueError(f"Unknown OPTION_TYP values in streamlit signed gamma view: {unknown}")
    return signed


def _safe_metric_value(value):
    """
    Purpose:
        Safely normalize metric value while preserving fallback behavior.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (Any): Input associated with value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if value in (None, ""):
        return "-"
    return esc(value)


def _normalize_symbol_input_state(key: str = "control_symbol"):
    """
    Purpose:
        Normalize the symbol widget value in Streamlit session state.

    Context:
        This callback runs from the Symbol widget `on_change` hook so the
        displayed input value is uppercased safely within Streamlit's callback
        lifecycle.

    Inputs:
        key (str): Session-state key for the symbol input widget.

    Returns:
        None: The function mutates `st.session_state` in place.
    """
    raw_value = st.session_state.get(key)
    normalized = (str(raw_value or "").strip().upper() or DEFAULT_SYMBOL)
    st.session_state[key] = normalized


def _safe_float(value):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        value (Any): Raw value supplied by the caller.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _display_path(path_value: str) -> str:
    """
    Purpose:
        Render path for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path_value (str): Input associated with path value.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if not path_value:
        return "-"
    path = Path(path_value)
    if len(path_value) <= 96:
        return esc(path_value)
    return esc(f"{path.parent.name}/{path.name}")


def _normalize_display_value(value):
    """
    Purpose:
        Normalize display value into the repository-standard form.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (Any): Input associated with value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple, set, dict)):
        try:
            return json.dumps(value, ensure_ascii=True, default=str, sort_keys=isinstance(value, dict))
        except TypeError:
            return str(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _prepare_display_frame(data) -> pd.DataFrame:
    """
    Purpose:
        Process prepare display frame for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        data (Any): Input associated with data.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    if frame.empty:
        return frame

    frame.columns = [str(column) for column in frame.columns]

    for column in frame.columns:
        if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column]):
            normalized = frame[column].map(_normalize_display_value)
            non_null = normalized.dropna()

            if non_null.empty:
                frame[column] = normalized.astype("string")
                continue

            if non_null.map(lambda value: isinstance(value, bool)).all():
                frame[column] = normalized.astype("boolean")
                continue

            if non_null.map(lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)).all():
                frame[column] = pd.to_numeric(normalized, errors="coerce")
                continue

            frame[column] = normalized.map(lambda value: None if value is None else str(value)).astype("string")

    return frame


def _render_dataframe(data, *, hide_index: bool = True):
    """
    Purpose:
        Render dataframe for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        data (Any): Input associated with data.
        hide_index (bool): Boolean flag associated with hide_index.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    st.dataframe(
        _prepare_display_frame(data),
        use_container_width=True,
        hide_index=hide_index,
    )


def _render_enhanced_ranked_strikes(ranked: pd.DataFrame):
    """Render the enhanced ranked-strikes table with factor scores and diagnostics."""
    has_enhanced = "enhanced_strike_score" in ranked.columns

    # --- Display columns ---
    core_cols = ["strike", "last_price", "volume", "open_interest", "iv", "score"]
    factor_cols = [
        "enhanced_strike_score", "liquidity_score", "gamma_magnetism",
        "dealer_pressure", "convexity_score", "premium_efficiency",
    ]
    distance_cols = ["distance_from_spot_pts", "distance_from_spot_pct"]
    flag_cols = ["tradable_intraday", "tradable_overnight", "liquidity_ok", "premium_reasonable"]
    context_cols = ["gamma_regime", "spot_vs_flip", "dealer_hedging_bias", "vol_surface_regime"]

    if has_enhanced:
        display_cols = [c for c in core_cols + factor_cols + distance_cols + flag_cols if c in ranked.columns]
    else:
        display_cols = [c for c in core_cols if c in ranked.columns]

    display = ranked[display_cols].copy()

    # Sort by enhanced score when available, otherwise base score
    sort_col = "enhanced_strike_score" if has_enhanced else "score"
    if sort_col in display.columns:
        display = display.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Highlight top strike row via Styler
    def _highlight_top(row):
        if row.name == 0:
            return ["background-color: rgba(76, 175, 80, 0.15)"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_highlight_top, axis=1)

    # Format factor scores to 2 decimal places where applicable
    factor_format = {c: "{:.2f}" for c in factor_cols if c in display.columns}
    if "distance_from_spot_pct" in display.columns:
        factor_format["distance_from_spot_pct"] = "{:.2f}%"
    styled = styled.format(factor_format, na_rep="-")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Show market structure context below the table
    if has_enhanced and not ranked.empty:
        ctx_values = {c: ranked.iloc[0].get(c) for c in context_cols if c in ranked.columns and ranked.iloc[0].get(c)}
        if ctx_values:
            ctx_parts = [f"**{k.replace('_', ' ').title()}**: {v}" for k, v in ctx_values.items() if v]
            if ctx_parts:
                st.caption("Market Structure: " + " · ".join(ctx_parts))


def _badge_class_for_regime(regime: str) -> str:
    """
    Purpose:
        Process badge class for regime for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        regime (str): Input associated with regime.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    regime = (regime or "").upper()
    if regime == "RISK_ON":
        return "oqe-badge-risk-on"
    if regime == "RISK_OFF":
        return "oqe-badge-risk-off"
    if regime == "EVENT_LOCKDOWN":
        return "oqe-badge-lockdown"
    return "oqe-badge-neutral"


def _badge_class_for_trade_status(status: str) -> str:
    """
    Purpose:
        Process badge class for trade status for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        status (str): Input associated with status.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    status = (status or "").upper()
    if status == "TRADE":
        return "oqe-badge-trade"
    if status == "WATCHLIST":
        return "oqe-badge-watch"
    if status in {"NO_SIGNAL", "DATA_INVALID"}:
        return "oqe-badge-no-signal"
    return "oqe-badge-blocked"


def _badge_class_for_decision_classification(decision_classification: str, trade_status: str) -> str:
    decision = (decision_classification or "").upper()
    trade_status = (trade_status or "").upper()
    if trade_status == "TRADE" or decision == "TRADE_READY":
        return "oqe-decision-ready"
    if trade_status in {"DATA_INVALID", "NO_TRADE", "BUDGET_FAIL"} or "BLOCK" in decision:
        return "oqe-decision-blocked"
    if "WATCHLIST" in decision or "AMBIGUOUS" in decision:
        return "oqe-decision-watchlist"
    if "INACTIVE" in decision:
        return "oqe-decision-inactive"
    return "oqe-badge-neutral"


def _render_explainability_scorecard(trade: dict):
    if not isinstance(trade, dict):
        return

    decision_classification = trade.get("decision_classification") or "UNKNOWN"
    trade_status = trade.get("trade_status")
    missing_requirements = trade.get("missing_signal_requirements")
    missing_requirements = missing_requirements if isinstance(missing_requirements, list) else []
    blocked_by = trade.get("blocked_by")
    blocked_by = blocked_by if isinstance(blocked_by, list) else []
    next_trigger = trade.get("likely_next_trigger")
    setup_activation_score = trade.get("setup_activation_score")
    setup_maturity_score = trade.get("setup_maturity_score")
    confidence = trade.get("explainability_confidence")

    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Explainability Scorecard")
    cols = st.columns([1.05, 0.9, 0.9, 1.0, 1.15])

    decision_badge_class = _badge_class_for_decision_classification(decision_classification, trade_status)
    cols[0].markdown(
        (
            '<div class="oqe-mini-scorecard">'
            '<div class="oqe-mini-title">Decision</div>'
            f'<div class="oqe-badge {decision_badge_class}" style="margin-bottom:0.35rem;">{esc(decision_classification)}</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(trade_status)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    cols[1].markdown(
        (
            '<div class="oqe-mini-scorecard">'
            '<div class="oqe-mini-title">Setup Quality</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(trade.get("setup_quality"))}</div>'
            f'<div class="oqe-mini-title" style="margin-top:0.4rem;">State</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(trade.get("setup_state"))}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    blocked_value = esc(", ".join(blocked_by)) if blocked_by else "-"
    cols[2].markdown(
        (
            '<div class="oqe-mini-scorecard">'
            '<div class="oqe-mini-title">Missing / Blocked</div>'
            f'<div class="oqe-mini-value">{len(missing_requirements)} missing</div>'
            f'<div class="oqe-mini-title" style="margin-top:0.4rem;">Blocked By</div>'
            f'<div class="oqe-mini-value">{blocked_value}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    cols[3].markdown(
        (
            '<div class="oqe-mini-scorecard">'
            '<div class="oqe-mini-title">Activation / Maturity</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(setup_activation_score)} / {_safe_metric_value(setup_maturity_score)}</div>'
            f'<div class="oqe-mini-title" style="margin-top:0.4rem;">Confidence</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(confidence)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    cols[4].markdown(
        (
            '<div class="oqe-mini-scorecard">'
            '<div class="oqe-mini-title">Likely Next Trigger</div>'
            f'<div class="oqe-mini-value">{_safe_metric_value(next_trigger)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def _list_replay_files(replay_dir: str, symbol: str, kind: str, source_label: str | None = None):
    """
    Purpose:
        List replay files available to the current workflow.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        replay_dir (str): Input associated with replay dir.
        symbol (str): Underlying symbol or index identifier.
        kind (str): Input associated with kind.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    replay_path = Path(replay_dir)
    if not replay_path.exists():
        return [], []

    symbol = (symbol or "").strip().upper()
    if kind == "spot":
        pattern = f"{symbol}_spot_snapshot_*.json"
        return sorted(str(path) for path in replay_path.glob(pattern)), []
    else:
        return list_replay_chain_snapshots(
            symbol,
            replay_dir=replay_dir,
            source_label=source_label,
        )

    return [], []


def _list_replay_directories() -> list[str]:
    """Return directories under the project root that contain snapshot files."""
    root = PROJECT_ROOT
    candidates = ["debug_samples", "data_store", "backtests"]
    dirs = []
    for name in candidates:
        path = root / name
        if path.is_dir():
            dirs.append(name)
    # Also pick up any other top-level dirs that contain snapshot files.
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name not in candidates and not entry.name.startswith((".", "__")):
            if any(entry.glob("*_spot_snapshot_*.json")) or any(entry.glob("*_option_chain_snapshot_*")):
                dirs.append(entry.name)
    return dirs if dirs else ["debug_samples"]


def _list_replay_source_labels(replay_dir: str, symbol: str) -> list[str]:
    """Scan chain snapshot filenames to extract available source labels."""
    replay_path = Path(replay_dir)
    if not replay_path.is_dir():
        return ["ANY"]
    symbol = (symbol or "").strip().upper()
    labels: set[str] = set()
    for path in replay_path.glob(f"{symbol}_*_option_chain_snapshot_*"):
        name = path.name
        # Filename pattern: {SYMBOL}_{SOURCE}_option_chain_snapshot_...
        remainder = name[len(symbol) + 1:]  # strip "{SYMBOL}_"
        idx = remainder.find("_option_chain_snapshot_")
        if idx > 0:
            labels.add(remainder[:idx])
    ordered = sorted(labels)
    return ["ANY"] + ordered if ordered else ["ANY"]


def _extract_snapshot_timestamp(path_str: str):
    """
    Purpose:
        Extract snapshot timestamp from the supplied payload.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path_str (str): Input associated with path str.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2}(?:\.\d+)?\+\d{2}[-:]\d{2})", path_str)
    if not match:
        return None
    token = match.group(1).replace("-", ":", 2)
    token = token.replace("-", ":", 1) if "+" in token and token.rsplit("+", 1)[-1].count("-") == 1 else token
    token = re.sub(r"(?<=\+\d{2})-(?=\d{2}$)", ":", token)
    try:
        return pd.to_datetime(token, errors="coerce")
    except Exception as exc:
        _warn_once(
            "streamlit_snapshot_timestamp_parse_failed",
            "streamlit_app: failed to parse snapshot timestamp from %r (%s)",
            path_str,
            exc,
        )
        return None


def _select_default_option(options):
    """
    Purpose:
        Process select default option for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        options (Any): Input associated with options.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if not options:
        return ""
    return options[-1]


def _nearest_spot_for_chain(chain_path: str, spot_options):
    """
    Purpose:
        Process nearest spot for chain for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        chain_path (str): Input associated with chain path.
        spot_options (Any): Input associated with spot options.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    chain_ts = _extract_snapshot_timestamp(chain_path or "")
    if chain_ts is None or not spot_options:
        return _select_default_option(spot_options)

    candidates = []
    for spot_path in spot_options:
        spot_ts = _extract_snapshot_timestamp(spot_path)
        if spot_ts is None:
            continue
        candidates.append((abs((chain_ts - spot_ts).total_seconds()), spot_path))
    if not candidates:
        return _select_default_option(spot_options)
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _render_key_value_table(title: str, values: dict):
    """
    Purpose:
        Render key value table for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        title (str): Input associated with title.
        values (dict): Input associated with values.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    st.markdown(f'<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader(title)
    frame = pd.DataFrame([{"field": key, "value": value} for key, value in values.items()])
    _render_dataframe(frame)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_key_value_list(title: str, values: dict):
    """
    Purpose:
        Render key value list for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        title (str): Input associated with title.
        values (dict): Input associated with values.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    st.markdown(f'<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader(title)
    for key, value in values.items():
        st.markdown(f"**{key}**")
        if isinstance(value, (dict, list)):
            st.json(value, expanded=False)
        else:
            st.write(value)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_runbar(result: dict):
    """
    Purpose:
        Render runbar for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result (dict): Input associated with result.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    timestamp = ((result.get("spot_summary") or {}).get("timestamp")) or "-"
    pills = [
        ("Mode", result.get("mode")),
        ("Source", result.get("source")),
        ("Symbol", result.get("symbol")),
        ("Snapshot Time", timestamp),
    ]
    html = "".join(
        f'<div class="oqe-runpill">{label}: {_safe_metric_value(value)}</div>'
        for label, value in pills
    )
    st.markdown(f'<div class="oqe-runbar">{html}</div>', unsafe_allow_html=True)


def _render_trade_metrics(trade: dict):
    """
    Purpose:
        Render the trade metrics for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        trade (dict): Trade payload produced by the signal engine, or `None` for no-trade outcomes.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    status = trade.get("trade_status", "UNKNOWN")
    st.markdown(
        f'<div class="oqe-badge {_badge_class_for_trade_status(status)}">{esc(status)}</div>',
        unsafe_allow_html=True,
    )

    # Primary execution fields — the operator's at-a-glance trade ticket
    direction = trade.get("direction")
    option_type = trade.get("option_type")
    direction_label = _safe_metric_value(direction)
    if option_type:
        direction_label = f"{_safe_metric_value(direction)} ({esc(str(option_type))})"

    cols = st.columns(6)
    primary_items = [
        ("Direction", direction_label),
        ("Strike", trade.get("strike")),
        ("Entry Price", trade.get("entry_price")),
        ("Target", trade.get("target")),
        ("Stop Loss", trade.get("stop_loss")),
        ("Expiry", trade.get("selected_expiry")),
    ]
    for col, (label, value) in zip(cols, primary_items):
        col.markdown(
            f"""
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">{label}</div>
                <div class="oqe-summary-value">{_safe_metric_value(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Secondary sizing and quality metrics
    kpi_cols = st.columns(6)
    lots_value = "-"
    current_lots = trade.get("number_of_lots")
    suggested_lots = trade.get("macro_suggested_lots")
    if current_lots is not None or suggested_lots is not None:
        lots_value = f"{_safe_metric_value(current_lots)} / {_safe_metric_value(suggested_lots)}"

    kpi_cols[0].metric("Strength", _safe_metric_value(trade.get("trade_strength")))
    kpi_cols[1].metric("Signal Quality", _safe_metric_value(trade.get("signal_quality")))
    kpi_cols[2].metric("Data Quality", _safe_metric_value(trade.get("data_quality_score")))
    kpi_cols[3].metric("Lots (Curr / Sugg)", lots_value)
    kpi_cols[4].metric("Capital Required", _safe_metric_value(trade.get("capital_required")))
    kpi_cols[5].metric("Event Risk", _safe_metric_value(trade.get("macro_event_risk_score")))


def _render_decision_panel(trade: dict):
    """
    Purpose:
        Render the decision panel for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        trade (dict): Trade payload produced by the signal engine, or `None` for no-trade outcomes.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    focus_cards = [
        ("Decision Class", trade.get("decision_classification")),
        ("Setup State", trade.get("setup_state")),
        ("Direction Source", trade.get("direction_source")),
        ("Execution Regime", trade.get("execution_regime")),
        ("Signal Regime", trade.get("signal_regime")),
        ("Provider Health", (trade.get("provider_health") or {}).get("summary_status")),
        ("Macro Adjustment", trade.get("macro_adjustment_score")),
        ("Position Size Multiplier", trade.get("macro_position_size_multiplier")),
        ("No-Trade Code", trade.get("no_trade_reason_code")),
        ("Watchlist", trade.get("watchlist_flag")),
    ]
    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Decision Summary")
    for row_start in range(0, len(focus_cards), 3):
        row = focus_cards[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            col.metric(label, _safe_metric_value(value))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_overnight_risk_card(trade: dict):
    """Render a color-coded overnight hold assessment card."""
    from app.terminal_output import resolve_overnight_hold_assessment

    assessment = resolve_overnight_hold_assessment(trade)
    suggested = assessment["overnight_hold_suggested"]

    class_map = {"YES": "oqe-overnight-yes", "HOLD_WITH_CAUTION": "oqe-overnight-caution", "NO": "oqe-overnight-no"}
    label_map = {"YES": "HOLD ALLOWED", "HOLD_WITH_CAUTION": "HOLD WITH CAUTION", "NO": "DO NOT HOLD"}

    card_class = class_map.get(suggested, "oqe-overnight-unknown")
    label = label_map.get(suggested, esc(suggested))

    st.markdown(
        f'<div class="oqe-overnight-card {card_class}">'
        f'<span class="oqe-overnight-label">{label}</span>'
        f'<span class="oqe-overnight-confidence">'
        f'Confidence: {esc(assessment["overnight_hold_confidence"])}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    detail_cols = st.columns(3)
    detail_cols[0].metric("Gap Risk Score", _safe_metric_value(assessment["overnight_gap_risk_score"]))
    detail_cols[1].metric("Risk Penalty", _safe_metric_value(assessment["overnight_risk_penalty"]))
    detail_cols[2].metric("Suggested", label)

    if assessment["overnight_hold_reason"]:
        st.caption(f"Reason: {assessment['overnight_hold_reason']}")
    if assessment["overnight_constraints"]:
        with st.expander("Overnight Constraints", expanded=False):
            for c in assessment["overnight_constraints"]:
                st.text(f"• {c}")


def _render_signal_confidence_card(trade: dict):
    """Render a gauge-style signal confidence card with component breakdown."""
    from analytics.signal_confidence import compute_signal_confidence

    result = compute_signal_confidence(trade)
    score = result["confidence_score"]
    level = result["confidence_level"]

    color_map = {
        "VERY_HIGH": "#166534",
        "HIGH": "#1e40af",
        "MODERATE": "#92400e",
        "LOW": "#9a3412",
        "UNRELIABLE": "#991b1b",
    }
    dark_color_map = {
        "VERY_HIGH": "#4ade80",
        "HIGH": "#60a5fa",
        "MODERATE": "#fbbf24",
        "LOW": "#fb923c",
        "UNRELIABLE": "#f87171",
    }
    accent_map = {
        "VERY_HIGH": "#22c55e",
        "HIGH": "#3b82f6",
        "MODERATE": "#eab308",
        "LOW": "#f97316",
        "UNRELIABLE": "#ef4444",
    }
    color = color_map.get(level, "#475569")
    dark_color = dark_color_map.get(level, "#94a3b8")
    accent = accent_map.get(level, "#94a3b8")

    # SVG gauge arc
    pct = max(0, min(100, score))
    arc_len = pct * 1.8  # 0–180 degrees mapped
    rad = math.radians(180 - arc_len)
    ex = 50 + 40 * math.cos(rad)
    ey = 55 - 40 * math.sin(rad)
    large = 1 if arc_len > 90 else 0

    gauge_svg = (
        f'<svg viewBox="0 0 100 60" width="220" height="132">'
        f'<path d="M10,55 A40,40 0 0,1 90,55" fill="none" stroke="var(--oqe-gauge-track, rgba(30,41,59,0.15))" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="M10,55 A40,40 0 {large},1 {ex:.1f},{ey:.1f}" fill="none" stroke="{accent}" stroke-width="6" stroke-linecap="round"/>'
        f'<text x="50" y="48" text-anchor="middle" fill="var(--oqe-gauge-score, {color})" font-size="14" font-weight="700">{score}</text>'
        f'<text x="50" y="58" text-anchor="middle" fill="var(--oqe-gauge-sublabel, #6b7280)" font-size="6">{level}</text>'
        f'</svg>'
    )

    st.markdown(
        f'<div class="oqe-confidence-card" style="--oqe-conf-color:{color};--oqe-conf-dark-color:{dark_color}">'
        f'<div class="oqe-confidence-title">'
        f'Signal Confidence</div>'
        f'{gauge_svg}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Component breakdown table
    components = [
        ("Signal Strength", result["signal_strength_component"], "30%"),
        ("Confirmation", result["confirmation_component"], "25%"),
        ("Market Stability", result["market_stability_component"], "20%"),
        ("Data Integrity", result["data_integrity_component"], "15%"),
        ("Option Efficiency", result["option_efficiency_component"], "10%"),
    ]
    rows = []
    for name, val, weight in components:
        bar_color = "#22c55e" if val >= 70 else "#eab308" if val >= 40 else "#ef4444"
        rows.append({"Component": name, "Score": round(val, 1), "Weight": weight})

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def _render_macro_news_section(macro_news_state: dict, headline_state: dict):
    """
    Purpose:
        Render the macro news section for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        macro_news_state (dict): Headline-driven macro-news state produced by the news stack.
        headline_state (dict): Headline-ingestion state produced by the news service.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    st.markdown(f'<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Macro / News Regime")
    regime = macro_news_state.get("macro_regime", "MACRO_NEUTRAL")
    st.markdown(
        f'<div class="oqe-badge {_badge_class_for_regime(regime)}">{esc(regime)}</div>',
        unsafe_allow_html=True,
    )

    summary = {
        "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
        "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
        "news_confidence_score": macro_news_state.get("news_confidence_score"),
        "headline_velocity": macro_news_state.get("headline_velocity"),
        "headline_count": macro_news_state.get("headline_count"),
        "classified_headline_count": macro_news_state.get("classified_headline_count"),
        "event_lockdown_flag": macro_news_state.get("event_lockdown_flag"),
        "neutral_fallback": macro_news_state.get("neutral_fallback"),
    }
    st.dataframe(
        _prepare_display_frame([{"field": key, "value": value} for key, value in summary.items()]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Macro Regime Reasons", expanded=True):
        st.json(macro_news_state.get("macro_regime_reasons", []), expanded=True)

    with st.expander("Headline Diagnostics", expanded=False):
        for key, value in {
            "provider_name": headline_state.get("provider_name"),
            "data_available": headline_state.get("data_available"),
            "is_stale": headline_state.get("is_stale"),
            "neutral_fallback": headline_state.get("neutral_fallback"),
            "warnings": headline_state.get("warnings"),
            "issues": headline_state.get("issues"),
            "provider_metadata": headline_state.get("provider_metadata"),
        }.items():
            st.markdown(f"**{key}**")
            if isinstance(value, (dict, list)):
                st.json(value, expanded=False)
            else:
                st.write(value)

    classification_preview = macro_news_state.get("classification_preview")
    if classification_preview:
        with st.expander("Classification Preview", expanded=False):
            st.json(classification_preview, expanded=False)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_option_chain_charts(option_chain: pd.DataFrame):
    """
    Purpose:
        Render the option chain charts for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        option_chain (pd.DataFrame): Option-chain snapshot in dataframe form.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    if option_chain is None or option_chain.empty:
        st.info("No option-chain data available for charts.")
        return

    df = option_chain.copy()
    numeric_cols = ["strikePrice", "lastPrice", "openInterest", "changeinOI", "impliedVolatility", "GAMMA", "DELTA", "VEGA", "THETA", "VANNA", "CHARM"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    chart_cols = st.columns(2)

    if {"strikePrice", "OPTION_TYP", "openInterest"}.issubset(df.columns):
        oi_view = (
            df.dropna(subset=["strikePrice"])
            .pivot_table(index="strikePrice", columns="OPTION_TYP", values="openInterest", aggfunc="sum")
            .sort_index()
            .fillna(0)
        )
        with chart_cols[0]:
            st.markdown("**Open Interest by Strike**")
            st.bar_chart(oi_view, use_container_width=True)

    if {"strikePrice", "OPTION_TYP", "impliedVolatility"}.issubset(df.columns):
        iv_view = (
            df[df["impliedVolatility"] > 0]
            .pivot_table(index="strikePrice", columns="OPTION_TYP", values="impliedVolatility", aggfunc="mean")
            .sort_index()
        )
        with chart_cols[1]:
            st.markdown("**IV Smile**")
            if not iv_view.empty:
                st.line_chart(iv_view, use_container_width=True)
            else:
                st.caption("No positive IV values available.")

    greek_cols = st.columns(2)
    if {"strikePrice", "OPTION_TYP", "GAMMA"}.issubset(df.columns):
        gamma_view = (
            df.dropna(subset=["strikePrice", "GAMMA"])
            .assign(signed_gamma=lambda frame: frame["GAMMA"] * _signed_option_type(frame["OPTION_TYP"]))
            .groupby("strikePrice", as_index=True)["signed_gamma"]
            .sum()
            .sort_index()
        )
        with greek_cols[0]:
            st.markdown("**Signed Gamma by Strike**")
            if not gamma_view.empty:
                st.line_chart(gamma_view, use_container_width=True)
            else:
                st.caption("Gamma data unavailable.")

    if {"strikePrice", "lastPrice"}.issubset(df.columns):
        ltp_view = (
            df.dropna(subset=["strikePrice"])
            .pivot_table(index="strikePrice", columns="OPTION_TYP", values="lastPrice", aggfunc="mean")
            .sort_index()
        )
        with greek_cols[1]:
            st.markdown("**Option Premium Curve**")
            st.line_chart(ltp_view, use_container_width=True)

    if {"strikePrice", "changeinOI", "OPTION_TYP"}.issubset(df.columns):
        coi_view = (
            df.dropna(subset=["strikePrice"])
            .pivot_table(index="strikePrice", columns="OPTION_TYP", values="changeinOI", aggfunc="sum")
            .sort_index()
            .fillna(0)
        )
        st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
        st.markdown("**Change in OI by Strike**")
        st.bar_chart(coi_view, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if {"strikePrice", "OPTION_TYP", "openInterest"}.issubset(df.columns):
        pcr_view = (
            df.dropna(subset=["strikePrice"])
            .pivot_table(index="strikePrice", columns="OPTION_TYP", values="openInterest", aggfunc="sum")
            .sort_index()
            .fillna(0)
        )
        st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
        st.markdown("**Put-Call OI Ratio by Strike (ATM-Centered)**")
        if {"PE", "CE"}.issubset(pcr_view.columns):
            pcr_series = pcr_view["PE"] / pcr_view["CE"].replace(0, pd.NA)
            pcr_series = pcr_series.dropna()
            if not pcr_series.empty:
                strikes = list(pcr_series.index)
                atm_reference = None
                if "underlyingValue" in df.columns:
                    spot_series = pd.to_numeric(df["underlyingValue"], errors="coerce").dropna()
                    if not spot_series.empty:
                        atm_reference = float(spot_series.median())
                if atm_reference is None:
                    oi_total = pcr_view["PE"].fillna(0) + pcr_view["CE"].fillna(0)
                    if not oi_total.empty:
                        atm_reference = float(oi_total.idxmax())

                if atm_reference is None:
                    atm_strike = strikes[len(strikes) // 2]
                else:
                    atm_strike = min(strikes, key=lambda strike: abs(strike - atm_reference))

                atm_idx = strikes.index(atm_strike)
                max_window = max(1, min(20, len(strikes) // 2 if len(strikes) > 2 else 1))
                default_window = min(8, max_window)
                strike_window = st.slider(
                    "PCR window (ATM ± N strikes)",
                    min_value=1,
                    max_value=max_window,
                    value=default_window,
                    key="structure_pcr_atm_window",
                    help="Focuses PCR around ATM for cleaner intraday read.",
                )

                start_idx = max(0, atm_idx - strike_window)
                end_idx = min(len(strikes), atm_idx + strike_window + 1)
                centered_pcr = pcr_series.iloc[start_idx:end_idx]

                st.line_chart(centered_pcr, use_container_width=True)
                st.caption(
                    f"ATM anchor: {atm_strike:.2f}. Showing {len(centered_pcr)} strikes around ATM. "
                    "Values above 1 indicate relatively higher put OI vs call OI at that strike."
                )
            else:
                st.caption("PCR unavailable: CE OI is zero or missing across strikes.")
        else:
            st.caption("PCR unavailable: both CE and PE legs are required.")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_regime_rollout_panel(result: dict):
    """Render shadow-rollout diagnostics and the session-by-session validation table."""
    if not isinstance(result, dict):
        return

    regime_eval = result.get("regime_pack_evaluation") or {}
    shadow_comparison = result.get("shadow_comparison") or {}
    shadow_summary = result.get("shadow_validation_summary") or {}
    latest_session = shadow_summary.get("latest_session_validation") or {}

    if not regime_eval and not shadow_comparison and not shadow_summary:
        return

    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Regime Rollout Validation")

    policy_alert_count = int(shadow_summary.get("policy_alert_count") or 0)
    if latest_session.get("policy_alert"):
        breaches = ", ".join(latest_session.get("breached_limits") or []) or "policy limits"
        st.error(f"Session alert: disagreement rates breached policy limits for {latest_session.get('session_date') or '-'} ({breaches}).")
    elif policy_alert_count > 0:
        st.warning(f"Historical rollout alert sessions detected: {policy_alert_count}. Review the session summary below.")
    elif latest_session.get("alert_level") == "WATCH":
        st.warning("Session disagreement is elevated versus the clean baseline. Review the rollout diagnostics.")
    elif latest_session:
        st.success("Latest session is within rollout policy limits.")

    headline_cols = st.columns(7)
    headline_cols[0].metric("Shadow Candidate", _safe_metric_value(regime_eval.get("shadow_candidate_pack")))
    headline_cols[1].metric("Route Reason", _safe_metric_value(regime_eval.get("reason")))
    headline_cols[2].metric("Log Status", _safe_metric_value(result.get("shadow_log_status")))
    headline_cols[3].metric("Latest Session", _safe_metric_value(latest_session.get("session_date")))
    headline_cols[4].metric("Alert Level", _safe_metric_value(latest_session.get("alert_level")))
    headline_cols[5].metric("Policy Alerts", _safe_metric_value(policy_alert_count))
    headline_cols[6].metric("Total Evaluations", _safe_metric_value(shadow_summary.get("shadow_event_count")))

    if shadow_comparison:
        detail_cols = st.columns(4)
        detail_cols[0].metric("Validation Status", _safe_metric_value(shadow_comparison.get("validation_status")))
        detail_cols[1].metric("Delta Strength", _safe_metric_value(shadow_comparison.get("delta_trade_strength")))
        detail_cols[2].metric("Decision Diverged", _safe_metric_value("YES" if shadow_comparison.get("decision_disagreement_flag") else "NO"))
        detail_cols[3].metric("Status Diverged", _safe_metric_value("YES" if shadow_comparison.get("trade_status_disagreement_flag") else "NO"))

    session_rows = shadow_summary.get("session_validation_summary") or []
    if session_rows:
        session_frame = pd.DataFrame(session_rows)
        preferred_cols = [
            "session_date",
            "baseline_pack_name",
            "shadow_pack_name",
            "snapshot_count",
            "decision_disagreement_rate",
            "trade_status_disagreement_rate",
            "average_delta_trade_strength",
            "validation_status",
            "alert_level",
            "policy_alert",
            "breached_limits",
            "session_buckets_observed",
        ]
        present_cols = [col for col in preferred_cols if col in session_frame.columns]
        st.caption("Session-by-session shadow validation summary")
        _render_dataframe(session_frame[present_cols] if present_cols else session_frame)
    else:
        st.caption("Session-by-session shadow validation will populate as shadow comparisons are logged.")

    with st.expander("Rollout Diagnostics", expanded=False):
        st.json(
            {
                "regime_pack_evaluation": regime_eval,
                "shadow_comparison": shadow_comparison,
                "latest_session_validation": latest_session,
            },
            expanded=False,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_signal_processing_diagnostics(trade: dict):
    """
    Purpose:
        Render signal processing diagnostics panel showing runtime improvements.

    Context:
        Function inside the `streamlit app` module. Displays time decay, score calibration,
        path filtering, and regime-conditional thresholds applied during signal generation.

    Inputs:
        trade (dict): Trade payload containing signal processing metadata.

    Returns:
        None: Side effect only.

    Notes:
        New dashboard section highlighting the four core signal processing improvements.
    """
    if not trade:
        return

    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Signal Processing Diagnostics")

    time_decay_elapsed = trade.get("time_decay_elapsed_minutes")
    time_decay_factor = trade.get("time_decay_factor")
    time_decay_enabled = trade.get("time_decay_enabled")
    time_decay_applied = trade.get("time_decay_applied")
    time_decay_fallback_used = trade.get("time_decay_fallback_used")
    time_decay_elapsed_source = trade.get("time_decay_elapsed_source")
    path_aware_status = trade.get("path_aware_status")
    path_aware_score_penalty = trade.get("path_aware_score_penalty")
    path_aware_entry_veto = trade.get("path_aware_entry_veto")
    path_aware_mae_zscore = trade.get("path_aware_mae_zscore")
    path_aware_reasons = trade.get("path_aware_reasons")

    effective_size_cap = trade.get("effective_size_cap")
    macro_size_applied = trade.get("macro_size_applied")
    gamma_regime = trade.get("gamma_regime")
    regime_threshold_adjustments = trade.get("regime_threshold_adjustments")
    min_trade_strength_threshold = trade.get("min_trade_strength_threshold")
    min_composite_score_threshold = trade.get("min_composite_score_threshold")

    runtime_composite_score = trade.get("runtime_composite_score")
    score_calibration_enabled = trade.get("score_calibration_enabled")
    score_calibration_applied = trade.get("score_calibration_applied")
    score_calibration_backend = trade.get("score_calibration_backend")
    score_calibration_artifact_path = trade.get("score_calibration_artifact_path")

    if not time_decay_enabled:
        time_decay_status = "Disabled"
    elif time_decay_applied and time_decay_fallback_used:
        time_decay_status = "Fallback Applied"
    elif time_decay_applied:
        time_decay_status = "Applied"
    else:
        time_decay_status = "Enabled / Baseline"
    if score_calibration_enabled:
        runtime_score_status = "Applied" if score_calibration_applied else "Enabled / Fallback"
    else:
        runtime_score_status = "Disabled"
    path_filter_status = path_aware_status or "Unavailable"
    regime_adjustment_active = bool(regime_threshold_adjustments) or (macro_size_applied is True)
    regime_adjustment_status = "Adjusted" if regime_adjustment_active else "Baseline"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">Time Decay</div>
                <div class="oqe-summary-value">{}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">
                    {} min | factor {}
                </div>
            </div>
            """.format(
                esc(time_decay_status),
                esc("-" if time_decay_elapsed is None else round(time_decay_elapsed, 1)),
                esc("-" if time_decay_factor is None else round(time_decay_factor, 3))
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">Runtime Score</div>
                <div class="oqe-summary-value">{}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">
                    Composite: {}
                </div>
            </div>
            """.format(
                esc(runtime_score_status),
                esc("-" if runtime_composite_score is None else round(runtime_composite_score, 0))
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">Path Filtering</div>
                <div class="oqe-summary-value">{}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">
                    Penalty: {} | veto: {}
                </div>
            </div>
            """.format(
                esc(path_filter_status),
                esc("-" if path_aware_score_penalty is None else path_aware_score_penalty),
                esc("YES" if path_aware_entry_veto else "NO")
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">Regime Adjustments</div>
                <div class="oqe-summary-value">{}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">
                    {} | cap: {}
                </div>
            </div>
            """.format(
                esc(regime_adjustment_status),
                esc(gamma_regime or "-"),
                esc("-" if effective_size_cap is None else round(effective_size_cap, 2))
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Detailed Signal Processing", expanded=False):
        detail_cols = st.columns(2)

        with detail_cols[0]:
            st.markdown("**Time Decay Model**")
            st.json({
                "time_decay_enabled": time_decay_enabled,
                "time_decay_applied": time_decay_applied,
                "time_decay_fallback_used": time_decay_fallback_used,
                "time_decay_elapsed_source": time_decay_elapsed_source,
                "elapsed_minutes": None if time_decay_elapsed is None else round(time_decay_elapsed, 2),
                "decay_factor": None if time_decay_factor is None else round(time_decay_factor, 6),
            }, expanded=False)

        with detail_cols[1]:
            st.markdown("**Score Calibration**")
            st.json({
                "score_calibration_enabled": score_calibration_enabled,
                "score_calibration_applied": score_calibration_applied,
                "score_calibration_backend": score_calibration_backend,
                "score_calibration_artifact_path": score_calibration_artifact_path,
                "runtime_composite_score": None if runtime_composite_score is None else round(runtime_composite_score, 0),
            }, expanded=False)

        with detail_cols[0]:
            st.markdown("**Path-Aware Filtering**")
            st.json({
                "path_aware_status": path_aware_status,
                "path_aware_score_penalty": path_aware_score_penalty,
                "path_aware_entry_veto": path_aware_entry_veto,
                "path_aware_mae_zscore": None if path_aware_mae_zscore is None else round(path_aware_mae_zscore, 4),
                "path_aware_reasons": path_aware_reasons,
            }, expanded=False)

        with detail_cols[1]:
            st.markdown("**Regime-Conditional Thresholds**")
            st.json({
                "gamma_regime": gamma_regime,
                "effective_size_cap": None if effective_size_cap is None else round(effective_size_cap, 2),
                "size_applied": macro_size_applied,
                "min_trade_strength_threshold": min_trade_strength_threshold,
                "min_composite_score_threshold": min_composite_score_threshold,
                "regime_threshold_adjustments": regime_threshold_adjustments,
            }, expanded=False)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_macro_lab(macro_news_state: dict, headline_records: pd.DataFrame, macro_event_state: dict):
    """
    Purpose:
        Render macro lab for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        macro_news_state (dict): Headline-driven macro state produced by the news layer.
        headline_records (pd.DataFrame): Input associated with headline records.
        macro_event_state (dict): Scheduled-event state produced by the macro-event layer.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    left, right = st.columns([1.1, 0.9])
    with left:
        _render_macro_news_section(macro_news_state, {
            "provider_name": None,
            "data_available": len(headline_records) > 0,
            "is_stale": macro_news_state.get("neutral_fallback"),
            "neutral_fallback": macro_news_state.get("neutral_fallback"),
            "warnings": [],
            "issues": [],
            "provider_metadata": {},
        })
    with right:
        _render_key_value_table("Scheduled Event State", macro_event_state)

    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Headline Records")
    if headline_records is not None and not headline_records.empty:
        _render_dataframe(headline_records)
        if {"timestamp", "headline"}.issubset(headline_records.columns):
            st.caption(f"Loaded {len(headline_records)} normalized headlines.")
    else:
        st.info("No normalized headline records are available for this run.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_workstation(result: dict):
    """
    Purpose:
        Render workstation for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result (dict): Input associated with result.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    full_trade = result.get("trade") or result.get("trade_audit")
    execution_trade = result.get("execution_trade")
    if execution_trade is None and isinstance(full_trade, dict):
        execution_trade = full_trade.get("execution_trade")
    trade = full_trade or execution_trade

    if execution_trade or trade:
        _render_trade_metrics(execution_trade or trade)
        _render_decision_panel(execution_trade or trade)
        _render_signal_confidence_card(execution_trade or trade)
        _render_overnight_risk_card(execution_trade or trade)
    else:
        st.warning("No trade payload was returned for this snapshot.")

    overview_tab, structure_tab, diagnostics_tab = st.tabs(
        ["Overview", "Structure", "Diagnostics"]
    )

    with overview_tab:
        if trade:
            _render_explainability_scorecard(trade)
            _render_signal_processing_diagnostics(trade)
        _render_regime_rollout_panel(result)

        top_left, top_right = st.columns(2)
        with top_left:
            _render_key_value_table("Spot Validation", result.get("spot_validation", {}))
            _render_key_value_table("Spot Snapshot", result.get("spot_summary", {}))
            _render_key_value_table("Macro Event Risk", result.get("macro_event_state", {}))
        with top_right:
            _render_key_value_table("Option Chain Validation", result.get("option_chain_validation", {}))
            _render_macro_news_section(result.get("macro_news_state", {}), result.get("headline_state", {}))

        global_market_snapshot = result.get("global_market_snapshot", {}) or {}
        global_market_inputs = global_market_snapshot.get("market_inputs", {}) or {}
        lower_left, lower_right = st.columns(2)
        with lower_left:
            _render_key_value_table("Global Risk State", result.get("global_risk_state", {}))
        with lower_right:
            _render_key_value_table(
                "Global Market Snapshot",
                {
                    "provider": global_market_snapshot.get("provider"),
                    "data_available": global_market_snapshot.get("data_available"),
                    "stale": global_market_snapshot.get("stale"),
                    "oil_change_24h": global_market_inputs.get("oil_change_24h"),
                    "US VIX Change 24h": global_market_inputs.get("vix_change_24h"),
                    "India VIX Level": global_market_inputs.get("india_vix_level"),
                    "India VIX Change 24h": global_market_inputs.get("india_vix_change_24h"),
                    "sp500_change_24h": global_market_inputs.get("sp500_change_24h"),
                    "us10y_change_bp": global_market_inputs.get("us10y_change_bp"),
                    "usdinr_change_24h": global_market_inputs.get("usdinr_change_24h"),
                    "warnings": global_market_snapshot.get("warnings"),
                },
            )

        if trade:
            with st.expander("Full Trader View", expanded=False):
                _render_dataframe(result.get("trader_view_rows"))

    with structure_tab:
        _render_option_chain_charts(result.get("option_chain_frame"))
        ranked = result.get("ranked_strikes")
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
            st.subheader(f"Ranked Strikes ({trade.get('selected_expiry') or '-'})")
            _render_enhanced_ranked_strikes(ranked)
            st.markdown("</div>", unsafe_allow_html=True)

    with diagnostics_tab:
        if trade:
            diagnostics = {
                label: value
                for label, value in (
                    ("direction_source", trade.get("direction_source")),
                    ("decision_classification", trade.get("decision_classification")),
                    ("setup_state", trade.get("setup_state")),
                    ("setup_quality", trade.get("setup_quality")),
                    ("watchlist_flag", trade.get("watchlist_flag")),
                    ("watchlist_reason", trade.get("watchlist_reason")),
                    ("no_trade_reason_code", trade.get("no_trade_reason_code")),
                    ("no_trade_reason", trade.get("no_trade_reason")),
                    ("missing_signal_requirements", trade.get("missing_signal_requirements")),
                    ("setup_upgrade_conditions", trade.get("setup_upgrade_conditions")),
                    ("likely_next_trigger", trade.get("likely_next_trigger")),
                    ("option_efficiency_status", trade.get("option_efficiency_status")),
                    ("option_efficiency_reason", trade.get("option_efficiency_reason")),
                    ("global_risk_status", trade.get("global_risk_status")),
                    ("global_risk_reason", trade.get("global_risk_reason")),
                    ("macro_news_status", trade.get("macro_news_status")),
                    ("macro_news_reason", trade.get("macro_news_reason")),
                    ("India VIX Level", trade.get("india_vix_level")),
                    ("India VIX Change 24h", trade.get("india_vix_change_24h")),
                    ("macro_adjustment_reasons", trade.get("macro_adjustment_reasons")),
                    ("macro_regime_reasons", trade.get("macro_regime_reasons")),
                    ("confirmation_status", trade.get("confirmation_status")),
                    ("confirmation_reasons", trade.get("confirmation_reasons")),
                    ("scoring_breakdown", trade.get("scoring_breakdown")),
                    ("option_chain_validation", trade.get("option_chain_validation")),
                    ("spot_validation", trade.get("spot_validation")),
                )
                if value is not None
            }
            _render_key_value_list("Trade Diagnostics", diagnostics)

        _render_key_value_list(
            "Run Metadata",
            {
                "mode": result.get("mode"),
                "source": result.get("source"),
                "symbol": result.get("symbol"),
                "replay_paths": result.get("replay_paths"),
                "saved_paths": result.get("saved_paths"),
            },
        )

        st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
        st.subheader("Option Chain Preview")
        _render_dataframe(pd.DataFrame(result.get("option_chain_preview", [])))
        st.markdown("</div>", unsafe_allow_html=True)


def _render_run_paths(result: dict):
    """
    Purpose:
        Render run paths for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result (dict): Input associated with result.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    replay_paths = result.get("replay_paths") or {}
    saved_paths = result.get("saved_paths") or {}

    if not replay_paths and not saved_paths:
        return

    st.markdown('<div class="oqe-path-card">', unsafe_allow_html=True)
    if replay_paths:
        st.markdown('<div class="oqe-path-label">Replay Spot</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="oqe-path-value">{_display_path(replay_paths.get("spot"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="oqe-path-label" style="margin-top:0.55rem;">Replay Option Chain</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="oqe-path-value">{_display_path(replay_paths.get("chain"))}</div>', unsafe_allow_html=True)
    if saved_paths:
        st.markdown('<div class="oqe-path-label" style="margin-top:0.75rem;">Saved Spot Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="oqe-path-value">{_display_path(saved_paths.get("spot"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="oqe-path-label" style="margin-top:0.55rem;">Saved Option Chain Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="oqe-path-value">{_display_path(saved_paths.get("chain"))}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_replay_selection_banner(result: dict):
    """Render transparent replay-file selection diagnostics in replay mode."""
    if (result.get("mode") or "").upper() != "REPLAY":
        return

    selection = result.get("replay_selection") or {}
    if not isinstance(selection, dict) or not selection:
        return

    reason = selection.get("selection_reason") or "unknown"
    source = selection.get("source_label") or result.get("source") or "-"
    replay_paths = result.get("replay_paths") or {}
    spot = selection.get("spot_path") or replay_paths.get("spot")
    chain = selection.get("chain_path") or replay_paths.get("chain")
    skipped = selection.get("skipped_chain_files") or []

    reason_map = {
        "manual_selection": "Manual replay files selected in sidebar.",
        "latest_valid_for_source": "Auto-selected latest valid chain snapshot for the selected source.",
        "latest_valid_any_source": "Auto-selected latest valid chain snapshot across all sources.",
        "no_valid_chain_snapshot": "No valid chain snapshot found after skipping empty/invalid files.",
        "replay_dir_missing": "Replay directory not found.",
    }
    message = reason_map.get(reason, f"Replay selection reason: {reason}")

    st.info(
        f"Replay Selection: {message} | Source={source} | Spot={_display_path(spot)} | Chain={_display_path(chain)}"
    )
    if skipped:
        with st.expander(f"Skipped invalid chain files ({len(skipped)})", expanded=False):
            _render_dataframe(pd.DataFrame(skipped))


def _render_replay_tools(result: dict):
    """
    Purpose:
        Render replay tools for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        result (dict): Input associated with result.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    replay_paths = result.get("replay_paths") or {}
    saved_paths = result.get("saved_paths") or {}

    left, right = st.columns(2)
    with left:
        _render_key_value_table(
            "Snapshot Context",
            {
                "mode": result.get("mode"),
                "source": result.get("source"),
                "symbol": result.get("symbol"),
                "timestamp": (result.get("spot_summary") or {}).get("timestamp"),
                "option_chain_rows": result.get("option_chain_rows"),
            },
        )
    with right:
        _render_key_value_table(
            "Replay / Saved Files",
            {
                "replay_spot": _display_path(replay_paths.get("spot")),
                "replay_chain": _display_path(replay_paths.get("chain")),
                "saved_spot": _display_path(saved_paths.get("spot")),
                "saved_chain": _display_path(saved_paths.get("chain")),
            },
        )


def _render_research_metric_card(label: str, value: str):
    """
    Purpose:
        Render the research metric card for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        label (str): Human-readable label shown to the operator or attached to a record.
        value (str): Raw value supplied by the caller.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    st.markdown(
        f"""
        <div class="oqe-summary-card">
            <div class="oqe-summary-label">{esc(label)}</div>
            <div class="oqe-summary-value">{esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_research_table(title: str, frame: pd.DataFrame, *, caption: str | None = None):
    """
    Purpose:
        Render research table for operator-facing or report output.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        title (str): Input associated with title.
        frame (pd.DataFrame): Input associated with frame.
        caption (str | None): Input associated with caption.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.caption(caption)
    if frame is None or frame.empty:
        st.info("No research data is available for this view yet.")
    else:
        _render_dataframe(frame)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_signal_research_dashboard():
    """
    Purpose:
        Render the signal research dashboard for reporting or presentation.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        None: Side effect only.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    dataset = load_signals_dataset()

    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Signal Evaluation Summary")
    st.caption(f"Canonical dataset: {_display_path(str(SIGNAL_DATASET_PATH))}")

    if dataset.empty:
        st.info("No signal evaluation rows are available yet. Run a live or replay snapshot to start building the research dataset.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    report = build_research_report(dataset)
    total_signals = len(dataset)
    scored_signals = int(pd.to_numeric(dataset.get("composite_signal_score", pd.Series(dtype="float64")), errors="coerce").notna().sum())
    completed_outcomes = int(dataset.get("outcome_status", pd.Series(dtype="object")).astype(str).str.upper().eq("COMPLETE").sum())
    avg_composite_score = _safe_float(pd.to_numeric(dataset.get("composite_signal_score", pd.Series(dtype="float64")), errors="coerce").mean())
    avg_move_probability = _safe_float(pd.to_numeric(dataset.get("move_probability", pd.Series(dtype="float64")), errors="coerce").mean())
    avg_hit_rate = _safe_float(pd.to_numeric(report["move_probability_calibration"].get("actual_hit_rate", pd.Series(dtype="float64")), errors="coerce").mean())

    metric_cols = st.columns(6)
    metric_values = [
        ("Signals", f"{total_signals}"),
        ("Scored Rows", f"{scored_signals}"),
        ("Complete Outcomes", f"{completed_outcomes}"),
        ("Avg Composite", "-" if avg_composite_score is None else f"{avg_composite_score:.2f}"),
        ("Avg Move Prob", "-" if avg_move_probability is None else f"{avg_move_probability:.2%}"),
        ("Avg Hit Rate", "-" if avg_hit_rate is None else f"{avg_hit_rate:.2%}"),
    ]
    for column, (label, value) in zip(metric_cols, metric_values):
        with column:
            _render_research_metric_card(label, value)

    st.markdown("</div>", unsafe_allow_html=True)

    score_cols = st.columns([1.25, 1.25, 1.1])
    with score_cols[0]:
        _render_research_table(
            "Average Score by Signal Quality",
            report["average_score_by_signal_quality"],
            caption="Shows how the component and composite scores behave across quality buckets.",
        )
    with score_cols[1]:
        _render_research_table(
            "Move Probability Calibration",
            report["move_probability_calibration"],
            caption="Compares predicted move probability buckets with realized hit behavior.",
        )
    with score_cols[2]:
        _render_research_table(
            "Average Realized Return by Horizon",
            report["average_realized_return_by_horizon"],
            caption="Helps assess how quickly signals monetize after capture.",
        )

    performance_cols = st.columns(2)
    with performance_cols[0]:
        _render_research_table(
            "Hit Rate by Trade Strength",
            report["hit_rate_by_trade_strength"],
            caption="Useful for checking whether stronger setups are actually more reliable.",
        )
    with performance_cols[1]:
        _render_research_table(
            "Hit Rate by Macro Regime",
            report["hit_rate_by_macro_regime"],
            caption="Highlights whether some macro states are materially better for signal performance.",
        )

    regime_cols = st.columns([1.2, 0.8])
    with regime_cols[0]:
        _render_research_table(
            "Top Regime Fingerprints",
            report["regime_fingerprint_performance"],
            caption="Condition clusters that currently show the strongest composite scores and hit rates.",
        )
    with regime_cols[1]:
        _render_research_table(
            "Signal Count by Regime",
            report["signal_count_by_regime"],
            caption="Quick coverage view so we can separate robust clusters from thin samples.",
        )

    # ── ML Research Panel (optional, non-intrusive) ─────────────────
    _render_ml_research_panel(dataset)


def _render_ml_research_panel(dataset: pd.DataFrame):
    """Render optional ML research metrics panel. Safe no-op if ML is disabled."""
    try:
        from research.ml_models.ml_config import ML_RESEARCH_ENABLED
        if not ML_RESEARCH_ENABLED:
            return
    except ImportError:
        return

    with st.expander("ML Research (Dual-Model Shadow)", expanded=False):
        st.caption("GBT ranking + LogReg calibration — observational only, does NOT affect trades")

        ml_cols = ["ml_rank_score", "ml_confidence_score", "ml_rank_bucket",
                    "ml_confidence_bucket", "ml_agreement_with_engine"]

        if not all(c in dataset.columns for c in ml_cols[:2]):
            st.info("ML scores not yet available in dataset. Run evaluation to populate.")
            return

        scored = dataset.copy()
        scored["_ml_rank_score_num"] = pd.to_numeric(scored["ml_rank_score"], errors="coerce")
        scored["_ml_confidence_score_num"] = pd.to_numeric(
            scored.get("ml_confidence_score"),
            errors="coerce",
        )
        scored = scored[scored["_ml_rank_score_num"].notna()]
        if scored.empty:
            st.info("No signals with ML scores yet.")
            return

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            avg_rank = scored["_ml_rank_score_num"].mean()
            st.metric("Avg Rank Score", f"{avg_rank:.3f}" if avg_rank == avg_rank else "—")
        with m2:
            avg_conf = scored["_ml_confidence_score_num"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.3f}" if pd.notna(avg_conf) else "—")
        with m3:
            if "ml_agreement_with_engine" in scored.columns:
                agree_pct = (scored["ml_agreement_with_engine"].astype(str).str.upper() == "YES").mean()
                st.metric("Agreement Rate", f"{agree_pct:.1%}")
            else:
                st.metric("Agreement Rate", "—")
        with m4:
            st.metric("ML Coverage", f"{len(scored)}/{len(dataset)}")

        if "ml_rank_bucket" in scored.columns:
            bucket_summary = (
                scored.groupby("ml_rank_bucket")
                .agg(
                    n=("_ml_rank_score_num", "count"),
                    avg_rank=("_ml_rank_score_num", "mean"),
                    avg_conf=("_ml_confidence_score_num", "mean"),
                )
                .round(4)
                .reset_index()
            )
            st.dataframe(bucket_summary, use_container_width=True, hide_index=True)


def _inject_autorefresh(interval_seconds: int):
    """
    Purpose:
        Process inject autorefresh for downstream use.
    
    Context:
        Internal helper within the application layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        interval_seconds (int): Input associated with interval seconds.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    interval_ms = max(int(interval_seconds * 1000), 5000)
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {interval_ms});
        </script>
        """,
        height=0,
        width=0,
    )


def main():
    """
    Purpose:
        Run the module entry point for command-line or operational execution.

    Context:
        Function inside the `streamlit app` module. The module sits in the application layer that presents runtime state through operator-facing interfaces and sinks.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Exit status or workflow result returned by the implementation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    st.markdown(
        """
        <div class="oqe-hero">
            <div class="oqe-hero-title">Options Quant Engine</div>
            <div class="oqe-hero-subtitle">Trader workstation for live analysis, replay inspection, and macro/news review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    seed_control_state()

    with st.sidebar:
        st.header("Run Controls")
        mode = st.radio("Mode", ["LIVE", "REPLAY"], horizontal=True, key="control_mode")
        st.text_input("Symbol", key="control_symbol", on_change=_normalize_symbol_input_state)
        symbol = (st.session_state.get("control_symbol") or DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL

        auto_refresh = False
        refresh_seconds = 30

        if mode == "LIVE":
            st.caption("Live broker/public-data snapshot")
            live_source = st.session_state.get("control_source", DEFAULT_DATA_SOURCE)
            if live_source not in DATA_SOURCE_OPTIONS:
                live_source = DEFAULT_DATA_SOURCE
                st.session_state["control_source"] = live_source
            source_index = DATA_SOURCE_OPTIONS.index(live_source)
            source = st.selectbox("Data Source", DATA_SOURCE_OPTIONS, index=source_index, key="control_source")
            save_live_snapshots = st.checkbox("Save live snapshots", key="control_save_live_snapshots")
            auto_refresh = st.checkbox("Auto-refresh live view", key="control_auto_refresh")
            if auto_refresh:
                refresh_seconds = st.slider("Refresh every (sec)", min_value=10, max_value=300, step=5, key="control_refresh_seconds")
            else:
                refresh_seconds = int(st.session_state.get("control_refresh_seconds", 30))
        else:
            st.caption("Replay from saved market snapshots")
            replay_dir_options = _list_replay_directories()
            saved_replay_dir = st.session_state.get("control_replay_dir", "debug_samples")
            replay_dir_index = replay_dir_options.index(saved_replay_dir) if saved_replay_dir in replay_dir_options else 0
            replay_dir = st.selectbox("Replay Directory", replay_dir_options, index=replay_dir_index, key="control_replay_dir")
            source_labels = _list_replay_source_labels(replay_dir, symbol)
            saved_source = st.session_state.get("control_source", source_labels[0] if source_labels else "ANY")
            source_index = source_labels.index(saved_source) if saved_source in source_labels else 0
            source = st.selectbox("Replay Source Label", source_labels, index=source_index, key="control_source")
            save_live_snapshots = False

        st.divider()
        st.caption("Sizing")
        apply_budget_constraint = st.checkbox("Apply budget constraint", key="control_apply_budget_constraint")
        lot_size = st.number_input("Lot Size", min_value=1, step=1, key="control_lot_size")
        requested_lots = st.number_input("Requested Lots", min_value=1, step=1, key="control_requested_lots")
        max_capital = st.number_input("Max Capital Per Trade", min_value=0.0, step=1000.0, key="control_max_capital")

        provider_credentials = {}
        if mode == "LIVE" and source == "ZERODHA":
            st.divider()
            st.markdown("**Zerodha Credentials**")
            provider_credentials["api_key"] = st.text_input("API Key", value=os.getenv("ZERODHA_API_KEY", ""), type="password")
            provider_credentials["api_secret"] = st.text_input("API Secret", value="", type="password")
            provider_credentials["access_token"] = st.text_input("Access Token", value="", type="password")
        elif mode == "LIVE" and source == "ICICI":
            st.divider()
            st.markdown("**ICICI Breeze Credentials**")
            provider_credentials["api_key"] = st.text_input("API Key", value=os.getenv("ICICI_BREEZE_API_KEY", ""), type="password")
            provider_credentials["secret_key"] = st.text_input("Secret Key", value="", type="password")
            provider_credentials["session_token"] = st.text_input("Session Token", value="", type="password")

        replay_spot = None
        replay_chain = None
        if mode != "REPLAY":
            replay_dir = "debug_samples"
        if mode == "REPLAY":
            st.markdown("**Replay Snapshots**")
            spot_files, _ = _list_replay_files(replay_dir, symbol, "spot")
            requested_source_label = None if source == "ANY" else source
            chain_files, skipped_chain_files = _list_replay_files(
                replay_dir,
                symbol,
                "chain",
                source_label=requested_source_label,
            )
            fallback_used = False
            if not chain_files and requested_source_label is not None:
                chain_files, skipped_chain_files = _list_replay_files(
                    replay_dir,
                    symbol,
                    "chain",
                    source_label=None,
                )
                fallback_used = bool(chain_files)

            default_chain = _select_default_option(chain_files)
            default_spot = _nearest_spot_for_chain(default_chain, spot_files) if default_chain else _select_default_option(spot_files)
            spot_options = [""] + spot_files
            chain_options = [""] + chain_files
            saved_replay_spot = st.session_state.get("control_replay_spot", "")
            saved_replay_chain = st.session_state.get("control_replay_chain", "")
            replay_spot = st.selectbox(
                "Replay Spot Snapshot",
                options=spot_options,
                index=spot_options.index(saved_replay_spot) if saved_replay_spot in spot_options else (spot_options.index(default_spot) if default_spot in spot_options else 0),
                key="control_replay_spot",
                help="Defaults to the nearest spot snapshot for the latest matching option-chain snapshot.",
            )
            replay_chain = st.selectbox(
                "Replay Option Chain Snapshot",
                options=chain_options,
                index=chain_options.index(saved_replay_chain) if saved_replay_chain in chain_options else (chain_options.index(default_chain) if default_chain in chain_options else 0),
                key="control_replay_chain",
                help="Defaults to the latest matching option-chain snapshot.",
            )

            with st.expander("Replay Snapshot Inventory", expanded=False):
                st.write(f"Spot snapshots found: {len(spot_files)}")
                st.write(f"Option-chain snapshots found: {len(chain_files)}")
                if fallback_used:
                    st.info("No valid chain snapshots were found for the selected source label. Showing latest valid chains across any source.")
                if skipped_chain_files:
                    st.warning(f"Skipped {len(skipped_chain_files)} invalid/empty chain snapshot file(s).")
                if replay_spot:
                    st.caption(f"Selected spot: {replay_spot}")
                if replay_chain:
                    st.caption(f"Selected chain: {replay_chain}")

        run_button = st.button("Run Snapshot", type="primary", use_container_width=True)

    persist_control_state(mode, symbol=symbol, source=source, replay_dir=replay_dir if mode == "REPLAY" else None)

    # Detect a reload triggered by the auto-refresh timer (query param survives browser reloads).
    auto_run_triggered = (
        mode == "LIVE"
        and auto_refresh
        and query_param_bool("auto_run", False)
    )
    should_run = run_button or auto_run_triggered

    if not should_run and "last_result" not in st.session_state:
        st.info("Choose settings in the sidebar, then click `Run Snapshot`.")
        return

    if should_run:
        with st.spinner("Running engine snapshot..."):
            st.session_state["last_result"] = run_engine_snapshot(
                symbol=symbol,
                mode=mode,
                source=source,
                apply_budget_constraint=apply_budget_constraint,
                requested_lots=int(requested_lots),
                lot_size=int(lot_size),
                max_capital=float(max_capital),
                provider_credentials=provider_credentials,
                replay_spot=replay_spot or None,
                replay_chain=replay_chain or None,
                replay_dir=replay_dir,
                save_live_snapshots=save_live_snapshots,
            )
        st.session_state["last_mode"] = mode
        st.session_state["last_symbol"] = symbol
        st.session_state["auto_refresh"] = auto_refresh
        st.session_state["refresh_seconds"] = refresh_seconds
        # Arm the next auto-run so the following browser reload also executes automatically.
        if auto_refresh:
            st.query_params["auto_run"] = "1"
        else:
            st.query_params.pop("auto_run", None)

    result = st.session_state.get("last_result")
    if not result:
        return

    if not result.get("ok"):
        st.error(result.get("error", "Engine run failed."))
        return

    _render_runbar(result)
    _render_run_paths(result)
    _render_replay_selection_banner(result)

    snapshot_tab, research_tab, replay_tools_tab, macro_lab_tab = st.tabs(
        ["Current Snapshot", "Signal Research", "Replay Tools", "Macro / News Lab"]
    )

    with snapshot_tab:
        _render_workstation(result)

    with research_tab:
        _render_signal_research_dashboard()

    with replay_tools_tab:
        _render_replay_tools(result)

    with macro_lab_tab:
        _render_macro_lab(
            result.get("macro_news_state", {}),
            result.get("headline_records", pd.DataFrame()),
            result.get("macro_event_state", {}),
        )

    if st.session_state.get("last_mode") == "LIVE" and st.session_state.get("auto_refresh"):
        st.caption(f"Live auto-refresh is enabled every {st.session_state.get('refresh_seconds', 30)} seconds.")
        _inject_autorefresh(st.session_state.get("refresh_seconds", 30))


if __name__ == "__main__":
    main()
