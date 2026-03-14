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

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 220, 176, 0.38), transparent 28%),
            radial-gradient(circle at top right, rgba(182, 225, 204, 0.35), transparent 30%),
            linear-gradient(180deg, #f5efe4 0%, #f7f5ef 45%, #f2f0ea 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3efe6 0%, #ebe6db 100%);
        border-right: 1px solid rgba(30, 41, 59, 0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(30, 41, 59, 0.12);
        border-radius: 0.85rem 0.85rem 0 0;
        color: #334155 !important;
        font-weight: 700;
        padding: 0.65rem 0.95rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important;
        background: rgba(255, 255, 255, 0.88);
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        background: rgba(255, 253, 247, 0.96) !important;
        border-bottom-color: rgba(255, 253, 247, 0.96) !important;
        box-shadow: 0 -2px 0 0 #d97706 inset;
    }
    .stMarkdown, .stText, .stCaption, label, p, li, div {
        color: #1f2937;
    }
    h1, h2, h3 {
        color: #111827 !important;
    }
    .stSelectbox label, .stTextInput label, .stNumberInput label, .stCheckbox label, .stRadio label {
        color: #334155 !important;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] * {
        color: #1f2937;
    }
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] small,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stCaption {
        color: #1f2937 !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="select"] > div,
    div[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
    div[data-testid="stSidebar"] textarea,
    div[data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.92) !important;
        color: #111827 !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] [data-baseweb="select"] div[role="button"],
    div[data-testid="stSidebar"] [data-baseweb="select"] span,
    div[data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder,
    div[data-testid="stSidebar"] textarea::placeholder {
        color: #334155 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #334155 !important;
    }
    div[data-testid="stSidebar"] details summary,
    div[data-testid="stSidebar"] details summary * {
        color: #1f2937 !important;
    }
    div[data-testid="stSidebar"] [role="radiogroup"] label,
    div[data-testid="stSidebar"] .stCheckbox label {
        color: #1f2937 !important;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] hr {
        border-color: rgba(30, 41, 59, 0.12);
    }
    div[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #c96b28 0%, #b45309 100%);
        color: #fff8f1;
        border: none;
        font-weight: 700;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #b45309 0%, #92400e 100%);
        color: white;
    }
    .oqe-path-card {
        border: 1px solid rgba(30, 41, 59, 0.10);
        background: rgba(255, 255, 255, 0.74);
        border-radius: 0.9rem;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1rem;
    }
    .oqe-path-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6b7280;
        margin-bottom: 0.15rem;
    }
    .oqe-path-value {
        font-size: 0.92rem;
        color: #111827;
        word-break: break-word;
    }
    .oqe-hero {
        border: 1px solid rgba(30, 41, 59, 0.12);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        background: rgba(255, 253, 247, 0.85);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .oqe-hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #18212f;
        margin-bottom: 0.15rem;
    }
    .oqe-hero-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
    }
    .oqe-runbar {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .oqe-runpill {
        border-radius: 999px;
        padding: 0.38rem 0.8rem;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(30, 41, 59, 0.10);
        font-size: 0.78rem;
        color: #334155;
        font-weight: 700;
    }
    .oqe-summary-card {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 0.85rem;
        padding: 0.85rem 0.95rem;
        background: rgba(255, 253, 247, 0.88);
        min-height: 90px;
    }
    .oqe-summary-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.28rem;
    }
    .oqe-summary-value {
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.25;
        color: #111827;
        word-break: break-word;
    }
    .oqe-badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        margin-bottom: 0.8rem;
    }
    .oqe-badge-neutral {
        background: rgba(148, 163, 184, 0.18);
        color: #475569;
    }
    .oqe-badge-risk-on {
        background: rgba(34, 197, 94, 0.16);
        color: #166534;
    }
    .oqe-badge-risk-off {
        background: rgba(239, 68, 68, 0.16);
        color: #991b1b;
    }
    .oqe-badge-lockdown {
        background: rgba(249, 115, 22, 0.18);
        color: #9a3412;
    }
    .oqe-badge-trade {
        background: rgba(16, 185, 129, 0.16);
        color: #065f46;
    }
    .oqe-badge-watch {
        background: rgba(245, 158, 11, 0.18);
        color: #92400e;
    }
    .oqe-badge-no-signal {
        background: rgba(100, 116, 139, 0.18);
        color: #475569;
    }
    .oqe-badge-blocked {
        background: rgba(239, 68, 68, 0.16);
        color: #991b1b;
    }
    .oqe-panel {
        border: 1px solid rgba(30, 41, 59, 0.10);
        border-radius: 1rem;
        padding: 0.8rem 0.9rem 0.55rem 0.9rem;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_metric_value(value):
    return "-" if value in (None, "") else value


def _safe_float(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _display_path(path_value: str) -> str:
    if not path_value:
        return "-"
    path = Path(path_value)
    if len(path_value) <= 96:
        return path_value
    return f"{path.parent.name}/{path.name}"


def _badge_class_for_regime(regime: str) -> str:
    regime = (regime or "").upper()
    if regime == "RISK_ON":
        return "oqe-badge-risk-on"
    if regime == "RISK_OFF":
        return "oqe-badge-risk-off"
    if regime == "EVENT_LOCKDOWN":
        return "oqe-badge-lockdown"
    return "oqe-badge-neutral"


def _badge_class_for_trade_status(status: str) -> str:
    status = (status or "").upper()
    if status == "TRADE":
        return "oqe-badge-trade"
    if status == "WATCHLIST":
        return "oqe-badge-watch"
    if status in {"NO_SIGNAL", "DATA_INVALID"}:
        return "oqe-badge-no-signal"
    return "oqe-badge-blocked"


def _list_replay_files(replay_dir: str, symbol: str, kind: str):
    replay_path = Path(replay_dir)
    if not replay_path.exists():
        return []

    symbol = (symbol or "").strip().upper()
    if kind == "spot":
        pattern = f"{symbol}_spot_snapshot_*.json"
    else:
        pattern = f"{symbol}_*_option_chain_snapshot_*"

    return sorted(str(path) for path in replay_path.glob(pattern))


def _extract_snapshot_timestamp(path_str: str):
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2}(?:\.\d+)?\+\d{2}[-:]\d{2})", path_str)
    if not match:
        return None
    token = match.group(1).replace("-", ":", 2)
    token = token.replace("-", ":", 1) if "+" in token and token.rsplit("+", 1)[-1].count("-") == 1 else token
    token = re.sub(r"(?<=\+\d{2})-(?=\d{2}$)", ":", token)
    try:
        return pd.to_datetime(token, errors="coerce")
    except Exception:
        return None


def _select_default_option(options):
    if not options:
        return ""
    return options[-1]


def _nearest_spot_for_chain(chain_path: str, spot_options):
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
    st.markdown(f'<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader(title)
    frame = pd.DataFrame([{"field": key, "value": value} for key, value in values.items()])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_key_value_list(title: str, values: dict):
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
    status = trade.get("trade_status", "UNKNOWN")
    st.markdown(
        f'<div class="oqe-badge {_badge_class_for_trade_status(status)}">{status}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    summary_items = [
        ("Direction", trade.get("direction")),
        ("Strike", trade.get("strike")),
        ("Entry", trade.get("entry_price")),
        ("Strength", trade.get("trade_strength")),
        ("Signal Regime", trade.get("signal_regime")),
    ]
    for col, (label, value) in zip(cols, summary_items):
        col.markdown(
            f"""
            <div class="oqe-summary-card">
                <div class="oqe-summary-label">{label}</div>
                <div class="oqe-summary-value">{_safe_metric_value(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi_cols = st.columns(4)
    lots_value = "-"
    current_lots = trade.get("number_of_lots")
    suggested_lots = trade.get("macro_suggested_lots")
    if current_lots is not None or suggested_lots is not None:
        lots_value = f"{_safe_metric_value(current_lots)} / {_safe_metric_value(suggested_lots)}"

    kpi_cols[0].metric("Signal Quality", _safe_metric_value(trade.get("signal_quality")))
    kpi_cols[1].metric("Data Quality", _safe_metric_value(trade.get("data_quality_score")))
    kpi_cols[2].metric("Lots (Current / Suggested)", lots_value)
    kpi_cols[3].metric("Event Risk", _safe_metric_value(trade.get("macro_event_risk_score")))


def _render_decision_panel(trade: dict):
    focus_cards = [
        ("Direction Source", trade.get("direction_source")),
        ("Execution Regime", trade.get("execution_regime")),
        ("Selected Expiry", trade.get("selected_expiry")),
        ("Target / Stop", f"{trade.get('target')} / {trade.get('stop_loss')}"),
        ("Capital Required", trade.get("capital_required")),
        ("Provider Health", (trade.get("provider_health") or {}).get("summary_status")),
        ("Macro Adjustment", trade.get("macro_adjustment_score")),
        ("Position Size Multiplier", trade.get("macro_position_size_multiplier")),
    ]
    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Decision Summary")
    for row_start in range(0, len(focus_cards), 3):
        row = focus_cards[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            col.metric(label, _safe_metric_value(value))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_macro_news_section(macro_news_state: dict, headline_state: dict):
    st.markdown(f'<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader("Macro / News Regime")
    regime = macro_news_state.get("macro_regime", "MACRO_NEUTRAL")
    st.markdown(
        f'<div class="oqe-badge {_badge_class_for_regime(regime)}">{regime}</div>',
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
        pd.DataFrame([{"field": key, "value": value} for key, value in summary.items()]),
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
            .assign(signed_gamma=lambda frame: frame["GAMMA"] * frame["OPTION_TYP"].map({"CE": 1.0, "PE": -1.0}).fillna(0.0))
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


def _render_macro_lab(macro_news_state: dict, headline_records: pd.DataFrame, macro_event_state: dict):
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
        st.dataframe(headline_records, use_container_width=True, hide_index=True)
        if {"timestamp", "headline"}.issubset(headline_records.columns):
            st.caption(f"Loaded {len(headline_records)} normalized headlines.")
    else:
        st.info("No normalized headline records are available for this run.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_workstation(result: dict):
    trade = result.get("trade")
    if trade:
        _render_trade_metrics(trade)
        _render_decision_panel(trade)
    else:
        st.warning("No trade payload was returned for this snapshot.")

    overview_tab, structure_tab, diagnostics_tab = st.tabs(
        ["Overview", "Structure", "Diagnostics"]
    )

    with overview_tab:
        top_left, top_right = st.columns(2)
        with top_left:
            _render_key_value_table("Spot Validation", result.get("spot_validation", {}))
            _render_key_value_table("Spot Snapshot", result.get("spot_summary", {}))
            _render_key_value_table("Macro Event Risk", result.get("macro_event_state", {}))
        with top_right:
            _render_key_value_table("Option Chain Validation", result.get("option_chain_validation", {}))
            _render_macro_news_section(result.get("macro_news_state", {}), result.get("headline_state", {}))

        if trade:
            with st.expander("Full Trader View", expanded=False):
                st.dataframe(result.get("trader_view_rows"), use_container_width=True, hide_index=True)

    with structure_tab:
        _render_option_chain_charts(result.get("option_chain_frame"))
        ranked = result.get("ranked_strikes")
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
            st.subheader(f"Ranked Strikes ({trade.get('selected_expiry') or '-'})")
            st.dataframe(ranked, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with diagnostics_tab:
        if trade:
            diagnostics = {
                key: trade.get(key)
                for key in (
                    "direction_source",
                    "macro_adjustment_reasons",
                    "macro_regime_reasons",
                    "confirmation_status",
                    "confirmation_reasons",
                    "scoring_breakdown",
                    "option_chain_validation",
                    "spot_validation",
                )
                if key in trade
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
        st.dataframe(pd.DataFrame(result.get("option_chain_preview", [])), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_run_paths(result: dict):
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


def _render_replay_tools(result: dict):
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
    st.markdown(
        f"""
        <div class="oqe-summary-card">
            <div class="oqe-summary-label">{label}</div>
            <div class="oqe-summary-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_research_table(title: str, frame: pd.DataFrame, *, caption: str | None = None):
    st.markdown('<div class="oqe-panel">', unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.caption(caption)
    if frame is None or frame.empty:
        st.info("No research data is available for this view yet.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_signal_research_dashboard():
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
    scored_signals = int(dataset.get("composite_signal_score", pd.Series(dtype="float64")).notna().sum())
    completed_outcomes = int(dataset.get("outcome_status", pd.Series(dtype="object")).astype(str).str.upper().eq("COMPLETE").sum())
    avg_composite_score = _safe_float(dataset.get("composite_signal_score", pd.Series(dtype="float64")).mean())
    avg_move_probability = _safe_float(dataset.get("move_probability", pd.Series(dtype="float64")).mean())
    avg_hit_rate = _safe_float(report["move_probability_calibration"].get("actual_hit_rate", pd.Series(dtype="float64")).mean())

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


def _inject_autorefresh(interval_seconds: int):
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
    st.markdown(
        """
        <div class="oqe-hero">
            <div class="oqe-hero-title">Options Quant Engine</div>
            <div class="oqe-hero-subtitle">Trader workstation for live analysis, replay inspection, and macro/news review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Controls")
        mode = st.radio("Mode", ["LIVE", "REPLAY"], index=0, horizontal=True)
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL

        auto_refresh = False
        refresh_seconds = 30

        if mode == "LIVE":
            st.caption("Live broker/public-data snapshot")
            source = st.selectbox("Data Source", DATA_SOURCE_OPTIONS, index=DATA_SOURCE_OPTIONS.index(DEFAULT_DATA_SOURCE))
            save_live_snapshots = st.checkbox("Save live snapshots", value=False)
            auto_refresh = st.checkbox("Auto-refresh live view", value=False)
            if auto_refresh:
                refresh_seconds = st.slider("Refresh every (sec)", min_value=10, max_value=300, value=30, step=5)
        else:
            st.caption("Replay from saved market snapshots")
            source = st.text_input("Replay Source Label", value="REPLAY").strip().upper() or "REPLAY"
            save_live_snapshots = False

        st.divider()
        st.caption("Sizing")
        apply_budget_constraint = st.checkbox("Apply budget constraint", value=False)
        lot_size = st.number_input("Lot Size", min_value=1, value=int(LOT_SIZE), step=1)
        requested_lots = st.number_input("Requested Lots", min_value=1, value=int(NUMBER_OF_LOTS), step=1)
        max_capital = st.number_input("Max Capital Per Trade", min_value=0.0, value=float(MAX_CAPITAL_PER_TRADE), step=1000.0)

        provider_credentials = {}
        if mode == "LIVE" and source == "ZERODHA":
            st.divider()
            st.markdown("**Zerodha Credentials**")
            provider_credentials["api_key"] = st.text_input("API Key", value=os.getenv("ZERODHA_API_KEY", ""))
            provider_credentials["api_secret"] = st.text_input("API Secret", value="", type="password")
            provider_credentials["access_token"] = st.text_input("Access Token", value="", type="password")
        elif mode == "LIVE" and source == "ICICI":
            st.divider()
            st.markdown("**ICICI Breeze Credentials**")
            provider_credentials["api_key"] = st.text_input("API Key", value=os.getenv("ICICI_BREEZE_API_KEY", ""))
            provider_credentials["secret_key"] = st.text_input("Secret Key", value="", type="password")
            provider_credentials["session_token"] = st.text_input("Session Token", value="", type="password")

        replay_spot = None
        replay_chain = None
        replay_dir = "debug_samples"
        if mode == "REPLAY":
            st.markdown("**Replay Inputs**")
            replay_dir = st.text_input("Replay Directory", value="debug_samples").strip() or "debug_samples"
            spot_files = _list_replay_files(replay_dir, symbol, "spot")
            chain_files = _list_replay_files(replay_dir, symbol, "chain")
            default_chain = _select_default_option(chain_files)
            default_spot = _nearest_spot_for_chain(default_chain, spot_files) if default_chain else _select_default_option(spot_files)
            spot_options = [""] + spot_files
            chain_options = [""] + chain_files
            replay_spot = st.selectbox(
                "Replay Spot Snapshot",
                options=spot_options,
                index=spot_options.index(default_spot) if default_spot in spot_options else 0,
                help="Defaults to the nearest spot snapshot for the latest matching option-chain snapshot.",
            )
            replay_chain = st.selectbox(
                "Replay Option Chain Snapshot",
                options=chain_options,
                index=chain_options.index(default_chain) if default_chain in chain_options else 0,
                help="Defaults to the latest matching option-chain snapshot.",
            )

            with st.expander("Replay Snapshot Inventory", expanded=False):
                st.write(f"Spot snapshots found: {len(spot_files)}")
                st.write(f"Option-chain snapshots found: {len(chain_files)}")
                if replay_spot:
                    st.caption(f"Selected spot: {replay_spot}")
                if replay_chain:
                    st.caption(f"Selected chain: {replay_chain}")

        run_button = st.button("Run Snapshot", type="primary", use_container_width=True)

    if not run_button and "last_result" not in st.session_state:
        st.info("Choose settings in the sidebar, then click `Run Snapshot`.")
        return

    if run_button:
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

    result = st.session_state.get("last_result")
    if not result:
        return

    if not result.get("ok"):
        st.error(result.get("error", "Engine run failed."))
        return

    _render_runbar(result)
    _render_run_paths(result)

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
