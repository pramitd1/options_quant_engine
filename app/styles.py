"""
Module: styles.py

Purpose:
    Centralize CSS styles for the Streamlit operator interface.

Role in the System:
    Part of the application layer. Keeps presentation styles separate from
    rendering logic and state management.
"""

OQE_GLOBAL_CSS = """
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
.oqe-mini-scorecard {
    border: 1px solid rgba(30, 41, 59, 0.10);
    border-radius: 1rem;
    background: rgba(255, 255, 255, 0.74);
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.85rem;
}
.oqe-mini-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 0.2rem;
    font-weight: 700;
}
.oqe-mini-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
    word-break: break-word;
}
.oqe-decision-watchlist {
    background: rgba(245, 158, 11, 0.18);
    color: #92400e;
}
.oqe-decision-inactive {
    background: rgba(100, 116, 139, 0.18);
    color: #334155;
}
.oqe-decision-blocked {
    background: rgba(239, 68, 68, 0.16);
    color: #991b1b;
}
.oqe-decision-ready {
    background: rgba(16, 185, 129, 0.16);
    color: #065f46;
}

/* Overnight risk card */
.oqe-overnight-card {
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
.oqe-overnight-label {
    font-weight: 700;
    font-size: 1.1em;
}
.oqe-overnight-confidence {
    margin-left: 12px;
    font-size: 0.9em;
    color: #6b7280;
}
.oqe-overnight-yes {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.oqe-overnight-yes .oqe-overnight-label { color: #166534; }
.oqe-overnight-caution {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.oqe-overnight-caution .oqe-overnight-label { color: #92400e; }
.oqe-overnight-no {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.oqe-overnight-no .oqe-overnight-label { color: #991b1b; }
.oqe-overnight-unknown {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.oqe-overnight-unknown .oqe-overnight-label { color: #475569; }

/* Signal confidence card */
.oqe-confidence-card {
    background: rgba(255, 253, 247, 0.88);
    border: 1px solid rgba(30, 41, 59, 0.12);
    border-radius: 1rem;
    padding: 12px 16px;
    margin: 8px 0;
    text-align: center;
}
.oqe-confidence-title {
    color: var(--oqe-conf-color, #475569);
    font-weight: 700;
    font-size: 1em;
    margin-bottom: 4px;
}

/* ── Dark-mode overrides ── */
[data-theme="dark"] .stApp,
.stApp[data-theme="dark"] {
    background:
        radial-gradient(circle at top left, rgba(180, 120, 60, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(60, 130, 90, 0.12), transparent 30%),
        linear-gradient(180deg, #0e1117 0%, #131720 45%, #0e1117 100%) !important;
}
[data-theme="dark"] section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131720 0%, #181d28 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-theme="dark"] .stMarkdown,
[data-theme="dark"] .stText,
[data-theme="dark"] .stCaption,
[data-theme="dark"] label,
[data-theme="dark"] p,
[data-theme="dark"] li,
[data-theme="dark"] div {
    color: #e2e8f0;
}
[data-theme="dark"] h1,
[data-theme="dark"] h2,
[data-theme="dark"] h3 {
    color: #f1f5f9 !important;
}
[data-theme="dark"] .stSelectbox label,
[data-theme="dark"] .stTextInput label,
[data-theme="dark"] .stNumberInput label,
[data-theme="dark"] .stCheckbox label,
[data-theme="dark"] .stRadio label {
    color: #cbd5e1 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
[data-theme="dark"] div[data-testid="stSidebar"] p,
[data-theme="dark"] div[data-testid="stSidebar"] span,
[data-theme="dark"] div[data-testid="stSidebar"] small,
[data-theme="dark"] div[data-testid="stSidebar"] label,
[data-theme="dark"] div[data-testid="stSidebar"] .stCaption {
    color: #e2e8f0 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
[data-theme="dark"] div[data-testid="stSidebar"] textarea,
[data-theme="dark"] div[data-testid="stSidebar"] input {
    background: rgba(30, 41, 59, 0.85) !important;
    color: #f1f5f9 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="select"] div[role="button"],
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder,
[data-theme="dark"] div[data-testid="stSidebar"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #94a3b8 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] details summary,
[data-theme="dark"] div[data-testid="stSidebar"] details summary * {
    color: #e2e8f0 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] [role="radiogroup"] label,
[data-theme="dark"] div[data-testid="stSidebar"] .stCheckbox label {
    color: #e2e8f0 !important;
}
[data-theme="dark"] div[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1);
}
[data-theme="dark"] div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    color: #fff;
}
[data-theme="dark"] div[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: #fff;
}
[data-theme="dark"] .stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.55);
    border-color: rgba(255, 255, 255, 0.10);
    color: #cbd5e1 !important;
}
[data-theme="dark"] .stTabs [data-baseweb="tab"]:hover {
    color: #f1f5f9 !important;
    background: rgba(30, 41, 59, 0.75);
}
[data-theme="dark"] .stTabs [aria-selected="true"] {
    color: #f1f5f9 !important;
    background: rgba(15, 23, 42, 0.85) !important;
    border-bottom-color: rgba(15, 23, 42, 0.85) !important;
    box-shadow: 0 -2px 0 0 #f59e0b inset;
}
[data-theme="dark"] .oqe-hero {
    background: rgba(15, 23, 42, 0.72);
    border-color: rgba(255, 255, 255, 0.08);
}
[data-theme="dark"] .oqe-hero-title {
    color: #f1f5f9;
}
[data-theme="dark"] .oqe-hero-subtitle {
    color: #94a3b8;
}
[data-theme="dark"] .oqe-runpill {
    background: rgba(30, 41, 59, 0.65);
    border-color: rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
}
[data-theme="dark"] .oqe-panel {
    background: rgba(15, 23, 42, 0.55);
    border-color: rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
}
[data-theme="dark"] .oqe-summary-card {
    background: rgba(30, 41, 59, 0.65);
    border-color: rgba(255, 255, 255, 0.1);
}
[data-theme="dark"] .oqe-summary-label {
    color: #94a3b8;
}
[data-theme="dark"] .oqe-summary-value {
    color: #f1f5f9;
}
[data-theme="dark"] .oqe-mini-scorecard {
    background: rgba(30, 41, 59, 0.6);
    border-color: rgba(255, 255, 255, 0.08);
}
[data-theme="dark"] .oqe-mini-title {
    color: #94a3b8;
}
[data-theme="dark"] .oqe-mini-value {
    color: #f1f5f9;
}
[data-theme="dark"] .oqe-path-card {
    background: rgba(30, 41, 59, 0.55);
    border-color: rgba(255, 255, 255, 0.08);
}
[data-theme="dark"] .oqe-path-label {
    color: #94a3b8;
}
[data-theme="dark"] .oqe-path-value {
    color: #e2e8f0;
}
[data-theme="dark"] .oqe-badge-neutral {
    background: rgba(148, 163, 184, 0.22);
    color: #cbd5e1;
}
[data-theme="dark"] .oqe-badge-risk-on {
    background: rgba(34, 197, 94, 0.20);
    color: #4ade80;
}
[data-theme="dark"] .oqe-badge-risk-off {
    background: rgba(239, 68, 68, 0.20);
    color: #f87171;
}
[data-theme="dark"] .oqe-badge-lockdown {
    background: rgba(249, 115, 22, 0.22);
    color: #fb923c;
}
[data-theme="dark"] .oqe-badge-trade {
    background: rgba(16, 185, 129, 0.20);
    color: #34d399;
}
[data-theme="dark"] .oqe-badge-watch {
    background: rgba(245, 158, 11, 0.22);
    color: #fbbf24;
}
[data-theme="dark"] .oqe-badge-no-signal {
    background: rgba(100, 116, 139, 0.22);
    color: #94a3b8;
}
[data-theme="dark"] .oqe-badge-blocked {
    background: rgba(239, 68, 68, 0.20);
    color: #f87171;
}
[data-theme="dark"] .oqe-decision-watchlist {
    background: rgba(245, 158, 11, 0.22);
    color: #fbbf24;
}
[data-theme="dark"] .oqe-decision-inactive {
    background: rgba(100, 116, 139, 0.22);
    color: #cbd5e1;
}
[data-theme="dark"] .oqe-decision-blocked {
    background: rgba(239, 68, 68, 0.20);
    color: #f87171;
}
[data-theme="dark"] .oqe-decision-ready {
    background: rgba(16, 185, 129, 0.20);
    color: #34d399;
}

/* Dark-mode overrides for overnight risk card */
[data-theme="dark"] .oqe-overnight-card {
    border-color: rgba(255, 255, 255, 0.1);
}
[data-theme="dark"] .oqe-overnight-confidence {
    color: #94a3b8;
}
[data-theme="dark"] .oqe-overnight-yes {
    background: rgba(34, 197, 94, 0.18);
}
[data-theme="dark"] .oqe-overnight-yes .oqe-overnight-label { color: #4ade80; }
[data-theme="dark"] .oqe-overnight-caution {
    background: rgba(245, 158, 11, 0.18);
}
[data-theme="dark"] .oqe-overnight-caution .oqe-overnight-label { color: #fbbf24; }
[data-theme="dark"] .oqe-overnight-no {
    background: rgba(239, 68, 68, 0.18);
}
[data-theme="dark"] .oqe-overnight-no .oqe-overnight-label { color: #f87171; }
[data-theme="dark"] .oqe-overnight-unknown {
    background: rgba(100, 116, 139, 0.2);
}
[data-theme="dark"] .oqe-overnight-unknown .oqe-overnight-label { color: #94a3b8; }

/* Dark-mode overrides for signal confidence card */
[data-theme="dark"] .oqe-confidence-card {
    background: rgba(15, 23, 42, 0.72);
    border-color: rgba(255, 255, 255, 0.1);
}
[data-theme="dark"] .oqe-confidence-title {
    color: var(--oqe-conf-dark-color, #94a3b8);
}
[data-theme="dark"] .oqe-confidence-card {
    --oqe-gauge-track: rgba(148, 163, 184, 0.2);
    --oqe-gauge-sublabel: #94a3b8;
}

/* Dark-mode metric and dataframe contrast */
[data-theme="dark"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}
[data-theme="dark"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-theme="dark"] [data-testid="stMetricDelta"] {
    color: #cbd5e1 !important;
}
[data-theme="dark"] .stDataFrame,
[data-theme="dark"] .stTable {
    color: #e2e8f0;
}
[data-theme="dark"] [data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
}
[data-theme="dark"] [data-testid="stExpander"] details {
    border-color: rgba(255, 255, 255, 0.08);
}
/* Dropdown menu options (for selectbox/multiselect) */
[data-theme="dark"] [data-baseweb="menu"] {
    background: #1e293b !important;
}
[data-theme="dark"] [data-baseweb="menu"] li {
    color: #e2e8f0 !important;
}
[data-theme="dark"] [data-baseweb="menu"] li:hover {
    background: rgba(255, 255, 255, 0.08) !important;
}
[data-theme="dark"] [data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.75) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}
[data-theme="dark"] [data-baseweb="select"] div[role="button"],
[data-theme="dark"] [data-baseweb="select"] span {
    color: #f1f5f9 !important;
}
[data-theme="dark"] [data-baseweb="select"] svg {
    fill: #94a3b8 !important;
}
[data-theme="dark"] [data-baseweb="base-input"] > div {
    background: rgba(30, 41, 59, 0.75) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}
[data-theme="dark"] [data-baseweb="base-input"] input {
    color: #f1f5f9 !important;
}
</style>
"""
