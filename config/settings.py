"""
Module: settings.py

Purpose:
    Define repository-wide runtime defaults such as symbols, capital settings, and provider options.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _csv_env_list(name: str) -> list[str]:
    """
    Purpose:
        Process csv env list for downstream use.
    
    Context:
        Internal helper within the configuration layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        name (str): Input associated with name.
    
    Returns:
        list[str]: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_or_placeholder(name: str) -> str:
    """
    Purpose:
        Process env or placeholder for downstream use.
    
    Context:
        Internal helper within the configuration layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        name (str): Input associated with name.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    return os.getenv(name, f"YOUR_{name}")


def get_zerodha_runtime_config() -> dict:
    """
    Purpose:
        Return the Zerodha runtime configuration bundle used by data ingestion.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    return {
        "api_key": _env_or_placeholder("ZERODHA_API_KEY"),
        "api_secret": _env_or_placeholder("ZERODHA_API_SECRET"),
        "access_token": _env_or_placeholder("ZERODHA_ACCESS_TOKEN"),
    }


def get_icici_runtime_config() -> dict:
    """
    Purpose:
        Return the ICICI runtime configuration bundle used by data ingestion.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    return {
        "api_key": _env_or_placeholder("ICICI_BREEZE_API_KEY"),
        "secret_key": _env_or_placeholder("ICICI_BREEZE_SECRET_KEY"),
        "session_token": _env_or_placeholder("ICICI_BREEZE_SESSION_TOKEN"),
    }


# ================================
# Engine Core Settings
# ================================

DEFAULT_SYMBOL = "NIFTY"
DEFAULT_DATA_SOURCE = "ICICI"

# Terminal output verbosity: COMPACT, STANDARD, or FULL_DEBUG
OUTPUT_MODE = os.getenv("OQE_OUTPUT_MODE", "COMPACT").upper().strip()

# Runtime environment label used for production safety checks.
# Accepted production aliases: PROD, PRODUCTION.
RUNTIME_ENV = os.getenv("OQE_RUNTIME_ENV", "DEV").upper().strip()
IS_PRODUCTION = RUNTIME_ENV in {"PROD", "PRODUCTION"}

REFRESH_INTERVAL = 10
NSE_REFRESH_INTERVAL = 12
ICICI_REFRESH_INTERVAL = 8

MAX_RETRIES = 3
QUOTE_BATCH_SIZE = 200


# ================================
# Trading Parameters
# ================================

TARGET_PROFIT_PERCENT = 30
STOP_LOSS_PERCENT = 15

MAX_CAPITAL_PER_TRADE = 50000
RISK_SCORE = 5

LOT_SIZE = 65
NUMBER_OF_LOTS = 1
RISK_FREE_RATE = 0.06
DIVIDEND_YIELD = 0.0


# ================================
# Analytics Thresholds
# ================================

HIGH_GAMMA_THRESHOLD = 1e6
MIN_OPEN_INTEREST = 100
LIQUIDITY_VOID_OI_THRESHOLD = 50
VOL_EXPANSION_THRESHOLD = 1.3


# ================================
# Move Predictor Configuration
# ================================

# Active ML model from the registry. Set to a registry model name
# (e.g. "GBT_shallow_v1") to load that persisted model into the blended
# pipeline ML leg.
# Leave empty/None to skip registry-model loading; the blended path will still
# use the built-in ML heuristic leg (and only falls back to pure rule if the
# ML leg is unavailable at runtime).
ACTIVE_MODEL = os.getenv("OQE_ACTIVE_MODEL", "").strip() or None

# Prediction method — controls the pluggable predictor architecture.
# Options: "blended" (default), "pure_ml", "pure_rule", "research_dual_model",
#          "research_decision_policy", "ev_sizing", "research_rank_gate",
#          "research_uncertainty_adjusted"
# Set via env OQE_PREDICTION_METHOD or override here.
PREDICTION_METHOD = os.getenv("OQE_PREDICTION_METHOD", "blended").strip() or "blended"


# ================================
# Backtesting Configuration
# ================================

BACKTEST_YEARS = 5
INTRADAY_INTERVAL = "1d"

# Data source mode for backtesting / tuning.
# "historical" — real NSE bhav-copy data from data_store/historical/
# "live"       — synthetic (Black-Scholes) chain built on the fly
# "combined"   — historical first, live-synthetic appended for missing dates
BACKTEST_DATA_SOURCE = os.getenv("BACKTEST_DATA_SOURCE", "historical").strip().lower()

BACKTEST_STRIKE_STEP = 50
BACKTEST_STRIKE_RANGE = 10
BACKTEST_DEFAULT_IV = 18.0

BACKTEST_MIN_TRADE_STRENGTH = 25
BACKTEST_SIGNAL_PERSISTENCE = 1
BACKTEST_MAX_HOLD_BARS = 5
BACKTEST_ENTRY_SLIPPAGE_BPS = 10
BACKTEST_EXIT_SLIPPAGE_BPS = 10
BACKTEST_SPREAD_BPS = 15
BACKTEST_COMMISSION_PER_ORDER = 20.0
BACKTEST_ENABLE_BUDGET = False
BACKTEST_STARTING_CAPITAL = 500000

WF_TRAIN_RATIO = 0.7
WF_MIN_TRAIN_SAMPLES = 100

SWEEP_SIGNAL_PERSISTENCE_GRID = [1, 2, 3]
SWEEP_MAX_HOLD_BARS_GRID = [3, 5, 8]
SWEEP_TP_GRID = [20, 30, 40]
SWEEP_SL_GRID = [10, 15, 20]

MC_SIMULATIONS = 1000

MAX_WORKERS = 2
ENABLE_PARALLEL_SWEEP = False
SWEEP_PROGRESS_EVERY = 5


# ================================
# Data Source Configuration
# ================================

DATA_SOURCE_OPTIONS = [
    "NSE",
    "ZERODHA",
    "ICICI",
]


# ================================
# Zerodha API Credentials
# ================================

ZERODHA_API_KEY = _env_or_placeholder("ZERODHA_API_KEY")
ZERODHA_API_SECRET = _env_or_placeholder("ZERODHA_API_SECRET")
ZERODHA_ACCESS_TOKEN = _env_or_placeholder("ZERODHA_ACCESS_TOKEN")

API_KEY = ZERODHA_API_KEY
API_SECRET = ZERODHA_API_SECRET
ACCESS_TOKEN = ZERODHA_ACCESS_TOKEN


# ================================
# ICICI Breeze Credentials
# ================================

ICICI_BREEZE_API_KEY = _env_or_placeholder("ICICI_BREEZE_API_KEY")
ICICI_BREEZE_SECRET_KEY = _env_or_placeholder("ICICI_BREEZE_SECRET_KEY")
ICICI_BREEZE_SESSION_TOKEN = _env_or_placeholder("ICICI_BREEZE_SESSION_TOKEN")

# Optional manual fallback expiry. Leave blank to rely on dynamic expiry generation.
ICICI_DEFAULT_EXPIRY_DATE = os.getenv("ICICI_DEFAULT_EXPIRY_DATE", "").strip()

# Optional manual overrides. Comma-separated ISO timestamps are supported via env vars.
# If these are empty, the loader generates upcoming weekly expiries automatically.
ICICI_SYMBOL_EXPIRY_CANDIDATES = {
    "NIFTY": _csv_env_list("ICICI_NIFTY_EXPIRIES"),
    "BANKNIFTY": _csv_env_list("ICICI_BANKNIFTY_EXPIRIES"),
    "FINNIFTY": _csv_env_list("ICICI_FINNIFTY_EXPIRIES"),
}

NSE_DEBUG = os.getenv("NSE_DEBUG", "false").strip().lower() == "true"
ICICI_DEBUG = os.getenv("ICICI_DEBUG", "false").strip().lower() == "true"


# ================================
# Scheduled Macro Event Risk
# ================================

MACRO_EVENT_FILTER_ENABLED = os.getenv("MACRO_EVENT_FILTER_ENABLED", "true").strip().lower() == "true"
MACRO_EVENT_SCHEDULE_FILE = os.getenv("MACRO_EVENT_SCHEDULE_FILE", "").strip()
MACRO_EVENT_PRE_EVENT_WARNING_MINUTES = int(os.getenv("MACRO_EVENT_PRE_EVENT_WARNING_MINUTES", "180"))
MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES = int(os.getenv("MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES", "30"))
MACRO_EVENT_EVENT_DURATION_MINUTES = int(os.getenv("MACRO_EVENT_EVENT_DURATION_MINUTES", "10"))
MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES = int(os.getenv("MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES", "30"))
MACRO_EVENT_WATCH_RISK_THRESHOLD = int(os.getenv("MACRO_EVENT_WATCH_RISK_THRESHOLD", "45"))
MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD = int(os.getenv("MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD", "70"))

# Optional built-in mock/default schedule. Prefer using MACRO_EVENT_SCHEDULE_FILE
# for local schedules, but keep the interface centralized here.
DEFAULT_MACRO_EVENT_SCHEDULE: list[dict] = []


# ================================
# Headline Ingestion Foundation
# ================================

HEADLINE_INGESTION_ENABLED = os.getenv("HEADLINE_INGESTION_ENABLED", "true").strip().lower() == "true"
HEADLINE_PROVIDER = os.getenv("HEADLINE_PROVIDER", "MOCK").strip().upper()
HEADLINE_MOCK_FILE = os.getenv("HEADLINE_MOCK_FILE", "config/mock_headlines.example.json").strip()
HEADLINE_RSS_URLS = _csv_env_list("HEADLINE_RSS_URLS")
HEADLINE_RSS_TIMEOUT_SECONDS = int(os.getenv("HEADLINE_RSS_TIMEOUT_SECONDS", "8"))
HEADLINE_RSS_USER_AGENT = os.getenv(
    "HEADLINE_RSS_USER_AGENT",
    "options_quant_engine/1.0 (+headline-ingestion-rss)"
).strip()
HEADLINE_STALE_MINUTES = int(os.getenv("HEADLINE_STALE_MINUTES", "45"))
HEADLINE_MAX_RECORDS = int(os.getenv("HEADLINE_MAX_RECORDS", "50"))


# ================================
# Event Intelligence (Options NLP)
# ================================

EVENT_INTELLIGENCE_ENABLED = os.getenv("EVENT_INTELLIGENCE_ENABLED", "true").strip().lower() == "true"
EVENT_INTELLIGENCE_LLM_ENABLED = os.getenv("EVENT_INTELLIGENCE_LLM_ENABLED", "false").strip().lower() == "true"
EVENT_INTELLIGENCE_LLM_PROVIDER = os.getenv("EVENT_INTELLIGENCE_LLM_PROVIDER", "OPENAI").strip().upper()
EVENT_INTELLIGENCE_LLM_MODEL = os.getenv("EVENT_INTELLIGENCE_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
EVENT_INTELLIGENCE_LLM_TIMEOUT_SECONDS = float(os.getenv("EVENT_INTELLIGENCE_LLM_TIMEOUT_SECONDS", "8"))
EVENT_INTELLIGENCE_LLM_TEMPERATURE = float(os.getenv("EVENT_INTELLIGENCE_LLM_TEMPERATURE", "0"))
EVENT_INTELLIGENCE_LLM_MIN_CONFIDENCE = float(os.getenv("EVENT_INTELLIGENCE_LLM_MIN_CONFIDENCE", "0.62"))


# ================================
# Macro / News Aggregation
# ================================

HEADLINE_BURST_LOOKBACK_MINUTES = int(os.getenv("HEADLINE_BURST_LOOKBACK_MINUTES", "60"))
HEADLINE_VELOCITY_BASE_COUNT = int(os.getenv("HEADLINE_VELOCITY_BASE_COUNT", "5"))
MACRO_NEWS_SENTIMENT_ON_THRESHOLD = float(os.getenv("MACRO_NEWS_SENTIMENT_ON_THRESHOLD", "18"))
MACRO_NEWS_SENTIMENT_OFF_THRESHOLD = float(os.getenv("MACRO_NEWS_SENTIMENT_OFF_THRESHOLD", "-18"))
MACRO_NEWS_RISK_BIAS_THRESHOLD = float(os.getenv("MACRO_NEWS_RISK_BIAS_THRESHOLD", "0.22"))


# ================================
# Global Market Risk Feature Data
# ================================

GLOBAL_MARKET_DATA_ENABLED = os.getenv("GLOBAL_MARKET_DATA_ENABLED", "true").strip().lower() == "true"
GLOBAL_MARKET_LOOKBACK_DAYS = int(os.getenv("GLOBAL_MARKET_LOOKBACK_DAYS", "90"))
GLOBAL_MARKET_STALE_DAYS = int(os.getenv("GLOBAL_MARKET_STALE_DAYS", "5"))


def _validate_runtime_settings() -> None:
    allowed_backtest_sources = {"historical", "live", "combined"}
    if BACKTEST_DATA_SOURCE not in allowed_backtest_sources:
        raise ValueError(
            "Invalid BACKTEST_DATA_SOURCE=%r. Allowed values: %s"
            % (BACKTEST_DATA_SOURCE, sorted(allowed_backtest_sources))
        )

    allowed_headline_providers = {"MOCK", "RSS"}
    if HEADLINE_PROVIDER not in allowed_headline_providers:
        raise ValueError(
            "Invalid HEADLINE_PROVIDER=%r. Allowed values: %s"
            % (HEADLINE_PROVIDER, sorted(allowed_headline_providers))
        )

    non_negative_checks = {
        "MACRO_EVENT_PRE_EVENT_WARNING_MINUTES": MACRO_EVENT_PRE_EVENT_WARNING_MINUTES,
        "MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES": MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES,
        "MACRO_EVENT_EVENT_DURATION_MINUTES": MACRO_EVENT_EVENT_DURATION_MINUTES,
        "MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES": MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES,
        "HEADLINE_STALE_MINUTES": HEADLINE_STALE_MINUTES,
    }
    for field_name, field_value in non_negative_checks.items():
        if field_value < 0:
            raise ValueError(f"{field_name} must be >= 0 (got {field_value})")

    positive_checks = {
        "HEADLINE_MAX_RECORDS": HEADLINE_MAX_RECORDS,
        "GLOBAL_MARKET_LOOKBACK_DAYS": GLOBAL_MARKET_LOOKBACK_DAYS,
        "GLOBAL_MARKET_STALE_DAYS": GLOBAL_MARKET_STALE_DAYS,
    }
    for field_name, field_value in positive_checks.items():
        if field_value <= 0:
            raise ValueError(f"{field_name} must be > 0 (got {field_value})")

    if MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD < MACRO_EVENT_WATCH_RISK_THRESHOLD:
        raise ValueError(
            "MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD must be >= "
            "MACRO_EVENT_WATCH_RISK_THRESHOLD"
        )


_validate_runtime_settings()


# ================================
# Logging Configuration
# ================================

LOG_LEVEL = "INFO"
LOG_FILE = "quant_engine.log"


# ================================
# Project Directory Structure
# ================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data_store")
BACKTEST_DIR = os.path.join(BASE_DIR, "backtests")
MODEL_DIR = os.path.join(BASE_DIR, "models_store")
LOG_DIR = os.path.join(BASE_DIR, "logs")
IV_SURFACE_DIR = os.path.join(DATA_DIR, "iv_surface")

for directory in [
    DATA_DIR,
    BACKTEST_DIR,
    MODEL_DIR,
    LOG_DIR,
    IV_SURFACE_DIR
]:
    os.makedirs(directory, exist_ok=True)
