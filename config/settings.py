"""
Global configuration settings for the Options Quant Engine
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _csv_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_or_placeholder(name: str) -> str:
    return os.getenv(name, f"YOUR_{name}")


def get_zerodha_runtime_config() -> dict:
    return {
        "api_key": _env_or_placeholder("ZERODHA_API_KEY"),
        "api_secret": _env_or_placeholder("ZERODHA_API_SECRET"),
        "access_token": _env_or_placeholder("ZERODHA_ACCESS_TOKEN"),
    }


def get_icici_runtime_config() -> dict:
    return {
        "api_key": _env_or_placeholder("ICICI_BREEZE_API_KEY"),
        "secret_key": _env_or_placeholder("ICICI_BREEZE_SECRET_KEY"),
        "session_token": _env_or_placeholder("ICICI_BREEZE_SESSION_TOKEN"),
    }


# ================================
# Engine Core Settings
# ================================

DEFAULT_SYMBOL = "NIFTY"
DEFAULT_DATA_SOURCE = "NSE"

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
STRIKE_WINDOW_STEPS = 8
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

MOVE_PROB_THRESHOLD = 0.60
LARGE_MOVE_POINTS = (150, 300)


# ================================
# Backtesting Configuration
# ================================

BACKTEST_YEARS = 5
INTRADAY_INTERVAL = "1d"

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
