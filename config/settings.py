"""
Global configuration settings for the Options Quant Engine
"""

import os


# ================================
# Engine Core Settings
# ================================

DEFAULT_SYMBOL = "NIFTY"
REFRESH_INTERVAL = 10
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

# Walk-forward ML
WF_TRAIN_RATIO = 0.7
WF_MIN_TRAIN_SAMPLES = 100

# Parameter sweep
SWEEP_SIGNAL_PERSISTENCE_GRID = [1, 2, 3]
SWEEP_MAX_HOLD_BARS_GRID = [3, 5, 8]
SWEEP_TP_GRID = [20, 30, 40]
SWEEP_SL_GRID = [10, 15, 20]

# Monte Carlo
MC_SIMULATIONS = 1000

# Parallel backtest
MAX_WORKERS = 2
ENABLE_PARALLEL_SWEEP = False   # safer default on small laptops / macOS
SWEEP_PROGRESS_EVERY = 5


# ================================
# Data Source Configuration
# ================================

DATA_SOURCE_OPTIONS = [
    "NSE",
    "ZERODHA"
]


# ================================
# Zerodha API Credentials
# ================================

API_KEY = os.getenv("ZERODHA_API_KEY", "YOUR_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET", "YOUR_API_SECRET")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")


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