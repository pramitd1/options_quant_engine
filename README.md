# Options Quant Engine

Options Quant Engine is a live options analytics, signal-generation, and research-evaluation system focused on Indian index and stock options. It combines live option-chain ingestion, dealer/gamma/liquidity analytics, macro/news overlays, terminal and Streamlit interfaces, replay tooling, and a canonical signal research dataset that can be enriched over time with realized market outcomes.

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in provider credentials in `.env` if you want to use Zerodha or ICICI.

### 2. Run the terminal engine

```bash
python main.py
```

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Refresh realized signal outcomes

```bash
python scripts/update_signal_outcomes.py
```

### 5. Run research reports

```bash
python scripts/signal_evaluation_report.py
```

## Status

- Beta phase
- ICICI live flow is currently the most exercised broker path
- Zerodha integration exists, but is less exercised end to end
- NSE public-data mode is useful for research and lightweight monitoring, but remains inherently brittle

## What The System Does

The repository supports three practical workflows:

1. Live signal generation
   - Terminal loop via `main.py`
   - Streamlit app via `app/streamlit_app.py`
2. Replay and validation
   - Snapshot replay via `main.py --replay`
   - Replay regression tooling under `backtest/`
3. Research dataset and evaluation
   - Canonical signal capture into `research/signal_evaluation/signals_dataset.csv`
   - Realized outcome enrichment
   - Signal evaluation scoring
   - Grouped research reporting

## Main Entry Points

- `main.py`: interactive terminal engine
- `app/streamlit_app.py`: Streamlit interface for live or replay snapshots
- `backtest/backtest_runner.py`: historical backtest runner
- `scripts/update_signal_outcomes.py`: refresh pending realized outcomes in the canonical signal dataset
- `scripts/signal_evaluation_report.py`: grouped research reporting over the canonical signal dataset
- `config/generate_token.py`: helper for Zerodha token generation

## Architecture

### Live Engine Flow

1. Spot price and intraday spot context are fetched from `data/spot_downloader.py`
2. Option-chain data is loaded through `data/data_source_router.py`
3. Provider output is normalized in `data/provider_normalization.py`
4. Market data is validated through `data/option_chain_validation.py`
5. `engine/trading_engine.py` computes analytics, direction, scoring, exits, sizing, and final trade status
6. Macro/news overlays are applied through `macro/` and `news/`
7. Terminal or Streamlit views render the result
8. Signal snapshots are captured into the canonical research dataset

### Research Evaluation Flow

1. A signal snapshot is converted into a canonical row by `research/signal_evaluation/evaluator.py`
2. The row is upserted into `research/signal_evaluation/signals_dataset.csv`
3. Later, realized spot outcomes are merged into the same row by `scripts/update_signal_outcomes.py`
4. Evaluation scores and grouped reports are produced from the canonical dataset

## Key Features

- Live option-chain support:
  - NSE
  - Zerodha
  - ICICI Breeze
- Market structure analytics:
  - gamma exposure
  - gamma flip
  - dealer inventory
  - dealer hedging flow
  - dealer hedging simulator
  - gamma walls
  - gamma path and squeeze detection
  - intraday gamma shift
  - volatility regime
  - volatility surface
- Flow and liquidity analytics:
  - options flow imbalance
  - smart money flow
  - liquidity heatmap
  - liquidity voids
  - liquidity vacuum zones
  - dealer liquidity map
- Macro/news overlay:
  - scheduled event risk
  - deterministic headline classification
  - macro/news regime aggregation
  - conservative confirmation and sizing adjustments
- Signal policy:
  - weighted directional consensus
  - configurable scoring
  - symbol-aware intraday thresholds
  - signal regime and execution regime classification
- Research system:
  - canonical one-row-per-signal dataset
  - dedupe-safe upsert by `signal_id`
  - realized outcome enrichment
  - evaluation scoring
  - regime fingerprinting
  - grouped research reporting

## Repository Guide

```text
options_quant_engine/
├── app/
├── analytics/
├── backtest/
├── config/
├── data/
├── engine/
├── macro/
├── models/
├── news/
├── research/
├── scripts/
├── strategy/
├── tests/
├── visualization/
├── main.py
├── requirements.txt
└── README.md
```

### Important Directories

- `analytics/`: market-structure analytics used by live and research flows
- `app/`: app-facing wrappers and the Streamlit UI
- `backtest/`: historical replay and evaluation stack
- `config/`: environment, thresholds, scoring policy, symbol microstructure settings
- `data/`: broker adapters, validation, replay helpers, and spot/chain loaders
- `engine/`: main signal-generation pipeline
- `research/signal_evaluation/`: canonical signal dataset, enrichment, scoring, reporting
- `scripts/`: operational helpers for dataset refresh and reporting
- `tests/`: unit coverage for macro/news, live engine policy, and signal evaluation

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Environment variables are loaded automatically from `.env` if present.

### Recommended setup

```bash
cp .env.example .env
```

Then fill in the credentials for the provider you want to use.

### Important environment variables

Zerodha:

```bash
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token
```

ICICI Breeze:

```bash
ICICI_BREEZE_API_KEY=your_api_key
ICICI_BREEZE_SECRET_KEY=your_secret_key
ICICI_BREEZE_SESSION_TOKEN=your_session_token
```

Optional ICICI expiry overrides:

```bash
ICICI_DEFAULT_EXPIRY_DATE=
ICICI_NIFTY_EXPIRIES=
ICICI_BANKNIFTY_EXPIRIES=
ICICI_FINNIFTY_EXPIRIES=
```

Optional provider debug flags:

```bash
NSE_DEBUG=false
ICICI_DEBUG=false
```

Optional macro/news settings:

```bash
MACRO_EVENT_FILTER_ENABLED=true
MACRO_EVENT_SCHEDULE_FILE=config/india_macro_schedule.json
HEADLINE_PROVIDER=MOCK
HEADLINE_MOCK_FILE=config/mock_headlines.example.json
HEADLINE_RSS_URLS=
```

### Important config modules

- `config/settings.py`: runtime defaults, broker config, backtest settings, paths
- `config/signal_policy.py`: direction/scoring calibration for live signals
- `config/symbol_microstructure.py`: symbol-aware live thresholds
- `config/signal_evaluation_scoring.py`: evaluation scoring weights and thresholds

## How To Run

### Terminal live engine

```bash
python main.py
```

You will be prompted for:

- symbol
- source
- broker credentials when needed
- budget constraints and lot settings

The terminal loop prints:

- spot validation
- option-chain validation
- macro/news regime
- trader view
- dealer positioning dashboard
- diagnostics
- ranked strikes

Stop with `Ctrl+C`.

### Streamlit app interface

Run the Streamlit UI with:

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- live mode
- replay mode
- budget-aware inputs
- provider credentials
- snapshot saving
- trade summary
- signal research dashboard
- ranked strikes
- macro/news diagnostics
- provider and validation diagnostics

The Streamlit workspace includes a dedicated `Signal Research` tab that reads from the canonical dataset at `research/signal_evaluation/signals_dataset.csv` and summarizes:

- signal counts and scoring coverage
- average composite score and move probability
- hit rate by trade strength and macro regime
- realized return by horizon
- move-probability calibration
- top-performing regime fingerprints

If `streamlit` is installed from `requirements.txt`, the app should be available immediately after environment setup.

### Replay mode

Replay a saved snapshot pair:

```bash
python main.py --replay \
  --replay-spot debug_samples/NIFTY_spot_snapshot_2026-03-13T15-25-00+05-30.json \
  --replay-chain debug_samples/NIFTY_ICICI_option_chain_snapshot_2026-03-13T17-53-29.968000+05-30.csv
```

### Replay regression

```bash
python -m backtest.replay_regression --symbol NIFTY --source ICICI --replay-dir debug_samples
```

### Backtesting

```bash
python backtest/backtest_runner.py
```

Supports:

- single backtest
- parameter sweep

## Canonical Signal Research Dataset

The system treats the signal dataset as a canonical research dataset, not an append-only log.

Dataset path:

```text
research/signal_evaluation/signals_dataset.csv
```

Rules:

- one signal = one row
- primary key = `signal_id`
- rows are updated over time
- duplicates are removed on canonical write
- schema is kept stable and normalized

### Signal capture policy

Signal capture can be filtered using these policies:

- `TRADE_ONLY`
- `ACTIONABLE`
- `ALL_SIGNALS`

Terminal usage:

```bash
python main.py --signal-capture-policy TRADE_ONLY
```

Default behavior remains `ALL_SIGNALS`.

## Research Dataset Operations

### Refresh pending realized outcomes

```bash
python scripts/update_signal_outcomes.py
```

Optional cutoff:

```bash
python scripts/update_signal_outcomes.py --as-of "2026-03-14T15:25:00+05:30"
```

### Run grouped research reports

```bash
python scripts/signal_evaluation_report.py
```

## What The Research Dataset Stores

The canonical dataset includes:

- signal identity and timestamp
- signal context:
  - symbol
  - direction
  - trade status
  - signal regime
  - execution regime
  - macro regime
  - gamma regime
  - flow state
  - quality scores
  - probabilities
- provider and data quality diagnostics
- realized outcomes:
  - 5m
  - 15m
  - 30m
  - 60m
  - same-day close
  - next-day open
  - next-day close
- evaluation scores:
  - direction score
  - magnitude score
  - timing score
  - tradeability score
  - composite signal score
- regime fingerprint fields for later condition-cluster analysis

## Research Reporting Layer

The reporting layer provides grouped analysis such as:

- hit rate by trade strength
- hit rate by macro regime
- average score by signal quality
- average realized return by horizon
- signal count by regime
- move-probability calibration
- regime fingerprint performance

This is designed to help answer:

- which signals work
- which regimes work best
- whether move probability calibrates to realized outcomes
- which market-condition fingerprints produce the best signals

## Tests

Run the main targeted suite:

```bash
python -m unittest \
  tests/test_signal_evaluation_reports.py \
  tests/test_signal_evaluation_dataset.py \
  tests/test_macro_news_layer.py \
  tests/test_option_chain_validation.py \
  tests/test_live_engine_policy.py
```

## Historical Data Notes

Backtests rely on `data/historical_option_chain.py`.

Behavior:

- if a cached CSV exists in `data_store/`, it is loaded
- otherwise the system downloads spot history and builds a synthetic option-chain approximation

This means historical testing can run without a paid historical option-chain vendor, but synthetic-chain assumptions still matter.

## Installed Dependencies

Main dependencies include:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `yfinance`
- `requests`
- `scipy`
- `kiteconnect`
- `breeze-connect`
- `python-dotenv`
- `streamlit`

## Notes

- The Streamlit app and terminal engine both use the same core signal-generation path
- The research dataset is canonical by design and should be treated as a research table, not a log file
- The best next step after live-market accumulation is calibration review on the captured dataset rather than more rule growth by default
