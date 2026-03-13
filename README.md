# Options Quant Engine

An interactive options analytics and signal-generation engine for Indian index and stock options. The project combines live option-chain ingestion, dealer/gamma/liquidity analytics, ML-assisted move probability scoring, and a research backtesting stack that can generate synthetic historical option chains when cached data is not available.

## What This Project Does

The engine is built around two workflows:

1. Live terminal workflow via `main.py`
   - Pulls spot price data with `yfinance`
   - Pulls option-chain data from NSE, Zerodha, or ICICI Breeze
   - Supports index options and stock options such as `RELIANCE`, `SBIN`, and `TCS`
   - Computes dealer positioning, gamma structure, volatility, liquidity, and flow analytics
   - Produces a directional trade idea with strike, entry, target, stop, quality, and budget-aware sizing
   - Prints a terminal dashboard with validation, trader view, dealer structure, diagnostics, and ranked strikes

2. Research workflow via `backtest/backtest_runner.py`
   - Runs single historical backtests or parameter sweeps
   - Uses cached historical option-chain data if present
   - Automatically builds a synthetic historical option chain from spot history if no cache exists
   - Reports trade logs, PnL metrics, and Monte Carlo reshuffle statistics

## High-Level Architecture

### Entry Points

- `main.py`: interactive live engine loop
- `backtest/backtest_runner.py`: backtest runner with single-run and parameter-sweep modes
- `config/generate_token.py`: helper script for generating a Zerodha access token manually
- `.env.example`: sample environment-variable template for broker credentials and debug flags

### Core Flow

1. Spot price is fetched from `data/spot_downloader.py`
2. Option-chain data is routed through `data/data_source_router.py`
3. `engine/trading_engine.py` normalizes the chain and computes all analytics
4. Strategy modules assign direction, strike, exits, score, and optional lot optimization
5. Visualization modules print a terminal dashboard
6. Backtest modules reuse the same trade-generation engine on historical snapshots

## Features

- Live option-chain support for:
  - NSE public data
  - Zerodha/Kite data
  - ICICI Breeze data
- Dealer and gamma analytics:
  - gamma exposure
  - gamma flip
  - dealer inventory position
  - dealer hedging flow and hedging simulator
  - market gamma regime and top gamma strikes
  - gamma wall classification
  - intraday gamma shift
  - gamma path / squeeze detection
- Liquidity and flow analytics:
  - options flow imbalance
  - smart money flow
  - liquidity heatmap
  - liquidity voids
  - liquidity vacuum zones
  - dealer liquidity map
- Volatility analytics:
  - volatility regime
  - volatility surface / ATM IV regime
- Signal generation:
  - CALL/PUT direction selection
  - strike selection
  - target/stop calculation
  - trade-strength scoring with scoring breakdown
  - signal-quality classification
- Position sizing:
  - optional budget-aware lot optimization
- ML / probability layer:
  - feature builder
  - training-time random-forest scaffold
  - deterministic live ML fallback wrapper
  - large-move rule-based probability
  - hybrid rule + ML move probability
- Research tooling:
  - intraday-style historical replay backtester
  - walk-forward retraining scaffold
  - parameter sweep
  - Monte Carlo reshuffle
  - performance metrics

## Repository Structure

```text
options_quant_engine/
├── main.py
├── requirements.txt
├── config/
├── data/
├── analytics/
├── strategy/
├── models/
├── engine/
├── backtest/
├── visualization/
└── data_store/
```

### Folder Guide

#### `analytics/`

Market-structure analytics used by both live trading and backtests.

- `gamma_exposure.py`: approximate gamma exposure / GEX
- `gamma_flip.py`: gamma flip level and spot-vs-flip logic
- `dealer_inventory.py`: long-gamma vs short-gamma dealer stance
- `dealer_gamma_path.py`: gamma path simulation and squeeze detection
- `dealer_hedging_flow.py`: directional hedging pressure
- `dealer_hedging_simulator.py`: move-based hedging simulation and bias
- `market_gamma_map.py`: market gamma map, regime classification, largest gamma strikes
- `gamma_walls.py`: support/resistance gamma walls
- `intraday_gamma_shift.py`: change in gamma profile between snapshots
- `options_flow_imbalance.py`: call/put flow imbalance
- `smart_money_flow.py`: unusual volume based flow classification
- `liquidity_heatmap.py`: strongest liquidity strikes
- `liquidity_void.py`: low-OI gaps and void signals
- `liquidity_vacuum.py`: breakout-zone detection
- `dealer_liquidity_map.py`: next support/resistance, squeeze zone, vacuum summary
- `volatility_regime.py`: realized-vol regime
- `volatility_surface.py`: ATM IV and IV regime
- `greeks_engine.py`: Black-Scholes Greek enrichment plus aggregate Greek exposures and regimes
- `flow_utils.py`: front-expiry / near-ATM helpers for flow analytics

#### `engine/`

- `trading_engine.py`: the main decision engine. It normalizes data, calls analytics, blends rule-based and ML signals, computes trade strength, sets exits, and returns the final trade payload.

#### `strategy/`

Execution and scoring helpers.

- `trade_strength.py`: weighted scoring model and scoring breakdown
- `exit_model.py`: target/stop from config percentages
- `budget_optimizer.py`: lot optimization under capital limits
- `strike_selector.py`: ATM strike helper
- `signal_builder.py`: simple signal scaffold

#### `models/`

Move prediction and feature engineering.

- `feature_builder.py`: converts market state into a feature vector
- `move_predictor.py`: training-time `RandomForestClassifier` scaffold
- `ml_move_predictor.py`: live inference wrapper with deterministic heuristic fallback
- `large_move_probability.py`: rule-based large move probability

#### `data/`

Data acquisition, routing, and historical data generation.

- `data_source_router.py`: unified source selection
- `nse_option_chain_downloader.py`: NSE option-chain fetcher
- `zerodha_option_chain.py`: Zerodha instrument + quote based chain builder
- `icici_breeze_option_chain.py`: ICICI Breeze chain builder with Security Master-based expiry resolution and IV fallback estimation
- `spot_downloader.py`: current spot price via `yfinance`
- `historical_option_chain.py`: cache loader or synthetic historical chain builder
- `historical_iv_surface.py`: optional historical IV-surface lookup for synthetic backtests
- `intraday_downloader.py`: 5-minute spot downloader utility
- `synthetic_option_chain.py`: synthetic chain helper
- `instrument_loader.py`: instrument universe utility
- `replay_loader.py`: saved spot/option-chain snapshot loader for after-hours replay

#### `backtest/`

Historical replay and evaluation.

- `intraday_backtester.py`: replays historical snapshots and simulates trade entries/exits
- `backtest_runner.py`: CLI wrapper for single-run and parameter-sweep modes
- `performance_metrics.py`: PnL, win-rate, drawdown, Sharpe, expectancy
- `pnl_engine.py`: trade-level PnL including transaction costs
- `monte_carlo.py`: reshuffled-path robustness analysis
- `parameter_sweep.py`: grid builder and ranking
- `walk_forward.py`: walk-forward retraining scaffold

#### `visualization/`

Terminal dashboards and analytics printouts.

- `dealer_dashboard.py`: main terminal dashboard used by `main.py`
- `liquidity_dashboard.py`

#### `config/`

Global settings and credentials.

- `settings.py`: thresholds, lot size, capital limits, refresh interval, backtest config, directories, broker env var reads, and provider debug flags
- `generate_token.py`: helper to exchange a Zerodha request token for an access token

#### `data_store/`

Local cache for datasets and generated history. Example data already exists under `data_store/NIFTY/`.

## How Trade Generation Works

`engine/trading_engine.py` is the center of the system.

It:

1. Normalizes the option-chain schema from different data sources
2. Normalizes source-specific fields and enriches the chain with Greeks when they are not provided directly
3. Computes:
   - gamma / gamma flip / gamma regime
   - vanna / charm exposures and regimes
   - dealer position and hedging bias
   - options flow and smart-money flow
   - liquidity levels, voids, vacuum zones
   - market gamma map and gamma clusters
   - volatility regime and ATM IV regime
   - intraday gamma shift if a previous snapshot exists
4. Chooses trade direction from a ruleset that blends flow, gamma/flip, dealer positioning, and vanna/charm structure
5. Selects a strike near spot
6. Computes:
   - rule-based large-move probability
   - ML move probability
   - hybrid move probability
7. Scores the setup using `strategy/trade_strength.py`
8. Applies target/stop logic
9. Optionally reduces lot count if the trade breaches capital constraints
10. Returns a final trade payload including trader view fields and dashboard fields

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Replay Testing

Canonical single-session replay:

```bash
python main.py --replay \
  --replay-spot debug_samples/NIFTY_spot_snapshot_2026-03-13T15-25-00+05-30.json \
  --replay-chain debug_samples/NIFTY_ICICI_option_chain_snapshot_2026-03-13T17-53-29.968000+05-30.csv
```

Canonical replay bias/regression check:

```bash
python -m backtest.replay_regression --symbol NIFTY --source ICICI --replay-dir debug_samples
```

The regression harness summarizes:

- `CALL` count
- `PUT` count
- `NO_SIGNAL` count
- direction-source frequency
- latest replay cases used in the sample

Dependencies listed in the repository:

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

## Configuration

Main configuration lives in `config/settings.py`. Environment variables are loaded automatically from a local `.env` file if present.

### Recommended setup

```bash
cp .env.example .env
```

Then fill in the credentials for the provider you want to use.

### Environment variables

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

### Runtime credential prompts

If you choose `ZERODHA` or `ICICI` in `main.py`, the app will prompt for the required broker credentials. Press Enter to reuse values already loaded from `.env` or your shell environment.

### Important settings

- `DEFAULT_SYMBOL`
- `DEFAULT_DATA_SOURCE`
- `REFRESH_INTERVAL`
- `NSE_REFRESH_INTERVAL`
- `ICICI_REFRESH_INTERVAL`
- `TARGET_PROFIT_PERCENT`
- `STOP_LOSS_PERCENT`
- `MAX_CAPITAL_PER_TRADE`
- `LOT_SIZE`
- `NUMBER_OF_LOTS`
- analytics thresholds such as `HIGH_GAMMA_THRESHOLD` and `VOL_EXPANSION_THRESHOLD`
- backtest settings such as:
  - `BACKTEST_YEARS`
  - `BACKTEST_STRIKE_STEP`
  - `BACKTEST_STRIKE_RANGE`
  - `BACKTEST_DEFAULT_IV`
  - `BACKTEST_SIGNAL_PERSISTENCE`
  - `BACKTEST_MAX_HOLD_BARS`
  - `BACKTEST_ENABLE_BUDGET`
  - `MC_SIMULATIONS`
  - `MAX_WORKERS`

If you use Zerodha, generate the access token using the Kite login flow and the helper in `config/generate_token.py`.

## How To Run

### Live engine

```bash
python main.py
```

You will be prompted for:

- symbol: `NIFTY / BANKNIFTY / FINNIFTY / STOCK`
- if you choose `STOCK`, the app will prompt for the actual stock-option underlying symbol such as `RELIANCE`, `SBIN`, or `TCS`
- data source: `NSE`, `ZERODHA`, or `ICICI`
- broker credentials when `ZERODHA` or `ICICI` is selected
- whether budget constraints should be applied
- optional lot size / lot count / capital limit if budget mode is enabled

Then the engine enters a refresh loop and prints:

- spot validation
- spot price
- option-chain validation
- trader view
- dealer positioning dashboard
- quant trade signal
- diagnostics
- ranked strikes for the selected expiry

Stop it with `Ctrl+C`.

### Backtesting

```bash
python backtest/backtest_runner.py
```

The runner supports:

- single backtest
- parameter sweep

Single backtest output includes:

- summary metrics
- Monte Carlo reshuffle results
- last 5 trades

Parameter sweep output includes:

- ranked result rows sorted by total PnL and Sharpe ratio

### Walk-forward retraining

There is no standalone CLI for this module. `backtest/walk_forward.py` currently provides a reusable function:

- `walk_forward_retrain(feature_rows, labels)`

Use it from a notebook, script, or future CLI wrapper.

## Historical Data Behavior

Backtests rely on `data/historical_option_chain.py`.

Behavior:

- If a cached CSV already exists in `data_store/`, it is loaded
- If no cached historical option chain exists, the engine:
  - downloads spot history from `yfinance`
  - optionally loads a historical IV surface if available
  - builds a synthetic option chain across strikes around ATM
  - prices options using a Black-Scholes-style approximation
  - saves the generated dataset back to `data_store/<SYMBOL>/`

This means backtests can run without a real historical option-chain vendor, but they are partly synthetic unless you provide your own cached data.

## Expected Outputs

### Trader View

Typical fields include:

- symbol
- spot
- direction
- direction source
- selected expiry
- strike
- option type
- entry price
- target
- stop loss
- trade strength
- signal quality
- trade status
- budget fields
- large move probability
- ML move probability
- hybrid move probability

### Dealer Dashboard

Typical fields include:

- gamma exposure
- gamma flip level
- market gamma and regime
- dealer inventory
- hedging flow and bias
- intraday gamma state
- volatility regime / ATM IV regime
- flow and smart-money state
- gamma event
- walls, voids, and vacuum zones
- dealer liquidity map
- scoring breakdown

### Diagnostics and ranked strikes

The live output also includes:

- option-chain validation stats such as row count, CE/PE counts, priced rows, and IV rows
- a compact diagnostics block with supporting analytics
- a ranked strikes table for the currently selected expiry

## Assumptions and Limitations

- The live engine is analytics-driven and prints trade ideas; it does not place broker orders.
- `main.py` accepts `STOCK` as a shortcut, then prompts for the actual underlying symbol; live stock-option support depends on whether that symbol is available from the selected provider.
- Zerodha support requires valid Kite credentials and instrument access.
- ICICI support requires valid Breeze credentials and a live session token.
- NSE endpoints can change or rate-limit requests.
- ICICI expiry resolution uses ICICI metadata and configured fallbacks; manual expiry overrides are optional, not required.
- For stock options, spot lookup uses Yahoo Finance NSE cash tickers such as `.NS`, while broker requests continue to use the clean underlying code such as `RELIANCE`.
- Some missing Greeks are approximated in `engine/trading_engine.py` when a source does not provide them.
- Some IV values may be estimated from market inputs when the live provider does not supply usable implied volatility.
- Historical backtests may be based on synthetic option chains rather than real archived option-chain snapshots.
- The walk-forward module is a scaffold, not a complete training pipeline.
- Visualization is terminal-based; there is no web UI in the current codebase.
- There is no formal test suite in this repository yet.

## Suggested Workflow

### For live analytics

1. Configure dependencies
2. Create `.env` from `.env.example` or be ready to enter broker credentials interactively
3. Run `python main.py`
4. For stock options, choose `STOCK` and then enter the real underlying symbol
5. Start with `NSE` if you want a public data path, or `ICICI` / `ZERODHA` if you want broker-backed live data
6. Observe trader view, dashboard output, and refresh behavior

### For research

1. Run `python backtest/backtest_runner.py`
2. Start with a single backtest for `NIFTY`
3. Review cached data generated under `data_store/`
4. Use parameter sweep to compare persistence / hold-bar settings
5. Extend walk-forward retraining if you want a fuller ML workflow

## Possible Next Improvements

- add a proper CLI with argument flags instead of interactive prompts
- separate live data adapters from research data generators
- add unit tests for analytics and trade generation
- persist trade logs to CSV or a database
- add model training and serialization workflow
- build a richer dashboard or notebook layer for research

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

For historical research:

```bash
python backtest/backtest_runner.py
```
