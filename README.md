# Options Quant Engine

`options_quant_engine` is a live + research derivatives engine for Indian index and stock options. It combines live option-chain ingestion, dealer/gamma/liquidity analytics, macro/news risk controls, convexity overlays, option-efficiency evaluation, replay tooling, and a canonical signal research dataset.

The system is designed to be signal-evaluation-first:

- it generates signals from market data
- it evaluates those signals against subsequent market behavior
- it tunes and validates from the signal evaluation dataset
- it does not depend on your manually executed trades for research, tuning, or promotion

The system is designed as a layered trade engine, not a single-factor signal script:

1. build microstructure state
2. infer direction conservatively
3. estimate move quality and path risk
4. apply overlay/risk layers
5. rank strikes and size exposure
6. log the signal into a research-safe canonical dataset

The current repository does not contain a live order-routing engine. Broker integrations here are used for market data access, not for automatic execution.

## Quick Start

### Environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Windows Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Fill in provider credentials only for the routes you want to use.

Notes:

- the commands elsewhere in this README are written in Unix-style form because
  the repo has primarily been developed on macOS
- on Windows, once the virtual environment is activated, the Python commands
  are usually the same, for example `python main.py` and
  `streamlit run app/streamlit_app.py`
- if PowerShell blocks activation, you may need:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Terminal Engine

```bash
python main.py
```

### Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### Refresh Research Outcomes

```bash
python scripts/update_signal_outcomes.py
```

### Research Reporting

```bash
python scripts/signal_evaluation_report.py
```

### Daily Research Report

```bash
python scripts/daily_research_report.py
```

### Multiyear Backtest

```bash
python scripts/run_multiyear_backtest.py
```

### Build ML Model Registry

```bash
python scripts/build_model_registry.py
```

## Repo Hygiene Policy

To keep the repository maintainable and reproducible, use these artifact path conventions.

- Runtime validation and profiling outputs: store under `research/runtime_validation/`.
- Backtest logs and comparison outputs: store under `research/runtime_validation/backtest_runs/`.
- Signal-evaluation reports and generated tables: store under `research/signal_evaluation/reports/`.
- One-off analysis outputs from ad-hoc scripts: store under `research/` with a date-stamped subfolder.
- Audit reports, deployment reviews, and ad-hoc documentation snapshots: store under `documentation/` (local archive; intentionally excluded from git).
- Avoid writing generated artifacts to repository root.

Commit hygiene:

- Keep code changes, script reorganization, and generated research artifacts in separate commits.
- When moving scripts, keep compatibility wrappers at legacy entrypoints until automation is migrated.
- Do not delete intermediate or final research artifacts that are needed for auditability.

## Current System Shape

The engine now has five important overlay packages plus a dedicated research-governance stack sitting on top of the core microstructure signal path:

- `macro/` and `news/`: scheduled event risk, headline classification, macro/news aggregation
- `risk/global_risk_*`: external/global regime classification, overnight gap risk, volatility expansion risk
- `risk/gamma_vol_acceleration_*`: convexity and acceleration overlay
- `risk/dealer_hedging_pressure_*`: dealer-flow and pinning/acceleration overlay
- `risk/option_efficiency_*`: expected move and option-buying efficiency overlay
- `tuning/`: parameter registry, named packs, experiment runner, advanced search, automated campaigns, promotion, and reporting
  plus walk-forward and regime-aware validation
- `strategy/`: strike selection, confirmation scoring, enhanced strike scoring with market-microstructure factors, exit model, budget optimization, and trade strength evaluation
- `utils/`: centralized numeric helpers (`clip`, `safe_float`, `safe_div`, `to_python_number`), math functions (`norm_pdf`, `norm_cdf`), and timestamp utilities (`coerce_timestamp`)

These layers are intentionally modifiers and filters. They do not replace the core directional engine.

## Main Workflows

### 1. Live Signal Generation

- terminal loop in [main.py](main.py)
- Streamlit app in [streamlit_app.py](app/streamlit_app.py)
- shared orchestration in [engine_runner.py](app/engine_runner.py)

### 2. Replay and Validation

- replay through `main.py --replay`
- shared replay/backtest orchestration through [run_preloaded_engine_snapshot](app/engine_runner.py)
- snapshot-based validation under [backtest/](backtest)
- deterministic scenario runners for global risk, gamma-vol, dealer pressure, and option efficiency

### 3. Historical Backtesting

- runner in [holistic_backtest_runner.py](backtest/holistic_backtest_runner.py)
- three data-source modes controlled by `BACKTEST_DATA_SOURCE` (or interactive prompt):
  - **historical**: real NSE bhav-copy option chains with Newton-Raphson IV computation via [historical_data_adapter.py](data/historical_data_adapter.py)
  - **live**: synthetic option-chain reconstruction from spot bars (legacy default)
  - **combined**: historical chains where available, synthetic fallback elsewhere
- the historical pipeline is powered by [historical_data_download.py](scripts/historical_data_download.py) which downloads NSE F&O bhav copies, merges them into a single Parquet database, and downloads spot history via yfinance
- the merged Parquet dataset (`data_store/historical/`) is excluded from version control due to size (~12 GB); run the download script locally to build it
- performance conclusions should be interpreted in the context of the available data granularity

### 4. Research Dataset and Evaluation

- live signal dataset in `research/signal_evaluation/signals_dataset.csv`
- backtest dataset in `research/signal_evaluation/backtest_signals_dataset.parquet` (7,404 signals from 10-year multi-year backtest)
- schema and upsert rules in [dataset.py](research/signal_evaluation/dataset.py)
- row-building and outcome enrichment in [evaluator.py](research/signal_evaluation/evaluator.py) and [market_data.py](research/signal_evaluation/market_data.py)
- this dataset is the primary calibration and validation source for tuning, walk-forward analysis, and promotion decisions
- all research data artifacts (CSV, Parquet, checkpoints, reports) are excluded from version control and regenerated locally

### 5. Parameter Research and Governance

- central parameter registry in [registry.py](tuning/registry.py)
- named packs in [parameter_packs](config/parameter_packs)
- research-generated candidate packs in `research/parameter_tuning/candidate_packs/`
- runtime policy resolution in [policy_resolver.py](config/policy_resolver.py)
- tuning-facing compatibility wrappers in [runtime.py](tuning/runtime.py)
- objective evaluation and experiments in [objectives.py](tuning/objectives.py) and [experiments.py](tuning/experiments.py)
- search, promotion, and ledger inspection in [search.py](tuning/search.py), [promotion.py](tuning/promotion.py), and [reporting.py](tuning/reporting.py)
- automated group tuning campaigns in [campaigns.py](tuning/campaigns.py)
- walk-forward split engine and regime-aware validation in [walk_forward.py](tuning/walk_forward.py), [regimes.py](tuning/regimes.py), and [validation.py](tuning/validation.py)
- live shadow comparison and rollout logging in [shadow.py](tuning/shadow.py)

Important distinction:

- `tuning/` contains the parameter-tuning and promotion code
- `research/parameter_tuning/` stores runtime-generated research artifacts, ledgers, reports, state, and candidate packs

### 6. ML Research Evaluation

- policy robustness analysis: `python research/ml_evaluation/policy_robustness/robustness_runner.py`
- rank-gate + confidence-sizing research: `python research/ml_evaluation/rank_gate_sizing/rank_gate_sizing_runner.py`
- predictor method comparison: `python research/ml_evaluation/predictor_comparison/predictor_comparison_runner.py`
- each runner reads from the canonical signal datasets, produces structured reports (`.md`, `.json`, `.csv`), and generates visualizations (`.png`)
- all research outputs are version-controlled alongside the runner scripts

## Scoring Modes (Production Defaults)

The engine supports both `continuous` and `discrete` scoring in key decision modules.
Production defaults are now set to `continuous` to reduce threshold-jump behavior while keeping hard safety gates intact.

Current defaults:

- `trade_strength.runtime_thresholds.trade_strength_scoring_mode = "continuous"`
- `confirmation_filter.core.confirmation_scoring_mode = "continuous"`
- `strike_selection.scoring.strike_scoring_mode = "continuous"`

Where these are defined:

- `config/signal_policy.py`
- `config/strike_selection_policy.py`

Behavioral notes:

- Continuous mode smooths score contribution curves for market-state factors.
- Final hard protections remain discrete by design (for example, veto and safety gates).
- You can switch any module back to `discrete` through policy overrides for controlled A/B checks.

Replay A/B helper for confirmation mode:

```bash
python scripts/backtest/compare_confirmation_mode_replay.py \
  --snapshot-dir data_store/replay_snapshots \
  --max-snapshots 200
```

This generates CSV/JSON/Markdown artifacts that quantify score deltas and decision-impact deltas between `continuous` and `discrete` confirmation scoring.

Full cross-predictor scoring mode comparison:

```bash
python scripts/backtest/compare_scoring_modes_full_backtest.py
```

Runs all 8 predictor methods under both `discrete` and `continuous` trade-strength scoring (16 backtest stages total) and writes CSV/JSON/Markdown artifacts to `scripts/backtest/reports/`. Use this to measure per-method accuracy, trade volume, and composite score deltas between scoring modes before changing production defaults.

## Architecture

### Core Live Flow

1. spot data and intraday context are loaded from `data/spot_downloader.py`
2. option-chain data is routed through broker/public adapters in `data/`
3. [engine_runner.py](app/engine_runner.py) now splits loading from evaluation:
   - `run_engine_snapshot(...)` loads live/replay inputs
   - `run_preloaded_engine_snapshot(...)` evaluates already-loaded snapshots
4. provider output is normalized and validated
5. expiry selection is resolved before trade generation
6. [signal_engine.py](engine/signal_engine.py) assembles:
   - market state
   - probability state
   - directional vote
   - trade strength
   - strike selection
   - trade payload
7. trade output is split into:
   - `execution_trade` for operator/execution-facing fields
   - `trade_audit` for research, diagnostics, and governance fields
   - `trade` as the backward-compatible merged payload
8. [trading_engine.py](engine/trading_engine.py) remains as a backward-compatible facade over the canonical signal engine
9. helper domains are separated under `engine/trading_support/`
10. overlay layers modify risk, ranking, confirmation, and overnight handling
11. the result is rendered in the terminal or Streamlit and optionally written into the research dataset

### Overlay Stack

The engine currently applies overlays in this order:

1. macro/news adjustments
2. global risk regime
3. gamma-vol acceleration
4. dealer hedging pressure
5. option efficiency

That order matters:

- macro/global layers handle exogenous regime stress
- gamma-vol and dealer pressure handle convexity and path amplification
- option efficiency asks whether the option itself is worth buying relative to the expected move

### Gamma Flip-Zone Awareness

The signal pipeline includes structural awareness of the gamma flip zone — the price level where dealer gamma exposure crosses zero. When spot sits at or near the flip, the microstructure becomes unpredictable: hedging flows can amplify moves in either direction, and the engine has no directional edge.

This awareness is applied in two places:

1. **Trade strength dampener** (`strategy/trade_strength.py`): a `flip_zone_dampener_score` component applies a configurable penalty when `spot_vs_flip == AT_FLIP` and the gamma regime is unfavorable (NEGATIVE_GAMMA: -12 pts, NEUTRAL_GAMMA: -8 pts). Positive gamma at the flip gets no penalty because mean-reversion support exists.

2. **Confirmation filter** (`strategy/confirmation_filters.py`): a `flip_zone_gamma_score` component penalizes the confirmation total when the same AT_FLIP + non-positive gamma condition holds (-3 pts for negative gamma, -2 pts for neutral). This can degrade the confirmation status from CONFIRMED to MIXED or CONFLICT.

3. **Direction vote breadth** (`engine/trading_support/signal_state.py`): the signal state now includes `direction_vote_count` — the number of independent vote sources behind the chosen direction. A thin base (e.g., 2 sources like FLOW+CHARM) is visible to downstream consumers for sizing or urgency decisions.

The net effect: the engine continues to produce signals when the market moves through the flip zone, but the trade suggestions reflect the structural uncertainty. A typical AT_FLIP + neutral/negative gamma signal degrades from TRADE to WATCHLIST territory, which is the correct behavior — observe but don't act until the market resolves directionally.

All flip-zone weights are tunable via `TRADE_STRENGTH_WEIGHTS` and `CONFIRMATION_FILTER_CONFIG` in `config/signal_policy.py` and are automatically registered in the parameter tuning registry.

### Research Flow

1. each captured signal is assigned a stable `signal_id`
2. one signal maps to one canonical row
3. rows are updated as realized outcomes arrive
4. no-direction rows remain neutral during enrichment rather than being forced bearish
5. the dataset can be grouped by regime fingerprints, score buckets, probability buckets, and overlay states
6. actual broker fills or discretionary trades are intentionally outside this calibration loop

## Repository Guide

```text
options_quant_engine/
├── analytics/          # dealer/gamma/liquidity/volatility analytics
├── app/                # shared runner + Streamlit app
├── archive/            # historical reference (old trade-centric backtest README)
├── backtest/           # backtests, replay helpers, scenario runners
├── backtests/          # raw backtest output artifacts (git-ignored)
├── config/             # runtime, scoring, and overlay policies
│   └── parameter_packs/# versioned parameter pack overrides
├── data/               # provider adapters, validation, historical loaders and adapters
├── data_store/         # on-disk data caches — historical Parquet, spot history, IV surface (git-ignored)
├── debug_samples/      # sample debug snapshots (git-ignored)
├── engine/             # signal engine, compat facades, trading support, and pluggable predictors
│   ├── predictors/     # MovePredictor protocol, factory, built-in and research predictors
│   └── trading_support/# probability state, market state, signal state, trade modifier helpers
├── logs/               # runtime log output (git-ignored)
├── macro/              # scheduled-event and macro/news logic
├── models/             # move probability, ML model predictor, feature builders, and trained predictor serialization
├── models_store/       # serialized model registry (git-ignored; rebuild via scripts/build_model_registry.py)
├── news/               # deterministic news classification
├── research/           # signal evaluation, decision policies, ML evaluation, parameter-tuning artifacts
│   ├── architecture_analysis/ # prediction pipeline architecture reports
│   ├── decision_policy/# decision-policy definitions, engine, config, evaluation
│   ├── ml_evaluation/  # robustness analysis, rank-gate sizing, predictor comparison, EV-based sizing research
│   ├── ml_models/      # GBT ranking + LogReg calibration research models
│   ├── ml_research/    # ML research results, calibration reports, shadow predictions
│   ├── parameter_tuning/# parameter-tuning artifacts and promotion state
│   └── signal_evaluation/ # canonical signal dataset, evaluator, daily reports
├── risk/               # overlay layers and regime models
├── scripts/            # operational helpers, historical-data download, ML research, model registry builder, daily reports, multiyear backtest
├── strategy/           # confirmation, strike selection, enhanced scoring, exits, sizing, trade strength
├── tests/              # regression and scenario coverage (238 tests)
├── tuning/             # registry, packs, experiments, search, validation, promotion code
├── utils/              # centralized numerics, math helpers, timestamp utilities
├── documentation/      # system monograph, academic papers, signal state dictionary, research notes
├── main.py
└── README.md
```

## Key Files

- [signal_engine.py](engine/signal_engine.py): canonical signal assembly
- [policy_resolver.py](config/policy_resolver.py): runtime policy-resolution layer that breaks the config/tuning cycle
- [engine_runner.py](app/engine_runner.py): loader wrapper plus shared preloaded snapshot orchestration
- [trading_engine.py](engine/trading_engine.py): backward-compatible facade for signal generation imports
- [trading_engine_support.py](engine/trading_engine_support.py): backward-compatible facade over `engine/trading_support/`
- [strike_selector.py](strategy/strike_selector.py): strike ranking and optional candidate hooks
- [global_risk_layer.py](risk/global_risk_layer.py): global risk facade
- [gamma_vol_acceleration_layer.py](risk/gamma_vol_acceleration_layer.py): convexity acceleration overlay
- [dealer_hedging_pressure_layer.py](risk/dealer_hedging_pressure_layer.py): dealer pressure overlay
- [option_efficiency_layer.py](risk/option_efficiency_layer.py): expected move / option efficiency overlay
- [historical_data_adapter.py](data/historical_data_adapter.py): NSE bhav-copy adapter with Newton-Raphson IV
- [historical_data_download.py](scripts/historical_data_download.py): multi-source historical data downloader
- [terminal_output.py](app/terminal_output.py): terminal rendering with verbosity modes (COMPACT/STANDARD/FULL_DEBUG)
- [signal_confidence.py](analytics/signal_confidence.py): multi-factor signal confidence scoring
- [enhanced_strike_scoring.py](strategy/enhanced_strike_scoring.py): institutional-grade five-factor strike scoring
- [spot_history.py](data/spot_history.py): intraday spot price history caching (yfinance-backed)
- [daily_research_report.py](research/signal_evaluation/daily_research_report.py): automated daily research narrative and analysis
- [runtime_metadata.py](engine/runtime_metadata.py): operator-facing trade decision metadata schema
- [ml_move_predictor.py](models/ml_move_predictor.py): production ML prediction wrapper (MLMovePredictor)
- [feature_builder.py](models/feature_builder.py): routes between heuristic (7-feature) and ML (33-feature) paths
- [expanded_feature_builder.py](models/expanded_feature_builder.py): 33-feature extraction for ML models
- [trained_predictor.py](models/trained_predictor.py): shared class for registry serialization
- [policy_engine.py](research/decision_policy/policy_engine.py): decision-policy evaluation engine
- [policy_definitions.py](research/decision_policy/policy_definitions.py): core policy gate definitions
- [robustness_runner.py](research/ml_evaluation/policy_robustness/robustness_runner.py): 10-year policy robustness analysis
- [rank_gate_sizing_runner.py](research/ml_evaluation/rank_gate_sizing/rank_gate_sizing_runner.py): rank-gate + confidence-sizing research
- [predictor_comparison_runner.py](research/ml_evaluation/predictor_comparison/predictor_comparison_runner.py): predictor method comparison
- [run_multiyear_backtest.py](scripts/run_multiyear_backtest.py): 10-year multi-symbol backtest runner
- [compare_scoring_modes_full_backtest.py](scripts/backtest/compare_scoring_modes_full_backtest.py): all-predictor discrete vs continuous scoring mode comparison (artifacts → `scripts/backtest/reports/`)
- [dataset.py](research/signal_evaluation/dataset.py): canonical schema
- [evaluator.py](research/signal_evaluation/evaluator.py): research row builder and outcome enrichment
- [registry.py](tuning/registry.py): parameter registry and metadata
- [experiments.py](tuning/experiments.py): experiment runner and ledger
- [promotion.py](tuning/promotion.py): baseline/candidate/live workflow

## Configuration

Environment variables are loaded from `.env` when present.

### Backtest Data Source

```bash
BACKTEST_DATA_SOURCE=historical   # historical | live | combined
```

Set to `historical` to use real NSE bhav-copy option chains (requires the
merged Parquet dataset under `data_store/historical/`).  Set to `live` for
synthetic chain reconstruction.  Set to `combined` for historical-first with
synthetic fallback.  The backtest runner also prompts interactively if the
variable is not set.

### Prediction Method

```bash
OQE_PREDICTION_METHOD=blended   # blended | pure_ml | pure_rule | research_dual_model | research_decision_policy | ev_sizing | research_rank_gate | research_uncertainty_adjusted
```

Set to `blended` (default) for the production rule + ML weighted blend. Set to `pure_ml` to use only the ML leg, `pure_rule` for only the rule-based heuristic, `research_dual_model` to use the research GBT ranking + LogReg calibration dual-model, `research_decision_policy` to use the decision-policy layer that applies ALLOW/BLOCK/DOWNGRADE policies over the dual-model output, `ev_sizing` to use expected-value-based sizing from conditional return tables (blocks negative-EV signals, scales positive-EV proportionally), `research_rank_gate` to block low-rank signals with a research rank threshold, or `research_uncertainty_adjusted` to downweight high-uncertainty signals using dual-model disagreement and confidence ambiguity. The backtester also accepts a per-run `prediction_method` parameter that overrides this setting for that run only.

### Common Provider Settings

Zerodha:

```bash
ZERODHA_API_KEY=
ZERODHA_API_SECRET=
ZERODHA_ACCESS_TOKEN=
```

ICICI Breeze:

```bash
ICICI_BREEZE_API_KEY=
ICICI_BREEZE_SECRET_KEY=
ICICI_BREEZE_SESSION_TOKEN=
```

### Macro / News Settings

```bash
OQE_RUNTIME_ENV=DEV
MACRO_EVENT_FILTER_ENABLED=true
MACRO_EVENT_SCHEDULE_FILE=config/india_macro_schedule.json
HEADLINE_PROVIDER=RSS
HEADLINE_MOCK_FILE=config/mock_headlines.example.json
HEADLINE_RSS_URLS=https://www.livemint.com/rss/markets
```

Production safety guard:

- when `OQE_RUNTIME_ENV` is set to `PROD` or `PRODUCTION`, `HEADLINE_PROVIDER=MOCK` is rejected at startup
- production deployments should use `HEADLINE_PROVIDER=RSS` (or another live provider)

### Global Market Overlay Settings

```bash
GLOBAL_MARKET_DATA_ENABLED=true
GLOBAL_MARKET_LOOKBACK_DAYS=90
GLOBAL_MARKET_STALE_DAYS=5
```

## Parameter Packs

Named packs currently live under [parameter_packs](config/parameter_packs):

- `baseline_v1`: registry-default pack, intended to preserve current behavior
- `macro_overlay_v1`: stronger macro/global caution candidate
- `overnight_focus_v1`: more conservative overnight selection candidate
- `experimental_v1`: research-only pack for offline experiments
- `candidate_v1`: reserved promotion slot

Pack format is JSON and supports inheritance through `parent` plus a flat `overrides` map keyed by stable parameter ids such as `trade_strength.scoring.flow_call_bullish` or `global_risk.core.risk_adjustment_extreme`.

The governed tuning surface now extends well beyond the original threshold packs and includes:

- signal-engine data-quality, execution, probability, and trade-modifier policies
- analytics feature thresholds for flow imbalance, smart-money flow, and volatility regime
- strike-selection heuristics
- large-move probability coefficients
- event-window risk policy
- category-level headline rule multipliers
- scalar headline rule weights
- raw overlay feature coefficients for global risk, gamma-vol acceleration, dealer hedging pressure, and option efficiency

The parameter registry now covers the main engine and research groups end to end, so tuning campaigns can act on a materially broader but still auditable surface instead of leaving the overlay math buried in code.

## Promotion And Shadow Mode

The production-governance layer now supports four explicit pack roles:

- `baseline`: trusted comparison reference
- `candidate`: pack under review
- `shadow`: pack evaluated in live conditions without controlling execution
- `live`: pack currently authoritative for real-time engine decisions

Promotion state and audit files live under `research/parameter_tuning/`:

- `promotion_state.json`
- `promotion_ledger.jsonl`
- `shadow_mode_log.jsonl`

These are generated runtime artifacts and are intentionally excluded from source control.

Candidate parameter packs created during governed tuning are also stored under:

- `research/parameter_tuning/candidate_packs/`

Shadow mode is conservative:

- the authoritative/live pack remains the only pack allowed to control the returned trade decision
- the shadow pack runs in parallel on the same live snapshot
- candidate outputs are compared and logged side by side
- canonical signal capture remains tied to the authoritative path only

## Tuning Workflow

The tuning subsystem is designed for controlled research, not naive profit chasing:

1. baseline defaults are registered with metadata and safety flags
2. a named parameter pack overrides only the keys under study
3. experiments evaluate packs against the canonical signal dataset with time-based train/validation splitting
4. objective scores combine hit rate, composite quality, tradeability, target reachability, drawdown proxy, stability, and frequency sanity checks
5. walk-forward validation adds explicit out-of-sample split metrics, regime summaries, and robustness scoring
6. search supports bounded random search, Latin hypercube exploration, coordinate-descent refinement, and registry-driven group campaigns
7. tuning writes a reviewable candidate pack and comparison report without modifying production
8. promotion from `baseline` to `candidate` to `live` can now consume out-of-sample and robustness hooks in addition to sample-count, stability, signal-frequency checks, and explicit manual approval

The practical implication is that the system is now much closer to a governed quantitative research stack:

- the engine generates signals
- the signal evaluation dataset records what the market actually did afterward
- the tuning framework searches registry-governed parameter groups
- walk-forward and regime-aware validation decide whether changes generalize
- promotion and shadow mode handle rollout conservatively
- production packs are not overwritten automatically by tuning

Structured research outputs are written under `research/parameter_tuning/` when experiment persistence is enabled.

## Signal-Evaluation-First Policy

Research, tuning, validation, and promotion are designed around one principle:

- compare the engine's generated signal with what the market actually did afterward

That means:

- parameter tuning is based on the canonical signal evaluation dataset
- walk-forward and regime-aware validation are based on signal outcomes, not personal trade history
- promotion and shadow-mode comparisons are based on signal behavior and robustness
- manual or real broker trades may still be logged operationally later, but they are not a learning source for this system

## Testing

Targeted regression:

```bash
pytest -q tests/test_live_engine_policy.py tests/test_signal_evaluation_dataset.py
```

Full suite (238 tests):

```bash
pytest -q
```

Warning governance (local + CI):

- known benign platform noise is allowlisted: `urllib3.exceptions.NotOpenSSLWarning` on macOS LibreSSL environments
- risky warnings are not broadly suppressed
- in CI (`CI=1`), warnings are promoted to errors by `tests/conftest.py`, with only the known urllib3 macOS SSL warning exempted

CI-style strict warning run:

```bash
CI=1 .venv/bin/python -m pytest -q
```

Notes:

- `pytest.ini` keeps the urllib3 allowlist explicit and narrow
- model deserialization version-mismatch warnings (for example sklearn `InconsistentVersionWarning`) are treated as actionable risk and should be resolved by model/runtime version alignment
- report-generation numeric summaries guard empty/all-NaN slices to avoid silent invalid-statistics warnings- data integrity tests (`test_live_data_anomaly_detection.py`) validate option chain consistency, IV anomalies, and spot price jumps
- macro integration tests (`test_historical_macro_parity.py`) verify historical and live parity for event-based signals
Parameter tuning framework:

```bash
pytest -q tests/test_parameter_tuning_framework.py
```

Predictor architecture and decision policies:

```bash
pytest -q tests/test_predictor_architecture.py tests/test_decision_policy.py
```

Governed parameter workflow:

```bash
python scripts/parameter_governance.py evaluate-current
python scripts/parameter_governance.py tune --group trade_strength --group option_efficiency
python scripts/parameter_governance.py approve-candidate --reviewer your_name
python scripts/parameter_governance.py promote-candidate --approved-by your_name
```

Automated group tuning campaign:

```python
from tuning import run_group_tuning_campaign

campaign = run_group_tuning_campaign(
    "baseline_v1",
    groups=["trade_strength", "confirmation_filter", "option_efficiency"],
    walk_forward_config={
        "split_type": "rolling",
        "train_window_days": 180,
        "validation_window_days": 60,
        "minimum_train_rows": 50,
        "minimum_validation_rows": 20,
    },
)
```

## Pluggable Predictor Architecture

The probability stack uses a pluggable predictor architecture under `engine/predictors/`. The active prediction method is selected via config and can be overridden per-run in backtests or at runtime.

### Available Prediction Methods

| Method | Description |
|---|---|
| `blended` | **Default.** Production pipeline — weighted blend of rule + ML legs |
| `pure_ml` | ML leg only — rule leg suppressed |
| `pure_rule` | Rule leg only — ML leg suppressed |
| `research_dual_model` | Research dual-model — GBT ranking + LogReg calibration |
| `research_decision_policy` | Decision-policy layer — dual-model + ALLOW/BLOCK/DOWNGRADE policies |
| `ev_sizing` | EV-based sizing — uses conditional return tables to compute per-signal expected value; blocks negative-EV, scales positive-EV proportionally |

### Switching Prediction Method

Set `OQE_PREDICTION_METHOD` in `.env` or as an environment variable:

```bash
OQE_PREDICTION_METHOD=pure_ml python main.py
```

Or override per backtest run without changing the global setting:

```python
result = run_holistic_backtest(
    "NIFTY",
    start_date="2024-01-01",
    prediction_method="pure_ml",
)
```

Or use the context manager for ad-hoc overrides:

```python
from engine.predictors import prediction_method_override

with prediction_method_override("research_dual_model"):
    result = run_preloaded_engine_snapshot(...)
```

### Key Modules

- `engine/predictors/protocol.py` — `MovePredictor` Protocol and `PredictionResult` dataclass
- `engine/predictors/factory.py` — singleton factory with registry-based resolution
- `engine/predictors/builtin_predictors.py` — built-in predictors (blended, pure_ml, pure_rule)
- `engine/predictors/research_predictor.py` — research dual-model predictor (GBT + LogReg)
- `engine/predictors/decision_policy_predictor.py` — decision-policy predictor (dual-model + policy overlay)
- `engine/predictors/ev_sizing_predictor.py` — EV-based sizing predictor (conditional return tables + expected value)
- `research/decision_policy/` — policy definitions, engine, evaluation, and configuration

### Registering Custom Predictors

```python
from engine.predictors.factory import register_predictor

register_predictor("my_custom", MyCustomPredictor)
```

Custom predictors must implement the `MovePredictor` protocol (a `name` property and a `predict(market_ctx)` method returning `PredictionResult`).

## ML Model Registry

The probability stack supports both a rule-based heuristic and trained ML models. The model registry lives under `models_store/registry/` (git-ignored; regenerable).

### Building the Registry

```bash
python scripts/build_model_registry.py
```

This trains and serializes all research models from the backtest dataset. Each model gets a versioned directory with `model.joblib` and `meta.json`.

### Activating an ML Model

Set `ACTIVE_MODEL` in `config/settings.py` or via environment variable:

```bash
OQE_ACTIVE_MODEL=GBT_shallow_v1 python main.py
```

When `ACTIVE_MODEL` is set, the feature builder produces 33-feature expanded vectors (via `models/expanded_feature_builder.py`) and the probability stack loads the corresponding registry model. When unset, the system falls back to the 7-feature rule-based heuristic.

Note: `ACTIVE_MODEL` controls which trained model the ML leg uses internally. `PREDICTION_METHOD` controls which prediction strategy (blended, pure_ml, pure_rule, research_dual_model, research_decision_policy, ev_sizing) composes the final `hybrid_move_probability`. Both settings are independent and composable.

### Research Scripts

- `scripts/ml_signal_research.py` — multi-model comparison pipeline (AUC, Brier, ECE, stability)
- `scripts/ml_calibration_research.py` — calibration analysis (Platt scaling, isotonic, reliability curves)
- `scripts/build_model_registry.py` — serialize research models into production registry

### Key Classes

- `models/ml_move_predictor.py::MLMovePredictor` — production ML prediction wrapper
- `models/trained_predictor.py::TrainedMovePredictor` — shared class for registry serialization
- `models/feature_builder.py` — routes between 7-feature heuristic and 33-feature ML paths
- `models/expanded_feature_builder.py` — 33-feature extraction for ML models

## Decision Policy Layer

The decision-policy layer sits on top of the dual-model probability stack and applies explicit ALLOW / BLOCK / DOWNGRADE decisions to each signal before it reaches execution. Policies are defined in `research/decision_policy/` and exposed as a predictor via `engine/predictors/decision_policy_predictor.py`.

### Policy Definitions

| Policy | Description |
|---|---|
| `dual_threshold` | ALLOW when both GBT rank ≥ threshold AND LogReg confidence ≥ threshold |
| `agreement_only` | ALLOW only when both models agree on direction |
| `rank_filter_bottom_20pct` | Block bottom 20% by GBT rank score |
| `rank_filter_bottom_30pct` | Block bottom 30% by GBT rank score |
| `rank_filter_bottom_40pct` | Block bottom 40% by GBT rank score |

### Key Modules

- `research/decision_policy/policy_definitions.py` — core policy gate definitions
- `research/decision_policy/policy_engine.py` — policy evaluation engine
- `research/decision_policy/policy_evaluation.py` — metrics computation (hit rate, returns, Sharpe)
- `research/decision_policy/policy_config.py` — configuration constants and thresholds

## ML Research Evaluation Framework

The `research/ml_evaluation/` directory contains a suite of research evaluation runners that operate on the canonical signal datasets and produce structured reports, visualizations, and JSON artifacts. All outputs are preserved alongside the runner scripts.

### 1. Policy Robustness Analysis

Runner: `research/ml_evaluation/policy_robustness/robustness_runner.py`

Comprehensive 10-year robustness analysis of all decision policies across 7,404 backtest signals. Covers:

- yearly stability breakdown (2016–2025)
- regime-conditional performance (gamma, macro, global risk regimes)
- rank and confidence threshold sweeps
- filter attribution (ALLOW vs BLOCK effectiveness)
- efficiency frontier across retention vs quality trade-off

Key finding: `dual_threshold` achieves 0.74 hit rate and +18.98 bps per signal at 48.66% retention. Regime analysis confirms consistent improvement across positive, neutral, and negative gamma environments.

### 2. Rank-Gate + Confidence-Sizing Research

Runner: `research/ml_evaluation/rank_gate_sizing/rank_gate_sizing_runner.py`

Tests the hypothesis that rank (GBT) should be used solely for signal filtering while confidence (LogReg) drives position sizing:

- three rank thresholds: block bottom 20% / 30% / 40%
- three confidence-sizing tiers: low 0.5×, medium 1.0×, high 1.5×

Best result: `rank_gate_40` — 0.74 hit rate, +22.1 bps unsized → +26.1 bps sized (+17.8% improvement), Sharpe 0.362, cumulative +36,304 bps over 10 years.

### 3. Predictor Comparison

Runner: `research/ml_evaluation/predictor_comparison/predictor_comparison_runner.py`

Head-to-head evaluation of the original 5 predictor methods on both cumulative (279 rows) and backtest (7,404 rows) datasets (the `ev_sizing` method was added later — see cross-method comparison in `research/ml_evaluation/ev_and_regime_policy/`):

| Predictor | Hit Rate | Avg Return (bps) | Sharpe | Retention |
|---|---|---|---|---|
| **decision_policy** | **0.74** | **18.98** | **0.348** | **40.91%** |
| research_dual | 0.67 | 10.92 | 0.172 | 46.8% |
| pure_ml | 0.58 | 5.03 | 0.078 | 59.35% |
| blended | 0.50 | -2.36 | -0.030 | 99.85% |
| pure_rule | 0.50 | -2.49 | -0.033 | 94.26% |

The `decision_policy` predictor dominates on quality metrics; `blended` and `pure_rule` retain nearly all signals but at breakeven-to-negative expected return. Policy filtering is the decisive performance driver.

### 4. EV-Based Sizing & Regime-Switching Policy

Runner: `research/ml_evaluation/ev_and_regime_policy/runner.py`

Two research modules: (1) EV-based sizing using conditional return tables, and (2) regime-switching policy selection. Combined cross-method comparison across 33 sizing/filtering variants in 5 categories:

- EV sizing outperforms confidence sizing: Sharpe 0.349 vs 0.344, avg return +32.89 vs +15.08 bps, cumulative +49,724 vs +22,808 bps (2.2× improvement)
- Best overall Sharpe: `rank_gate_40` at 0.362
- Regime switching did not improve over static policies on the evaluated dataset (single volatility regime)
- 25 output artifacts (JSON, CSV, MD, PNG charts) generated

### 5. Parameter Tuning Research

Runner: `research/ml_evaluation/parameter_tuning/`

ML-informed parameter tuning research with phase-based runners for grid search, focused sweeps, and walk-forward validation.

### Additional ML Evaluation Reports

The `research/ml_evaluation/` root directory also contains:

- `ml_evaluation_runner.py` — base ML model evaluation and comparison
- `ml_calibration_report.py` — Platt scaling, isotonic calibration, and reliability curves
- `ml_comparison_report.py` — side-by-side model scoring across architectures
- `ml_ranking_report.py` — cross-model rank-stability analysis
- `ml_filter_simulation.py` — ML-based signal filtering simulation
- `engine_metrics_evaluation.py` — engine-level metric evaluation against ML baselines
- `decision_policy_comparison.md` — narrative comparison of policy alternatives

## Nomenclature

The trade payload uses explicit names to avoid ambiguity:

| Payload Key | Source | Description |
|---|---|---|
| `hybrid_move_probability` | active predictor (`engine/predictors/`) | Final probability output from the active prediction method |
| `rule_move_probability` | `models/large_move_probability.py` | Rule-based heuristic probability |
| `ml_move_probability` | `models/ml_move_predictor.py` | ML model probability (when active) |
| `predictor_name` | `engine/predictors/` | Name of the active predictor that produced the probability |
| `large_move_probability` | alias of `hybrid_move_probability` | Legacy alias — prefer `hybrid_move_probability` |
| `flow_signal` | `analytics/options_flow_imbalance.py` | Raw options flow imbalance label |
| `final_flow_signal` | consensus of flow + smart money | Merged directional flow signal |
| `macro_news_volatility_shock_score` | `macro/engine_adjustments.py` | Headline-derived volatility shock |
| `market_volatility_shock_score` | `risk/global_risk_features.py` | VIX/cross-asset volatility shock |
| `volatility_regime` | `analytics/volatility_regime.py` | External name for internal `vol_regime` |
| `dealer_hedging_bias` | `analytics/dealer_hedging_flow.py` | External name for internal `hedging_bias` |

## Notes

- The engine is intentionally conservative about missing or stale inputs and should degrade to neutral rather than inventing state.
- Global risk, gamma-vol, dealer pressure, and option efficiency are overlays, not standalone direction engines.
- The system is intentionally decoupled from actual executed trades; the canonical signal dataset is the research truth source.
- The remaining research challenge is no longer basic parameter centralization; it is disciplined calibration and validation of the now much larger governed surface without overfitting.
