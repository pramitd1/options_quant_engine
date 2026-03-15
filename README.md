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

## Current System Shape

The engine now has five important overlay packages plus a dedicated research-governance stack sitting on top of the core microstructure signal path:

- `macro/` and `news/`: scheduled event risk, headline classification, macro/news aggregation
- `risk/global_risk_*`: external/global regime classification, overnight gap risk, volatility expansion risk
- `risk/gamma_vol_acceleration_*`: convexity and acceleration overlay
- `risk/dealer_hedging_pressure_*`: dealer-flow and pinning/acceleration overlay
- `risk/option_efficiency_*`: expected move and option-buying efficiency overlay
- `tuning/`: parameter registry, named packs, experiment runner, advanced search, automated campaigns, promotion, and reporting
  plus walk-forward and regime-aware validation

These layers are intentionally modifiers and filters. They do not replace the core directional engine.

## Main Workflows

### 1. Live Signal Generation

- terminal loop in [main.py](main.py)
- Streamlit app in [streamlit_app.py](app/streamlit_app.py)
- shared orchestration in [engine_runner.py](app/engine_runner.py)

### 2. Replay and Validation

- replay through `main.py --replay`
- snapshot-based validation under [backtest/](backtest)
- deterministic scenario runners for global risk, gamma-vol, dealer pressure, and option efficiency

### 3. Historical Backtesting

- runner in [backtest_runner.py](backtest/backtest_runner.py)
- default historical builder is bar-based and uses synthetic option-chain reconstruction unless richer data is provided
- performance conclusions should be interpreted in the context of the available bar granularity

### 4. Research Dataset and Evaluation

- canonical dataset in [signals_dataset.csv](research/signal_evaluation/signals_dataset.csv)
- schema and upsert rules in [dataset.py](research/signal_evaluation/dataset.py)
- the canonical CSV now has a durable SQLite sidecar for faster, safer local research reads and updates
- live capture appends new signal rows in constant time and only rewrites the canonical store when an existing `signal_id` needs an update
- row-building and outcome enrichment in [evaluator.py](research/signal_evaluation/evaluator.py) and [market_data.py](research/signal_evaluation/market_data.py)
- this dataset is the primary calibration and validation source for tuning, walk-forward analysis, and promotion decisions

### 5. Parameter Research and Governance

- central parameter registry in [registry.py](tuning/registry.py)
- named packs in [parameter_packs](config/parameter_packs)
- research-generated candidate packs in `research/parameter_tuning/candidate_packs/`
- runtime pack activation in [runtime.py](tuning/runtime.py)
- objective evaluation and experiments in [objectives.py](tuning/objectives.py) and [experiments.py](tuning/experiments.py)
- search, promotion, and ledger inspection in [search.py](tuning/search.py), [promotion.py](tuning/promotion.py), and [reporting.py](tuning/reporting.py)
- automated group tuning campaigns in [campaigns.py](tuning/campaigns.py)
- walk-forward split engine and regime-aware validation in [walk_forward.py](tuning/walk_forward.py), [regimes.py](tuning/regimes.py), and [validation.py](tuning/validation.py)
- live shadow comparison and rollout logging in [shadow.py](tuning/shadow.py)

Important distinction:

- `tuning/` contains the parameter-tuning and promotion code
- `research/parameter_tuning/` stores runtime-generated research artifacts, ledgers, reports, state, and candidate packs

## Architecture

### Core Live Flow

1. spot data and intraday context are loaded from `data/spot_downloader.py`
2. option-chain data is routed through broker/public adapters in `data/`
3. provider output is normalized and validated
4. expiry selection is resolved before trade generation
5. [signal_engine.py](engine/signal_engine.py) assembles:
   - market state
   - probability state
   - directional vote
   - trade strength
   - strike selection
   - trade payload
6. [trading_engine.py](engine/trading_engine.py) remains as a backward-compatible facade over the canonical signal engine
7. helper domains are separated under `engine/trading_support/`
8. overlay layers modify risk, ranking, confirmation, and overnight handling
9. the result is rendered in the terminal or Streamlit and optionally written into the research dataset

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
├── backtest/           # backtests, replay helpers, scenario runners
├── backtests/          # generated backtest output directory
├── config/             # runtime, scoring, and overlay policies
│   └── parameter_packs/# versioned parameter pack overrides
├── data/               # provider adapters, validation, historical loaders
├── engine/             # signal engine, compat facades, and trading support domains
├── macro/              # scheduled-event and macro/news logic
├── models/             # move probability and ML support
├── news/               # deterministic news classification
├── research/           # signal evaluation + parameter-tuning artifacts
├── risk/               # overlay layers and regime models
├── scripts/            # operational helpers
├── strategy/           # confirmation, strike selection, exits, sizing
├── tests/              # regression and scenario coverage
├── tuning/             # registry, packs, experiments, search, validation, promotion code
├── main.py
└── README.md
```

## Key Files

- [signal_engine.py](engine/signal_engine.py): canonical signal assembly
- [trading_engine.py](engine/trading_engine.py): backward-compatible facade for signal generation imports
- [trading_engine_support.py](engine/trading_engine_support.py): backward-compatible facade over `engine/trading_support/`
- [strike_selector.py](strategy/strike_selector.py): strike ranking and optional candidate hooks
- [global_risk_layer.py](risk/global_risk_layer.py): global risk facade
- [gamma_vol_acceleration_layer.py](risk/gamma_vol_acceleration_layer.py): convexity acceleration overlay
- [dealer_hedging_pressure_layer.py](risk/dealer_hedging_pressure_layer.py): dealer pressure overlay
- [option_efficiency_layer.py](risk/option_efficiency_layer.py): expected move / option efficiency overlay
- [dataset.py](research/signal_evaluation/dataset.py): canonical schema
- [evaluator.py](research/signal_evaluation/evaluator.py): research row builder and outcome enrichment
- [registry.py](tuning/registry.py): parameter registry and metadata
- [experiments.py](tuning/experiments.py): experiment runner and ledger
- [promotion.py](tuning/promotion.py): baseline/candidate/live workflow

## Configuration

Environment variables are loaded from `.env` when present.

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
MACRO_EVENT_FILTER_ENABLED=true
MACRO_EVENT_SCHEDULE_FILE=config/india_macro_schedule.json
HEADLINE_PROVIDER=MOCK
HEADLINE_MOCK_FILE=config/mock_headlines.example.json
HEADLINE_RSS_URLS=
```

### Global Market Overlay Settings

```bash
GLOBAL_MARKET_DATA_ENABLED=true
GLOBAL_MARKET_LOOKBACK_DAYS=40
GLOBAL_MARKET_STALE_DAYS=2
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

## Research Note

The maintained research note source is:

- [quant_note_trade_signal_logic.md](research/quant_note_trade_signal_logic.md)

Local rendered research-note artifacts are generated under `documentation/`, but
that folder is intentionally excluded from git for now.

If you want to regenerate local publishable artifacts, use the markdown sources
under `research/` and `documentation/research_notes/`.

## Testing

Targeted regression:

```bash
pytest -q tests/test_live_engine_policy.py tests/test_signal_evaluation_dataset.py
```

Full suite:

```bash
pytest -q
```

Parameter tuning framework:

```bash
pytest -q tests/test_parameter_tuning_framework.py
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

## Notes

- The engine is intentionally conservative about missing or stale inputs and should degrade to neutral rather than inventing state.
- Global risk, gamma-vol, dealer pressure, and option efficiency are overlays, not standalone direction engines.
- The system is intentionally decoupled from actual executed trades; the canonical signal dataset is the research truth source.
- The remaining research challenge is no longer basic parameter centralization; it is disciplined calibration and validation of the now much larger governed surface without overfitting.
