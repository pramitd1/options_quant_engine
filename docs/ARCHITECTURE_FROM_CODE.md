# Architecture Reconstructed From Code

## What This Document Describes

This is the architecture the repository actually implements today, reconstructed from imports and orchestration code.

It is not an idealized target architecture.

## True Entrypoints

### Production And Operator Entrypoints

| Entrypoint | What It Activates |
| --- | --- |
| `main.py` | Interactive CLI for live and replay runs. Calls `app.engine_runner.run_engine_snapshot(...)`, which now acts as the market-data loading wrapper around the shared `run_preloaded_engine_snapshot(...)` orchestration path. |
| `app/streamlit_app.py` | Browser-based operator/research workstation. Also calls `run_engine_snapshot(...)` and reads research reports/datasets. |
| `smoke_macro_news.py` | Smoke-test entrypoint for scheduled-event risk, headline ingestion, and macro-news aggregation only. |

### Research And Operational Entrypoints

| Entrypoint | What It Activates |
| --- | --- |
| `scripts/update_signal_outcomes.py` | Calls `research.signal_evaluation.update_signal_dataset_outcomes(...)` to backfill realized outcomes into the canonical signal dataset. |
| `scripts/signal_evaluation_report.py` | Calls `research.signal_evaluation.write_signal_evaluation_report(...)` to build markdown/JSON reports from the canonical signal dataset. |
| `scripts/parameter_governance.py` | Front door to governed tuning, review context, manual approval, and promotion. |

### Backtest And Replay Entrypoints

| Entrypoint | What It Activates |
| --- | --- |
| `backtest/backtest_runner.py` | Interactive backtest runner. Uses `backtest.intraday_backtester.run_intraday_backtest(...)` and parameter sweeps. |
| `backtest/replay_regression.py` | Loads replay snapshots and routes them through `app.engine_runner.run_preloaded_engine_snapshot(...)` so replay uses the same macro/news/global-risk assembly stages as live runtime after data loading. |
| `backtest/*_scenario_runner.py` | Specialized scenario runners for macro news, global risk, dealer pressure, gamma-vol acceleration, and option-efficiency layers. |

## Actual Layer Map

### 1. Data Ingestion / Data Access

Primary modules:

- `data.data_source_router`
- `data.spot_downloader`
- `data.zerodha_option_chain`
- `data.nse_option_chain_downloader`
- `data.icici_breeze_option_chain`
- `data.replay_loader`
- `data.global_market_snapshot`
- `news.service`

Role in the system:

- fetch live spot and option-chain data
- normalize provider outputs
- fetch global-market context
- fetch headline/news context
- load replay snapshots and historical datasets

Separation quality:

- Good overall.
- Provider-specific code is mostly contained in `data/*`.
- Live runtime still decides too much of the ingestion flow inside `app.engine_runner`.

### 2. Normalization / Preprocessing

Primary modules:

- `data.provider_normalization`
- `data.option_chain_validation`
- `data.expiry_resolver`
- `engine.trading_support.common.normalize_option_chain`

Role in the system:

- standardize provider schemas
- resolve the active expiry slice
- validate spot and option-chain freshness/health
- enrich chains into the schema expected by analytics and the signal engine

Separation quality:

- Good, but spread across `data/*` and `engine/trading_support/common.py`.

### 3. Analytics / Feature Computation

Primary modules:

- `analytics/gamma_*`
- `analytics/liquidity_*`
- `analytics/dealer_*`
- `analytics/volatility_*`
- `analytics/greeks_engine.py`
- `analytics/options_flow_imbalance.py`
- `analytics/smart_money_flow.py`

Role in the system:

- compute structural market features from the normalized option chain
- infer dealer positioning, flip levels, walls, liquidity voids, IV regime, vanna/charm regimes, and flow context

Separation quality:

- Strong.
- Analytics remain upstream of strategy/risk/research.

### 4. Market-State Assembly

Primary module:

- `engine.trading_support.market_state._collect_market_state(...)`

Role in the system:

- fan-in point from many analytics calculators
- returns one denormalized market-state bundle for downstream consumers

Why it matters:

- This is the point where isolated analytics stop being independent calculations and become part of one signal-ready state object.

### 5. Probability Layer

Primary modules:

- `engine.trading_support.probability`
- `models.feature_builder`
- `models.large_move_probability`
- `models.ml_move_predictor`

Role in the system:

- convert market-state features into:
  - rule-based move probability
  - ML move probability
  - blended hybrid move probability

Separation quality:

- Good.
- Probability is a real intermediate layer, not hidden inside trade scoring.

### 6. Signal Engine / Direction / Scoring

Primary modules:

- `engine.trading_support.signal_state`
- `strategy.trade_strength`
- `strategy.confirmation_filters`
- `engine.signal_engine`

Role in the system:

- infer direction
- score the directional thesis
- apply confirmation filters
- classify signal quality and execution regime

Where each decision happens:

- Direction inference:
  - `engine.trading_support.signal_state.decide_direction(...)`
- Base scoring:
  - `strategy.trade_strength.compute_trade_strength(...)`
- Confirmation:
  - `strategy.confirmation_filters.compute_confirmation_filters(...)`
- Assembly into one signal-state payload:
  - `engine.trading_support.signal_state._compute_signal_state(...)`

### 7. Risk Overlays

Primary modules:

- `macro.scheduled_event_risk`
- `macro.macro_news_aggregator`
- `risk.global_risk_features`
- `risk.global_risk_layer`
- `risk.gamma_vol_acceleration_layer`
- `risk.dealer_hedging_pressure_layer`
- `risk.option_efficiency_layer`
- `engine.trading_support.trade_modifiers`

Role in the system:

- downgrade, cap, or veto otherwise-valid signals
- attach overlay-specific diagnostics and score adjustments

Important distinction:

- Some macro/news logic is built before `generate_trade(...)` in `app.engine_runner`.
- The final risk gate is still inside `engine.signal_engine`, especially through `evaluate_global_risk_layer(...)`.

### 8. Strategy / Trade Construction

Primary modules:

- `strategy.strike_selector`
- `strategy.exit_model`
- `strategy.budget_optimizer`

Role in the system:

- rank candidate option contracts after the directional thesis survives gating
- choose strike and option type
- set target and stop-loss
- optimize lots and capital usage

### 9. Research Logging / Signal Capture

Primary modules:

- `app.runtime_sinks`
- `config.policy_resolver`
- `research.signal_evaluation.evaluator`
- `research.signal_evaluation.dataset`
- `research.signal_evaluation.policy`

Role in the system:

- decide whether a runtime snapshot should be captured
- turn the runtime payload into a canonical signal-evaluation row
- persist the row to the research dataset

Integration point:

- This happens in the runtime layer, not in the signal engine itself.
- Runtime payloads now expose both `execution_trade` and `trade_audit`, while preserving the legacy merged `trade` payload for compatibility.

### 10. Evaluation / Reporting / Tuning / Promotion

Primary modules:

- `research.signal_evaluation.market_data`
- `research.signal_evaluation.evaluator`
- `research.signal_evaluation.reporting`
- `tuning.experiments`
- `tuning.validation`
- `tuning.search`
- `tuning.campaigns`
- `tuning.governance`
- `tuning.promotion`

Role in the system:

- backfill realized outcomes
- score signal quality
- build research reports
- run governed parameter search
- compare candidate vs production packs
- manage candidate, shadow, and live assignments

### 11. Runtime Policy Resolution

Primary modules:

- `config.policy_resolver`
- `tuning.runtime`
- `tuning.packs`

Role in the system:

- resolve the active parameter pack without pulling configuration getters back through the full tuning registry
- keep live runtime, backtests, replay, and tuning experiments on the same override context
- leave `tuning.runtime` as the tuning-facing compatibility surface for governance code

## Actual Signal Path

## Live Runtime Path

1. `main.py` or `app/streamlit_app.py` calls `app.engine_runner.run_engine_snapshot(...)`, or a historical workflow calls `run_preloaded_engine_snapshot(...)`.
2. `run_engine_snapshot(...)` loads raw market inputs when the caller has not already done so.
3. `_prepare_snapshot_context(...)` builds the shared runtime context:
   - spot snapshot
   - option chain
   - scheduled-event state
   - headline ingestion state
   - global-market snapshot or a neutral historical fallback
4. `_prepare_snapshot_context(...)` resolves expiry and validates the option chain.
5. `app.engine_runner` enters `_evaluate_snapshot_for_pack(...)`.
6. `_evaluate_snapshot_for_pack(...)` activates the current parameter pack through `tuning.runtime.temporary_parameter_pack(...)`.
7. `_evaluate_snapshot_for_pack(...)` builds:
   - `macro_news_state` via `macro.macro_news_aggregator.build_macro_news_state(...)`
   - `global_risk_state` via `risk.build_global_risk_state(...)`
8. `_evaluate_snapshot_for_pack(...)` calls `engine.signal_engine.generate_trade(...)`.

## Inside `generate_trade(...)`

1. Normalize the option chain with `engine.trading_support.common.normalize_option_chain(...)`.
2. Build market state with `engine.trading_support.market_state._collect_market_state(...)`.
3. Build probability state with `engine.trading_support.probability._compute_probability_state(...)`.
4. Build signal state with `engine.trading_support.signal_state._compute_signal_state(...)`.
5. Apply macro-news adjustments and overlay modifier scores.
6. Build overlay states:
   - gamma/vol acceleration
   - dealer hedging pressure
   - option efficiency
7. Accumulate adjusted trade strength.
8. Build the verbose base payload.
9. Run the final pre-trade gate:
   - `risk.global_risk_layer.evaluate_global_risk_layer(...)`
10. If the signal survives gating, rank contracts with `strategy.strike_selector.select_best_strike(...)`.
11. Build contract-level option-efficiency state.
12. Compute exits with `strategy.exit_model.calculate_exit(...)`.
13. Optimize lots with `strategy.budget_optimizer.optimize_lots(...)`.
14. Split the final payload into `execution_trade` and `trade_audit` while preserving the legacy merged payload.
15. Emit the final trade/no-trade payload.

## After `generate_trade(...)`

1. `app.engine_runner` builds a snapshot-level result payload around the trade.
2. `app.runtime_sinks.DefaultShadowEvaluationSink` may re-run the snapshot under a shadow parameter pack.
3. `app.runtime_sinks.DefaultSignalCaptureSink` may persist the signal to the research dataset.

## Where Specific Decisions Happen

| Concern | Module / Function |
| --- | --- |
| Direction inference | `engine.trading_support.signal_state.decide_direction(...)` |
| Trade-strength scoring | `strategy.trade_strength.compute_trade_strength(...)` |
| Confirmation | `strategy.confirmation_filters.compute_confirmation_filters(...)` |
| Move probability | `engine.trading_support.probability._compute_probability_state(...)` |
| Macro-event state | `macro.scheduled_event_risk.evaluate_scheduled_event_risk(...)` |
| Headline aggregation | `macro.macro_news_aggregator.build_macro_news_state(...)` |
| Global-risk feature build | `risk.global_risk_features.build_global_risk_features(...)` |
| Final risk gate | `risk.global_risk_layer.evaluate_global_risk_layer(...)` |
| Strike ranking | `strategy.strike_selector.select_best_strike(...)` |
| Exit construction | `strategy.exit_model.calculate_exit(...)` |
| Budget sizing | `strategy.budget_optimizer.optimize_lots(...)` |
| Signal capture | `research.signal_evaluation.save_signal_evaluation(...)` |
| Outcome backfill | `research.signal_evaluation.update_signal_dataset_outcomes(...)` |
| Tuning experiment | `tuning.experiments.run_parameter_experiment(...)` |
| Governed campaign | `tuning.campaigns.run_group_tuning_campaign(...)` |
| Candidate creation workflow | `tuning.governance.run_controlled_tuning_workflow(...)` |

## How Research And Tuning Connect Back To Production

The connection is not inside the engine core. It happens through two explicit surfaces:

### Parameter Packs

- `config/*` getters now resolve effective values through `config.policy_resolver`.
- Live runtime and shadow runtime both activate parameter packs via `temporary_parameter_pack(...)`.
- This lets the same computational code run under:
  - live pack
  - candidate pack
  - shadow pack
  - experiment overrides

### Signal Evaluation Dataset

- Live runtime may persist each qualifying snapshot through `save_signal_evaluation(...)`.
- Evaluation later enriches those rows with realized outcomes.
- Tuning experiments read that canonical dataset and score parameter packs against it.

So the research/tuning stack connects to production in two ways:

- parameterization of the live engine
- reuse of live-captured signal outputs as the tuning dataset

## Where The Code Respects Intended Separation

- Data ingestion is mostly upstream.
- Analytics are cleanly upstream of signal generation.
- Signal assembly, strategy, and risk overlays are explicit stages.
- Research and tuning are mostly downstream consumers.

## Where The Implementation Has Drifted

- `config` is no longer coupled directly back to the tuning registry. Runtime policy resolution now lives in `config.policy_resolver`, with `tuning.runtime` preserved as a compatibility layer for governance code.
- Runtime orchestration mixes production concerns with research capture and shadow governance.
- Backtest parity is improved for the main entrypoints because `intraday_backtester` and `replay_regression` now use `run_preloaded_engine_snapshot(...)`, but historical paths still rely on neutral global-market fallbacks when aligned cross-asset data is unavailable.

## Practical Interpretation

For a future refactor, the repo already has usable seams:

- `market_state`
- `probability_state`
- `signal_state`
- overlay layers
- runtime sinks
- research dataset
- parameter runtime context

Those seams are real and visible in code. The architecture does not need to be invented from scratch; it needs to be hardened around those existing boundaries.
