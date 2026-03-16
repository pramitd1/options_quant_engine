# Code Dependency Audit

## Scope And Method

This audit was reconstructed from the codebase itself, not from folder names or README claims.

- Scope: 154 non-test Python modules under `analytics`, `app`, `backtest`, `config`, `data`, `engine`, `macro`, `models`, `news`, `research`, `risk`, `scripts`, `strategy`, `tuning`, plus top-level entrypoints.
- Method:
  - Parsed internal Python imports to build a module graph and folder-level dependency summary.
  - Read the live runtime path (`main.py`, `app/engine_runner.py`, `engine/signal_engine.py`).
  - Read the research, evaluation, tuning, promotion, and backtest entrypoints.
  - Checked for architectural cycles, layer violations, and hidden coupling.
- Uncertainty:
  - This is an import-level and orchestration-level audit. It does not prove every dynamic call path.
  - Some compatibility helpers such as `_call_first(...)` deliberately hide concrete function names behind fallback lookup.

## Executive Findings

The codebase mostly follows a sensible quant-engine split:

`data -> analytics -> engine/trading_support -> engine.signal_engine -> strategy/risk -> app/runtime -> research/tuning`

The strongest part of the architecture is the computational core:

- `analytics` stays free of `strategy`, `research`, and `tuning`.
- `risk` stays free of `research` and `tuning`.
- `strategy` stays free of `research` and `tuning`.
- `engine` depends on `strategy`, `risk`, `analytics`, and `models`, but not directly on `research` or `tuning`.

The main architectural improvement now in code is the configuration boundary:

- `config/*` policy getters resolve live parameter-pack overrides through `config.policy_resolver`.
- `config.policy_resolver` depends on `tuning.packs`, not on `tuning.registry`.
- `tuning.runtime` remains the tuning-facing compatibility surface and imports `tuning.registry` lazily only where governance code still needs registry defaults or serialization.
- This removes the old direct `config <-> tuning.runtime <-> tuning.registry` cycle from the production policy-resolution path.

The remaining major drift is data parity rather than orchestration parity:

- Live runtime goes through `app/engine_runner.py`, which builds macro-event state, headline state, global-market state, global-risk state, signal capture, and shadow evaluation.
- The main backtest and replay utilities now reuse `run_preloaded_engine_snapshot(...)`, so they share the same post-load orchestration path.
- Historical analysis still uses neutral global-market fallbacks when aligned cross-asset snapshots are unavailable, so parity is improved but not perfect.

## Folder-Level Dependency Summary

Observed high-signal folder couplings from imports:

| Source | Main Dependencies | What It Means |
| --- | --- | --- |
| `app` | `data`, `engine`, `macro`, `news`, `research`, `tuning` | Runtime orchestration is the integration layer where production, research capture, and shadow evaluation meet. |
| `engine` | `analytics`, `config`, `models`, `risk`, `strategy` | Core signal assembly is layered but very central. |
| `risk` | `config`, `risk` | Overlay logic is self-contained and policy-driven. |
| `strategy` | `config` | Trade construction is mostly isolated from the rest of the repo. |
| `research` | `config`, `data`, `research` | Evaluation/reporting stack is mostly downstream and read-oriented. |
| `tuning` | `config`, `research`, `tuning`, `macro` | Governance depends on both configuration and research datasets. |
| `config` | `tuning.packs` | Configuration is dynamically parameter-pack aware, but active resolution is now isolated behind `config.policy_resolver` instead of the full tuning runtime/registry stack. |
| `backtest` | `config`, `data`, `engine`, `macro`, `news`, `risk` | Backtests and scenario runners touch many layers, especially when simulating overlays. |

Import graph highlights:

- `config.settings` has the highest inbound dependency count in the repo.
- `config.policy_resolver` is now the key runtime-policy abstraction that decouples config getters from tuning governance internals.
- `tuning.runtime` remains heavily reused, but more as a compatibility and governance surface than as the direct config entrypoint.
- `engine.signal_engine` is the most central production computation module.
- `app.engine_runner` is the highest-fan-out production orchestrator.
- `tuning.registry` is the most central governance/configuration module.

## Most Central Modules

### Runtime And Signal Path

| Module | Why It Is Central |
| --- | --- |
| `app.engine_runner` | Real runtime orchestrator. Pulls market data, validates, resolves parameter packs, builds macro/global risk context, calls the signal engine, triggers signal capture, and runs shadow evaluation. |
| `engine.signal_engine` | Final trade assembly point. Combines market state, probabilities, confirmation, overlays, strike selection, sizing, and final trade status. |
| `engine.trading_support.market_state` | Fan-in point for the analytics layer. Converts many analytics modules into one denormalized market-state bundle. |
| `engine.trading_support.probability` | Bridges analytics outputs into rule/ML/blended move probabilities. |
| `engine.trading_support.signal_state` | Computes direction, trade-strength scoring, confirmation, signal quality, and execution regime. |
| `risk.global_risk_layer` | Final pre-trade macro/global risk gate. |
| `strategy.strike_selector` | Main trade-construction ranking engine after direction and overlay gating survive. |

### Research And Governance

| Module | Why It Is Central |
| --- | --- |
| `research.signal_evaluation.evaluator` | Converts runtime payloads into signal-evaluation rows and later backfills realized outcomes. |
| `research.signal_evaluation.reporting` | Builds structured research reports from the canonical signal dataset. |
| `config.policy_resolver` | Injects active parameter-pack overrides into config getters without depending on the registry. |
| `tuning.runtime` | Tuning-facing compatibility surface for active pack context, shadow mode, and registry serialization. |
| `tuning.registry` | Defines the tunable universe and groups across the whole engine. |
| `tuning.experiments` | Runs a parameter pack against the signal-evaluation dataset and optional walk-forward validation. |
| `tuning.governance` | High-level research-to-candidate workflow orchestration. |
| `tuning.promotion` | Stores promotion state and candidate/live/shadow assignments. |

## Peripheral And Leaf Modules

Many analytics calculators are intentionally leaf-like:

- `analytics.gamma_exposure`
- `analytics.gamma_walls`
- `analytics.liquidity_vacuum`
- `analytics.liquidity_void`
- `analytics.volatility_surface`
- `analytics.dealer_gamma_path`
- `analytics.dealer_hedging_flow`

That is a healthy pattern. These modules mainly perform isolated calculations and are consumed through `engine.trading_support.market_state`.

Other weakly connected modules are operational or scenario tools:

- `smoke_macro_news.py`
- `backtest/*_scenario_runner.py`
- `scripts/update_signal_outcomes.py`
- `scripts/signal_evaluation_report.py`

These are peripheral by design.

## Circular Dependency Findings

### Runtime Cycle Status

The previous production-path cycle:

`config policy getter -> tuning.runtime -> tuning.registry -> config policy modules`

has been broken in code by introducing `config.policy_resolver`.

Current behavior:

- config and macro policy getters depend on `config.policy_resolver`
- `config.policy_resolver` depends on `tuning.packs`
- `tuning.registry` still imports config and macro modules to enumerate tunable defaults
- `tuning.runtime` still imports `tuning.registry` lazily for tuning/governance concerns

Interpretation:

- the production configuration path is now acyclic
- the tuning/governance stack still depends heavily on config modules, which is acceptable but remains a high-coupling boundary

### What Was Not Found

- No evidence that `analytics` imports `strategy`, `research`, or `tuning`.
- No evidence that `risk` imports `research` or `tuning`.
- No evidence that `strategy` imports `research` or `tuning`.
- No evidence that `engine` imports `research` or `tuning` directly.

That is an important positive result: the computational core is cleaner than the configuration/governance surface.

## Hidden Coupling

### 1. Runtime Depends On Research Capture

`app/runtime_sinks.py` imports:

- `research.signal_evaluation.save_signal_evaluation`
- `research.signal_evaluation.should_capture_signal`
- `tuning.shadow.append_shadow_log`
- `tuning.shadow.compare_shadow_trade_outputs`

This means the live runtime is not just producing trades. It is also the attachment point for:

- research dataset persistence
- capture policy enforcement
- candidate-vs-production shadow comparison

That is operationally useful, but it tightly couples runtime orchestration to research/governance concerns.

### 2. Signal Engine Emits A Research-Grade Payload

`engine.signal_engine.generate_trade(...)` does not emit a minimal execution instruction. It emits a very large audit payload containing:

- analytics diagnostics
- move probability components
- confirmation breakdown
- macro/news fields
- global risk fields
- overlay diagnostics
- ranked strike candidates
- sizing details

This is convenient and valuable, but it means the core trade-construction module is also a serialization/reporting surface.

### 3. Historical Paths Still Approximate Some Live Inputs

`backtest.intraday_backtester` and `backtest.replay_regression` now call `run_preloaded_engine_snapshot(...)` rather than `generate_trade(...)` directly.

That removes the earlier orchestration bypass for:

- macro-event evaluation
- headline-state assembly
- macro-news aggregation
- global-risk state construction
- result payload shaping

Remaining parity risk comes from the quality of the historical inputs:

- synthesized spot OHLC context in bar-based backtests
- neutral global-market fallbacks when historical cross-asset context is unavailable

### 4. Research Dataset Is The Tuning Substrate

`tuning.experiments` reads `research.signal_evaluation.dataset`.

That is intentional and sensible, but it means:

- tuning quality depends directly on the fidelity of signal capture and outcome backfills
- production runtime correctness and research dataset correctness are tightly linked

If signal capture drifts, tuning and promotion quality drift with it.

## Modules With Too Many Responsibilities

### `app.engine_runner`

Responsibilities currently combined:

- data acquisition
- preloaded snapshot orchestration
- replay loading
- provider credential setup
- expiry resolution
- validation
- macro-event evaluation
- headline ingestion
- global-market snapshot acquisition
- parameter-pack switching
- signal-engine execution
- research capture
- shadow evaluation
- result payload assembly

This is the clearest orchestration hotspot in the repo.

### `engine.signal_engine`

Responsibilities currently combined:

- analytics-state consumption
- probability consumption
- direction handling
- scoring and overlay accumulation
- global-risk gating
- strike selection
- contract-level option-efficiency evaluation
- position sizing
- final message and payload shaping

This is the core signal brain, but it is also close to becoming an orchestration layer in its own right.

### `tuning.registry`

Responsibilities currently combined:

- parameter definition registry
- group metadata
- config-policy introspection
- macro-config exposure
- tuning search metadata

It is both the schema of tunable parameters and the bridge back into configuration modules.

## Architecture Match Vs Drift

### Where The Code Matches A Clean Layered Design

- Analytics feed market-state assembly rather than strategy directly.
- Probability assembly is a separate layer between analytics and signal scoring.
- Strategy code is mostly isolated to confirmation, trade-strength, strike selection, exits, and budget sizing.
- Risk overlays are explicit modules instead of hidden inside the signal engine.
- Research, evaluation, reporting, tuning, and promotion are mostly downstream consumers of captured signal data.

### Where The Implementation Has Drifted

- Configuration is dynamic and parameter-pack aware, so `config` is not a pure static upstream layer.
- Live runtime is coupled to research capture and shadow governance through `app.runtime_sinks`.
- Backtests and replay utilities now share the same post-load orchestration path as live runtime, but historical data still approximates some live inputs.
- `engine.signal_engine` doubles as both decision engine and canonical trade-report serializer.

## Maintainability Assessment

### Stable Abstractions

- `analytics/*` leaf calculators
- `strategy/exit_model.py`
- `strategy/budget_optimizer.py`
- `research.signal_evaluation.dataset`
- `research.signal_evaluation.market_data`

These modules are relatively focused and easy to reason about.

### Likely Refactor Hotspots

- `app/engine_runner.py`
- `engine/signal_engine.py`
- `tuning/runtime.py`
- `tuning/registry.py`
- `config/*_policy.py` getter pattern

These are the modules most likely to accumulate additional coupling as the engine evolves.

## Recommended Hardening Steps

1. Continue shrinking `app.engine_runner` around its new orchestration stages:
   `market_input_loader`, `snapshot_context_builder`, `signal_executor`, `runtime_postprocessors`.

2. Continue migrating operator surfaces toward `execution_trade` while keeping `trade_audit` for research and governance.

3. Treat `config.policy_resolver` as the only production policy-resolution boundary:
   keep registry-dependent behavior on the tuning side of the repo.

4. Improve historical data fidelity around the shared parity path:
   especially point-in-time global-market context and richer intraday spot summaries.

5. Keep research/tuning hooks attached at the `app` layer, not in the engine core:
   the code is already mostly following this boundary and should preserve it.
