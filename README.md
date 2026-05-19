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

## Live Governance Layers

The current options engine now includes a governed live decision stack on top of the raw signal logic:

- market-data readiness scoring for operator triage
- historical outcome and regime-segment guards before TRADE promotion
- portfolio concentration and book-heat throttles for same-way crowding
- session-level loss budget and cooldown controls
- same-day trade-slot caps with constrained operator override support
- a final promotion gate that can require replay validation before live TRADE routing

These controls are designed to keep the engine evaluation-first and desk-safe while still preserving useful watchlist intelligence when a setup is not yet executable.

## Operational Doctrine

Use this repository as a governed signal and research engine.

- Keep the workflow signal-first, evaluation-first, and policy-governed.
- Treat this system as a signal engine, not an execution router.
- Change runtime behavior through parameter packs first, not ad-hoc code edits.
- Base promotion decisions on signal-evaluation evidence, not discretionary trade anecdotes.

## Current Operating Snapshot

As of the latest local review on 2026-05-18:

- the default option-chain source is `ICICI` through `OQE_DEFAULT_DATA_SOURCE`
- `baseline_v1` remains the active runtime parameter pack unless the operator explicitly sets another pack
- `candidate_v1` contains the reviewed `composite_signal_score >= 85` threshold candidate, but it is not active runtime behavior until deliberately selected and reconciled
- live CLI runs default to saving raw spot and option-chain snapshots for replayable signal research
- the signal dataset captures market-data provenance, canonical PCR fields, selected-option entry premium, and selected-option bid/ask/mid when provider quotes are available
- selected-contract option premium paths can be reconstructed from saved option-chain snapshots for later P&L analysis
- regime-parameter artifacts remain research-only unless fresh-forward validation, explicit runtime toggles, and audit output are added
- forward backlog, audit reports, research reports, and plan documents are local-only artifacts; the root README is the only versioned operator document

## Direction Head Governance Note

Direction-head promotion governance in this repository is currently signal-quality-first by design.

- Promotion gates are evaluated on directional signal quality metrics (for example directional accuracy delta and directional return delta).
- Trade-level confirmation is optional for this engine objective because the system is intended to generate high-quality decision signals for discretionary use, not to auto-route execution.
- The policy is encoded in the promotion matrix runner and CI workflow defaults.

## Compact Table Of Contents

- [Current Operating Snapshot](#current-operating-snapshot)
- [Quick Start](#quick-start)
- [10-Minute Runbook](#10-minute-runbook)
- [Main Workflows](#main-workflows)
- [Repair Queue Ops](#repair-queue-ops)
- [Parameter Packs](#parameter-packs)
- [Promotion And Shadow Mode](#promotion-and-shadow-mode)
- [Tuning Workflow](#tuning-workflow)
- [Testing](#testing)
- [Configuration](#configuration)

## Quick Start

### Environment

Recommended runtime: Python 3.11.x.

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill in provider credentials only for the routes you want to use. The default live option-chain route is ICICI; switch deliberately with `OQE_DEFAULT_DATA_SOURCE=ZERODHA` only when that provider is intended.

Notes:

- on macOS, prefer `python3.11` when multiple Python versions are installed
- the commands elsewhere in this README are written in Unix-style form because
  the repo has primarily been developed on macOS
- on Windows, once the virtual environment is activated, the Python commands
  are usually the same, for example `python main.py` and
  `python -m streamlit run app/streamlit_app.py`
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
python -m streamlit run app/streamlit_app.py
```

If your shell resolves to system Python instead of the repo virtual environment, use an explicit path:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Common launch pitfalls:

- use `app/streamlit_app.py` (not `streamlit_app.py` from repo root)
- use `python -m streamlit ...` (not `python streamlit ...`)
- if you see `No module named streamlit`, install deps in the same interpreter used to launch:

```bash
python -m pip install -r requirements.txt
```

Troubleshooting:

- if Streamlit exits with code `143`, the process was terminated externally (for example Ctrl+C, terminal cleanup, or a kill signal), not a missing-package failure
- verify interpreter selection when multiple Python versions are installed:

```bash
python3 --version
.venv/bin/python --version
```

- on this repo, prefer launching with the venv-qualified interpreter to avoid system-Python fallback:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

**Structure Tab Visualization:**

The Structure tab now includes four market-structure charts plus a new ATM-centered Put-Call OI Ratio (PCR) widget:

1. **Open Interest by Strike** — Bar chart of PE/CE cumulative OI across strikes
2. **IV Smile** — Line chart showing implied volatility skew by strike
3. **Option Premium Curve** — Line chart of last prices (CE/PE) by strike  
4. **Change in OI by Strike** — Bar chart of PE/CE daily OI changes
5. **Put-Call OI Ratio (ATM-Centered)** — Line chart of PE/CE OI ratio focused around At-The-Money (ATM) with adjustable ±N strikes window for intraday actionability

The PCR chart:

- Auto-anchors to ATM using median underlying value when available, or highest combined OI as fallback
- Includes slider to zoom in/out (±1 to ±20 strikes, default ±8)
- Shows only the focused window for cleaner intraday reads
- Displays PCR anchor and signal count in caption

### Refresh Research Outcomes

```bash
python scripts/update_signal_outcomes.py
```

To enrich selected-contract option premium paths for future P&L research, run:

```bash
python scripts/update_signal_outcomes.py --option-premium-paths
```

Live CLI runs default to saving spot and option-chain snapshots, so this premium-path enrichment can reconstruct 5m/15m/30m/60m/120m option marks from the saved chain files.

For end-of-day local maintenance, refresh the cumulative dataset and premium paths with:

```bash
python scripts/refresh_cumulative_signal_dataset.py
```

Generated signal datasets, SQLite mirrors, audit outputs, research reports, plan documents, and saved snapshots are intentionally ignored by git.

### Research Reporting

```bash
python scripts/signal_evaluation_report.py
```

### Daily Research Report

```bash
python scripts/ops/run_daily_research_workflow.py --date YYYY-MM-DD --include-cumulative
```

### Signal-Quality Model Audit

```bash
python scripts/ops/run_signal_quality_model_audit.py
```

This research-only audit measures probability calibration, regime-conditioned
calibration bias, feature stability, and EV/risk-adjusted ranking strength from
the signal-evaluation dataset. It writes JSON, Markdown, and CSV diagnostics
under `research/signal_evaluation/reports/signal_quality_model_audit/` and does
not run the engine, change data sources, alter parameter packs, or place
trades.

### Probability Calibration Experiment

```bash
python scripts/ops/run_probability_calibration_experiment.py
```

This research-only experiment compares raw probabilities against fitted
calibration mappings on a chronological train/holdout split using
quality-approved labels. It writes review artifacts under
`research/signal_evaluation/reports/probability_calibration_experiment/` and
does not apply the selected mapping to runtime config or parameter packs.

### Segmented Probability Calibration Experiment

```bash
python scripts/ops/run_segmented_probability_calibration_experiment.py
```

This research-only experiment searches for calibration candidates inside
regime slices and recent training windows. It emits guarded review artifacts
under
`research/signal_evaluation/reports/segmented_probability_calibration_experiment/`
and keeps runtime probabilities unchanged unless a future human-controlled
promotion workflow explicitly adopts a vetted bundle.

### Segmented Probability Forward Shadow

```bash
python scripts/ops/run_segmented_probability_forward_shadow.py
```

This research-only validator applies the latest segmented calibration candidate
bundle to labeled rows with explicit routing policies such as
`candidate_priority`, `regime_first`, and `recency_first`. It compares raw
versus shadow-calibrated probabilities, writes review artifacts under
`research/signal_evaluation/reports/segmented_probability_forward_shadow/`,
and leaves runtime probabilities, data sources, parameter packs, and execution
behavior unchanged.

### Segmented Probability EV Shadow Evaluation

```bash
python scripts/ops/run_segmented_probability_ev_shadow_evaluation.py
```

This research-only evaluator scores the same segmented calibration routes by
realized trading usefulness: top-bucket hit rate, signed return,
MAE/MFE-adjusted return, spread/liquidity quality, option-chain quality, and
regime-conditioned payoff. It writes JSON, Markdown, and CSV artifacts under
`research/signal_evaluation/reports/segmented_probability_ev_shadow_evaluation/`
without changing runtime probabilities, data sources, parameter packs, or
execution behavior.

### Segmented Probability EV Rejection Attribution

```bash
python scripts/ops/run_segmented_probability_ev_rejection_attribution.py
```

When EV shadow evaluation rejects a segmented-probability candidate, this
research-only attribution pass explains the failure by comparing the raw top
bucket against the shadow-calibrated top bucket, highlighting damaging
candidate routes, shadow-only promoted rows, regime pockets, and policy-level
alternatives. It writes JSON, Markdown, and CSV artifacts under
`research/signal_evaluation/reports/segmented_probability_ev_rejection_attribution/`
and leaves runtime probabilities, data sources, parameter packs, and execution
behavior unchanged.

### Segmented Probability Guarded EV Experiment

```bash
python scripts/ops/run_segmented_probability_guarded_ev_experiment.py
```

After EV rejection attribution identifies damaging routes or top-bucket
selection damage, this research-only experiment tests guarded alternatives:
quarantining EV-negative candidate routes, enforcing raw-rank preservation, and
combining both. It compares each variant against the raw top bucket and the
current rejected shadow ranking, then writes advisory JSON, Markdown, and CSV
artifacts under
`research/signal_evaluation/reports/segmented_probability_guarded_ev_experiment/`.
It never edits the candidate bundle, runtime config, parameter packs, data
sources, or execution behavior.

### Segmented Probability Guarded Candidate Bundle

```bash
python scripts/ops/run_segmented_probability_guarded_candidate_bundle.py
```

When the guarded EV experiment passes, this research-only generator writes a
new candidate bundle that removes EV-negative candidate routes and records the
raw-rank preservation policy as governance metadata. The bundle remains
approval-gated and requires guard-aware forward shadow, guard-aware EV shadow,
EV rejection attribution, and readiness checks before any manual review. It
writes artifacts under
`research/signal_evaluation/reports/segmented_probability_guarded_candidate_bundle/`
and never edits runtime config, parameter packs, data sources, or execution
behavior.

### Segmented Probability Guard-Aware Shadow Validation

```bash
python scripts/ops/run_segmented_probability_guarded_shadow_validation.py
```

This research-only validator consumes the guarded candidate bundle and applies
its raw-rank preservation policy during shadow evaluation. It reports both
calibration behavior and EV/risk top-bucket behavior using guarded ranking
probabilities, writes JSON, Markdown, and CSV artifacts under
`research/signal_evaluation/reports/segmented_probability_guarded_shadow_validation/`,
and leaves runtime probabilities, data sources, parameter packs, and execution
behavior unchanged.

### Segmented Probability Forward Shadow Accumulator

```bash
python scripts/ops/run_segmented_probability_forward_shadow_accumulator.py
```

This research-only accumulator runs the forward-shadow validator in auto mode,
appends the latest replay/true-forward state to an audit history, and refreshes
a dashboard showing whether enough post-candidate labels have arrived for true
forward validation. It automatically keeps using holdout replay until the
configured forward-label sample threshold is met, then switches to true
post-candidate rows without changing runtime behavior.

### Segmented Probability Forward Shadow Readiness Gate

```bash
python scripts/ops/run_segmented_probability_forward_shadow_readiness.py
```

This research-only gate reads the latest forward-shadow, accumulation,
candidate-staleness, EV/risk shadow, and guard-aware shadow validation
artifacts. When guard-aware validation is present, it uses that guarded
EV/ranking evidence as the primary payoff gate and keeps the legacy EV shadow
artifact as fallback context. It blocks manual calibration review unless true
post-candidate validation has enough labels, the recommended routing policy is
stable, route regressions are absent, schemas are valid, candidate staleness is
`ACTIVE_REVIEW`, the bundle is not expired or superseded, no material
forward-label population shift is detected, guard-aware top-bucket payoff and
hit rate do not regress, quarantined routes have zero top-bucket exposure, the
guarded bundle remains research-only and approval-gated, and all side-effect
flags remain false. `--allow-holdout-replay-guarded-validation` can be used for
explicit research review of guarded holdout replay evidence, but it is not a
runtime adoption approval.

### Segmented Probability Shadow Soak

```bash
python scripts/ops/run_segmented_probability_shadow_soak.py
```

This is the one-line daily research loop for the guarded segmented-probability
candidate. It first refreshes pending realized outcomes using the local spot
history store by default, then appends forward-shadow accumulation history,
refreshes candidate staleness, refreshes legacy EV shadow context, reruns
guard-aware validation, reruns the readiness gate, and writes a compact
soak-status report under
`research/signal_evaluation/reports/segmented_probability_shadow_soak/`.
The command tracks true-forward sample progress, no-new-label days, guarded
EV/ranking deltas, quarantined-route exposure, original and guarded staleness
state, readiness status, outcome-refresh progress, post-guarded true-forward label accumulation,
and bundle hash immutability. It also appends a guarded soak history CSV so
operators can see whether quality-approved labels after the guarded bundle are
actually increasing across sessions. It never edits runtime config, parameter
packs, data sources, candidate bundles, or execution behavior. Use
`--outcome-refresh-source skip` to inspect the existing dataset without
refreshing labels, or `--outcome-refresh-source default_provider` when an
external research backfill provider is explicitly desired.

### Segmented Probability Guarded Candidate Staleness

```bash
python scripts/ops/run_segmented_probability_guarded_candidate_staleness.py
```

This research-only governance check evaluates the guarded candidate bundle
separately from the original source candidate. It marks the guarded bundle as
`GUARDED_ACCUMULATING_FORWARD_LABELS`, `GUARDED_ACTIVE_REVIEW`,
`GUARDED_STALE_WATCH`, `GUARDED_EXPIRED`, `GUARDED_SUPERSEDED`, or
`GUARDED_BLOCKED` using guarded-bundle age, post-guarded data, post-guarded
label shift, guarded routing-policy stability from the soak history, and newer
guarded-bundle detection. It never changes runtime config, parameter packs,
data sources, candidate bundles, or execution behavior.

### Monday Readiness Preflight

```bash
python scripts/ops/run_monday_readiness_preflight.py
```

This read-only preflight summarizes the selected option data source, latest
dataset state, guarded-bundle staleness, shadow-soak blockers, and the exact
next commands to run once market data starts flowing. It does not fetch
providers, refresh outcomes, change data sources, alter parameter packs, write
candidate artifacts, or execute trades. Pass `--option-chain path/to/snapshot.csv
--spot 22500 --as-of 2026-05-18T09:20:00+05:30` to validate a saved
option-chain snapshot without contacting a live provider.

### Segmented Probability Candidate Staleness

```bash
python scripts/ops/run_segmented_probability_candidate_staleness.py
```

This research-only governance check marks a candidate bundle as
`ACTIVE_REVIEW`, `STALE_WATCH`, `EXPIRED`, or `SUPERSEDED` using candidate age,
new post-candidate data, forward-label population shift, routing-policy
stability, and newer bundle detection. It writes JSON/Markdown artifacts under
`research/signal_evaluation/reports/segmented_probability_candidate_staleness/`
and never changes runtime config, parameter packs, data sources, or execution
behavior.

### Daily Readiness Dashboard

```bash
python scripts/daily_readiness_dashboard.py --input-file research/signal_evaluation/cumulative_signals.parquet --output-dir research/daily_readiness
```

This produces a CSV summary, text report, and optional JSON artifacts when `--write-json` is supplied.

### Offline Pack Replay Suite (Resumable)

```bash
python scripts/ops/run_offline_replay_pack_suite.py
```

Resume an existing suite directory without re-running completed tasks:

```bash
python scripts/ops/run_offline_replay_pack_suite.py \
  --resume-dir research/parameter_tuning/offline_replay_runs/suite_YYYYMMDD_HHMMSS
```

### Multiyear Backtest

```bash
python scripts/run_multiyear_backtest.py
```

### Build ML Model Registry

```bash
python scripts/build_model_registry.py
```

## 10-Minute Runbook

This is the fastest practical onboarding path for a new operator.

1. Set up local dev runtime (2-3 minutes)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python main.py
```

Expected outcome: start the engine and confirm it prints a structured snapshot without execution routing.

1. Run replay smoke check (2-3 minutes)

```bash
python main.py --replay
```

Expected outcome: confirm the deterministic snapshot evaluation path runs end-to-end.

1. Run governance sanity checks (2-3 minutes)

```bash
python scripts/parameter_governance.py evaluate-current
python scripts/parameter_governance.py tune --group trade_strength --group option_efficiency
```

Expected outcome: confirm governed comparison artifacts and candidate deltas are generated under `research/parameter_tuning/`.

1. Rehearse promotion flow (2 minutes)

```bash
python scripts/parameter_governance.py approve-candidate --reviewer your_name
python scripts/parameter_governance.py promote-candidate --approved-by your_name
```

Expected outcome: confirm promotion state and ledger update with an auditable trail, without requiring ad-hoc code edits.

If you only have 10 minutes, run steps 1 and 2 first, then run steps 3 and 4 before any production-facing policy change.

## Repo Hygiene Policy

To keep the repository maintainable and reproducible, use these artifact path conventions.

- This repository is scoped to the options engine itself. Future cross-asset platform code, scaffolds, roadmaps, and platform-level planning should live in the parent `Quant Engines/` folder, not inside this repo.
- Human-readable operator docs, plans, reviews, summaries, and deployment guides are local-only artifacts. Keep them in the ignored `documentation/` archive or the parent local docs tree. Only the root `README.md` should remain versioned here.
- Runtime validation and profiling outputs: store under `research/runtime_validation/`.
- Backtest logs and comparison outputs: store under `research/runtime_validation/backtest_runs/`.
- Signal-evaluation reports and generated tables: store under `research/signal_evaluation/reports/`.
- One-off analysis outputs from ad-hoc scripts: store under `research/` with a date-stamped subfolder.
- Audit reports, deployment reviews, and ad-hoc docs snapshots: store in the local docs archive (intentionally excluded from git).
- Avoid writing generated artifacts to repository root.
- Keep archival runbooks and review memos local-only unless their durable operating rules are summarized in this README.

Commit hygiene:

- Keep code changes, script reorganization, and generated research artifacts in separate commits.
- When moving scripts, keep compatibility wrappers at legacy entrypoints until automation is migrated.
- Do not delete intermediate or final research artifacts that are needed for auditability.

Offline replay suite hygiene:

- Use `scripts/ops/run_offline_replay_pack_suite.py` for baseline-vs-candidate replay checks.
- Reuse the same suite directory with `--resume-dir` so completed tasks are skipped via checkpoint state.
- Each invocation writes a dated sub-run under `subruns/` and appends `run_history.csv`.
- The suite is offline-only and does not mutate production runtime behavior.

## Current System Shape

The engine now has five important overlay packages plus a dedicated research-governance stack sitting on top of the core microstructure signal path:

- `macro/` and `news/`: scheduled event risk, headline classification, macro/news aggregation
- `risk/global_risk_*`: external/global regime classification, overnight gap risk, volatility expansion risk
- `risk/gamma_vol_acceleration_*`: convexity and acceleration overlay
- `risk/dealer_hedging_pressure_*`: dealer-flow and pinning/acceleration overlay
- `risk/option_efficiency_*`: expected move and option-buying efficiency overlay
- `models/heston/`: optional research-only stochastic-volatility diagnostics; Black-Scholes remains the live Greek engine
- `tuning/`: parameter registry, named packs, experiment runner, advanced search, automated campaigns, promotion, and reporting
  plus walk-forward and regime-aware validation
- `strategy/`: strike selection, confirmation scoring, direction reversal control, enhanced strike scoring with market-microstructure factors, exit model, budget optimization, and trade strength evaluation
- `utils/`: centralized numeric helpers (`clip`, `safe_float`, `safe_div`, `to_python_number`), math functions (`norm_pdf`, `norm_cdf`), and timestamp utilities (`coerce_timestamp`)

These layers are intentionally modifiers and filters. They do not replace the core directional engine.

## Main Workflows

### Canonical Script Map

- Live engine loop: `python main.py`
- Replay snapshot run: `python main.py --replay`
- Daily research workflow: `python scripts/ops/run_daily_research_workflow.py --date YYYY-MM-DD --include-cumulative`
- Empirical regime outcome tables: `python scripts/ops/run_regime_outcome_tables.py`
- Daily+cumulative signal reports: `python scripts/reports/run_signal_evaluation_reports.py`
- Signal-report PDFs (canonical): `python scripts/reports/generate_professional_pdfs.py`
- Watchlist second-pass scheduler: `python scripts/schedule_watchlist_second_pass.py --target-date YYYY-MM-DD`
- Watchlist realized evaluation (scheduler target): `python scripts/watchlist_realized_evaluation.py`
- PCR backfill from saved chains: `python scripts/ops/backfill_pcr_fields.py --write`
- Option premium path backfill from saved chains: `python scripts/ops/backfill_option_premium_paths.py --write`
- Offline Heston diagnostics backfill: `python scripts/ops/backfill_heston_research_features.py --write`
- Heston research diagnostics: `python scripts/ops/run_heston_research_report.py`
- Parameter governance and tuning: `python scripts/parameter_governance.py ...`
- Offline replay pack suite: `python scripts/ops/run_offline_replay_pack_suite.py ...`

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
- optional Heston research fields are captured when `OQE_HESTON_RESEARCH_ENABLED=true`; they are diagnostic only and never alter trade decisions
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
- offline replay pack suite with resumable checkpoints in [run_offline_replay_pack_suite.py](scripts/ops/run_offline_replay_pack_suite.py)

Important distinction:

- `tuning/` contains the parameter-tuning and promotion code
- `research/parameter_tuning/` stores runtime-generated research artifacts, ledgers, reports, state, and candidate packs

## Repair Queue Ops

Use these commands for legacy contract-repair operations.

Generate HIGH-only repair artifacts and review queue (safe dry-run):

```bash
python scripts/backfill_signal_contract_fields.py \
  --dataset cumulative \
  --apply-repair-proposals \
  --emit-audit \
  --emit-repair-proposals \
  --high-confidence-only \
  --dry-run
```

Apply MEDIUM+HIGH proposals and promote repaired copy into source cumulative dataset (with backup):

```bash
python scripts/backfill_signal_contract_fields.py \
  --dataset cumulative \
  --apply-repair-proposals \
  --emit-audit \
  --emit-repair-proposals \
  --min-proposal-confidence MEDIUM \
  --promote-repaired-dataset
```

Approve selected MEDIUM rows from an existing review queue without regenerating proposals:

```bash
python scripts/approve_repair_review_queue.py \
  --review-queue-csv research/signal_evaluation/backfill_audit/contract_match_audit_YYYYMMDD_HHMMSS/repair_review_queue.csv \
  --dataset-path research/signal_evaluation/signals_dataset_cumul.csv \
  --signal-ids "signal_id_1,signal_id_2" \
  --promote-approved
```

Run daily refresh with automatic HIGH-only review artifact emission:

```bash
python scripts/schedule_refresh_cumulative_daily.py \
  --dataset-path research/signal_evaluation/signals_dataset_cumul.csv \
  --emit-high-only-review-artifacts
```

Enable automatic HIGH-confidence promotion in production cron/task:

```cron
# every day at 18:35 IST
35 18 * * * cd /Users/pramitdutta/Desktop/Quant\ Engines/options_quant_engine && \
  /Users/pramitdutta/Desktop/Quant\ Engines/options_quant_engine/.venv/bin/python \
  scripts/schedule_refresh_cumulative_daily.py \
  --dataset-path research/signal_evaluation/signals_dataset_cumul.csv \
  --emit-high-only-review-artifacts \
  --promote-high-confidence-repairs
```

### 6. ML Research Evaluation

- policy robustness analysis: `python research/ml_evaluation/policy_robustness/robustness_runner.py`
- rank-gate + confidence-sizing research: `python research/ml_evaluation/rank_gate_sizing/rank_gate_sizing_runner.py`
- predictor method comparison: `python research/ml_evaluation/predictor_comparison/predictor_comparison_runner.py`
- each runner reads from the canonical signal datasets, produces structured reports (`.md`, `.json`, `.csv`), and generates visualizations (`.png`)
- generated research outputs are stored under `research/` for local audit and may be git-ignored; runner scripts are the reproducible source of truth

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

## March 2026 Integration Update

This repository now integrates additional analytics into direction inference,
trade-strength scoring, confirmation, and risk overlays.

Integrated signals:

- OI velocity
- risk reversal skew and momentum
- ATM volume PCR
- gamma flip drift
- expiry-conditioned max pain pinning

Core implementation surfaces:

- direction votes and signal state: `engine/trading_support/signal_state.py`
- trade-strength components: `strategy/trade_strength.py`
- confirmation PCR path: `strategy/confirmation_filters.py`
- overlay enrichments:
  - `risk/gamma_vol_acceleration_features.py`
  - `risk/dealer_hedging_pressure_features.py`
- orchestration pass-through: `engine/signal_engine.py`
- runtime policy defaults: `config/signal_policy.py`

Default runtime toggles (all enabled):

- `use_oi_velocity_in_direction = 1`
- `use_rr_in_direction = 1`
- `use_pcr_in_confirmation = 1`
- `use_flip_drift_in_overlays = 1`
- `use_max_pain_expiry_overlay = 1`

Selected default thresholds:

- `oi_velocity_vote_on = 0.18`
- `rr_skew_put_dominant = 0.75`
- `rr_skew_call_dominant = -0.75`
- `volume_pcr_atm_put_dominant = 1.20`
- `volume_pcr_atm_call_dominant = 0.80`
- `gamma_flip_drift_pts_vote_on = 80`
- `max_pain_overlay_max_dte = 2`
- `max_pain_pin_distance_pts_min = 80`

Recommended staged rollout order:

1. `use_pcr_in_confirmation`
2. `use_oi_velocity_in_direction`
3. `use_rr_in_direction`
4. `use_flip_drift_in_overlays`
5. `use_max_pain_expiry_overlay`

Validation status for this integration:

- targeted integration tests passed when the integration landed
- rerun the current regression suite before using these notes for a fresh promotion decision

Detailed implementation and rollout notes should stay in the ignored local documentation tree; durable operating rules belong in this README.

## Pluggable Predictor Architecture

The engine now supports multiple prediction methods for move direction and magnitude. Switch between methods without code changes using configuration.

### Available Prediction Methods

| Method | Type | AUC (test) | Best For |
| --- | --- | --- | --- |
| `blended` (default) | Hybrid | N/A | Production: balances interpretability with performance |
| `pure_ml` | ML only | 0.65 | Research: isolate ML contribution |
| `pure_rule` | Rule only | N/A | Research: validate heuristic baselines |
| `research_dual_model` | Ensemble | 0.6525 (GBT) | Research: two-model ensemble with calibration |
| `research_decision_policy` | Policy overlay | 0.74 (hit rate) | Research: dual-model with policy filtering |
| `ev_sizing` | EV-based | — | Research: size by expected value |

### Switching Methods

Set environment variable or config:

```bash
export OQE_PREDICTION_METHOD=pure_ml
python main.py
```

Or in backtests:

```bash
python scripts/run_multiyear_backtest.py --prediction-method research_dual_model
```

For more details, see the local developer guide for pluggable predictor architecture.

## ML Model Registry and Selection

Trained ML models are stored in the local registry under `models_store/registry/`. That directory is git-ignored; rebuild or refresh it with the model-registry script when needed.

### Available Models

Trained on 60-minute directional accuracy (`correct_60m_all` target, 2,701 samples):

| Model | AUC | ECE | Criteria | Notes |
| --- | --- | --- | --- | --- |
| `GBT_shallow_v1` | **0.6525** | 0.1235 | 3/4 | Highest AUC; requires calibration |
| `GBT_shallow_platt_v1` | 0.6637 | 0.1094 | 3/4 | GBT + Platt calibration |
| `LogReg_ElasticNet_v1` ★ | 0.6295 | **0.0818** | **4/4 ALL** | Best calibration; passes all criteria |
| `LogReg_L2_v1` | 0.6248 | 0.1729 | 3/4 | L2 regularization baseline |
| `RF_shallow_v1` | 0.6232 | 0.1029 | 3/4 | Random forest baseline |

**Criteria**: AUC ≥ 0.60, Ranking stability, Generalization, Calibration (ECE ≤ 0.10).

### Switching Active Model

```bash
export OQE_ACTIVE_MODEL=GBT_shallow_v1
python main.py
```

Or in configuration (`config/settings.py`):

```python
ACTIVE_MODEL = "LogReg_ElasticNet_v1"  # Best calibration
```

Without `ACTIVE_MODEL` set, the system falls back to legacy `move_predictor_v2.joblib`. To build or update the registry:

```bash
python scripts/build_model_registry.py
```

For details, see the local developer guide section on model registry and switching.

## Decision Policy Layer

Signals can be evaluated under configurable decision policies for filtering or downsizing.

### Available Policies

- **agreement_only**: Require multi-source directional agreement
- **rank_filter**: Block bottom K% by confidence  
- **dual_threshold**: Dual-model predictions with policy thresholds
- **sizing_simulation**: Conditional size adjustments

### Policy Performance (7,404 signals)

| Method | Signals | Hit Rate 60m | Avg Return 60m | vs Baseline |
| --- | --- | --- | --- | --- |
| Baseline (all) | 100% | 50.35% | -2.60 bps | — |
| Research Dual-Model | 54.6% | 67.48% | +10.92 bps | +17.13pp |
| **Dual Threshold Policy** | **48.6%** | **74.12%** | **+18.98 bps** | **+23.77pp** |
| Rank Filter 30% | 70.0% | 70.97% | +19.34 bps | +20.62pp |

Test under any policy with:

```bash
python scripts/run_multiyear_backtest.py --prediction-method research_decision_policy
```

See the local developer guide section on the decision policy layer for implementation details.

## Performance Profile

Recent optimizations achieve **31-45x speedup** over baseline engine (warm runs):

| Component | Baseline | Optimized | Improvement |
| --- | --- | --- | --- |
| Full engine | 1,250-1,810ms | **~40ms** | **31-45x** |
| Total analytics | ~46ms | **25ms** | 1.8x |
| Greeks enrichment | 21.8ms | **1.9ms** | 11.5x |
| Engine overhead | 900-1,400ms | **14.7ms** | 61-95x |

### Key Optimization Insights

1. **Config resolution cache** (~70% speedup): Eliminates repeated dataclass instantiation per signal
2. **TTE parse cache** (11.5x): Pre-caches time-to-expiry calculations per unique expiry
3. **yfinance TTL cache** (~350ms/tick): 5-minute cache for market snapshots
4. **Redundant call removal**: Eliminated duplicate `attach_trade_views` and `split_trade_payload` iterations
5. **Probability hot-path simplification**: model feature construction now uses a single rich `build_features` call with legacy fallback instead of duplicate builds

For profiling details, see the local developer guide section on performance caching systems.

### Reproducible Hotspot Benchmarks (p50/p95)

To quantify runtime hotspots with reproducible traces (and keep optimization passes measurable), use:

```bash
python scripts/micro_benchmark_hotspots.py --iterations 200 --warmup 25
```

This script benchmarks:

1. spot-history load latency
2. event-feature aggregation latency

and writes both summary + raw artifacts under `debug_samples/performance/`:

- `micro_benchmark_hotspots_<timestamp>.json`
- `micro_benchmark_hotspots_<timestamp>_raw.csv`

To compare two benchmark runs (absolute and percent deltas):

```bash
python scripts/compare_micro_benchmark_hotspots.py
```

By default this auto-selects the latest two summaries. You can also pass explicit files:

```bash
python scripts/compare_micro_benchmark_hotspots.py \
  --baseline debug_samples/performance/micro_benchmark_hotspots_<old>.json \
  --current debug_samples/performance/micro_benchmark_hotspots_<new>.json
```

The compare script prints delta tables and writes a comparison artifact:

- `micro_benchmark_hotspots_comparison_<timestamp>.json`

## March 2026 Runtime Stability Updates

Recent production-hardening updates focused on evaluation correctness, regime semantics, and hot-path efficiency:

1. **Strict `as_of` gating in signal evaluation**: Outcome enrichment now hard-limits realized paths to the requested `as_of` timestamp and prevents accidental forward leakage when running partial backfills.

1. **Regime-conditional threshold defaults aligned to intent**: `POSITIVE_GAMMA` uses looser thresholds, longer holds, and a larger sizing multiplier, while `NEGATIVE_GAMMA` uses tighter thresholds, shorter holds, and a smaller sizing multiplier.

1. **Volume PCR edge-case handling**: Zero-call / positive-put volume is now treated as an extreme put-dominant reading rather than `UNAVAILABLE`.

1. **ML predictor hot-reload behavior**: Probability predictor now reloads when `ACTIVE_MODEL` or registry artifact signature changes, avoiding stale-model behavior in long-running sessions.

1. **Gamma normalization consistency**: Fallback gamma-exposure distance now uses spot-normalized moneyness scaling for consistency across gamma analytics paths.

Validation snapshot:

- the stability changes passed the then-current regression suite when they landed
- use the current `pytest -q` result as the authority for new changes

## Parameter Tuning and Governance

The tuning pipeline supports grid search, walk-forward validation, and regime-aware testing.

### Quick Parameter Search

```bash
python tuning/search.py --param signal_policy.trade_strength_floor --min 50 --max 80 --step 5
```

### Recent Findings (March 2026)

Optimal thresholds on 7,404-signal backtest dataset:

| Parameter | Optimal Value | Holdout HR | Signal Count |
| --- | --- | --- | --- |
| `trade_strength_floor` | 60 | 100% | 25 |
| `composite_score_floor` | 75 | — | — |
| `tradeability_floor` | 65 | — | — |
| `move_probability_floor` | 0.60 | — | — |
| `option_efficiency_floor` | 40 | — | — |
| `global_risk_cap` | 70 | — | — |

Tighter thresholds improve hit rate from 62.4% → 100% (on holdout set) but reduce signal volume from 93 → 25.

### Current Tuning Caveats

- `evaluation_thresholds.selection.move_probability_floor` is probability-scaled (`0.0` to `1.0`); score-like floors remain `0` to `100`.
- **Score-computation groups** (`trade_strength`, `confirmation_filter`, `large_move_probability`) don't affect pre-computed backtest datasets — must re-run signal generation to tune these.

For full governance workflow details, see the local developer guide tuning-workflow section.

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

### Direction Confirmation Stickiness & Reversal Control

#### Problem

Historical analysis revealed that confirmation status exhibits high persistence across direction reversals: when the engine reverses its directional bias (CALL → PUT or vice versa), the new direction frequently inherits a STRONG_CONFIRMATION or CONFIRMED status even at the reversal snapshot itself. This "stickiness" creates false confidence in direction changes and can lead to whipsaws.

#### Solution: Three-Tier Reversal Control

The engine now provides three complementary mechanisms to manage reversal stickiness, all tunable via `CONFIRMATION_FILTER_CONFIG` in `config/signal_policy.py`:

#### 1. Reversal Veto (recommended, default: 1 step)

The most effective mechanism. Forces newly-reversed directions to MIXED status for a configurable grace period (0−6 steps):

- `reversal_veto_steps = 0`: No veto (baseline reversal stickiness = 100%)
- `reversal_veto_steps = 1`: **RECOMMENDED** ⟹ reversal snapshots demoted to MIXED, flip_persist_ratio = 0%
- `reversal_veto_steps = 2+`: Extended grace period (unnecessary at veto_steps=1)

**Sweep Results** (from live NIFTY dataset): a 1-step veto eliminates 100% of reversal stickiness (flip_persist_ratio: 1.0 → 0.0) while maintaining overall self-transition rate at 0.63.

**Usage in Live/Replay:**

```python
# In app/engine_runner.py, pass reversal_age to generate_trade():
signal = generate_trade(
    ...,
    previous_direction=prior_direction,
    reversal_age=snapshot_age_since_last_flip,
)
```

#### 2. Direction-Change Penalty (bounded: 0−6 points)

Applies a one-time deduction to confirmation score only at the reversal snapshot (reversal_age=0):

- `direction_change_penalty = 0.0`: No penalty
- `direction_change_penalty = 1.0−6.0`: Scales confirmation score downward by fixed amount

Note: Single-penalty alone is insufficient to break stickiness on this dataset (reversal scores: 8.27−13.54). Useful for fine-tuning in combination with veto.

#### 3. Post-Reversal Decay Model (advanced)

Extends the penalty across N snapshots post-reversal with geometric decay:

- `direction_change_decay_steps`: window length (0−20)
- `direction_change_decay_factor`: decay multiplier per step (0.0−1.0, default 0.5)

Effective penalty at step k: `base_penalty × (decay_factor ^ k)`

Example: penalty=4.0, factor=0.5, steps=3

- Step 0 (reversal): -4.0
- Step 1: -2.0
- Step 2: -1.0

**Sweep Analysis**: Decay model alone is mathematically insufficient to demote observed reversal scores (would require penalty ≈ 8−12). Primary value is educational — demonstrates why threshold adjustment or veto is necessary.

#### Configuration & Tuning

Within `config/signal_policy.py` → `CONFIRMATION_FILTER_CONFIG`:

```python
"reversal_veto_steps": 1,                               # NEW (recommended: 1)
"direction_change_penalty": 0.0,                         # NEW
"direction_change_decay_steps": 0,                        # NEW
"direction_change_decay_factor": 0.5,                     # NEW
```

All are automatically exposed in the tuning registry under `confirmation_filter.core.*` and can be searched/swept:

```bash
python tuning/search.py --param confirmation_filter.core.reversal_veto_steps \
  --min 0 --max 6 --step 1 --objective flip_persist_ratio
```

#### Analysis Artifacts

Generated sweep outputs are local-only research artifacts. Regenerate them when needed instead of relying on versioned CSV or memo files.

Run fresh analysis:

```bash
python scripts/analyze_direction_confirmation_stickiness.py
```

This generates CSV artifacts plus detailed markdown memo covering:

- Baseline confirmation stickiness metrics
- Reversal persistence analysis
- Sweep results for all three mechanisms
- Margin analysis explaining why single-penalty fails

#### Implementation Notes

- `reversal_age` is threaded through: `generate_trade()` → `_compute_signal_state()` → `compute_confirmation_filters()`
- Default value is `None` everywhere (backward compatible, no behavioral change unless explicitly tuned)
- Veto is applied after all other confirmation factors (open, close, flow, hedge, gamma, etc.) are scored
- Veto forces `status = "MIXED"` if the computed rank would be STRONG/CONFIRMED within the grace window
- All three mechanisms coexist and are independent (can mix/match for research)

#### Test Coverage

All three mechanisms are unit-tested:

```bash
python -m pytest tests/test_confirmation_filters.py::test_reversal_veto_forces_mixed_on_reversal_snapshot -v
python -m pytest tests/test_confirmation_filters.py::test_direction_change_penalty_reduces_confirmation_score_on_reversal -v
python -m pytest tests/test_confirmation_filters.py::test_post_reversal_decay_applies_at_step_1 -v
```

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
├── tests/              # regression and scenario coverage
├── tuning/             # registry, packs, experiments, search, validation, promotion code
├── utils/              # centralized numerics, math helpers, timestamp utilities
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

### External Market Data (USD/INR, WTI Crude Oil)

Cross-market spillover signals feed macro/statistical context, global risk checks, and future regime work.

Historical cache files, when present locally:

```bash
data/cache/usd_inr_historical_365d.csv       # 257 trading days
data/cache/wti_historical_365d.csv           # 251 trading days
```

Live/real-time quotes (optional setup):

```bash
FINNHUB_API_KEY=YOUR_FINNHUB_API_KEY
```

Local setup and handoff notes can be kept under the ignored `documentation/` tree when needed. Cross-asset platform planning should live in the parent `Quant Engines/` folder instead of this repo.

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

### Heston Research Diagnostics

Black-Scholes remains the default live Greek engine. Heston is optional and writes research-only diagnostics into the signal-evaluation dataset for later tests on hit rate, option efficiency, volatility-expansion detection, and strike selection.

```bash
OQE_HESTON_RESEARCH_ENABLED=false
OQE_HESTON_CALIBRATION_MAX_ROWS=80
OQE_HESTON_CALIBRATION_MIN_ROWS=8
OQE_HESTON_CALIBRATION_ERROR_REJECT=0.35
OQE_HESTON_CALIBRATION_TIMEOUT_SECONDS=2.5
OQE_HESTON_BOUND_GUARD_TOLERANCE_PCT=0.01
OQE_HESTON_BOUND_GUARD_REJECT_COUNT=3
OQE_HESTON_PRICE_GAP_WEAK_PCT=60
OQE_HESTON_PRICE_GAP_REJECT_PCT=100
OQE_HESTON_SHORT_TTE_WEAK_DAYS=1
OQE_HESTON_SHORT_TTE_REJECT_DAYS=0.05
OQE_HESTON_SELECTED_IV_LOW_PCT=3
OQE_HESTON_SELECTED_IV_HIGH_PCT=150
```

Report generation:

```bash
python scripts/ops/backfill_heston_research_features.py --write
python scripts/ops/run_heston_research_report.py
```

The backfill runner reads stored `saved_chain_snapshot_path` rows and appends Heston diagnostics after the fact, so the live engine can keep `OQE_HESTON_RESEARCH_ENABLED=false`. Heston quality guards downgrade or reject diagnostics when calibrated parameters pin to optimizer bounds, selected contracts are too close to expiry, or selected-contract Black-Scholes versus Heston price gaps become extreme. If selected IV is proxy, missing, or outside sanity bounds, the price-gap guard is suppressed and reported separately so calibration quality is not confused with a weak BS comparison input. The report summarizes calibration quality by day, TTE bucket, expiry context, selected-IV quality, direction, provider health, guard-flag counts, Heston parameter stability, correlation screens, and ML feature-importance screens. Generated artifacts are local-only under `research/signal_evaluation/reports/heston_research/`.

### Common Provider Settings

Default provider:

```bash
OQE_DEFAULT_DATA_SOURCE=ICICI   # ICICI | ZERODHA
OQE_ICICI_REFRESH_INTERVAL=8
```

Zerodha:

```bash
ZERODHA_API_KEY=
ZERODHA_API_SECRET=
ZERODHA_ACCESS_TOKEN=
```

Generate/update Zerodha access token in one command (auto-writes `.env`):

1. Open login URL and complete consent:

```text
https://kite.trade/connect/login?api_key=YOUR_API_KEY&v=3
```

1. Either copy the `request_token` from the redirect URL, or keep the full redirect URL.

1. Run:

```bash
.venv/bin/python config/generate_token.py \
  --api-key "YOUR_API_KEY" \
  --api-secret "YOUR_API_SECRET" \
  --request-token "FRESH_REQUEST_TOKEN"
```

Or pass the full callback URL directly:

```bash
.venv/bin/python config/generate_token.py \
  --api-key "YOUR_API_KEY" \
  --api-secret "YOUR_API_SECRET" \
  --redirect-url "http://127.0.0.1:8000/?action=login&type=login&status=success&request_token=FRESH_REQUEST_TOKEN"
```

On success the script prints the access token and updates `ZERODHA_ACCESS_TOKEN` in project `.env` automatically. Request tokens expire quickly, so run this immediately after login.

ICICI Breeze:

```bash
ICICI_BREEZE_API_KEY=
ICICI_BREEZE_SECRET_KEY=
ICICI_BREEZE_SESSION_TOKEN=
```

### Macro / News Settings

```bash
OQE_RUNTIME_ENV=DEV
OQE_ALLOW_LIVE_SECRETS=true   # set only for trusted local use when live broker secrets are present
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
OQE_GIFT_NIFTY_SOURCE=ICICI
OQE_GIFT_NIFTY_ICICI_CANDIDATES="NDX:NIFTY;NDX:GIFTNIFTY;NDX:GIFT NIFTY;NDX:NIFTY:futures;NDX:GIFTNIFTY:futures"
OQE_GIFT_NIFTY_ICICI_CACHE_TTL_SECONDS=60
OQE_GIFT_NIFTY_TICKER=      # optional explicit yfinance fallback; do not use ^NSEI as GIFT data
```

GIFT NIFTY is sourced from ICICI Breeze by default. If the ICICI quote is
unavailable, the engine leaves the GIFT lead neutral unless an explicit
non-proxy fallback ticker is configured.

## Parameter Packs

Named packs currently live under [parameter_packs](config/parameter_packs). Notable packs:

- `baseline_v1`: registry-default pack, intended to preserve current behavior
- `macro_overlay_v1`: stronger macro/global caution candidate
- `overnight_focus_v1`: more conservative overnight selection candidate
- `experimental_v1`: research-only pack for offline experiments
- `candidate_v1`: promotion slot currently carrying the reviewed `composite_signal_score >= 85` candidate
- `caution_conditional_shadow_*`: shadow/research caution variants retained for validation and comparison

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

### Consistency Escalation Policy Tuning (No Code Changes)

The signal engine supports regime-aware consistency escalation through policy config.

Policy module:

- `config/signal_consistency_policy.py`

Parameter-pack keys:

- `signal_engine.consistency.default_trade_escalation_min_severity`
- `signal_engine.consistency.trade_escalation_regime_map`

Severity ordering used by escalation:

- `NONE < LOW < MEDIUM < HIGH < CRITICAL`

Condition grammar for `trade_escalation_regime_map` keys:

- `gamma=...;global_risk=...;vol=...;confirmation=...`
- supported condition keys are `gamma`, `global_risk`, `vol`, `confirmation`
- matching is exact after normalization
- most specific rule wins (more conditions = higher priority)

Example parameter pack override:

```json
{
  "name": "consistency_governance_v1",
  "parent": "baseline_v1",
  "overrides": {
    "signal_engine.consistency.default_trade_escalation_min_severity": "HIGH",
    "signal_engine.consistency.trade_escalation_regime_map": {
      "gamma=NEGATIVE_GAMMA;global_risk=RISK_OFF": "MEDIUM",
      "gamma=NEGATIVE_GAMMA;vol=VOL_EXPANSION": "MEDIUM",
      "vol=NORMAL_VOL;confirmation=CONFIRMED": "HIGH",
      "vol=NORMAL_VOL;confirmation=STRONG_CONFIRMATION": "HIGH"
    }
  }
}
```

Practical guidance:

- Make stressed regimes stricter by lowering threshold (for example `MEDIUM`).
- Keep benign, confirmed regimes less sensitive with a higher threshold (for example `HIGH`).
- Apply changes via packs, run replay/backtests, then promote through governance workflow.

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

Signal-evaluation threshold shadow mode can be run end to end with one command:

```bash
python scripts/ops/run_threshold_shadow_mode.py
```

That command automatically builds threshold governance, the policy experiment
sandbox, shadow signal-retention simulation, promotion-readiness review,
manual promotion package, post-promotion monitor, and adoption reconciliation
artifacts. It is advisory research automation only: it does not change runtime
thresholds, parameter packs, signal generation, or execution behavior.

When the shadow review reaches `PROMOTION_READY`, the same workflow also writes
a manual promotion review package. To rebuild that package or record a human
approve/reject/defer decision explicitly:

```bash
python scripts/ops/run_threshold_promotion_review.py --review-action DEFERRED --reviewer "<name>"
```

The decision is appended to `threshold_promotion_review_ledger.csv`; recording
approval is an audit action only and still does not apply a runtime change.

After an `APPROVED` ledger entry exists, post-promotion monitoring can compare
newer signal outcomes against the shadow evidence that justified approval:

```bash
python scripts/ops/run_threshold_post_promotion_monitor.py
```

The monitor can classify the approved threshold as healthy, watch,
deteriorating, insufficient-data, or skipped-no-approval. It can recommend
manual review, but it does not apply or revert configuration.

To verify whether an approved threshold is actually active in the current
runtime policy view:

```bash
python scripts/ops/run_threshold_adoption_reconciliation.py
```

The reconciliation report can classify the adoption state as approved-but-not-adopted,
adopted manually, mismatched, manually rolled back, or unknown. It
only reports consistency between the ledger, promotion package, active
parameter-pack policy, and post-promotion monitor.

To produce the exact parameter-pack patch needed for manual adoption, run the
advisory adoption helper:

```bash
python scripts/ops/run_threshold_adoption_helper.py
```

The helper writes a reviewable adoption plan and diff, but it does not edit a
parameter pack unless `--apply` is supplied explicitly. A typical controlled
manual adoption sequence is:

```bash
python scripts/ops/run_threshold_adoption_helper.py --target-parameter-pack config/parameter_packs/candidate_v1.json
python scripts/ops/run_threshold_adoption_replay_gate.py --require-ready
python scripts/ops/run_threshold_adoption_helper.py --target-parameter-pack config/parameter_packs/candidate_v1.json --apply
python scripts/ops/run_threshold_adoption_reconciliation.py --parameter-pack config/parameter_packs/candidate_v1.json --require-adopted
OQE_PARAMETER_PACK=candidate_v1 python scripts/ops/run_threshold_adoption_reconciliation.py --require-adopted
python scripts/ops/run_threshold_runtime_activation_marker.py --candidate-pack candidate_v1 --threshold-value 85
python scripts/ops/run_threshold_signal_rollout_monitor.py --fail-on-blocked
python scripts/ops/run_threshold_adoption_history.py
OQE_PARAMETER_PACK=candidate_v1 python scripts/ops/run_threshold_post_activation_verification.py
```

The replay gate applies the proposed override in a temporary runtime context
and checks that only the intended selection-policy key changes, selected signal
sets move in the expected stricter/looser direction, output columns stay stable,
and data-source/provenance fields are preserved. The first reconciliation
validation checks the patched parameter-pack file; the second checks the active
runtime policy after the operator deliberately selects that pack. The runtime
activation marker records the human-controlled start of candidate-pack signal
generation so earlier baseline rows do not pollute the candidate rollout
window. After that marker exists, LIVE signal capture under a different
parameter pack is skipped and flagged rather than persisted to the research
dataset; the guard does not switch packs, edit config, or place trades. The
rollout monitor then emits a daily signal-only status such as
`CANDIDATE_SIGNAL_ROLLOUT_HEALTHY`, `CANDIDATE_SIGNAL_ROLLOUT_WATCH`, or
`CANDIDATE_SIGNAL_ROLLOUT_BLOCKED`, confirming candidate-pack traceability,
resolved threshold value, baseline-versus-candidate selection, provenance
preservation, outcome-label readiness, and absence of order/execution
side-effect fields. The adoption-history command appends that latest state to
`research/signal_evaluation/reports/threshold_adoption_history/threshold_adoption_history.csv`
and refreshes a compact dashboard showing whether the approved threshold is
unadopted, active in candidate-pack signal generation, mixed, mismatched, or
rolled back. The post-activation verifier runs the strict rollout monitor,
appends adoption history, confirms the active runtime pack matches the
activation marker, and exits non-zero unless candidate-pack traceability,
threshold consistency, label readiness, and execution side-effect checks are
all clean. These critical JSON artifacts are checked against lightweight
schema contracts before they are written, so missing fields or type drift fail
fast instead of silently breaking downstream gates. None of these commands
places trades or changes execution behavior.

Before recording a real `APPROVED` ledger decision, the full approval path can
be rehearsed against real artifacts with a sandbox ledger:

```bash
python scripts/ops/run_threshold_promotion_dry_run.py
```

The dry-run creates a sandbox APPROVED decision, runs post-promotion monitoring,
and runs adoption reconciliation without touching the real promotion ledger or
runtime parameter packs.

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

Full suite:

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
- report-generation numeric summaries guard empty/all-NaN slices to avoid silent invalid-statistics warnings
- data integrity tests (`test_live_data_anomaly_detection.py`) validate option chain consistency, IV anomalies, and spot price jumps
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

## Extended Notes

Canonical topic sections are kept earlier in this README. Use these links for the primary operator-facing definitions:

- Predictor architecture: [Pluggable Predictor Architecture](#pluggable-predictor-architecture)
- Model registry: [ML Model Registry and Selection](#ml-model-registry-and-selection)
- Decision policies: [Decision Policy Layer](#decision-policy-layer)

For deeper implementation details:

- Predictor modules: `engine/predictors/`
- Decision-policy research package: `research/decision_policy/`
- ML research/evaluation package: `research/ml_evaluation/`

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
| --- | --- | --- | --- | --- |
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
| --- | --- | --- |
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
