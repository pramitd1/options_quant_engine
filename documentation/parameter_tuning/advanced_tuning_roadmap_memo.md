---
title: "Advanced Parameter Tuning Roadmap"
subtitle: "Signal-Evaluation-First Statistical and Machine Learning Research Plan"
author: "Codex"
date: "2026-03-14"
---

# Executive Summary

This memo defines the recommended roadmap for extending `options_quant_engine`
from governed parameter search into more advanced statistical, machine
learning, and eventually neural optimization methods.

The governing principle is unchanged:

`market data -> signal generation -> signal evaluation dataset -> tuning / validation / promotion`

Actual executed trades remain outside the learning loop. The objective is not
to create an online self-modifying strategy. The objective is to build a
research-to-production calibration program that is statistically disciplined,
regime-aware, auditable, and safe.

The recommended sequence is:

1. strengthen statistical tuning and parameter sensitivity analysis
2. build experiment meta-analysis and surrogate optimization
3. add regime-conditional optimization
4. add machine-learning-based pack recommendation
5. use neural methods only if the experiment ledger becomes sufficiently large

This sequence is intentionally conservative. It maximizes research signal while
containing overfitting and governance risk.

# Architecture Diagram

```text
                               +----------------------+
                               |   Live / Replay Data |
                               +----------+-----------+
                                          |
                                          v
+----------------+              +---------+----------+
| Parameter      |              | Signal Generation  |
| Registry       +------------->+ Engine             |
| Packs          |              | (authoritative)    |
+-------+--------+              +---------+----------+
        |                                 |
        |                                 v
        |                      +----------+-----------+
        |                      | Signal Evaluation    |
        |                      | Dataset              |
        |                      +----------+-----------+
        |                                 |
        |                                 v
        |                      +----------+-----------+
        |                      | Walk-Forward /       |
        |                      | Regime Validation    |
        |                      +----------+-----------+
        |                                 |
        |                                 v
        |                      +----------+-----------+
        |                      | Objective Framework  |
        |                      | Robustness Metrics   |
        |                      +----------+-----------+
        |                                 |
        |                                 v
        |      +--------------------------+---------------------------+
        |      |                                                      |
        v      v                                                      v
+-------+------+---------+        +-----------------------------------+--+
| Statistical Search     |        | Experiment Meta-Dataset / Surrogates |
| Grid / LHS / Random    |        | Effect Models / Bayesian Search      |
+-------+------+---------+        +-------------------+------------------+
        |      |                                      |
        +------+------------------+-------------------+
                               |
                               v
                  +------------+-------------+
                  | Candidate Parameter Pack |
                  +------------+-------------+
                               |
                               v
                  +------------+-------------+
                  | Promotion Governance     |
                  | Shadow Mode / Approval   |
                  +------------+-------------+
                               |
                               v
                  +------------+-------------+
                  | Live Pack / Rollback     |
                  +--------------------------+
```

# Strategic Principles

## Research Boundary

- Tuning must remain based on the signal evaluation dataset, not actual fills.
- Promotion must depend on out-of-sample and regime-aware validation, not one
  favorable period.
- Advanced methods may propose parameter packs, but they do not bypass
  validation, shadow mode, approval, or rollback controls.

## Optimization Philosophy

- Optimize for robust signal quality, not just one-dimensional profit.
- Prefer interpretable methods until there is clear evidence that additional
  model complexity is justified.
- Treat parameter groups as structured families rather than independent knobs
  whenever possible.

# Phased Implementation Plan

| Phase | Objective | Main methods | Promotion to next phase requires |
|---|---|---|---|
| 1 | statistical baseline and sensitivity map | grid, bounded random, Latin hypercube, coordinate descent | stable out-of-sample performance and repeatable sensitivity reports |
| 2 | experiment meta-analysis | regularized regression, GAMs, bootstrap, interaction analysis | clear parameter-effect evidence and tighter search bounds |
| 3 | surrogate optimization | Bayesian optimization, TPE, boosted-tree surrogates, multi-objective search | consistent surrogate usefulness over naive search |
| 4 | regime-conditional tuning | hierarchical models, multi-task optimization, regime-conditioned surrogates | better robustness without fragmenting sample sizes |
| 5 | ML pack recommendation | ranking models, contextual offline policy selection, calibrated classifiers | shadow-mode evidence that recommendations add value |
| 6 | neural meta-models | deep surrogates, tabular neural meta-models, constrained meta-learning | sufficient experiment scale and robust governance controls |

# Methods By Phase

## Phase 1: Statistical Tuning Baseline

### Purpose

Create a strong non-ML baseline so later advanced methods learn from stable and
governed research outputs.

### Scope

- finalize objective definitions
- standardize walk-forward and regime-aware validation configuration
- run parameter-group campaigns rather than indiscriminate whole-space search
- generate sensitivity reports for each major parameter family

### Recommended methods

- bounded grid search for discrete thresholds
- Latin hypercube sampling for medium-dimensional continuous groups
- random search for broad but bounded parameter groups
- coordinate descent for threshold families with interpretable monotonic effects
- ablation studies to test whether a parameter group actually matters

### Deliverables

- baseline benchmark packs
- sensitivity report by parameter group
- interaction map for key parameter families
- freeze / tune / deprioritize classification for each group

## Phase 2: Experiment Meta-Analysis

### Purpose

Model the experiment ledger itself to understand which parameters matter, where
they matter, and how stable they are.

### Recommended methods

- elastic-net or ridge regression on experiment-level outcomes
- generalized additive models for nonlinear but interpretable parameter effects
- bootstrap confidence intervals for effect stability
- interaction screening for coupled parameter groups
- hierarchical models where regime buckets are treated as partial pools

### Deliverables

- parameter effect size report
- uncertainty bands by regime
- parameter instability warnings
- recommended bounds tightening

## Phase 3: Surrogate Optimization

### Purpose

Use learned approximations of the objective surface to propose more promising
candidate packs than naive search alone.

### Recommended methods

- Gaussian-process Bayesian optimization for smaller search spaces
- Tree-structured Parzen Estimator for mixed or larger search spaces
- gradient-boosted surrogate models for flexible response modeling
- constrained multi-objective optimization over:
  - out-of-sample score
  - robustness score
  - signal frequency
  - drawdown proxy
  - regime collapse penalties

### Operating rule

Surrogates may propose packs. They do not approve packs.

### Deliverables

- expected-improvement candidate proposals
- uncertainty-aware candidate ranking
- surrogate-versus-naive-search performance comparison

## Phase 4: Regime-Conditional Tuning

### Purpose

Allow parameter logic to adapt at the research level to persistent regime
differences without turning the engine into an unstable regime-switching system.

### Best candidate groups

- `macro_news`
- `global_risk`
- `gamma_vol_acceleration`
- `dealer_pressure`
- `option_efficiency`

### Recommended methods

- hierarchical Bayesian priors across regime buckets
- regime-conditioned surrogate models
- multi-task optimization across regime slices
- shrinkage toward a global baseline to prevent fragmentation

### Deliverables

- regime-conditional candidate packs
- regime transfer and stability report
- robustness comparison versus one-size-fits-all baseline packs

## Phase 5: Machine Learning Pack Recommendation

### Purpose

Use machine learning to recommend among approved or candidate pack families,
not to mutate the live engine directly.

### Recommended methods

- calibrated classification models for candidate-over-baseline likelihood
- ranking models for pack-family selection
- contextual offline policy selection using regime and diagnostic features
- conservative bandit-style analysis in offline and shadow mode only

### Deployment rule

This layer begins in shadow mode. It does not control live execution.

### Deliverables

- recommendation score
- confidence estimate
- explanatory driver summary
- shadow comparison against static baseline selection

## Phase 6: Neural Meta-Models

### Purpose

Capture higher-order interactions only after the experiment ledger is rich
enough that simpler methods have saturated.

### Candidate methods

- deep surrogate models for large experiment spaces
- tabular neural networks for pack-to-outcome mapping
- neural density models for promising parameter regions
- constrained meta-learning over split and regime tasks

### Preconditions

- sufficiently large experiment ledger
- robust out-of-sample and regime-aware validation
- clear evidence that simpler methods plateau
- maintained auditability through explicit candidate-pack generation

# Recommended File and Module Design

## Phase 1 and 2 Modules

- `tuning/sensitivity.py`
  - parameter-group sensitivity reports
  - ablation studies
  - local response analysis
- `tuning/meta_dataset.py`
  - converts experiment ledger rows into model-ready experiment-level datasets
- `tuning/effects.py`
  - regression, GAM, and uncertainty analysis over experiments

## Phase 3 Modules

- `tuning/surrogates.py`
  - surrogate model interfaces
- `tuning/bayes_search.py`
  - Bayesian optimization orchestration
- `tuning/multi_objective.py`
  - constrained objective composition and Pareto filtering

## Phase 4 Modules

- `tuning/regime_models.py`
  - regime-conditioned response models
- `tuning/regime_search.py`
  - regime-aware campaign execution

## Phase 5 Modules

- `tuning/pack_recommender.py`
  - pack recommendation model
- `tuning/shadow_analysis.py`
  - shadow-mode evaluation of recommendation quality

## Phase 6 Modules

- `tuning/neural_surrogates.py`
  - optional deep surrogate implementations
- `tuning/meta_learning.py`
  - constrained neural meta-learning research hooks

## Shared Design Requirements

- every module must read from existing parameter packs and registries
- all proposals must output explicit candidate packs
- all evaluations must flow through existing validation and promotion systems
- every advanced method must log to the experiment ledger with full metadata

# Risks and Governance Controls

## Core Risks

### Overfitting

- optimization may lock onto one favorable period
- regime-specific tuning may fragment already limited samples
- selection thresholds may be tuned to make the dataset look better rather than
  improve signal quality

### Leakage

- improper walk-forward design can allow future information into tuning
- experiment meta-models can accidentally learn from mixed in-sample and
  out-of-sample labels if ledger hygiene is weak

### Research-to-production drift

- a method that improves research metrics may fail in shadow mode
- a recommender can look strong offline but introduce unstable live behavior

### Governance failure

- an opaque optimizer can produce packs that are hard to interpret or audit
- aggressive automation can outrun human review and rollback controls

## Mandatory Controls

- time-based splits only
- out-of-sample metrics weighted more heavily than in-sample metrics
- regime-aware reporting before any promotion decision
- minimum sample count and signal-frequency constraints
- explicit robustness-score requirement
- shadow-mode probation before candidate can become live
- manual approval checkpoint for any live promotion
- rollback path and predecessor tracking for every live change

## Model-Specific Controls

### For statistical search

- bounded search ranges only
- monotonic sanity checks for threshold families

### For surrogate models

- require uncertainty estimates where possible
- compare against naive-search baselines
- never use surrogate-predicted score as the final decision criterion

### For ML recommendation

- require calibration checks
- deploy only in shadow mode first
- log every recommendation and disagreement with authoritative logic

### For neural methods

- use only after experiment-scale threshold is met
- require explicit candidate-pack generation
- ban direct online parameter mutation

# Evaluation and Decision Framework

## Core Metrics

- out-of-sample objective score
- walk-forward split stability
- regime-aware stability
- signal count and signal-frequency ratio versus baseline
- direction hit rate
- target reachability quality
- tradeability quality
- drawdown proxy
- regime collapse penalty
- robustness score

## Decision Layers

### Candidate generation

Methods may search, rank, or recommend.

### Candidate validation

All candidates must pass:

- walk-forward validation
- regime-aware validation
- robustness review
- sample sufficiency checks

### Candidate deployment

All candidates must pass:

- shadow-mode review
- baseline versus candidate comparison
- explicit approval
- rollback readiness

# Practical Milestone Plan

## Milestone 1

Build:

- `tuning/sensitivity.py`
- `tuning/meta_dataset.py`
- experiment-level sensitivity and effect reports

Expected value:

- fastest increase in research clarity with low governance risk

## Milestone 2

Build:

- `tuning/surrogates.py`
- `tuning/bayes_search.py`
- constrained surrogate-guided search campaigns

Expected value:

- better search efficiency without changing promotion safety

## Milestone 3

Build:

- `tuning/regime_models.py`
- `tuning/regime_search.py`

Expected value:

- clearer understanding of when certain parameter groups should remain global
  versus regime-conditioned

## Milestone 4

Build:

- `tuning/pack_recommender.py`
- shadow evaluation of pack recommendation

Expected value:

- research-grade adaptive selection without live automation risk

## Milestone 5

Research only:

- neural surrogates and constrained meta-learning

Expected value:

- only justified if the experiment ledger becomes large enough

# Recommendations

## What to do next

1. build experiment meta-analysis and sensitivity reporting first
2. add surrogate-guided candidate proposal second
3. defer neural methods until the experiment ledger is materially larger

## What not to do yet

1. do not build online self-tuning
2. do not optimize directly on actual trade fills
3. do not let ML bypass walk-forward, regime-aware validation, shadow mode, or
   approval

# Appendix A: Advanced Methods and Best Use Cases

| Method | Best use | Why |
|---|---|---|
| grid search | small discrete threshold families | maximally interpretable |
| Latin hypercube | bounded medium-dimensional continuous groups | better coverage than naive random search |
| coordinate descent | ordered threshold families | efficient local refinement |
| elastic-net / ridge | experiment effect analysis | stable effect-size estimation |
| GAM | nonlinear but interpretable response modeling | good intermediate step before opaque ML |
| Bayesian optimization | expensive evaluation surfaces | sample-efficient candidate proposal |
| TPE | mixed-type larger search spaces | practical for pack search |
| gradient-boosted surrogates | flexible experiment meta-modeling | strong baseline surrogate |
| hierarchical models | regime-conditional shrinkage | reduces fragmentation risk |
| ranking models | pack recommendation | fits baseline-versus-candidate choice |
| neural surrogates | very large experiment ledgers | captures higher-order interactions |

# Appendix B: Governance Rules

1. No live parameter mutation without explicit candidate-pack creation.
2. No promotion based only on in-sample results.
3. No promotion without robustness and regime review.
4. No promotion without shadow evidence once shadow support exists for that
   workflow.
5. No neural method may directly control live behavior.
