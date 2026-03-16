# Dependency Diagrams

The diagrams below are based on the code paths and import relationships observed in the repository.

## 1. High-Level Architecture

```mermaid
flowchart TD
    A[Data Layer<br/>spot_downloader, data_source_router, provider loaders] --> B[Runtime Context Builder<br/>run_engine_snapshot / run_preloaded_engine_snapshot]
    B --> C[Normalization And Validation<br/>provider_normalization, expiry_resolver, option_chain_validation]
    H[Policy Resolution<br/>config.policy_resolver] --> G
    C --> D[Analytics Layer<br/>gamma, liquidity, flow, IV, Greeks]
    D --> E[Market-State Assembly<br/>engine.trading_support.market_state]
    E --> F[Probability Layer<br/>engine.trading_support.probability<br/>models.feature_builder / large_move_probability / ml_move_predictor]
    E --> G[Signal-State Layer<br/>engine.trading_support.signal_state]
    F --> G
    G --> I[Signal Engine<br/>engine.signal_engine]
    J[Macro And News Context<br/>scheduled_event_risk, macro_news_aggregator, news.service] --> I
    K[Risk Overlays<br/>global_risk, gamma_vol, dealer_pressure, option_efficiency] --> I
    I --> L[Strategy / Trade Construction<br/>strike_selector, exit_model, budget_optimizer]
    L --> M[Trade Payload Split<br/>execution_trade + trade_audit + legacy trade]
    M --> N[Runtime Sinks<br/>signal capture, shadow evaluation]
    N --> O[Signal Evaluation Dataset]
    O --> P[Evaluation / Reporting]
    O --> Q[Tuning / Validation / Promotion]
```

This is the architecture the live runtime follows most closely.

## 2. Folder Dependency Diagram

```mermaid
flowchart LR
    main[main.py] --> app[app]
    app --> data[data]
    app --> macro[macro]
    app --> news[news]
    app --> engine[engine]
    app --> research[research]
    app --> tuning[tuning]

    data --> config[config]
    macro --> config
    news --> config
    risk[risk] --> config
    strategy[strategy] --> config
    engine --> config

    engine --> analytics[analytics]
    engine --> models[models]
    engine --> strategy
    engine --> risk

    research --> data
    research --> config

    config --> policy[config.policy_resolver]
    policy --> tuning

    tuning --> research
    tuning --> config
    tuning --> macro

    backtest[backtest] --> app
```

The key architectural wrinkle is now localized: configuration remains dynamically parameter-pack aware, but that dependency is isolated behind `config.policy_resolver` rather than routed through the full tuning runtime/registry stack.

## 3. Data-To-Signal Pipeline

```mermaid
flowchart TD
    A[run_engine_snapshot or run_preloaded_engine_snapshot] --> B[_load_market_inputs when needed]
    A --> C[_prepare_snapshot_context]
    B --> C
    C --> D[spot_snapshot + option_chain]
    D --> E[resolve_selected_expiry / filter_option_chain_by_expiry]
    E --> F[validate_option_chain]
    D --> G[evaluate_scheduled_event_risk]
    D --> H[headline_service.fetch]
    D --> I[build_global_market_snapshot or neutral historical fallback]
    G --> J[build_macro_news_state]
    J --> K[build_global_risk_state]
    F --> L[_evaluate_snapshot_for_pack]
    K --> L
    L --> M[generate_trade]

    M --> N[normalize_option_chain]
    N --> O[_collect_market_state]
    O --> P[_compute_probability_state]
    O --> Q[_compute_signal_state]
    P --> Q
    Q --> R[overlay score accumulation]
    R --> S[evaluate_global_risk_layer]
    S --> T[select_best_strike]
    T --> U[build_option_efficiency_state]
    U --> V[calculate_exit / optimize_lots]
    V --> W[execution_trade + trade_audit + legacy trade]
```

This is the real production signal path as implemented today.

## 4. Research / Tuning / Promotion Workflow

```mermaid
flowchart TD
    A[Live Or Replay Runtime] --> B[DefaultSignalCaptureSink]
    B --> C[save_signal_evaluation]
    C --> D[research.signal_evaluation.dataset]
    D --> E[update_signal_dataset_outcomes]
    E --> F[write_signal_evaluation_report]
    D --> G[run_parameter_experiment]
    G --> H[run_walk_forward_validation]
    H --> I[run_group_tuning_campaign]
    I --> J[run_controlled_tuning_workflow]
    J --> K[candidate pack written]
    J --> L[candidate vs production report]
    K --> M[promotion_state / promotion_ledger]
    M --> N[shadow pack / live pack assignments]
```

The research dataset is the bridge between production signal generation and governed tuning.

## 5. Candidate-Vs-Production Parameter Workflow

```mermaid
flowchart LR
    A[config/* policy defaults] --> B[tuning.registry]
    B --> C[tuning.packs]
    C --> D[config.policy_resolver active context]
    D --> E[config getters resolve live overrides]
    E --> F[app.engine_runner]
    F --> G[engine.signal_engine]

    H[run_controlled_tuning_workflow] --> I[run_group_tuning_campaign]
    I --> J[run_parameter_experiment]
    J --> K[candidate overrides]
    K --> L[materialize_candidate_parameter_pack]
    L --> M[promotion_state candidate]
    M --> N[shadow mode in runtime]
    N --> O[DefaultShadowEvaluationSink]
    O --> P[shadow log and comparison]
    P --> Q[promote_candidate]
    Q --> R[promotion_state live]
    R --> D
```

This diagram shows the closed loop between configuration, runtime parameter activation, tuning, shadow evaluation, and eventual promotion.

## 6. Central-Module Dependency Diagram

```mermaid
flowchart TD
    A[app.engine_runner] --> B[data.*]
    A --> C[macro.*]
    A --> D[news.service]
    A --> E[risk.build_global_risk_state]
    A --> F[engine.signal_engine]
    A --> G[app.runtime_sinks]
    A --> H[tuning.runtime]
    A --> I[tuning.promotion]

    F --> J[engine.trading_support.market_state]
    F --> K[engine.trading_support.probability]
    F --> L[engine.trading_support.signal_state]
    F --> M[risk.global_risk_layer]
    F --> N[risk.gamma_vol_acceleration_layer]
    F --> O[risk.dealer_hedging_pressure_layer]
    F --> P[risk.option_efficiency_layer]
    F --> Q[strategy.strike_selector]
    F --> R[strategy.exit_model]
    F --> S[strategy.budget_optimizer]
```

This is the smallest useful picture of the runtime center of gravity in the repository.

## Diagram Notes

- The production core is centered on `app.engine_runner` and `engine.signal_engine`.
- The computational core is cleaner than the operational boundary around it.
- The tuning/configuration loop is still the highest-coupling boundary, but the production policy-resolution path is cleaner because it now terminates at `config.policy_resolver`.
