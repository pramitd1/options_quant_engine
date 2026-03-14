# Parameter Tuning Research

This directory stores auditable outputs from the parameter registry and tuning framework.

Expected artifacts:

- `experiment_ledger.jsonl`: append-only experiment results
- `tuning_campaign_ledger.jsonl`: automated group-level tuning campaign outputs
- `promotion_state.json`: current `baseline`, `candidate`, `shadow`, and `live` pack mapping
- `promotion_ledger.jsonl`: promotion, shadow, approval, and rollback audit trail
- `shadow_mode_log.jsonl`: side-by-side live baseline vs shadow comparisons

Validation artifacts are embedded inside each experiment record and include:

- walk-forward split definitions
- split-level out-of-sample metrics
- regime summaries
- robustness metrics
- baseline-vs-candidate comparison summaries when requested

Promotion state supports explicit pack roles:

- `baseline`
- `candidate`
- `shadow`
- `live`

The code for these workflows lives in:

- [experiments.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/experiments.py)
- [campaigns.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/campaigns.py)
- [promotion.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/promotion.py)
- [reporting.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/reporting.py)
- [walk_forward.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/walk_forward.py)
- [regimes.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/regimes.py)
- [validation.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/validation.py)
- [shadow.py](/Users/pramitdutta/Desktop/options_quant_engine/tuning/shadow.py)

Search methods now include:

- bounded random search
- Latin hypercube search for broader continuous parameter groups
- coordinate-descent refinement for threshold-heavy groups
- registry-driven group tuning campaigns that chain coarse exploration with local refinement

These files are generated at runtime and should be treated as local research artifacts rather than committed source files.
