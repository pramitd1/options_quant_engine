# Parameter Tuning Research

This directory stores auditable outputs from the parameter registry and tuning framework.

It is an artifact store, not the implementation package:

- `tuning/` contains the tuning, validation, reporting, governance, and promotion code
- `research/parameter_tuning/` contains the ledgers, reports, state files, and research-generated candidate packs produced by that code

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

- [experiments.py](../../tuning/experiments.py)
- [campaigns.py](../../tuning/campaigns.py)
- [promotion.py](../../tuning/promotion.py)
- [reporting.py](../../tuning/reporting.py)
- [walk_forward.py](../../tuning/walk_forward.py)
- [regimes.py](../../tuning/regimes.py)
- [validation.py](../../tuning/validation.py)
- [shadow.py](../../tuning/shadow.py)

Search methods now include:

- bounded random search
- Latin hypercube search for broader continuous parameter groups
- coordinate-descent refinement for threshold-heavy groups
- registry-driven group tuning campaigns that chain coarse exploration with local refinement

These files are generated at runtime and should be treated as local research artifacts rather than committed source files.
