# Operational Scripts

This folder contains one-off and operational maintenance scripts that are useful
for diagnostics, backfills, and rollout checks.

## Scripts

- `apply_ml_inference_backfill.py`
  - Retroactively populates missing ML scores in cumulative signals.
  - Run: `.venv/bin/python scripts/ops/apply_ml_inference_backfill.py`

- `diagnose_ml_inference_gap.py`
  - End-to-end diagnosis for ML score coverage and inference pipeline health.
  - Run: `.venv/bin/python scripts/ops/diagnose_ml_inference_gap.py`

- `run_phase0_shadow_verdict.py`
  - Executes Phase 0 shadow comparison (`blended` vs `research_rank_gate`) and emits GO/NO-GO verdict using rollout KPI gates.
  - Run: `.venv/bin/python scripts/ops/run_phase0_shadow_verdict.py`
  - Optional date: `.venv/bin/python scripts/ops/run_phase0_shadow_verdict.py --date YYYY-MM-DD`

- `run_offline_replay_pack_suite.py`
  - Runs offline-only baseline vs candidate pack replay comparisons across rolling windows and writes auditable artifacts under `research/parameter_tuning/offline_replay_runs/`.
  - Includes checkpoint/resume support so evaluations can continue later from exactly where they stopped.
  - Each invocation writes a dated sub-run folder under `subruns/` and appends to `run_history.csv` so prior summaries are never overwritten.
  - Run: `.venv/bin/python scripts/ops/run_offline_replay_pack_suite.py`
  - Custom candidates/windows: `.venv/bin/python scripts/ops/run_offline_replay_pack_suite.py --candidates macro_overlay_v1 overnight_focus_v1 --windows all 30 60 90`
  - Resume prior run: `.venv/bin/python scripts/ops/run_offline_replay_pack_suite.py --resume-dir research/parameter_tuning/offline_replay_runs/suite_YYYYMMDD_HHMMSS`
