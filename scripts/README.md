# Scripts Organization

This directory is organized by purpose.

## Root-level stable entrypoints

- daily_research_report.py
- signal_evaluation_report.py
- update_signal_outcomes.py
- run_multiyear_backtest.py
- build_model_registry.py
- parameter_governance.py
- historical_data_download.py
- render_pdf.py
- generate_documentation_pdfs.sh

## Subfolders

- backtest/: comparative and monitoring utilities for long-running backtests
- data_prep/: historical/macroeconomic data build and audit utilities
- simulations/: synthetic regime replay tools
- ad_hoc/: one-off diagnostics and analysis scripts
- ops/: operational diagnostics, rollout checks, and resumable offline replay pack suites

Notes:

- The active full-window comparative entrypoint remains at repository root while an existing run is in progress.
- After that run completes, it can be moved into scripts/backtest with a compatibility wrapper if desired.
- For resumable baseline-vs-candidate replay evaluation, use `scripts/ops/run_offline_replay_pack_suite.py` and resume with `--resume-dir`.

PDF generation workflow:

- Use `scripts/generate_documentation_pdfs.sh` for strict all-documentation rendering.
- The run report is written to `documentation/_pdf_assets/render_reports/docs_pdf_render_latest.json`.
