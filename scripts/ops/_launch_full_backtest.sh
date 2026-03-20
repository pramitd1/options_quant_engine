#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT"
LOG="$ROOT/research/ml_evaluation/run_logs/compare_scoring_modes_full_backtest_20260320.log"
nohup "$ROOT/.venv/bin/python" "$ROOT/scripts/backtest/compare_scoring_modes_full_backtest.py" >> "$LOG" 2>&1 &
echo "Launched PID=$! — logs appending to $LOG"
