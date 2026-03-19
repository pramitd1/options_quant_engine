#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_FULL="backtest_fullwindow_2016_2026.log"
LOG_REMAIN="backtest_remaining_optimized.log"

ORIG_PID=$(ps -ef | grep 'python comparative_backtest_all_predictors.py' | grep -v grep | awk 'NR==1{print $2}')

if [[ -z "${ORIG_PID:-}" ]]; then
  echo "No active comparative_backtest_all_predictors.py process found; starting optimized remaining run now." | tee -a "$LOG_REMAIN"
  python comparative_backtest_all_predictors.py --methods pure_rule pure_ml research_dual_model research_decision_policy ev_sizing >> "$LOG_REMAIN" 2>&1
  exit 0
fi

echo "Watching PID $ORIG_PID for blended completion..." | tee -a "$LOG_REMAIN"

while kill -0 "$ORIG_PID" 2>/dev/null; do
  if grep -q "Backtest complete:" "$LOG_FULL"; then
    echo "Detected blended completion marker. Stopping original PID $ORIG_PID and switching to optimized remaining methods." | tee -a "$LOG_REMAIN"
    kill "$ORIG_PID" 2>/dev/null || true
    sleep 2
    break
  fi
  sleep 20
done

echo "Starting optimized remaining-method run at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_REMAIN"
python comparative_backtest_all_predictors.py --methods pure_rule pure_ml research_dual_model research_decision_policy ev_sizing >> "$LOG_REMAIN" 2>&1
