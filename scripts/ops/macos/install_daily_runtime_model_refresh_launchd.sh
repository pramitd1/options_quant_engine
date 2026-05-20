#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
TEMPLATE_PATH="$PROJECT_ROOT/scripts/ops/macos/com.optionsquant.runtime_model_refresh.plist.template"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$TARGET_DIR/com.optionsquant.runtime_model_refresh.plist"
PYTHON_BIN_DEFAULT="${OQE_PYTHON:-$WORKSPACE_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN_DEFAULT" ]]; then
  PYTHON_BIN_DEFAULT="$PROJECT_ROOT/.venv/bin/python"
fi
PYTHON_BIN="${1:-$PYTHON_BIN_DEFAULT}"
FAILURE_WEBHOOK_URL="${RUNTIME_MODEL_REFRESH_FAILURE_WEBHOOK_URL:-}"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary is not executable: $PYTHON_BIN" >&2
  echo "Pass explicit path as first arg, e.g.:" >&2
  echo "  $0 /Users/you/Quant Engines/.venv/bin/python" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
mkdir -p "$PROJECT_ROOT/logs"

python3 - <<'PY' "$TEMPLATE_PATH" "$TARGET_PLIST" "$PROJECT_ROOT" "$PYTHON_BIN" "$FAILURE_WEBHOOK_URL"
import pathlib
import sys

template_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
project_root = sys.argv[3]
python_bin = sys.argv[4]
webhook = sys.argv[5]

text = template_path.read_text(encoding="utf-8")
text = text.replace("__PROJECT_ROOT__", project_root)
text = text.replace("__PYTHON_BIN__", python_bin)
text = text.replace("__FAILURE_WEBHOOK_URL__", webhook)
out_path.write_text(text, encoding="utf-8")
PY

launchctl unload "$TARGET_PLIST" >/dev/null 2>&1 || true
launchctl load "$TARGET_PLIST"

echo "Installed and loaded LaunchAgent: $TARGET_PLIST"
echo "Schedule: daily at 18:05 local time"
echo "Verify: launchctl list | grep com.optionsquant.runtime_model_refresh"
