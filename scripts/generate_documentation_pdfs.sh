#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="./.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

REPORT_DIR="documentation/_pdf_assets/render_reports"
REPORT_PATH="$REPORT_DIR/docs_pdf_render_latest.json"

mkdir -p "$REPORT_DIR"

"$PYTHON_BIN" scripts/render_pdf.py \
  --all-docs \
  --strict \
  --report-json "$REPORT_PATH"

echo "PDF generation complete. Report: $REPORT_PATH"
