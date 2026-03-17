#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_MD="$ROOT_DIR/documentation/research_notes/quant_note_trade_signal_logic.md"
STYLE_CSS="$ROOT_DIR/documentation/research_notes/quant_note_trade_signal_logic_style.css"
OUTPUT_DIR="$ROOT_DIR/documentation/research_notes"
OUTPUT_HTML="$OUTPUT_DIR/quant_note_trade_signal_logic_polished.html"
OUTPUT_PDF="$OUTPUT_DIR/quant_note_trade_signal_logic_polished.pdf"

# ---- Try the Python renderer first (multi-backend, most robust) ----
VENV_PYTHON="$ROOT_DIR/.venv/bin/python3"
if [ -x "$VENV_PYTHON" ]; then
  echo "Attempting Python PDF renderer..."
  CONDA_LIB="${CONDA_PREFIX:-}/lib"
  OQE_LIB="${PDF_TOOLS_ROOT:-$HOME/.local/oqe-pdf-tools}/lib"
  DYLD_FALLBACK_LIBRARY_PATH="$CONDA_LIB:$OQE_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-}" \
    "$VENV_PYTHON" "$ROOT_DIR/scripts/render_pdf.py" \
      "$SOURCE_MD" \
      -o "$OUTPUT_PDF" \
      --css "$STYLE_CSS" && {
    echo "Rendered via Python renderer:"
    echo "  PDF : $OUTPUT_PDF"
    exit 0
  }
  echo "Python renderer failed — falling back to shell pipeline."
fi

# ---- Legacy fallback: Pandoc + Chrome headless ----
source "$ROOT_DIR/scripts/activate_pdf_tools.sh" >/dev/null 2>&1 || true
mkdir -p "$OUTPUT_DIR"

PANDOC_BIN="${PDF_TOOLS_ROOT:-$HOME/.local/oqe-pdf-tools}/bin/pandoc"
if ! command -v "$PANDOC_BIN" >/dev/null 2>&1; then
  PANDOC_BIN="$(command -v pandoc 2>/dev/null || true)"
fi
if [ -z "$PANDOC_BIN" ]; then
  echo "Error: pandoc not found." >&2
  exit 1
fi

"$PANDOC_BIN" "$SOURCE_MD" \
  --standalone \
  --from gfm+yaml_metadata_block \
  --to html5 \
  --embed-resources \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --css "$STYLE_CSS" \
  -o "$OUTPUT_HTML"

# Try Chrome, Chromium, or google-chrome in order
CHROME_BIN=""
for candidate in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "$(command -v chromium-browser 2>/dev/null || true)" \
  "$(command -v chromium 2>/dev/null || true)" \
  "$(command -v google-chrome 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    CHROME_BIN="$candidate"
    break
  fi
done

if [ -z "$CHROME_BIN" ]; then
  echo "Warning: Chrome/Chromium not found — HTML generated but PDF skipped." >&2
  echo "  HTML: $OUTPUT_HTML"
  exit 1
fi

"$CHROME_BIN" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$OUTPUT_PDF" \
  "file://$OUTPUT_HTML"

echo "Rendered:"
echo "  HTML: $OUTPUT_HTML"
echo "  PDF : $OUTPUT_PDF"
