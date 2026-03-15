#!/usr/bin/env bash

PDF_TOOLS_ROOT="${PDF_TOOLS_ROOT:-$HOME/.local/oqe-pdf-tools}"

if [ ! -d "$PDF_TOOLS_ROOT" ]; then
  echo "PDF tools environment not found at $PDF_TOOLS_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$PDF_TOOLS_ROOT/bin:$PATH"
export FONTCONFIG_PATH="$PDF_TOOLS_ROOT/etc/fonts"
export FC_CONFIG_DIR="$PDF_TOOLS_ROOT/etc/fonts"
export FC_CONFIG_FILE="$PDF_TOOLS_ROOT/etc/fonts/fonts.conf"
export XDG_CACHE_HOME="/tmp/oqe-pdf-tools-cache"

mkdir -p "$XDG_CACHE_HOME/fontconfig"

echo "Activated PDF toolchain from $PDF_TOOLS_ROOT"
echo "Available: tectonic, pandoc, IBM Plex Sans fonts"
