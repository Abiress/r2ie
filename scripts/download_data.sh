#!/usr/bin/env bash
# Download the Tiny Shakespeare corpus used by the full R2IE quickstart.
# Tests do NOT use this file (they use the bundled inline fixture); this is
# only for the end-to-end training quickstart experience.
set -euo pipefail

DATA_DIR="${1:-data}"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

mkdir -p "$DATA_DIR"
OUT="$DATA_DIR/tinyshakespeare.txt"

if [ -f "$OUT" ]; then
    echo "[skip] $OUT already exists."
else
    echo "[info] downloading tinyshakespeare -> $OUT"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "$OUT"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$OUT" "$URL"
    else
        echo "[error] neither curl nor wget is available." >&2
        exit 1
    fi
    echo "[done] saved $OUT ($(wc -c < "$OUT") bytes)"
fi
