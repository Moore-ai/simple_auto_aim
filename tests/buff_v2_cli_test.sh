#!/bin/bash
set -euo pipefail

buff_binary="${1:-./build/buff}"
"$buff_binary" --help >/dev/null
if "$buff_binary" configs/standard.yaml --mode=2 >/dev/null 2>&1; then
  echo "invalid rune mode was accepted" >&2
  exit 1
fi
