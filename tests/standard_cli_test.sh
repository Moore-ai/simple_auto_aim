#!/bin/bash
set -euo pipefail

standard_binary="${1:-./build/standard}"
help_output="$("$standard_binary" --help)"
if [[ "$help_output" != *"0=小符，1=大符，2=自瞄普通目标和前哨站"* ]] ||
   [[ "$help_output" != *"--mode (value:2)"* ]]; then
  echo "standard help does not describe mode 2 as the default auto-aim mode" >&2
  exit 1
fi

if "$standard_binary" configs/standard.yaml --mode=3 >/dev/null 2>&1; then
  echo "invalid mode was accepted" >&2
  exit 1
fi
