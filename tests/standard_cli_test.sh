#!/bin/bash
set -euo pipefail

standard_binary="${1:-./build/standard}"
help_output="$("$standard_binary" --help)"
if [[ "$help_output" != *"0=自瞄普通目标和前哨站，1=小符，2=大符"* ]] ||
   [[ "$help_output" != *"--mode (value:0)"* ]]; then
  echo "standard help does not describe mode 0 as the default auto-aim mode" >&2
  exit 1
fi

if [[ "$help_output" != *"--target-color"* || "$help_output" != *"none"* ||
      "$help_output" != *"必填"* ]]; then
  echo "standard help does not describe the required target color and none option" >&2
  exit 1
fi

set +e
output="$("$standard_binary" /nonexistent/standard-cli-test.yaml 2>&1)"
status=$?
set -e
if [[ "$status" != 2 || "$output" != *"target-color"* ]]; then
  echo "missing target color was not rejected before runtime startup" >&2
  exit 1
fi

for color in green RED ''; do
  set +e
  output="$("$standard_binary" configs/standard.yaml --target-color="$color" 2>&1)"
  status=$?
  set -e
  if [[ "$status" != 2 || "$output" != *"target-color"* ]]; then
    echo "invalid target color '$color' was not rejected before runtime startup" >&2
    exit 1
  fi
done

for color in red blue none; do
  set +e
  output="$("$standard_binary" /nonexistent/standard-cli-test.yaml --target-color="$color" 2>&1)"
  status=$?
  set -e
  if [[ "$status" != 1 || "$output" != *"Failed to load file"* ]]; then
    echo "valid target color '$color' did not reach configuration loading" >&2
    exit 1
  fi
done

if "$standard_binary" configs/standard.yaml --mode=3 --target-color=none >/dev/null 2>&1; then
  echo "invalid mode was accepted" >&2
  exit 1
fi
