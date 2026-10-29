#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AUTO_AIM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${AUTO_AIM_CONFIG:-configs/standard.yaml}"
MODE="${AUTO_AIM_MODE:-0}"
TARGET_COLOR="${AUTO_AIM_TARGET_COLOR:-none}"
START_DELAY="${AUTO_AIM_START_DELAY:-5}"

cd "$PROJECT_DIR"
if [[ ! -x build/standard ]]; then
    echo "Missing build/standard. Run: cmake -S . -B build && cmake --build build --target standard -j2" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration file not found: $CONFIG" >&2
    exit 1
fi
if [[ ! "$MODE" =~ ^[012]$ || ! "$TARGET_COLOR" =~ ^(red|blue|none)$ ||
      ! "$START_DELAY" =~ ^[0-9]+$ ]]; then
    echo "Invalid service environment: mode=0/1/2, target color=red/blue/none, delay=non-negative integer" >&2
    exit 1
fi

# systemd does not inherit the interactive shell's SDK library paths.
for library_dir in /opt/MVS/lib/64 /opt/MVS/lib/aarch64 /usr/local/runtime/lib/intel64 \
                   /usr/local/runtime/lib/aarch64; do
    if [[ -d "$library_dir" ]]; then
        export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

sleep "$START_DELAY"
echo "[auto_aim] project=$PROJECT_DIR config=$CONFIG mode=$MODE target-color=$TARGET_COLOR"
exec ./build/standard "$CONFIG" "--mode=$MODE" "--target-color=$TARGET_COLOR"
