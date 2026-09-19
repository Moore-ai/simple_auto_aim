#!/usr/bin/env bash

set -euo pipefail

if cmake --build "$1" --target auto_buff; then
  echo "旧 auto_buff target 仍属于活跃构建图" >&2
  exit 1
fi
