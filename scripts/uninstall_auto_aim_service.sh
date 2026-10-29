#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run with sudo: sudo bash scripts/uninstall_auto_aim_service.sh" >&2
    exit 1
fi

systemctl disable --now simple
rm -f /etc/systemd/system/simple.service
systemctl daemon-reload
systemctl reset-failed simple 2>/dev/null || true
echo "Removed simple.service; kept /etc/default/simple for future installation."
