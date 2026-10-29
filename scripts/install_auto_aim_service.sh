#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_PATH=/etc/systemd/system/simple.service
ENV_PATH=/etc/default/simple

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run with sudo: sudo bash scripts/install_auto_aim_service.sh" >&2
    exit 1
fi
USER_NAME="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"
if [[ -z "$USER_NAME" || "$USER_NAME" == root ]]; then
    echo "Run with sudo from the non-root account that will run the service." >&2
    exit 1
fi
if [[ ! -d /run/systemd/system ]]; then
    echo "systemd is not running. Install this service on the Ubuntu deployment machine." >&2
    exit 1
fi
if [[ ! -x "$PROJECT_DIR/build/standard" ]]; then
    echo "Build first: cmake -S . -B build && cmake --build build --target standard -j2" >&2
    exit 1
fi

if [[ ! -f "$ENV_PATH" ]]; then
    cat > "$ENV_PATH" <<'EOF'
# Change these values, then run: sudo systemctl restart simple
# Relative config paths are resolved from the project root.
AUTO_AIM_CONFIG="configs/standard.yaml"
AUTO_AIM_MODE="0"
AUTO_AIM_TARGET_COLOR="none"
AUTO_AIM_START_DELAY="5"
EOF
    chmod 0644 "$ENV_PATH"
else
    echo "Keeping existing $ENV_PATH"
fi

# WorkingDirectory takes a literal path; ExecStart uses quoted command arguments.
UNIT_PROJECT_DIR="${PROJECT_DIR//%/%%}"
EXEC_PROJECT_DIR="${UNIT_PROJECT_DIR//\\/\\\\}"
EXEC_PROJECT_DIR="${EXEC_PROJECT_DIR//\"/\\\"}"
EXEC_PROJECT_DIR="${EXEC_PROJECT_DIR//\$/\$\$}"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Simple Auto Aim
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$UNIT_PROJECT_DIR
EnvironmentFile=$ENV_PATH
ExecStart=/bin/bash "$EXEC_PROJECT_DIR/scripts/auto_aim_service.sh"
Restart=always
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
chmod 0644 "$UNIT_PATH"
systemctl daemon-reload
systemctl enable simple
systemctl restart simple

echo "Installed and started simple.service as $USER_NAME"
echo "Status: systemctl status simple"
echo "Logs:   journalctl -u simple -f"
