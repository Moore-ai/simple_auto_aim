#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/project with spaces/build" "$TEST_DIR/project with spaces/configs"
touch "$TEST_DIR/project with spaces/configs/standard.yaml"
cat > "$TEST_DIR/project with spaces/build/standard" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$PWD" "$@" > "$SERVICE_TEST_OUTPUT"
exit 23
EOF
chmod +x "$TEST_DIR/project with spaces/build/standard"
export SERVICE_TEST_OUTPUT="$TEST_DIR/output"

# Catch wrong working directories, missing required arguments, and swallowed exit codes.
set +e
AUTO_AIM_PROJECT_DIR="$TEST_DIR/project with spaces" AUTO_AIM_START_DELAY=0 \
  bash "$PROJECT_DIR/scripts/auto_aim_service.sh"
status=$?
set -e
[[ "$status" == 23 ]] || { echo "service did not propagate standard's exit code" >&2; exit 1; }
printf '%s\n' "$TEST_DIR/project with spaces" configs/standard.yaml --mode=0 \
  --target-color=none > "$TEST_DIR/expected"
diff -u "$TEST_DIR/expected" "$SERVICE_TEST_OUTPUT"

# Custom configuration paths must remain a single argument.
touch "$TEST_DIR/project with spaces/configs/custom config.yaml"
set +e
AUTO_AIM_PROJECT_DIR="$TEST_DIR/project with spaces" AUTO_AIM_START_DELAY=0 \
  AUTO_AIM_CONFIG="configs/custom config.yaml" AUTO_AIM_MODE=2 AUTO_AIM_TARGET_COLOR=blue \
  bash "$PROJECT_DIR/scripts/auto_aim_service.sh"
status=$?
set -e
[[ "$status" == 23 ]]
printf '%s\n' "$TEST_DIR/project with spaces" 'configs/custom config.yaml' --mode=2 \
  --target-color=blue > "$TEST_DIR/expected"
diff -u "$TEST_DIR/expected" "$SERVICE_TEST_OUTPUT"

# Invalid settings must not reach the hardware executable.
rm "$SERVICE_TEST_OUTPUT"
if AUTO_AIM_PROJECT_DIR="$TEST_DIR/project with spaces" AUTO_AIM_START_DELAY=0 \
   AUTO_AIM_MODE=3 bash "$PROJECT_DIR/scripts/auto_aim_service.sh"; then
  echo "invalid service mode was accepted" >&2
  exit 1
fi
[[ ! -e "$SERVICE_TEST_OUTPUT" ]]

# Missing build output must fail with a useful diagnosis before launching hardware.
set +e
output="$(AUTO_AIM_PROJECT_DIR="$TEST_DIR" AUTO_AIM_START_DELAY=0 \
  bash "$PROJECT_DIR/scripts/auto_aim_service.sh" 2>&1)"
status=$?
set -e
[[ "$status" != 0 && "$output" == *"cmake --build"* ]]
echo "auto aim service tests passed"
