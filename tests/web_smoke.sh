#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
frame_path=/dev/shm/simple_auto_aim_frame
data_path=/dev/shm/simple_auto_aim_data.json
log_path=/dev/shm/simple_auto_aim_log.json
web_pid=""

for path in "$frame_path" "$data_path" "$log_path"; do
  if [ -e "$path" ]; then
    echo "测试中止：IPC 文件已存在：$path" >&2
    exit 1
  fi
done

cleanup() {
  if [ -n "$web_pid" ]; then
    kill "$web_pid" 2>/dev/null || true
    wait "$web_pid" 2>/dev/null || true
  fi
  rm -f "$frame_path" "$data_path" "$log_path"
}
trap cleanup EXIT INT TERM

"$project_root/.venv/bin/python" - "$frame_path" "$data_path" "$log_path" <<'PY'
import base64
import json
import struct
import sys

frame_path, data_path, log_path = sys.argv[1:]
jpeg = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAA"
    "AAAAAAAAAAAAA/9oACAEDAQE/AX//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AX//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAE"
    "BAAY/Aqf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMB"
    "AT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)
assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
with open(frame_path, "wb") as stream:
    stream.truncate(2 * 1024 * 1024)
    stream.seek(0)
    stream.write(struct.pack("<I", len(jpeg)))
    stream.write(jpeg)
with open(data_path, "w", encoding="utf-8") as stream:
    json.dump({"time": [1.0], "yaw": [0.1], "pitch": [0.2]}, stream)
with open(log_path, "w", encoding="utf-8") as stream:
    json.dump({"frame_id": 1, "mode": "AUTO_AIM"}, stream)
PY

cd "$project_root"
./run_web.sh >/tmp/simple_auto_aim_web_smoke.log 2>&1 &
web_pid=$!

for _ in $(seq 1 50); do
  if curl --fail --silent http://127.0.0.1:9000/data >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

index_headers="$(curl --fail --silent --show-error --dump-header - --output /dev/null http://127.0.0.1:9000/)"
index_body="$(curl --fail --silent --show-error http://127.0.0.1:9000/)"
data_headers="$(curl --fail --silent --show-error --dump-header - --output /dev/null 'http://127.0.0.1:9000/data?max_points=10')"
data_body="$(curl --fail --silent --show-error 'http://127.0.0.1:9000/data?max_points=10')"
log_headers="$(curl --fail --silent --show-error --dump-header - --output /dev/null http://127.0.0.1:9000/log)"
log_body="$(curl --fail --silent --show-error http://127.0.0.1:9000/log)"

set +e
video_body=/tmp/simple_auto_aim_video
video_headers="$(curl --silent --max-time 2 --dump-header - --output "$video_body" http://127.0.0.1:9000/video)"
video_status=$?
set -e
[ "$video_status" -eq 0 ] || [ "$video_status" -eq 28 ]
grep -q -- "--frame" "$video_body"

grep -qi '^content-type: text/html' <<<"$index_headers"
grep -q 'Web 调试器' <<<"$index_body"
grep -qi '^content-type: application/json' <<<"$data_headers"
grep -qi '^cache-control: no-store, max-age=0' <<<"$data_headers"
grep -q '"yaw":\[0.1\]' <<<"$data_body"
grep -qi '^content-type: application/json' <<<"$log_headers"
grep -qi '^cache-control: no-store, max-age=0' <<<"$log_headers"
grep -q '"frame_id":1' <<<"$log_body"
grep -qi '^content-type: multipart/x-mixed-replace; boundary=frame' <<<"$video_headers"
grep -qi '^cache-control: no-store, max-age=0' <<<"$video_headers"
grep -qi '^x-accel-buffering: no' <<<"$video_headers"

echo "Web smoke test passed."
