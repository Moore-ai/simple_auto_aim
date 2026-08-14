#!/usr/bin/env bash
set -eu

project_root="$(cd "$(dirname "$0")" && pwd)"
venv_python="$project_root/.venv/bin/python"

if [ ! -x "$venv_python" ]; then
  python3 -m venv "$project_root/.venv" || true
fi

if [ ! -x "$venv_python" ]; then
  echo "无法创建 $venv_python" >&2
  exit 1
fi

if ! "$venv_python" -c 'import fastapi, jinja2, uvicorn' >/dev/null 2>&1; then
  if "$venv_python" -m pip --version >/dev/null 2>&1; then
    "$venv_python" -m pip install -r "$project_root/requirements-web.txt"
  else
    site_packages="$("$venv_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    pip3 install --target "$site_packages" -r "$project_root/requirements-web.txt"
  fi
fi

exec "$venv_python" "$project_root/web.py"
