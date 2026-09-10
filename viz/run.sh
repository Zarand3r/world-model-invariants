#!/usr/bin/env bash
# Start the bench. `./viz/run.sh` serves the built UI from the API process on one port;
# `./viz/run.sh dev` runs Vite alongside it with hot reload and a proxy to the API.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8130}"

if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "port $PORT is already in use — set PORT=... and retry" >&2
  exit 1
fi

if [[ "${1:-}" == "dev" ]]; then
  [[ -d viz/web/node_modules ]] || (cd viz/web && npm install)
  uv run python -m uvicorn viz.server.app:app --host 127.0.0.1 --port "$PORT" --reload &
  API=$!
  trap 'kill $API 2>/dev/null || true' EXIT
  cd viz/web && npm run dev
else
  if [[ ! -d viz/web/dist ]]; then
    echo "building the UI (first run only)…"
    (cd viz/web && npm install && npm run build)
  fi
  echo "bench on http://127.0.0.1:$PORT"
  exec uv run python -m uvicorn viz.server.app:app --host 127.0.0.1 --port "$PORT"
fi
