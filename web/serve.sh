#!/usr/bin/env bash
# Serve the packaged WebApp locally (required — browsers block WASM from file://).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8080}"
echo "BlendCraft website → http://127.0.0.1:${PORT}/"
echo "Open that URL in Chrome/Edge/Firefox. Ctrl+C to stop."
if command -v python3 >/dev/null 2>&1; then
  cd "$DIR" && python3 -m http.server "$PORT"
elif command -v python >/dev/null 2>&1; then
  cd "$DIR" && python -m http.server "$PORT"
else
  echo "Need python3 to serve static files." >&2
  exit 1
fi
