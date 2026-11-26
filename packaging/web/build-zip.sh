#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
# Absolute path — after we cd into web/dist, a relative OUT would break.
if [[ "$OUT" != /* ]]; then
  OUT="$(pwd)/$OUT"
fi
mkdir -p "$OUT"
if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  echo "web/dist missing; building…"
  bash "$ROOT/web/build.sh"
fi
if [[ ! -d "$ROOT/web/dist" ]] || [[ -z "$(ls -A "$ROOT/web/dist" 2>/dev/null)" ]]; then
  echo "error: web/dist is empty or missing after build" >&2
  exit 1
fi
ZIP="$OUT/BlendCraft-web.zip"
rm -f "$ZIP"
(
  cd "$ROOT/web/dist"
  zip -r "$ZIP" .
)
echo "Web ZIP: $ZIP ($(wc -c < "$ZIP") bytes)"
