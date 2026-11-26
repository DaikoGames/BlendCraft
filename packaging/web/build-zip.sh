#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT"
if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  bash "$ROOT/web/build.sh"
fi
(
  cd "$ROOT/web/dist"
  zip -r "$OUT/BlendCraft-web.zip" .
)
echo "Web ZIP: $OUT/BlendCraft-web.zip"
