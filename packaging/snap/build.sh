#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT"
cd "$ROOT/packaging/snap"
snapcraft pack --output "$OUT"
echo "Snap written under $OUT"
