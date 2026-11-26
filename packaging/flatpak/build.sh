#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT" "$OUT/flatpak-build" "$OUT/flatpak-repo"
flatpak-builder --force-clean --repo="$OUT/flatpak-repo" "$OUT/flatpak-build" \
  "$ROOT/packaging/flatpak/games.daiko.BlendCraft.json"
flatpak build-bundle "$OUT/flatpak-repo" "$OUT/BlendCraft.flatpak" games.daiko.BlendCraft
echo "Flatpak bundle: $OUT/BlendCraft.flatpak"
