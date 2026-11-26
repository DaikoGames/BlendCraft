#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
ARCH="${ARCH:-x86_64}"
mkdir -p "$OUT/BlendCraft-linux-$ARCH"
BIN=""
for c in \
  "$ROOT/target/${ARCH}-unknown-linux-gnu/release/forge3d" \
  "$ROOT/target/release/forge3d"; do
  [[ -x "$c" ]] && BIN="$c" && break
done
[[ -n "$BIN" ]] || { cargo build --release -p forge-desktop; BIN="$ROOT/target/release/forge3d"; }
cp "$BIN" "$OUT/BlendCraft-linux-$ARCH/forge3d"
cp "$ROOT/packaging/appimage/BlendCraft.desktop" "$OUT/BlendCraft-linux-$ARCH/" 2>/dev/null || true
cp "$ROOT/LICENSE.md" "$OUT/BlendCraft-linux-$ARCH/" 2>/dev/null || true
cp "$ROOT/README.md" "$OUT/BlendCraft-linux-$ARCH/" 2>/dev/null || true
tar -C "$OUT" -czf "$OUT/BlendCraft-linux-$ARCH.tar.gz" "BlendCraft-linux-$ARCH"
echo "Tarball: $OUT/BlendCraft-linux-$ARCH.tar.gz"
