#!/usr/bin/env bash
# Stage a FreeBSD binary tarball (run on FreeBSD after cargo build).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT/BlendCraft-freebsd"
cp "$ROOT/target/release/forge3d" "$OUT/BlendCraft-freebsd/"
tar -C "$OUT" -czf "$OUT/BlendCraft-freebsd.tar.gz" BlendCraft-freebsd
echo "BSD tarball: $OUT/BlendCraft-freebsd.tar.gz"
