#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT"
BIN_SRC=""
for c in \
  "$ROOT/target/x86_64-unknown-linux-gnu/release/forge3d" \
  "$ROOT/target/release/forge3d"; do
  [[ -x "$c" ]] && BIN_SRC="$c" && break
done
if [[ -z "$BIN_SRC" ]]; then
  cargo build --release -p forge-desktop --target x86_64-unknown-linux-gnu
  BIN_SRC="$ROOT/target/x86_64-unknown-linux-gnu/release/forge3d"
fi
# nfpm paths are relative to the config file directory
CFG="$ROOT/packaging/linux/nfpm.yaml"
# Rewrite src to absolute via temp config
TMP=$(mktemp)
sed "s|src: ../../target/x86_64-unknown-linux-gnu/release/forge3d|src: $BIN_SRC|" "$CFG" > "$TMP"
if ! command -v nfpm >/dev/null 2>&1; then
  curl -fsSL https://github.com/goreleaser/nfpm/releases/download/v2.41.1/nfpm_2.41.1_Linux_x86_64.tar.gz \
    | tar -xz -C /tmp nfpm
  NFPM=/tmp/nfpm
else
  NFPM=nfpm
fi
"$NFPM" package -f "$TMP" -p deb -t "$OUT"
"$NFPM" package -f "$TMP" -p rpm -t "$OUT"
rm -f "$TMP"
echo "deb/rpm packages in $OUT"
