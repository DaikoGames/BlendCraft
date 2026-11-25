#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}/..")" && pwd)"
cd "$ROOT"

TARGET="wasm32-unknown-unknown"
OUT="$ROOT/web/dist"

if ! command -v cargo >/dev/null 2>&1; then echo 'Error: Cargo is required.' >&2; exit 1; fi
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo 'Error: wasm-bindgen CLI is required. Install the version matching Cargo.lock (0.2.130):' >&2
  echo '  cargo install wasm-bindgen-cli --version 0.2.130 --locked' >&2
  exit 1
fi

rustup target add "$TARGET"
# Avoid serving an old WASM binary or stale wasm-bindgen glue after source changes.
rm -rf "$OUT"
cargo clean --target "$TARGET"
cargo build --release -p forge-web --target "$TARGET"
WASM="$ROOT/target/$TARGET/release/forge_web.wasm"
mkdir -p "$OUT"
wasm-bindgen --target web --out-dir "$OUT" "$WASM"
cp "$ROOT/web/index.html" "$ROOT/web/style.css" "$ROOT/web/bootstrap.js" "$OUT/"
echo "Web build ready: $OUT"
echo "Serve it with: python3 -m http.server 8080 --directory web/dist"
