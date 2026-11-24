#!/usr/bin/env bash
# Build the browser package into web-dist/ (run from repo root or from web/).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
rustup target add wasm32-unknown-unknown
cargo build --release -p forge-app --target wasm32-unknown-unknown
command -v wasm-bindgen >/dev/null || cargo install wasm-bindgen-cli --locked
OUT="$REPO/web-dist"
rm -rf "$OUT"
mkdir -p "$OUT"
wasm-bindgen --target web --out-dir "$OUT" --out-name forge_app \
  target/wasm32-unknown-unknown/release/forge3d.wasm
cp web/index.html "$OUT/"
cp web/serve.sh "$OUT/"
chmod +x "$OUT/serve.sh"
cp crates/app/logo.png "$OUT/" 2>/dev/null || true
echo "Built $OUT"
echo "Run:  cd web-dist && ./serve.sh"
ls -lh "$OUT"
