#!/usr/bin/env bash
# Local website server for BlendCraft (WASM needs http://, correct MIME, and built files).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8080}"

# Prefer a packaged dist folder if the user built into web-dist next to the repo.
CANDIDATES=(
  "$ROOT"
  "$ROOT/../web-dist"
  "$ROOT/../../web-dist"
)

SITE=""
for d in "${CANDIDATES[@]}"; do
  if [[ -f "$d/index.html" && -f "$d/forge_app.js" ]]; then
    SITE="$(cd "$d" && pwd)"
    break
  fi
done

if [[ -z "$SITE" ]]; then
  echo "ERROR: forge_app.js not found next to index.html." >&2
  echo "" >&2
  echo "Build the website package first, e.g.:" >&2
  echo "  rustup target add wasm32-unknown-unknown" >&2
  echo "  cargo build --release -p forge-app --target wasm32-unknown-unknown" >&2
  echo "  cargo install wasm-bindgen-cli --locked" >&2
  echo "  mkdir -p web-dist && wasm-bindgen --target web --out-dir web-dist --out-name forge_app \\" >&2
  echo "    target/wasm32-unknown-unknown/release/forge3d.wasm" >&2
  echo "  cp web/index.html web/serve.sh web-dist/" >&2
  echo "  cp crates/app/logo.png web-dist/ 2>/dev/null || true" >&2
  echo "  cd web-dist && ./serve.sh" >&2
  echo "" >&2
  echo "Looked in:" >&2
  for d in "${CANDIDATES[@]}"; do echo "  - $d" >&2; done
  exit 1
fi

echo "BlendCraft website"
echo "  root: $SITE"
echo "  url:  http://127.0.0.1:${PORT}/"
echo "Open that URL (not file://). Ctrl+C to stop."
echo ""

cd "$SITE"
export PORT
python3 - << 'PY'
import http.server
import socketserver
import os
import mimetypes

PORT = int(os.environ.get("PORT", "8080"))

# Ensure WASM / JS module MIME types (required by browsers).
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **getattr(http.server.SimpleHTTPRequestHandler, "extensions_map", {}),
        ".wasm": "application/wasm",
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".json": "application/json",
        ".css": "text/css",
        ".html": "text/html",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        # Local dev: avoid sticky caches while iterating on wasm builds.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        # Not a CORS issue on same origin; headers help if tools probe cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReuseTCPServer(("127.0.0.1", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
PY
