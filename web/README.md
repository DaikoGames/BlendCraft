# BlendCraft Web

This folder contains the browser shell for the WebAssembly build. It is separate from the desktop packaging and does not replace desktop sources or dependencies.


## Build

From the repository root, install the WebAssembly target and matching `wasm-bindgen` CLI:


```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
./web/build.sh
```

The script builds the app for `wasm32-unknown-unknown` without default features (so the native Python-scripting feature is not enabled), generates the wasm-bindgen JavaScript, and copies the browser shell into `web/dist/`.


## Run locally

```bash
python3 -m http.server 8080 --directory web/dist
```

Then open `http://localhost:8080`. Do not open `index.html` directly with `file://`; browsers require a local HTTP server for WebAssembly module loading.


## Notes

- `web/dist` is generated output and is not committed as source.
- Browser APIs and WASM limitations may mean some desktop-only features are unavailable in Web.
- The Web build has no server/backend dependency; it runs locally in the browser.
