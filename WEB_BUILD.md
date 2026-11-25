# BlendCraft desktop and web builds

BlendCraft has separate target dependency sets and feature selections:

- **Desktop (native):** default `desktop` feature, embedded Python, video export, native file dialogs, and `ufbx` FBX import.
- **WebAssembly:** build with `--no-default-features --features web`; native-only crates are excluded. Browser FBX import uses the built-in parser (not `ufbx`).

## Desktop

```sh
cargo build --release -p forge-app
```

## WebAssembly

Install the target once:

```sh
rustup target add wasm32-unknown-unknown
```

Build:
```sh
cargo build --release -p forge-app --target wasm32-unknown-unknown --no-default-features --features web
```

The wasm build needs a browser integration page/loader configured for the generated wasm-bindgen output and a WebGPU-capable browser. This source package's web target changes have not been compiled in this environment (Rust/Cargo are unavailable), so treat this as a source-level compatibility patch, not a verified successful build.
