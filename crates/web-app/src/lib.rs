//! Browser entry point for BlendCraft. This crate only builds for `wasm32`;
//! the native desktop executable lives in `crates/desktop` and has a separate
//! dependency graph.
#![cfg(target_arch = "wasm32")]

use forge_app::App;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::web::EventLoopExtWebSys;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    // In the browser the event loop must not block: `spawn_app` returns immediately
    // and hands control back to the JS event loop (`run_app` would throw).
    event_loop.spawn_app(App::new(None));
}
