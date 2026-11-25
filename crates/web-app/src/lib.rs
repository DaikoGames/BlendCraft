use forge_app::App;
use winit::event_loop::{ControlFlow, EventLoop};

/// Browser entry point. All browser-only startup code lives in this crate;
/// the native desktop executable has a separate launcher and dependency graph.
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let _ = event_loop.run_app(&mut App::new(None));
}
