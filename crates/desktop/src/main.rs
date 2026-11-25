use forge_app::{load_cached_fonts, App};
use std::path::PathBuf;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    load_cached_fonts();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let initial = std::env::args().nth(1).map(PathBuf::from);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(initial)).expect("run");
}
