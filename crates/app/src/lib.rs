mod fonts_io;

/// Load locally cached and system fonts on native platforms. No network access is used.
pub fn load_cached_fonts() {
    fonts_io::load_cached_fonts();
}

mod gizmo;
mod keymap;
mod navgizmo;
mod overlay;
mod recent;
mod scripting;
mod tools;
mod transform;
mod ui;

use egui_wgpu::wgpu;
use forge_core::{
    project, Constraint, Document, EditDomain, EnvironmentMap, FixedView, ObjectKind, OrbitCamera,
    SameKind, SculptSettings,
};
use forge_render::{Highlight, RenderEngine, Renderer, ViewSettings};
use glam::{Mat4, Vec2};
use keymap::{Action, Binding};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tools::{anchors, apply_selection, point_in_polygon, SelShape, SelectOp, Tool, ToolState};
use transform::{Modal, ModalKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditXformKind {
    Move,
    Rotate,
    Scale,
}

/// Active mouse drag for extruding a new bone from a selected bone tip.
struct BoneExtrudeDrag {
    bone_idx: u32,
    /// Fixed head in armature local space
    head: glam::Vec3,
}

/// Snapshot-based transform for selected armature bones. The same tool works
/// on rest geometry in Armature Edit Mode and pose channels in Pose Mode.
struct BoneXform {
    kind: EditXformKind,
    start_cursor: (f64, f64),
    bones: Vec<(u32, [f32; 3], [f32; 3], [f32; 4], [f32; 3], [f32; 3])>,
    /// Original rest endpoints for every bone, used to preserve connected joints.
    rest_joints: Vec<(glam::Vec3, glam::Vec3)>,
    center: glam::Vec3,
    inv_world: glam::Mat4,
}

struct EditGizmoDrag {
    axis: usize,
    /// World-space axis direction
    axis_dir: glam::Vec3,
    origin: glam::Vec3,
    start_param: f32,
    verts: Vec<(u32, glam::Vec3)>,
    inv_world: glam::Mat4,
    /// Fixed (selected, _, signs) pairs baked at drag start
    mirror_pairs: Vec<(u32, u32, glam::Vec3)>,
    /// Full mesh positions at drag start (for split-vert symmetry)
    mesh_snap: Vec<[f32; 3]>,
}

/// Modal transform for selected mesh components in Edit Mode (G / R / S).
struct EditXform {
    kind: EditXformKind,
    start_cursor: (f64, f64),
    verts: Vec<(u32, glam::Vec3)>,
    center: glam::Vec3,
    inv_world: glam::Mat4,
    /// true = drag while holding LMB (click-drag verts); false = modal G/R/S
    confirm_on_release: bool,
    /// Fixed (selected, _, signs) pairs baked at transform start
    mirror_pairs: Vec<(u32, u32, glam::Vec3)>,
    /// Full mesh positions at transform start (for split-vert symmetry)
    mesh_snap: Vec<[f32; 3]>,
}

use ui::{Rename, RenderProgressUi, UiState};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

#[derive(serde::Serialize, serde::Deserialize)]
struct UvMapFile {
    format: String,
    version: u32,
    object_name: String,
    vertex_count: usize,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    uvs: Vec<[f32; 2]>,
}

struct Gfx {
    window: Arc<Window>,
    renderer: Renderer,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

/// A box / lasso / circle selection drag in progress.
struct SelDrag {
    tool: Tool,
    start: (f64, f64),
    points: Vec<Vec2>,
    op: SelectOp,
}

/// Interactive Edit Mode tool drag (Ctrl+LMB): amount follows mouse.
struct ToolDrag {
    tool: Tool,
    start: (f64, f64),
    /// Last applied parameter (so we can re-apply from checkpoint)
    last_amount: f32,
}

pub struct App {
    gfx: Option<Gfx>,
    /// WebAssembly only: GPU init is asynchronous in the browser, so the renderer
    /// is delivered through this slot and picked up on the next window event.
    #[cfg(target_arch = "wasm32")]
    pending_gpu: std::rc::Rc<std::cell::RefCell<Option<anyhow::Result<Renderer>>>>,
    #[cfg(target_arch = "wasm32")]
    web_window: Option<Arc<Window>>,
    #[cfg(target_arch = "wasm32")]
    gpu_init_started: bool,
    egui_ctx: egui::Context,
    doc: Document,
    camera: OrbitCamera,
    settings: ViewSettings,
    ts: ToolState,
    gizmo: gizmo::Gizmo,
    modal: Option<Modal>,
    edit_xform: Option<EditXform>,
    /// Hovered axis of the Edit Mode move gizmo (0=X,1=Y,2=Z)
    edit_gizmo_hover: Option<usize>,
    edit_gizmo_drag: Option<EditGizmoDrag>,
    bone_xform_drag: Option<BoneXform>,
    bone_extrude_drag: Option<BoneExtrudeDrag>,
    sel_drag: Option<SelDrag>,
    /// Completed measurements remain visible until explicitly cleared.
    measurements: Vec<(glam::Vec3, glam::Vec3)>,
    /// Start and live endpoint for the measurement currently being drawn.
    measure_a: Option<glam::Vec3>,
    measure_b: Option<glam::Vec3>,
    /// Annotate tool: strokes as lists of world-space points
    annotations: Vec<Vec<glam::Vec3>>,
    /// Stroke currently being drawn; only active after an explicit LMB press.
    annotate_stroke: Option<Vec<glam::Vec3>>,
    annotate_drag_active: bool,
    /// Ctrl+LMB interactive edit tool
    tool_drag: Option<ToolDrag>,
    sculpt: SculptSettings,
    /// True while LMB is held painting in Sculpt Mode
    sculpt_stroke: bool,
    /// Last world hit for grab delta
    sculpt_last_hit: Option<glam::Vec3>,
    /// Cursor pos of last sculpt sample (throttle density)
    sculpt_last_sample: Option<(f64, f64)>,
    /// True while LMB is held for a texture-paint stroke.
    paint_stroke: bool,
    paint_last_cursor: Option<(f64, f64)>,
    ui: UiState,
    bindings: Vec<Binding>,
    recent: recent::Recent,
    /// Some(object id) while looking through that camera
    camera_view: Option<u64>,
    cursor: (f64, f64),
    press: (f64, f64),
    right_press: (f64, f64),
    left: bool,
    right: bool,
    middle: bool,
    dragged: bool,
    right_dragged: bool,
    mods: ModifiersState,
    /// 3D viewport in physical pixels: x, y, w, h
    viewport: [f32; 4],
    /// Last non-zero 3D panel size (physical px). Used for F12 when `viewport`
    /// is temporarily zeroed (Rendering workspace / progress overlay).
    last_3d_viewport: [f32; 4],
    /// (out_w, out_h, pixel size, camera view) the Cycles buffer was last sized for.
    cycles_res_key: (u32, u32, u32, bool),
    uploaded: u64,
    last_anim_tick: web_time::Instant,
    /// Fractional part (0..1) of the current frame while playing, for smooth motion.
    anim_subframe: f32,
    /// Object transforms changed by playback only; GPU matrices need a refresh.
    anim_gpu_dirty: bool,
    /// Transient Rapier world, rebuilt from the saved scene when playback starts.
    physics_runtime: Option<forge_core::PhysicsRuntime>,
    physics_runtime_doc_version: u64,
    /// Running eased transition of the orbit camera to an axis view.
    view_anim: Option<ViewAnim>,
    title: String,
    initial: Option<PathBuf>,
    /// Cycle through overlapping objects under the cursor (Alt-click or re-click).
    pick_cycle_hits: Vec<usize>,
    pick_cycle_index: usize,
    pick_cycle_cursor: (i32, i32),
}

fn highlights(doc: &Document) -> Vec<Highlight> {
    doc.scene
        .objects
        .iter()
        .map(|o| {
            if doc.selection.contains(&o.id) {
                if doc.active == Some(o.id) {
                    Highlight::Active
                } else {
                    Highlight::Selected
                }
            } else {
                Highlight::None
            }
        })
        .collect()
}

/// Smooth camera move to an axis-aligned view (yaw, pitch pairs).
struct ViewAnim {
    from: (f32, f32),
    to: (f32, f32),
    start: web_time::Instant,
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn err_string<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn edit_sel_world_center(doc: &Document) -> Option<(glam::Vec3, glam::Mat4)> {
    let session = doc.edit.as_ref()?;
    if session.selected_vert_indices().is_empty() {
        return None;
    }
    let id = session.object_id;
    let idx = doc.scene.objects.iter().position(|o| o.id == id)?;
    let world = doc.scene.world_matrices()[idx];
    let local = session.selection_center();
    Some((world.transform_point3(local), world))
}

fn draw_edit_gizmo_ui(
    ctx: &egui::Context,
    rect: egui::Rect,
    camera: &OrbitCamera,
    viewport: [f32; 4],
    doc: &Document,
    hover: Option<usize>,
    drag_axis: Option<usize>,
) {
    let Some((origin, _)) = edit_sel_world_center(doc) else {
        return;
    };
    let dist = (origin - camera.target).length().max(0.5);
    let len = (dist * 0.15).clamp(0.3, 8.0);
    let ppp = ctx.pixels_per_point();
    let vp = [
        rect.min.x * ppp,
        rect.min.y * ppp,
        rect.width() * ppp,
        rect.height() * ppp,
    ];
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("edit_gizmo"),
        ))
        .with_clip_rect(rect);
    let pt = |p: glam::Vec2| egui::pos2(p.x / ppp, p.y / ppp);
    use crate::tools::project_px;
    let colors = [
        egui::Color32::from_rgb(230, 60, 60),
        egui::Color32::from_rgb(80, 200, 80),
        egui::Color32::from_rgb(70, 120, 240),
    ];
    let active = drag_axis.or(hover);
    let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
    for (i, dir) in axes.iter().enumerate() {
        let on = active == Some(i);
        let col = if on {
            egui::Color32::from_rgb(255, 220, 60)
        } else {
            colors[i]
        };
        let (Some(p0), Some(p1)) = (
            project_px(camera, vp, origin),
            project_px(camera, vp, origin + *dir * len),
        ) else {
            continue;
        };
        let stroke = egui::Stroke::new(if on { 3.5 } else { 2.5_f32 }, col);
        painter.line_segment([pt(p0), pt(p1)], stroke);
        let d2 = (p1 - p0).normalize_or_zero();
        let n = glam::Vec2::new(-d2.y, d2.x);
        let base = p1 - d2 * 14.0 * ppp;
        painter.add(egui::Shape::convex_polygon(
            vec![
                pt(p1 + d2 * 6.0 * ppp),
                pt(base + n * 6.0 * ppp),
                pt(base - n * 6.0 * ppp),
            ],
            col,
            egui::Stroke::NONE,
        ));
    }
    if let Some(pc) = project_px(camera, vp, origin) {
        painter.circle_filled(pt(pc), 5.0, egui::Color32::WHITE);
    }
    let _ = viewport;
}

fn save_render_image(path: &std::path::Path, rgba: &[u8], width: u32, height: u32, format: u8) -> Result<(), String> {
    let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "Render dimensions overflow".to_string())?;
    if rgba.len() < expected { return Err("Render pixel buffer is incomplete".into()); }
    match format {
        1 => {
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for px in rgba[..expected].chunks_exact(4) { rgb.extend_from_slice(&px[..3]); }
            image::save_buffer_with_format(path, &rgb, width, height, image::ColorType::Rgb8, image::ImageFormat::Jpeg)
                .map_err(|e| e.to_string())
        }
        2 => {
            let pixels: Vec<f32> = rgba[..expected].chunks_exact(4)
                .flat_map(|px| px.iter().map(|v| *v as f32 / 255.0))
                .collect();
            let image = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::from_raw(width, height, pixels)
                .ok_or_else(|| "Could not construct OpenEXR image buffer".to_string())?;
            image::DynamicImage::ImageRgba32F(image).save_with_format(path, image::ImageFormat::OpenExr)
                .map_err(|e| e.to_string())
        }
        _ => image::save_buffer_with_format(path, &rgba[..expected], width, height, image::ColorType::Rgba8, image::ImageFormat::Png)
            .map_err(|e| e.to_string()),
    }
}

impl App {
    pub fn new(initial: Option<PathBuf>) -> Self {
        Self {
            gfx: None,
            #[cfg(target_arch = "wasm32")]
            pending_gpu: Default::default(),
            #[cfg(target_arch = "wasm32")]
            web_window: None,
            #[cfg(target_arch = "wasm32")]
            gpu_init_started: false,
            egui_ctx: egui::Context::default(),
            doc: Document::new(),
            camera: OrbitCamera::default(),
            settings: ViewSettings::default(),
            ts: ToolState::new(),
            gizmo: gizmo::Gizmo::new(),
            modal: None,
            edit_xform: None,
            edit_gizmo_hover: None,
            edit_gizmo_drag: None,
            bone_xform_drag: None,
            bone_extrude_drag: None,
            sel_drag: None,
            measurements: Vec::new(),
            measure_a: None,
            measure_b: None,
            annotations: Vec::new(),
            annotate_stroke: None,
            annotate_drag_active: false,
            tool_drag: None,
            sculpt: SculptSettings::default(),
            sculpt_stroke: false,
            sculpt_last_hit: None,
            sculpt_last_sample: None,
            paint_stroke: false,
            paint_last_cursor: None,
            ui: {
                let mut ui = UiState::default();
                ui.render_width = 1920;
                ui.render_height = 1080;
                ui.render_percent = 100;
                // One resolution (Output X × Y × %) drives the 3D view, Render Image and Animation.
                ui.render_use_viewport = false;
                // Panel defaults used to be filled in lazily only when the Render panel was drawn,
                // so the Cycles view ran with 1 sample until it was opened. Set them up front.
                ui.render_samples_viewport = 64;
                ui.render_samples = 128;
                ui.cycles_pixel_size = 1;
                ui.cycles_max_dim = 4096;
                ui.render_anim_frames = 36;
                ui.render_fps = 24;
                ui
            },
            bindings: keymap::bindings(),
            recent: recent::Recent::load(),
            camera_view: None,
            cursor: (0.0, 0.0),
            press: (0.0, 0.0),
            right_press: (0.0, 0.0),
            left: false,
            right: false,
            middle: false,
            dragged: false,
            right_dragged: false,
            mods: ModifiersState::empty(),
            viewport: [0.0, 0.0, 1280.0, 720.0],
            last_3d_viewport: [0.0, 0.0, 1280.0, 720.0],
            cycles_res_key: (0, 0, 0, false),
            uploaded: u64::MAX,
            last_anim_tick: web_time::Instant::now(),
            anim_subframe: 0.0,
            anim_gpu_dirty: false,
            physics_runtime: None,
            physics_runtime_doc_version: 0,
            view_anim: None,
            title: String::new(),
            initial,
            pick_cycle_hits: Vec::new(),
            pick_cycle_index: 0,
            pick_cycle_cursor: (i32::MIN, i32::MIN),
        }
    }

    // ------------------------------------------------------------------ helpers
    fn ctrl(&self) -> bool {
        self.mods.control_key() || self.mods.super_key()
    }

    fn in_viewport(&self, p: (f64, f64)) -> bool {
        let [x, y, w, h] = self.viewport;
        let (px, py) = (p.0 as f32, p.1 as f32);
        px >= x && py >= y && px < x + w && py < y + h
    }

    fn frame(&mut self, selected_only: bool) {
        let b = if selected_only {
            self.doc.selected_bounds()
        } else {
            None
        };
        if let Some(b) = b.or_else(|| self.doc.scene.bounds()) {
            self.camera.frame(b);
        }
    }

    /// Raycast against ground plane y=0 or return point along ray at distance 5.

    /// Tools that support Ctrl+LMB interactive drag in Edit Mode.
    fn is_ctrl_lmb_tool(tool: Tool) -> bool {
        matches!(
            tool,
            Tool::ExtrudeRegion
                | Tool::InsetFaces
                | Tool::Bevel
                | Tool::LoopCut
                | Tool::Knife
                | Tool::PolyBuild
                | Tool::Spin
                | Tool::Smooth
                | Tool::EdgeSlide
                | Tool::ShrinkFatten
                | Tool::Shear
                | Tool::RipRegion
        )
    }

    fn begin_tool_drag(&mut self) {
        if !self.doc.in_edit_mode() {
            self.status("Tab into Edit Mode first");
            return;
        }
        let empty = self
            .doc
            .edit
            .as_ref()
            .map(|e| e.selection.is_empty())
            .unwrap_or(true);
        // LoopCut/Knife can work with empty selection differently; others need selection
        let needs_sel = !matches!(self.ts.tool, Tool::Knife | Tool::LoopCut);
        if empty && needs_sel {
            self.status(format!(
                "{}: select geometry first (1/2/3 + click)",
                self.ts.tool.name()
            ));
            return;
        }
        // Checkpoint once at start; we'll undo+reapply each frame for live feedback
        self.doc.checkpoint();
        self.tool_drag = Some(ToolDrag {
            tool: self.ts.tool,
            start: self.cursor,
            last_amount: 0.0,
        });
        self.status(format!(
            "{}: drag to adjust, release to confirm",
            self.ts.tool.name()
        ));
    }

    fn update_tool_drag(&mut self) {
        let Some(drag) = self.tool_drag.as_ref() else {
            return;
        };
        let tool = drag.tool;
        let start = drag.start;
        let dy = (start.1 - self.cursor.1) as f32; // mouse up = positive
        let dx = (self.cursor.0 - start.0) as f32;
        let amount = match tool {
            Tool::ExtrudeRegion => dy * 0.005,
            Tool::InsetFaces => (dy.abs() * 0.002).clamp(0.01, 0.49),
            Tool::Bevel => (dy.abs() * 0.002).max(0.0),
            Tool::ShrinkFatten => dy * 0.004,
            Tool::Smooth => (dy.abs() * 0.002).clamp(0.0, 1.0),
            Tool::EdgeSlide => dx * 0.004,
            Tool::Shear => dx * 0.004,
            Tool::Spin => dy.abs() * 0.8,
            _ => dy * 0.005,
        };
        // Restore pre-drag snapshot, re-save it, then apply without extra checkpoints
        self.doc.undo();
        self.doc.checkpoint();
        // Apply directly on the edit session (avoid edit_* wrappers that checkpoint again)
        if let Some(e) = self.doc.edit.as_mut() {
            match tool {
                Tool::ExtrudeRegion => {
                    e.extrude_faces(amount);
                    self.status(format!("Extrude: {:.3}  (release to confirm)", amount));
                }
                Tool::InsetFaces => {
                    e.inset_faces(amount);
                    self.status(format!("Inset: {:.3}  (release to confirm)", amount));
                }
                Tool::Bevel => {
                    e.bevel(amount);
                    self.status(format!("Bevel: {:.3}  (release to confirm)", amount));
                }
                Tool::ShrinkFatten => {
                    e.shrink_fatten(amount);
                    self.status(format!(
                        "Shrink/Fatten: {:.3}  (release to confirm)",
                        amount
                    ));
                }
                Tool::Smooth => {
                    e.smooth_verts(amount);
                    self.status(format!("Smooth: {:.2}  (release to confirm)", amount));
                }
                Tool::EdgeSlide => {
                    e.edge_slide(amount);
                    self.status(format!("Edge Slide: {:.3}  (release to confirm)", amount));
                }
                Tool::Shear => {
                    e.shear(0, amount);
                    self.status(format!("Shear: {:.3}  (release to confirm)", amount));
                }
                Tool::LoopCut => {
                    e.loop_cut();
                    self.status("Loop Cut");
                    self.doc.sync_edit_mesh();
                    self.tool_drag = None;
                    return;
                }
                Tool::PolyBuild => {
                    e.poly_build(amount.abs().max(0.05));
                    self.status(format!("Poly Build: {:.3}", amount.abs()));
                    self.doc.sync_edit_mesh();
                    self.tool_drag = None;
                    return;
                }
                Tool::Spin => {
                    let ang = amount.clamp(15.0, 360.0);
                    e.spin(8, ang);
                    self.status(format!("Spin: {:.0}°  (release to confirm)", ang));
                }
                Tool::RipRegion => {
                    e.rip_region();
                    self.status("Rip Region");
                    self.doc.sync_edit_mesh();
                    self.tool_drag = None;
                    return;
                }
                Tool::Knife => {
                    self.doc.sync_edit_mesh();
                    self.apply_knife_now();
                    self.tool_drag = None;
                    return;
                }
                _ => {}
            }
            self.doc.sync_edit_mesh();
        }
        if let Some(d) = self.tool_drag.as_mut() {
            d.last_amount = amount;
        }
    }

    fn end_tool_drag(&mut self) {
        if self.tool_drag.take().is_some() {
            self.status("Tool confirmed");
        }
    }

    fn apply_knife_now(&mut self) {
        if !self.doc.in_edit_mode() {
            return;
        }
        let center = self
            .doc
            .edit
            .as_ref()
            .map(|e| e.selection_center())
            .unwrap_or(glam::Vec3::ZERO);
        let world_center = if let Some(session) = self.doc.edit.as_ref() {
            if let Some(idx) = self
                .doc
                .scene
                .objects
                .iter()
                .position(|o| o.id == session.object_id)
            {
                let w = self.doc.scene.world_matrices()[idx];
                w.transform_point3(center)
            } else {
                center
            }
        } else {
            center
        };
        let normal = (self.camera.target - self.camera.eye()).normalize_or_zero();
        if let Some(session) = self.doc.edit.as_ref() {
            if let Some(idx) = self
                .doc
                .scene
                .objects
                .iter()
                .position(|o| o.id == session.object_id)
            {
                let inv = self.doc.scene.world_matrices()[idx].inverse();
                let local_pt = inv.transform_point3(world_center);
                let local_n = inv.transform_vector3(normal).normalize_or_zero();
                self.doc.edit_knife_bisect(local_pt, local_n);
                self.status("Knife bisect");
            }
        }
    }

    fn sync_sculpt_brush_from_tool(&mut self) {
        use forge_core::SculptBrush as B;
        self.sculpt.brush = match self.ts.tool {
            Tool::SculptDrawSharp => B::DrawSharp,
            Tool::SculptClay => B::Clay,
            Tool::SculptClayStrips => B::ClayStrips,
            Tool::SculptLayer => B::Layer,
            Tool::SculptInflate => B::Inflate,
            Tool::SculptBlob => B::Blob,
            Tool::SculptCrease => B::Crease,
            Tool::SculptSmooth => B::Smooth,
            Tool::SculptFlatten => B::Flatten,
            Tool::SculptFill => B::Fill,
            Tool::SculptScrape => B::Scrape,
            Tool::SculptPinch => B::Pinch,
            Tool::SculptGrab => B::Grab,
            Tool::SculptSnakeHook => B::SnakeHook,
            Tool::SculptNudge => B::Nudge,
            Tool::SculptThumb => B::Thumb,
            _ => B::Draw,
        };
    }

    fn paint_sample(&mut self) {
        if let Some(uv) = self.paint_sample_at(self.cursor) {
            match self.ts.tool {
                Tool::PaintErase => {
                    let _ = self.doc.erase_stroke_uv(&[uv], self.ts.paint_radius, self.ts.paint_strength);
                }
                Tool::PaintSoften => {
                    let _ = self.doc.soften_stroke_uv(&[uv], self.ts.paint_radius, self.ts.paint_strength);
                }
                Tool::PaintSmear => {
                    let _ = self.doc.smear_stroke_uv(&[uv], self.ts.paint_radius, self.ts.paint_strength);
                }
                Tool::PaintFill => {
                    let _ = self.doc.fill_uv(uv, self.ts.paint_color, 24);
                }
                _ => {
                    let _ = self.doc.paint_at_uv(uv, self.ts.paint_color, self.ts.paint_radius, self.ts.paint_strength);
                }
            }
        }
    }

    fn vertex_paint_settings(&self) -> forge_core::VertexPaintSettings {
        use forge_core::VertexPaintBrush as B;
        let brush = match self.ts.tool {
            Tool::VertexPaintSoften => B::Soften,
            Tool::VertexPaintBlur => B::Blur,
            Tool::VertexPaintSmear => B::Smear,
            _ => B::Draw,
        };
        forge_core::VertexPaintSettings {
            brush,
            color: self.ts.vpaint_color,
            radius: self.ts.vpaint_radius,
            strength: self.ts.vpaint_strength,
        }
    }

    fn weight_paint_settings(&self) -> forge_core::WeightPaintSettings {
        use forge_core::WeightPaintBrush as B;
        let brush = match self.ts.tool {
            Tool::WeightPaintBlur => B::Blur,
            Tool::WeightPaintAverage => B::Average,
            Tool::WeightPaintSmear => B::Smear,
            _ => B::Draw,
        };
        forge_core::WeightPaintSettings {
            brush,
            weight: self.ts.wpaint_weight,
            radius: self.ts.wpaint_radius,
            strength: self.ts.wpaint_strength,
            normalize: self.ts.wpaint_normalize,
        }
    }

    fn ray_from_cursor(&self, cursor: (f64, f64)) -> Option<(glam::Vec3, glam::Vec3)> {
        let (vx, vy, vw, vh) = (
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        if vw < 1.0 || vh < 1.0 {
            return None;
        }
        let nx = ((cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((cursor.1 as f32 - vy) / vh) * 2.0;
        Some(self.camera.ray((nx, ny), vw / vh))
    }

    /// Ray-hit active mesh and return object-local position for vertex/weight paint.
    fn paint_local_hit(&self, cursor: (f64, f64)) -> Option<glam::Vec3> {
        let id = self.doc.active?;
        let idx = self.doc.scene.objects.iter().position(|o| o.id == id)?;
        let (origin, dir) = self.ray_from_cursor(cursor)?;
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let o_l = inv.transform_point3(origin);
        let d_l = inv.transform_vector3(dir).normalize_or_zero();
        let mesh = &self.doc.scene.objects[idx].mesh;
        let mut best_t = f32::INFINITY;
        let mut hit = None;
        for tri in mesh.indices.chunks_exact(3) {
            if tri.iter().any(|&i| i as usize >= mesh.vertices.len()) {
                continue;
            }
            let a = glam::Vec3::from(mesh.vertices[tri[0] as usize].position);
            let b = glam::Vec3::from(mesh.vertices[tri[1] as usize].position);
            let c = glam::Vec3::from(mesh.vertices[tri[2] as usize].position);
            if let Some(t) = ray_triangle(o_l, d_l, a, b, c) {
                if t > 1e-5 && t < best_t {
                    best_t = t;
                    hit = Some(o_l + d_l * t);
                }
            }
        }
        hit
    }

    fn vertex_or_weight_paint_sample(&mut self) {
        let Some(hit) = self.paint_local_hit(self.cursor) else {
            return;
        };
        if self.doc.vertex_paint_mode {
            let s = self.vertex_paint_settings();
            let _ = self.doc.vertex_paint_at(&[hit], &s);
        } else if self.doc.weight_paint_mode {
            let s = self.weight_paint_settings();
            let _ = self.doc.weight_paint_at(&[hit], &s);
        }
    }

    fn paint_sample_at(&self, cursor: (f64, f64)) -> Option<glam::Vec2> {
        if !self.doc.paint_mode {
            return None;
        }
        let (vx, vy, vw, vh) = (
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        if vw < 1.0
            || vh < 1.0
            || cursor.0 < vx as f64
            || cursor.1 < vy as f64
            || cursor.0 > (vx + vw) as f64
            || cursor.1 > (vy + vh) as f64
        {
            return None;
        }
        let nx = ((cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((cursor.1 as f32 - vy) / vh) * 2.0;
        let (origin, dir) = self.camera.ray((nx, ny), vw / vh);
        // Raycast active mesh for UV
        let Some(id) = self.doc.active else {
            return None;
        };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return None;
        };
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(origin);
        let local_d = inv.transform_vector3(dir).normalize_or_zero();
        let mesh = &self.doc.scene.objects[idx].mesh;
        let mut best: Option<(f32, glam::Vec2)> = None;
        for tri in mesh.indices.chunks_exact(3) {
            if tri
                .iter()
                .any(|&index| index as usize >= mesh.vertices.len())
            {
                continue;
            }
            let a = glam::Vec3::from(mesh.vertices[tri[0] as usize].position);
            let b = glam::Vec3::from(mesh.vertices[tri[1] as usize].position);
            let c = glam::Vec3::from(mesh.vertices[tri[2] as usize].position);
            // Möller–Trumbore
            let e1 = b - a;
            let e2 = c - a;
            let p = local_d.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-10 {
                continue;
            }
            let inv_det = 1.0 / det;
            let tvec = local_o - a;
            let u = tvec.dot(p) * inv_det;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = tvec.cross(e1);
            let v = local_d.dot(q) * inv_det;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e2.dot(q) * inv_det;
            if t <= 1e-5 {
                continue;
            }
            if best.map(|(bt, _)| t < bt).unwrap_or(true) {
                let ua = glam::Vec2::from(mesh.vertices[tri[0] as usize].uv);
                let ub = glam::Vec2::from(mesh.vertices[tri[1] as usize].uv);
                let uc = glam::Vec2::from(mesh.vertices[tri[2] as usize].uv);
                let uv = ua * (1.0 - u - v) + ub * u + uc * v;
                best = Some((t, uv));
            }
        }
        let Some((_, uv)) = best else { return None };
        Some(uv)
    }

    fn sculpt_sample(&mut self) {
        if !self.doc.sculpt_mode {
            return;
        }
        // Throttle: at least ~6 px between samples (prevents crash / meltdown)
        if let Some((lx, ly)) = self.sculpt_last_sample {
            let dist = ((self.cursor.0 - lx).powi(2) + (self.cursor.1 - ly).powi(2)).sqrt();
            if dist < 6.0 {
                return;
            }
        }
        self.sculpt_last_sample = Some(self.cursor);

        self.sync_sculpt_brush_from_tool();
        self.sculpt.invert = self.ctrl();
        let (vx, vy, vw, vh) = (
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        if vw < 1.0 || vh < 1.0 {
            return;
        }
        let nx = ((self.cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((self.cursor.1 as f32 - vy) / vh) * 2.0;
        let (origin, dir) = self.camera.ray((nx, ny), vw / vh);
        let grab_delta = if matches!(
            self.sculpt.brush,
            forge_core::SculptBrush::Grab
                | forge_core::SculptBrush::SnakeHook
                | forge_core::SculptBrush::Nudge
                | forge_core::SculptBrush::Thumb
        ) {
            let dx = (self.cursor.0 - self.press.0) as f32 * 0.0025 * self.camera.distance;
            let dy = (self.press.1 - self.cursor.1) as f32 * 0.0025 * self.camera.distance;
            let view = self.camera.view().inverse();
            let right = view.x_axis.truncate();
            let up = view.y_axis.truncate();
            Some(right * dx + up * dy)
        } else {
            None
        };
        let _ = self
            .doc
            .sculpt_stroke(origin, dir, &self.sculpt, grab_delta);
        // Live status: vertex count + dyntopo
        if let Some(id) = self.doc.active {
            if let Some(obj) = self.doc.scene.objects.iter().find(|o| o.id == id) {
                let n = obj.mesh.vertices.len();
                if self.sculpt.dyntopo {
                    self.status(format!(
                        "Dyntopo ON · relative {:.0}% · {} verts  ([ ] or Alt+Wheel)",
                        self.sculpt.dyntopo_detail * 100.0,
                        n
                    ));
                }
            }
        }
    }

    fn raycast_ground_or_mesh(&self) -> Option<glam::Vec3> {
        let (vx, vy, vw, vh) = (
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        if vw < 1.0 || vh < 1.0 {
            return None;
        }
        let nx = ((self.cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((self.cursor.1 as f32 - vy) / vh) * 2.0;
        let (origin, dir) = self.camera.ray((nx, ny), vw / vh);
        // Ground plane y = 0
        if dir.y.abs() > 1e-6 {
            let t = -origin.y / dir.y;
            if t > 0.0 {
                return Some(origin + dir * t);
            }
        }
        Some(origin + dir * 5.0)
    }

    fn status(&mut self, s: impl Into<String>) {
        let s = s.into();
        log::info!("{s}");
        self.ui.status = s;
    }

    /// Update the render progress overlay and paint one UI frame so the user sees %.
    fn set_render_progress(&mut self, title: impl Into<String>, detail: impl Into<String>, fraction: f32) {
        self.ui.render_progress = Some(RenderProgressUi {
            title: title.into(),
            detail: detail.into(),
            fraction: fraction.clamp(0.0, 1.0),
        });
        self.pump_progress_ui();
    }

    fn clear_render_progress(&mut self) {
        self.ui.render_progress = None;
    }

    /// Paint only the progress/done overlays on top of the current swapchain.
    fn pump_progress_ui(&mut self) {
        let Some(g) = self.gfx.as_mut() else {
            return;
        };
        let (w, h) = g.renderer.size();
        if w < 2 || h < 2 {
            return;
        }
        let ppp = g.window.scale_factor() as f32;
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32 / ppp, h as f32 / ppp),
        ));
        raw.predicted_dt = 1.0 / 60.0;
        // egui 0.29 uses pixels_per_point on the context / screen descriptor, not always on RawInput.
        let progress = self.ui.render_progress.clone();
        let done = self.ui.render_done_message.clone();
        // Temporarily apply progress for the overlay draw without needing full UiState mutability games.
        let out = self.egui_ctx.run(raw, |ctx| {
            if progress.is_some() || done.is_some() {
                let screen = ctx.screen_rect();
                egui::Area::new(egui::Id::new("render_modal_dim_pump"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(screen.min)
                    .show(ctx, |ui| {
                        let _ = ui.allocate_rect(screen, egui::Sense::hover());
                        ui.painter().rect_filled(
                            screen,
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
                        );
                    });
            }
            if let Some(p) = progress.as_ref() {
                let frac = p.fraction.clamp(0.0, 1.0);
                let pct = (frac * 100.0).round() as i32;
                let card = egui::Frame::none()
                    .fill(egui::Color32::from_rgb(0x2c, 0x2c, 0x30))
                    .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x5a, 0x5a, 0x62)))
                    .inner_margin(egui::Margin::symmetric(14.0, 12.0))
                    .rounding(6.0);
                egui::Window::new("Rendering…")
                    .id(egui::Id::new("render_progress_window_pump"))
                    .title_bar(false)
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .order(egui::Order::Tooltip)
                    .frame(card)
                    .show(ctx, |ui| {
                        ui.set_width(280.0);
                        ui.label(
                            egui::RichText::new(&p.title)
                                .strong()
                                .size(14.0)
                                .color(egui::Color32::from_rgb(0xf0, 0xf0, 0xf2)),
                        );
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(&p.detail)
                                .size(11.5)
                                .color(egui::Color32::from_rgb(0xa8, 0xa8, 0xb0)),
                        );
                        ui.add_space(8.0);
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .show_percentage()
                                .desired_width(ui.available_width()),
                        );
                        ui.label(
                            egui::RichText::new(format!("{pct}%"))
                                .monospace()
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0x88, 0xcc, 0xff)),
                        );
                    });
            }
        });
        let tris = self.egui_ctx.tessellate(out.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: ppp,
        };
        let textures = out.textures_delta;
        let er = &mut g.egui_renderer;
        // Zero viewport → skip 3D/path-trace; only clear + draw the progress overlay.
        let res = g.renderer.render(
            &self.camera,
            &self.settings,
            [0.0, 0.0, 0.0, 0.0],
            &mut |device, queue, enc, view| {
                for (id, delta) in &textures.set {
                    er.update_texture(device, queue, *id, delta);
                }
                let bufs = er.update_buffers(device, queue, enc, &tris, &screen);
                {
                    let mut rp = enc
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui-progress"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        })
                        .forget_lifetime();
                    er.render(&mut rp, &tris, &screen);
                }
                for id in &textures.free {
                    er.free_texture(id);
                }
                bufs
            },
        );
        if let Err(e) = res {
            log::warn!("progress UI render: {e}");
        }
        g.window.request_redraw();
    }

    fn confirm_discard(&self) -> bool {
        if !self.doc.dirty {
            return true;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            rfd::MessageDialog::new()
                .set_title("Unsaved changes")
                .set_description("The current scene has unsaved changes. Discard them?")
                .set_buttons(rfd::MessageButtons::YesNo)
                .show()
                == rfd::MessageDialogResult::Yes
        }
        #[cfg(target_arch = "wasm32")]
        {
            true
        }
    }

    fn reset_view_state(&mut self) {
        self.modal = None;
        self.sel_drag = None;
        self.camera_view = None;
        self.camera.fixed = None;
        self.uploaded = u64::MAX;
    }

    // ------------------------------------------------------------------ camera view
    /// Handle a click / drag on the viewport navigation gizmo.
    fn apply_nav(&mut self, action: navgizmo::NavAction) {
        use navgizmo::NavAction as N;
        match action {
            N::SnapView { yaw, pitch } => {
                // Snapping to an axis leaves camera view, like Blender.
                if self.camera_view.is_some() {
                    self.camera_view = None;
                    self.camera.fixed = None;
                }
                self.view_anim = Some(ViewAnim {
                    from: (self.camera.yaw, self.camera.pitch),
                    to: (yaw, pitch),
                    start: web_time::Instant::now(),
                });
            }
            N::Orbit(dx, dy) => {
                self.view_anim = None;
                self.camera.orbit(dx, dy);
            }
            N::Pan(dx, dy) => self.camera.pan(dx, dy),
            N::Zoom(amount) => self.camera.zoom(amount),
            N::ToggleCameraView => self.toggle_camera_view(),
            N::TogglePerspective => {
                self.camera.ortho = !self.camera.ortho;
                self.status(if self.camera.ortho {
                    "Orthographic view"
                } else {
                    "Perspective view"
                });
            }
        }
    }

    /// Advance the eased axis-view transition, if one is running.
    fn step_view_anim(&mut self) {
        let Some((from, to, start)) = self.view_anim.as_ref().map(|a| (a.from, a.to, a.start))
        else {
            return;
        };
        let t = (start.elapsed().as_secs_f32() / 0.18).clamp(0.0, 1.0);
        let e = t * t * (3.0 - 2.0 * t);
        let tau = std::f32::consts::TAU;
        let mut dyaw = (to.0 - from.0) % tau;
        if dyaw > std::f32::consts::PI {
            dyaw -= tau;
        } else if dyaw < -std::f32::consts::PI {
            dyaw += tau;
        }
        self.camera.yaw = from.0 + dyaw * e;
        self.camera.pitch = from.1 + (to.1 - from.1) * e;
        if t >= 1.0 {
            self.camera.yaw = to.0;
            self.camera.pitch = to.1;
            self.view_anim = None;
        }
    }

    /// Fixed view for a camera object at the scene's current transforms.
    fn fixed_view_for(&self, id: u64) -> Option<FixedView> {
        let i = self.doc.scene.index_of(id)?;
        match &self.doc.scene.objects[i].kind {
            ObjectKind::Camera(c) => {
                let w = self.doc.scene.world_matrices()[i];
                let (_, r, t) = w.to_scale_rotation_translation();
                Some(FixedView {
                    eye: t,
                    view: Mat4::from_rotation_translation(r, t).inverse(),
                    fov_y: c.fov_deg.to_radians(),
                    near: c.near,
                    far: c.far,
                    ortho_height: c.ortho.then_some(c.ortho_size),
                })
            }
            _ => None,
        }
    }

    fn sync_camera_view(&mut self) {
        let Some(id) = self.camera_view else {
            self.camera.fixed = None;
            return;
        };
        let view = self.fixed_view_for(id);
        if view.is_none() {
            self.camera_view = None;
        }
        self.camera.fixed = view;
    }

    /// Camera for offline renders — WYSIWYG with the 3D viewport.
    /// Uses exactly the view currently on screen (orbit *or* locked camera view).
    /// Does NOT silently switch to a different scene camera object.
    fn render_camera(&self) -> OrbitCamera {
        let mut cam = self.camera.clone();
        // Refresh fixed view from the active camera-view object so animation keys apply.
        if let Some(id) = self.camera_view {
            if let Some(v) = self.fixed_view_for(id) {
                cam.fixed = Some(v);
            }
        }
        if cam.fixed.is_some() {
            cam.ortho = false;
        }
        cam
    }

    fn toggle_camera_view(&mut self) {
        if self.camera_view.is_some() {
            self.camera_view = None;
            self.camera.fixed = None;
            self.status("Left camera view");
            return;
        }
        let is_cam = |o: &forge_core::Object| matches!(o.kind, ObjectKind::Camera(_));
        let from_active = self
            .doc
            .active_index()
            .filter(|&i| is_cam(&self.doc.scene.objects[i]));
        let idx = from_active.or_else(|| self.doc.scene.objects.iter().position(is_cam));
        match idx {
            Some(i) => {
                self.camera_view = Some(self.doc.scene.objects[i].id);
                self.sync_camera_view();
                self.status("Camera view (press 0 to leave)");
            }
            None => self.status("No camera in the scene. Use Add > Camera."),
        }
    }

    // ------------------------------------------------------------------ picking / cursor
    fn pick(&mut self, extend: bool) {
        if self.doc.in_edit_mode() {
            self.pick_edit(extend);
            return;
        }
        let [vx, vy, vw, vh] = self.viewport;
        if vw < 1.0 || vh < 1.0 {
            return;
        }
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (o, d) = self.camera.ray((nx, ny), vw / vh);
        let hits = self.doc.scene.pick_all_hits(o, d);
        let cursor_key = (self.cursor.0 as i32 / 4, self.cursor.1 as i32 / 4);
        let alt = self.mods.alt_key();
        // Cycle when Alt is held, or when clicking the same spot again with multiple hits.
        let same_spot = cursor_key == self.pick_cycle_cursor;
        let cycle = !extend
            && hits.len() > 1
            && (alt || (same_spot && !self.pick_cycle_hits.is_empty()));
        let chosen = if hits.is_empty() {
            self.pick_cycle_hits.clear();
            self.pick_cycle_index = 0;
            None
        } else if cycle {
            let hit_ids: Vec<usize> = hits.iter().map(|(i, _)| *i).collect();
            if hit_ids != self.pick_cycle_hits {
                self.pick_cycle_hits = hit_ids;
                self.pick_cycle_index = 0;
            } else {
                self.pick_cycle_index = (self.pick_cycle_index + 1) % self.pick_cycle_hits.len();
            }
            self.pick_cycle_cursor = cursor_key;
            Some(self.pick_cycle_hits[self.pick_cycle_index])
        } else {
            self.pick_cycle_hits = hits.iter().map(|(i, _)| *i).collect();
            self.pick_cycle_index = 0;
            self.pick_cycle_cursor = cursor_key;
            Some(hits[0].0)
        };
        match (chosen, extend) {
            (Some(i), true) => self.doc.select_toggle(i),
            (Some(i), false) => {
                self.doc.select_only(i);
                if self.pick_cycle_hits.len() > 1 {
                    let name = self
                        .doc
                        .scene
                        .objects
                        .get(i)
                        .map(|o| o.name.as_str())
                        .unwrap_or("?");
                    self.status(format!(
                        "Selected {} ({}/{}) — Alt+click or re-click to cycle",
                        name,
                        self.pick_cycle_index + 1,
                        self.pick_cycle_hits.len()
                    ));
                }
            }
            (None, false) => self.doc.select_none(),
            (None, true) => {}
        }
        if !self.doc.armature_edit
            && !self.doc.pose_mode
            && matches!(
                self.ts.tool,
                Tool::BoneAdd | Tool::BoneExtrude | Tool::BoneDelete | Tool::PoseClear
            )
        {
            self.ts.tool = Tool::Select;
        }
    }

    /// If verts are selected, start a free Move grab (same as G). Returns true if started.
    /// Select component under cursor and start a hold-to-drag move.

    fn edit_selection_world_center(&self) -> Option<(glam::Vec3, glam::Mat4, usize)> {
        let session = self.doc.edit.as_ref()?;
        if session.selected_vert_indices().is_empty() {
            return None;
        }
        let id = session.object_id;
        let idx = self.doc.scene.objects.iter().position(|o| o.id == id)?;
        let world = self.doc.scene.world_matrices()[idx];
        let local = session.selection_center();
        let origin = world.transform_point3(local);
        Some((origin, world, idx))
    }

    fn edit_gizmo_len(&self, origin: glam::Vec3) -> f32 {
        let dist = (origin - self.camera.target).length().max(0.5);
        (dist * 0.15).clamp(0.3, 8.0)
    }

    fn edit_gizmo_hover_test(&mut self) {
        self.edit_gizmo_hover = None;
        let Some((origin, _, _)) = self.edit_selection_world_center() else {
            return;
        };
        let len = self.edit_gizmo_len(origin);
        // (viewport components unused here)
        let cam = &self.camera;
        let vp = self.viewport;
        use crate::tools::project_px;
        let cursor = glam::Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
        let mut best: Option<(usize, f32)> = None;
        for (i, dir) in axes.iter().enumerate() {
            let (Some(p0), Some(p1)) = (
                project_px(cam, vp, origin),
                project_px(cam, vp, origin + *dir * len),
            ) else {
                continue;
            };
            // distance from cursor to segment
            let ab = p1 - p0;
            let t = ((cursor - p0).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
            let closest = p0 + ab * t;
            let d = (cursor - closest).length();
            if d < 12.0 && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        self.edit_gizmo_hover = best.map(|(i, _)| i);
    }

    fn begin_edit_gizmo_drag(&mut self) -> bool {
        let Some(axis) = self.edit_gizmo_hover else {
            return false;
        };
        let Some(session) = self.doc.edit.as_ref() else {
            return false;
        };
        let verts = session.snapshot_positions();
        if verts.is_empty() {
            return false;
        }
        let Some((origin, world, _)) = self.edit_selection_world_center() else {
            return false;
        };
        let axis_dir = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][axis];
        let (ro, rd) = {
            let [vx, vy, vw, vh] = self.viewport;
            let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
            let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
            self.camera.ray((nx, ny), vw / vh)
        };
        // param along axis from ray
        let _start = {
            let w = rd.cross(axis_dir);
            let den = w.length_squared();
            if den < 1e-10 {
                0.0
            } else {
                (origin - ro).cross(rd).dot(w) / den // rough
            }
        };
        // better: closest points between ray and axis line
        let start = {
            let u = axis_dir;
            let v = rd;
            let w0 = origin - ro;
            let a = u.dot(u);
            let b = u.dot(v);
            let c = v.dot(v);
            let d = u.dot(w0);
            let e = v.dot(w0);
            let den = a * c - b * b;
            if den.abs() < 1e-10 {
                0.0
            } else {
                (b * e - c * d) / den
            }
        };
        let mesh_snap: Vec<[f32; 3]> =
            session.mesh.vertices.iter().map(|v| v.position).collect();
        let selected: Vec<u32> = verts.iter().map(|(i, _)| *i).collect();
        let mirror_pairs =
            forge_core::bake_mirror_pairs(&mesh_snap, &selected, self.doc.mesh_symmetry);
        self.doc.checkpoint();
        self.edit_gizmo_drag = Some(EditGizmoDrag {
            axis,
            axis_dir,
            origin,
            start_param: start,
            verts,
            inv_world: world.inverse(),
            mirror_pairs,
            mesh_snap,
        });
        true
    }

    fn update_edit_gizmo_drag(&mut self) {
        let Some(d) = self.edit_gizmo_drag.as_ref() else {
            return;
        };
        let (ro, rd) = {
            let [vx, vy, vw, vh] = self.viewport;
            let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
            let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
            self.camera.ray((nx, ny), vw / vh)
        };
        let u = d.axis_dir;
        let v = rd;
        let w0 = d.origin - ro;
        let a = u.dot(u);
        let b = u.dot(v);
        let c = v.dot(v);
        let d_ = u.dot(w0);
        let e = v.dot(w0);
        let den = a * c - b * b;
        let t = if den.abs() < 1e-10 {
            0.0
        } else {
            (b * e - c * d_) / den
        };
        let delta_world = d.axis_dir * (t - d.start_param);
        let delta_local = d.inv_world.transform_vector3(delta_world);
        let positions: Vec<(u32, glam::Vec3)> = d
            .verts
            .iter()
            .map(|(vi, p)| (*vi, *p + delta_local))
            .collect();
        let pairs = d.mirror_pairs.clone();
        let mesh_snap = self
            .edit_gizmo_drag
            .as_ref()
            .map(|d| d.mesh_snap.clone())
            .unwrap_or_default();
        if let Some(session) = self.doc.edit.as_mut() {
            session.set_vert_positions(&positions);
            if !pairs.is_empty() && !mesh_snap.is_empty() {
                let mut pos: Vec<[f32; 3]> =
                    session.mesh.vertices.iter().map(|v| v.position).collect();
                forge_core::apply_mirror_pairs_full(&mut pos, &mesh_snap, &positions, &pairs);
                for (i, p) in pos.into_iter().enumerate() {
                    if let Some(v) = session.mesh.vertices.get_mut(i) {
                        v.position = p;
                    }
                }
                session.set_vert_positions(&positions);
                session.mesh.compute_normals();
            }
        }
        self.doc.sync_edit_mesh();
    }

    fn end_edit_gizmo_drag(&mut self) {
        self.edit_gizmo_drag = None;
    }

    /// Start an extrusion directly from a visible bone joint. The new bone
    /// begins exactly at the clicked head or tail and is placed by dragging.
    fn start_bone_extrude_from_joint(&mut self, parent: u32, from_head: bool) -> bool {
        if !self.doc.armature_edit {
            return false;
        }
        let joint = {
            let Some(arm) = self.doc.active_armature() else {
                return false;
            };
            let Some(bone) = arm.bones.get(parent as usize) else {
                return false;
            };
            if from_head {
                bone.head_v()
            } else {
                bone.tail_v()
            }
        };
        self.doc.checkpoint();
        let new_index = self
            .doc
            .active_armature_mut()
            .and_then(|arm| arm.extrude_from_joint(parent, from_head));
        let Some(new_index) = new_index else {
            self.doc.undo();
            return false;
        };
        let head = joint;
        self.bone_extrude_drag = Some(BoneExtrudeDrag {
            bone_idx: new_index,
            head,
        });
        self.update_bone_extrude();
        self.status(if from_head {
            "Extruding branch from bone head"
        } else {
            "Extruding child from bone tail"
        });
        true
    }

    /// Hit-test the visible head/tail handles. Returns true when a drag starts.
    fn try_start_bone_joint_extrude(&mut self) -> bool {
        if !self.doc.armature_edit {
            return false;
        }
        let Some(id) = self.doc.active else {
            return false;
        };
        let Some(object_index) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return false;
        };
        let world = self.doc.scene.world_matrices()[object_index];
        let Some(arm) = self.doc.active_armature() else {
            return false;
        };
        let cursor = glam::Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let mut best: Option<(f32, u32, bool)> = None;
        for (index, bone) in arm.bones.iter().enumerate() {
            for (point, from_head) in [(bone.head_v(), true), (bone.tail_v(), false)] {
                let point_world = world.transform_point3(point);
                let Some(screen) = tools::project_px(&self.camera, self.viewport, point_world)
                else {
                    continue;
                };
                let distance = (screen - cursor).length();
                if distance <= 10.0 && best.map_or(true, |(d, _, _)| distance < d) {
                    best = Some((distance, index as u32, from_head));
                }
            }
        }
        if let Some((_, bone, from_head)) = best {
            return self.start_bone_extrude_from_joint(bone, from_head);
        }
        false
    }

    /// Blender-style: extrude bone from selected tip, then grab the new tip with the mouse.
    fn start_bone_extrude(&mut self) {
        if !self.doc.armature_edit {
            self.status("Tab into Armature Edit Mode first");
            return;
        }
        self.doc.checkpoint();
        let new_idx = {
            let Some(a) = self.doc.active_armature_mut() else {
                self.status("Select an Armature");
                return;
            };
            if a.selected.is_empty() {
                if let Some(act) = a.active {
                    a.selected.insert(act);
                }
            }
            if a.selected.is_empty() {
                self.status("Select a bone to extrude from");
                return;
            }
            a.extrude_selected();
            a.active
        };
        let Some(bi) = new_idx else {
            self.status("Extrude failed");
            return;
        };
        let head = self
            .doc
            .active_armature()
            .and_then(|a| a.bones.get(bi as usize))
            .map(|b| glam::Vec3::from(b.head))
            .unwrap_or(glam::Vec3::ZERO);
        self.bone_extrude_drag = Some(BoneExtrudeDrag { bone_idx: bi, head });
        self.status("Extrude bone — move mouse to place tip, LMB confirm, RMB/Esc cancel");
    }

    fn update_bone_extrude(&mut self) {
        let Some(drag) = self.bone_extrude_drag.as_ref() else {
            return;
        };
        let bi = drag.bone_idx;
        let head = drag.head;
        let id = match self.doc.active {
            Some(id) => id,
            None => return,
        };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let [vx, vy, vw, vh] = self.viewport;
        if vw < 1.0 || vh < 1.0 {
            return;
        }
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (ro, rd) = self.camera.ray((nx, ny), vw / vh);
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(ro);
        let local_d = inv.transform_vector3(rd).normalize_or_zero();
        // Plane through head, facing the camera (armature local)
        let cam_fwd_world = (self.camera.target - self.camera.eye()).normalize_or_zero();
        let plane_n = inv.transform_vector3(cam_fwd_world).normalize_or_zero();
        let denom = local_d.dot(plane_n);
        let tip = if denom.abs() > 1e-6 {
            let t = (head - local_o).dot(plane_n) / denom;
            if t > 0.0 {
                local_o + local_d * t
            } else {
                head + glam::Vec3::Y * 0.3
            }
        } else {
            head + glam::Vec3::Y * 0.3
        };
        // Keep a minimum length
        let mut tip = tip;
        if (tip - head).length() < 0.05 {
            tip = head + (tip - head).normalize_or_zero() * 0.05;
            if (tip - head).length() < 0.04 {
                tip = head + glam::Vec3::Y * 0.25;
            }
        }
        if let Some(a) = self.doc.active_armature_mut() {
            if let Some(b) = a.bones.get_mut(bi as usize) {
                b.head = head.to_array();
                b.tail = tip.to_array();
            }
        }
        self.doc.touch();
    }

    fn confirm_bone_extrude(&mut self) {
        self.bone_extrude_drag = None;
        self.status("Bone extruded");
    }

    fn cancel_bone_extrude(&mut self) {
        self.bone_extrude_drag = None;
        // Undo the checkpoint taken at start of extrude
        self.doc.undo();
        self.status("Extrude cancelled");
    }

    fn start_bone_xform(&mut self, kind: EditXformKind, confirm_on_release: bool) {
        let Some(arm) = self.doc.active_armature() else {
            self.status("Select an Armature first");
            return;
        };
        let mut indices: Vec<u32> = arm.selected.iter().copied().collect();
        if indices.is_empty() {
            if let Some(active) = arm.active {
                indices.push(active);
            }
        }
        if indices.is_empty() {
            self.status("Select one or more bones first");
            return;
        }
        let snapshots: Vec<_> = indices
            .into_iter()
            .filter_map(|i| {
                arm.bones
                    .get(i as usize)
                    .map(|b| (i, b.head, b.tail, b.pose_rot, b.pose_loc, b.pose_scale))
            })
            .collect();
        if snapshots.is_empty() {
            return;
        }
        let rest_joints: Vec<_> = arm.bones.iter().map(|b| (b.head_v(), b.tail_v())).collect();
        let center = snapshots.iter().fold(glam::Vec3::ZERO, |sum, b| {
            sum + glam::Vec3::from(b.1) + glam::Vec3::from(b.2)
        }) / (snapshots.len() as f32 * 2.0);
        let Some(id) = self.doc.active else { return };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let inv_world = self.doc.scene.world_matrices()[idx].inverse();
        self.doc.checkpoint();
        self.bone_xform_drag = Some(BoneXform {
            kind,
            start_cursor: self.cursor,
            bones: snapshots,
            rest_joints,
            center,
            inv_world,
        });
        let name = match kind {
            EditXformKind::Move => "Move",
            EditXformKind::Rotate => "Rotate",
            EditXformKind::Scale => "Scale",
        };
        self.status(if confirm_on_release {
            format!("Bone {name} — drag and release to confirm")
        } else {
            format!("Bone {name} — LMB confirm, RMB/Esc cancel")
        });
    }

    fn update_bone_xform(&mut self) {
        let Some(xf) = self.bone_xform_drag.as_ref() else {
            return;
        };
        let dx = (self.cursor.0 - xf.start_cursor.0) as f32;
        let dy = (self.cursor.1 - xf.start_cursor.1) as f32;
        let inv_view = self.camera.view().inverse();
        let right = inv_view.x_axis.truncate().normalize_or_zero();
        let up = inv_view.y_axis.truncate().normalize_or_zero();
        let view_axis = inv_view.z_axis.truncate().normalize_or_zero();
        let dist = (self.camera.target - inv_view.w_axis.truncate())
            .length()
            .max(0.1);
        let world_delta = right * (dx * dist * 0.0025) + up * (-dy * dist * 0.0025);
        let local_delta = xf.inv_world.transform_vector3(world_delta);
        let local_axis = xf
            .inv_world
            .transform_vector3(view_axis)
            .normalize_or_zero();
        let angle = dx * 0.01;
        let rotation = glam::Quat::from_axis_angle(local_axis, angle);
        let factor = (-dy * 0.01).exp().clamp(0.01, 100.0);
        let center = xf.center;
        let kind = xf.kind;
        let snapshots = xf.bones.clone();
        let rest_joints = xf.rest_joints.clone();
        let selected_bones: std::collections::BTreeSet<u32> =
            snapshots.iter().map(|b| b.0).collect();
        let pose_mode = self.doc.pose_mode;
        let mesh_symmetry = self.doc.mesh_symmetry;
        if let Some(arm) = self.doc.active_armature_mut() {
            // Collect mirror targets before mut borrow of individual bones
            let mirror_pairs: Vec<(u32, Option<u32>)> = if mesh_symmetry.any() {
                snapshots
                    .iter()
                    .map(|(i, ..)| (*i, arm.mirror_bone_index(*i)))
                    .collect()
            } else {
                snapshots.iter().map(|(i, ..)| (*i, None)).collect()
            };
            for (i, head, tail, pose_rot, pose_loc, pose_scale) in snapshots.iter().copied() {
                let Some(b) = arm.bones.get_mut(i as usize) else {
                    continue;
                };
                if pose_mode {
                    match kind {
                        EditXformKind::Move => {
                            let p = glam::Vec3::from(pose_loc) + local_delta;
                            b.pose_loc = p.to_array();
                        }
                        EditXformKind::Rotate => {
                            let rest_dir = (glam::Vec3::from(tail) - glam::Vec3::from(head))
                                .normalize_or_zero();
                            let rest_rot = if rest_dir.length_squared() < 1e-10 {
                                glam::Quat::IDENTITY
                            } else {
                                glam::Quat::from_rotation_arc(glam::Vec3::Y, rest_dir)
                            };
                            let bone_axis = rest_rot.conjugate() * local_axis;
                            let bone_delta = glam::Quat::from_axis_angle(bone_axis, angle);
                            let q = glam::Quat::from_xyzw(
                                pose_rot[0],
                                pose_rot[1],
                                pose_rot[2],
                                pose_rot[3],
                            )
                            .normalize();
                            let q = (bone_delta * q).normalize();
                            b.pose_rot = [q.x, q.y, q.z, q.w];
                        }
                        EditXformKind::Scale => {
                            b.pose_scale = (glam::Vec3::from(pose_scale) * factor)
                                .clamp(glam::Vec3::splat(0.01), glam::Vec3::splat(100.0))
                                .to_array();
                        }
                    }
                } else {
                    let h = glam::Vec3::from(head);
                    let t = glam::Vec3::from(tail);
                    let (h, t) = match kind {
                        EditXformKind::Move => (h + local_delta, t + local_delta),
                        EditXformKind::Rotate => (
                            center + rotation * (h - center),
                            center + rotation * (t - center),
                        ),
                        EditXformKind::Scale => (
                            center + (h - center) * factor,
                            center + (t - center) * factor,
                        ),
                    };
                    b.head = h.to_array();
                    b.tail = t.to_array();
                }
            }
            // X-mirror: apply mirrored pose/rest transforms to counterpart bones
            if mesh_symmetry.any() {
                let mdelta = forge_core::mirror_vec_sym(mesh_symmetry, local_delta);
                let maxis = forge_core::mirror_vec_sym(mesh_symmetry, local_axis).normalize_or_zero();
                let mcenter = forge_core::mirror_point_sym(mesh_symmetry, center);
                for (i, head, tail, pose_rot, pose_loc, pose_scale) in snapshots.iter().copied() {
                    let Some((_, Some(mi))) = mirror_pairs.iter().find(|(si, _)| *si == i) else {
                        continue;
                    };
                    let mi = *mi;
                    if selected_bones.contains(&mi) {
                        continue;
                    }
                    let Some(mb) = arm.bones.get_mut(mi as usize) else {
                        continue;
                    };
                    if pose_mode {
                        match kind {
                            EditXformKind::Move => {
                                let p = glam::Vec3::from(pose_loc) + mdelta;
                                mb.pose_loc = p.to_array();
                            }
                            EditXformKind::Rotate => {
                                let rest_dir = (glam::Vec3::from(tail) - glam::Vec3::from(head))
                                    .normalize_or_zero();
                                let rest_rot = if rest_dir.length_squared() < 1e-10 {
                                    glam::Quat::IDENTITY
                                } else {
                                    glam::Quat::from_rotation_arc(glam::Vec3::Y, rest_dir)
                                };
                                let bone_axis = rest_rot.conjugate() * maxis;
                                let bone_delta = glam::Quat::from_axis_angle(bone_axis, angle);
                                let q = glam::Quat::from_xyzw(
                                    pose_rot[0],
                                    pose_rot[1],
                                    pose_rot[2],
                                    pose_rot[3],
                                )
                                .normalize();
                                let q = forge_core::mirror_quat_sym(mesh_symmetry, (bone_delta * q).normalize());
                                mb.pose_rot = [q.x, q.y, q.z, q.w];
                            }
                            EditXformKind::Scale => {
                                mb.pose_scale = (glam::Vec3::from(pose_scale) * factor)
                                    .clamp(glam::Vec3::splat(0.01), glam::Vec3::splat(100.0))
                                    .to_array();
                            }
                        }
                    } else {
                        let h = forge_core::mirror_point_sym(mesh_symmetry, glam::Vec3::from(head));
                        let t = forge_core::mirror_point_sym(mesh_symmetry, glam::Vec3::from(tail));
                        let (h, t) = match kind {
                            EditXformKind::Move => (h + mdelta, t + mdelta),
                            EditXformKind::Rotate => {
                                let mrot = forge_core::mirror_quat_sym(mesh_symmetry, rotation);
                                (mcenter + mrot * (h - mcenter), mcenter + mrot * (t - mcenter))
                            }
                            EditXformKind::Scale => (
                                mcenter + (h - mcenter) * factor,
                                mcenter + (t - mcenter) * factor,
                            ),
                        };
                        mb.head = h.to_array();
                        mb.tail = t.to_array();
                    }
                }
            }
            if !pose_mode {
                arm.propagate_connected_rest_joints(&rest_joints, &selected_bones);
            }
        }
        if pose_mode {
            self.doc.apply_all_skinning();
        }
        self.doc.touch();
    }

    fn finish_bone_xform(&mut self, cancel: bool) {
        if self.bone_xform_drag.take().is_some() {
            if cancel {
                self.doc.undo();
                self.status("Bone transform cancelled");
            } else {
                self.status("Bone transform confirmed");
            }
        }
    }

    fn start_bone_pose_rotate(&mut self) {
        self.start_bone_xform(EditXformKind::Rotate, false);
    }

    fn try_pick_bone(&mut self, extend: bool) -> bool {
        let Some(session_arm) = self.doc.active_armature() else {
            return false;
        };
        let id = self.doc.active.unwrap();
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return false;
        };
        let [vx, vy, vw, vh] = self.viewport;
        if vw < 1.0 || vh < 1.0 {
            return false;
        }
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (ro, rd) = self.camera.ray((nx, ny), vw / vh);
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(ro);
        let local_d = inv.transform_vector3(rd);
        let dist = (world.w_axis.truncate() - self.camera.target)
            .length()
            .max(0.5);
        // Generous threshold so bones inside / behind meshes are still easy to click
        let thresh = (dist * 0.22).max(0.15);
        let pick = if self.doc.armature_edit {
            session_arm.pick_bone_rest(local_o, local_d, thresh)
        } else {
            session_arm.pick_bone(local_o, local_d, thresh)
        };
        let Some(bi) = pick else { return false };
        let was_selected = session_arm.selected.contains(&bi);
        let preserve_multi_selection = self.ts.tool.is_gizmo() && was_selected && !extend;
        if !preserve_multi_selection {
            self.doc.checkpoint();
        }
        let selected_after_pick = if let Some(a) = self.doc.active_armature_mut() {
            // Clicking an already-selected bone with a transform tool should
            // preserve the whole multi-selection so it can be transformed.
            if preserve_multi_selection {
                a.active = Some(bi);
            } else {
                a.select_bone(bi, extend);
            }
            a.selected.contains(&bi)
        } else {
            false
        };
        self.status(format!("Bone {}", bi + 1));
        if selected_after_pick && self.ts.tool.is_gizmo() {
            let kind = match self.ts.tool {
                Tool::Move => EditXformKind::Move,
                Tool::Rotate => EditXformKind::Rotate,
                Tool::Scale => EditXformKind::Scale,
                _ => EditXformKind::Move,
            };
            self.start_bone_xform(kind, true);
        }
        true
    }

    fn try_edit_click_drag(&mut self, extend: bool) -> bool {
        let Some(session) = self.doc.edit.as_ref() else {
            return false;
        };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return false;
        };
        let [vx, vy, vw, vh] = self.viewport;
        if vw < 1.0 || vh < 1.0 {
            return false;
        }
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (ro, rd) = self.camera.ray((nx, ny), vw / vh);
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(ro);
        let local_d = inv.transform_vector3(rd);
        let dist = (world.w_axis.truncate() - self.camera.target)
            .length()
            .max(0.5);
        let thresh = dist * 0.08;
        // Edge mode: pick in screen space so clicking the drawn line always hits
        let pick = if session.selection.domain == forge_core::EditDomain::Edge {
            let cam = self.camera.clone();
            let vp = self.viewport;
            let project = move |p: glam::Vec3| -> Option<(f32, f32)> {
                crate::tools::project_px(&cam, vp, p).map(|v| (v.x, v.y))
            };
            session
                .pick_edge_screen(
                    world,
                    &project,
                    (self.cursor.0 as f32, self.cursor.1 as f32),
                    16.0, // ~16px hit radius around the drawn line
                )
                .or_else(|| session.pick_component(local_o, local_d, thresh * 3.0))
        } else {
            session.pick_component(local_o, local_d, thresh)
        };
        let Some(pick) = pick else {
            if !extend {
                self.doc.edit_select_none();
            }
            return false;
        };
        self.doc.edit_apply_pick(pick, extend);
        // If we now have verts to move, start drag immediately
        let has = self
            .doc
            .edit
            .as_ref()
            .map(|e| !e.selected_vert_indices().is_empty())
            .unwrap_or(false);
        if has {
            self.start_edit_xform_ex(EditXformKind::Move, true);
            return true;
        }
        true // selected but nothing to move (shouldn't happen)
    }

    fn begin_edit_axis_grab(&mut self) -> bool {
        let Some(session) = self.doc.edit.as_ref() else {
            return false;
        };
        if session.selected_vert_indices().is_empty() {
            return false;
        }
        self.start_edit_xform(EditXformKind::Move);
        true
    }

    fn start_edit_xform(&mut self, kind: EditXformKind) {
        self.start_edit_xform_ex(kind, false);
    }

    fn start_edit_xform_ex(&mut self, kind: EditXformKind, confirm_on_release: bool) {
        let Some(session) = self.doc.edit.as_ref() else {
            return;
        };
        let verts = session.snapshot_positions();
        if verts.is_empty() {
            self.status("Nothing selected");
            return;
        }
        let center = session.selection_center();
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let world = self.doc.scene.world_matrices()[idx];
        let inv_world = world.inverse();
        let mesh_snap: Vec<[f32; 3]> =
            session.mesh.vertices.iter().map(|v| v.position).collect();
        let selected: Vec<u32> = verts.iter().map(|(i, _)| *i).collect();
        let mirror_pairs =
            forge_core::bake_mirror_pairs(&mesh_snap, &selected, self.doc.mesh_symmetry);
        self.doc.checkpoint();
        self.edit_xform = Some(EditXform {
            kind,
            start_cursor: self.cursor,
            verts,
            center,
            inv_world,
            confirm_on_release,
            mirror_pairs,
            mesh_snap,
        });
        if confirm_on_release {
            self.status("Dragging selection…");
        } else {
            let name = match kind {
                EditXformKind::Move => "Move",
                EditXformKind::Rotate => "Rotate",
                EditXformKind::Scale => "Scale",
            };
            self.status(format!("{name} — LMB confirm, RMB/Esc cancel"));
        }
    }

    fn update_edit_xform(&mut self) {
        let Some(xf) = self.edit_xform.as_ref() else {
            return;
        };
        let dx = (self.cursor.0 - xf.start_cursor.0) as f32;
        let dy = (self.cursor.1 - xf.start_cursor.1) as f32;
        let inv_view = self.camera.view().inverse();
        let right = inv_view.x_axis.truncate().normalize_or_zero();
        let up = inv_view.y_axis.truncate().normalize_or_zero();
        let dist = (self.camera.target - inv_view.w_axis.truncate())
            .length()
            .max(0.1);

        let positions: Vec<(u32, glam::Vec3)> = match xf.kind {
            EditXformKind::Move => {
                let scale = dist * 0.0025;
                let world_delta = right * (dx * scale) + up * (-dy * scale);
                let local_delta = xf.inv_world.transform_vector3(world_delta);
                xf.verts
                    .iter()
                    .map(|(vi, p)| (*vi, *p + local_delta))
                    .collect()
            }
            EditXformKind::Rotate => {
                let angle = dx * 0.01;
                let axis = xf
                    .inv_world
                    .transform_vector3(inv_view.z_axis.truncate().normalize_or_zero())
                    .normalize_or_zero();
                let q = glam::Quat::from_axis_angle(axis, angle);
                xf.verts
                    .iter()
                    .map(|(vi, p)| (*vi, xf.center + q * (*p - xf.center)))
                    .collect()
            }
            EditXformKind::Scale => {
                let factor = (1.0 + dx * 0.005).max(0.01);
                let f = glam::Vec3::splat(factor);
                xf.verts
                    .iter()
                    .map(|(vi, p)| (*vi, xf.center + (*p - xf.center) * f))
                    .collect()
            }
        };
        // Take pairs out without holding xf borrow across mut session
        let pairs = self
            .edit_xform
            .as_ref()
            .map(|x| x.mirror_pairs.clone())
            .unwrap_or_default();
        let mesh_snap = self
            .edit_xform
            .as_ref()
            .map(|x| x.mesh_snap.clone())
            .unwrap_or_default();
        if let Some(session) = self.doc.edit.as_mut() {
            session.set_vert_positions(&positions);
            if !pairs.is_empty() && !mesh_snap.is_empty() {
                let mut pos: Vec<[f32; 3]> =
                    session.mesh.vertices.iter().map(|v| v.position).collect();
                forge_core::apply_mirror_pairs_full(&mut pos, &mesh_snap, &positions, &pairs);
                for (i, p) in pos.into_iter().enumerate() {
                    if let Some(v) = session.mesh.vertices.get_mut(i) {
                        v.position = p;
                    }
                }
                session.set_vert_positions(&positions);
                session.mesh.compute_normals();
            }
        }
        self.doc.sync_edit_mesh();
    }

    fn confirm_edit_xform(&mut self) {
        if self.edit_xform.take().is_some() {
            // checkpoint was taken at start; current state is the result
            self.status("Edit Mode");
        }
    }

    fn cancel_edit_xform(&mut self) {
        if let Some(xf) = self.edit_xform.take() {
            if let Some(session) = self.doc.edit.as_mut() {
                session.set_vert_positions(&xf.verts);
            }
            self.doc.sync_edit_mesh();
            // discard the checkpoint taken at start of the grab
            self.doc.drop_last_checkpoint();
            self.status("Transform cancelled");
        }
    }

    fn pick_edit(&mut self, extend: bool) {
        let Some(session) = self.doc.edit.as_ref() else {
            return;
        };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let [vx, vy, vw, vh] = self.viewport;
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (ro, rd) = self.camera.ray((nx, ny), vw / vh);
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(ro);
        let local_d = inv.transform_vector3(rd);
        // threshold ~ 12px in world units near the object
        let dist = (world.w_axis.truncate() - self.camera.target)
            .length()
            .max(0.5);
        let thresh = dist * 0.03;
        let pick = session.pick_component(local_o, local_d, thresh);
        match pick {
            Some(p) => self.doc.edit_apply_pick(p, extend),
            None if !extend => self.doc.edit_select_none(),
            None => {}
        }
    }

    fn set_cursor_at_mouse(&mut self) {
        let [vx, vy, vw, vh] = self.viewport;
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (o, d) = self.camera.ray((nx, ny), vw / vh);
        let p = if let Some((_, t)) = self.doc.scene.pick_hit(o, d) {
            Some(o + d * t)
        } else if d.y.abs() > 1e-6 && -o.y / d.y > 0.0 {
            Some(o + d * (-o.y / d.y))
        } else {
            None
        };
        if let Some(p) = p {
            self.doc.scene.cursor = p;
            self.doc.touch();
        }
    }

    // ------------------------------------------------------------------ selection tools
    fn begin_sel_drag(&mut self, shift: bool, ctrl: bool) {
        let op = if shift {
            SelectOp::Extend
        } else if ctrl {
            SelectOp::Subtract
        } else {
            SelectOp::Replace
        };
        let p = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        self.sel_drag = Some(SelDrag {
            tool: self.ts.tool,
            start: self.cursor,
            points: vec![p],
            op,
        });
        if self.doc.armature_edit || self.doc.pose_mode {
            self.doc.checkpoint();
        }
        if self.ts.tool == Tool::Circle {
            if op == SelectOp::Replace {
                if self.doc.armature_edit || self.doc.pose_mode {
                    if let Some(a) = self.doc.active_armature_mut() {
                        a.selected.clear();
                        a.active = None;
                    }
                } else {
                    self.doc.select_none();
                }
            }
            self.circle_paint(op);
        }
    }

    fn circle_paint(&mut self, op: SelectOp) {
        let c = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let r = self.ts.circle_radius;
        if self.doc.armature_edit || self.doc.pose_mode {
            let bone_op = if op == SelectOp::Replace {
                SelectOp::Extend
            } else {
                op
            };
            self.select_bones_in_shape(bone_op, |p| (p - c).length() <= r);
            return;
        }
        let hits: Vec<usize> = anchors(&self.doc, &self.camera, self.viewport)
            .into_iter()
            .filter(|(_, p)| (*p - c).length() <= r)
            .map(|(i, _)| i)
            .collect();
        apply_selection(
            &mut self.doc,
            &hits,
            if op == SelectOp::Subtract {
                SelectOp::Subtract
            } else {
                SelectOp::Extend
            },
        );
    }

    fn update_sel_drag(&mut self) {
        let p = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let (tool, op) = match self.sel_drag.as_mut() {
            Some(d) => {
                if d.tool == Tool::Lasso
                    && d.points.last().map_or(true, |l| (*l - p).length() > 3.0)
                {
                    d.points.push(p);
                }
                (d.tool, d.op)
            }
            None => return,
        };
        if tool == Tool::Circle {
            self.circle_paint(op);
        }
    }

    fn finish_sel_drag(&mut self) {
        let Some(d) = self.sel_drag.take() else {
            return;
        };
        if d.tool == Tool::Circle {
            return;
        }
        let cur = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let start = Vec2::new(d.start.0 as f32, d.start.1 as f32);
        if (cur - start).length() < 4.0 {
            if self.doc.armature_edit || self.doc.pose_mode {
                self.try_pick_bone(d.op == SelectOp::Extend);
            } else {
                self.pick(d.op == SelectOp::Extend);
            }
            return;
        }
        let (lo, hi) = (start.min(cur), start.max(cur));
        if self.doc.armature_edit || self.doc.pose_mode {
            let tool = d.tool;
            let points = d.points.clone();
            self.select_bones_in_shape(d.op, |p| match tool {
                Tool::BoxSelect => p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y,
                _ => point_in_polygon(p, &points),
            });
            return;
        }
        if self.doc.in_edit_mode() {
            let extend = d.op != SelectOp::Replace;
            self.finish_edit_sel_rect(lo, hi, extend);
            return;
        }
        let hits: Vec<usize> = anchors(&self.doc, &self.camera, self.viewport)
            .into_iter()
            .filter(|(_, p)| match d.tool {
                Tool::BoxSelect => p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y,
                _ => point_in_polygon(*p, &d.points),
            })
            .map(|(i, _)| i)
            .collect();
        apply_selection(&mut self.doc, &hits, d.op);
    }

    fn select_bones_in_shape(&mut self, op: SelectOp, mut contains: impl FnMut(Vec2) -> bool) {
        let Some(id) = self.doc.active else { return };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let world = self.doc.scene.world_matrices()[idx];
        let cam = self.camera.clone();
        let vp = self.viewport;
        let Some(arm) = self.doc.active_armature() else {
            return;
        };
        let deform = if self.doc.pose_mode {
            Some(arm.deform_matrices_hierarchical())
        } else {
            None
        };
        let mut hits = Vec::new();
        for (i, b) in arm.bones.iter().enumerate() {
            let (head, tail) = if let Some(mats) = deform.as_ref() {
                let dir = b.direction();
                let rest_rot = if dir.length_squared() < 1e-10 {
                    glam::Quat::IDENTITY
                } else {
                    glam::Quat::from_rotation_arc(glam::Vec3::Y, dir)
                };
                let rest = glam::Mat4::from_rotation_translation(rest_rot, b.head_v());
                let posed = mats[i] * rest;
                (
                    posed.transform_point3(glam::Vec3::ZERO),
                    posed.transform_point3(glam::Vec3::new(0.0, b.length(), 0.0)),
                )
            } else {
                (b.head_v(), b.tail_v())
            };
            let mid = world.transform_point3((head + tail) * 0.5);
            let head_world = world.transform_point3(head);
            let tail_world = world.transform_point3(tail);
            let mut inside = false;
            for point in [mid, head_world, tail_world] {
                if let Some(px) = crate::tools::project_px(&cam, vp, point) {
                    if contains(px) {
                        inside = true;
                        break;
                    }
                }
            }
            if inside {
                hits.push(i as u32);
            }
        }
        if let Some(arm) = self.doc.active_armature_mut() {
            match op {
                SelectOp::Replace => arm.selected.clear(),
                SelectOp::Extend | SelectOp::Subtract => {}
            }
            for i in hits {
                if op == SelectOp::Subtract {
                    arm.selected.remove(&i);
                } else {
                    arm.selected.insert(i);
                    arm.active = Some(i);
                }
            }
            if arm
                .active
                .map_or(false, |active| !arm.selected.contains(&active))
            {
                arm.active = arm.selected.iter().next_back().copied();
            }
            if op == SelectOp::Replace && arm.selected.is_empty() {
                arm.active = None;
            }
        }
        self.doc.touch();
    }

    fn finish_edit_sel_rect(&mut self, lo: Vec2, hi: Vec2, extend: bool) {
        let Some(session) = self.doc.edit.as_mut() else {
            return;
        };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else {
            return;
        };
        let world = self.doc.scene.world_matrices()[idx];
        let cam = self.camera.clone();
        let vp = self.viewport;
        session.select_in_rect(
            world,
            &move |w| crate::tools::project_px(&cam, vp, w).map(|p| (p.x, p.y)),
            (lo.x, lo.y),
            (hi.x, hi.y),
            extend,
        );
    }

    fn sel_shape(&self) -> Option<SelShape> {
        let d = self.sel_drag.as_ref()?;
        let cur = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        match d.tool {
            Tool::BoxSelect => Some(SelShape::Box {
                a: Vec2::new(d.start.0 as f32, d.start.1 as f32),
                b: cur,
            }),
            Tool::Lasso => Some(SelShape::Lasso(d.points.clone())),
            _ => None,
        }
    }

    // ------------------------------------------------------------------ modal transforms
    fn start_modal(&mut self, kind: ModalKind, checkpoint: bool) {
        if self.modal.is_some() {
            return;
        }
        match Modal::start(
            kind,
            &mut self.doc,
            &self.camera,
            self.viewport,
            self.cursor,
            self.ts.pivot,
            checkpoint,
        ) {
            Some(m) => {
                self.modal = Some(m);
                self.update_modal();
            }
            None => self.status("Select an object first"),
        }
    }

    fn update_modal(&mut self) {
        let ctrl = self.ctrl();
        if let Some(m) = self.modal.as_mut() {
            m.update(
                &mut self.doc,
                &self.camera,
                self.viewport,
                self.cursor,
                &self.ts.snap,
                ctrl,
            );
        }
    }

    fn end_modal(&mut self, cancel: bool) {
        if let Some(m) = self.modal.take() {
            if cancel {
                m.cancel(&mut self.doc);
            } else {
                m.confirm(&mut self.doc);
                // Auto Key: record Loc/Rot/Scale for the selection after a transform.
                if self.doc.scene.animation.auto_key && !self.doc.in_edit_mode() {
                    self.insert_keyframe(true, true, true);
                }
            }
        }
    }

    fn modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Escape => self.end_modal(true),
            KeyCode::Enter | KeyCode::NumpadEnter => self.end_modal(false),
            _ => {
                let shift = self.mods.shift_key();
                if let Some(m) = self.modal.as_mut() {
                    m.key(code, shift);
                }
                self.update_modal();
            }
        }
    }

    // ------------------------------------------------------------------ files
    fn open_project(&mut self, path: PathBuf) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            self.status("Open from path is not available on WebAssembly".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let res = std::fs::read(&path)
                .map_err(err_string)
                .and_then(|b| project::from_bytes(&b).map_err(err_string));
            match res {
                Ok(scene) => {
                    self.doc = Document::from_scene(scene, Some(path.clone()));
                    self.physics_runtime = None;
                    self.recent.add(&path);
                    self.reset_view_state();
                    self.frame(false);
                    self.status(format!("Opened {}", path.display()));
                }
                Err(e) => self.status(format!("Open failed: {e}")),
            }
        }
    }

    /// Import every model (.fbx / .obj / .glb) inside a Unity `.unitypackage`.
    fn import_unitypackage(&mut self, path: &Path) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            self.status("Unity package import is not available on WebAssembly yet");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            match forge_io::load_unitypackage(path) {
                Ok((models, notes)) => {
                    if models.is_empty() {
                        let extra = notes.first().map(|n| format!(" ({n})")).unwrap_or_default();
                        self.status(format!(
                            "No importable models (.fbx/.obj/.glb) found in {}{extra}",
                            path.display()
                        ));
                        return;
                    }
                    let count = models.len();
                    let mut objects = 0usize;
                    for (_unity_path, scene) in models {
                        objects += scene.objects.len();
                        self.doc.import_scene(scene);
                    }
                    self.frame(false);
                    let skipped = if notes.is_empty() {
                        String::new()
                    } else {
                        format!(" · {} asset(s) skipped", notes.len())
                    };
                    self.status(format!(
                        "Imported {count} model(s), {objects} object(s) from {}{skipped}",
                        path.display()
                    ));
                }
                Err(e) => self.status(format!("Import failed: {e:#}")),
            }
        }
    }

    /// Import an Alembic cache: meshes (with per-frame sequences) and animated transforms.
    fn import_alembic(&mut self, path: &Path) {
        let fps = self.doc.scene.animation.fps as f32;
        match forge_io::load_alembic(path, fps, 1.0) {
            Ok(imp) => {
                let count = imp.objects.len();
                let mut scene = forge_core::Scene::default();
                let mut extras = Vec::new();
                for item in imp.objects {
                    extras.push((item.sequence, item.keys));
                    scene.objects.push(item.object);
                }
                self.doc.import_scene(scene);
                let total = self.doc.scene.objects.len();
                let mut frames = 0usize;
                for (k, (sequence, keys)) in extras.into_iter().enumerate() {
                    let id = self.doc.scene.objects[total - count + k].id;
                    if let Some(seq) = sequence {
                        frames = frames.max(seq.len());
                        self.doc.scene.mesh_caches.insert(id, seq);
                    }
                    if !keys.is_empty() {
                        let mut anim = forge_core::ObjectAnimation::default();
                        anim.keys = keys;
                        anim.sort();
                        self.doc.scene.animation.objects.insert(id, anim);
                    }
                }
                if imp.end_frame > imp.start_frame {
                    self.doc.scene.animation.frame_start = imp.start_frame;
                    self.doc.scene.animation.frame_end = imp.end_frame;
                    let current = self.doc.scene.animation.frame_current;
                    self.doc.scene.animation.set_frame(current);
                }
                self.doc.touch();
                self.apply_animation_frame();
                self.frame(false);
                let notes = if imp.notes.is_empty() { String::new() } else { format!(" · {}", imp.notes.join("; ")) };
                self.status(format!(
                    "Imported {count} object(s) from {} ({frames} cached frame(s)){notes}",
                    path.display()
                ));
            }
            Err(e) => self.status(format!("Alembic import failed: {e:#}")),
        }
    }

    fn import_path(&mut self, path: &Path) {
        let ext = ext_of(path);
        if ext == "unitypackage" {
            self.import_unitypackage(path);
            return;
        }
        if ext == "abc" {
            self.import_alembic(path);
            return;
        }
        let res = match ext.as_str() {
            "glb" | "gltf" => forge_io::load_gltf(path),
            "obj" => forge_io::load_obj(path),
            "fbx" => forge_io::load_fbx(path),
            _ => {
                self.status(format!("Unsupported file type: .{ext}"));
                return;
            }
        };
        match res {
            Ok(scene) => {
                let n = scene.objects.len();
                let has_skin = scene.objects.iter().any(|o| o.skin.is_some());
                self.doc.import_scene(scene);
                // Show the imported rest / bind pose. Apply skinning so the mesh
                // matches the bones, but do NOT evaluate animation keys — that was
                // forcing the first frame of the clip onto every bone.
                // Scrubbing the timeline or pressing play still applies animation.
                if has_skin {
                    self.doc.apply_all_skinning();
                }
                self.frame(false);
                self.status(format!("Imported {n} object(s) from {}", path.display()));
            }
            Err(e) => self.status(format!("Import failed: {e:#}")),
        }
    }

    fn load_hdri(&mut self, path: &Path) {
        match forge_io::load_environment(path) {
            Ok(env) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer.set_environment(&env);
                }
                self.status(format!("Environment: {}", path.display()));
            }
            Err(e) => self.status(format!("HDRI failed: {e:#}")),
        }
    }

    /// Drag & drop: decide by extension.
    fn handle_path(&mut self, path: PathBuf) {
        match ext_of(&path).as_str() {
            "forge" => {
                if self.confirm_discard() {
                    self.open_project(path);
                }
            }
            "hdr" => self.load_hdri(&path),
            _ => self.import_path(&path),
        }
    }

    fn save(&mut self, save_as: bool) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = save_as;
            self.status("Save is not available on WebAssembly yet".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = match (&self.doc.path, save_as) {
                (Some(p), false) => p.clone(),
                _ => {
                    let Some(mut p) = rfd::FileDialog::new()
                        .add_filter("BlendCraft project", &["forge"])
                        .set_file_name("untitled.forge")
                        .save_file()
                    else {
                        return;
                    };
                    if p.extension().is_none() {
                        p.set_extension("forge");
                    }
                    p
                }
            };
            // Edit Mode owns a working mesh copy. Commit it before serializing so
            // UV edits and geometry changes cannot be omitted from the project.
            self.doc.sync_edit_mesh();
            // Weight Paint temporarily overwrites vertex colors with the heat map.
            // Restore real colors (and keep groups/skin) so paintings save correctly.
            let was_weight = self.doc.weight_paint_mode;
            if was_weight {
                self.doc.exit_weight_paint_mode();
            }
            // Sync any painted vertex groups into skin before writing the file.
            let skinned: Vec<u64> = self
                .doc
                .scene
                .objects
                .iter()
                .filter(|o| o.skin.is_some())
                .map(|o| o.id)
                .collect();
            for id in skinned {
                self.doc.sync_skin_from_vertex_groups(id);
            }
            let result = project::to_bytes(&self.doc.scene)
                .and_then(|b| std::fs::write(&path, b).map_err(Into::into));
            if was_weight {
                let _ = self.doc.enter_weight_paint_mode();
            }
            match result {
                Ok(()) => {
                    self.doc.path = Some(path.clone());
                    self.doc.dirty = false;
                    self.recent.add(&path);
                    self.status(format!(
                        "Saved {} (vertex colors, weight groups & skin bindings included)",
                        path.display()
                    ));
                }
                Err(e) => self.status(format!("Save failed: {e:#}")),
            }
        }
    }

    fn export(&mut self, format: &str, selected_only: bool) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (format, selected_only);
            self.status("Export is not available on WebAssembly yet".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (filter, ext, default) = match format {
                "glb" => ("glTF Binary", "glb", "model.glb"),
                "fbx" => ("Autodesk FBX", "fbx", "model.fbx"),
                "abc" => ("Alembic", "abc", "scene.abc"),
                "usda" => ("USD ASCII", "usda", "scene.usda"),
                _ => ("Wavefront OBJ", "obj", "model.obj"),
            };
            let Some(mut p) = rfd::FileDialog::new()
                .add_filter(filter, &[ext])
                .set_file_name(default)
                .save_file()
            else {
                return;
            };
            if p.extension().is_none() {
                p.set_extension(ext);
            }
            let only = selected_only.then_some(&self.doc.selection);
            let res = match format {
                "glb" => forge_io::export_glb(&self.doc.scene, &p, only),
                "fbx" => forge_io::export_fbx(&self.doc.scene, &p, only),
                "abc" => forge_io::export_alembic(&self.doc.scene, &p, only),
                "usda" => forge_io::export_usda(&self.doc.scene, &p, only),
                _ => forge_io::export_obj(&self.doc.scene, &p, only),
            };
            match res {
                Ok(()) => self.status(format!("Exported {}", p.display())),
                Err(e) => self.status(format!("Export failed: {e:#}")),
            }
        }
    }

    // ------------------------------------------------------------------ actions
    fn apply(&mut self, a: Action, el: &ActiveEventLoop) {
        use Action as A;
        match a {
            A::New => {
                if self.confirm_discard() {
                    self.doc = Document::new();
                    self.physics_runtime = None;
                    self.reset_view_state();
                    self.frame(false);
                }
            }
            A::Open => {
                #[cfg(target_arch = "wasm32")]
                {
                    self.status("Open file is not available on WebAssembly yet".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if self.confirm_discard() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("BlendCraft project", &["forge"])
                            .pick_file()
                        {
                            self.open_project(p);
                        }
                    }
                }
            }
            A::OpenRecent(p) => {
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = p;
                    self.status("Open recent is not available on WebAssembly yet".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if !p.exists() {
                        self.recent.remove(&p);
                        self.status(format!("File no longer exists: {}", p.display()));
                    } else if self.confirm_discard() {
                        self.open_project(p);
                    }
                }
            }
            A::ClearRecent => self.recent.clear(),
            A::Save => self.save(false),
            A::SaveAs => self.save(true),
            A::Import => {
                #[cfg(target_arch = "wasm32")]
                {
                    self.status("Import is not available on WebAssembly yet".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("3D models", &["glb", "gltf", "obj", "fbx", "abc", "unitypackage"])
                        .pick_file()
                    {
                        self.import_path(&p);
                    }
                }
            }
            A::ExportGlb => self.export("glb", false),
            A::ExportObj => self.export("obj", false),
            A::ExportGlbSelected => self.export("glb", true),
            A::ExportObjSelected => self.export("obj", true),
            A::ExportFbx => self.export("fbx", false),
            A::ExportFbxSelected => self.export("fbx", true),
            A::ExportAlembic => self.export("abc", false),
            A::ExportAlembicSelected => self.export("abc", true),
            A::ExportUsda => self.export("usda", false),
            A::ExportUsdaSelected => self.export("usda", true),
            A::Quit => {
                if self.confirm_discard() {
                    el.exit();
                }
            }

            A::Undo => {
                // Abort any in-progress drag first (without double-undo)
                if self.edit_xform.is_some() {
                    self.cancel_edit_xform();
                    self.status("Undo");
                    return;
                }
                if self.edit_gizmo_drag.is_some() {
                    // restore from checkpoint taken at drag start
                    self.edit_gizmo_drag = None;
                    if self.doc.undo() {
                        self.status("Undo");
                    }
                    return;
                }
                if self.gizmo.dragging() {
                    self.gizmo.end();
                    // gizmo already checkpointed at begin — undo the transform
                    if self.doc.undo() {
                        self.status("Undo");
                    }
                    return;
                }
                if self.doc.undo() {
                    self.status(format!("Undo ({} left)", self.doc.undo_len()));
                } else {
                    self.status("Nothing to undo");
                }
            }
            A::Redo => {
                if self.doc.redo() {
                    self.status(format!("Redo ({} left)", self.doc.redo_len()));
                } else {
                    self.status("Nothing to redo");
                }
            }
            A::Duplicate => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_duplicate();
                    self.status("Duplicated faces");
                } else if self.doc.duplicate(false) {
                    self.start_modal(ModalKind::Move, false);
                }
            }
            A::DuplicateLinked => {
                if self.doc.duplicate(true) {
                    self.start_modal(ModalKind::Move, false);
                }
            }
            A::Delete => {
                if self.doc.pose_mode {
                    self.status("Switch to Armature Edit Mode to delete bones");
                } else if self.doc.armature_edit && self.doc.active_armature().is_some() {
                    self.doc.armature_delete_bones();
                    self.status("Deleted bones");
                } else if self.doc.edit.is_some() {
                    self.doc.edit_delete();
                    self.status("Deleted selection");
                } else {
                    self.doc.delete_selected();
                }
            }
            A::Extrude => {
                if self.doc.armature_edit && self.doc.active_armature().is_some() {
                    self.start_bone_extrude();
                } else if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Extrude: select verts/edges/faces first (1/2/3 for domain, click components)");
                    } else {
                        self.doc.edit_extrude(0.25);
                        self.status("Extruded (E)");
                    }
                } else if self.doc.active_armature().is_some() {
                    self.status("Press Tab for Armature Edit Mode, then E");
                } else {
                    self.status("Extrude: Tab into Edit Mode first");
                }
            }
            A::InsetFaces => {
                if self.doc.in_edit_mode() {
                    let ok = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| {
                            !e.selection.faces.is_empty()
                                || (!e.selection.verts.is_empty()
                                    && e.selection.domain == forge_core::EditDomain::Vertex)
                        })
                        .unwrap_or(false);
                    if !ok {
                        self.status("Inset: switch to Face select (3) and select faces, or select face verts");
                    } else {
                        self.doc.edit_inset(0.25);
                        self.status("Inset Faces (I)");
                    }
                } else {
                    self.status("Inset: Tab into Edit Mode first");
                }
            }
            A::Bevel => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Bevel: select edges/verts/faces first");
                    } else {
                        self.doc.edit_bevel(0.05);
                        self.status("Bevel (Ctrl+B)");
                    }
                } else {
                    self.status("Bevel: Tab into Edit Mode first");
                }
            }
            A::LoopCut => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.edges.is_empty() && e.selection.faces.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Loop Cut: select edges (2) or faces (3) first");
                    } else {
                        self.doc.edit_loop_cut();
                        self.status("Loop Cut (Ctrl+R)");
                    }
                } else {
                    self.status("Loop Cut: Tab into Edit Mode first");
                }
            }
            A::Knife => {
                if self.doc.in_edit_mode() {
                    // Bisect through selection center with view plane normal
                    let center = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection_center())
                        .unwrap_or(glam::Vec3::ZERO);
                    // Transform center to world
                    let world_center = if let Some(session) = self.doc.edit.as_ref() {
                        if let Some(idx) = self
                            .doc
                            .scene
                            .objects
                            .iter()
                            .position(|o| o.id == session.object_id)
                        {
                            let w = self.doc.scene.world_matrices()[idx];
                            w.transform_point3(center)
                        } else {
                            center
                        }
                    } else {
                        center
                    };
                    let normal = (self.camera.target - self.camera.eye()).normalize_or_zero();
                    // Convert plane into object local space
                    if let Some(session) = self.doc.edit.as_ref() {
                        if let Some(idx) = self
                            .doc
                            .scene
                            .objects
                            .iter()
                            .position(|o| o.id == session.object_id)
                        {
                            let inv = self.doc.scene.world_matrices()[idx].inverse();
                            let local_pt = inv.transform_point3(world_center);
                            let local_n = inv.transform_vector3(normal).normalize_or_zero();
                            self.doc.edit_knife_bisect(local_pt, local_n);
                            self.status("Knife bisect (view plane)");
                        }
                    }
                } else {
                    self.status("Knife only works in Edit Mode");
                }
            }
            A::PolyBuild => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_poly_build(0.25);
                    self.status("Poly Build");
                } else {
                    self.status("Poly Build only works in Edit Mode (select edge/verts)");
                }
            }
            A::Spin => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_spin(8, 360.0);
                    self.status("Spin");
                } else {
                    self.status("Spin only works in Edit Mode");
                }
            }
            A::SmoothVerts => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Smooth: select vertices first (1)");
                    } else {
                        self.doc.edit_smooth(0.5);
                        self.status("Smooth");
                    }
                } else {
                    self.status("Smooth: Tab into Edit Mode first");
                }
            }
            A::EdgeSlide => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Edge Slide: select edges first (2)");
                    } else {
                        self.doc.edit_edge_slide(0.1);
                        self.status("Edge Slide (G G)");
                    }
                } else {
                    self.status("Edge Slide: Tab into Edit Mode first");
                }
            }
            A::ShrinkFatten => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Shrink/Fatten: select verts/faces first");
                    } else {
                        self.doc.edit_shrink_fatten(0.1);
                        self.status("Shrink/Fatten (Alt+S)");
                    }
                } else {
                    self.status("Shrink/Fatten: Tab into Edit Mode first");
                }
            }
            A::Shear => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Shear: select verts first");
                    } else {
                        self.doc.edit_shear(0, 0.25);
                        self.status("Shear (Ctrl+Shift+Alt+S)");
                    }
                } else {
                    self.status("Shear: Tab into Edit Mode first");
                }
            }
            A::RipRegion => {
                if self.doc.in_edit_mode() {
                    let empty = self
                        .doc
                        .edit
                        .as_ref()
                        .map(|e| e.selection.is_empty())
                        .unwrap_or(true);
                    if empty {
                        self.status("Rip: select verts first");
                    } else {
                        self.doc.edit_rip();
                        self.status("Rip Region (V)");
                    }
                } else {
                    self.status("Rip: Tab into Edit Mode first");
                }
            }
            A::Subdivide => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty { self.status("Subdivide: select vertices, edges, or faces first"); }
                    else { self.doc.edit_subdivide(); self.status("Subdivided selected geometry"); }
                } else { self.status("Subdivide: enter Edit Mode first"); }
            }
            A::FlipNormals => {
                if self.doc.in_edit_mode() { self.doc.edit_flip_normals(); self.status("Flipped face normals"); }
                else { self.status("Flip Normals: enter Edit Mode first"); }
            }
            A::MergeVerts => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_merge_verts();
                    self.status("Merged vertices");
                }
            }
            A::WeldVerts => {
                if self.doc.in_edit_mode() {
                    let selected = self.doc.edit.as_ref().map(|e| e.selected_vert_indices().len()).unwrap_or(0);
                    if selected < 2 {
                        self.status("Merge by Distance: select at least two vertices first");
                    } else {
                        self.doc.edit_weld_verts(1.0e-5);
                        self.status("Merged coincident selected vertices");
                    }
                } else {
                    self.status("Merge by Distance: enter Edit Mode first");
                }
            }
            A::Rename => {
                if let Some(i) = self.doc.active_index() {
                    self.ui.renaming = Some(Rename::Object(self.doc.scene.objects[i].id));
                    self.ui.rename_buf = self.doc.scene.objects[i].name.clone();
                    self.ui.rename_focus = true;
                }
            }
            A::MakeSingleUser => self.doc.make_single_user(),

            A::SelectAll => {
                if self.doc.uv_mode {
                    self.doc.uv_select_all();
                } else if self.doc.armature_edit || self.doc.pose_mode {
                    if let Some(a) = self.doc.active_armature_mut() {
                        a.selected = (0..a.bones.len() as u32).collect();
                        a.active = a.selected.iter().next_back().copied();
                    }
                    self.doc.touch();
                } else if self.doc.in_edit_mode() {
                    self.doc.edit_select_all();
                } else {
                    self.doc.select_all();
                }
            }
            A::SelectNone => {
                if self.doc.uv_mode {
                    self.doc.uv_select_none();
                } else if self.doc.armature_edit || self.doc.pose_mode {
                    if let Some(a) = self.doc.active_armature_mut() {
                        a.selected.clear();
                        a.active = None;
                    }
                    self.doc.touch();
                } else if self.doc.in_edit_mode() {
                    self.doc.edit_select_none();
                } else {
                    self.doc.select_none();
                }
            }
            A::SelectInvert => {
                if self.doc.armature_edit || self.doc.pose_mode {
                    if let Some(a) = self.doc.active_armature_mut() {
                        let old = a.selected.clone();
                        a.selected = (0..a.bones.len() as u32)
                            .filter(|i| !old.contains(i))
                            .collect();
                        a.active = a.selected.iter().next_back().copied();
                    }
                    self.doc.touch();
                } else if self.doc.in_edit_mode() {
                    self.doc.edit_invert_selection();
                } else {
                    self.doc.select_invert();
                }
            }
            A::SelectRandom => self.doc.select_random(0.5),
            A::SelectChildren => self.doc.select_children(),
            A::SelectParent => self.doc.select_parent(),
            A::SelectSameType => self.doc.select_same(SameKind::Type),
            A::SelectSameCollection => self.doc.select_same(SameKind::Collection),
            A::SelectSameMaterial => self.doc.select_same(SameKind::Material),
            A::SelectLinked => self.doc.select_same(SameKind::Linked),

            A::SetTool(t) => {
                if t != Tool::Annotate {
                    self.annotate_drag_active = false;
                    self.annotate_stroke = None;
                }
                self.ts.tool = t;
                self.sel_drag = None;
            }
            A::CycleTool => {
                self.ts.tool = match self.ts.tool {
                    Tool::Move => Tool::Rotate,
                    Tool::Rotate => Tool::Scale,
                    Tool::Scale => Tool::Select,
                    _ => Tool::Move,
                };
            }
            A::ToggleSnap => self.ts.snap.enabled = !self.ts.snap.enabled,
            A::ModalMove => {
                if self.doc.armature_edit || self.doc.pose_mode {
                    self.start_bone_xform(EditXformKind::Move, false);
                } else if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Move);
                } else {
                    self.start_modal(ModalKind::Move, true);
                }
            }
            A::ModalRotate => {
                if self.doc.armature_edit || self.doc.pose_mode {
                    self.start_bone_pose_rotate();
                } else if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Rotate);
                } else {
                    self.start_modal(ModalKind::Rotate, true);
                }
            }
            A::ModalScale => {
                if self.doc.armature_edit || self.doc.pose_mode {
                    self.start_bone_xform(EditXformKind::Scale, false);
                } else if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Scale);
                } else {
                    self.start_modal(ModalKind::Scale, true);
                }
            }

            A::Hide => self.doc.hide_selected(),
            A::HideUnselected => self.doc.hide_unselected(),
            A::ShowAll => self.doc.show_all(),
            A::ParentSet => match self.doc.parent_set() {
                Ok(()) => self.status("Parented selection to the active object"),
                Err(e) => self.status(e),
            },
            A::ParentClear => self.doc.parent_clear(false),
            A::ParentClearKeep => self.doc.parent_clear(true),
            A::NewCollection => {
                let id = self.doc.new_collection();
                self.ui.renaming = Some(Rename::Collection(id));
                self.ui.rename_buf = self
                    .doc
                    .scene
                    .collections
                    .iter()
                    .find(|c| c.id == id)
                    .map(|c| c.name.clone())
                    .unwrap_or_default();
                self.ui.rename_focus = true;
            }
            A::OpenMoveToCollection => self.ui.show_move_popup = true,
            A::MoveToCollection(c) => self.doc.move_selected_to_collection(c),
            A::Join => match self.doc.join() {
                Ok(()) => self.status("Joined meshes"),
                Err(e) => self.status(e),
            },
            A::ShadeSmooth => {
                self.doc.shade_smooth_selected();
                self.status("Shade Smooth");
            }
            A::ShadeFlat => {
                self.doc.shade_flat_selected();
                self.status("Shade Flat");
            }
            A::SeparateLoose => {
                let n = self.doc.separate_loose();
                self.status(if n == 0 {
                    "Nothing to separate (mesh is one piece)".to_string()
                } else {
                    format!("Separated {n} new object(s)")
                });
            }
            A::SeparateMaterial => {
                let n = self.doc.separate_material();
                self.status(if n == 0 {
                    "Nothing to separate (needs a mesh with several materials)".to_string()
                } else {
                    format!("Separated {n} new object(s)")
                });
            }
            A::SetOrigin(m) => self.doc.set_origin(m),
            A::GeometryToOrigin => self.doc.geometry_to_origin(),
            A::Apply { loc, rot, scale } => {
                if self.doc.in_edit_mode() {
                    // Ctrl+A is "Apply transforms" in Object Mode; in Edit Mode treat as Select All
                    self.doc.edit_select_all();
                    self.status("Select all (Edit Mode)");
                } else {
                    self.doc.apply_transform(loc, rot, scale);
                }
            }
            A::ClearLocation => self.doc.clear_transform(true, false, false),
            A::ClearRotation => self.doc.clear_transform(false, true, false),
            A::ClearScale => self.doc.clear_transform(false, false, true),
            A::Align(axis, mode, to) => self.doc.align(axis, mode, to),
            A::SnapSelToCursor => self.doc.selection_to_cursor(),
            A::SnapSelToActive => self.doc.selection_to_active(),
            A::SnapCursorToSel => self.doc.cursor_to_selection(),
            A::SnapCursorToActive => self.doc.cursor_to_active(),
            A::SnapCursorToOrigin => self.doc.cursor_to_origin(),
            A::AddConstraint(k) => {
                if let Some(i) = self.doc.active_index() {
                    self.doc.checkpoint();
                    self.doc.scene.objects[i]
                        .constraints
                        .push(Constraint::new(k));
                }
            }
            A::ClearConstraints => {
                if let Some(i) = self.doc.active_index() {
                    self.doc.checkpoint();
                    self.doc.scene.objects[i].constraints.clear();
                }
            }

            A::Add(p) => self.doc.add_primitive(p),
            A::AddKind(k) => self.doc.add_kind(k),
            A::AddImagePlane => {
                #[cfg(target_arch = "wasm32")]
                {
                    self.status("Add Image Plane is not available on WebAssembly yet".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg"])
                        .pick_file()
                    {
                        match forge_io::load_image(&p) {
                            Ok(t) => {
                                let name = t.name.clone();
                                self.doc.add_image_plane(&name, t);
                                self.status(format!("Added image plane {name}"));
                            }
                            Err(e) => self.status(format!("Image failed: {e:#}")),
                        }
                    }
                }
            }
            A::LoadHdri => {
                #[cfg(target_arch = "wasm32")]
                {
                    self.status("Load HDRI is not available on WebAssembly yet".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("HDR image", &["hdr", "png", "jpg", "jpeg"])
                        .pick_file()
                    {
                        self.load_hdri(&p);
                    }
                }
            }
            A::ResetSky => {
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer
                        .set_environment(&EnvironmentMap::procedural_sky(512, 256));
                }
            }

            A::FrameAll => self.frame(false),
            A::FrameSelected => self.frame(true),
            A::CameraView => self.toggle_camera_view(),
            A::SetShading(m) => self.settings.shading = m,
            A::CycleWire => self.settings.wireframe = self.settings.wireframe.next(),
            A::ToggleBackground => self.settings.show_background = !self.settings.show_background,
            A::ExposureDown => self.settings.exposure = (self.settings.exposure / 1.15).max(0.05),
            A::ExposureUp => self.settings.exposure = (self.settings.exposure * 1.15).min(20.0),
            A::EnvRotate(d) => self.settings.env_rotation += 0.1 * d as f32,
            A::RenderImage => self.render_image(),
            A::RunPythonScript => self.run_python_script(),
            A::OpenPythonScript => self.open_python_script(),
            A::SavePythonScript => self.save_python_script(),
            A::RenderCompositor => self.render_compositor(),
            A::SaveCompositor => self.save_compositor(),
            A::RenderAnimation => self.render_animation(),
            A::InsertKeyframeLocRotScale => self.insert_keyframe(true, true, true),
            A::InsertKeyframeLocation => self.insert_keyframe(true, false, false),
            A::InsertKeyframeRotation => self.insert_keyframe(false, true, false),
            A::InsertKeyframeScale => self.insert_keyframe(false, false, true),
            A::DeleteKeyframe => self.delete_keyframe(),
            A::ToggleAnimationPlay => {
                self.doc.scene.animation.playing = !self.doc.scene.animation.playing;
                if self.doc.scene.animation.playing {
                    self.last_anim_tick = web_time::Instant::now();
                    self.status("Animation playing");
                } else {
                    self.status("Animation paused");
                }
            }
            A::ToggleAutoKey => {
                self.doc.scene.animation.auto_key = !self.doc.scene.animation.auto_key;
                self.status(if self.doc.scene.animation.auto_key {
                    "Auto Key ON"
                } else {
                    "Auto Key OFF"
                });
            }
            A::AnimScrub => self.apply_animation_frame(),

            A::ToggleShortcuts => self.ui.show_shortcuts = !self.ui.show_shortcuts,
            A::ShowAbout => self.ui.show_about = true,
            A::SavePaintImage => self.save_paint_image(),
            A::UvSaveMap => self.save_uv_map(),
            A::UvLoadMap => self.load_uv_map(),
            A::TogglePaintMode => match self.doc.toggle_paint_mode() {
                Ok(()) => {
                    if self.doc.paint_mode {
                        self.ts.tool = Tool::PaintBrush;
                        self.status("Texture Paint – LMB paint · Erase tool · Wheel radius");
                    } else {
                        self.ts.tool = Tool::Select;
                        self.status("Object Mode");
                    }
                }
                Err(e) => self.status(e),
            },
            A::ToggleVertexPaintMode => match self.doc.toggle_vertex_paint_mode() {
                Ok(()) => {
                    if self.doc.vertex_paint_mode {
                        self.ts.tool = Tool::VertexPaintDraw;
                        // Sensible brush size from mesh bounds so the first stroke is visible
                        if let Some(i) = self.doc.active_index() {
                            let mesh = &self.doc.scene.objects[i].mesh;
                            if !mesh.vertices.is_empty() {
                                let mut min = glam::Vec3::splat(f32::INFINITY);
                                let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
                                for v in &mesh.vertices {
                                    let p = glam::Vec3::from(v.position);
                                    min = min.min(p);
                                    max = max.max(p);
                                }
                                let extent = (max - min).length().max(1e-3);
                                // ~8% of diagonal — similar to Blender's default feel on a unit cube
                                self.ts.vpaint_radius = (extent * 0.08).clamp(0.02, 5.0);
                            }
                        }
                        self.status("Vertex Paint – LMB paint · adjust Radius/Color in header");
                    } else {
                        self.ts.tool = Tool::Select;
                        self.status("Object Mode");
                    }
                }
                Err(e) => self.status(e),
            },
            A::ToggleWeightPaintMode => match self.doc.toggle_weight_paint_mode() {
                Ok(()) => {
                    if self.doc.weight_paint_mode {
                        self.ts.tool = Tool::WeightPaintDraw;
                        if let Some(i) = self.doc.active_index() {
                            let mesh = &self.doc.scene.objects[i].mesh;
                            if !mesh.vertices.is_empty() {
                                let mut min = glam::Vec3::splat(f32::INFINITY);
                                let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
                                for v in &mesh.vertices {
                                    let p = glam::Vec3::from(v.position);
                                    min = min.min(p);
                                    max = max.max(p);
                                }
                                let extent = (max - min).length().max(1e-3);
                                self.ts.wpaint_radius = (extent * 0.08).clamp(0.02, 5.0);
                            }
                        }
                        self.status("Weight Paint – LMB paint · Weight/Radius/Group in header");
                    } else {
                        self.ts.tool = Tool::Select;
                        self.status("Object Mode");
                    }
                }
                Err(e) => self.status(e),
            },
            A::ToggleXMirror => {
                // Cycle: toggle X (Blender primary default)
                self.doc.mesh_symmetry = self.doc.mesh_symmetry.toggle_x();
                self.doc.version = self.doc.version.wrapping_add(1);
                self.status(format!(
                    "Mesh Symmetry X={} Y={} Z={}",
                    self.doc.mesh_symmetry.x, self.doc.mesh_symmetry.y, self.doc.mesh_symmetry.z
                ));
            }
            A::ToggleMeshSymmetryX => {
                self.doc.mesh_symmetry = self.doc.mesh_symmetry.toggle_x();
                self.doc.version = self.doc.version.wrapping_add(1);
                self.status(format!("Mesh Symmetry X={}", self.doc.mesh_symmetry.x));
            }
            A::ToggleMeshSymmetryY => {
                self.doc.mesh_symmetry = self.doc.mesh_symmetry.toggle_y();
                self.doc.version = self.doc.version.wrapping_add(1);
                self.status(format!("Mesh Symmetry Y={}", self.doc.mesh_symmetry.y));
            }
            A::ToggleMeshSymmetryZ => {
                self.doc.mesh_symmetry = self.doc.mesh_symmetry.toggle_z();
                self.doc.version = self.doc.version.wrapping_add(1);
                self.status(format!("Mesh Symmetry Z={}", self.doc.mesh_symmetry.z));
            }
            A::SetMirrorAxis(axis) => {
                self.doc.mesh_symmetry = axis.to_sym();
                self.doc.version = self.doc.version.wrapping_add(1);
                self.status(format!(
                    "Mesh Symmetry X={} Y={} Z={}",
                    self.doc.mesh_symmetry.x, self.doc.mesh_symmetry.y, self.doc.mesh_symmetry.z
                ));
            }
            A::ToggleUvMode => {
                match self.doc.toggle_uv_mode() {
                    Ok(()) => {
                        if self.doc.uv_mode {
                            self.ts.tool = Tool::UvSelect;
                            // Auto smart unwrap if UVs look unset (all near zero)
                            if let Some(id) = self.doc.active {
                                if let Some(obj) =
                                    self.doc.scene.objects.iter().find(|o| o.id == id)
                                {
                                    let max_u = obj
                                        .mesh
                                        .vertices
                                        .iter()
                                        .map(|v| v.uv[0].abs() + v.uv[1].abs())
                                        .fold(0.0f32, f32::max);
                                    if max_u < 1e-5 {
                                        self.doc.uv_unwrap_smart();
                                    }
                                }
                            }
                            self.status(
                                "UV Editing – unwrap tools in toolbar · select & drag in UV Editor",
                            );
                        } else {
                            self.ts.tool = Tool::Select;
                            self.status("Object Mode");
                        }
                    }
                    Err(e) => self.status(e),
                }
            }
            A::UvUnwrapSmart => {
                self.doc.uv_unwrap_smart();
                self.status("Smart UV Project");
            }
            A::UvUnwrapCube => {
                self.doc.uv_unwrap_cube();
                self.status("Cube Projection");
            }
            A::UvUnwrapPlanar => {
                self.doc.uv_unwrap_planar();
                self.status("Project from View (planar)");
            }
            A::UvPack => {
                self.doc.uv_pack();
                self.status("Pack Islands");
            }
            A::UvReset => {
                self.doc.uv_unwrap_reset();
                self.status("Reset UVs");
            }
            A::UvExportLayout => self.export_uv_layout(),
            A::ToggleSculptMode => {
                match self.doc.toggle_sculpt_mode() {
                    Ok(()) => {
                        if self.doc.sculpt_mode {
                            self.ts.tool = Tool::SculptDraw;
                            self.status("Sculpt: LMB paint · Alt+LMB / MMB / RMB orbit · Shift+pan · D dyntopo");
                        } else {
                            self.ts.tool = Tool::Select;
                            self.status("Object Mode");
                        }
                    }
                    Err(e) => self.status(e),
                }
            }
            A::ToggleEditMode => {
                // Drop any object-gizmo drag so it cannot move the whole object
                if self.gizmo.dragging() {
                    self.gizmo.end();
                }
                self.edit_xform = None;
                match self.doc.toggle_edit_mode() {
                    Ok(()) => {
                        if self.doc.armature_edit {
                            self.ts.tool = Tool::Select;
                            self.status(
                                "Armature Edit Mode — select bone, E extrude tip, X delete"
                                    .to_string(),
                            );
                        } else if self.doc.in_edit_mode() {
                            self.ts.tool = Tool::Select;
                            self.status("Edit Mode — select points, then G to move".to_string());
                        } else {
                            self.ts.tool = Tool::Select;
                            self.status("Object Mode".to_string());
                        }
                        self.frame(false);
                    }
                    Err(e) => self.status(e),
                }
            }
            A::TogglePoseMode => {
                if self.gizmo.dragging() {
                    self.gizmo.end();
                }
                self.bone_xform_drag = None;
                match self.doc.toggle_pose_mode() {
                    Ok(()) => {
                        self.ts.tool = Tool::Select;
                        self.status(if self.doc.pose_mode {
                            "Pose Mode — select bones and use G/R/S to transform"
                        } else {
                            "Object Mode"
                        });
                    }
                    Err(e) => self.status(e),
                }
            }
            A::EditDomainVertex => {
                self.doc.set_edit_domain(EditDomain::Vertex);
                self.status("Vertex select".to_string());
            }
            A::EditDomainEdge => {
                self.doc.set_edit_domain(EditDomain::Edge);
                self.status("Edge select".to_string());
            }
            A::EditDomainFace => {
                self.doc.set_edit_domain(EditDomain::Face);
                self.status("Face select".to_string());
            }
            A::ArmatureAddBone => {
                if self.doc.armature_edit {
                    self.doc.armature_add_bone();
                    self.status("Bone added");
                } else {
                    self.status("Enter Armature Edit Mode to add bones");
                }
            }
            A::ArmatureExtrude => {
                if self.doc.armature_edit {
                    self.start_bone_extrude();
                } else {
                    self.status("Press Tab for Armature Edit Mode, then Extrude");
                }
            }
            A::ArmatureDeleteBones => {
                if self.doc.armature_edit {
                    self.doc.armature_delete_bones();
                    self.status("Deleted bones");
                } else {
                    self.status("Enter Armature Edit Mode to delete bones");
                }
            }
            A::ArmatureClearPose => {
                if self.doc.pose_mode {
                    self.doc.armature_clear_pose();
                    self.status("Selected bone pose cleared");
                } else {
                    self.status("Enter Pose Mode to clear bone poses");
                }
            }
            A::ArmatureBindMeshes => match self.doc.bind_selected_to_armature() {
                Ok(n) => self.status(format!("Bound {n} mesh(es) to armature")),
                Err(e) => self.status(e),
            },
        }
    }

    /// Save UV coordinates as an independent, portable BlendCraft UV-map file.
    fn save_uv_map(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Saving UV maps is not available on WebAssembly yet");
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.doc.sync_edit_mesh();
            let Some(id) = self.doc.active else {
                self.status("Select a mesh before saving its UV map");
                return;
            };
            let Some(obj) = self.doc.scene.objects.iter().find(|o| o.id == id) else {
                self.status("Active object no longer exists");
                return;
            };
            if obj.mesh.vertices.is_empty() || obj.mesh.indices.is_empty() {
                self.status("Active mesh has no UV faces");
                return;
            }
            let file = UvMapFile {
                format: "BlendCraft UV Map".to_string(),
                version: 1,
                object_name: obj.name.clone(),
                vertex_count: obj.mesh.vertices.len(),
                positions: obj.mesh.vertices.iter().map(|v| v.position).collect(),
                indices: obj.mesh.indices.clone(),
                uvs: obj.mesh.vertices.iter().map(|v| v.uv).collect(),
            };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("BlendCraft UV Map", &["uvmap"])
                .set_file_name(&format!(
                    "{}.uvmap",
                    obj.name.replace('/', "_").replace('\\', "_")
                ))
                .save_file()
            else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("uvmap");
            }
            match serde_json::to_vec_pretty(&file) {
                Ok(bytes) => match std::fs::write(&path, bytes) {
                    Ok(()) => self.status(format!("Saved UV map to {}", path.display())),
                    Err(e) => self.status(format!("UV map save failed: {e}")),
                },
                Err(e) => self.status(format!("UV map serialization failed: {e}")),
            }
        }
    }

    /// Load a standalone UV-map file, requiring an exact mesh topology match.
    fn load_uv_map(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Loading UV maps is not available on WebAssembly yet");
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("BlendCraft UV Map", &["uvmap"])
                .pick_file()
            else {
                return;
            };
            let loaded: Result<UvMapFile, String> = std::fs::read(&path)
                .map_err(|e| format!("Could not read UV map: {e}"))
                .and_then(|bytes| {
                    serde_json::from_slice(&bytes).map_err(|e| format!("Invalid UV map file: {e}"))
                });
            let file = match loaded {
                Ok(f) => f,
                Err(e) => {
                    self.status(e);
                    return;
                }
            };
            if file.format != "BlendCraft UV Map" || file.version != 1 {
                self.status("Unsupported UV map file format or version");
                return;
            }
            let Some(id) = self.doc.active else {
                self.status("Select the target mesh before loading a UV map");
                return;
            };
            let current_mesh = self
                .doc
                .edit
                .as_ref()
                .filter(|e| e.object_id == id)
                .map(|e| e.mesh.clone())
                .or_else(|| {
                    self.doc
                        .scene
                        .objects
                        .iter()
                        .find(|o| o.id == id)
                        .map(|o| (*o.mesh).clone())
                });
            let Some(current_mesh) = current_mesh else {
                self.status("Active object no longer exists");
                return;
            };
            if current_mesh.vertices.len() != file.vertex_count
                || current_mesh.indices != file.indices
                || file.uvs.len() != file.vertex_count
                || file.positions.len() != file.vertex_count
                || current_mesh
                    .vertices
                    .iter()
                    .zip(file.positions.iter())
                    .any(|(v, p)| v.position != *p)
            {
                self.status("UV map does not match this mesh geometry/topology. Select the original mesh or matching duplicate.");
                return;
            }
            if file.uvs.iter().flatten().any(|v| !v.is_finite()) {
                self.status("UV map contains invalid coordinates");
                return;
            }
            self.doc.checkpoint();
            let mut mesh = current_mesh;
            for (vertex, uv) in mesh.vertices.iter_mut().zip(file.uvs.iter()) {
                vertex.uv = *uv;
            }
            if let Some(session) = self.doc.edit.as_mut().filter(|e| e.object_id == id) {
                session.mesh = mesh.clone();
            }
            if let Some(obj) = self.doc.scene.objects.iter_mut().find(|o| o.id == id) {
                obj.mesh = Arc::new(mesh);
                obj.primitive = None;
            }
            self.doc.touch();
            self.status(format!(
                "Loaded UV map '{}' from {}",
                file.object_name,
                path.display()
            ));
        }
    }

    /// Export the active UV map as a transparent PNG reference image.
    fn export_uv_layout(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("UV layout export is not available on WebAssembly yet");
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mesh = if let Some(session) = self.doc.edit.as_ref() {
                Some(session.mesh.clone())
            } else {
                self.doc
                    .active
                    .and_then(|id| self.doc.scene.objects.iter().find(|o| o.id == id))
                    .map(|o| (*o.mesh).clone())
            };
            let Some(mesh) = mesh else {
                self.status("Select a mesh before exporting its UV layout");
                return;
            };
            if mesh.indices.len() < 3 {
                self.status("The active mesh has no UV faces");
                return;
            }
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("uv-layout.png")
                .save_file()
            else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("png");
            }

            const SIZE: u32 = 1024;
            let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
            let mut internal_diagonals = std::collections::HashSet::new();
            for q in mesh.quad_faces() {
                let (a, b) = (q[1].min(q[3]), q[1].max(q[3]));
                internal_diagonals.insert((a, b));
            }
            let mut draw_line = |a: glam::Vec2, b: glam::Vec2| {
                if !a.is_finite() || !b.is_finite() {
                    return;
                }
                // Clip to the standard 0..1 UV tile, as Blender's UV layout export does.
                if a.x < 0.0
                    || a.x > 1.0
                    || a.y < 0.0
                    || a.y > 1.0
                    || b.x < 0.0
                    || b.x > 1.0
                    || b.y < 0.0
                    || b.y > 1.0
                {
                    return;
                }
                let ax = a.x * (SIZE - 1) as f32;
                let ay = (1.0 - a.y) * (SIZE - 1) as f32;
                let bx = b.x * (SIZE - 1) as f32;
                let by = (1.0 - b.y) * (SIZE - 1) as f32;
                let steps = ((bx - ax).abs().max((by - ay).abs())).ceil().max(1.0) as u32;
                for step in 0..=steps {
                    let t = step as f32 / steps as f32;
                    let x = (ax + (bx - ax) * t).round() as i32;
                    let y = (ay + (by - ay) * t).round() as i32;
                    for oy in -1..=1 {
                        for ox in -1..=1 {
                            let px = x + ox;
                            let py = y + oy;
                            if px < 0 || py < 0 || px >= SIZE as i32 || py >= SIZE as i32 {
                                continue;
                            }
                            let i = ((py as u32 * SIZE + px as u32) * 4) as usize;
                            rgba[i..i + 4].copy_from_slice(&[245, 205, 90, 255]);
                        }
                    }
                }
            };
            for tri in mesh.indices.chunks_exact(3) {
                if tri.iter().any(|&i| i as usize >= mesh.vertices.len()) {
                    continue;
                }
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let edge = (a.min(b), a.max(b));
                    if internal_diagonals.contains(&edge) {
                        continue;
                    }
                    draw_line(
                        glam::Vec2::from(mesh.vertices[a as usize].uv),
                        glam::Vec2::from(mesh.vertices[b as usize].uv),
                    );
                }
            }
            match image::save_buffer(&path, &rgba, SIZE, SIZE, image::ColorType::Rgba8) {
                Ok(()) => self.status(format!("Exported UV layout to {}", path.display())),
                Err(e) => self.status(format!("UV layout export failed: {e}")),
            }
        }
    }

    /// Save the active material's base-color paint image as a local PNG.
    fn save_paint_image(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Saving paint images is not available on WebAssembly yet");
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(id) = self.doc.active else {
                self.status("No active paint object");
                return;
            };
            let image = self
                .doc
                .scene
                .objects
                .iter()
                .find(|o| o.id == id)
                .and_then(|o| o.material.or(Some(0)))
                .and_then(|mi| self.doc.scene.materials.get(mi))
                .and_then(|m| m.base_color_tex)
                .and_then(|ti| self.doc.scene.textures.get(ti))
                .cloned();
            let Some(image) = image else {
                self.status("No paint image assigned");
                return;
            };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("paint-texture.png")
                .save_file()
            else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("png");
            }
            match image::save_buffer(
                &path,
                &image.rgba8,
                image.width,
                image.height,
                image::ColorType::Rgba8,
            ) {
                Ok(()) => self.status(format!("Saved paint image to {}", path.display())),
                Err(e) => self.status(format!("Could not save paint image: {e}")),
            }
        }
    }

    /// Capture the current 3D viewport (no UI) to a PNG the user chooses.

    /// The ONE resolution setting: Output X × Y × %.
    /// It drives the camera frame, the Cycles 3D view buffer, Render Image and
    /// Render Animation, so changing it changes all of them.
    fn render_resolution(&self) -> (u32, u32) {
        let pct = self.ui.render_percent.clamp(1, 200) as f32 / 100.0;
        let w = ((self.ui.render_width.max(64) as f32) * pct).round() as u32;
        let h = ((self.ui.render_height.max(64) as f32) * pct).round() as u32;
        (w.clamp(64, 7680), h.clamp(64, 4320))
    }

    fn insert_keyframe(&mut self, loc: bool, rot: bool, scl: bool) {
        if self.doc.in_edit_mode() || self.doc.sculpt_mode || self.doc.uv_mode {
            self.status("Keyframes: switch to Object Mode");
            return;
        }
        let frame = self.doc.scene.animation.frame_current;
        let ids: Vec<u64> = if self.doc.selection.is_empty() {
            self.doc.active.into_iter().collect()
        } else {
            self.doc.selection.iter().copied().collect()
        };
        if ids.is_empty() {
            self.status("Select an object to keyframe");
            return;
        }
        self.doc.checkpoint();
        let mut n = 0;
        for id in ids {
            let Some(idx) = self.doc.scene.index_of(id) else {
                continue;
            };
            let o = &self.doc.scene.objects[idx];
            let loc_v = loc.then_some(o.translation);
            let rot_v = rot.then_some(o.rotation);
            let scl_v = scl.then_some(o.scale);
            self.doc
                .scene
                .animation
                .objects
                .entry(id)
                .or_default()
                .insert_or_merge(frame, loc_v, rot_v, scl_v);
            n += 1;
        }
        self.doc.touch();
        self.status(format!(
            "Inserted keyframe at frame {frame} on {n} object(s)"
        ));
    }

    fn delete_keyframe(&mut self) {
        let frame = self.doc.scene.animation.frame_current;
        let ids: Vec<u64> = if self.doc.selection.is_empty() {
            self.doc.active.into_iter().collect()
        } else {
            self.doc.selection.iter().copied().collect()
        };
        self.doc.checkpoint();
        let mut n = 0;
        for id in ids {
            if let Some(oa) = self.doc.scene.animation.objects.get_mut(&id) {
                if oa.has_any_at(frame) {
                    oa.delete_at(frame);
                    n += 1;
                }
            }
        }
        self.doc.touch();
        self.apply_animation_frame();
        self.status(format!("Deleted keys at frame {frame} ({n} object(s))"));
    }

    /// Write evaluated animation onto object transforms for the current frame.
    fn apply_animation_frame(&mut self) {
        self.anim_subframe = 0.0;
        let frame = self.doc.scene.animation.frame_current as f32;
        self.eval_animation_at(frame);
        let current_frame = self.doc.scene.animation.frame_current;
        forge_core::apply_baked_frame(&mut self.doc.scene, current_frame);
        self.doc.touch();
    }

    /// True when a light is animated directly or sits under an animated parent.
    fn animation_moves_lights(&self) -> bool {
        let anim = &self.doc.scene.animation.objects;
        let animated = |id: u64| anim.get(&id).map_or(false, |a| !a.keys.is_empty());
        self.doc.scene.objects.iter().any(|o| {
            if !matches!(o.kind, ObjectKind::Light(_)) {
                return false;
            }
            let mut cur = Some(o.id);
            let mut guard = 0;
            while let Some(id) = cur {
                if animated(id) {
                    return true;
                }
                guard += 1;
                if guard > 64 {
                    break;
                }
                cur = self
                    .doc
                    .scene
                    .index_of(id)
                    .and_then(|i| self.doc.scene.objects[i].parent);
            }
            false
        })
    }

    /// Evaluate keyframes at a (possibly fractional) frame and write the object
    /// transforms. Does not bump the document version, so playback does not force
    /// a full GPU scene rebuild every tick.
    fn eval_animation_at(&mut self, frame: f32) {
        let keys: Vec<(
            u64,
            (Option<glam::Vec3>, Option<glam::Quat>, Option<glam::Vec3>),
        )> = self
            .doc
            .scene
            .animation
            .objects
            .iter()
            .map(|(id, oa)| (*id, oa.eval(frame)))
            .collect();
        for (id, (loc, rot, scl)) in keys {
            let Some(idx) = self.doc.scene.index_of(id) else {
                continue;
            };
            if self.doc.scene.rigid_bodies.get(&id).is_some_and(|body|
                body.enabled && body.body_type == forge_core::RigidBodyType::Active
                    && body.dynamic && !body.animated
            ) {
                continue;
            }
            let o = &mut self.doc.scene.objects[idx];
            if let Some(v) = loc {
                o.translation = v;
            }
            if let Some(v) = rot {
                o.rotation = v;
            }
            if let Some(v) = scl {
                o.scale = v;
            }
        }
        // Evaluate imported skeletal tracks after object transforms. Pose values are
        // deltas from the armature's imported rest pose; reset first so switching
        // clips or scrubbing before the first key cannot leave stale bone poses.
        let active_clip = self.doc.scene.animation.active_clip;
        let bone_tracks = active_clip
            .and_then(|index| self.doc.scene.animation.bone_tracks.get(index))
            .cloned().unwrap_or_default();
        let mut armature_ids: Vec<u64> = self.doc.scene.animation.bone_tracks.iter()
            .flat_map(|clip_tracks| clip_tracks.keys().copied()).collect();
        armature_ids.sort_unstable();
        armature_ids.dedup();
        for &armature_id in &armature_ids {
            let Some(idx) = self.doc.scene.index_of(armature_id) else { continue };
            let tracks = bone_tracks.get(&armature_id);
            if let ObjectKind::Armature(armature) = &mut self.doc.scene.objects[idx].kind {
                for (bone_index, bone) in armature.bones.iter_mut().enumerate() {
                    // Absolute-local (glTF) bones restore bind local TRS so unkeyed
                    // channels stay at rest; delta bones still reset to identity.
                    bone.reset_pose_to_rest();
                    if let Some(track) = tracks.and_then(|tracks| tracks.get(&(bone_index as u32))) {
                        let (loc, rot, scale) = track.eval(frame);
                        if let Some(v) = loc { bone.pose_loc = v.to_array(); }
                        if let Some(v) = rot { bone.pose_rot = v.normalize().to_array(); }
                        if let Some(v) = scale { bone.pose_scale = v.to_array(); }
                    }
                }
            }
        }
        if !armature_ids.is_empty() {
            self.doc.apply_all_skinning();
        }
        // Mesh sequence caches (fluid bakes, Alembic): swap in the mesh for this frame.
        if forge_core::apply_mesh_caches(&mut self.doc.scene, frame) {
            let runtime_in_sync = self.physics_runtime.is_some()
                && self.physics_runtime_doc_version == self.doc.version;
            self.doc.version += 1;
            if runtime_in_sync {
                self.physics_runtime_doc_version = self.doc.version;
            }
        }
    }

    fn render_image(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Render Image is not available on WebAssembly yet".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.gfx.is_none() {
                self.status("Renderer not ready");
                return;
            }
            let (format_name, extension, dialog_filter) = match self.ui.render_format {
                1 => ("JPEG", "jpg", "JPEG image"),
                2 => ("OpenEXR", "exr", "OpenEXR image"),
                _ => ("PNG", "png", "PNG image"),
            };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter(dialog_filter, &[extension])
                .set_file_name(&format!("render.{extension}"))
                .save_file()
            else { return; };
            path.set_extension(extension);
            let (w, h) = self.render_resolution();
            // Ensure objects sit on the current timeline frame.
            self.apply_animation_frame();
            if let Some(g) = self.gfx.as_mut() {
                g.renderer
                    .set_scene(&self.doc.scene, &highlights(&self.doc));
                if self.settings.engine == RenderEngine::Cycles {
                    g.renderer.sync_cycles(&self.doc.scene);
                }
            }
            self.uploaded = self.doc.version;
            let cam = self.render_camera();
            let mut settings = self.settings.clone();
            settings.engine = RenderEngine::from_u8(self.ui.render_engine);
            // Same engine + exposure as the viewport. Sample count from Render settings
            // (viewport may still be converging — wait for the sample counter to catch up).
            if settings.engine == RenderEngine::Cycles {
                settings.cycles_samples = self.ui.render_samples.max(1);
            }
            let samples = settings.cycles_samples.max(1);
            let engine_name = settings.engine.name();
            self.set_render_progress(
                format!("Rendering image ({engine_name})"),
                format!("{w} × {h}"),
                0.0,
            );
            let result = if settings.engine == RenderEngine::Cycles {
                // Step samples in the app so we can pump the % UI between them.
                {
                    let g = self.gfx.as_mut().unwrap();
                    g.renderer.cycles_offline_begin(&cam, &settings, w, h);
                }
                let mut err: Option<String> = None;
                for i in 0..samples {
                    {
                        let g = self.gfx.as_mut().unwrap();
                        if !g.renderer.cycles_offline_step(&cam, &settings, w, h) {
                            err = Some("Cycles produced zero samples".into());
                            break;
                        }
                    }
                    let frac = (i + 1) as f32 / samples as f32;
                    self.set_render_progress(
                        format!("Rendering image (Cycles)"),
                        format!("{w} × {h} · sample {}/{samples}", i + 1),
                        frac,
                    );
                }
                if let Some(e) = err {
                    Err(e)
                } else {
                    let g = self.gfx.as_mut().unwrap();
                    Ok(g.renderer.cycles_offline_finish(w, h))
                }
            } else {
                let g = self.gfx.as_mut().unwrap();
                g.renderer
                    .capture_with(&cam, &settings, w, h, false)
                    .map_err(|e| e.to_string())
            };
            self.clear_render_progress();
            match result {
                Ok(rgba) => {
                    if rgba.len() < (w as usize).saturating_mul(h as usize).saturating_mul(4) {
                        self.status("Render failed: incomplete pixel buffer");
                        self.ui.render_done_message =
                            Some("Render failed: incomplete pixel buffer.".into());
                        return;
                    }
                    self.ui.compositor_render_rgba = rgba.clone();
                    self.ui.compositor_render_width = w;
                    self.ui.compositor_render_height = h;
                    self.ui.compositor_render_serial =
                        self.ui.compositor_render_serial.wrapping_add(1).max(1);
                    // Force the Rendering workspace texture to rebuild for this frame.
                    self.ui.render_result_texture_serial = 0;
                    self.ui.compositor_output_key = None;
                    self.ui.compositor_output_rgba.clear();
                    self.ui.compositor_output_texture = None;
                    self.ui.compositor_eval_error = false;
                    match save_render_image(&path, &rgba, w, h, self.ui.render_format) {
                        Ok(()) => {
                            let msg = format!(
                                "Render finished.\n\nSaved to:\n{}\n\n{} × {} · {} samples · {format_name}",
                                path.display(), w, h, settings.cycles_samples.max(1)
                            );
                            self.status(format!("Saved render to {}", path.display()));
                            self.ui.render_done_message = Some(msg);
                        }
                        Err(e) => {
                            self.status(&format!("Render failed: {e}"));
                            self.ui.render_done_message = Some(format!("Render failed:\n{e}"));
                        }
                    }
                }
                Err(e) => {
                    self.status(format!("Render failed: {e}"));
                    self.ui.render_done_message = Some(format!("Render failed:\n{e}"));
                }
            }
        }
    }

    fn render_compositor(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Compositor rendering is not available on WebAssembly yet");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (w, h) = self.render_resolution();
            self.apply_animation_frame();
            if let Some(g) = self.gfx.as_mut() {
                g.renderer
                    .set_scene(&self.doc.scene, &highlights(&self.doc));
            } else {
                self.status("Renderer not ready");
                return;
            }
            self.uploaded = self.doc.version;
            let cam = self.render_camera();
            let result = {
                let g = self.gfx.as_mut().unwrap();
                g.renderer
                    .capture_with(&cam, &self.settings, w, h, false)
            };
            match result {
                Ok(rgba)
                    if rgba.len() == (w as usize).saturating_mul(h as usize).saturating_mul(4) =>
                {
                    self.ui.compositor_render_rgba = rgba;
                    self.ui.compositor_render_width = w;
                    self.ui.compositor_render_height = h;
                    self.ui.compositor_render_serial =
                        self.ui.compositor_render_serial.wrapping_add(1).max(1);
                    self.ui.render_result_texture_serial = 0;
                    self.ui.compositor_output_key = None;
                    self.ui.compositor_output_rgba.clear();
                    self.ui.compositor_output_texture = None;
                    self.ui.compositor_eval_error = false;
                    self.status("Render complete — processing compositor nodes");
                }
                Ok(_) => self.status("Render failed: incomplete pixel buffer"),
                Err(e) => self.status(&format!("Render failed: {e}")),
            }
        }
    }

    fn save_compositor(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Saving compositor output is not available on WebAssembly yet");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (width, height) = (
                self.ui.compositor_render_width,
                self.ui.compositor_render_height,
            );
            if self.ui.compositor_render_rgba.is_empty() || width == 0 || height == 0 {
                self.status("Render an image before saving the compositor output");
                return;
            }
            // Blender: Composite node is the final file output (Viewer is preview only).
            let rgba = self
                .doc
                .scene
                .compositor_graph
                .evaluate_composite(width, height, &self.ui.compositor_render_rgba)
                .or_else(|| {
                    if !self.ui.compositor_output_rgba.is_empty() {
                        Some(self.ui.compositor_output_rgba.clone())
                    } else {
                        None
                    }
                });
            let Some(rgba) = rgba else {
                self.status("Compositor graph produced no Composite output");
                return;
            };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("composited.png")
                .save_file()
            else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("png");
            }
            match image::save_buffer(&path, &rgba, width, height, image::ColorType::Rgba8) {
                Ok(()) => self.status(&format!("Saved compositor output to {}", path.display())),
                Err(e) => self.status(&format!("Compositor save failed: {e}")),
            }
        }
    }

    fn run_python_script(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.ui.python_script_output =
                "Python scripting is not available on WebAssembly".to_owned();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.ui.python_script_rx.is_some() {
                return;
            }
            let source = self.ui.python_script.clone();
            let snapshot = scripting::ScriptSnapshot::from_document(&self.doc);
            let document_version = self.doc.version;
            let (tx, rx) = std::sync::mpsc::channel();
            self.ui.python_script_rx = Some(rx);
            self.ui.python_script_running = true;
            self.ui.python_script_output = "Running embedded Python…".to_owned();
            let spawn = std::thread::Builder::new()
                .name("blendcraft-python".into())
                .spawn(move || {
                    let result = scripting::execute(source, snapshot);
                    let _ = tx.send(scripting::ScriptCompletion {
                        document_version,
                        result,
                    });
                });
            if let Err(error) = spawn {
                self.ui.python_script_rx = None;
                self.ui.python_script_running = false;
                self.ui.python_script_output = format!("Could not start Python worker: {error}");
            }
        }
    }

    fn open_python_script(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.ui.python_script_output =
                "Opening Python scripts is not available on WebAssembly".to_owned();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Python script", &["py"])
                .pick_file()
            {
                match std::fs::read_to_string(&path) {
                    Ok(source) => {
                        self.ui.python_script = source;
                        self.ui.python_script_path = Some(path);
                        self.ui.python_script_output.clear();
                    }
                    Err(error) => {
                        self.ui.python_script_output = format!("Could not open Python script: {error}");
                    }
                }
            }
        }
    }

    fn save_python_script(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.ui.python_script_output =
                "Saving Python scripts is not available on WebAssembly".to_owned();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = self.ui.python_script_path.clone().or_else(|| {
                rfd::FileDialog::new()
                    .add_filter("Python script", &["py"])
                    .set_file_name("script.py")
                    .save_file()
            });
            let Some(mut path) = path else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("py");
            }
            match std::fs::write(&path, self.ui.python_script.as_bytes()) {
                Ok(()) => {
                    self.ui.python_script_path = Some(path.clone());
                    self.ui.python_script_output = format!("Saved {}", path.display());
                }
                Err(error) => {
                    self.ui.python_script_output = format!("Could not save Python script: {error}");
                }
            }
        }
    }

    fn render_animation(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Render Animation is not available on WebAssembly yet".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Action may run before the per-frame settings sync — mirror UI now.
            self.settings.engine = RenderEngine::from_u8(self.ui.render_engine);

            let Some(dir) = rfd::FileDialog::new()
                .set_title("Choose folder for animation frames")
                .pick_folder()
            else {
                self.status("Animation render cancelled (no folder chosen)");
                return;
            };
            if self.gfx.is_none() {
                self.status("Renderer not ready");
                self.ui.render_done_message = Some("Renderer not ready.".into());
                return;
            }
            let start = self.doc.scene.animation.frame_start;
            let end = self.doc.scene.animation.frame_end.max(start);
            let restore = self.doc.scene.animation.frame_current;
            let fps = self.doc.scene.animation.fps.max(1);
            let (w, h) = self.render_resolution();
            let cycles_anim = self.settings.engine == RenderEngine::Cycles;
            let mut ok = 0u32;
            let mut err_msg: Option<String> = None;
            let total = (end - start + 1) as u32;
            let t0 = web_time::Instant::now();

            {
                let highlight = highlights(&self.doc);
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer.set_scene(&self.doc.scene, &highlight);
                    if cycles_anim {
                        g.renderer.sync_cycles(&self.doc.scene);
                    }
                }
                self.uploaded = self.doc.version;
            }

            let frames_dir = dir.join("frames");
            if let Err(e) = std::fs::create_dir_all(&frames_dir) {
                let msg = format!("Could not create frames folder: {e}");
                self.status(msg.clone());
                self.ui.render_done_message = Some(msg);
                return;
            }
            // Cycles: always write PNGs so progress is visible on disk even if AVI fails.
            // GPU engines: skip PNGs when the AVI writer is available (faster).
            let write_png_frames = cycles_anim;

            let avi_path = dir.join("animation.avi");
            let mut video = match forge_video::VideoWriter::create(&avi_path, w, h, fps, 90) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::warn!("video writer init failed: {e}");
                    None
                }
            };

            // Cap Cycles samples for animation so a 250-frame shot does not take all night.
            // Full quality remains available via single-frame F12.
            let cycles_samples = if cycles_anim {
                self.ui.render_samples.max(1).min(48)
            } else {
                1
            };

            self.set_render_progress(
                "Rendering animation",
                format!(
                    "Frames {start}–{end} · {w}×{h}{}",
                    if cycles_anim {
                        format!(" · Cycles {cycles_samples} spp")
                    } else {
                        String::new()
                    }
                ),
                0.0,
            );

            for frame in start..=end {
                self.doc.scene.animation.frame_current = frame;
                self.doc.scene.animation.playing = false;
                self.apply_animation_frame();

                let highlight = highlights(&self.doc);
                let cam = self.render_camera();
                let mut settings = self.settings.clone();
                if cycles_anim {
                    settings.engine = RenderEngine::Cycles;
                    settings.cycles_samples = cycles_samples;
                }
                let frame_i = (frame - start) as u32;
                let result = {
                    let g = self.gfx.as_mut().unwrap();
                    if cycles_anim {
                        g.renderer.set_scene(&self.doc.scene, &highlight);
                        g.renderer.sync_cycles(&self.doc.scene);
                        g.renderer.capture_with(&cam, &settings, w, h, false)
                    } else {
                        g.renderer.update_transforms(&self.doc.scene, &highlight);
                        g.renderer.capture_with(&cam, &settings, w, h, true)
                    }
                };
                match result {
                    Ok(rgba) => {
                        let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
                        if rgba.len() < need {
                            err_msg = Some(format!(
                                "Animation render failed at frame {frame}: buffer {} < {need} bytes",
                                rgba.len()
                            ));
                            break;
                        }
                        ok += 1;
                        if write_png_frames || video.is_none() {
                            let path = frames_dir.join(format!("frame_{:04}.png", frame));
                            if let Err(e) =
                                image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8)
                            {
                                err_msg = Some(format!("Failed to write {}: {e}", path.display()));
                                break;
                            }
                        }
                        if let Some(v) = video.as_mut() {
                            if let Err(e) = v.push_rgba(rgba) {
                                err_msg = Some(format!("Video encoding failed: {e}"));
                                break;
                            }
                        }
                        let frac = (frame_i + 1) as f32 / total.max(1) as f32;
                        self.set_render_progress(
                            "Rendering animation",
                            format!(
                                "Frame {frame} / {end}  ({}/{} done) · {w}×{h}",
                                frame_i + 1,
                                total
                            ),
                            frac,
                        );
                    }
                    Err(e) => {
                        err_msg = Some(format!("Animation render failed at frame {frame}: {e}"));
                        break;
                    }
                }
            }
            self.clear_render_progress();

            // Restore timeline position and transforms.
            self.doc.scene.animation.frame_current = restore;
            self.apply_animation_frame();
            if let Some(g) = self.gfx.as_mut() {
                g.renderer
                    .update_transforms(&self.doc.scene, &highlights(&self.doc));
            }
            self.uploaded = self.doc.version;

            let elapsed = t0.elapsed().as_secs_f32();
            let fps_render = if elapsed > 0.0 {
                ok as f32 / elapsed
            } else {
                0.0
            };

            // Finish the encoder threads and finalize the AVI index.
            let mut video_note = String::new();
            match video.take() {
                Some(v) => match v.finish() {
                    Ok(stats) if err_msg.is_none() && stats.frames > 0 => {
                        video_note = format!(" · video {}", avi_path.display());
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if err_msg.is_none() {
                            err_msg = Some(format!("Video encoding failed: {e}"));
                        }
                    }
                },
                None => {
                    if err_msg.is_none() && ok > 0 {
                        video_note = " · video writer unavailable (PNG sequence in frames/)".into();
                    }
                }
            }

            if let Some(msg) = err_msg {
                self.status(msg.clone());
                self.ui.render_done_message = Some(msg);
            } else {
                let msg = format!(
                    "Animation render finished.\n\n{ok}/{total} frames ({w}×{h}) in {elapsed:.1}s ({fps_render:.1} fps)\nFrames {start}–{end}{video_note}"
                );
                self.status(msg.clone());
                self.ui.render_done_message = Some(msg);
            }
        }
    }

    fn shortcut(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        if matches!(code, KeyCode::Enter | KeyCode::NumpadEnter) && self.bone_xform_drag.is_some() {
            self.finish_bone_xform(false);
            return;
        }
        if code == KeyCode::Escape {
            if self.ts.tool == Tool::Measure && self.measure_a.is_some() {
                self.measure_a = None;
                self.measure_b = None;
                self.status("Measurement cancelled");
                return;
            }
            if self.bone_xform_drag.is_some() {
                self.finish_bone_xform(true);
                return;
            }
            if self.bone_extrude_drag.is_some() {
                self.cancel_bone_extrude();
                return;
            }
            if self.edit_xform.is_some() {
                self.cancel_edit_xform();
                return;
            }
            self.sel_drag = None;
            if self.doc.armature_edit {
                self.doc.exit_armature_edit();
                self.ts.tool = Tool::Select;
                self.status("Object Mode");
                return;
            }
            if self.doc.in_edit_mode() {
                self.doc.exit_edit_mode(true);
                self.status("Object Mode");
            }
            return;
        }
        if self.ts.tool == Tool::Measure && code == KeyCode::Delete {
            self.measurements.clear();
            self.measure_a = None;
            self.measure_b = None;
            self.status("All measurements cleared");
            return;
        }
        let (ctrl, shift, alt) = (self.ctrl(), self.mods.shift_key(), self.mods.alt_key());
        // Sculpt Mode: D toggles Dyntopo, [ ] adjust detail size
        if self.doc.sculpt_mode && !ctrl && !shift && !alt {
            if code == KeyCode::KeyD {
                self.sculpt.dyntopo = !self.sculpt.dyntopo;
                self.status(if self.sculpt.dyntopo {
                    format!(
                        "Dyntopo ON · relative {:.0}% · [ ] denser/coarser · Alt+Wheel",
                        self.sculpt.dyntopo_detail * 100.0
                    )
                } else {
                    "Dyntopo OFF – press D to add verts while sculpting".into()
                });
                return;
            }
            if code == KeyCode::BracketLeft {
                // denser = smaller relative %
                self.sculpt.dyntopo_detail = (self.sculpt.dyntopo_detail - 0.05).clamp(0.15, 1.5);
                self.status(format!(
                    "Dyntopo relative detail: {:.0}% (smaller % = denser mesh)",
                    self.sculpt.dyntopo_detail * 100.0
                ));
                return;
            }
            if code == KeyCode::BracketRight {
                self.sculpt.dyntopo_detail = (self.sculpt.dyntopo_detail + 0.05).clamp(0.15, 1.5);
                self.status(format!(
                    "Dyntopo relative detail: {:.0}% (larger % = fewer verts)",
                    self.sculpt.dyntopo_detail * 100.0
                ));
                return;
            }
        }
        if self.ui.scripting_mode && alt && !ctrl && !shift && code == KeyCode::KeyP {
            self.apply(Action::RunPythonScript, el);
            return;
        }
        if let Some(a) = keymap::find_action(&self.bindings, ctrl, shift, alt, code) {
            self.apply(a, el);
        }
    }

    // ------------------------------------------------------------------ input
    fn mouse_button(&mut self, state: ElementState, button: MouseButton, consumed: bool) {
        let down = state == ElementState::Pressed;
        if self.bone_xform_drag.is_some() {
            if button == MouseButton::Left && !down {
                self.finish_bone_xform(false);
            } else if button == MouseButton::Right && down {
                self.finish_bone_xform(true);
            }
            return;
        }
        if self.bone_extrude_drag.is_some() {
            if button == MouseButton::Left && !down {
                self.confirm_bone_extrude();
            } else if button == MouseButton::Right && down {
                self.cancel_bone_extrude();
            }
            return;
        }
        if let Some(xf) = self.edit_xform.as_ref() {
            let on_release = xf.confirm_on_release;
            if on_release {
                // Click-drag: release LMB confirms, RMB cancels
                if button == MouseButton::Left && !down {
                    self.confirm_edit_xform();
                } else if button == MouseButton::Right && down {
                    self.cancel_edit_xform();
                }
            } else if down {
                match button {
                    MouseButton::Left => self.confirm_edit_xform(),
                    MouseButton::Right => self.cancel_edit_xform(),
                    _ => {}
                }
            }
            return;
        }
        if self.modal.is_some() {
            if down {
                match button {
                    MouseButton::Left => self.end_modal(false),
                    MouseButton::Right => self.end_modal(true),
                    _ => {}
                }
            }
            return;
        }
        let start = down && !consumed && self.in_viewport(self.cursor);
        let (shift, ctrl) = (self.mods.shift_key(), self.ctrl());
        match button {
            MouseButton::Left => {
                if down {
                    if !start {
                        return;
                    }
                    // --- Texture Paint ---
                    if self.doc.paint_mode {
                        let alt = self.mods.alt_key();
                        if alt {
                            self.left = true;
                            self.press = self.cursor;
                            self.dragged = false;
                            return;
                        }
                        self.doc.checkpoint();
                        self.paint_stroke = true;
                        self.paint_last_cursor = Some(self.cursor);
                        self.paint_sample();
                        return;
                    }
                    // --- Vertex / Weight Paint ---
                    if self.doc.vertex_paint_mode || self.doc.weight_paint_mode {
                        let alt = self.mods.alt_key();
                        if alt {
                            self.left = true;
                            self.press = self.cursor;
                            self.dragged = false;
                            return;
                        }
                        self.doc.checkpoint();
                        self.paint_stroke = true;
                        self.paint_last_cursor = Some(self.cursor);
                        self.vertex_or_weight_paint_sample();
                        return;
                    }
                    // --- Sculpt Mode ---
                    if self.doc.sculpt_mode {
                        // Alt+LMB = orbit / Alt+Shift+LMB = pan (Blender "Emulate 3 Button Mouse")
                        // so you can navigate without damaging the sculpt.
                        let alt = self.mods.alt_key();
                        if alt {
                            self.left = true;
                            self.press = self.cursor;
                            self.dragged = false;
                            return;
                        }
                        // Plain LMB = sculpt stroke
                        self.doc.checkpoint();
                        self.sculpt_stroke = true;
                        self.sculpt_last_hit = None;
                        self.sculpt_last_sample = None;
                        self.press = self.cursor;
                        self.sculpt_sample();
                        return;
                    }
                    // --- Measure tool ---
                    if self.ts.tool == Tool::Measure {
                        if let Some(hit) = self.raycast_ground_or_mesh() {
                            if self.measure_a.is_none() {
                                self.measure_a = Some(hit);
                                self.measure_b = Some(hit);
                                self.status("Measure: click second point");
                            } else {
                                let a = self.measure_a.unwrap();
                                self.measurements.push((a, hit));
                                self.measure_a = None;
                                self.measure_b = None;
                                self.status(format!(
                                    "Distance: {:.4} m · {} measurements",
                                    (hit - a).length(),
                                    self.measurements.len()
                                ));
                            }
                        }
                        return;
                    }
                    // --- Annotate tool ---
                    if self.ts.tool == Tool::Annotate {
                        self.annotate_drag_active = false;
                        self.annotate_stroke = None;
                        if let Some(hit) = self.raycast_ground_or_mesh() {
                            self.annotate_stroke = Some(vec![hit]);
                            self.annotate_drag_active = true;
                            self.status("Annotate: drag while holding LMB; release to finish");
                        }
                        self.left = true;
                        self.press = self.cursor;
                        self.dragged = false;
                        return;
                    }
                    // --- Transform tool: use move gizmo ---
                    if self.ts.tool == Tool::Transform
                        && !(self.doc.armature_edit || self.doc.pose_mode)
                    {
                        self.ts.tool = Tool::Move;
                        if self.settings.show_gizmo && self.gizmo.begin(
                            self.cursor,
                            &self.camera,
                            self.viewport,
                            &mut self.doc,
                            &self.ts,
                        ) {
                            return;
                        }
                        self.ts.tool = Tool::Transform;
                    }
                    if self.doc.armature_edit
                        && matches!(self.ts.tool, Tool::Select | Tool::BoneExtrude)
                        && self.try_start_bone_joint_extrude()
                    {
                        return;
                    }
                    if self.doc.armature_edit && self.ts.tool == Tool::BoneExtrude {
                        if self.try_pick_bone(false) {
                            self.start_bone_extrude();
                            if self.bone_extrude_drag.is_some() {
                                self.update_bone_extrude();
                            }
                        } else {
                            self.status("Click a bone to extrude it");
                        }
                        return;
                    }
                    // In armature modes, gizmo handles take priority over bone picking.
                    if (self.doc.armature_edit || self.doc.pose_mode)
                        && self.ts.tool.is_gizmo()
                        && self.settings.show_gizmo
                        && self.gizmo.begin(
                            self.cursor,
                            &self.camera,
                            self.viewport,
                            &mut self.doc,
                            &self.ts,
                        )
                    {
                        return;
                    }
                    if (self.doc.armature_edit || self.doc.pose_mode)
                        && self.doc.active_armature().is_some()
                        && matches!(
                            self.ts.tool,
                            Tool::Select
                                | Tool::Move
                                | Tool::Rotate
                                | Tool::Scale
                                | Tool::Transform
                        )
                    {
                        if self.try_pick_bone(shift) {
                            return;
                        }
                    }
                    if self.doc.in_edit_mode() && self.doc.edit.is_some() {
                        // Ctrl+LMB: interactive edit tool (Extrude, Inset, Bevel, ...)
                        if ctrl && Self::is_ctrl_lmb_tool(self.ts.tool) {
                            self.begin_tool_drag();
                            return;
                        }
                        // Axis arrows first (like Object Mode gizmo)
                        if self.settings.show_gizmo && self.begin_edit_gizmo_drag() {
                            return;
                        }
                        // Click a vertex: select + free drag
                        if self.try_edit_click_drag(shift) {
                            return;
                        }
                        // Missed: orbit camera
                        self.left = true;
                        self.press = self.cursor;
                        self.dragged = false;
                        return;
                    }
                    if self.ts.tool.is_gizmo()
                        && !(self.doc.armature_edit || self.doc.pose_mode)
                        && self.settings.show_gizmo
                        && self.gizmo.begin(
                            self.cursor,
                            &self.camera,
                            self.viewport,
                            &mut self.doc,
                            &self.ts,
                        )
                    {
                        return;
                    }
                    match self.ts.tool {
                        Tool::BoxSelect | Tool::Lasso | Tool::Circle => {
                            self.begin_sel_drag(shift, ctrl)
                        }
                        _ => {
                            self.left = true;
                            self.press = self.cursor;
                            self.dragged = false;
                        }
                    }
                } else if self.paint_stroke {
                    self.paint_stroke = false;
                    self.paint_last_cursor = None;
                    self.status("Paint stroke done");
                } else if self.sculpt_stroke {
                    self.sculpt_stroke = false;
                    self.sculpt_last_hit = None;
                    self.status("Sculpt stroke done");
                } else if self.tool_drag.is_some() {
                    self.end_tool_drag();
                } else if self.edit_gizmo_drag.is_some() {
                    self.end_edit_gizmo_drag();
                } else if self.gizmo.dragging() {
                    self.gizmo.end();
                } else if self.sel_drag.is_some() {
                    self.finish_sel_drag();
                } else if self.left {
                    self.left = false;
                    // Finish annotate stroke
                    if self.ts.tool == Tool::Annotate {
                        self.annotate_drag_active = false;
                        if let Some(stroke) = self.annotate_stroke.take() {
                            if stroke.len() >= 2 {
                                self.annotations.push(stroke);
                                self.status(format!(
                                    "Annotation saved ({} strokes)",
                                    self.annotations.len()
                                ));
                            }
                        }
                    } else if !self.dragged {
                        self.pick(shift);
                    }
                }
            }
            MouseButton::Right => {
                if down {
                    // In Object Mode, a viewport right-click selects the object under
                    // the pointer and is reserved for the shared Object context menu.
                    // Keep RMB camera navigation for non-Object work modes / modifiers.
                    if start
                        && !self.doc.in_edit_mode()
                        && !self.doc.pose_mode
                        && !self.doc.sculpt_mode
                        && !self.doc.uv_mode
                        && !self.doc.paint_mode
                        && !shift
                    {
                        self.pick(false);
                        self.right = false;
                        return;
                    }
                    self.right = start;
                    self.right_press = self.cursor;
                    self.right_dragged = false;
                } else {
                    if self.right && !self.right_dragged && shift {
                        self.set_cursor_at_mouse();
                    }
                    self.right = false;
                }
            }
            MouseButton::Middle => self.middle = start,
            _ => {}
        }
    }

    fn cursor_moved(&mut self, x: f64, y: f64) {
        let (dx, dy) = ((x - self.cursor.0) as f32, (y - self.cursor.1) as f32);
        self.cursor = (x, y);
        if self.annotate_drag_active && (!self.left || self.ts.tool != Tool::Annotate) {
            self.annotate_drag_active = false;
            self.annotate_stroke = None;
        }
        let (shift, ctrl) = (self.mods.shift_key(), self.ctrl());
        // Texture painting is a continuous stroke, not a single click stamp.
        // Interpolate screen-space samples so fast pointer movement cannot leave gaps.
        if self.paint_stroke {
            let from = self.paint_last_cursor.unwrap_or((x, y));
            let distance = ((x - from.0).powi(2) + (y - from.1).powi(2)).sqrt();
            let steps = (distance / 4.0).ceil().max(1.0) as usize;
            if self.doc.vertex_paint_mode || self.doc.weight_paint_mode {
                let mut pts = Vec::with_capacity(steps);
                for step in 1..=steps {
                    let t = step as f64 / steps as f64;
                    let sample = (from.0 + (x - from.0) * t, from.1 + (y - from.1) * t);
                    if let Some(hit) = self.paint_local_hit(sample) {
                        pts.push(hit);
                    }
                }
                if !pts.is_empty() {
                    if self.doc.vertex_paint_mode {
                        let s = self.vertex_paint_settings();
                        let _ = self.doc.vertex_paint_at(&pts, &s);
                    } else {
                        let s = self.weight_paint_settings();
                        let _ = self.doc.weight_paint_at(&pts, &s);
                    }
                }
            } else {
                let mut uvs = Vec::with_capacity(steps);
                for step in 1..=steps {
                    let t = step as f64 / steps as f64;
                    let sample = (from.0 + (x - from.0) * t, from.1 + (y - from.1) * t);
                    if let Some(uv) = self.paint_sample_at(sample) {
                        uvs.push(uv);
                    }
                }
                if !uvs.is_empty() {
                    match self.ts.tool {
                        Tool::PaintErase => {
                            let _ = self.doc.erase_stroke_uv(&uvs, self.ts.paint_radius, self.ts.paint_strength);
                        }
                        Tool::PaintSoften => {
                            let _ = self.doc.soften_stroke_uv(&uvs, self.ts.paint_radius, self.ts.paint_strength);
                        }
                        Tool::PaintSmear => {
                            let _ = self.doc.smear_stroke_uv(&uvs, self.ts.paint_radius, self.ts.paint_strength);
                        }
                        Tool::PaintFill => {
                            if let Some(uv) = uvs.last() {
                                let _ = self.doc.fill_uv(*uv, self.ts.paint_color, 24);
                            }
                        }
                        _ => {
                            let _ = self.doc.paint_stroke_uv(&uvs, self.ts.paint_color, self.ts.paint_radius, self.ts.paint_strength);
                        }
                    }
                }
            }
            self.paint_last_cursor = Some((x, y));
            return;
        }
        // Live measure second point
        if self.ts.tool == Tool::Measure && self.measure_a.is_some() {
            if let Some(hit) = self.raycast_ground_or_mesh() {
                self.measure_b = Some(hit);
                let a = self.measure_a.unwrap();
                let dist = (hit - a).length();
                self.status(format!("Distance: {:.4} m", dist));
            }
        }
        // Annotate stroke while dragging
        if self.annotate_drag_active && self.left && self.ts.tool == Tool::Annotate {
            if let Some(hit) = self.raycast_ground_or_mesh() {
                if let Some(s) = self.annotate_stroke.as_mut() {
                    if s.last().map(|p| (*p - hit).length() > 0.02).unwrap_or(true) {
                        s.push(hit);
                    }
                }
            }
            // Annotation owns this drag; do not also orbit the camera or
            // feed the same mouse movement to another viewport interaction.
            return;
        }
        if self.sculpt_stroke {
            self.sculpt_sample();
            return;
        }
        if self.tool_drag.is_some() {
            self.update_tool_drag();
            return;
        }
        if self.bone_extrude_drag.is_some() {
            self.update_bone_extrude();
            return;
        }
        if self.bone_xform_drag.is_some() {
            self.update_bone_xform();
            return;
        }
        if self.edit_gizmo_drag.is_some() {
            self.update_edit_gizmo_drag();
            return;
        }
        if self.edit_xform.is_some() {
            self.update_edit_xform();
            return;
        }
        if self.modal.is_some() {
            self.update_modal();
            return;
        }
        if self.gizmo.dragging() && self.doc.edit.is_none() {
            self.gizmo.update(
                self.cursor,
                &self.camera,
                self.viewport,
                &mut self.doc,
                &self.ts,
                ctrl,
            );
            return;
        }
        if self.sel_drag.is_some() {
            self.update_sel_drag();
            return;
        }
        if !self.settings.show_gizmo {
            self.edit_gizmo_hover = None;
        } else if !self.left && !self.middle && !self.right {
            if self.doc.edit.is_some() {
                self.edit_gizmo_hover_test();
            } else {
                self.gizmo.hover_test(
                    self.cursor,
                    &self.camera,
                    self.viewport,
                    &self.doc,
                    &self.ts,
                );
            }
        }
        if self.left {
            let d = ((x - self.press.0).powi(2) + (y - self.press.1).powi(2)).sqrt();
            if d > 4.0 {
                self.dragged = true;
            }
            if self.dragged {
                if shift {
                    self.camera.pan(dx, dy)
                } else {
                    self.camera.orbit(dx, dy)
                }
            }
        }
        if self.middle {
            if shift {
                self.camera.pan(dx, dy)
            } else {
                self.camera.orbit(dx, dy)
            }
        }
        if self.right {
            let d = ((x - self.right_press.0).powi(2) + (y - self.right_press.1).powi(2)).sqrt();
            if d > 4.0 {
                self.right_dragged = true;
            }
            if self.right_dragged {
                // Unity-style viewport navigation in both paint and sculpt modes:
                // hold RMB and move to orbit; Shift+RMB pans. Keep the legacy
                // RMB-pan behavior in the other workspaces.
                if self.doc.paint_mode || self.doc.sculpt_mode {
                    if shift {
                        self.camera.pan(dx, dy);
                    } else {
                        self.camera.orbit(dx, dy);
                    }
                } else {
                    self.camera.pan(dx, dy);
                }
            }
        }
    }

    // ------------------------------------------------------------------ frame
    fn redraw(&mut self, el: &ActiveEventLoop) {
        // Advance timeline when playing (real-time FPS).
        // Accumulates time so slow frames still keep overall speed correct:
        // at 24 fps, 24 steps ≈ 1 second of wall clock. The leftover time is turned
        // into a fractional frame so motion is evaluated every display refresh
        // (smooth) instead of only 24 times per second (choppy camera).
        if self.doc.scene.animation.playing {
            let fps = self.doc.scene.animation.fps.max(1) as f32;
            let frame_dt = 1.0 / fps;
            let mut elapsed = self.last_anim_tick.elapsed().as_secs_f32();
            // Cap catch-up so a long hitch does not jump the whole timeline at once.
            let max_steps = 8u32;
            let mut advanced_frames = Vec::new();
            while elapsed >= frame_dt && advanced_frames.len() < max_steps as usize {
                elapsed -= frame_dt;
                self.doc.scene.animation.step();
                advanced_frames.push(self.doc.scene.animation.frame_current);
            }
            if !advanced_frames.is_empty() {
                // Keep residual time so we do not drift slow.
                self.last_anim_tick = web_time::Instant::now()
                    - std::time::Duration::from_secs_f32(elapsed.max(0.0));
            }
            let steps = advanced_frames.len();
            self.anim_subframe = (elapsed / frame_dt).clamp(0.0, 0.999);
            let frame = self.doc.scene.animation.frame_current as f32 + self.anim_subframe;
            let mut physics_changed = false;
            if self.doc.scene.physics_cache.is_some() {
                self.eval_animation_at(frame);
                let current_frame = self.doc.scene.animation.frame_current;
                physics_changed = forge_core::apply_baked_frame(
                    &mut self.doc.scene,
                    current_frame,
                );
                self.physics_runtime = None;
            } else if steps > 0 {
                // Build the runtime from the scene at the frame playback started,
                // before evaluating the next animation frame. Then evaluate and
                // simulate each frame in order so catch-up never skips physics time.
                if self.physics_runtime.is_some()
                    && self.physics_runtime_doc_version != self.doc.version
                {
                    // User edits made while paused invalidate the transient solver state.
                    self.physics_runtime = None;
                }
                if self.physics_runtime.is_none() {
                    self.physics_runtime = Some(forge_core::PhysicsRuntime::new(&self.doc.scene));
                    self.physics_runtime_doc_version = self.doc.version;
                }
                for simulation_frame in advanced_frames {
                    self.eval_animation_at(simulation_frame as f32);
                    if let Some(runtime) = self.physics_runtime.as_mut() {
                        physics_changed |= runtime.step_scene(&mut self.doc.scene, frame_dt);
                    }
                }
                self.eval_animation_at(frame);
            } else {
                self.eval_animation_at(frame);
            }
            if self.animation_moves_lights() {
                // Light positions are baked into the scene upload: full refresh.
                self.doc.touch();
                if self.physics_runtime.is_some() {
                    self.physics_runtime_doc_version = self.doc.version;
                }
            } else {
                self.anim_gpu_dirty = true;
            }
            if physics_changed {
                self.anim_gpu_dirty = true;
                self.doc.dirty = true;
            }
        } else {
            if self.anim_subframe != 0.0 {
                // Playback just stopped: snap back onto the whole frame.
                self.apply_animation_frame();
            }
        }
        self.step_view_anim();
        // Camera view follows the (already updated) camera object this very frame.
        self.sync_camera_view();

        // Camera used for BOTH the egui overlays (edit-mode vertices/edges, gizmos) and the
        // GPU pass of this frame. Nav-gizmo orbit/pan/zoom are applied after the UI is built,
        // so rendering with the live camera made the overlay trail one frame behind the mesh.
        let frame_camera = self.camera.clone();

        // 1) build the UI
        let raw = match self.gfx.as_mut() {
            Some(g) => g.egui_state.take_egui_input(&g.window),
            None => return,
        };
        let shape = self.sel_shape();
        let modal_text = self.modal.as_ref().map(|m| m.status());
        let circle = (self.ts.tool == Tool::Circle && self.in_viewport(self.cursor)).then(|| {
            (
                Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32),
                self.ts.circle_radius,
            )
        });
        let mut actions = Vec::new();
        let mut nav_actions: Vec<navgizmo::NavAction> = Vec::new();
        // The zoom / pan / camera / projection buttons are Object Mode only.
        let nav_full = !self.doc.in_edit_mode()
            && !self.doc.sculpt_mode
            && !self.doc.uv_mode
            && !self.doc.paint_mode
            && !self.doc.pose_mode
            && !self.doc.armature_edit;
        let nav_in_camera = self.camera_view.is_some();
        let mut rect = egui::Rect::NOTHING;
        let rendering_ws =
            (self.ui.rendering_mode || self.ui.compositing_mode || self.ui.scripting_mode)
                && !self.doc.uv_mode
                && !self.doc.paint_mode
                && !self.doc.sculpt_mode;
        let out = self.egui_ctx.run(raw, |ctx| {
            let (a, r) = ui::draw(
                ctx,
                &mut self.doc,
                &mut self.settings,
                &mut self.ts,
                &mut self.ui,
                &self.bindings,
                &self.recent.paths,
            );
            // Rendering workspace: no 3D overlays / gizmos / tool HUD — only the Image editor UI.
            // Also hide gizmos while a render progress/done modal is up so they cannot
            // paint on top of the dialog (they use Order::Foreground; modal uses Tooltip).
            let render_modal = self.ui.render_progress.is_some()
                || self.ui.render_done_message.is_some();
            if !rendering_ws && !render_modal {
                tools::draw_toolbar(ctx, r, &mut self.ts);
                if self.settings.show_gizmo {
                    nav_actions = navgizmo::draw(ctx, &self.camera, r, nav_full, nav_in_camera);
                }
                let measure_preview = match (self.measure_a, self.measure_b) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                };
                let extra = overlay::Extra {
                    shape: shape.as_ref(),
                    modal: modal_text.as_deref(),
                    circle,
                    measurements: &self.measurements,
                    measure_preview,
                    annotations: &self.annotations,
                    annotate_stroke: self.annotate_stroke.as_deref(),
                    xray: self.settings.xray,
                    overlays: self.settings.show_overlays,
                };
                overlay::draw(ctx, &self.camera, r, &self.doc, &extra);
                if !self.settings.show_gizmo {
                    // gizmos hidden
                } else if self.doc.edit.is_some() {
                    draw_edit_gizmo_ui(
                        ctx,
                        r,
                        &self.camera,
                        self.viewport,
                        &self.doc,
                        self.edit_gizmo_hover,
                        self.edit_gizmo_drag.as_ref().map(|d| d.axis),
                    );
                } else {
                    self.gizmo.draw(ctx, &self.camera, r, &self.doc, &self.ts);
                }
            }
            actions = a;
            rect = r;
        });
        if let Some(g) = self.gfx.as_mut() {
            g.egui_state
                .handle_platform_output(&g.window, out.platform_output.clone());
        }
        let ppp = out.pixels_per_point;
        // Zero viewport in Rendering mode so the GPU 3D pass draws nothing into the window.
        if rendering_ws {
            self.viewport = [0.0, 0.0, 0.0, 0.0];
        } else {
            let mut vx = rect.min.x * ppp;
            let mut vy = rect.min.y * ppp;
            let mut vw = rect.width() * ppp;
            let mut vh = rect.height() * ppp;
            // Camera view: letterbox to the *output* aspect so F12 matches what you see.
            if self.camera_view.is_some() && vw > 1.0 && vh > 1.0 {
                let (rw, rh) = if self.ui.render_width >= 64 && self.ui.render_height >= 64 {
                    (self.ui.render_width as f32, self.ui.render_height as f32)
                } else {
                    (vw, vh)
                };
                let target = (rw / rh.max(1.0)).max(0.05);
                let panel = vw / vh.max(1.0);
                if panel > target {
                    // Panel wider than render → pillarbox left/right
                    let new_w = vh * target;
                    vx += (vw - new_w) * 0.5;
                    vw = new_w;
                } else if panel < target {
                    // Panel taller than render → letterbox top/bottom
                    let new_h = vw / target;
                    vy += (vh - new_h) * 0.5;
                    vh = new_h;
                }
            }
            self.viewport = [vx, vy, vw, vh];
            if vw >= 32.0 && vh >= 32.0 {
                self.last_3d_viewport = [vx, vy, vw, vh];
            }
        }

        // 2) apply menu actions
        for a in actions {
            self.apply(a, el);
        }
        for a in nav_actions {
            self.apply_nav(a);
        }
        self.sync_camera_view();

        // 3) window title
        let name = self
            .doc
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or("Untitled".to_string(), |n| n.to_string_lossy().into_owned());
        let title = format!(
            "{}{} - BlendCraft",
            name,
            if self.doc.dirty { "*" } else { "" }
        );
        if title != self.title {
            if let Some(g) = &self.gfx {
                g.window.set_title(&title);
            }
            self.title = title;
        }

        // Sync Blender-style render engine from the Render properties panel.
        self.settings.engine = RenderEngine::from_u8(self.ui.render_engine);
        // One resolution for everything: Output X × Y × %. The Cycles view buffer uses it
        // (exactly, in camera view; same pixel density otherwise). Pixel Size only divides
        // the live view for speed.
        let (out_w, out_h) = self.render_resolution();
        let want_pix = match self.ui.cycles_pixel_size {
            1 | 2 | 4 | 8 => self.ui.cycles_pixel_size,
            _ => 1,
        };
        let mut view_size = if self.camera_view.is_some() {
            (out_w, out_h)
        } else if self.viewport[2] >= 2.0 && self.viewport[3] >= 2.0 {
            let k = out_w.max(out_h) as f32 / self.viewport[2].max(self.viewport[3]);
            (
                ((self.viewport[2] * k).round() as u32).max(2),
                ((self.viewport[3] * k).round() as u32).max(2),
            )
        } else {
            (out_w, out_h)
        };
        let long = view_size.0.max(view_size.1);
        if long > 4096 {
            let k = 4096.0 / long as f32;
            view_size = (
                ((view_size.0 as f32 * k).round() as u32).max(2),
                ((view_size.1 as f32 * k).round() as u32).max(2),
            );
        }
        let res_key = (out_w, out_h, want_pix, self.camera_view.is_some());
        let res_changed = res_key != self.cycles_res_key
            || self.settings.cycles_pixel_size != want_pix;
        self.cycles_res_key = res_key;
        self.settings.cycles_max_dim = 4096;
        self.settings.cycles_pixel_size = want_pix;
        self.settings.cycles_view_size = [view_size.0, view_size.1];
        if self.settings.engine == RenderEngine::Cycles {
            self.settings.cycles_samples = self.ui.render_samples_viewport.max(1);
        } else if self.settings.engine == RenderEngine::Workbench {
            // Solid mode: leave shadows as user set.
        }

        // 4) sync scene to the GPU when something changed
        let Some(g) = self.gfx.as_mut() else { return };
        if self.doc.version != self.uploaded {
            let hl = highlights(&self.doc);
            g.renderer.set_scene(&self.doc.scene, &hl);
            self.uploaded = self.doc.version;
            self.anim_gpu_dirty = false;
        } else if self.anim_gpu_dirty {
            // Playback only moved transforms: cheap matrix refresh, no scene rebuild.
            let hl = highlights(&self.doc);
            g.renderer.update_transforms(&self.doc.scene, &hl);
            self.anim_gpu_dirty = false;
        }
        // Shader-graph edits often write materials without bumping doc.version.
        // When Cycles is active, rebuild path-tracer data if shading inputs changed.
        if self.settings.engine == RenderEngine::Cycles {
            if res_changed {
                g.renderer.invalidate_cycles_resolution();
            }
            g.renderer.sync_cycles(&self.doc.scene);
        }
        // Status after gfx borrow ends (cannot mut-borrow self while `g` is held).
        let cycles_res_msg = if self.settings.engine == RenderEngine::Cycles && res_changed {
            Some(format!(
                "Cycles view: {} × {} px (Output {} × {}) · Pixel Size {}×",
                self.settings.cycles_view_size[0] / self.settings.cycles_pixel_size.max(1),
                self.settings.cycles_view_size[1] / self.settings.cycles_pixel_size.max(1),
                out_w,
                out_h,
                self.settings.cycles_pixel_size.max(1)
            ))
        } else {
            None
        };

        // 5) render 3D + egui overlay
        let (w, h) = g.renderer.size();
        let tris = self.egui_ctx.tessellate(out.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: ppp,
        };
        let textures = out.textures_delta;
        let er = &mut g.egui_renderer;
        let res = g.renderer.render(
            &frame_camera,
            &self.settings,
            self.viewport,
            &mut |device, queue, enc, view| {
                for (id, delta) in &textures.set {
                    er.update_texture(device, queue, *id, delta);
                }
                let bufs = er.update_buffers(device, queue, enc, &tris, &screen);
                {
                    let mut rp = enc
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        })
                        .forget_lifetime();
                    er.render(&mut rp, &tris, &screen);
                }
                for id in &textures.free {
                    er.free_texture(id);
                }
                bufs
            },
        );
        if let Err(e) = res {
            log::error!("render error: {e}");
        }
        g.window.request_redraw();
        drop(g);
        if let Some(msg) = cycles_res_msg {
            self.status(msg);
        }
    }
}

impl App {
    /// Build the egui integration around a ready renderer and load the initial file.
    fn finish_gfx_init(&mut self, window: Arc<Window>, renderer: Renderer) {
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            Some(renderer.device().limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1, false);
        self.gfx = Some(Gfx {
            window,
            renderer,
            egui_state,
            egui_renderer,
        });
        if let Some(p) = self.initial.take() {
            match ext_of(&p).as_str() {
                "forge" => self.open_project(p),
                "glb" | "gltf" | "obj" => {
                    self.doc = Document::from_scene(Default::default(), None);
                    self.physics_runtime = None;
                    self.import_path(&p);
                    self.doc.dirty = false;
                }
                _ => self.handle_path(p),
            }
        } else {
            self.frame(false);
        }
    }

    /// WebAssembly: pick up the renderer once the async WebGPU/WebGL init finished.
    #[cfg(target_arch = "wasm32")]
    fn poll_pending_gpu(&mut self) {
        if self.gfx.is_some() {
            return;
        }
        let result = self.pending_gpu.borrow_mut().take();
        match result {
            Some(Ok(renderer)) => {
                if let Some(window) = self.web_window.take() {
                    self.finish_gfx_init(window, renderer);
                    if let Some(g) = self.gfx.as_ref() {
                        g.window.request_redraw();
                    }
                }
            }
            Some(Err(e)) => {
                log::error!("GPU init failed: {e:#}");
                self.web_window = None;
            }
            None => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            if self.gpu_init_started {
                return;
            }
        }
        let mut attrs = Window::default_attributes()
            .with_title("BlendCraft")
            .with_inner_size(LogicalSize::new(1400, 850));
        // Browser: create a <canvas> and attach it to the page body.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            attrs = attrs.with_append(true);
        }
        // Window / taskbar icon from the project logo (hand-eye OK mark).
        // Resize to 256² so every platform accepts it (raw 1254² can fail silently).
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(img) = image::load_from_memory(include_bytes!("../logo.png")) {
                let img = img.resize_exact(256, 256, image::imageops::FilterType::Lanczos3);
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                if let Ok(icon) = winit::window::Icon::from_rgba(rgba.into_raw(), w, h) {
                    attrs = attrs.with_window_icon(Some(icon));
                }
            }
        }
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        let size = window.inner_size();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let renderer = pollster::block_on(Renderer::new(window.clone(), size.width, size.height))
                .expect("init GPU");
            self.finish_gfx_init(window, renderer);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.gpu_init_started = true;
            self.web_window = Some(window.clone());
            let slot = self.pending_gpu.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result =
                    Renderer::new(window.clone(), size.width.max(1), size.height.max(1)).await;
                *slot.borrow_mut() = Some(result);
                window.request_redraw();
            });
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        {
            self.poll_pending_gpu();
        }
        let mut consumed = false;
        if let Some(g) = self.gfx.as_mut() {
            consumed = g.egui_state.on_window_event(&g.window, &event).consumed;
        }
        match event {
            WindowEvent::CloseRequested => {
                if self.confirm_discard() {
                    el.exit();
                }
            }
            WindowEvent::Resized(s) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer.resize(s.width, s.height);
                }
            }
            WindowEvent::DroppedFile(p) => self.handle_path(p),
            WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
            WindowEvent::MouseInput { state, button, .. } => {
                self.mouse_button(state, button, consumed)
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed && self.in_viewport(self.cursor) && self.modal.is_none() {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                    };
                    if self.doc.paint_mode {
                        if self.mods.shift_key() {
                            self.ts.paint_strength =
                                (self.ts.paint_strength + y * 0.05).clamp(0.05, 1.0);
                            self.status(format!("Paint strength: {:.2}", self.ts.paint_strength));
                        } else {
                            self.ts.paint_radius =
                                (self.ts.paint_radius * (1.0 + 0.1 * y)).clamp(0.005, 0.25);
                            self.status(format!("Paint radius: {:.3}", self.ts.paint_radius));
                        }
                    } else if self.doc.sculpt_mode {
                        if self.mods.alt_key() {
                            self.sculpt.dyntopo_detail =
                                (self.sculpt.dyntopo_detail - y * 0.05).clamp(0.15, 1.5);
                            self.status(format!(
                                "Dyntopo relative detail: {:.0}%{}",
                                self.sculpt.dyntopo_detail * 100.0,
                                if self.sculpt.dyntopo {
                                    ""
                                } else {
                                    " – press D to enable"
                                }
                            ));
                        } else if self.mods.shift_key() {
                            self.sculpt.strength =
                                (self.sculpt.strength + y * 0.05).clamp(0.05, 1.0);
                            self.status(format!("Brush strength: {:.2}", self.sculpt.strength));
                        } else {
                            self.sculpt.radius =
                                (self.sculpt.radius * (1.0 + 0.1 * y)).clamp(0.05, 5.0);
                            self.status(format!("Brush radius: {:.2}", self.sculpt.radius));
                        }
                    } else if self.ts.tool == Tool::Circle {
                        self.ts.circle_radius =
                            (self.ts.circle_radius * (1.0 + 0.1 * y)).clamp(6.0, 600.0);
                    } else {
                        self.camera.zoom(y);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                // Undo/Redo via logical character so Ctrl+Z works on German QWERTZ (Z/Y swapped)
                {
                    use winit::keyboard::Key;
                    let ctrl = self.ctrl();
                    let shift = self.mods.shift_key();
                    let is_undo = ctrl
                        && !shift
                        && matches!(
                            &event.logical_key,
                            Key::Character(c) if c.eq_ignore_ascii_case("z")
                        );
                    let is_redo = ctrl
                        && ((shift
                            && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("z")))
                            || matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("y")));
                    if is_undo {
                        self.apply(keymap::Action::Undo, el);
                        return;
                    }
                    if is_redo {
                        self.apply(keymap::Action::Redo, el);
                        return;
                    }
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    if self.modal.is_some() {
                        self.modal_key(code);
                    } else if !self.egui_ctx.wants_keyboard_input() {
                        let viewport_key = matches!(
                            code,
                            KeyCode::Tab
                                | KeyCode::Digit1
                                | KeyCode::Digit2
                                | KeyCode::Digit3
                                | KeyCode::KeyG
                                | KeyCode::KeyR
                                | KeyCode::KeyS
                                | KeyCode::KeyE
                                | KeyCode::KeyX
                                | KeyCode::KeyA
                                | KeyCode::KeyZ
                                | KeyCode::KeyY
                                | KeyCode::Delete
                                | KeyCode::Escape
                        );
                        if !consumed || viewport_key {
                            self.shortcut(code, el);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => self.redraw(el),
            _ => {}
        }
    }
}


fn ray_triangle(o: glam::Vec3, d: glam::Vec3, v0: glam::Vec3, v1: glam::Vec3, v2: glam::Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = o - v0;
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 1e-5 {
        Some(t)
    } else {
        None
    }
}

