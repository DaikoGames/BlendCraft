mod gizmo;
mod keymap;
mod overlay;
mod recent;
mod fonts_io;
mod tools;
mod transform;
mod ui;

use egui_wgpu::wgpu;
use forge_core::{project, Constraint, Document, EditDomain, EnvironmentMap, FixedView, ObjectKind, OrbitCamera, SameKind, SculptSettings};
use forge_render::{Highlight, Renderer, ViewSettings};
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

/// Modal transform for selected mesh components in Edit Mode (G / R / S).
struct BoneExtrudeDrag {
    bone_idx: u32,
    /// Fixed head in armature local space
    head: glam::Vec3,
}

struct EditGizmoDrag {
    axis: usize,
    /// World-space axis direction
    axis_dir: glam::Vec3,
    origin: glam::Vec3,
    start_param: f32,
    verts: Vec<(u32, glam::Vec3)>,
    inv_world: glam::Mat4,
}

struct EditXform {
    kind: EditXformKind,
    start_cursor: (f64, f64),
    verts: Vec<(u32, glam::Vec3)>,
    center: glam::Vec3,
    inv_world: glam::Mat4,
    /// true = drag while holding LMB (click-drag verts); false = modal G/R/S
    confirm_on_release: bool,
}


use ui::{Rename, UiState};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
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

struct App {
    gfx: Option<Gfx>,
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
    bone_pose_drag: Option<(f64, f64)>,
    bone_extrude_drag: Option<BoneExtrudeDrag>,
    sel_drag: Option<SelDrag>,
    /// Measure tool: first click point in world space
    measure_a: Option<glam::Vec3>,
    /// Measure tool: second point (live while moving)
    measure_b: Option<glam::Vec3>,
    /// Annotate tool: strokes as lists of world-space points
    annotations: Vec<Vec<glam::Vec3>>,
    /// Stroke currently being drawn
    annotate_stroke: Option<Vec<glam::Vec3>>,
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
    uploaded: u64,
    last_anim_tick: std::time::Instant,
    title: String,
    initial: Option<PathBuf>,
}

fn highlights(doc: &Document) -> Vec<Highlight> {
    doc.scene
        .objects
        .iter()
        .map(|o| {
            if doc.selection.contains(&o.id) {
                if doc.active == Some(o.id) { Highlight::Active } else { Highlight::Selected }
            } else {
                Highlight::None
            }
        })
        .collect()
}

fn ext_of(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
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
    let Some((origin, _)) = edit_sel_world_center(doc) else { return };
    let dist = (origin - camera.target).length().max(0.5);
    let len = (dist * 0.15).clamp(0.3, 8.0);
    let ppp = ctx.pixels_per_point();
    let vp = [rect.min.x * ppp, rect.min.y * ppp, rect.width() * ppp, rect.height() * ppp];
    let painter = ctx
        .layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("edit_gizmo")))
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
        let stroke = egui::Stroke::new(if on { 3.5 } else { 2.5 }, col);
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

impl App {
    fn new(initial: Option<PathBuf>) -> Self {
        Self {
            gfx: None,
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
            bone_pose_drag: None,
            bone_extrude_drag: None,
            sel_drag: None,
            measure_a: None,
            measure_b: None,
            annotations: Vec::new(),
            annotate_stroke: None,
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
                ui.render_use_viewport = false;
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
            uploaded: u64::MAX,
            last_anim_tick: std::time::Instant::now(),
            title: String::new(),
            initial,
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
        let b = if selected_only { self.doc.selected_bounds() } else { None };
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
        let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
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
        self.status(format!("{}: drag to adjust, release to confirm", self.ts.tool.name()));
    }

    fn update_tool_drag(&mut self) {
        let Some(drag) = self.tool_drag.as_ref() else { return };
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
                    self.status(format!("Shrink/Fatten: {:.3}  (release to confirm)", amount));
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
        let center = self.doc.edit.as_ref().map(|e| e.selection_center()).unwrap_or(glam::Vec3::ZERO);
        let world_center = if let Some(session) = self.doc.edit.as_ref() {
            if let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == session.object_id) {
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
            if let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == session.object_id) {
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
            let color = if self.ts.tool == Tool::PaintErase { [0.78, 0.78, 0.78, 1.0] } else { self.ts.paint_color };
            let _ = self.doc.paint_at_uv(uv, color, self.ts.paint_radius, self.ts.paint_strength);
        }
    }

    fn paint_sample_at(&self, cursor: (f64, f64)) -> Option<glam::Vec2> {
        if !self.doc.paint_mode {
            return None;
        }
        let (vx, vy, vw, vh) = (self.viewport[0], self.viewport[1], self.viewport[2], self.viewport[3]);
        if vw < 1.0 || vh < 1.0
            || cursor.0 < vx as f64 || cursor.1 < vy as f64
            || cursor.0 > (vx + vw) as f64 || cursor.1 > (vy + vh) as f64
        {
            return None;
        }
        let nx = ((cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((cursor.1 as f32 - vy) / vh) * 2.0;
        let (origin, dir) = self.camera.ray((nx, ny), vw / vh);
        // Raycast active mesh for UV
        let Some(id) = self.doc.active else { return None };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return None };
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(origin);
        let local_d = inv.transform_vector3(dir).normalize_or_zero();
        let mesh = &self.doc.scene.objects[idx].mesh;
        let mut best: Option<(f32, glam::Vec2)> = None;
        for tri in mesh.indices.chunks_exact(3) {
            if tri.iter().any(|&index| index as usize >= mesh.vertices.len()) { continue; }
            let a = glam::Vec3::from(mesh.vertices[tri[0] as usize].position);
            let b = glam::Vec3::from(mesh.vertices[tri[1] as usize].position);
            let c = glam::Vec3::from(mesh.vertices[tri[2] as usize].position);
            // Möller–Trumbore
            let e1 = b - a;
            let e2 = c - a;
            let p = local_d.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-10 { continue; }
            let inv_det = 1.0 / det;
            let tvec = local_o - a;
            let u = tvec.dot(p) * inv_det;
            if !(0.0..=1.0).contains(&u) { continue; }
            let q = tvec.cross(e1);
            let v = local_d.dot(q) * inv_det;
            if v < 0.0 || u + v > 1.0 { continue; }
            let t = e2.dot(q) * inv_det;
            if t <= 1e-5 { continue; }
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
        let (vx, vy, vw, vh) = (self.viewport[0], self.viewport[1], self.viewport[2], self.viewport[3]);
        if vw < 1.0 || vh < 1.0 {
            return;
        }
        let nx = ((self.cursor.0 as f32 - vx) / vw) * 2.0 - 1.0;
        let ny = 1.0 - ((self.cursor.1 as f32 - vy) / vh) * 2.0;
        let (origin, dir) = self.camera.ray((nx, ny), vw / vh);
        let grab_delta = if matches!(self.sculpt.brush, forge_core::SculptBrush::Grab | forge_core::SculptBrush::SnakeHook | forge_core::SculptBrush::Nudge | forge_core::SculptBrush::Thumb) {
            let dx = (self.cursor.0 - self.press.0) as f32 * 0.0025 * self.camera.distance;
            let dy = (self.press.1 - self.cursor.1) as f32 * 0.0025 * self.camera.distance;
            let view = self.camera.view().inverse();
            let right = view.x_axis.truncate();
            let up = view.y_axis.truncate();
            Some(right * dx + up * dy)
        } else {
            None
        };
        let _ = self.doc.sculpt_stroke(origin, dir, &self.sculpt, grab_delta);
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
        let (vx, vy, vw, vh) = (self.viewport[0], self.viewport[1], self.viewport[2], self.viewport[3]);
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
    fn sync_camera_view(&mut self) {
        let Some(id) = self.camera_view else {
            self.camera.fixed = None;
            return;
        };
        let view = match self.doc.scene.index_of(id) {
            Some(i) => match &self.doc.scene.objects[i].kind {
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
            },
            None => None,
        };
        if view.is_none() {
            self.camera_view = None;
        }
        self.camera.fixed = view;
    }

    fn toggle_camera_view(&mut self) {
        if self.camera_view.is_some() {
            self.camera_view = None;
            self.camera.fixed = None;
            self.status("Left camera view");
            return;
        }
        let is_cam = |o: &forge_core::Object| matches!(o.kind, ObjectKind::Camera(_));
        let from_active = self.doc.active_index().filter(|&i| is_cam(&self.doc.scene.objects[i]));
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
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (o, d) = self.camera.ray((nx, ny), vw / vh);
        match (self.doc.scene.pick(o, d), extend) {
            (Some(i), true) => self.doc.select_toggle(i),
            (Some(i), false) => self.doc.select_only(i),
            (None, false) => self.doc.select_none(),
            (None, true) => {}
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
        let Some((origin, _, _)) = self.edit_selection_world_center() else { return };
        let len = self.edit_gizmo_len(origin);
        let [vx, vy, vw, vh] = self.viewport;
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
            ) else { continue };
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
        let Some(axis) = self.edit_gizmo_hover else { return false };
        let Some(session) = self.doc.edit.as_ref() else { return false };
        let verts = session.snapshot_positions();
        if verts.is_empty() {
            return false;
        }
        let Some((origin, world, _)) = self.edit_selection_world_center() else { return false };
        let axis_dir = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][axis];
        let (ro, rd) = {
            let [vx, vy, vw, vh] = self.viewport;
            let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
            let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
            self.camera.ray((nx, ny), vw / vh)
        };
        // param along axis from ray
        let start = {
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
        self.doc.checkpoint();
        self.edit_gizmo_drag = Some(EditGizmoDrag {
            axis,
            axis_dir,
            origin,
            start_param: start,
            verts,
            inv_world: world.inverse(),
        });
        true
    }

    fn update_edit_gizmo_drag(&mut self) {
        let Some(d) = self.edit_gizmo_drag.as_ref() else { return };
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
        if let Some(session) = self.doc.edit.as_mut() {
            session.set_vert_positions(&positions);
        }
        self.doc.sync_edit_mesh();
    }

    fn end_edit_gizmo_drag(&mut self) {
        self.edit_gizmo_drag = None;
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
        let Some(drag) = self.bone_extrude_drag.as_ref() else { return };
        let bi = drag.bone_idx;
        let head = drag.head;
        let id = match self.doc.active {
            Some(id) => id,
            None => return,
        };
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return };
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

    fn start_bone_pose_rotate(&mut self) {
        if self.doc.active_armature().map(|a| a.active.is_none()).unwrap_or(true) {
            self.status("Select a bone first");
            return;
        }
        self.doc.checkpoint();
        self.bone_pose_drag = Some(self.cursor);
        self.status("Pose rotate — move mouse, LMB confirm, RMB/Esc cancel");
    }

    fn try_pick_bone(&mut self, extend: bool) -> bool {
        let Some(session_arm) = self.doc.active_armature() else { return false };
        let id = self.doc.active.unwrap();
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return false };
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
        let dist = (world.w_axis.truncate() - self.camera.target).length().max(0.5);
        let thresh = dist * 0.12;
        let pick = session_arm.pick_bone(local_o, local_d, thresh);
        let Some(bi) = pick else { return false };
        self.doc.checkpoint();
        if let Some(a) = self.doc.active_armature_mut() {
            a.select_bone(bi, extend);
        }
        self.status(format!("Bone {}", bi + 1));
        true
    }

    /// Rotate active bone pose with mouse delta (call from ModalRotate when armature selected).
    fn pose_rotate_active_bone(&mut self, dx: f32, dy: f32) {
        let Some(a) = self.doc.active_armature_mut() else { return };
        let Some(ai) = a.active else { return };
        let Some(bone) = a.bones.get_mut(ai as usize) else { return };
        let q = glam::Quat::from_xyzw(bone.pose_rot[0], bone.pose_rot[1], bone.pose_rot[2], bone.pose_rot[3]);
        let delta = glam::Quat::from_euler(glam::EulerRot::YXZ, dx * 0.01, dy * 0.01, 0.0);
        let nq = (delta * q).normalize();
        bone.pose_rot = [nq.x, nq.y, nq.z, nq.w];
        drop(a);
        self.doc.apply_all_skinning();
    }

    fn try_edit_click_drag(&mut self, extend: bool) -> bool {
        let Some(session) = self.doc.edit.as_ref() else { return false };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return false };
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
        let dist = (world.w_axis.truncate() - self.camera.target).length().max(0.5);
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
        let Some(session) = self.doc.edit.as_ref() else { return false };
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
        let Some(session) = self.doc.edit.as_ref() else { return };
        let verts = session.snapshot_positions();
        if verts.is_empty() {
            self.status("Nothing selected");
            return;
        }
        let center = session.selection_center();
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return };
        let world = self.doc.scene.world_matrices()[idx];
        let inv_world = world.inverse();
        self.doc.checkpoint();
        self.edit_xform = Some(EditXform {
            kind,
            start_cursor: self.cursor,
            verts,
            center,
            inv_world,
            confirm_on_release,
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
        let Some(xf) = self.edit_xform.as_ref() else { return };
        let dx = (self.cursor.0 - xf.start_cursor.0) as f32;
        let dy = (self.cursor.1 - xf.start_cursor.1) as f32;
        let inv_view = self.camera.view().inverse();
        let right = inv_view.x_axis.truncate().normalize_or_zero();
        let up = inv_view.y_axis.truncate().normalize_or_zero();
        let dist = (self.camera.target - inv_view.w_axis.truncate()).length().max(0.1);

        let positions: Vec<(u32, glam::Vec3)> = match xf.kind {
            EditXformKind::Move => {
                let scale = dist * 0.0025;
                let world_delta = right * (dx * scale) + up * (-dy * scale);
                let local_delta = xf.inv_world.transform_vector3(world_delta);
                xf.verts.iter().map(|(vi, p)| (*vi, *p + local_delta)).collect()
            }
            EditXformKind::Rotate => {
                let angle = dx * 0.01;
                let axis = xf.inv_world.transform_vector3(
                    inv_view.z_axis.truncate().normalize_or_zero()
                ).normalize_or_zero();
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
        if let Some(session) = self.doc.edit.as_mut() {
            session.set_vert_positions(&positions);
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
        let Some(session) = self.doc.edit.as_ref() else { return };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return };
        let [vx, vy, vw, vh] = self.viewport;
        let nx = (self.cursor.0 as f32 - vx) / vw * 2.0 - 1.0;
        let ny = 1.0 - (self.cursor.1 as f32 - vy) / vh * 2.0;
        let (ro, rd) = self.camera.ray((nx, ny), vw / vh);
        let world = self.doc.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(ro);
        let local_d = inv.transform_vector3(rd);
        // threshold ~ 12px in world units near the object
        let dist = (world.w_axis.truncate() - self.camera.target).length().max(0.5);
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
        self.sel_drag = Some(SelDrag { tool: self.ts.tool, start: self.cursor, points: vec![p], op });
        if self.ts.tool == Tool::Circle {
            if op == SelectOp::Replace {
                self.doc.select_none();
            }
            self.circle_paint(op);
        }
    }

    fn circle_paint(&mut self, op: SelectOp) {
        let c = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let r = self.ts.circle_radius;
        let hits: Vec<usize> = anchors(&self.doc, &self.camera, self.viewport).into_iter().filter(|(_, p)| (*p - c).length() <= r).map(|(i, _)| i).collect();
        apply_selection(&mut self.doc, &hits, if op == SelectOp::Subtract { SelectOp::Subtract } else { SelectOp::Extend });
    }

    fn update_sel_drag(&mut self) {
        let p = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let (tool, op) = match self.sel_drag.as_mut() {
            Some(d) => {
                if d.tool == Tool::Lasso && d.points.last().map_or(true, |l| (*l - p).length() > 3.0) {
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
        let Some(d) = self.sel_drag.take() else { return };
        if d.tool == Tool::Circle {
            return;
        }
        let cur = Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32);
        let start = Vec2::new(d.start.0 as f32, d.start.1 as f32);
        if (cur - start).length() < 4.0 {
            self.pick(d.op == SelectOp::Extend);
            return;
        }
        let (lo, hi) = (start.min(cur), start.max(cur));
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

    fn finish_edit_sel_rect(&mut self, lo: Vec2, hi: Vec2, extend: bool) {
        let Some(session) = self.doc.edit.as_mut() else { return };
        let id = session.object_id;
        let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == id) else { return };
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
            Tool::BoxSelect => Some(SelShape::Box { a: Vec2::new(d.start.0 as f32, d.start.1 as f32), b: cur }),
            Tool::Lasso => Some(SelShape::Lasso(d.points.clone())),
            _ => None,
        }
    }

    // ------------------------------------------------------------------ modal transforms
    fn start_modal(&mut self, kind: ModalKind, checkpoint: bool) {
        if self.modal.is_some() {
            return;
        }
        match Modal::start(kind, &mut self.doc, &self.camera, self.viewport, self.cursor, self.ts.pivot, checkpoint) {
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
            m.update(&mut self.doc, &self.camera, self.viewport, self.cursor, &self.ts.snap, ctrl);
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
            let res = std::fs::read(&path).map_err(err_string).and_then(|b| project::from_bytes(&b).map_err(err_string));
            match res {
                Ok(scene) => {
                    self.doc = Document::from_scene(scene, Some(path.clone()));
                    self.recent.add(&path);
                    self.reset_view_state();
                    self.frame(false);
                    self.status(format!("Opened {}", path.display()));
                }
                Err(e) => self.status(format!("Open failed: {e}")),
            }
        }
    }

    fn import_path(&mut self, path: &Path) {
        let ext = ext_of(path);
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
                self.doc.import_scene(scene);
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
                    let Some(mut p) = rfd::FileDialog::new().add_filter("BlendCraft project", &["forge"]).set_file_name("untitled.forge").save_file() else {
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
            match project::to_bytes(&self.doc.scene).and_then(|b| std::fs::write(&path, b).map_err(Into::into)) {
                Ok(()) => {
                    self.doc.path = Some(path.clone());
                    self.doc.dirty = false;
                    self.recent.add(&path);
                    self.status(format!("Saved {}", path.display()));
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
                _ => ("Wavefront OBJ", "obj", "model.obj"),
            };
            let Some(mut p) = rfd::FileDialog::new().add_filter(filter, &[ext]).set_file_name(default).save_file() else {
                return;
            };
            if p.extension().is_none() {
                p.set_extension(ext);
            }
            let only = selected_only.then_some(&self.doc.selection);
            let res = match format {
                "glb" => forge_io::export_glb(&self.doc.scene, &p, only),
                "fbx" => forge_io::export_fbx(&self.doc.scene, &p, only),
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
                        if let Some(p) = rfd::FileDialog::new().add_filter("BlendCraft project", &["forge"]).pick_file() {
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
                    if let Some(p) = rfd::FileDialog::new().add_filter("3D models", &["glb", "gltf", "obj", "fbx"]).pick_file() {
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
                if self.doc.armature_edit && self.doc.active_armature().is_some() {
                    self.doc.armature_delete_bones();
                    self.status("Deleted bones");
                } else if self.doc.edit.is_some() {
                    self.doc.edit_delete();
                    self.status("Deleted selection");
                } else if self.doc.active_armature().is_some() {
                    self.doc.armature_delete_bones();
                    self.status("Deleted bones");
                } else {
                    self.doc.delete_selected();
                }
            },
            A::Extrude => {
                if self.doc.armature_edit && self.doc.active_armature().is_some() {
                    self.start_bone_extrude();
                } else if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
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
            },
            A::InsetFaces => {
                if self.doc.in_edit_mode() {
                    let ok = self.doc.edit.as_ref().map(|e| {
                        !e.selection.faces.is_empty()
                            || (!e.selection.verts.is_empty() && e.selection.domain == forge_core::EditDomain::Vertex)
                    }).unwrap_or(false);
                    if !ok {
                        self.status("Inset: switch to Face select (3) and select faces, or select face verts");
                    } else {
                        self.doc.edit_inset(0.25);
                        self.status("Inset Faces (I)");
                    }
                } else {
                    self.status("Inset: Tab into Edit Mode first");
                }
            },
            A::Bevel => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Bevel: select edges/verts/faces first");
                    } else {
                        self.doc.edit_bevel(0.05);
                        self.status("Bevel (Ctrl+B)");
                    }
                } else {
                    self.status("Bevel: Tab into Edit Mode first");
                }
            },
            A::LoopCut => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| {
                        e.selection.edges.is_empty() && e.selection.faces.is_empty()
                    }).unwrap_or(true);
                    if empty {
                        self.status("Loop Cut: select edges (2) or faces (3) first");
                    } else {
                        self.doc.edit_loop_cut();
                        self.status("Loop Cut (Ctrl+R)");
                    }
                } else {
                    self.status("Loop Cut: Tab into Edit Mode first");
                }
            },
            A::Knife => {
                if self.doc.in_edit_mode() {
                    // Bisect through selection center with view plane normal
                    let center = self.doc.edit.as_ref()
                        .map(|e| e.selection_center())
                        .unwrap_or(glam::Vec3::ZERO);
                    // Transform center to world
                    let world_center = if let Some(session) = self.doc.edit.as_ref() {
                        if let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == session.object_id) {
                            let w = self.doc.scene.world_matrices()[idx];
                            w.transform_point3(center)
                        } else { center }
                    } else { center };
                    let normal = (self.camera.target - self.camera.eye()).normalize_or_zero();
                    // Convert plane into object local space
                    if let Some(session) = self.doc.edit.as_ref() {
                        if let Some(idx) = self.doc.scene.objects.iter().position(|o| o.id == session.object_id) {
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
            },
            A::PolyBuild => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_poly_build(0.25);
                    self.status("Poly Build");
                } else {
                    self.status("Poly Build only works in Edit Mode (select edge/verts)");
                }
            },
            A::Spin => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_spin(8, 360.0);
                    self.status("Spin");
                } else {
                    self.status("Spin only works in Edit Mode");
                }
            },
            A::SmoothVerts => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Smooth: select vertices first (1)");
                    } else {
                        self.doc.edit_smooth(0.5);
                        self.status("Smooth");
                    }
                } else {
                    self.status("Smooth: Tab into Edit Mode first");
                }
            },
            A::EdgeSlide => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Edge Slide: select edges first (2)");
                    } else {
                        self.doc.edit_edge_slide(0.1);
                        self.status("Edge Slide (G G)");
                    }
                } else {
                    self.status("Edge Slide: Tab into Edit Mode first");
                }
            },
            A::ShrinkFatten => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Shrink/Fatten: select verts/faces first");
                    } else {
                        self.doc.edit_shrink_fatten(0.1);
                        self.status("Shrink/Fatten (Alt+S)");
                    }
                } else {
                    self.status("Shrink/Fatten: Tab into Edit Mode first");
                }
            },
            A::Shear => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Shear: select verts first");
                    } else {
                        self.doc.edit_shear(0, 0.25);
                        self.status("Shear (Ctrl+Shift+Alt+S)");
                    }
                } else {
                    self.status("Shear: Tab into Edit Mode first");
                }
            },
            A::RipRegion => {
                if self.doc.in_edit_mode() {
                    let empty = self.doc.edit.as_ref().map(|e| e.selection.is_empty()).unwrap_or(true);
                    if empty {
                        self.status("Rip: select verts first");
                    } else {
                        self.doc.edit_rip();
                        self.status("Rip Region (V)");
                    }
                } else {
                    self.status("Rip: Tab into Edit Mode first");
                }
            },
            A::MergeVerts => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_merge_verts();
                    self.status("Merged vertices");
                }
            },
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
                } else if self.doc.in_edit_mode() {
                    self.doc.edit_select_all();
                } else {
                    self.doc.select_all();
                }
            },
            A::SelectNone => {
                if self.doc.uv_mode {
                    self.doc.uv_select_none();
                } else if self.doc.in_edit_mode() {
                    self.doc.edit_select_none();
                } else {
                    self.doc.select_none();
                }
            },
            A::SelectInvert => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_invert_selection();
                } else {
                    self.doc.select_invert();
                }
            },
            A::SelectRandom => self.doc.select_random(0.5),
            A::SelectChildren => self.doc.select_children(),
            A::SelectParent => self.doc.select_parent(),
            A::SelectSameType => self.doc.select_same(SameKind::Type),
            A::SelectSameCollection => self.doc.select_same(SameKind::Collection),
            A::SelectSameMaterial => self.doc.select_same(SameKind::Material),
            A::SelectLinked => self.doc.select_same(SameKind::Linked),

            A::SetTool(t) => {
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
                if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Move);
                } else {
                    self.start_modal(ModalKind::Move, true);
                }
            },
            A::ModalRotate => {
                if self.doc.active_armature().is_some() && self.doc.edit.is_none() {
                    self.start_bone_pose_rotate();
                } else if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Rotate);
                } else {
                    self.start_modal(ModalKind::Rotate, true);
                }
            },
            A::ModalScale => {
                if self.doc.in_edit_mode() {
                    self.start_edit_xform(EditXformKind::Scale);
                } else {
                    self.start_modal(ModalKind::Scale, true);
                }
            },

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
                self.ui.rename_buf = self.doc.scene.collections.iter().find(|c| c.id == id).map(|c| c.name.clone()).unwrap_or_default();
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
                self.status(if n == 0 { "Nothing to separate (mesh is one piece)".to_string() } else { format!("Separated {n} new object(s)") });
            }
            A::SeparateMaterial => {
                let n = self.doc.separate_material();
                self.status(if n == 0 { "Nothing to separate (needs a mesh with several materials)".to_string() } else { format!("Separated {n} new object(s)") });
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
            },
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
                    self.doc.scene.objects[i].constraints.push(Constraint::new(k));
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
                    if let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg"]).pick_file() {
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
                    if let Some(p) = rfd::FileDialog::new().add_filter("HDR image", &["hdr", "png", "jpg", "jpeg"]).pick_file() {
                        self.load_hdri(&p);
                    }
                }
            }
            A::ResetSky => {
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer.set_environment(&EnvironmentMap::procedural_sky(512, 256));
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
            A::RenderAnimation => self.render_animation(),
            A::InsertKeyframeLocRotScale => self.insert_keyframe(true, true, true),
            A::InsertKeyframeLocation => self.insert_keyframe(true, false, false),
            A::InsertKeyframeRotation => self.insert_keyframe(false, true, false),
            A::InsertKeyframeScale => self.insert_keyframe(false, false, true),
            A::DeleteKeyframe => self.delete_keyframe(),
            A::ToggleAnimationPlay => {
                self.doc.scene.animation.playing = !self.doc.scene.animation.playing;
                if self.doc.scene.animation.playing {
                    self.last_anim_tick = std::time::Instant::now();
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
            A::TogglePaintMode => {
                match self.doc.toggle_paint_mode() {
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
                }
            },
            A::ToggleUvMode => {
                match self.doc.toggle_uv_mode() {
                    Ok(()) => {
                        if self.doc.uv_mode {
                            self.ts.tool = Tool::UvSelect;
                            // Auto smart unwrap if UVs look unset (all near zero)
                            if let Some(id) = self.doc.active {
                                if let Some(obj) = self.doc.scene.objects.iter().find(|o| o.id == id) {
                                    let max_u = obj.mesh.vertices.iter().map(|v| v.uv[0].abs() + v.uv[1].abs()).fold(0.0f32, f32::max);
                                    if max_u < 1e-5 {
                                        self.doc.uv_unwrap_smart();
                                    }
                                }
                            }
                            self.status("UV Editing – unwrap tools in toolbar · select & drag in UV Editor");
                        } else {
                            self.ts.tool = Tool::Select;
                            self.status("Object Mode");
                        }
                    }
                    Err(e) => self.status(e),
                }
            },
            A::UvUnwrapSmart => {
                self.doc.uv_unwrap_smart();
                self.status("Smart UV Project");
            },
            A::UvUnwrapCube => {
                self.doc.uv_unwrap_cube();
                self.status("Cube Projection");
            },
            A::UvUnwrapPlanar => {
                self.doc.uv_unwrap_planar();
                self.status("Project from View (planar)");
            },
            A::UvPack => {
                self.doc.uv_pack();
                self.status("Pack Islands");
            },
            A::UvReset => {
                self.doc.uv_unwrap_reset();
                self.status("Reset UVs");
            },
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
            },
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
                            self.status("Armature Edit Mode — select bone, E extrude tip, X delete".to_string());
                        } else if self.doc.in_edit_mode() {
                            self.ts.tool = Tool::Select;
                            self.status("Edit Mode — select points, then G to move".to_string());
                        } else {
                            self.status("Object Mode".to_string());
                        }
                        self.frame(false);
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
                self.doc.armature_add_bone();
                self.status("Bone added");
            }
            A::ArmatureExtrude => {
                if self.doc.armature_edit {
                    self.start_bone_extrude();
                } else {
                    self.status("Press Tab for Armature Edit Mode, then Extrude");
                }
            }
            A::ArmatureDeleteBones => {
                self.doc.armature_delete_bones();
                self.status("Deleted bones");
            }
            A::ArmatureClearPose => {
                self.doc.armature_clear_pose();
                self.status("Pose cleared");
            }
            A::ArmatureBindMeshes => {
                match self.doc.bind_selected_to_armature() {
                    Ok(n) => self.status(format!("Bound {n} mesh(es) to armature")),
                    Err(e) => self.status(e),
                }
            }

        }
    }

    /// Save UV coordinates as an independent, portable BlendCraft UV-map file.
    fn save_uv_map(&mut self) {
        #[cfg(target_arch = "wasm32")]
        { self.status("Saving UV maps is not available on WebAssembly yet"); return; }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.doc.sync_edit_mesh();
            let Some(id) = self.doc.active else { self.status("Select a mesh before saving its UV map"); return };
            let Some(obj) = self.doc.scene.objects.iter().find(|o| o.id == id) else { self.status("Active object no longer exists"); return };
            if obj.mesh.vertices.is_empty() || obj.mesh.indices.is_empty() { self.status("Active mesh has no UV faces"); return; }
            let file = UvMapFile {
                format: "BlendCraft UV Map".to_string(), version: 1,
                object_name: obj.name.clone(), vertex_count: obj.mesh.vertices.len(),
                positions: obj.mesh.vertices.iter().map(|v| v.position).collect(),
                indices: obj.mesh.indices.clone(),
                uvs: obj.mesh.vertices.iter().map(|v| v.uv).collect(),
            };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("BlendCraft UV Map", &["uvmap"])
                .set_file_name(&format!("{}.uvmap", obj.name.replace('/', "_").replace('\\', "_")))
                .save_file() else { return };
            if path.extension().is_none() { path.set_extension("uvmap"); }
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
        { self.status("Loading UV maps is not available on WebAssembly yet"); return; }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(path) = rfd::FileDialog::new().add_filter("BlendCraft UV Map", &["uvmap"]).pick_file() else { return };
            let loaded: Result<UvMapFile, String> = std::fs::read(&path)
                .map_err(|e| format!("Could not read UV map: {e}"))
                .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| format!("Invalid UV map file: {e}")));
            let file = match loaded { Ok(f) => f, Err(e) => { self.status(e); return; } };
            if file.format != "BlendCraft UV Map" || file.version != 1 {
                self.status("Unsupported UV map file format or version"); return;
            }
            let Some(id) = self.doc.active else { self.status("Select the target mesh before loading a UV map"); return };
            let current_mesh = self.doc.edit.as_ref().filter(|e| e.object_id == id).map(|e| e.mesh.clone())
                .or_else(|| self.doc.scene.objects.iter().find(|o| o.id == id).map(|o| (*o.mesh).clone()));
            let Some(current_mesh) = current_mesh else { self.status("Active object no longer exists"); return };
            if current_mesh.vertices.len() != file.vertex_count || current_mesh.indices != file.indices
                || file.uvs.len() != file.vertex_count || file.positions.len() != file.vertex_count
                || current_mesh.vertices.iter().zip(file.positions.iter()).any(|(v, p)| v.position != *p) {
                self.status("UV map does not match this mesh geometry/topology. Select the original mesh or matching duplicate."); return;
            }
            if file.uvs.iter().flatten().any(|v| !v.is_finite()) {
                self.status("UV map contains invalid coordinates"); return;
            }
            self.doc.checkpoint();
            let mut mesh = current_mesh;
            for (vertex, uv) in mesh.vertices.iter_mut().zip(file.uvs.iter()) { vertex.uv = *uv; }
            if let Some(session) = self.doc.edit.as_mut().filter(|e| e.object_id == id) {
                session.mesh = mesh.clone();
            }
            if let Some(obj) = self.doc.scene.objects.iter_mut().find(|o| o.id == id) {
                obj.mesh = Arc::new(mesh);
                obj.primitive = None;
            }
            self.doc.touch();
            self.status(format!("Loaded UV map '{}' from {}", file.object_name, path.display()));
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
                self.doc.active.and_then(|id| self.doc.scene.objects.iter().find(|o| o.id == id))
                    .map(|o| (*o.mesh).clone())
            };
            let Some(mesh) = mesh else { self.status("Select a mesh before exporting its UV layout"); return };
            if mesh.indices.len() < 3 { self.status("The active mesh has no UV faces"); return; }
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("uv-layout.png")
                .save_file() else { return };
            if path.extension().is_none() { path.set_extension("png"); }

            const SIZE: u32 = 1024;
            let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
            let mut internal_diagonals = std::collections::HashSet::new();
            for q in mesh.quad_faces() {
                let (a, b) = (q[1].min(q[3]), q[1].max(q[3]));
                internal_diagonals.insert((a, b));
            }
            let mut draw_line = |a: glam::Vec2, b: glam::Vec2| {
                if !a.is_finite() || !b.is_finite() { return; }
                // Clip to the standard 0..1 UV tile, as Blender's UV layout export does.
                if a.x < 0.0 || a.x > 1.0 || a.y < 0.0 || a.y > 1.0
                    || b.x < 0.0 || b.x > 1.0 || b.y < 0.0 || b.y > 1.0 { return; }
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
                            if px < 0 || py < 0 || px >= SIZE as i32 || py >= SIZE as i32 { continue; }
                            let i = ((py as u32 * SIZE + px as u32) * 4) as usize;
                            rgba[i..i + 4].copy_from_slice(&[245, 205, 90, 255]);
                        }
                    }
                }
            };
            for tri in mesh.indices.chunks_exact(3) {
                if tri.iter().any(|&i| i as usize >= mesh.vertices.len()) { continue; }
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let edge = (a.min(b), a.max(b));
                    if internal_diagonals.contains(&edge) { continue; }
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
            let Some(id) = self.doc.active else { self.status("No active paint object"); return };
            let image = self.doc.scene.objects.iter().find(|o| o.id == id)
                .and_then(|o| o.material.or(Some(0)))
                .and_then(|mi| self.doc.scene.materials.get(mi))
                .and_then(|m| m.base_color_tex)
                .and_then(|ti| self.doc.scene.textures.get(ti))
                .cloned();
            let Some(image) = image else { self.status("No paint image assigned"); return };
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("paint-texture.png")
                .save_file() else { return };
            if path.extension().is_none() { path.set_extension("png"); }
            match image::save_buffer(&path, &image.rgba8, image.width, image.height, image::ColorType::Rgba8) {
                Ok(()) => self.status(format!("Saved paint image to {}", path.display())),
                Err(e) => self.status(format!("Could not save paint image: {e}")),
            }
        }
    }

    /// Capture the current 3D viewport (no UI) to a PNG the user chooses.

    /// Resolution for offline renders (image / animation).
    fn render_resolution(&self) -> (u32, u32) {
        if self.ui.render_use_viewport || self.ui.render_width < 64 || self.ui.render_height < 64 {
            (
                (self.viewport[2].max(64.0) as u32).clamp(64, 7680),
                (self.viewport[3].max(64.0) as u32).clamp(64, 4320),
            )
        } else {
            let pct = self.ui.render_percent.clamp(1, 200) as f32 / 100.0;
            let w = ((self.ui.render_width.max(64) as f32) * pct).round() as u32;
            let h = ((self.ui.render_height.max(64) as f32) * pct).round() as u32;
            (w.clamp(64, 7680), h.clamp(64, 4320))
        }
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
            let Some(idx) = self.doc.scene.index_of(id) else { continue };
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
        self.status(format!("Inserted keyframe at frame {frame} on {n} object(s)"));
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
        let frame = self.doc.scene.animation.frame_current as f32;
        let keys: Vec<(u64, (Option<glam::Vec3>, Option<glam::Quat>, Option<glam::Vec3>))> = self
            .doc
            .scene
            .animation
            .objects
            .iter()
            .map(|(id, oa)| (*id, oa.eval(frame)))
            .collect();
        for (id, (loc, rot, scl)) in keys {
            let Some(idx) = self.doc.scene.index_of(id) else { continue };
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
        self.doc.touch();
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
            let Some(mut path) = rfd::FileDialog::new()
                .add_filter("PNG image", &["png"])
                .set_file_name("render.png")
                .save_file()
            else {
                return;
            };
            if path.extension().is_none() {
                path.set_extension("png");
            }
            let (w, h) = self.render_resolution();
            // Ensure objects sit on the current timeline frame.
            self.apply_animation_frame();
            if let Some(g) = self.gfx.as_mut() {
                g.renderer.set_scene(&self.doc.scene, &highlights(&self.doc));
            }
            self.uploaded = self.doc.version;
            let cam = self.camera.clone();
            let settings = self.settings.clone();
            let result = {
                let g = self.gfx.as_mut().unwrap();
                g.renderer.capture_with(&cam, &settings, w, h, false)
            };
            match result {
                Ok(rgba) => {
                    if rgba.len() < (w as usize).saturating_mul(h as usize).saturating_mul(4) {
                        self.status("Render failed: incomplete pixel buffer");
                        return;
                    }
                    match image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                        Ok(()) => self.status(&format!("Saved render to {}", path.display())),
                        Err(e) => self.status(&format!("Render failed: {e}")),
                    }
                }
                Err(e) => self.status(&format!("Render failed: {e}")),
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
            let Some(dir) = rfd::FileDialog::new()
                .set_title("Choose folder for animation frames")
                .pick_folder()
            else {
                return;
            };
            if self.gfx.is_none() {
                self.status("Renderer not ready");
                return;
            }
            // Scene animation frames (object keyframes). Fast GPU path: transform-only
            // updates + reused capture targets + no shadow pass while baking video.
            let start = self.doc.scene.animation.frame_start;
            let end = self.doc.scene.animation.frame_end.max(start);
            let restore = self.doc.scene.animation.frame_current;
            let fps = self.doc.scene.animation.fps.max(1);
            let (w, h) = self.render_resolution();
            let mut ok = 0u32;
            let mut err_msg: Option<String> = None;
            let total = (end - start + 1) as u32;
            let t0 = std::time::Instant::now();

            // Ensure GPU scene is fully built once, then only update matrices.
            {
                let highlight = highlights(&self.doc);
                if let Some(g) = self.gfx.as_mut() {
                    g.renderer.set_scene(&self.doc.scene, &highlight);
                }
                self.uploaded = self.doc.version;
            }

            let frames_dir = dir.join("frames");
            let _ = std::fs::create_dir_all(&frames_dir);

            // ---- Speed strategy (target ~2+ fps on typical hardware) ----
            // 1. Fast GPU path already skips shadows and reuses offscreen targets.
            // 2. Do NOT write a PNG every frame by default when video export works
            //    (PNG compress + disk is often slower than the GPU itself).
            // 3. JPEG encoding runs on a pool of worker threads so the next frame can
            //    start rendering while previous ones are compressed in parallel.
            // 4. The in-flight queue is bounded so RAM stays reasonable.
            // Resolution still comes from Render settings — use 50% / 1280×720
            // if you need more speed on weaker GPUs.
            let write_png_frames = false; // set true only if you need the image sequence

            // Pure-Rust MJPEG/AVI writer (crates/video): frames are JPEG-compressed on a
            // pool of worker threads and streamed to disk, so no C/C++ encoder is needed.
            let avi_path = dir.join("animation.avi");
            let mut video = match forge_video::VideoWriter::create(&avi_path, w, h, fps, 90) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::warn!("video writer init failed: {e}");
                    None
                }
            };

            for frame in start..=end {
                self.doc.scene.animation.frame_current = frame;
                self.doc.scene.animation.playing = false;
                self.apply_animation_frame();

                let highlight = highlights(&self.doc);
                let cam = self.camera.clone();
                let settings = self.settings.clone();
                let result = {
                    let g = self.gfx.as_mut().unwrap();
                    // Transform-only upload (avoids rebuilding materials/meshes every frame).
                    g.renderer.update_transforms(&self.doc.scene, &highlight);
                    // Fast capture: skip shadows, reuse offscreen targets.
                    g.renderer.capture_with(&cam, &settings, w, h, true)
                };
                match result {
                    Ok(rgba) => {
                        ok += 1;
                        // PNG sequence only when requested, or as a fallback if the
                        // video writer could not be created.
                        if write_png_frames || video.is_none() {
                            let path = frames_dir.join(format!("frame_{:04}.png", frame));
                            let _ = image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8);
                        }
                        if let Some(v) = video.as_mut() {
                            // Blocks only if the encoder queue is full (bounds RAM).
                            if let Err(e) = v.push_rgba(rgba) {
                                err_msg = Some(format!("Video encoding failed: {e}"));
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        err_msg = Some(format!("Animation render failed at frame {frame}: {e}"));
                        break;
                    }
                }
            }

            // Restore timeline position and transforms.
            self.doc.scene.animation.frame_current = restore;
            self.apply_animation_frame();
            if let Some(g) = self.gfx.as_mut() {
                g.renderer.update_transforms(&self.doc.scene, &highlights(&self.doc));
            }
            self.uploaded = self.doc.version;

            let elapsed = t0.elapsed().as_secs_f32();
            let fps_render = if elapsed > 0.0 { ok as f32 / elapsed } else { 0.0 };

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
                self.status(msg);
            } else {
                self.status(format!(
                    "Rendered {ok}/{total} frames ({w}×{h}) in {elapsed:.1}s ({fps_render:.1} fps) · {start}–{end}{video_note}"
                ));
            }
        }
    }

    fn shortcut(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        if code == KeyCode::Escape {
            if self.bone_extrude_drag.is_some() {
                self.cancel_bone_extrude();
                return;
            }
            if self.bone_pose_drag.is_some() {
                self.bone_pose_drag = None;
                self.doc.undo();
                self.status("Pose cancelled");
                return;
            }
            if self.edit_xform.is_some() {
                self.cancel_edit_xform();
                return;
            }
            self.sel_drag = None;
            if self.doc.armature_edit {
                self.doc.exit_armature_edit();
                self.status("Object Mode");
                return;
            }
            if self.doc.in_edit_mode() {
                self.doc.exit_edit_mode(true);
                self.status("Object Mode");
            }
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
        if let Some(a) = keymap::find_action(&self.bindings, ctrl, shift, alt, code) {
            self.apply(a, el);
        }
    }

    // ------------------------------------------------------------------ input
    fn mouse_button(&mut self, state: ElementState, button: MouseButton, consumed: bool) {
        let down = state == ElementState::Pressed;
        if self.bone_extrude_drag.is_some() {
            if button == MouseButton::Left && !down {
                self.confirm_bone_extrude();
            } else if button == MouseButton::Right && down {
                self.cancel_bone_extrude();
            }
            return;
        }
        if self.bone_pose_drag.is_some() {
            if button == MouseButton::Left && !down {
                self.bone_pose_drag = None;
                self.doc.apply_all_skinning();
                self.status("Pose applied");
            } else if button == MouseButton::Right && down {
                self.bone_pose_drag = None;
                self.doc.undo(); // revert checkpoint
                self.status("Pose cancelled");
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
                                self.measure_b = Some(hit);
                                let a = self.measure_a.unwrap();
                                let dist = (hit - a).length();
                                self.status(format!("Distance: {:.4} m", dist));
                                // keep for overlay; next click starts new measure
                                self.measure_a = Some(hit);
                                self.measure_b = None;
                            }
                        }
                        return;
                    }
                    // --- Annotate tool ---
                    if self.ts.tool == Tool::Annotate {
                        if let Some(hit) = self.raycast_ground_or_mesh() {
                            if self.annotate_stroke.is_none() {
                                self.annotate_stroke = Some(vec![hit]);
                            } else if let Some(s) = self.annotate_stroke.as_mut() {
                                s.push(hit);
                            }
                            self.status("Annotate: drag to draw, release to finish");
                        }
                        self.left = true;
                        self.press = self.cursor;
                        self.dragged = false;
                        return;
                    }
                    // --- Transform tool: use move gizmo ---
                    if self.ts.tool == Tool::Transform {
                        self.ts.tool = Tool::Move;
                        if self.gizmo.begin(self.cursor, &self.camera, self.viewport, &mut self.doc, &self.ts) {
                            return;
                        }
                        self.ts.tool = Tool::Transform;
                    }
                    if self.doc.active_armature().is_some() && self.doc.edit.is_none() {
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
                        if self.begin_edit_gizmo_drag() {
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
                        && self.gizmo.begin(self.cursor, &self.camera, self.viewport, &mut self.doc, &self.ts)
                    {
                        return;
                    }
                    match self.ts.tool {
                        Tool::BoxSelect | Tool::Lasso | Tool::Circle => self.begin_sel_drag(shift, ctrl),
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
                        if let Some(stroke) = self.annotate_stroke.take() {
                            if stroke.len() >= 2 {
                                self.annotations.push(stroke);
                                self.status(format!("Annotation saved ({} strokes)", self.annotations.len()));
                            }
                        }
                    } else if !self.dragged {
                        self.pick(shift);
                    }
                }
            }
            MouseButton::Right => {
                if down {
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
        let (shift, ctrl) = (self.mods.shift_key(), self.ctrl());
        // Texture painting is a continuous stroke, not a single click stamp.
        // Interpolate screen-space samples so fast pointer movement cannot leave gaps.
        if self.paint_stroke {
            let from = self.paint_last_cursor.unwrap_or((x, y));
            let distance = ((x - from.0).powi(2) + (y - from.1).powi(2)).sqrt();
            let steps = (distance / 4.0).ceil().max(1.0) as usize;
            let mut uvs = Vec::with_capacity(steps);
            for step in 1..=steps {
                let t = step as f64 / steps as f64;
                let sample = (from.0 + (x - from.0) * t, from.1 + (y - from.1) * t);
                if let Some(uv) = self.paint_sample_at(sample) { uvs.push(uv); }
            }
            if !uvs.is_empty() {
                let color = if self.ts.tool == Tool::PaintErase { [0.78, 0.78, 0.78, 1.0] } else { self.ts.paint_color };
                let _ = self.doc.paint_stroke_uv(&uvs, color, self.ts.paint_radius, self.ts.paint_strength);
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
        if self.ts.tool == Tool::Annotate && self.left {
            if let Some(hit) = self.raycast_ground_or_mesh() {
                if let Some(s) = self.annotate_stroke.as_mut() {
                    if s.last().map(|p| (*p - hit).length() > 0.02).unwrap_or(true) {
                        s.push(hit);
                    }
                }
            }
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
        if let Some(start) = self.bone_pose_drag {
            let dx = (self.cursor.0 - start.0) as f32;
            let dy = (self.cursor.1 - start.1) as f32;
            // reset to checkpoint pose then apply? simpler: accumulate from start by restoring undo top then applying
            // For smooth pose: store initial pose on start. For now apply delta from last frame:
            self.pose_rotate_active_bone(dx * 0.15, dy * 0.15);
            self.bone_pose_drag = Some(self.cursor);
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
        if self.gizmo.dragging() && !self.doc.in_edit_mode() {
            self.gizmo.update(self.cursor, &self.camera, self.viewport, &mut self.doc, &self.ts, ctrl);
            return;
        }
        if self.sel_drag.is_some() {
            self.update_sel_drag();
            return;
        }
        if !self.left && !self.middle && !self.right {
            if self.doc.in_edit_mode() {
                self.edit_gizmo_hover_test();
            } else {
                self.gizmo.hover_test(self.cursor, &self.camera, self.viewport, &self.doc, &self.ts);
            }
        }
        if self.left {
            let d = ((x - self.press.0).powi(2) + (y - self.press.1).powi(2)).sqrt();
            if d > 4.0 {
                self.dragged = true;
            }
            if self.dragged {
                if shift { self.camera.pan(dx, dy) } else { self.camera.orbit(dx, dy) }
            }
        }
        if self.middle {
            if shift { self.camera.pan(dx, dy) } else { self.camera.orbit(dx, dy) }
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
        self.sync_camera_view();

        // Advance timeline when playing (real-time FPS).
        // Accumulates time so slow frames still keep overall speed correct:
        // at 24 fps, 24 steps ≈ 1 second of wall clock.
        if self.doc.scene.animation.playing {
            let fps = self.doc.scene.animation.fps.max(1) as f32;
            let frame_dt = 1.0 / fps;
            let mut elapsed = self.last_anim_tick.elapsed().as_secs_f32();
            // Cap catch-up so a long hitch does not jump the whole timeline at once.
            let max_steps = 8u32;
            let mut steps = 0u32;
            while elapsed >= frame_dt && steps < max_steps {
                elapsed -= frame_dt;
                self.doc.scene.animation.step();
                steps += 1;
            }
            if steps > 0 {
                // Keep residual time so we do not drift slow.
                self.last_anim_tick = std::time::Instant::now()
                    - std::time::Duration::from_secs_f32(elapsed.max(0.0));
                self.apply_animation_frame();
            }
        }

        // 1) build the UI
        let raw = match self.gfx.as_mut() {
            Some(g) => g.egui_state.take_egui_input(&g.window),
            None => return,
        };
        let shape = self.sel_shape();
        let modal_text = self.modal.as_ref().map(|m| m.status());
        let circle = (self.ts.tool == Tool::Circle && self.in_viewport(self.cursor))
            .then(|| (Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32), self.ts.circle_radius));
        let mut actions = Vec::new();
        let mut rect = egui::Rect::NOTHING;
        let out = self.egui_ctx.run(raw, |ctx| {
            let (a, r) = ui::draw(ctx, &mut self.doc, &mut self.settings, &mut self.ts, &mut self.ui, &self.bindings, &self.recent.paths);
            tools::draw_toolbar(ctx, r, &mut self.ts);
            let measure = match (self.measure_a, self.measure_b) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            };
            let extra = overlay::Extra {
                shape: shape.as_ref(),
                modal: modal_text.as_deref(),
                circle,
                measure,
                annotations: &self.annotations,
                annotate_stroke: self.annotate_stroke.as_deref(),
            };
            overlay::draw(ctx, &self.camera, r, &self.doc, &extra);
            if self.doc.in_edit_mode() {
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
            actions = a;
            rect = r;
        });
        if let Some(g) = self.gfx.as_mut() {
            g.egui_state.handle_platform_output(&g.window, out.platform_output.clone());
        }
        let ppp = out.pixels_per_point;
        self.viewport = [rect.min.x * ppp, rect.min.y * ppp, rect.width() * ppp, rect.height() * ppp];

        // 2) apply menu actions
        for a in actions {
            self.apply(a, el);
        }
        self.sync_camera_view();

        // 3) window title
        let name = self.doc.path.as_ref().and_then(|p| p.file_name()).map_or("Untitled".to_string(), |n| n.to_string_lossy().into_owned());
        let title = format!("{}{} - BlendCraft", name, if self.doc.dirty { "*" } else { "" });
        if title != self.title {
            if let Some(g) = &self.gfx {
                g.window.set_title(&title);
            }
            self.title = title;
        }

        // 4) sync scene to the GPU when something changed
        let Some(g) = self.gfx.as_mut() else { return };
        if self.doc.version != self.uploaded {
            let hl = highlights(&self.doc);
            g.renderer.set_scene(&self.doc.scene, &hl);
            self.uploaded = self.doc.version;
        }

        // 5) render 3D + egui overlay
        let (w, h) = g.renderer.size();
        let tris = self.egui_ctx.tessellate(out.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: ppp };
        let textures = out.textures_delta;
        let er = &mut g.egui_renderer;
        let res = g.renderer.render(&self.camera, &self.settings, self.viewport, &mut |device, queue, enc, view| {
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
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
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
        });
        if let Err(e) = res {
            log::error!("render error: {e}");
        }
        g.window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let mut attrs = Window::default_attributes()
            .with_title("BlendCraft")
            .with_inner_size(LogicalSize::new(1400, 850));
        // Window icon from the bundled logo (hand-eye OK mark)
        if let Ok(img) = image::load_from_memory(include_bytes!("../logo.png")) {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            if let Ok(icon) = winit::window::Icon::from_rgba(rgba.into_raw(), w, h) {
                attrs = attrs.with_window_icon(Some(icon));
            }
        }
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        let renderer = pollster::block_on(Renderer::new(window.clone(), size.width, size.height)).expect("init GPU");
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            Some(renderer.device().limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1, false);
        self.gfx = Some(Gfx { window, renderer, egui_state, egui_renderer });
        if let Some(p) = self.initial.take() {
            match ext_of(&p).as_str() {
                "forge" => self.open_project(p),
                "glb" | "gltf" | "obj" => {
                    self.doc = Document::from_scene(Default::default(), None);
                    self.import_path(&p);
                    self.doc.dirty = false;
                }
                _ => self.handle_path(p),
            }
        } else {
            self.frame(false);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
            WindowEvent::MouseInput { state, button, .. } => self.mouse_button(state, button, consumed),
            WindowEvent::CursorMoved { position, .. } => self.cursor_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed && self.in_viewport(self.cursor) && self.modal.is_none() {
                    let y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                    };
                    if self.doc.paint_mode {
                        if self.mods.shift_key() {
                            self.ts.paint_strength = (self.ts.paint_strength + y * 0.05).clamp(0.05, 1.0);
                            self.status(format!("Paint strength: {:.2}", self.ts.paint_strength));
                        } else {
                            self.ts.paint_radius = (self.ts.paint_radius * (1.0 + 0.1 * y)).clamp(0.005, 0.25);
                            self.status(format!("Paint radius: {:.3}", self.ts.paint_radius));
                        }
                    } else if self.doc.sculpt_mode {
                        if self.mods.alt_key() {
                            self.sculpt.dyntopo_detail = (self.sculpt.dyntopo_detail - y * 0.05).clamp(0.15, 1.5);
                            self.status(format!(
                                "Dyntopo relative detail: {:.0}%{}",
                                self.sculpt.dyntopo_detail * 100.0,
                                if self.sculpt.dyntopo { "" } else { " – press D to enable" }
                            ));
                        } else if self.mods.shift_key() {
                            self.sculpt.strength = (self.sculpt.strength + y * 0.05).clamp(0.05, 1.0);
                            self.status(format!("Brush strength: {:.2}", self.sculpt.strength));
                        } else {
                            self.sculpt.radius = (self.sculpt.radius * (1.0 + 0.1 * y)).clamp(0.05, 5.0);
                            self.status(format!("Brush radius: {:.2}", self.sculpt.radius));
                        }
                    } else if self.ts.tool == Tool::Circle {
                        self.ts.circle_radius = (self.ts.circle_radius * (1.0 + 0.1 * y)).clamp(6.0, 600.0);
                    } else {
                        self.camera.zoom(y);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !event.repeat => {
                // Undo/Redo via logical character so Ctrl+Z works on German QWERTZ (Z/Y swapped)
                {
                    use winit::keyboard::Key;
                    let ctrl = self.ctrl();
                    let shift = self.mods.shift_key();
                    let is_undo = ctrl && !shift && matches!(
                        &event.logical_key,
                        Key::Character(c) if c.eq_ignore_ascii_case("z")
                    );
                    let is_redo = ctrl && (
                        (shift && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("z")))
                        || matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("y"))
                    );
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

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    fonts_io::load_cached_fonts();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let initial = std::env::args().nth(1).map(PathBuf::from);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(initial)).expect("run");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    fonts_io::load_cached_fonts();
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log");
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(None)).expect("run");
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log");
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let _ = event_loop.run_app(&mut App::new(None));
}
