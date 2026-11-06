mod gizmo;
mod keymap;
mod overlay;
mod recent;
mod fonts_io;
mod tools;
mod transform;
mod ui;

use egui_wgpu::wgpu;
use forge_core::{project, Constraint, Document, EditDomain, EnvironmentMap, FixedView, ObjectKind, OrbitCamera, SameKind};
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
    sel_drag: Option<SelDrag>,
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
            sel_drag: None,
            ui: UiState::default(),
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
        // Prefer grabbing vertices regardless of current domain
        let thresh = dist * 0.06;
        let mut tmp = session.clone();
        tmp.selection.domain = forge_core::EditDomain::Vertex;
        let pick_vert = tmp.pick_component(local_o, local_d, thresh);
        let pick_other = session.pick_component(local_o, local_d, thresh);
        let pick = pick_vert.or(pick_other);
        let Some(pick) = pick else {
            if !extend {
                self.doc.edit_select_none();
            }
            return false;
        };
        if matches!(pick, forge_core::EditPick::Vertex(_)) {
            if let Some(e) = self.doc.edit.as_mut() {
                e.selection.domain = forge_core::EditDomain::Vertex;
            }
        }
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
                if self.edit_xform.is_some() {
                    self.cancel_edit_xform();
                } else if self.doc.undo() {
                    self.status("Undo");
                } else {
                    self.status("Nothing to undo");
                }
            }
            A::Redo => {
                if self.doc.redo() {
                    self.status("Redo");
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
                if self.doc.in_edit_mode() {
                    self.doc.edit_delete();
                    self.status("Deleted selection");
                } else {
                    self.doc.delete_selected();
                }
            },
            A::Extrude => {
                if self.doc.in_edit_mode() {
                    self.doc.edit_extrude(0.25);
                    self.status("Extruded");
                } else {
                    self.status("Extrude only works in Edit Mode");
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
                if self.doc.in_edit_mode() {
                    self.doc.edit_select_all();
                } else {
                    self.doc.select_all();
                }
            },
            A::SelectNone => {
                if self.doc.in_edit_mode() {
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
                if self.doc.in_edit_mode() {
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

            A::ToggleShortcuts => self.ui.show_shortcuts = !self.ui.show_shortcuts,
            A::ShowAbout => self.ui.show_about = true,
            A::ToggleEditMode => {
                // Drop any object-gizmo drag so it cannot move the whole object
                if self.gizmo.dragging() {
                    self.gizmo.end();
                }
                self.edit_xform = None;
                match self.doc.toggle_edit_mode() {
                    Ok(()) => {
                        if self.doc.in_edit_mode() {
                            // Stay off the object Move/Rotate/Scale tools while editing verts
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

        }
    }

    /// Capture the current 3D viewport (no UI) to a PNG the user chooses.
    fn render_image(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.status("Render Image is not available on WebAssembly yet".to_string());
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(g) = self.gfx.as_mut() else {
                self.status("Renderer not ready");
                return;
            };
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
            let (w, h) = (
                self.viewport[2].max(64.0) as u32,
                self.viewport[3].max(64.0) as u32,
            );
            match g.renderer.capture(&self.camera, &self.settings, w, h) {
                Ok(rgba) => {
                    if let Err(e) = image::save_buffer(&path, &rgba, w, h, image::ColorType::Rgba8) {
                        self.status(&format!("Render failed: {e}"));
                    } else {
                        self.status(&format!("Saved render to {}", path.display()));
                    }
                }
                Err(e) => self.status(&format!("Render failed: {e}")),
            }
        }
    }

    fn shortcut(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        if code == KeyCode::Escape {
            if self.edit_xform.is_some() {
                self.cancel_edit_xform();
                return;
            }
            self.sel_drag = None;
            if self.doc.in_edit_mode() {
                self.doc.exit_edit_mode(true);
                self.status("Object Mode");
            }
            return;
        }
        let (ctrl, shift, alt) = (self.ctrl(), self.mods.shift_key(), self.mods.alt_key());
        if let Some(a) = keymap::find_action(&self.bindings, ctrl, shift, alt, code) {
            self.apply(a, el);
        }
    }

    // ------------------------------------------------------------------ input
    fn mouse_button(&mut self, state: ElementState, button: MouseButton, consumed: bool) {
        let down = state == ElementState::Pressed;
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
                    if self.doc.in_edit_mode() {
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
                } else if self.edit_gizmo_drag.is_some() {
                    self.end_edit_gizmo_drag();
                } else if self.gizmo.dragging() {
                    self.gizmo.end();
                } else if self.sel_drag.is_some() {
                    self.finish_sel_drag();
                } else if self.left {
                    self.left = false;
                    if !self.dragged {
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
                self.camera.pan(dx, dy);
            }
        }
    }

    // ------------------------------------------------------------------ frame
    fn redraw(&mut self, el: &ActiveEventLoop) {
        self.sync_camera_view();

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
            let extra = overlay::Extra { shape: shape.as_ref(), modal: modal_text.as_deref(), circle };
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
                    if self.ts.tool == Tool::Circle {
                        self.ts.circle_radius = (self.ts.circle_radius * (1.0 + 0.1 * y)).clamp(6.0, 600.0);
                    } else {
                        self.camera.zoom(y);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !event.repeat => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if self.modal.is_some() {
                        self.modal_key(code);
                    } else if !self.egui_ctx.wants_keyboard_input() {
                        // egui marks Tab as "consumed" for focus cycling — still allow our mode shortcuts
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
