mod gizmo;
mod keymap;
mod overlay;
mod recent;
mod tools;
mod transform;
mod ui;

use egui_wgpu::wgpu;
use forge_core::{project, Constraint, Document, EnvironmentMap, FixedView, ObjectKind, OrbitCamera, SameKind};
use forge_render::{Highlight, Renderer, ViewSettings};
use glam::{Mat4, Vec2};
use keymap::{Action, Binding};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tools::{anchors, apply_selection, point_in_polygon, SelShape, SelectOp, Tool, ToolState};
use transform::{Modal, ModalKind};
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
        rfd::MessageDialog::new()
            .set_title("Unsaved changes")
            .set_description("The current scene has unsaved changes. Discard them?")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show()
            == rfd::MessageDialogResult::Yes
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

    fn import_path(&mut self, path: &Path) {
        let ext = ext_of(path);
        let res = match ext.as_str() {
            "glb" | "gltf" => forge_io::load_gltf(path),
            "obj" => forge_io::load_obj(path),
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
        let path = match (&self.doc.path, save_as) {
            (Some(p), false) => p.clone(),
            _ => {
                let Some(mut p) = rfd::FileDialog::new().add_filter("Forge3D project", &["forge"]).set_file_name("untitled.forge").save_file() else {
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

    fn export(&mut self, glb: bool, selected_only: bool) {
        let (filter, ext, default) = if glb { ("glTF Binary", "glb", "model.glb") } else { ("Wavefront OBJ", "obj", "model.obj") };
        let Some(mut p) = rfd::FileDialog::new().add_filter(filter, &[ext]).set_file_name(default).save_file() else {
            return;
        };
        if p.extension().is_none() {
            p.set_extension(ext);
        }
        let only = selected_only.then_some(&self.doc.selection);
        let res = if glb { forge_io::export_glb(&self.doc.scene, &p, only) } else { forge_io::export_obj(&self.doc.scene, &p, only) };
        match res {
            Ok(()) => self.status(format!("Exported {}", p.display())),
            Err(e) => self.status(format!("Export failed: {e:#}")),
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
                if self.confirm_discard() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Forge3D project", &["forge"]).pick_file() {
                        self.open_project(p);
                    }
                }
            }
            A::OpenRecent(p) => {
                if !p.exists() {
                    self.recent.remove(&p);
                    self.status(format!("File no longer exists: {}", p.display()));
                } else if self.confirm_discard() {
                    self.open_project(p);
                }
            }
            A::ClearRecent => self.recent.clear(),
            A::Save => self.save(false),
            A::SaveAs => self.save(true),
            A::Import => {
                if let Some(p) = rfd::FileDialog::new().add_filter("3D models", &["glb", "gltf", "obj"]).pick_file() {
                    self.import_path(&p);
                }
            }
            A::ExportGlb => self.export(true, false),
            A::ExportObj => self.export(false, false),
            A::ExportGlbSelected => self.export(true, true),
            A::ExportObjSelected => self.export(false, true),
            A::Quit => {
                if self.confirm_discard() {
                    el.exit();
                }
            }

            A::Undo => {
                self.doc.undo();
            }
            A::Redo => {
                self.doc.redo();
            }
            A::Duplicate => {
                if self.doc.duplicate(false) {
                    self.start_modal(ModalKind::Move, false);
                }
            }
            A::DuplicateLinked => {
                if self.doc.duplicate(true) {
                    self.start_modal(ModalKind::Move, false);
                }
            }
            A::Delete => self.doc.delete_selected(),
            A::Rename => {
                if let Some(i) = self.doc.active_index() {
                    self.ui.renaming = Some(Rename::Object(self.doc.scene.objects[i].id));
                    self.ui.rename_buf = self.doc.scene.objects[i].name.clone();
                    self.ui.rename_focus = true;
                }
            }
            A::MakeSingleUser => self.doc.make_single_user(),

            A::SelectAll => self.doc.select_all(),
            A::SelectNone => self.doc.select_none(),
            A::SelectInvert => self.doc.select_invert(),
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
            A::ModalMove => self.start_modal(ModalKind::Move, true),
            A::ModalRotate => self.start_modal(ModalKind::Rotate, true),
            A::ModalScale => self.start_modal(ModalKind::Scale, true),

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
            A::Apply { loc, rot, scale } => self.doc.apply_transform(loc, rot, scale),
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
            A::LoadHdri => {
                if let Some(p) = rfd::FileDialog::new().add_filter("HDR image", &["hdr", "png", "jpg", "jpeg"]).pick_file() {
                    self.load_hdri(&p);
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

            A::ToggleShortcuts => self.ui.show_shortcuts = !self.ui.show_shortcuts,
            A::ShowAbout => self.ui.show_about = true,
        }
    }

    fn shortcut(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        if code == KeyCode::Escape {
            self.sel_drag = None;
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
                    if self.ts.tool.is_gizmo() && self.gizmo.begin(self.cursor, &self.camera, self.viewport, &mut self.doc, &self.ts) {
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
        if self.modal.is_some() {
            self.update_modal();
            return;
        }
        if self.gizmo.dragging() {
            self.gizmo.update(self.cursor, &self.camera, self.viewport, &mut self.doc, &self.ts, ctrl);
            return;
        }
        if self.sel_drag.is_some() {
            self.update_sel_drag();
            return;
        }
        if !self.left && !self.middle && !self.right {
            self.gizmo.hover_test(self.cursor, &self.camera, self.viewport, &self.doc, &self.ts);
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
            self.gizmo.draw(ctx, &self.camera, r, &self.doc, &self.ts);
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
        let title = format!("{}{} - Forge3D", name, if self.doc.dirty { "*" } else { "" });
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
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("Forge3D").with_inner_size(LogicalSize::new(1400, 850)))
                .expect("create window"),
        );
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
                    } else if !consumed && !self.egui_ctx.wants_keyboard_input() {
                        self.shortcut(code, el);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.redraw(el),
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let initial = std::env::args().nth(1).map(PathBuf::from);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(initial)).expect("run");
}
