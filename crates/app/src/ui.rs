//! egui interface: menu bar, outliner (collections + hierarchy), properties, status bar, shortcuts window.
use crate::keymap::{self, Action, Binding};
use crate::scripting::{ScriptCompletion, ScriptResult, ScriptSnapshot};
use crate::tools::{Tool, ToolState};
use egui::RichText;
use forge_core::shader_graph::{evaluate_graph_ex_cancellable, GraphEvalResult};
use forge_core::{
    apply_eval_to_material, bake_graph_texture, font_registry, free_font_catalog, AlignMode,
    AlignTo, AlphaMode, AnimationData, ArmatureData, CameraData, Constraint, ConstraintKind,
    Document, EmptyShape, ForceFieldData, ForceFieldType, GreasePencilData, LightData,
    LightProbeData, LightProbeType, LightType, Material, ObjectKind, OriginMode, Primitive,
    ShaderGraph, ShaderLink, ShaderNode, ShaderNodeKind, SpeakerData, SurfaceData, TextFont,
    TextureData, VolumeData,
};
use forge_core::{CompositorGraph, CompositorNodeKind};
use forge_render::{ShadingMode, ViewSettings, WireMode};
use glam::{EulerRot, Quat, Vec3};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
struct ShaderEvalKey {
    graph: ShaderGraph,
    material: usize,
    frame: i32,
    texture_id: usize,
}

struct ShaderEvalCompletion {
    key: ShaderEvalKey,
    supported: bool,
    textures: Vec<TextureData>,
    result: Option<GraphEvalResult>,
}

struct ShaderBakeCompletion {
    graph: ShaderGraph,
    material: usize,
    texture_id: usize,
    texture: TextureData,
}

pub struct CompositorEvalCompletion {
    graph: CompositorGraph,
    render_serial: u64,
    width: u32,
    height: u32,
    rgba: Option<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rename {
    Object(u64),
    Collection(u64),
}

#[derive(Default)]
pub struct UiState {
    pub show_shortcuts: bool,
    pub show_about: bool,
    pub show_font_library: bool,
    pub font_filter: String,
    pub font_status: String,
    pub show_move_popup: bool,
    /// Position of the shared Object Mode context menu opened inside the 3D viewport.
    pub viewport_context_menu: Option<egui::Pos2>,
    pub viewport_context_menu_rect: Option<egui::Rect>,
    pub status: String,
    pub shortcut_filter: String,
    pub outliner_filter: String,
    pub renaming: Option<Rename>,
    pub rename_buf: String,
    pub rename_focus: bool,
    /// Persistent egui handles for the 2D Texture Paint canvas. Keep handles alive
    /// across workspace switches so wgpu never submits a frame with a freed texture.
    pub paint_previews: HashMap<(usize, u32, u32), egui::TextureHandle>,
    pub paint_preview_versions: HashMap<(usize, u32, u32), u64>,
    pub paint_canvas_last_uv: Option<glam::Vec2>,
    /// Animation workspace: Dope Sheet + Timeline focused layout (Blender-like).
    pub animation_mode: bool,
    /// Rendering workspace: full render/output settings (Blender Render + Output props).
    pub rendering_mode: bool,
    /// Compositing workspace visibility.
    pub compositing_mode: bool,
    /// Python Scripting workspace visibility.
    pub scripting_mode: bool,
    pub python_script: String,
    pub python_script_output: String,
    pub python_script_path: Option<std::path::PathBuf>,
    pub python_script_rx: Option<Receiver<ScriptCompletion>>,
    pub python_script_running: bool,
    /// Shading workspace visibility and editable node-graph session state.
    pub shading_mode: bool,
    /// Shader editor canvas pan (pixels) and zoom (1 = 100%).
    pub shader_pan: [f32; 2],
    pub shader_zoom: f32,
    /// Previous frame node screen rects (so pan does not steal node drags).
    pub shader_last_node_rects: Vec<(u64, egui::Rect)>,
    pub shader_nodes: Vec<ShaderNode>,
    pub shader_links: Vec<ShaderLink>,
    pub shader_next_id: u64,
    pub shader_socket_pick: Option<(u64, bool, u8)>,
    pub shader_rgb_color: [u8; 4],
    pub shader_image_index: Option<usize>,
    pub shading_nodes_collapsed: bool,
    pub shader_drag_from: Option<(u64, bool, u8)>,
    pub shader_socket_positions: HashMap<(u64, bool, u8), [f32; 2]>,
    /// Cached egui textures for Image Texture node previews (keyed by texture index).
    pub shader_image_previews: HashMap<usize, egui::TextureHandle>,
    pub shader_image_preview_versions: HashMap<usize, u64>,
    /// Which material index the current UI graph belongs to.
    pub shader_material_mi: Option<usize>,
    pub shader_last_eval_graph: Option<ShaderGraph>,
    pub shader_last_eval_material: Option<usize>,
    pub shader_last_eval_frame: Option<i32>,
    pub shader_last_eval_texture_id: usize,
    pub shader_apply_pending: bool,
    pub shader_bake_pending: bool,
    shader_eval_rx: Option<Receiver<ShaderEvalCompletion>>,
    shader_eval_key: Option<ShaderEvalKey>,
    shader_eval_cancel: Option<Arc<AtomicBool>>,
    shader_eval_failed_key: Option<ShaderEvalKey>,
    shader_bake_rx: Option<Receiver<ShaderBakeCompletion>>,
    /// Offline render / output settings (Blender-style).
    pub render_width: u32,
    pub render_height: u32,
    /// 1–200; final pixels = resolution × percent / 100.
    pub render_percent: u32,
    pub render_use_viewport: bool,
    pub render_anim_frames: u32,
    pub render_fps: u32,
    /// Blender-like render property UI state (Rendering workspace).
    pub render_engine: u8, // 0 Workbench, 1 EEVEE, 2 Cycles (UI only where unsupported)
    pub render_samples_viewport: u32,
    pub render_samples: u32,
    pub render_temporal_reprojection: bool,
    pub render_jittered_shadows: bool,
    pub render_shadows: bool,
    pub render_raytracing: bool,
    pub render_motion_blur: bool,
    pub render_dof: bool,
    pub render_film_transparent: bool,
    pub render_simplify: bool,
    pub render_format: u8, // 0 PNG, 1 JPEG, 2 EXR
    pub compositor_pan: [f32; 2],
    pub compositor_zoom: f32,
    pub compositor_render_width: u32,
    pub compositor_render_height: u32,
    pub compositor_render_rgba: Vec<u8>,
    pub compositor_render_serial: u64,
    pub compositor_output_rgba: Vec<u8>,
    pub compositor_output_texture: Option<egui::TextureHandle>,
    pub compositor_output_key: Option<(u64, u64)>,
    pub compositor_eval_error: bool,
    pub compositor_eval_rx: Option<Receiver<CompositorEvalCompletion>>,
    pub compositor_eval_busy: bool,
    pub compositor_eval_requested_graph: Option<CompositorGraph>,
    pub compositor_eval_requested_serial: u64,
    pub compositor_eval_requested_at: Option<std::time::Instant>,
    pub compositor_link_from: Option<u64>,
    pub compositor_last_node_rects: Vec<(u64, egui::Rect)>,
    /// Properties editor context, Blender-like tabs.
    pub properties_tab: PropertiesTab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PropertiesTab {
    #[default]
    Object,
    Render,
    Output,
    Material,
}

/// Records one undo step per continuous edit (drag / typing / click), then applies it.
fn commit(doc: &mut Document, r: &egui::Response, apply: impl FnOnce(&mut Document)) {
    commit_if(doc, r.changed(), apply);
}

fn commit_if(doc: &mut Document, changed: bool, apply: impl FnOnce(&mut Document)) {
    if changed {
        if !doc.edit_open {
            doc.checkpoint();
            doc.edit_open = true;
        }
        apply(doc);
        doc.touch();
    }
}

fn lin_to_srgb8(c: f32) -> u8 {
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
fn srgb8_to_lin(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Menu entry that shows its keyboard shortcut automatically.
fn mi(
    ui: &mut egui::Ui,
    actions: &mut Vec<Action>,
    bindings: &[Binding],
    label: &str,
    action: Action,
    enabled: bool,
) {
    let sc = keymap::shortcut_text(bindings, &action);
    let r = ui.add_enabled(enabled, egui::Button::new(label).shortcut_text(sc));
    if r.clicked() {
        ui.close_menu();
        actions.push(action);
    }
}

fn soon(ui: &mut egui::Ui, text: &str) {
    ui.add_enabled(false, egui::Button::new(format!("{text}  (soon)")));
}

fn kind_tag(k: &ObjectKind) -> &'static str {
    match k {
        ObjectKind::Mesh => "M",
        ObjectKind::Empty { .. } => "E",
        ObjectKind::Camera(_) => "C",
        ObjectKind::Light(_) => "L",
        ObjectKind::Surface(_) => "S",
        ObjectKind::Volume(_) => "V",
        ObjectKind::GreasePencil(_) => "G",
        ObjectKind::Armature(_) => "A",
        ObjectKind::LightProbe(_) => "P",
        ObjectKind::Speaker(_) => "Sp",
        ObjectKind::ForceField(_) => "F",
    }
}

// ------------------------------------------------------------------ main entry
pub fn draw(
    ctx: &egui::Context,
    doc: &mut Document,
    settings: &mut ViewSettings,
    ts: &mut ToolState,
    st: &mut UiState,
    bindings: &[Binding],
    recent: &[std::path::PathBuf],
) -> (Vec<Action>, egui::Rect) {
    let mut actions: Vec<Action> = Vec::new();
    poll_python_script(doc, st);

    // a continuous edit ends when nothing is being dragged or typed
    if !ctx.input(|i| i.pointer.any_down()) && ctx.memory(|m| m.focused().is_none()) {
        doc.edit_open = false;
    }

    // Apply a Blender-Dark inspired theme once
    ctx.style_mut(|style| {
        let mut visuals = egui::Visuals::dark();
        // Blender-ish dark greys
        visuals.window_fill = egui::Color32::from_rgb(0x30, 0x30, 0x30);
        visuals.panel_fill = egui::Color32::from_rgb(0x38, 0x38, 0x38);
        visuals.extreme_bg_color = egui::Color32::from_rgb(0x28, 0x28, 0x28);
        visuals.faint_bg_color = egui::Color32::from_rgb(0x3e, 0x3e, 0x3e);
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(0x38, 0x38, 0x38);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x48, 0x48, 0x48);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x5a, 0x5a, 0x5a);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x5c, 0x7a, 0xa0);
        visuals.selection.bg_fill = egui::Color32::from_rgb(0x4a, 0x70, 0xa0);
        visuals.selection.stroke.color = egui::Color32::from_rgb(0x6a, 0x9a, 0xd0);
        // Keep egui's normal dark-theme text styling. A global override makes every
        // label/menu item look forcibly highlighted instead of using widget states.
        visuals.override_text_color = None;
        style.visuals = visuals;
        style.spacing.item_spacing = egui::vec2(4.0, 3.0);
        style.spacing.window_margin = egui::Margin::same(4.0);
    });

    // === Blender-like layout ===
    // Top: global menu bar
    egui::TopBottomPanel::top("menu")
        .exact_height(24.0)
        .show(ctx, |ui| {
            ui.set_height(24.0);
            menu_bar(ui, doc, settings, ts, st, bindings, recent, &mut actions);
        });

    // Top: 3D Viewport header (mode + select mode) – similar to Blender's header
    egui::TopBottomPanel::top("mode_bar")
        .exact_height(28.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let in_edit = doc.in_edit_mode();
                let in_pose = doc.pose_mode && doc.active_armature().is_some();
                let has_armature = doc.active_armature().is_some();
                let mesh_ok = doc
                    .active_index()
                    .map(|i| doc.scene.objects[i].kind.is_mesh())
                    .unwrap_or(false)
                    || doc.selection.iter().any(|id| {
                        doc.scene
                            .objects
                            .iter()
                            .any(|o| o.id == *id && o.kind.is_mesh())
                    });

                // Mode selector (Blender style)
                let in_sculpt = doc.sculpt_mode;
                // Object Mode: default transform workspace
                let in_object = !in_edit
                    && !in_pose
                    && !in_sculpt
                    && !doc.uv_mode
                    && !doc.paint_mode
                    && !st.shading_mode
                    && !st.animation_mode
                    && !st.rendering_mode
                    && !st.compositing_mode
                    && !st.scripting_mode;
                if ui
                    .add_enabled(
                        mesh_ok
                            || has_armature
                            || in_edit
                            || in_pose
                            || in_sculpt
                            || doc.uv_mode
                            || doc.paint_mode
                            || st.animation_mode
                            || st.rendering_mode
                            || st.compositing_mode
                            || st.scripting_mode,
                        egui::SelectableLabel::new(in_object, "Object Mode"),
                    )
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_pose {
                        actions.push(Action::TogglePoseMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if doc.uv_mode {
                        actions.push(Action::ToggleUvMode);
                    }
                    if doc.paint_mode {
                        actions.push(Action::TogglePaintMode);
                    }
                }
                // Animation workspace (Blender Animation tab structure)
                let in_anim = st.animation_mode
                    && !in_edit
                    && !in_pose
                    && !in_sculpt
                    && !doc.uv_mode
                    && !doc.paint_mode
                    && !st.shading_mode
                    && !st.rendering_mode
                    && !st.compositing_mode
                    && !st.scripting_mode;
                if ui.selectable_label(in_anim, "Animation").clicked() {
                    st.shading_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    st.animation_mode = true;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_pose {
                        actions.push(Action::TogglePoseMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if doc.uv_mode {
                        actions.push(Action::ToggleUvMode);
                    }
                    if doc.paint_mode {
                        actions.push(Action::TogglePaintMode);
                    }
                }
                // Rendering workspace (Blender Rendering tab)
                let in_render_ws = st.rendering_mode
                    && !in_edit
                    && !in_pose
                    && !in_sculpt
                    && !doc.uv_mode
                    && !doc.paint_mode
                    && !st.shading_mode
                    && !st.animation_mode
                    && !st.compositing_mode
                    && !st.scripting_mode;
                if ui.selectable_label(in_render_ws, "Rendering").clicked() {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    st.rendering_mode = true;
                    st.properties_tab = PropertiesTab::Render;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_pose {
                        actions.push(Action::TogglePoseMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if doc.uv_mode {
                        actions.push(Action::ToggleUvMode);
                    }
                    if doc.paint_mode {
                        actions.push(Action::TogglePaintMode);
                    }
                }
                // Always labeled "Edit Mode" (including when editing armatures)
                if ui
                    .add_enabled(
                        mesh_ok || in_edit || doc.active_armature().is_some(),
                        egui::SelectableLabel::new(in_edit, "Edit Mode"),
                    )
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if in_pose {
                        actions.push(Action::TogglePoseMode);
                        actions.push(Action::ToggleEditMode);
                    } else if !in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                }
                if ui
                    .add_enabled(
                        has_armature,
                        egui::SelectableLabel::new(in_pose, "Pose Mode"),
                    )
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    if in_edit && doc.armature_edit {
                        actions.push(Action::TogglePoseMode);
                    } else if !in_pose {
                        actions.push(Action::TogglePoseMode);
                    }
                }
                if ui
                    .add_enabled(
                        mesh_ok || in_sculpt,
                        egui::SelectableLabel::new(in_sculpt, "Sculpt Mode"),
                    )
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if doc.uv_mode {
                        actions.push(Action::ToggleUvMode);
                    }
                    if !in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                }
                let in_uv = doc.uv_mode;
                if ui
                    .add_enabled(
                        mesh_ok || in_uv,
                        egui::SelectableLabel::new(in_uv, "UV Editing"),
                    )
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if doc.paint_mode {
                        actions.push(Action::TogglePaintMode);
                    }
                    if !in_uv {
                        actions.push(Action::ToggleUvMode);
                    }
                }
                let in_paint = doc.paint_mode;
                if ui
                    .add_enabled(
                        mesh_ok || in_paint,
                        egui::SelectableLabel::new(in_paint, "Texture Paint"),
                    )
                    .clicked()
                {
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if in_uv {
                        actions.push(Action::ToggleUvMode);
                    }
                    if !in_paint {
                        actions.push(Action::TogglePaintMode);
                    }
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = false;
                }

                if ui
                    .selectable_label(st.compositing_mode, "Compositing")
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = true;
                    st.scripting_mode = false;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if in_uv {
                        actions.push(Action::ToggleUvMode);
                    }
                    if in_paint {
                        actions.push(Action::TogglePaintMode);
                    }
                }
                if ui
                    .selectable_label(st.scripting_mode, "Scripting")
                    .clicked()
                {
                    st.shading_mode = false;
                    st.animation_mode = false;
                    st.rendering_mode = false;
                    st.compositing_mode = false;
                    st.scripting_mode = true;
                    if in_edit {
                        actions.push(Action::ToggleEditMode);
                    }
                    if in_sculpt {
                        actions.push(Action::ToggleSculptMode);
                    }
                    if in_uv {
                        actions.push(Action::ToggleUvMode);
                    }
                    if in_paint {
                        actions.push(Action::TogglePaintMode);
                    }
                    if st.python_script.is_empty() {
                        st.python_script =
                            include_str!("../../../examples/blendcraft_blender_compat_test.py")
                                .to_owned();
                    }
                }
                if ui.selectable_label(st.shading_mode, "Shading").clicked() {
                    st.shading_mode = !st.shading_mode;
                    if st.shading_mode {
                        st.animation_mode = false;
                        st.rendering_mode = false;
                        st.compositing_mode = false;
                        st.scripting_mode = false;
                        if in_edit {
                            actions.push(Action::ToggleEditMode);
                        }
                        if in_sculpt {
                            actions.push(Action::ToggleSculptMode);
                        }
                        if in_uv {
                            actions.push(Action::ToggleUvMode);
                        }
                        if in_paint {
                            actions.push(Action::TogglePaintMode);
                        }
                        ensure_shader_nodes(st);
                    }
                }

                ui.separator();

                ui.add_enabled_ui(in_edit, |ui| {
                    let domain = doc.edit.as_ref().map(|e| e.selection.domain);
                    if ui
                        .selectable_label(domain == Some(forge_core::EditDomain::Vertex), "Vertex")
                        .clicked()
                    {
                        actions.push(Action::EditDomainVertex);
                    }
                    if ui
                        .selectable_label(domain == Some(forge_core::EditDomain::Edge), "Edge")
                        .clicked()
                    {
                        actions.push(Action::EditDomainEdge);
                    }
                    if ui
                        .selectable_label(domain == Some(forge_core::EditDomain::Face), "Face")
                        .clicked()
                    {
                        actions.push(Action::EditDomainFace);
                    }
                });

                if !mesh_ok && !in_edit && !in_sculpt && !st.compositing_mode && !st.scripting_mode
                {
                    ui.weak("Select a mesh to edit");
                }
            });
        });

    // Timeline / Dope Sheet (Blender Animation workspace style)
    // Compact strip in Object Mode; full Dope Sheet layout in Animation tab.
    // Timeline at bottom (Blender Rendering workspace keeps the timeline)
    if !doc.uv_mode
        && !doc.paint_mode
        && !doc.sculpt_mode
        && !st.compositing_mode
        && !st.scripting_mode
    {
        let timeline_h = if st.animation_mode && !st.rendering_mode {
            220.0
        } else {
            72.0
        };
        egui::TopBottomPanel::bottom("timeline")
            .exact_height(timeline_h)
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(0x2a, 0x2a, 0x2a))
                    .inner_margin(egui::Margin::symmetric(8.0, 4.0)),
            )
            .show(ctx, |ui| {
                if st.animation_mode && !st.rendering_mode {
                    draw_animation_workspace(ui, doc, st, &mut actions);
                } else {
                    draw_timeline(ui, doc, st, &mut actions);
                }
            });
    }

    // Bottom: status bar (Blender-style info bar)
    egui::TopBottomPanel::bottom("status")
        .exact_height(22.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let tris: usize = (0..doc.scene.objects.len())
                    .filter(|&i| doc.scene.is_visible(i))
                    .map(|i| doc.scene.objects[i].mesh.indices.len() / 3)
                    .sum();
                ui.label(format!("{} objects", doc.scene.objects.len()));
                ui.separator();
                ui.label(format!("{tris} triangles"));
                ui.separator();
                let n = doc.selection.len();
                match doc.active_index() {
                    Some(a) if n > 0 => ui.label(format!(
                        "{n} selected, active: {}",
                        doc.scene.objects[a].name
                    )),
                    _ => ui.label("Nothing selected"),
                };
                if ts.snap.enabled {
                    ui.separator();
                    ui.label("Snap on");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&st.status);
                });
            });
        });

    // Left: Toolbar (hidden in Rendering workspace — no 3D tools there)
    let screen_h = ctx.screen_rect().height();
    let toolbar_w = if screen_h < 700.0 { 36.0 } else { 40.0 };
    let btn = if screen_h < 700.0 { 26.0 } else { 30.0 };
    if !st.rendering_mode && !st.compositing_mode && !st.scripting_mode {
        egui::SidePanel::left("toolbar")
        .exact_width(toolbar_w)
        .resizable(false)
        .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x2b, 0x2b, 0x2b)))
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 1.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(3.0);
                        let tools = if doc.paint_mode {
                            &Tool::PAINT_MODE[..]
                        } else if doc.uv_mode {
                            &Tool::UV_MODE[..]
                        } else if doc.sculpt_mode {
                            &Tool::SCULPT_MODE[..]
                        } else if doc.armature_edit && doc.active_armature().is_some() {
                            &Tool::ARMATURE_EDIT_MODE[..]
                        } else if doc.pose_mode && doc.active_armature().is_some() {
                            &Tool::POSE_MODE[..]
                        } else if doc.in_edit_mode() {
                            &Tool::EDIT_MODE[..]
                        } else {
                            &Tool::OBJECT_MODE[..]
                        };
                        for &tool in tools {
                            let selected = ts.tool == tool;
                            let bg = if selected {
                                egui::Color32::from_rgb(0x47, 0x72, 0xb3)
                            } else {
                                egui::Color32::from_rgb(0x3a, 0x3a, 0x3a)
                            };
                            let desired = egui::vec2(toolbar_w - 4.0, btn);
                            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());
                            // background
                            ui.painter().rect_filled(rect, 3.0, bg);
                            if selected {
                                ui.painter().rect_stroke(
                                    rect,
                                    3.0,
                                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x6a, 0x9a, 0xd0)),
                                );
                            }
                            // icon
                            let icon_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(btn * 0.72, btn * 0.72));
                            draw_tool_icon(ui, tool, icon_rect, selected, doc.armature_edit, doc.pose_mode);
                            if resp.clicked() {
                                ts.tool = tool;
                                // Mesh ops are interactive via Ctrl+LMB in Edit Mode.
                                // Toolbar only selects the tool (except Add Cube / Cursor).
                                match tool {
                                    Tool::AddCube if !doc.in_edit_mode() => {
                                        actions.push(Action::Add(forge_core::Primitive::Cube { size: 1.0 }));
                                    }
                                    Tool::Cursor => actions.push(Action::SnapCursorToSel),
                                    Tool::BoneAdd => actions.push(Action::ArmatureAddBone),
                                    Tool::BoneExtrude => st.status = "Extrude Bone selected — click-drag from a selected bone tip in the viewport".into(),
                                    Tool::BoneDelete => actions.push(Action::ArmatureDeleteBones),
                                    Tool::PoseClear => actions.push(Action::ArmatureClearPose),
                                    Tool::UvUnwrapSmart => actions.push(Action::UvUnwrapSmart),
                                    Tool::UvUnwrapCube => actions.push(Action::UvUnwrapCube),
                                    Tool::UvUnwrapPlanar => actions.push(Action::UvUnwrapPlanar),
                                    Tool::UvReset => actions.push(Action::UvReset),
                                    Tool::UvPack => actions.push(Action::UvPack),
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
                                    | Tool::RipRegion => {
                                        st.status = format!(
                                            "{} selected – select geometry, then Ctrl+LMB drag",
                                            tool.name()
                                        );
                                    }
                                    _ => {}
                                }
                            }
                            let bone_mode = doc.armature_edit || doc.pose_mode;
                            let tooltip = if bone_mode {
                                match tool {
                                    Tool::Select => "Select bones",
                                    Tool::BoxSelect => "Box-select bones",
                                    Tool::Circle => "Circle-select bones",
                                    Tool::Lasso => "Lasso-select bones",
                                    Tool::Move => "Move selected bones",
                                    Tool::Rotate => "Rotate selected bones",
                                    Tool::Scale => "Scale selected bones",
                                    _ => tool.tooltip(),
                                }
                            } else { tool.tooltip() };
                            resp.on_hover_text(tooltip);
                        }
                        ui.add_space(3.0);
                    });
                });
        });
    } // !rendering_mode toolbar

    // UV Editing workspace (Blender-style):
    // LEFT = UV Editor · RIGHT/CENTER = 3D Edit view (Outliner/Properties hidden)
    if doc.uv_mode {
        egui::SidePanel::left("uv_editor")
            .resizable(true)
            .default_width(460.0)
            .width_range(280.0..=900.0)
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x2a, 0x2a, 0x2a)))
            .show(ctx, |ui| {
                draw_uv_editor(ui, doc, ts, &mut actions);
            });
    }

    // Texture Paint workspace: Blender-style split editor.
    // LEFT = the actual 2D image canvas (with UV overlay); RIGHT = the 3D viewport.
    if doc.paint_mode {
        egui::SidePanel::left("texture_paint_image_editor")
            .resizable(true)
            .default_width(480.0)
            .width_range(300.0..=900.0)
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x2a, 0x2a, 0x2a)))
            .show(ctx, |ui| {
                draw_texture_paint_editor(ui, ctx, doc, ts, st, &mut actions);
            });
    }

    // Shading workspace: a docked node editor alongside the live 3D viewport.
    if st.shading_mode {
        egui::TopBottomPanel::bottom("shading_node_editor")
            .resizable(!st.shading_nodes_collapsed)
            .default_height(if st.shading_nodes_collapsed {
                34.0
            } else {
                390.0
            })
            .height_range(if st.shading_nodes_collapsed {
                34.0..=34.0
            } else {
                180.0..=680.0
            })
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x20, 0x20, 0x20)))
            .show(ctx, |ui| {
                draw_shading_editor(ui, ctx, doc, st);
            });
    }

    // Right panel:
    // - Rendering workspace: Render / Output properties only (no 3D, no Outliner)
    // - Other workspaces: Outliner + Properties
    if st.rendering_mode && !doc.uv_mode && !doc.paint_mode {
        egui::SidePanel::right("render_properties_panel")
            .default_width(320.0)
            .width_range(260.0..=480.0)
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(0x2b, 0x2b, 0x2b))
                    .inner_margin(egui::Margin::symmetric(8.0, 6.0)),
            )
            .show(ctx, |ui| {
                draw_rendering_properties(ui, doc, st, &mut actions);
            });
    } else if !doc.uv_mode && !doc.paint_mode && !st.compositing_mode && !st.scripting_mode {
        egui::SidePanel::right("right_panel")
            .default_width(300.0)
            .width_range(220.0..=480.0)
            .show(ctx, |ui| {
                egui::TopBottomPanel::top("outliner_panel")
                    .resizable(true)
                    .default_height(280.0)
                    .height_range(120.0..=600.0)
                    .show_inside(ui, |ui| {
                        ui.heading("Outliner");
                        ui.separator();
                        outliner(ui, doc, st, &mut actions);
                    });

                properties(ui, doc, st, &mut actions);
            });
    }

    // Texture Paint uses the 3D viewport as its canvas; no detached editor window.

    // Rendering workspace (Blender): center = Render Result image only — no 3D view.
    if st.rendering_mode && !doc.uv_mode && !doc.paint_mode && !doc.sculpt_mode {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x1e, 0x1e, 0x1e)))
            .show(ctx, |ui| {
                draw_render_result_view(ui, st, &mut actions);
            });
    }

    if st.compositing_mode && !doc.uv_mode && !doc.paint_mode && !doc.sculpt_mode {
        egui::TopBottomPanel::bottom("compositor_node_editor")
            .resizable(true)
            .default_height(360.0)
            .height_range(220.0..=700.0)
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x20, 0x20, 0x20)))
            .show(ctx, |ui| {
                draw_compositing_editor(ui, ctx, doc, st, &mut actions)
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x1e, 0x1e, 0x1e)))
            .show(ctx, |ui| {
                draw_compositor_preview(ui, ctx, doc, st, &mut actions)
            });
    }

    if st.scripting_mode && !doc.uv_mode && !doc.paint_mode && !doc.sculpt_mode {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0x1e, 0x1e, 0x1e)))
            .show(ctx, |ui| draw_scripting_workspace(ui, st, &mut actions));
    }

    // Right-clicking inside the 3D viewport opens the exact same Object Mode menu
    // used by Outliner rows. The viewport is the remaining central area after panels.
    let viewport_rect = ctx.available_rect();
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        st.viewport_context_menu = None;
        st.viewport_context_menu_rect = None;
    }
    if ctx.input(|i| i.pointer.any_click()) && !ctx.input(|i| i.pointer.secondary_clicked()) {
        if let Some(pointer) = ctx.input(|i| i.pointer.interact_pos()) {
            if !st
                .viewport_context_menu_rect
                .map(|r| r.contains(pointer))
                .unwrap_or(false)
            {
                st.viewport_context_menu = None;
                st.viewport_context_menu_rect = None;
            }
        }
    }
    if ctx.input(|i| i.pointer.secondary_clicked()) {
        if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
            if viewport_rect.contains(pos)
                && doc.active_index().is_some()
                && !doc.in_edit_mode()
                && !doc.pose_mode
                && !doc.sculpt_mode
                && !doc.uv_mode
                && !doc.paint_mode
            {
                st.viewport_context_menu = Some(pos);
            }
        }
    }
    if let Some(pos) = st.viewport_context_menu {
        let response = egui::Area::new(egui::Id::new("viewport_object_context_menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(190.0);
                    object_context_menu(ui, doc, &mut actions);
                });
            });
        st.viewport_context_menu_rect = Some(response.response.rect);
        if !actions.is_empty() {
            st.viewport_context_menu = None;
            st.viewport_context_menu_rect = None;
        }
    }

    shortcuts_window(ctx, st, bindings);
    move_popup(ctx, doc, st, &mut actions);
    font_library_window(ctx, st);
    let mut about = st.show_about;
    egui::Window::new("About").open(&mut about).show(ctx, |ui| {
        ui.heading("BlendCraft");
        ui.label("A 3D suite for game assets, written in Rust.");
        ui.label("Press F1 for the keyboard shortcuts.");
    });
    st.show_about = about;

    (actions, ctx.available_rect())
}

// Helper: draw a Blender-ish tool icon into a rect using egui shapes.
// Kept in ui.rs as a function.

fn ensure_shader_nodes(st: &mut UiState) {
    if st.shader_rgb_color == [0; 4] {
        st.shader_rgb_color = [180, 180, 190, 255];
    }
    if st.shader_zoom < 0.15 {
        st.shader_zoom = 1.0;
    }
    if st.shader_nodes.is_empty() {
        let g = ShaderGraph::default();
        st.shader_nodes = g.nodes;
        st.shader_links = g.links;
        st.shader_next_id = g.next_id;
    }
}

fn graph_from_ui(st: &UiState) -> ShaderGraph {
    ShaderGraph {
        nodes: st.shader_nodes.clone(),
        links: st.shader_links.clone(),
        next_id: st.shader_next_id,
    }
}

fn load_graph_into_ui(st: &mut UiState, g: &ShaderGraph) {
    st.shader_nodes = g.nodes.clone();
    st.shader_links = g.links.clone();
    st.shader_next_id = g.next_id.max(1);
}

fn add_noise_ramp_demo(st: &mut UiState) {
    let mut next_id = st.shader_next_id.max(1);
    let mut create_node = |kind, pos| {
        let id = next_id;
        next_id += 1;
        ShaderNode::with_kind(id, kind, pos)
    };

    let principled_id = st
        .shader_nodes
        .iter()
        .find(|node| node.kind == ShaderNodeKind::Principled)
        .map(|node| node.id)
        .unwrap_or_else(|| {
            let node = create_node(ShaderNodeKind::Principled, [600.0, 100.0]);
            let id = node.id;
            st.shader_nodes.push(node);
            id
        });
    let output_id = st
        .shader_nodes
        .iter()
        .find(|node| node.kind == ShaderNodeKind::Output)
        .map(|node| node.id)
        .unwrap_or_else(|| {
            let node = create_node(ShaderNodeKind::Output, [920.0, 100.0]);
            let id = node.id;
            st.shader_nodes.push(node);
            id
        });

    let noise = create_node(ShaderNodeKind::Noise, [60.0, 120.0]);
    let noise_id = noise.id;
    let mut ramp = create_node(ShaderNodeKind::ColorRamp, [315.0, 120.0]);
    let ramp_id = ramp.id;
    ramp.value = 2.0;
    ramp.color = [0.0, 0.0, 0.0, 1.0];
    ramp.props[0..4].copy_from_slice(&[1.0; 4]);
    ramp.props[12] = 0.0;
    ramp.props[13] = 1.0;
    st.shader_nodes.extend([noise, ramp]);
    st.shader_links.retain(|link| {
        !(link.to == principled_id && link.to_socket == 0)
            && !(link.to == output_id && link.to_socket == 0)
    });
    st.shader_links.extend([
        ShaderLink {
            from: noise_id,
            from_socket: 0,
            to: ramp_id,
            to_socket: 0,
        },
        ShaderLink {
            from: ramp_id,
            from_socket: 0,
            to: principled_id,
            to_socket: 0,
        },
        ShaderLink {
            from: principled_id,
            from_socket: 0,
            to: output_id,
            to_socket: 0,
        },
    ]);
    st.shader_next_id = next_id;
    st.shader_apply_pending = true;
}

fn remap_graph_texture_indices(
    material: &mut Material,
    graph: &mut ShaderGraph,
    old_textures: &[TextureData],
) -> Vec<usize> {
    let mut remap = vec![None; old_textures.len()];
    let mut kept = Vec::with_capacity(old_textures.len());
    for (old_index, texture) in old_textures.iter().enumerate() {
        if !texture.name.starts_with("__sg_") {
            remap[old_index] = Some(kept.len());
            kept.push(old_index);
        }
    }
    let remap_slot = |slot: &mut Option<usize>| {
        if let Some(old) = *slot {
            *slot = remap.get(old).copied().flatten();
        }
    };
    remap_slot(&mut material.base_color_tex);
    remap_slot(&mut material.metallic_roughness_tex);
    remap_slot(&mut material.normal_tex);
    remap_slot(&mut material.occlusion_tex);
    remap_slot(&mut material.emissive_tex);
    for node in &mut graph.nodes {
        if let Some(old) = node.image_index {
            node.image_index = remap.get(old).copied().flatten();
        }
    }
    kept
}

fn sync_graph_live(ctx: &egui::Context, doc: &mut Document, mi: usize, st: &mut UiState) {
    let current_graph = graph_from_ui(st);
    let current_frame = doc.scene.animation.frame_current;
    let current_texture_id = Arc::as_ptr(&doc.scene.textures) as usize;
    let current_key = ShaderEvalKey {
        graph: current_graph.clone(),
        material: mi,
        frame: current_frame,
        texture_id: current_texture_id,
    };
    if st
        .shader_eval_key
        .as_ref()
        .is_some_and(|active| active != &current_key)
    {
        if let Some(cancel) = &st.shader_eval_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    let completion = st
        .shader_eval_rx
        .as_ref()
        .and_then(|rx| match rx.try_recv() {
            Ok(completion) => Some(Ok(completion)),
            Err(TryRecvError::Disconnected) => Some(Err(())),
            Err(TryRecvError::Empty) => None,
        });
    if let Some(completion) = completion {
        let active_key = st.shader_eval_key.clone();
        st.shader_eval_rx = None;
        st.shader_eval_key = None;
        st.shader_eval_cancel = None;
        match completion {
            Ok(completion) => {
                st.shader_eval_failed_key = None;
                let key = &completion.key;
                let is_current = key.graph == current_graph
                    && key.material == mi
                    && key.frame == current_frame
                    && key.texture_id == current_texture_id;
                if is_current {
                    if completion.supported {
                        st.shader_last_eval_graph = Some(key.graph.clone());
                        st.shader_last_eval_material = Some(mi);
                        st.shader_last_eval_frame = Some(current_frame);
                        st.shader_last_eval_texture_id = current_texture_id;
                        if doc.scene.materials[mi].shader_graph.as_ref() != Some(&current_graph) {
                            doc.scene.materials[mi].shader_graph = Some(current_graph.clone());
                            doc.touch();
                        }
                    } else if let Some(result) = completion.result {
                        let mut graph = graph_from_ui(st);
                        remap_graph_texture_indices(
                            &mut doc.scene.materials[mi],
                            &mut graph,
                            &doc.scene.textures,
                        );
                        st.shader_nodes = graph.nodes.clone();
                        let mut textures = completion.textures;
                        doc.scene.materials[mi].shader_graph = Some(graph.clone());
                        apply_eval_to_material(&mut doc.scene.materials[mi], &mut textures, result);
                        doc.scene.textures = Arc::new(textures);
                        st.shader_last_eval_graph = Some(graph);
                        st.shader_last_eval_material = Some(mi);
                        st.shader_last_eval_frame = Some(current_frame);
                        st.shader_last_eval_texture_id = Arc::as_ptr(&doc.scene.textures) as usize;
                        doc.touch();
                    }
                }
            }
            Err(()) => {
                st.shader_eval_failed_key = active_key;
                st.status = "Shader graph evaluation worker stopped unexpectedly".into();
            }
        }
    }

    let graph = graph_from_ui(st);
    let key = ShaderEvalKey {
        graph: graph.clone(),
        material: mi,
        frame: current_frame,
        texture_id: Arc::as_ptr(&doc.scene.textures) as usize,
    };
    let already_evaluated = st.shader_last_eval_graph.as_ref() == Some(&key.graph)
        && st.shader_last_eval_material == Some(key.material)
        && st.shader_last_eval_frame == Some(key.frame)
        && st.shader_last_eval_texture_id == key.texture_id;
    let already_failed = st.shader_eval_failed_key.as_ref() == Some(&key);
    if st.shader_eval_rx.is_none() && !already_evaluated && !already_failed {
        let eval_graph = graph.clone();
        let eval_material = doc.scene.materials[mi].clone();
        let source_textures = Arc::clone(&doc.scene.textures);
        let frame = current_frame as f32;
        let time_sec = frame / doc.scene.animation.fps.max(1) as f32;
        let (tx, rx) = mpsc::channel();
        let worker_key = key.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        match std::thread::Builder::new()
            .name("shader-graph-eval".into())
            .spawn(move || {
                if forge_render::supports_shader_graph(&eval_graph) {
                    let _ = tx.send(ShaderEvalCompletion {
                        key: worker_key,
                        supported: true,
                        textures: Vec::new(),
                        result: None,
                    });
                    return;
                }

                let mut eval_graph = eval_graph;
                let mut eval_material = eval_material;
                let kept_texture_indices = remap_graph_texture_indices(
                    &mut eval_material,
                    &mut eval_graph,
                    &source_textures,
                );
                let eval_textures: Vec<_> = kept_texture_indices
                    .iter()
                    .filter_map(|index| source_textures.get(*index).cloned())
                    .collect();
                let result = evaluate_graph_ex_cancellable(
                    &eval_graph,
                    &eval_material,
                    &eval_textures,
                    128,
                    false,
                    frame,
                    time_sec,
                    &worker_cancel,
                );
                let _ = tx.send(ShaderEvalCompletion {
                    key: worker_key,
                    supported: false,
                    textures: eval_textures,
                    result,
                });
            }) {
            Ok(_) => {
                st.shader_eval_rx = Some(rx);
                st.shader_eval_key = Some(key);
                st.shader_eval_cancel = Some(cancel);
                st.shader_eval_failed_key = None;
            }
            Err(error) => {
                st.shader_eval_failed_key = Some(key);
                st.status = format!("Could not start shader graph evaluation: {error}");
            }
        }
    }

    if st.shader_eval_rx.is_some() {
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

fn sync_graph_bake(ctx: &egui::Context, doc: &mut Document, mi: usize, st: &mut UiState) {
    let completion = st
        .shader_bake_rx
        .as_ref()
        .and_then(|rx| match rx.try_recv() {
            Ok(completion) => Some(Ok(completion)),
            Err(TryRecvError::Disconnected) => Some(Err(())),
            Err(TryRecvError::Empty) => None,
        });
    if let Some(completion) = completion {
        st.shader_bake_rx = None;
        match completion {
            Ok(completion) => {
                if completion.graph == graph_from_ui(st)
                    && completion.material == mi
                    && completion.texture_id == Arc::as_ptr(&doc.scene.textures) as usize
                {
                    let mut textures = (*doc.scene.textures).clone();
                    textures.push(completion.texture);
                    let texture_index = textures.len() - 1;
                    doc.scene.textures = Arc::new(textures);
                    doc.scene.materials[mi].base_color_tex = Some(texture_index);
                    st.shader_last_eval_graph = Some(graph_from_ui(st));
                    st.shader_last_eval_material = Some(mi);
                    st.shader_last_eval_frame = Some(doc.scene.animation.frame_current);
                    st.shader_last_eval_texture_id = Arc::as_ptr(&doc.scene.textures) as usize;
                    doc.touch();
                } else {
                    st.status =
                        "Graph changed during texture bake; bake it again to use the result".into();
                }
            }
            Err(()) => {
                st.status = "Shader texture bake worker stopped unexpectedly".into();
            }
        }
    }

    if st.shader_bake_pending && st.shader_bake_rx.is_none() && st.shader_eval_rx.is_none() {
        st.shader_bake_pending = false;
        let graph = graph_from_ui(st);
        let mut eval_graph = graph.clone();
        let mut material = doc.scene.materials[mi].clone();
        let source_textures = Arc::clone(&doc.scene.textures);
        let kept_texture_indices =
            remap_graph_texture_indices(&mut material, &mut eval_graph, &source_textures);
        let texture_id = Arc::as_ptr(&doc.scene.textures) as usize;
        let (tx, rx) = mpsc::channel();
        match std::thread::Builder::new()
            .name("shader-graph-texture-bake".into())
            .spawn(move || {
                let textures: Vec<_> = kept_texture_indices
                    .iter()
                    .filter_map(|index| source_textures.get(*index).cloned())
                    .collect();
                let texture = bake_graph_texture(&eval_graph, &material, &textures, 1024);
                let _ = tx.send(ShaderBakeCompletion {
                    graph,
                    material: mi,
                    texture_id,
                    texture,
                });
            }) {
            Ok(_) => st.shader_bake_rx = Some(rx),
            Err(error) => {
                st.status = format!("Could not start shader texture bake: {error}");
            }
        }
    }

    if st.shader_bake_rx.is_some() || st.shader_bake_pending {
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

fn shader_node_color(kind: ShaderNodeKind) -> egui::Color32 {
    match kind.category() {
        "Output" => egui::Color32::from_rgb(0x72, 0x91, 0x62),
        "Shader" => egui::Color32::from_rgb(0x9a, 0x6b, 0x3c),
        "Texture" => egui::Color32::from_rgb(0x3e, 0x78, 0x91),
        "Color" => egui::Color32::from_rgb(0x7d, 0x55, 0x86),
        "Vector" => egui::Color32::from_rgb(0x3e, 0x7b, 0x65),
        "Input" => egui::Color32::from_rgb(0x55, 0x55, 0x70),
        _ => egui::Color32::from_rgb(0x55, 0x55, 0x60),
    }
}

fn draw_shading_editor(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    doc: &mut Document,
    st: &mut UiState,
) {
    ensure_shader_nodes(st);
    let Some(object_index) = doc.active_index() else {
        ui.centered_and_justified(|ui| ui.weak("Select an object to edit its material."));
        return;
    };
    let mut current = doc.scene.objects[object_index]
        .material
        .filter(|m| *m < doc.scene.materials.len());

    // Blender-like editor header: material datablock, node tools and graph controls.
    egui::TopBottomPanel::top("shader_editor_header").show_inside(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("◈ Shader Editor").strong());
            if ui
                .small_button(if st.shading_nodes_collapsed {
                    "▴ Expand"
                } else {
                    "▾ Collapse"
                })
                .clicked()
            {
                st.shading_nodes_collapsed = !st.shading_nodes_collapsed;
            }
            ui.separator();
            let selected = current
                .map(|m| doc.scene.materials[m].name.clone())
                .unwrap_or_else(|| "No Material".into());
            egui::ComboBox::from_id_salt("shading_material_slot")
                .selected_text(selected)
                .width(180.0)
                .show_ui(ui, |ui| {
                    for (idx, mat) in doc.scene.materials.iter().enumerate() {
                        if ui
                            .selectable_label(current == Some(idx), &mat.name)
                            .clicked()
                        {
                            current = Some(idx);
                        }
                    }
                });
            if ui.button("＋ New").clicked() {
                doc.checkpoint();
                let mut mat = Material::default();
                mat.name = format!("Material.{:03}", doc.scene.materials.len() + 1);
                doc.scene.materials.push(mat);
                current = Some(doc.scene.materials.len() - 1);
                doc.scene.objects[object_index].material = current;
            }
            ui.separator();
            ui.menu_button("Add ▾", |ui| {
                let mut by_cat: std::collections::BTreeMap<&str, Vec<ShaderNodeKind>> =
                    std::collections::BTreeMap::new();
                for &k in ShaderNodeKind::all_kinds() {
                    by_cat.entry(k.category()).or_default().push(k);
                }
                for (group, items) in by_cat {
                    ui.menu_button(group, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(360.0)
                            .show(ui, |ui| {
                                for kind in items {
                                    if ui.button(kind.title()).clicked() {
                                        let id = st.shader_next_id.max(1);
                                        st.shader_next_id = id + 1;
                                        let col =
                                            80.0 + (st.shader_nodes.len() as f32 % 4.0) * 245.0;
                                        let row = 70.0
                                            + (st.shader_nodes.len() as f32 / 4.0).floor() * 210.0;
                                        let (color, props, value) =
                                            if kind == ShaderNodeKind::ColorRamp {
                                                // Blender default: 2 stops, black @0 → white @1
                                                let mut p = [0.0; 16];
                                                p[0] = 1.0;
                                                p[1] = 1.0;
                                                p[2] = 1.0;
                                                p[3] = 1.0; // stop1 white
                                                p[12] = 0.0;
                                                p[13] = 1.0; // positions
                                                ([0.0, 0.0, 0.0, 1.0], p, 2.0)
                                            } else {
                                                ([0.8, 0.8, 0.8, 1.0], [0.0; 16], 0.5)
                                            };
                                        let mut node = ShaderNode::with_kind(id, kind, [col, row]);
                                        node.value = value;
                                        node.color = color;
                                        node.props = props;
                                        st.shader_nodes.push(node);
                                        ui.close_menu();
                                    }
                                }
                            });
                    });
                }
            });
            if ui
                .button("Fit View")
                .on_hover_text("Reset pan/zoom and node layout")
                .clicked()
            {
                st.shader_pan = [0.0, 0.0];
                st.shader_zoom = 1.0;
                for (i, node) in st.shader_nodes.iter_mut().enumerate() {
                    node.ui_size = [0.0; 2];
                    node.pos = match node.kind {
                        ShaderNodeKind::Principled => [330.0, 70.0],
                        ShaderNodeKind::Output => [700.0, 105.0],
                        ShaderNodeKind::Rgb => [55.0, 90.0],
                        ShaderNodeKind::ImageTexture => [55.0, 300.0],
                        ShaderNodeKind::TexCoord => [55.0, 510.0],
                        _ => {
                            let col = 80.0 + (i as f32 % 4.0) * 245.0;
                            let row = 70.0 + (i as f32 / 4.0).floor() * 210.0;
                            [col, row]
                        }
                    };
                }
            }
            if ui
                .button("Apply Graph")
                .on_hover_text("Evaluate nodes into the material (textures & PBR slots)")
                .clicked()
            {
                // mi is not in scope here (header is above). Flag for apply after material is known.
                st.shader_apply_pending = true;
            }
            if ui
                .button("Bake…")
                .on_hover_text("Bake graph color to a new texture and assign as Base Color")
                .clicked()
            {
                st.shader_bake_pending = true;
            }
            if ui
                .button("Noise Demo")
                .on_hover_text("Add a connected Noise → Color Ramp → Principled example")
                .clicked()
            {
                add_noise_ramp_demo(st);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if st.shader_bake_pending || st.shader_bake_rx.is_some() {
                    ui.spinner();
                    ui.weak("Baking texture");
                } else if st.shader_eval_rx.is_some() {
                    ui.spinner();
                    ui.weak("Updating material");
                }
                ui.weak(
                    "Drag nodes/canvas · Drag sockets to connect · Right-click wires to remove",
                );
            });
        });
    });
    if st.shading_nodes_collapsed {
        return;
    }
    if doc.scene.objects[object_index].material != current {
        doc.checkpoint();
        doc.scene.objects[object_index].material = current;
    }
    let Some(mi) = current else {
        ui.centered_and_justified(|ui| {
            ui.weak("Create or select a material to start building its shader graph.")
        });
        return;
    };

    // Load persisted graph for this material once (or when empty).
    if let Some(g) = doc.scene.materials[mi].shader_graph.clone() {
        if st.shader_nodes.is_empty()
            || st.shader_nodes.iter().map(|n| n.id).collect::<Vec<_>>()
                != g.nodes.iter().map(|n| n.id).collect::<Vec<_>>()
        {
            // Only replace when the material graph differs from the UI session
            // and the UI was empty or just switched materials.
        }
    }
    // Prefer material graph when UI has only the default empty ensure state and material has one.
    if let Some(g) = &doc.scene.materials[mi].shader_graph {
        if !g.nodes.is_empty() && (st.shader_nodes.is_empty() || st.shader_material_mi != Some(mi))
        {
            load_graph_into_ui(st, g);
        }
    } else if st.shader_material_mi != Some(mi) {
        // Fresh material → default graph
        let g = ShaderGraph::default();
        load_graph_into_ui(st, &g);
        doc.scene.materials[mi].shader_graph = Some(g);
    }
    st.shader_material_mi = Some(mi);

    if st.shader_apply_pending {
        st.shader_apply_pending = false;
        st.shader_last_eval_graph = None;
    }
    // Keep graph editing responsive; CPU texture evaluation runs outside the UI frame.
    sync_graph_live(ctx, doc, mi, st);
    sync_graph_bake(ctx, doc, mi, st);

    // Persist graph continuously so project save keeps it.
    doc.scene.materials[mi].shader_graph = Some(graph_from_ui(st));

    // ---- Shader node editor (Blender-like look + movement) ----
    if st.shader_zoom < 0.2 {
        st.shader_zoom = 1.0;
    }
    let zoom = st.shader_zoom.clamp(0.25, 3.0);
    let canvas_size = egui::vec2(
        ui.available_width().max(280.0),
        ui.available_height().max(160.0),
    );
    let (canvas_rect, canvas_resp) =
        ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

    let pointer = ui.ctx().pointer_interact_pos();
    let over_node = pointer
        .map(|p| st.shader_last_node_rects.iter().any(|(_, r)| r.contains(p)))
        .unwrap_or(false);

    // Zoom only on empty canvas (never while pointer is over a node — no "scroll inside node").
    // Wheel: zoom view so zoom-out makes nodes SMALLER (Blender node editor).
    if canvas_resp.hovered() && !over_node {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let before = zoom;
            // egui reports positive Y for scrolling up, which should zoom in.
            let factor = (1.0 + scroll * 0.0015).clamp(0.82, 1.22);
            let after = (before * factor).clamp(0.25, 3.0);
            if let Some(p) = pointer.or(ui.ctx().pointer_hover_pos()) {
                let local = p - canvas_rect.min;
                let gx = (local.x - st.shader_pan[0]) / before;
                let gy = (local.y - st.shader_pan[1]) / before;
                st.shader_zoom = after;
                st.shader_pan[0] = local.x - gx * after;
                st.shader_pan[1] = local.y - gy * after;
            } else {
                st.shader_zoom = after;
            }
        }
    }
    let zoom = st.shader_zoom.clamp(0.25, 3.0);
    let pan0 = st.shader_pan[0];
    let pan1 = st.shader_pan[1];

    // Backdrop + grid
    {
        let paint = ui.painter_at(canvas_rect);
        paint.rect_filled(canvas_rect, 0.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));
        let step = (28.0 * zoom).max(12.0);
        let origin = canvas_rect.min + egui::vec2(pan0, pan1);
        let mut x = origin.x - ((origin.x - canvas_rect.left()) / step).ceil() * step;
        let mut i = 0i32;
        while x <= canvas_rect.right() {
            let major = i.rem_euclid(5) == 0;
            paint.line_segment(
                [
                    egui::pos2(x, canvas_rect.top()),
                    egui::pos2(x, canvas_rect.bottom()),
                ],
                egui::Stroke::new(
                    if major { 1.0_f32 } else { 0.5_f32 },
                    egui::Color32::from_gray(if major { 46 } else { 34 }),
                ),
            );
            x += step;
            i += 1;
        }
        let mut y = origin.y - ((origin.y - canvas_rect.top()) / step).ceil() * step;
        let mut j = 0i32;
        while y <= canvas_rect.bottom() {
            let major = j.rem_euclid(5) == 0;
            paint.line_segment(
                [
                    egui::pos2(canvas_rect.left(), y),
                    egui::pos2(canvas_rect.right(), y),
                ],
                egui::Stroke::new(
                    if major { 1.0_f32 } else { 0.5_f32 },
                    egui::Color32::from_gray(if major { 46 } else { 34 }),
                ),
            );
            y += step;
            j += 1;
        }
        paint.text(
            canvas_rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Drag nodes · Drag empty canvas to pan · MMB pans anywhere · Right-click wires to remove · {:.0}%",
                zoom * 100.0
            ),
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(120),
        );
    }

    st.shader_socket_positions.clear();
    let nodes = st.shader_nodes.clone();
    let mut node_rects: Vec<(u64, egui::Rect)> = Vec::new();
    let mut delete_node: Option<u64> = None;

    for node in nodes {
        let n_in = node.kind.inputs().len();
        let n_out = node.kind.outputs().len();
        let sock_rows = n_in.max(n_out).max(1);
        // Extra rows for built-in controls (color ramp, rgb, math…)
        let extra = match node.kind {
            ShaderNodeKind::ColorRamp => 9,
            ShaderNodeKind::Rgb | ShaderNodeKind::Value | ShaderNodeKind::Math => 1,
            ShaderNodeKind::ImageTexture => 1,
            _ => 0,
        };
        let default_w = match node.kind {
            ShaderNodeKind::Principled => 230.0,
            ShaderNodeKind::ColorRamp => 220.0,
            _ if node.kind.category() == "Texture" => 210.0,
            ShaderNodeKind::Output | ShaderNodeKind::WorldOutput | ShaderNodeKind::LightOutput => {
                160.0
            }
            _ => 170.0,
        };
        // FIXED height from content formula — not user-resizable, no inner scroll
        let fixed_h = (26.0 + (sock_rows + extra) as f32 * 18.0 + 12.0).clamp(56.0, 340.0);
        let gw = if node.ui_size[0] > 0.0 {
            node.ui_size[0]
        } else {
            default_w
        }
        .clamp(120.0, 500.0);
        let gh = if node.ui_size[1] > 0.0 {
            node.ui_size[1]
        } else {
            fixed_h
        }
        .clamp(fixed_h, 600.0);

        // Blender-like zoom: positions *and* node size scale together.
        // Inner style (fonts, spacing) is scaled by the same zoom so content matches the frame.
        let screen_pos = egui::pos2(
            canvas_rect.min.x + pan0 + node.pos[0] * zoom,
            canvas_rect.min.y + pan1 + node.pos[1] * zoom,
        );
        let sw = (gw * zoom).max(36.0);
        let sh = (gh * zoom).max(28.0);
        let approx = egui::Rect::from_min_size(screen_pos, egui::vec2(sw, sh));
        if !approx.intersects(canvas_rect.expand(8.0)) {
            continue;
        }

        let header_col = shader_node_color(node.kind);
        let area = egui::Area::new(egui::Id::new(("node", node.id)))
            .order(egui::Order::Middle)
            .current_pos(screen_pos)
            .movable(false)
            .constrain(false)
            .show(ctx, |ui| {
                ui.set_clip_rect(canvas_rect);
                let move_node = ui.interact(
                    egui::Rect::from_min_size(screen_pos, egui::vec2(sw, sh)),
                    ui.id().with("move_node"),
                    egui::Sense::drag(),
                );
                // Scale fonts + spacing with view zoom (so zoom-out = smaller, cramped nodes)
                {
                    let mut style = (*ui.ctx().style()).clone();
                    let z = zoom.clamp(0.35, 2.5);
                    style.spacing.item_spacing = egui::vec2(4.0 * z, 2.0 * z);
                    style.spacing.button_padding = egui::vec2(3.0 * z, 1.5 * z);
                    style.spacing.indent = 8.0 * z;
                    for ts in style.text_styles.values_mut() {
                        ts.size = (ts.size * z).clamp(7.0, 18.0);
                    }
                    ui.set_style(style);
                }
                let frame = egui::Frame::none()
                    .fill(egui::Color32::from_rgb(0x2b, 0x2b, 0x2b))
                    .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x14, 0x14, 0x14)))
                    .rounding(egui::Rounding::same((4.0 * zoom).clamp(2.0, 6.0)))
                    .inner_margin(egui::Margin::same(0.0));
                frame.show(ui, |ui| {
                    ui.set_width(sw);
                    ui.set_max_width(sw);
                    ui.set_min_width(sw);

                    // ---- Title bar ----
                    let title_h = (22.0 * zoom).clamp(14.0, 28.0);
                    let title_rect = ui
                        .allocate_exact_size(egui::vec2(sw, title_h), egui::Sense::hover())
                        .0;
                    ui.painter().rect_filled(
                        title_rect,
                        egui::Rounding {
                            nw: 4.0,
                            ne: 4.0,
                            sw: 0.0,
                            se: 0.0,
                        },
                        header_col,
                    );
                    // slight darker strip under title like Blender
                    ui.painter().line_segment(
                        [
                            egui::pos2(title_rect.left(), title_rect.bottom()),
                            egui::pos2(title_rect.right(), title_rect.bottom()),
                        ],
                        egui::Stroke::new(1.0_f32, egui::Color32::from_black_alpha(80)),
                    );
                    ui.painter().text(
                        title_rect.left_center() + egui::vec2(8.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        node.kind.title(),
                        egui::FontId::proportional((12.0 * zoom).clamp(8.0, 16.0)),
                        egui::Color32::WHITE,
                    );
                    let close_r = egui::Rect::from_center_size(
                        egui::pos2(title_rect.right() - 11.0, title_rect.center().y),
                        egui::vec2(14.0, 14.0),
                    );
                    let close = ui.interact(close_r, ui.id().with("x"), egui::Sense::click());
                    ui.painter().circle_filled(
                        close_r.center(),
                        6.0,
                        if close.hovered() {
                            egui::Color32::from_rgb(0xc0, 0x40, 0x40)
                        } else {
                            egui::Color32::from_black_alpha(40)
                        },
                    );
                    ui.painter().text(
                        close_r.center(),
                        egui::Align2::CENTER_CENTER,
                        "×",
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_gray(230),
                    );
                    if close.clicked() {
                        delete_node = Some(node.id);
                    }
                    if move_node.dragged() && !close.hovered() {
                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                            let d = ui.input(|i| i.pointer.delta());
                            n.pos[0] += d.x / zoom;
                            n.pos[1] += d.y / zoom;
                        }
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                    } else if move_node.hovered() && !close.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }

                    // ---- Body ----
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.add_space(6.0);
                        ui.vertical(|ui| {
                            ui.set_width(sw - 12.0);
                            ui.spacing_mut().item_spacing.y = 2.0;

                            // Compact property controls
                            match node.kind {
                                ShaderNodeKind::Wave => {
                                    let wave_types = ["Bands", "Rings"];
                                    let wave_type = (node.mode as usize).min(wave_types.len() - 1);
                                    egui::ComboBox::from_id_salt(("wave_type", node.id))
                                        .selected_text(wave_types[wave_type])
                                        .show_ui(ui, |ui| {
                                            for (i, label) in wave_types.iter().enumerate() {
                                                if ui.selectable_label(wave_type == i, *label).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.mode = i as u8;
                                                    }
                                                }
                                            }
                                        });
                                    let directions = if node.mode == 0 {
                                        &["X", "Y", "Z", "Diagonal"][..]
                                    } else {
                                        &["X", "Y", "Z", "Spherical"][..]
                                    };
                                    let property = if node.mode == 0 { 8 } else { 9 };
                                    let direction =
                                        (node.props[property].round() as usize).min(directions.len() - 1);
                                    egui::ComboBox::from_id_salt(("wave_direction", node.id))
                                        .selected_text(format!("Direction: {}", directions[direction]))
                                        .show_ui(ui, |ui| {
                                            for (i, label) in directions.iter().enumerate() {
                                                if ui.selectable_label(direction == i, *label).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.props[property] = i as f32;
                                                    }
                                                }
                                            }
                                        });
                                    let profiles = ["Sine", "Saw", "Triangle"];
                                    let profile = (node.props[10].round() as usize).min(profiles.len() - 1);
                                    egui::ComboBox::from_id_salt(("wave_profile", node.id))
                                        .selected_text(format!("Profile: {}", profiles[profile]))
                                        .show_ui(ui, |ui| {
                                            for (i, label) in profiles.iter().enumerate() {
                                                if ui.selectable_label(profile == i, *label).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.props[10] = i as f32;
                                                    }
                                                }
                                            }
                                        });
                                }
                                ShaderNodeKind::Brick => {
                                    let controls = if node.props[..4].iter().all(|value| *value == 0.0) {
                                        [0.5, 2.0, 1.0, 2.0]
                                    } else {
                                        [node.props[0], node.props[1], node.props[2], node.props[3]]
                                    };
                                    let mut offset = controls[0];
                                    if ui.add(egui::DragValue::new(&mut offset).range(0.0..=1.0).prefix("Offset ")).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.props[..4].copy_from_slice(&[offset, controls[1], controls[2], controls[3]]);
                                        }
                                    }
                                    let mut offset_frequency = controls[1].round() as i32;
                                    if ui.add(egui::DragValue::new(&mut offset_frequency).range(1..=99).prefix("Offset Frequency ")).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.props[..4].copy_from_slice(&[controls[0], offset_frequency as f32, controls[2], controls[3]]);
                                        }
                                    }
                                    let mut squash = controls[2];
                                    if ui.add(egui::DragValue::new(&mut squash).range(0.0..=1.0).prefix("Squash ")).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.props[..4].copy_from_slice(&[controls[0], controls[1], squash, controls[3]]);
                                        }
                                    }
                                    let mut squash_frequency = controls[3].round() as i32;
                                    if ui.add(egui::DragValue::new(&mut squash_frequency).range(1..=99).prefix("Squash Frequency ")).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.props[..4].copy_from_slice(&[controls[0], controls[1], controls[2], squash_frequency as f32]);
                                        }
                                    }
                                }
                                ShaderNodeKind::Magic => {
                                    let mut depth = if node.props[9] > 0.5 {
                                        node.props[8].round().clamp(0.0, 10.0) as i32
                                    } else {
                                        node.mode.saturating_add(2).min(10) as i32
                                    };
                                    if ui.add(egui::DragValue::new(&mut depth).range(0..=10).prefix("Depth ")).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.props[8] = depth as f32;
                                            n.props[9] = 1.0;
                                        }
                                    }
                                }
                                ShaderNodeKind::WhiteNoise => {
                                    let dimensions = ["1D", "2D", "3D", "4D"];
                                    let dimension = (node.mode as usize).min(dimensions.len() - 1);
                                    egui::ComboBox::from_id_salt(("white_noise_dimensions", node.id))
                                        .selected_text(format!("Dimensions: {}", dimensions[dimension]))
                                        .show_ui(ui, |ui| {
                                            for (i, label) in dimensions.iter().enumerate() {
                                                if ui.selectable_label(dimension == i, *label).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.mode = i as u8;
                                                    }
                                                }
                                            }
                                        });
                                    ui.label("Use Mapping to scale coordinates; White Noise has no Detail setting.");
                                }
                                ShaderNodeKind::ColorRamp => {
                                    use forge_core::{
                                        color_ramp_eval_stops, ramp_get_stops, ramp_hue_interpolation,
                                        ramp_set_stops, ColorRampStop, COLOR_RAMP_MAX_STOPS,
                                    };
                                    let (cols, poss, interp, color_mode) = ramp_get_stops(&node);
                                    let hue_mode = ramp_hue_interpolation(&node);
                                    let n_stops = cols.len();
                                    let mut active = {
                                        let a = node.name_prop.parse::<usize>().unwrap_or(0);
                                        a.min(n_stops.saturating_sub(1))
                                    };
                                    let bar_h = (14.0 * zoom).clamp(10.0, 18.0);
                                    let (bar_rect, _) = ui.allocate_exact_size(
                                        egui::vec2(ui.available_width().max(60.0), bar_h),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        bar_rect,
                                        1.0,
                                        egui::Color32::from_gray(28),
                                    );
                                    let segs = 64;
                                    for i in 0..segs {
                                        let t0 = i as f32 / segs as f32;
                                        let t1 = (i + 1) as f32 / segs as f32;
                                        let t = (t0 + t1) * 0.5;
                                        let color = color_ramp_eval_stops(
                                            t,
                                            &cols,
                                            &poss,
                                            if color_mode == 0 { interp } else { 0 },
                                            color_mode,
                                            hue_mode,
                                        );
                                        let x0 = bar_rect.left() + bar_rect.width() * t0;
                                        let x1 = bar_rect.left() + bar_rect.width() * t1;
                                        ui.painter().rect_filled(
                                            egui::Rect::from_min_max(
                                                egui::pos2(x0, bar_rect.top()),
                                                egui::pos2(x1, bar_rect.bottom()),
                                            ),
                                            0.0,
                                            egui::Color32::from_rgb(
                                                lin_to_srgb8(color[0]),
                                                lin_to_srgb8(color[1]),
                                                lin_to_srgb8(color[2]),
                                            ),
                                        );
                                    }
                                    ui.painter().rect_stroke(
                                        bar_rect,
                                        1.0,
                                        egui::Stroke::new(
                                            1.0_f32,
                                            egui::Color32::from_gray(100),
                                        ),
                                    );
                                    let interp = if color_mode == 0 { interp } else { 0 };
                                    let mut stops: Vec<ColorRampStop> = cols
                                        .iter()
                                        .copied()
                                        .zip(poss.iter().copied())
                                        .map(|(color, position)| ColorRampStop { color, position })
                                        .collect();
                                    for (index, position) in poss.iter().copied().enumerate() {
                                        let x = bar_rect.left() + position * bar_rect.width();
                                        let handle = egui::Rect::from_center_size(
                                            egui::pos2(x, bar_rect.bottom() + 4.0 * zoom),
                                            egui::vec2(12.0, 10.0),
                                        );
                                        let response = ui.interact(
                                            handle,
                                            ui.id().with(("ramp_stop", node.id, index)),
                                            egui::Sense::click_and_drag(),
                                        );
                                        if response.clicked() || response.dragged() {
                                            active = index;
                                            if let Some(n) = st
                                                .shader_nodes
                                                .iter_mut()
                                                .find(|n| n.id == node.id)
                                            {
                                                n.name_prop = index.to_string();
                                            }
                                        }
                                        if response.dragged() {
                                            if let Some(pointer) = response.interact_pointer_pos() {
                                                let lower = if index == 0 {
                                                    0.0
                                                } else {
                                                    poss[index - 1] + 0.001
                                                };
                                                let upper = poss
                                                    .get(index + 1)
                                                    .map(|next| next - 0.001)
                                                    .unwrap_or(1.0);
                                                stops[index].position = ((pointer.x - bar_rect.left())
                                                    / bar_rect.width())
                                                .clamp(lower.min(upper), upper.max(lower));
                                                if let Some(n) = st
                                                    .shader_nodes
                                                    .iter_mut()
                                                    .find(|n| n.id == node.id)
                                                {
                                                    ramp_set_stops(n, &stops);
                                                }
                                            }
                                        }
                                        let marker_color = egui::Color32::from_rgb(
                                            lin_to_srgb8(cols[index][0]),
                                            lin_to_srgb8(cols[index][1]),
                                            lin_to_srgb8(cols[index][2]),
                                        );
                                        let triangle = vec![
                                            egui::pos2(x - 5.0 * zoom, bar_rect.bottom() + 8.0 * zoom),
                                            egui::pos2(x + 5.0 * zoom, bar_rect.bottom() + 8.0 * zoom),
                                            egui::pos2(x, bar_rect.bottom() - 1.0 * zoom),
                                        ];
                                        ui.painter().add(egui::Shape::convex_polygon(
                                            triangle,
                                            marker_color,
                                            egui::Stroke::new(
                                                if active == index { 2.0 } else { 1.0_f32 },
                                                if active == index {
                                                    egui::Color32::WHITE
                                                } else {
                                                    egui::Color32::BLACK
                                                },
                                            ),
                                        ));
                                    }
                                    ui.add_space(8.0 * zoom);
                                    let interps =
                                        ["Linear", "Ease", "B-Spline", "Cardinal", "Constant"];
                                    let m = (interp as usize).min(interps.len() - 1);
                                    egui::ComboBox::from_id_salt(("ramp_interp", node.id))
                                        .selected_text(interps[m])
                                        .width((90.0 * zoom).clamp(60.0, 120.0))
                                        .show_ui(ui, |ui| {
                                            for (i, name) in interps.iter().enumerate() {
                                                if ui.selectable_label(m == i, *name).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.mode = (n.mode & !0x0f) | i as u8;
                                                    }
                                                }
                                            }
                                        });
                                    let color_modes = ["RGB", "HSV", "HSL"];
                                    let current_color_mode = (color_mode as usize)
                                        .min(color_modes.len() - 1);
                                    egui::ComboBox::from_id_salt(("ramp_color_mode", node.id))
                                        .selected_text(format!(
                                            "Color: {}",
                                            color_modes[current_color_mode]
                                        ))
                                        .show_ui(ui, |ui| {
                                            for (i, label) in color_modes.iter().enumerate() {
                                                if ui
                                                    .selectable_label(
                                                        current_color_mode == i,
                                                        *label,
                                                    )
                                                    .clicked()
                                                {
                                                    if let Some(n) = st
                                                        .shader_nodes
                                                        .iter_mut()
                                                        .find(|n| n.id == node.id)
                                                    {
                                                        n.mode = (n.mode & !0x30) | ((i as u8) << 4);
                                                    }
                                                }
                                            }
                                        });
                                    if color_mode != 0 {
                                        let hue_modes =
                                            ["Near", "Far", "Clockwise", "Counter-clockwise"];
                                        let current_hue_mode =
                                            (hue_mode as usize).min(hue_modes.len() - 1);
                                        egui::ComboBox::from_id_salt(("ramp_hue_mode", node.id))
                                            .selected_text(format!(
                                                "Hue: {}",
                                                hue_modes[current_hue_mode]
                                            ))
                                            .show_ui(ui, |ui| {
                                                for (i, label) in hue_modes.iter().enumerate() {
                                                    if ui
                                                        .selectable_label(
                                                            current_hue_mode == i,
                                                            *label,
                                                        )
                                                        .clicked()
                                                    {
                                                        if let Some(n) = st
                                                            .shader_nodes
                                                            .iter_mut()
                                                            .find(|n| n.id == node.id)
                                                        {
                                                            n.mode =
                                                                (n.mode & !0xc0) | ((i as u8) << 6);
                                                        }
                                                    }
                                                }
                                            });
                                    }
                                    ui.horizontal(|ui| {
                                        if ui.small_button("+").clicked()
                                            && n_stops < COLOR_RAMP_MAX_STOPS
                                        {
                                            let (insert_at, position) = if n_stops == 1 {
                                                (1, (poss[0] + 1.0) * 0.5)
                                            } else if active + 1 < n_stops {
                                                (
                                                    active + 1,
                                                    (poss[active] + poss[active + 1]) * 0.5,
                                                )
                                            } else {
                                                (
                                                    active,
                                                    (poss[active - 1] + poss[active]) * 0.5,
                                                )
                                            };
                                            let color = color_ramp_eval_stops(
                                                position,
                                                &cols,
                                                &poss,
                                                if color_mode == 0 { interp } else { 0 },
                                                color_mode,
                                                hue_mode,
                                            );
                                            stops.insert(
                                                insert_at,
                                                ColorRampStop { color, position },
                                            );
                                            if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                ramp_set_stops(n, &stops);
                                                n.name_prop = insert_at.to_string();
                                            }
                                            active = insert_at;
                                        }
                                        if ui.small_button("−").clicked() && n_stops > 1 {
                                            if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                stops.remove(active);
                                                ramp_set_stops(n, &stops);
                                                active = active.min(stops.len() - 1);
                                                n.name_prop = active.to_string();
                                            }
                                        }
                                        ui.label(format!("Stop {}", active + 1));
                                    });
                                    // Blender stores ramp colors in linear space; the picker uses sRGB.
                                    let mut rgb = [
                                        lin_to_srgb8(cols[active][0]),
                                        lin_to_srgb8(cols[active][1]),
                                        lin_to_srgb8(cols[active][2]),
                                    ];
                                    if ui.color_edit_button_srgb(&mut rgb).changed() {
                                        stops[active].color[..3].copy_from_slice(&[
                                            srgb8_to_lin(rgb[0]),
                                            srgb8_to_lin(rgb[1]),
                                            srgb8_to_lin(rgb[2]),
                                        ]);
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            ramp_set_stops(n, &stops);
                                        }
                                    }
                                    let mut alpha = stops[active].color[3];
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut alpha)
                                                .range(0.0..=1.0)
                                                .prefix("Alpha "),
                                        )
                                        .changed()
                                    {
                                        stops[active].color[3] = alpha;
                                        if let Some(n) = st
                                            .shader_nodes
                                            .iter_mut()
                                            .find(|n| n.id == node.id)
                                        {
                                            ramp_set_stops(n, &stops);
                                        }
                                    }
                                    let mut position = stops[active].position;
                                    let position_min = if active == 0 {
                                        0.0
                                    } else {
                                        stops[active - 1].position
                                    };
                                    let position_max = stops
                                        .get(active + 1)
                                        .map(|stop| stop.position)
                                        .unwrap_or(1.0);
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut position)
                                                .range(position_min..=position_max)
                                                .prefix("Position "),
                                        )
                                        .changed()
                                    {
                                        stops[active].position = position;
                                        if let Some(n) = st
                                            .shader_nodes
                                            .iter_mut()
                                            .find(|n| n.id == node.id)
                                        {
                                            ramp_set_stops(n, &stops);
                                        }
                                    }
                                }
                                ShaderNodeKind::Rgb => {
                                    let mut rgb = [
                                        lin_to_srgb8(node.color[0]),
                                        lin_to_srgb8(node.color[1]),
                                        lin_to_srgb8(node.color[2]),
                                    ];
                                    if ui.color_edit_button_srgb(&mut rgb).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.color = [srgb8_to_lin(rgb[0]), srgb8_to_lin(rgb[1]), srgb8_to_lin(rgb[2]), n.color[3]];
                                        }
                                    }
                                }
                                ShaderNodeKind::Math => {
                                    let ops = ["Add","Subtract","Multiply","Divide","Power","Logarithm","Sqrt","Inverse Sqrt",
                                        "Absolute","Exponent","Minimum","Maximum","Less Than","Greater Than","Sign","Compare",
                                        "Smooth Min","Smooth Max","Round","Floor","Ceil","Truncate","Fraction","Modulo",
                                        "Wrap","Snap","Ping-Pong","Sine","Cosine","Tangent","Arcsine","Arccosine","Arctangent",
                                        "Arctan2","Sinh","Cosh","Tanh","To Radians","To Degrees"];
                                    let mode = (node.mode as usize).min(ops.len() - 1);
                                    egui::ComboBox::from_id_salt(("math", node.id))
                                        .selected_text(ops[mode])
                                        .width((100.0 * zoom).clamp(70.0, 140.0))
                                        .show_ui(ui, |ui| {
                                            for (i, name) in ops.iter().enumerate() {
                                                if ui.selectable_label(mode == i, *name).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.mode = i as u8;
                                                    }
                                                }
                                            }
                                        });
                                }
                                ShaderNodeKind::Mix | ShaderNodeKind::MixShader => {
                                    let modes = ["Mix","Darken","Multiply","Lighten","Screen","Overlay","Soft Light","Linear Light","Difference","Exclusion","Add","Subtract","Divide"];
                                    let m = (node.mode as usize).min(modes.len() - 1);
                                    egui::ComboBox::from_id_salt(("mix", node.id))
                                        .selected_text(modes[m])
                                        .width((90.0 * zoom).clamp(60.0, 120.0))
                                        .show_ui(ui, |ui| {
                                            for (i, name) in modes.iter().enumerate() {
                                                if ui.selectable_label(m == i, *name).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                        n.mode = i as u8;
                                                    }
                                                }
                                            }
                                        });
                                }
                                ShaderNodeKind::ImageTexture | ShaderNodeKind::Environment => {
                                    let ntex = doc.scene.textures.len();
                                    egui::ComboBox::from_id_salt(("img", node.id))
                                        .selected_text(
                                            node.image_index.and_then(|i| doc.scene.textures.get(i)).map(|t| t.name.as_str()).unwrap_or("(none)")
                                        )
                                        .width((110.0 * zoom).clamp(70.0, 150.0))
                                        .show_ui(ui, |ui| {
                                            if ui.selectable_label(node.image_index.is_none(), "(none)").clicked() {
                                                if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) { n.image_index = None; }
                                            }
                                            for i in 0..ntex {
                                                if ui.selectable_label(node.image_index == Some(i), &doc.scene.textures[i].name).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) { n.image_index = Some(i); }
                                                }
                                            }
                                        });
                                    let interp = ["Linear", "Closest", "Cubic", "Smart"];
                                    let m = (node.mode as usize).min(3);
                                    egui::ComboBox::from_id_salt(("imgi", node.id))
                                        .selected_text(interp[m])
                                        .width((80.0 * zoom).clamp(50.0, 110.0))
                                        .show_ui(ui, |ui| {
                                            for (i, name) in interp.iter().enumerate() {
                                                if ui.selectable_label(m == i, *name).clicked() {
                                                    if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) { n.mode = i as u8; }
                                                }
                                            }
                                        });
                                }
                                ShaderNodeKind::Noise
                                | ShaderNodeKind::Voronoi
                                | ShaderNodeKind::Gabor
                                | ShaderNodeKind::Gradient => {
                                    // Dimensions / type mode
                                    let labels = match node.kind {
                                        ShaderNodeKind::Noise => &["1D","2D","3D","4D"][..],
                                        ShaderNodeKind::Voronoi => &["F1","F2","Smooth F1","Distance to Edge","N-Sphere"][..],
                                        ShaderNodeKind::Gradient => &["Linear","Quadratic","Easing","Diagonal","Spherical","Quadratic Sphere","Radial"][..],
                                        ShaderNodeKind::Gabor => &["2D","3D"][..],
                                        _ => &["Default"][..],
                                    };
                                    if labels.len() > 1 {
                                        let m = (node.mode as usize).min(labels.len() - 1);
                                        egui::ComboBox::from_id_salt(("texmode", node.id))
                                            .selected_text(labels[m])
                                            .width((100.0 * zoom).clamp(60.0, 130.0))
                                            .show_ui(ui, |ui| {
                                                for (i, name) in labels.iter().enumerate() {
                                                    if ui.selectable_label(m == i, *name).clicked() {
                                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                                            n.mode = i as u8;
                                                        }
                                                    }
                                                }
                                            });
                                    }
                                }
                                ShaderNodeKind::Script => {
                                    let mut expr = node.name_prop.clone();
                                    if ui.text_edit_singleline(&mut expr).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.name_prop = expr;
                                        }
                                    }
                                }
                                ShaderNodeKind::Attribute | ShaderNodeKind::UvMap | ShaderNodeKind::ColorAttribute => {
                                    let mut name = node.name_prop.clone();
                                    if ui.text_edit_singleline(&mut name).changed() {
                                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                                            n.name_prop = name;
                                        }
                                    }
                                }
                                _ => {}
                            }

                            // Blender: each INPUT is connectable (●) AND has a default value
                            // you can drag when nothing is linked. Linked value overrides.
                            {
                                let inputs = node.kind.inputs();
                                for (i, sock) in inputs.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        let _ = shader_socket_control(
                                            ui,
                                            st,
                                            node.id,
                                            false,
                                            i as u8,
                                            "●",
                                        );
                                        ui.label(sock.name);
                                        match sock.ty {
                                            forge_core::SocketType::Float => {
                                                let mut v = node.socket_default(i)[0];
                                                let (speed, range) = match sock.name {
                                                    "Scale" | "Detail" | "Distortion" | "Lacunarity" => {
                                                        (0.05, 0.0..=50.0)
                                                    }
                                                    "Fac" | "Factor" | "Metallic" | "Roughness"
                                                    | "Alpha" | "Saturation" | "Hue" | "Value" => {
                                                        (0.01, 0.0..=1.0)
                                                    }
                                                    "Strength" | "Emission Strength" | "Coat Weight"
                                                    | "Sheen Weight" | "Subsurface Weight"
                                                    | "Transmission Weight" => (0.02, 0.0..=10.0),
                                                    "IOR" => (0.01, 1.0..=3.0),
                                                    "Gamma" | "Bright" | "Contrast" => {
                                                        (0.02, -2.0..=4.0)
                                                    }
                                                    "Temperature" => (10.0, 800.0..=12000.0),
                                                    "Distance" | "Randomness" | "Smoothness" => {
                                                        (0.01, 0.0..=1.0)
                                                    }
                                                    "W" => (0.01, -10.0..=10.0),
                                                    _ => (0.02, -100.0..=100.0),
                                                };
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut v)
                                                            .speed(speed)
                                                            .range(range)
                                                            .min_decimals(2)
                                                            .max_decimals(3),
                                                    )
                                                    .changed()
                                                {
                                                    if let Some(n) = st
                                                        .shader_nodes
                                                        .iter_mut()
                                                        .find(|n| n.id == node.id)
                                                    {
                                                        n.set_socket_default(i, [v, 0.0, 0.0, 0.0]);
                                                    }
                                                }
                                            }
                                            forge_core::SocketType::Color
                                            | forge_core::SocketType::Shader => {
                                                // Default color when unlinked (still connectable via ●)
                                                let default = node.socket_default(i);
                                                let mut rgb = [
                                                    lin_to_srgb8(default[0]),
                                                    lin_to_srgb8(default[1]),
                                                    lin_to_srgb8(default[2]),
                                                ];
                                                if ui.color_edit_button_srgb(&mut rgb).changed() {
                                                    if let Some(n) = st
                                                        .shader_nodes
                                                        .iter_mut()
                                                        .find(|n| n.id == node.id)
                                                    {
                                                        n.set_socket_default(i, [
                                                            srgb8_to_lin(rgb[0]),
                                                            srgb8_to_lin(rgb[1]),
                                                            srgb8_to_lin(rgb[2]),
                                                            default[3],
                                                        ]);
                                                    }
                                                }
                                            }
                                            forge_core::SocketType::Vector => {
                                                let default = node.socket_default(i);
                                                let mut x = default[0];
                                                let mut y = default[1];
                                                let mut z = default[2];
                                                let mut changed = false;
                                                changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut x)
                                                            .speed(0.01)
                                                            .prefix("X "),
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut y)
                                                            .speed(0.01)
                                                            .prefix("Y "),
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .add(
                                                        egui::DragValue::new(&mut z)
                                                            .speed(0.01)
                                                            .prefix("Z "),
                                                    )
                                                    .changed();
                                                if changed {
                                                    if let Some(n) = st
                                                        .shader_nodes
                                                        .iter_mut()
                                                        .find(|n| n.id == node.id)
                                                    {
                                                        n.set_socket_default(i, [x, y, z, 0.0]);
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    });
                                }
                            }

                            // Outputs: connectable only (right side)
                            for (i, sock) in node.kind.outputs().iter().enumerate() {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let _ = shader_socket_control(
                                            ui,
                                            st,
                                            node.id,
                                            true,
                                            i as u8,
                                            "◉",
                                        );
                                        ui.label(sock.name);
                                    },
                                );
                            }

                        });
                        ui.add_space(6.0);
                    });
                    ui.add_space(4.0);

                    let content_height = ui.min_size().y;
                    if sh > content_height {
                        ui.add_space(sh - content_height);
                    }

                    // Resize the node on its left, right, or bottom border.
                    let node_r = ui.min_rect();
                    let grip = 8.0_f32; // hit target in screen px (easy to grab)
                    let right_grip = egui::Rect::from_min_max(
                        egui::pos2(node_r.right() - grip, node_r.top() + title_h),
                        egui::pos2(node_r.right() + grip * 0.5, node_r.bottom()),
                    );
                    let left_grip = egui::Rect::from_min_max(
                        egui::pos2(node_r.left() - grip * 0.5, node_r.top() + title_h),
                        egui::pos2(node_r.left() + grip, node_r.bottom()),
                    );
                    let bottom_grip = egui::Rect::from_min_max(
                        egui::pos2(node_r.left() + 16.0, node_r.bottom() - grip * 0.5),
                        egui::pos2(node_r.right() - 16.0, node_r.bottom() + grip * 0.5),
                    );
                    let rg = ui.interact(right_grip, ui.id().with("rw"), egui::Sense::drag());
                    let lg = ui.interact(left_grip, ui.id().with("lw"), egui::Sense::drag());
                    let bg = ui.interact(bottom_grip, ui.id().with("bh"), egui::Sense::drag());

                    // Visual cue when hovering a resize edge
                    if rg.hovered() || rg.dragged() {
                        ui.painter().line_segment(
                            [
                                egui::pos2(node_r.right() - 1.0, node_r.top() + title_h + 2.0),
                                egui::pos2(node_r.right() - 1.0, node_r.bottom() - 2.0),
                            ],
                            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(0xc0, 0xa0, 0x50)),
                        );
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if lg.hovered() || lg.dragged() {
                        ui.painter().line_segment(
                            [
                                egui::pos2(node_r.left() + 1.0, node_r.top() + title_h + 2.0),
                                egui::pos2(node_r.left() + 1.0, node_r.bottom() - 2.0),
                            ],
                            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(0xc0, 0xa0, 0x50)),
                        );
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if bg.hovered() || bg.dragged() {
                        ui.painter().line_segment(
                            [
                                egui::pos2(node_r.left() + 2.0, node_r.bottom() - 1.0),
                                egui::pos2(node_r.right() - 2.0, node_r.bottom() - 1.0),
                            ],
                            egui::Stroke::new(
                                2.0_f32,
                                egui::Color32::from_rgb(0xc0, 0xa0, 0x50),
                            ),
                        );
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                    }

                    if rg.dragged() {
                        let d = ui.input(|i| i.pointer.delta());
                        let nw = (gw + d.x / zoom.max(0.25)).clamp(120.0, 500.0);
                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                            n.ui_size[0] = nw;
                        }
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if lg.dragged() {
                        let d = ui.input(|i| i.pointer.delta());
                        let dx = d.x / zoom.max(0.25);
                        let nw = (gw - dx).clamp(120.0, 500.0);
                        let applied = gw - nw;
                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                            n.ui_size[0] = nw;
                            n.pos[0] += applied;
                        }
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if bg.dragged() {
                        let d = ui.input(|i| i.pointer.delta());
                        let min_height = (content_height / zoom.max(0.25)).max(fixed_h);
                        let nh = (gh + d.y / zoom.max(0.25))
                            .clamp(min_height, 600.0);
                        if let Some(n) = st.shader_nodes.iter_mut().find(|n| n.id == node.id) {
                            n.ui_size[1] = nh;
                        }
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                    }
                });
            });
        node_rects.push((node.id, area.response.rect));
    }

    st.shader_last_node_rects = node_rects.clone();
    if ctx.input(|i| i.pointer.any_released()) {
        st.shader_drag_from = None;
    }
    if let Some(id) = delete_node {
        st.shader_nodes.retain(|n| n.id != id);
        st.shader_links.retain(|l| l.from != id && l.to != id);
        if st.shader_socket_pick.map(|p| p.0) == Some(id) {
            st.shader_socket_pick = None;
        }
        if st.shader_drag_from.map(|p| p.0) == Some(id) {
            st.shader_drag_from = None;
        }
    }

    // Draw Blender-style Bezier wires beneath the nodes.
    let wire_painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("shader_wires"),
        ))
        .with_clip_rect(canvas_rect);
    let pointer = ctx.pointer_interact_pos();
    let pointer_over_node = pointer
        .map(|p| node_rects.iter().any(|(_, rect)| rect.contains(p)))
        .unwrap_or(false);
    let mut hovered_wire = None;
    for (link_index, link) in st.shader_links.iter().enumerate() {
        if let (Some((_, from)), Some((_, to))) = (
            node_rects.iter().find(|(id, _)| *id == link.from),
            node_rects.iter().find(|(id, _)| *id == link.to),
        ) {
            // Exact measured socket centres (keyed by node + direction + socket index).
            let a = st
                .shader_socket_positions
                .get(&(link.from, true, link.from_socket))
                .map(|p| egui::pos2(p[0], p[1]))
                .unwrap_or_else(|| egui::pos2(from.right(), from.center().y));
            let b = st
                .shader_socket_positions
                .get(&(link.to, false, link.to_socket))
                .map(|p| egui::pos2(p[0], p[1]))
                .unwrap_or_else(|| egui::pos2(to.left(), to.center().y));
            let points = shader_wire_points(a, b);
            let is_hovered =
                !pointer_over_node && pointer.is_some_and(|p| shader_wire_hit(p, points, 7.0));
            if is_hovered {
                hovered_wire = Some(link_index);
            }
            wire_painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                points,
                false,
                egui::Color32::TRANSPARENT,
                egui::Stroke::new(
                    if is_hovered { 5.0 } else { 3.0_f32 },
                    if is_hovered {
                        egui::Color32::from_rgb(0xff, 0x6a, 0x55)
                    } else {
                        egui::Color32::from_rgb(0xd5, 0xa0, 0x55)
                    },
                ),
            ));
        }
    }
    if let Some(pointer) = pointer.filter(|_| hovered_wire.is_some()) {
        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        wire_painter.text(
            pointer + egui::vec2(12.0, 12.0),
            egui::Align2::LEFT_TOP,
            "Right-click wire to remove",
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
        if ctx.input(|i| i.pointer.secondary_clicked()) {
            if let Some(index) = hovered_wire {
                st.shader_links.remove(index);
                st.shader_apply_pending = true;
            }
        }
    }

    let pan_delta = ctx.input(|i| {
        let pan_button = i.pointer.button_down(egui::PointerButton::Middle)
            || (i.pointer.button_down(egui::PointerButton::Secondary) && hovered_wire.is_none())
            || (canvas_resp.dragged_by(egui::PointerButton::Primary) && !pointer_over_node);
        (pointer.is_some_and(|p| canvas_rect.contains(p)) && pan_button)
            .then_some(i.pointer.delta())
    });
    if let Some(delta) = pan_delta {
        st.shader_pan[0] += delta.x;
        st.shader_pan[1] += delta.y;
    }

    if let Some((id, is_output, socket)) = st.shader_drag_from {
        if let Some(start) = st
            .shader_socket_positions
            .get(&(id, is_output, socket))
            .copied()
        {
            if let Some(pointer) = ctx.pointer_interact_pos() {
                let a = egui::pos2(start[0], start[1]);
                let b = pointer;
                let points = shader_wire_points(a, b);
                wire_painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                    points,
                    false,
                    egui::Color32::TRANSPARENT,
                    egui::Stroke::new(2.5_f32, egui::Color32::LIGHT_BLUE),
                ));
            }
        }
    }
    if let Some((id, is_output, _)) = st.shader_socket_pick {
        let name = st
            .shader_nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.kind.title())
            .unwrap_or("Node");
        ui.painter_at(canvas_rect).text(
            canvas_rect.left_top() + egui::vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Connecting: {name} {} — click the other socket",
                if is_output { "output" } else { "input" }
            ),
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgb(0xf0, 0xc0, 0x75),
        );
    }
}

fn socket_click(st: &mut UiState, node_id: u64, is_output: bool, socket: u8) {
    if let Some((other, other_is_output, other_socket)) = st.shader_socket_pick {
        if other != node_id && other_is_output != is_output {
            let (from, to, from_socket, to_socket) = if is_output {
                (node_id, other, socket, other_socket)
            } else {
                (other, node_id, other_socket, socket)
            };
            // Replace any existing link into the same target socket.
            st.shader_links
                .retain(|l| !(l.to == to && l.to_socket == to_socket));
            st.shader_links.push(ShaderLink {
                from,
                to,
                from_socket,
                to_socket,
            });
            st.shader_apply_pending = true;
        }
        st.shader_socket_pick = None;
    } else {
        st.shader_socket_pick = Some((node_id, is_output, socket));
    }
}

fn shader_wire_points(a: egui::Pos2, b: egui::Pos2) -> [egui::Pos2; 4] {
    let bend = ((b.x - a.x).abs() * 0.5).max(55.0);
    [
        a,
        egui::pos2(a.x + bend, a.y),
        egui::pos2(b.x - bend, b.y),
        b,
    ]
}

fn shader_wire_hit(pointer: egui::Pos2, points: [egui::Pos2; 4], tolerance: f32) -> bool {
    let point_at = |t: f32| {
        let u = 1.0 - t;
        egui::pos2(
            u.powi(3) * points[0].x
                + 3.0 * u.powi(2) * t * points[1].x
                + 3.0 * u * t.powi(2) * points[2].x
                + t.powi(3) * points[3].x,
            u.powi(3) * points[0].y
                + 3.0 * u.powi(2) * t * points[1].y
                + 3.0 * u * t.powi(2) * points[2].y
                + t.powi(3) * points[3].y,
        )
    };
    let mut previous = points[0];
    for step in 1..=32 {
        let current = point_at(step as f32 / 32.0);
        let segment = current - previous;
        let length_squared = segment.length_sq();
        let along = if length_squared > 0.0 {
            ((pointer - previous).dot(segment) / length_squared).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if pointer.distance_sq(previous + segment * along) <= tolerance * tolerance {
            return true;
        }
        previous = current;
    }
    false
}

#[cfg(test)]
mod shader_wire_tests {
    use super::{shader_wire_hit, shader_wire_points};

    #[test]
    fn wire_hit_test_tracks_the_visible_curve() {
        let points = shader_wire_points(egui::pos2(0.0, 0.0), egui::pos2(100.0, 100.0));
        assert!(shader_wire_hit(egui::pos2(50.0, 50.0), points, 7.0));
        assert!(!shader_wire_hit(egui::pos2(50.0, 90.0), points, 7.0));
    }
}

fn shader_socket_control(
    ui: &mut egui::Ui,
    st: &mut UiState,
    node_id: u64,
    is_output: bool,
    socket: u8,
    label: &str,
) -> egui::Response {
    let response = ui.add(egui::Button::new(label).sense(egui::Sense::click_and_drag()));
    // Wire endpoints: left edge for inputs, right edge for outputs (Blender-accurate)
    let p = if is_output {
        [response.rect.right(), response.rect.center().y]
    } else {
        [response.rect.left(), response.rect.center().y]
    };
    st.shader_socket_positions
        .insert((node_id, is_output, socket), p);
    if response.drag_started() {
        st.shader_drag_from = Some((node_id, is_output, socket));
        st.shader_socket_pick = None;
    }
    if response.clicked() {
        socket_click(st, node_id, is_output, socket);
    }
    if response.hovered() && ui.input(|i| i.pointer.any_released()) {
        if let Some((from_id, from_output, from_socket)) = st.shader_drag_from.take() {
            if from_id != node_id && from_output != is_output {
                let (from, to, fs, ts) = if from_output {
                    (from_id, node_id, from_socket, socket)
                } else {
                    (node_id, from_id, socket, from_socket)
                };
                st.shader_links
                    .retain(|link| !(link.to == to && link.to_socket == ts));
                st.shader_links.push(ShaderLink {
                    from,
                    to,
                    from_socket: fs,
                    to_socket: ts,
                });
                st.shader_apply_pending = true;
            }
        }
    }
    response
}

fn draw_texture_paint_editor(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    doc: &mut Document,
    ts: &mut ToolState,
    st: &mut UiState,
    actions: &mut Vec<Action>,
) {
    ui.heading("Texture Paint · Image Editor");
    ui.horizontal(|ui| {
        if ui
            .selectable_label(ts.tool == Tool::PaintBrush, "Draw")
            .clicked()
        {
            ts.tool = Tool::PaintBrush;
        }
        if ui
            .selectable_label(ts.tool == Tool::PaintErase, "Erase")
            .clicked()
        {
            ts.tool = Tool::PaintErase;
        }
        ui.separator();
        ui.label("Color");
        ui.color_edit_button_rgba_unmultiplied(&mut ts.paint_color);
    });
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut ts.paint_radius, 0.002..=0.25).text("Radius"));
        ui.add(egui::Slider::new(&mut ts.paint_strength, 0.01..=1.0).text("Strength"));
    });
    ui.horizontal(|ui| {
        if ui.button("Save Image As…").clicked() {
            actions.push(Action::SavePaintImage);
        }
        if ui.button("Save Project").clicked() {
            actions.push(Action::Save);
        }
    });
    ui.separator();

    let active_texture = doc
        .active
        .and_then(|id| doc.scene.objects.iter().find(|o| o.id == id))
        .and_then(|o| o.material.or(Some(0)))
        .and_then(|mi| doc.scene.materials.get(mi))
        .and_then(|m| m.base_color_tex);

    let Some(tex_idx) = active_texture else {
        st.paint_canvas_last_uv = None;
        ui.colored_label(
            egui::Color32::YELLOW,
            "No paintable base-color image is assigned.",
        );
        ui.label("Assign or create a base-color texture for the active mesh. The 3D viewport remains on the right.");
        return;
    };
    let Some(tex) = doc.scene.textures.get(tex_idx) else {
        st.paint_canvas_last_uv = None;
        ui.colored_label(
            egui::Color32::YELLOW,
            "The active material references a missing image.",
        );
        return;
    };
    if tex.width == 0
        || tex.height == 0
        || tex.rgba8.len() != tex.width as usize * tex.height as usize * 4
    {
        st.paint_canvas_last_uv = None;
        ui.colored_label(
            egui::Color32::YELLOW,
            "The active image has invalid pixel data.",
        );
        return;
    }

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&tex.name).strong());
        ui.weak(format!("{} × {} px", tex.width, tex.height));
        ui.weak("LMB: paint · Wheel: radius");
    });

    let key = (tex_idx, tex.width, tex.height);
    let image = egui::ColorImage::from_rgba_unmultiplied(
        [tex.width as usize, tex.height as usize],
        &tex.rgba8,
    );
    if let Some(handle) = st.paint_previews.get_mut(&key) {
        if st.paint_preview_versions.get(&key).copied() != Some(doc.version) {
            handle.set(image, egui::TextureOptions::LINEAR);
            st.paint_preview_versions.insert(key, doc.version);
        }
    } else {
        let handle = ctx.load_texture(
            "blendcraft_texture_paint_canvas",
            image,
            egui::TextureOptions::LINEAR,
        );
        st.paint_previews.insert(key, handle);
        st.paint_preview_versions.insert(key, doc.version);
    }

    // Fit the image into the left editor while preserving its true aspect ratio.
    let avail = ui.available_size().max(egui::vec2(64.0, 64.0));
    let scale = (avail.x / tex.width as f32)
        .min(avail.y / tex.height as f32)
        .max(0.01);
    let image_size = egui::vec2(tex.width as f32 * scale, tex.height as f32 * scale);
    let (canvas, response) = ui.allocate_exact_size(image_size, egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas);
    painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));
    if let Some(handle) = st.paint_previews.get(&key) {
        painter.image(
            handle.id(),
            canvas,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    // Overlay the active mesh's UV edges, just like Blender's Image Editor.
    let mesh = doc.active.and_then(|id| {
        doc.edit
            .as_ref()
            .filter(|e| e.object_id == id)
            .map(|e| &e.mesh)
            .or_else(|| {
                doc.scene
                    .objects
                    .iter()
                    .find(|o| o.id == id)
                    .map(|o| o.mesh.as_ref())
            })
    });
    if let Some(mesh) = mesh {
        // Match the renderer's texture coordinate convention: UV (0, 0) is
        // the top-left texel, so the overlay must not invert V here.
        let uv_to_screen = |uv: [f32; 2]| {
            egui::pos2(
                canvas.min.x + uv[0] * canvas.width(),
                canvas.min.y + uv[1] * canvas.height(),
            )
        };
        let edge = egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgba_unmultiplied(235, 190, 80, 190),
        );
        for tri in mesh.indices.chunks_exact(3) {
            let a = uv_to_screen(mesh.vertices[tri[0] as usize].uv);
            let b = uv_to_screen(mesh.vertices[tri[1] as usize].uv);
            let c = uv_to_screen(mesh.vertices[tri[2] as usize].uv);
            painter.line_segment([a, b], edge);
            painter.line_segment([b, c], edge);
            painter.line_segment([c, a], edge);
        }
    }

    let to_uv = |pos: egui::Pos2| {
        glam::Vec2::new(
            ((pos.x - canvas.min.x) / canvas.width()).clamp(0.0, 1.0),
            ((pos.y - canvas.min.y) / canvas.height()).clamp(0.0, 1.0),
        )
    };
    if response.drag_started() {
        doc.checkpoint();
        st.paint_canvas_last_uv = response.interact_pointer_pos().map(to_uv);
    }
    if response.clicked() && !response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            doc.checkpoint();
            let uv = to_uv(pos);
            let color = if ts.tool == Tool::PaintErase {
                [0.78, 0.78, 0.78, 1.0]
            } else {
                ts.paint_color
            };
            doc.paint_at_uv(uv, color, ts.paint_radius, ts.paint_strength);
        }
    }
    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            let current = to_uv(pos);
            let previous = st.paint_canvas_last_uv.unwrap_or(current);
            let distance = (current - previous).length();
            let steps =
                ((distance / (ts.paint_radius * 0.35).max(0.002)).ceil() as usize).clamp(1, 256);
            let samples: Vec<glam::Vec2> = (1..=steps)
                .map(|i| previous.lerp(current, i as f32 / steps as f32))
                .collect();
            let color = if ts.tool == Tool::PaintErase {
                [0.78, 0.78, 0.78, 1.0]
            } else {
                ts.paint_color
            };
            doc.paint_stroke_uv(&samples, color, ts.paint_radius, ts.paint_strength);
            st.paint_canvas_last_uv = Some(current);
        }
    }
    if !ui.input(|i| i.pointer.primary_down()) {
        st.paint_canvas_last_uv = None;
    }
}

fn draw_uv_editor(
    ui: &mut egui::Ui,
    doc: &mut Document,
    ts: &mut ToolState,
    actions: &mut Vec<Action>,
) {
    ui.heading("UV Editor");
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Smart Project").clicked() {
            actions.push(Action::UvUnwrapSmart);
        }
        if ui.button("Cube").clicked() {
            actions.push(Action::UvUnwrapCube);
        }
        if ui.button("Planar").clicked() {
            actions.push(Action::UvUnwrapPlanar);
        }
        if ui.button("Pack").clicked() {
            actions.push(Action::UvPack);
        }
        if ui.button("Reset").clicked() {
            actions.push(Action::UvReset);
        }
        if ui.button("Save UV Map…").clicked() {
            actions.push(Action::UvSaveMap);
        }
        if ui.button("Load UV Map…").clicked() {
            actions.push(Action::UvLoadMap);
        }
        if ui.button("Export Layout…").clicked() {
            actions.push(Action::UvExportLayout);
        }
    });
    ui.horizontal(|ui| {
        if ui.button("Select All").clicked() {
            actions.push(Action::SelectAll);
        }
        if ui.button("Select None").clicked() {
            actions.push(Action::SelectNone);
        }
        ui.label(format!("sel {}", doc.uv_selection.len()));
    });
    ui.separator();

    let Some(id) = doc.active else {
        ui.weak("No active mesh");
        return;
    };
    // Prefer Edit Mode session mesh (live UVs)
    let mesh = if let Some(session) = doc.edit.as_ref().filter(|s| s.object_id == id) {
        &session.mesh
    } else if let Some(obj) = doc.scene.objects.iter().find(|o| o.id == id) {
        &obj.mesh
    } else {
        ui.weak("Mesh not found");
        return;
    };

    let quad_count = mesh.quad_faces().len();
    let tri_count = mesh.indices.len() / 3;
    ui.label(format!(
        "Topology: {quad_count} quad faces detected · {} remaining triangles",
        tri_count.saturating_sub(quad_count * 2)
    ));

    // 2D UV canvas — always SQUARE (UV space is 1:1 even if the panel is wide/tall)
    let avail = ui.available_size();
    let side = avail.x.min((avail.y - 4.0).max(0.0)).max(180.0);
    let size = egui::vec2(side, side);
    ui.allocate_ui_with_layout(
        egui::vec2(avail.x, side),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            let _ = ui; // layout centers children
        },
    );
    // Center by adding side space then allocating the square
    let full = ui.available_width();
    let pad = ((full - side) * 0.5).max(0.0);
    let (rect, resp) = ui
        .horizontal(|ui| {
            ui.add_space(pad);
            ui.allocate_exact_size(size, egui::Sense::click_and_drag())
        })
        .inner;
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x1e, 0x1e, 0x1e));
    let margin = 12.0;
    let inner_side = (rect.width().min(rect.height()) - margin * 2.0).max(32.0);
    let inner = egui::Rect::from_center_size(rect.center(), egui::vec2(inner_side, inner_side));
    // Checkerboard (square cells)
    let cells = 8;
    for y in 0..cells {
        for x in 0..cells {
            if (x + y) % 2 == 0 {
                let x0 = inner.min.x + inner_side * x as f32 / cells as f32;
                let y0 = inner.min.y + inner_side * (cells - 1 - y) as f32 / cells as f32;
                let x1 = inner.min.x + inner_side * (x + 1) as f32 / cells as f32;
                let y1 = inner.min.y + inner_side * (cells - y) as f32 / cells as f32;
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
                    0.0,
                    egui::Color32::from_rgb(0x28, 0x28, 0x28),
                );
            }
        }
    }
    painter.rect_stroke(
        inner,
        0.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x66, 0x66, 0x66)),
    );

    let to_screen = |uv: [f32; 2]| {
        egui::pos2(
            inner.min.x + uv[0].clamp(-0.5, 1.5) * inner_side,
            inner.max.y - uv[1].clamp(-0.5, 1.5) * inner_side,
        )
    };

    // Draw UV edges
    let edge_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(0x88, 0xaa, 0xcc));
    for tri in mesh.indices.chunks_exact(3) {
        let pts = [
            to_screen(mesh.vertices[tri[0] as usize].uv),
            to_screen(mesh.vertices[tri[1] as usize].uv),
            to_screen(mesh.vertices[tri[2] as usize].uv),
        ];
        painter.line_segment([pts[0], pts[1]], edge_stroke);
        painter.line_segment([pts[1], pts[2]], edge_stroke);
        painter.line_segment([pts[2], pts[0]], edge_stroke);
    }

    // Draw UV verts
    for (i, v) in mesh.vertices.iter().enumerate() {
        let p = to_screen(v.uv);
        let selected = doc.uv_selection.contains(&(i as u32));
        let col = if selected {
            egui::Color32::from_rgb(0xff, 0xaa, 0x33)
        } else {
            egui::Color32::from_rgb(0xdd, 0xdd, 0xdd)
        };
        painter.circle_filled(p, if selected { 3.5 } else { 2.0 }, col);
    }

    // Click to select UV verts
    if resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let u = ((pos.x - inner.min.x) / inner.width()).clamp(-0.5, 1.5);
            let v = ((inner.max.y - pos.y) / inner.height()).clamp(-0.5, 1.5);
            let max_d = 12.0 / inner.width().max(1.0);
            if let Some(idx) = forge_core::pick_uv(mesh, glam::Vec2::new(u, v), max_d) {
                let shift = ui.input(|i| i.modifiers.shift);
                if !shift {
                    doc.uv_selection.clear();
                }
                if doc.uv_selection.contains(&idx) && shift {
                    doc.uv_selection.remove(&idx);
                } else {
                    doc.uv_selection.insert(idx);
                }
            } else if !ui.input(|i| i.modifiers.shift) {
                doc.uv_selection.clear();
            }
        }
    }

    // Drag the live Edit Mode mesh, not only the scene copy. This keeps the
    // UV canvas, 3D viewport and saved project in sync while the pointer moves.
    if resp.drag_started() && !doc.uv_selection.is_empty() {
        doc.checkpoint();
    }
    if resp.dragged() && !doc.uv_selection.is_empty() {
        let delta = ui.input(|i| i.pointer.delta());
        let du = delta.x / inner.width().max(1.0);
        let dv = -delta.y / inner.height().max(1.0);
        if matches!(ts.tool, Tool::UvMove | Tool::UvSelect) || ui.input(|i| i.modifiers.command) {
            doc.uv_translate_live(du, dv);
        } else if ts.tool == Tool::UvScale {
            doc.uv_scale_live((1.0 + du * 2.0).max(0.01));
        } else if ts.tool == Tool::UvRotate {
            doc.uv_rotate_live(du * 3.0);
        }
    }

    ui.label(
        egui::RichText::new("Click select · Shift multi · Drag move · Tools: Scale/Rotate")
            .weak()
            .size(11.0),
    );
}

fn draw_tool_icon(
    ui: &mut egui::Ui,
    tool: Tool,
    rect: egui::Rect,
    selected: bool,
    armature_edit: bool,
    pose_mode: bool,
) {
    let painter = ui.painter_at(rect);
    let c = rect.center();
    let s = rect.width().min(rect.height()) * 0.38;
    let col = if selected {
        egui::Color32::from_rgb(0xf0, 0xf0, 0xf0)
    } else {
        egui::Color32::from_rgb(0xc0, 0xc0, 0xc0)
    };
    let stroke = egui::Stroke::new(1.4_f32, col);
    let thick = egui::Stroke::new(1.8_f32, col);

    if armature_edit || pose_mode {
        match tool {
            Tool::Select => {
                let tip = c + egui::vec2(-s * 0.65, -s * 0.7);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        tip,
                        tip + egui::vec2(s * 0.45, s * 0.75),
                        tip + egui::vec2(s * 0.08, s * 0.62),
                        tip + egui::vec2(-s * 0.12, s * 0.95),
                    ],
                    col,
                    egui::Stroke::NONE,
                ));
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.25, s * 0.15),
                        c + egui::vec2(s * 0.55, -s * 0.2),
                    ],
                    thick,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.25, s * 0.15), s * 0.14, col);
                painter.circle_filled(c + egui::vec2(s * 0.55, -s * 0.2), s * 0.14, col);
            }
            Tool::BoxSelect => {
                painter.rect_stroke(
                    egui::Rect::from_center_size(c, egui::vec2(s * 1.55, s * 1.35)),
                    1.0,
                    stroke,
                );
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.45, s * 0.2),
                        c + egui::vec2(s * 0.45, -s * 0.2),
                    ],
                    thick,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.45, s * 0.2), s * 0.13, col);
                painter.circle_filled(c + egui::vec2(s * 0.45, -s * 0.2), s * 0.13, col);
            }
            Tool::Lasso => {
                let pts: Vec<_> = (0..14)
                    .map(|i| {
                        let t = i as f32 / 14.0 * std::f32::consts::TAU;
                        c + egui::vec2(s * 0.78 * t.cos(), s * 0.7 * t.sin())
                    })
                    .collect();
                painter.add(egui::Shape::closed_line(pts, stroke));
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.35, s * 0.2),
                        c + egui::vec2(s * 0.35, -s * 0.2),
                    ],
                    thick,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.35, s * 0.2), s * 0.12, col);
                painter.circle_filled(c + egui::vec2(s * 0.35, -s * 0.2), s * 0.12, col);
            }
            Tool::Circle => {
                painter.circle_stroke(c, s * 0.8, thick);
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.35, s * 0.15),
                        c + egui::vec2(s * 0.35, -s * 0.15),
                    ],
                    stroke,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.35, s * 0.15), s * 0.12, col);
                painter.circle_filled(c + egui::vec2(s * 0.35, -s * 0.15), s * 0.12, col);
            }
            Tool::Move => {
                // Bone plus four-way translation arrows.
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.55, 0.0),
                        c + egui::vec2(s * 0.55, 0.0),
                    ],
                    thick,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.58, 0.0), s * 0.16, col);
                painter.circle_filled(c + egui::vec2(s * 0.58, 0.0), s * 0.16, col);
                painter.line_segment(
                    [c + egui::vec2(0.0, -s * 0.8), c + egui::vec2(0.0, s * 0.8)],
                    stroke,
                );
                for (sgn, y) in [(-1.0, -s * 0.8), (1.0, s * 0.8)] {
                    let tip = c + egui::vec2(0.0, y);
                    painter.line_segment([tip, tip + egui::vec2(-s * 0.2, sgn * s * 0.25)], stroke);
                    painter.line_segment([tip, tip + egui::vec2(s * 0.2, sgn * s * 0.25)], stroke);
                }
            }
            Tool::Rotate => {
                painter.circle_stroke(c, s * 0.72, thick);
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.45, 0.0),
                        c + egui::vec2(s * 0.45, 0.0),
                    ],
                    stroke,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.45, 0.0), s * 0.14, col);
                painter.circle_filled(c + egui::vec2(s * 0.45, 0.0), s * 0.14, col);
            }
            Tool::Scale => {
                painter.line_segment(
                    [
                        c + egui::vec2(-s * 0.62, 0.0),
                        c + egui::vec2(s * 0.62, 0.0),
                    ],
                    thick,
                );
                painter.circle_filled(c + egui::vec2(-s * 0.62, 0.0), s * 0.18, col);
                painter.circle_filled(c + egui::vec2(s * 0.62, 0.0), s * 0.18, col);
                painter.line_segment(
                    [
                        c + egui::vec2(0.0, -s * 0.85),
                        c + egui::vec2(0.0, s * 0.85),
                    ],
                    stroke,
                );
                painter.circle_filled(c, s * 0.12, col);
            }
            _ => {}
        }
        if matches!(
            tool,
            Tool::Select
                | Tool::BoxSelect
                | Tool::Circle
                | Tool::Lasso
                | Tool::Move
                | Tool::Rotate
                | Tool::Scale
        ) {
            return;
        }
    }

    match tool {
        Tool::BoneAdd => {
            painter.line_segment(
                [c + egui::vec2(-s * 0.7, 0.0), c + egui::vec2(s * 0.7, 0.0)],
                thick,
            );
            painter.line_segment(
                [c + egui::vec2(0.0, -s * 0.7), c + egui::vec2(0.0, s * 0.7)],
                thick,
            );
            painter.circle_filled(c, s * 0.22, col);
        }
        Tool::BoneExtrude => {
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.65, s * 0.55),
                    c + egui::vec2(s * 0.45, -s * 0.35),
                ],
                thick,
            );
            painter.circle_filled(c + egui::vec2(-s * 0.65, s * 0.55), s * 0.18, col);
            let tip = c + egui::vec2(s * 0.72, -s * 0.62);
            painter.add(egui::Shape::convex_polygon(
                vec![
                    tip,
                    tip + egui::vec2(-s * 0.5, s * 0.05),
                    tip + egui::vec2(-s * 0.05, s * 0.5),
                ],
                col,
                egui::Stroke::NONE,
            ));
        }
        Tool::BoneDelete => {
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.7, -s * 0.7),
                    c + egui::vec2(s * 0.7, s * 0.7),
                ],
                thick,
            );
            painter.line_segment(
                [
                    c + egui::vec2(s * 0.7, -s * 0.7),
                    c + egui::vec2(-s * 0.7, s * 0.7),
                ],
                thick,
            );
        }
        Tool::PoseClear => {
            painter.circle_stroke(c, s * 0.72, stroke);
            painter.line_segment(
                [
                    c + egui::vec2(0.0, -s * 0.95),
                    c + egui::vec2(0.0, -s * 0.35),
                ],
                thick,
            );
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.55, s * 0.25),
                    c + egui::vec2(s * 0.55, s * 0.25),
                ],
                thick,
            );
        }
        Tool::Select => {
            // Arrow pointer (northwest)
            let tip = c + egui::vec2(-s * 0.55, -s * 0.55);
            let base_r = c + egui::vec2(s * 0.35, -s * 0.15);
            let base_b = c + egui::vec2(-s * 0.15, s * 0.35);
            let notch = c + egui::vec2(s * 0.05, s * 0.05);
            painter.add(egui::Shape::convex_polygon(
                vec![tip, base_r, notch, base_b],
                col,
                egui::Stroke::NONE,
            ));
        }
        Tool::BoxSelect => {
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 1.4, s * 1.1)),
                1.0,
                stroke,
            );
        }
        Tool::Lasso => {
            // Freeform loop
            let pts: Vec<_> = (0..12)
                .map(|i| {
                    let t = i as f32 / 12.0 * std::f32::consts::TAU;
                    let r = s * (0.7 + 0.25 * (t * 3.0).sin());
                    c + egui::vec2(r * t.cos(), r * t.sin() * 0.85)
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Tool::Circle => {
            painter.circle_stroke(c, s * 0.85, stroke);
        }
        Tool::Cursor => {
            // Crosshair
            painter.line_segment([c + egui::vec2(-s, 0.0), c + egui::vec2(s, 0.0)], stroke);
            painter.line_segment([c + egui::vec2(0.0, -s), c + egui::vec2(0.0, s)], stroke);
            painter.circle_stroke(c, s * 0.35, stroke);
        }
        Tool::Move => {
            // 4-way arrows
            let arm = s * 0.9;
            painter.line_segment([c - egui::vec2(arm, 0.0), c + egui::vec2(arm, 0.0)], thick);
            painter.line_segment([c - egui::vec2(0.0, arm), c + egui::vec2(0.0, arm)], thick);
            // arrow heads
            let h = s * 0.28;
            for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                let tip = c + egui::vec2(dx * arm, dy * arm);
                let p1 = tip - egui::vec2(dx * h + dy * h * 0.6, dy * h + dx * h * 0.6);
                let p2 = tip - egui::vec2(dx * h - dy * h * 0.6, dy * h - dx * h * 0.6);
                painter.add(egui::Shape::convex_polygon(
                    vec![tip, p1, p2],
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
        Tool::Rotate => {
            // Arc with arrow
            let r = s * 0.85;
            let mut pts = Vec::new();
            for i in 0..=14 {
                let t = -0.4 + (i as f32 / 14.0) * (std::f32::consts::TAU * 0.75);
                pts.push(c + egui::vec2(r * t.cos(), r * t.sin()));
            }
            painter.add(egui::Shape::line(pts.clone(), thick));
            if let (Some(&a), Some(&b)) = (pts.iter().rev().nth(1), pts.last()) {
                let raw = b - a;
                let len = (raw.x * raw.x + raw.y * raw.y).sqrt().max(1e-5);
                let dir = raw / len;
                let n = egui::vec2(-dir.y, dir.x);
                let tip = b;
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        tip,
                        tip - dir * s * 0.35 + n * s * 0.22,
                        tip - dir * s * 0.35 - n * s * 0.22,
                    ],
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
        Tool::Scale => {
            // Corner brackets expanding
            let o = s * 0.75;
            let i = s * 0.3;
            for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                let corner = c + egui::vec2(sx * o, sy * o);
                painter.line_segment([corner, corner - egui::vec2(sx * i, 0.0)], stroke);
                painter.line_segment([corner, corner - egui::vec2(0.0, sy * i)], stroke);
            }
        }
        Tool::Transform => {
            // Combined: cross + corner marks
            let arm = s * 0.7;
            painter.line_segment([c - egui::vec2(arm, 0.0), c + egui::vec2(arm, 0.0)], stroke);
            painter.line_segment([c - egui::vec2(0.0, arm), c + egui::vec2(0.0, arm)], stroke);
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 0.9, s * 0.9)),
                0.0,
                egui::Stroke::new(1.0_f32, col),
            );
        }
        Tool::Annotate => {
            // Pencil
            let a = c + egui::vec2(-s * 0.6, s * 0.55);
            let b = c + egui::vec2(s * 0.55, -s * 0.6);
            painter.line_segment([a, b], thick);
            painter.add(egui::Shape::convex_polygon(
                vec![
                    b,
                    b + egui::vec2(-s * 0.2, -s * 0.05),
                    b + egui::vec2(-s * 0.05, -s * 0.2),
                ],
                col,
                egui::Stroke::NONE,
            ));
        }
        Tool::Measure => {
            // Ruler
            let a = c + egui::vec2(-s * 0.9, s * 0.35);
            let b = c + egui::vec2(s * 0.9, -s * 0.35);
            painter.line_segment([a, b], thick);
            for i in 1..5 {
                let t = i as f32 / 5.0;
                let p = a + (b - a) * t;
                let raw = b - a;
                let len = (raw.x * raw.x + raw.y * raw.y).sqrt().max(1e-5);
                let n = egui::vec2(-raw.y, raw.x) / len * s * 0.2;
                painter.line_segment([p - n, p + n], stroke);
            }
        }
        Tool::AddCube => {
            // Isometric cube outline
            let d = s * 0.55;
            let top = c + egui::vec2(0.0, -d * 0.7);
            let fl = c + egui::vec2(-d, d * 0.1);
            let fr = c + egui::vec2(d, d * 0.1);
            let bl = c + egui::vec2(-d * 0.55, -d * 0.25);
            let br = c + egui::vec2(d * 0.55, -d * 0.25);
            let bot = c + egui::vec2(0.0, d * 0.85);
            painter.line_segment([top, br], stroke);
            painter.line_segment([top, bl], stroke);
            painter.line_segment([br, fr], stroke);
            painter.line_segment([bl, fl], stroke);
            painter.line_segment([fr, bot], stroke);
            painter.line_segment([fl, bot], stroke);
            painter.line_segment([fl, fr], stroke);
        }
        Tool::ExtrudeRegion => {
            // Face with arrow up
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(0.0, s * 0.25),
                    egui::vec2(s * 1.2, s * 0.7),
                ),
                1.0,
                stroke,
            );
            let tip = c + egui::vec2(0.0, -s * 0.9);
            painter.line_segment([c + egui::vec2(0.0, -s * 0.1), tip], thick);
            painter.add(egui::Shape::convex_polygon(
                vec![
                    tip,
                    tip + egui::vec2(-s * 0.28, s * 0.35),
                    tip + egui::vec2(s * 0.28, s * 0.35),
                ],
                col,
                egui::Stroke::NONE,
            ));
        }
        Tool::InsetFaces => {
            // Outer rect + inner rect
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 1.5, s * 1.5)),
                1.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 0.75, s * 0.75)),
                1.0,
                stroke,
            );
        }
        Tool::Bevel => {
            // Square with cut corner
            let o = s * 0.75;
            let pts = vec![
                c + egui::vec2(-o, -o),
                c + egui::vec2(o * 0.15, -o),
                c + egui::vec2(o, -o * 0.15),
                c + egui::vec2(o, o),
                c + egui::vec2(-o, o),
            ];
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Tool::LoopCut => {
            // Vertical cut lines on a face
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 1.4, s * 1.2)),
                1.0,
                stroke,
            );
            painter.line_segment(
                [c + egui::vec2(0.0, -s * 0.6), c + egui::vec2(0.0, s * 0.6)],
                thick,
            );
        }
        Tool::Knife => {
            // Blade diagonal
            let a = c + egui::vec2(-s * 0.7, s * 0.55);
            let b = c + egui::vec2(s * 0.65, -s * 0.6);
            painter.line_segment([a, b], thick);
            painter.add(egui::Shape::convex_polygon(
                vec![b, b + egui::vec2(-s * 0.25, 0.05), a + (b - a) * 0.3],
                col,
                egui::Stroke::NONE,
            ));
        }
        Tool::PolyBuild => {
            // Triangle + plus
            let pts = vec![
                c + egui::vec2(0.0, -s * 0.85),
                c + egui::vec2(s * 0.8, s * 0.55),
                c + egui::vec2(-s * 0.8, s * 0.55),
            ];
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Tool::Spin => {
            // Spiral-ish arc stack
            for k in 0..3 {
                let r = s * (0.35 + k as f32 * 0.25);
                let mut pts = Vec::new();
                for i in 0..=10 {
                    let t = i as f32 / 10.0 * std::f32::consts::PI * 1.2;
                    pts.push(c + egui::vec2(r * t.cos(), r * t.sin() * 0.7));
                }
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
        Tool::Smooth => {
            // Wavy line
            let mut pts = Vec::new();
            for i in 0..=12 {
                let t = i as f32 / 12.0;
                let x = -s + 2.0 * s * t;
                let y = (t * std::f32::consts::TAU * 1.5).sin() * s * 0.35;
                pts.push(c + egui::vec2(x, y));
            }
            painter.add(egui::Shape::line(pts, thick));
        }
        Tool::EdgeSlide => {
            // Horizontal double arrow on edge
            painter.line_segment([c + egui::vec2(-s, 0.0), c + egui::vec2(s, 0.0)], thick);
            for dir in [-1.0_f32, 1.0] {
                let tip = c + egui::vec2(dir * s, 0.0);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        tip,
                        tip + egui::vec2(-dir * s * 0.35, -s * 0.25),
                        tip + egui::vec2(-dir * s * 0.35, s * 0.25),
                    ],
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
        Tool::ShrinkFatten => {
            // Vertical double arrow (normals)
            painter.line_segment([c + egui::vec2(0.0, -s), c + egui::vec2(0.0, s)], thick);
            for dir in [-1.0_f32, 1.0] {
                let tip = c + egui::vec2(0.0, dir * s);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        tip,
                        tip + egui::vec2(-s * 0.25, -dir * s * 0.35),
                        tip + egui::vec2(s * 0.25, -dir * s * 0.35),
                    ],
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
        Tool::Shear => {
            // Parallelogram
            let pts = vec![
                c + egui::vec2(-s * 0.9, s * 0.55),
                c + egui::vec2(-s * 0.3, -s * 0.55),
                c + egui::vec2(s * 0.9, -s * 0.55),
                c + egui::vec2(s * 0.3, s * 0.55),
            ];
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Tool::RipRegion => {
            // Split gap
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.7, -s * 0.8),
                    c + egui::vec2(-s * 0.2, s * 0.8),
                ],
                thick,
            );
            painter.line_segment(
                [
                    c + egui::vec2(s * 0.2, -s * 0.8),
                    c + egui::vec2(s * 0.7, s * 0.8),
                ],
                thick,
            );
        }
        Tool::SculptDraw | Tool::SculptDrawSharp => {
            painter.circle_stroke(c, s * 0.7, stroke);
            painter.circle_filled(c, s * 0.22, col);
            if tool == Tool::SculptDrawSharp {
                painter.line_segment([c + egui::vec2(0.0, -s), c + egui::vec2(0.0, s)], stroke);
            }
        }
        Tool::SculptClay | Tool::SculptClayStrips => {
            painter.rect_filled(
                egui::Rect::from_center_size(
                    c + egui::vec2(0.0, s * 0.15),
                    egui::vec2(s * 1.3, s * 0.55),
                ),
                2.0,
                col,
            );
            painter.circle_filled(c + egui::vec2(0.0, -s * 0.25), s * 0.35, col);
        }
        Tool::SculptLayer => {
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 1.4, s * 0.4)),
                1.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(0.0, -s * 0.45),
                    egui::vec2(s * 1.4, s * 0.4),
                ),
                1.0,
                stroke,
            );
        }
        Tool::SculptInflate | Tool::SculptBlob => {
            painter.circle_stroke(c, s * 0.5, stroke);
            painter.circle_stroke(c, s * 0.85, stroke);
        }
        Tool::SculptCrease => {
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.8, s * 0.5),
                    c + egui::vec2(0.0, -s * 0.7),
                ],
                thick,
            );
            painter.line_segment(
                [
                    c + egui::vec2(s * 0.8, s * 0.5),
                    c + egui::vec2(0.0, -s * 0.7),
                ],
                thick,
            );
        }
        Tool::SculptSmooth => {
            let mut pts = Vec::new();
            for i in 0..=10 {
                let t = i as f32 / 10.0;
                pts.push(
                    c + egui::vec2(
                        -s + 2.0 * s * t,
                        (t * std::f32::consts::TAU).sin() * s * 0.3,
                    ),
                );
            }
            painter.add(egui::Shape::line(pts, thick));
        }
        Tool::SculptFlatten | Tool::SculptFill | Tool::SculptScrape => {
            painter.line_segment([c + egui::vec2(-s, 0.0), c + egui::vec2(s, 0.0)], thick);
            painter.line_segment(
                [c + egui::vec2(0.0, -s * 0.6), c + egui::vec2(0.0, s * 0.6)],
                stroke,
            );
        }
        Tool::SculptPinch => {
            painter.line_segment([c + egui::vec2(-s, -s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(s, -s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(-s, s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(s, s * 0.5), c], thick);
        }
        Tool::SculptGrab | Tool::SculptSnakeHook | Tool::SculptNudge | Tool::SculptThumb => {
            painter.line_segment(
                [c + egui::vec2(-s * 0.5, 0.0), c + egui::vec2(s * 0.5, 0.0)],
                thick,
            );
            painter.line_segment(
                [c + egui::vec2(0.0, -s * 0.5), c + egui::vec2(0.0, s * 0.5)],
                thick,
            );
            painter.circle_stroke(c, s * 0.75, stroke);
        }
        Tool::UvSelect => {
            // Cursor arrow
            let tip = c + egui::vec2(-s * 0.55, -s * 0.75);
            let mid = c + egui::vec2(-s * 0.1, -s * 0.1);
            let wing = c + egui::vec2(-s * 0.45, s * 0.15);
            painter.line_segment([tip, mid], thick);
            painter.line_segment([tip, wing], thick);
            painter.line_segment([mid, wing], stroke);
            painter.circle_stroke(c + egui::vec2(s * 0.35, s * 0.35), s * 0.35, stroke);
        }
        Tool::UvMove => {
            // 4-way arrows (like Blender move)
            for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                let end = c + egui::vec2(dx * s * 0.85, dy * s * 0.85);
                painter.line_segment([c, end], thick);
                let ortho = egui::vec2(-dy, dx) * s * 0.25;
                let back = end - egui::vec2(dx, dy) * s * 0.3;
                painter.line_segment([end, back + ortho], stroke);
                painter.line_segment([end, back - ortho], stroke);
            }
        }
        Tool::UvRotate => {
            // Arc + arrow
            painter.circle_stroke(c, s * 0.7, stroke);
            let a0 = c + egui::vec2(s * 0.7, 0.0);
            painter.line_segment([a0, a0 + egui::vec2(-s * 0.15, -s * 0.3)], thick);
            painter.line_segment([a0, a0 + egui::vec2(-s * 0.3, s * 0.05)], thick);
        }
        Tool::UvScale => {
            // Corner brackets (scale box)
            let r = s * 0.75;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = c + egui::vec2(sx * r, sy * r);
                painter.line_segment([corner, corner - egui::vec2(sx * s * 0.35, 0.0)], thick);
                painter.line_segment([corner, corner - egui::vec2(0.0, sy * s * 0.35)], thick);
            }
        }
        Tool::UvUnwrapSmart => {
            // Unfolded net: center + side faces
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(s * 0.55, s * 0.55)),
                0.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(s * 0.7, 0.0),
                    egui::vec2(s * 0.45, s * 0.55),
                ),
                0.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(-s * 0.7, 0.0),
                    egui::vec2(s * 0.45, s * 0.55),
                ),
                0.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(0.0, -s * 0.7),
                    egui::vec2(s * 0.55, s * 0.4),
                ),
                0.0,
                stroke,
            );
        }
        Tool::UvUnwrapCube => {
            // Cube perspective
            let o = c + egui::vec2(-s * 0.25, s * 0.15);
            let f = s * 0.55;
            painter.rect_stroke(
                egui::Rect::from_min_size(o - egui::vec2(0.0, f), egui::vec2(f, f)),
                0.0,
                stroke,
            );
            // top
            painter.line_segment(
                [
                    o - egui::vec2(0.0, f),
                    o + egui::vec2(s * 0.3, -f - s * 0.25),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    o + egui::vec2(f, -f),
                    o + egui::vec2(f + s * 0.3, -f - s * 0.25),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    o + egui::vec2(s * 0.3, -f - s * 0.25),
                    o + egui::vec2(f + s * 0.3, -f - s * 0.25),
                ],
                stroke,
            );
            // side
            painter.line_segment(
                [
                    o + egui::vec2(f, 0.0),
                    o + egui::vec2(f + s * 0.3, -s * 0.25),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    o + egui::vec2(f + s * 0.3, -s * 0.25),
                    o + egui::vec2(f + s * 0.3, -f - s * 0.25),
                ],
                stroke,
            );
        }
        Tool::UvUnwrapPlanar => {
            // Plane with projection rays
            painter.rect_stroke(
                egui::Rect::from_center_size(
                    c + egui::vec2(0.0, s * 0.25),
                    egui::vec2(s * 1.4, s * 0.15),
                ),
                0.0,
                stroke,
            );
            for x in [-0.5, 0.0, 0.5] {
                painter.line_segment(
                    [
                        c + egui::vec2(x * s, -s * 0.7),
                        c + egui::vec2(x * s * 0.7, s * 0.15),
                    ],
                    stroke,
                );
            }
        }
        Tool::UvReset => {
            // Single triangle UV
            let a = c + egui::vec2(-s * 0.7, s * 0.55);
            let b = c + egui::vec2(s * 0.7, s * 0.55);
            let d = c + egui::vec2(0.0, -s * 0.7);
            painter.line_segment([a, b], thick);
            painter.line_segment([b, d], thick);
            painter.line_segment([d, a], thick);
        }
        Tool::UvPack => {
            for (ox, oy) in [(-0.45, -0.45), (0.25, -0.4), (-0.4, 0.25), (0.2, 0.2)] {
                painter.rect_stroke(
                    egui::Rect::from_center_size(
                        c + egui::vec2(ox * s, oy * s),
                        egui::vec2(s * 0.5, s * 0.5),
                    ),
                    0.0,
                    stroke,
                );
            }
        }
        Tool::PaintBrush => {
            painter.circle_stroke(c, s * 0.75, stroke);
            painter.circle_filled(c, s * 0.25, col);
        }
        Tool::PaintErase => {
            painter.circle_stroke(c, s * 0.75, stroke);
            painter.line_segment(
                [
                    c + egui::vec2(-s * 0.4, -s * 0.4),
                    c + egui::vec2(s * 0.4, s * 0.4),
                ],
                thick,
            );
        }
    }
}

// ------------------------------------------------------------------ animation & rendering workspaces

/// Blender-like Animation workspace: transport + Dope Sheet channels + timeline ruler.
fn draw_animation_workspace(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
) {
    // Header: Playback Controls (shared with Timeline in Blender)
    draw_timeline_transport(ui, doc, actions);

    ui.add_space(4.0);
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Dope Sheet").strong());
        ui.weak("· channels = objects · diamonds = keyframes · click ruler to scrub");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Only Selected").clicked() {
                // visual hint only — filtering is already selection-aware in the channel list
            }
            ui.weak("Summary");
        });
    });

    // Dope Sheet body: channel list (left) + keyframe lanes (right)
    let anim = &doc.scene.animation;
    let start = anim.frame_start as f32;
    let end = anim.frame_end as f32;
    let span = (end - start).max(1.0);
    let current = anim.frame_current;

    let available = ui.available_size();
    let channel_w = 140.0;
    let row_h = 22.0;

    // Collect channels: selected objects first, then others that have keys
    let mut channels: Vec<(u64, String, bool)> = Vec::new();
    for o in &doc.scene.objects {
        let has_keys = doc
            .scene
            .animation
            .objects
            .get(&o.id)
            .map(|a| !a.keys.is_empty())
            .unwrap_or(false);
        let selected = doc.selection.contains(&o.id);
        if selected || has_keys {
            channels.push((o.id, o.name.clone(), selected));
        }
    }
    if channels.is_empty() {
        for o in &doc.scene.objects {
            channels.push((o.id, o.name.clone(), false));
            if channels.len() >= 8 {
                break;
            }
        }
    }

    let lanes_h = (channels.len() as f32 * row_h + 8.0)
        .min(available.y - 4.0)
        .max(row_h * 2.0);
    let (body_rect, body_resp) = ui.allocate_exact_size(
        egui::vec2(available.x, lanes_h),
        egui::Sense::click_and_drag(),
    );
    let painter = ui.painter_at(body_rect);
    painter.rect_filled(body_rect, 2.0, egui::Color32::from_rgb(0x1c, 0x1c, 0x1c));

    let keys_rect = egui::Rect::from_min_max(
        egui::pos2(body_rect.left() + channel_w, body_rect.top()),
        body_rect.max,
    );
    let x_of = |f: f32| keys_rect.left() + ((f - start) / span) * keys_rect.width();

    // Frame ticks on top of keys area
    let step = if span > 200.0 {
        20.0
    } else if span > 80.0 {
        10.0
    } else {
        5.0
    };
    let mut f = (start / step).floor() * step;
    while f <= end {
        let x = x_of(f);
        let major = (f as i32) % (step as i32 * 2) == 0;
        painter.line_segment(
            [
                egui::pos2(x, keys_rect.top()),
                egui::pos2(x, keys_rect.bottom()),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(if major { 55 } else { 40 })),
        );
        if major {
            painter.text(
                egui::pos2(x, keys_rect.top() + 1.0),
                egui::Align2::CENTER_TOP,
                format!("{}", f as i32),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(140),
            );
        }
        f += step;
    }

    // Channel rows + key diamonds
    for (row, (id, name, selected)) in channels.iter().enumerate() {
        let y0 = body_rect.top() + row as f32 * row_h;
        let y1 = y0 + row_h;
        let row_rect = egui::Rect::from_min_max(
            egui::pos2(body_rect.left(), y0),
            egui::pos2(body_rect.right(), y1),
        );
        if *selected {
            painter.rect_filled(
                row_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(0x3a, 0x5a, 0x8a, 60),
            );
        } else if row % 2 == 1 {
            painter.rect_filled(
                row_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 8),
            );
        }
        // Channel name
        painter.text(
            egui::pos2(body_rect.left() + 6.0, y0 + row_h * 0.5),
            egui::Align2::LEFT_CENTER,
            name,
            egui::FontId::proportional(12.0),
            if *selected {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_gray(180)
            },
        );
        // Divider under channel column
        painter.line_segment(
            [
                egui::pos2(body_rect.left() + channel_w, y0),
                egui::pos2(body_rect.left() + channel_w, y1),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(50)),
        );
        // Keys
        if let Some(oa) = doc.scene.animation.objects.get(id) {
            for k in &oa.keys {
                let x = x_of(k.frame as f32);
                let col = if k.frame == current {
                    egui::Color32::from_rgb(0xff, 0xcc, 0x33)
                } else if *selected {
                    egui::Color32::from_rgb(0x55, 0xaa, 0xff)
                } else {
                    egui::Color32::from_rgb(0x40, 0x80, 0xc0)
                };
                let c = egui::pos2(x, y0 + row_h * 0.5);
                let s = 4.5;
                let diamond = [
                    egui::pos2(c.x, c.y - s),
                    egui::pos2(c.x + s, c.y),
                    egui::pos2(c.x, c.y + s),
                    egui::pos2(c.x - s, c.y),
                ];
                painter.add(egui::Shape::convex_polygon(
                    diamond.to_vec(),
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
    }

    // Playhead
    let px = x_of(current as f32);
    painter.line_segment(
        [
            egui::pos2(px, body_rect.top()),
            egui::pos2(px, body_rect.bottom()),
        ],
        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(0xff, 0x66, 0x44)),
    );

    // Scrub on keys area
    if body_resp.dragged() || body_resp.clicked() {
        if let Some(pos) = body_resp.interact_pointer_pos() {
            if pos.x >= keys_rect.left() {
                let t = ((pos.x - keys_rect.left()) / keys_rect.width()).clamp(0.0, 1.0);
                let frame = (start + t * span).round() as i32;
                doc.scene.animation.frame_current = frame.clamp(
                    doc.scene.animation.frame_start,
                    doc.scene.animation.frame_end,
                );
                doc.scene.animation.playing = false;
                actions.push(Action::AnimScrub);
            }
        }
    }
    let _ = st; // quiet unused if no extra state yet
}

fn draw_timeline_transport(ui: &mut egui::Ui, doc: &mut Document, actions: &mut Vec<Action>) {
    let anim = &mut doc.scene.animation;
    if anim.frame_end < anim.frame_start {
        anim.frame_end = anim.frame_start;
    }
    anim.frame_current = anim.frame_current.clamp(anim.frame_start, anim.frame_end);

    ui.horizontal(|ui| {
        let play_label = if anim.playing { "⏸" } else { "▶" };
        if ui
            .button(play_label)
            .on_hover_text("Play / Pause")
            .clicked()
        {
            actions.push(Action::ToggleAnimationPlay);
        }
        if ui.button("⏮").on_hover_text("Jump to start").clicked() {
            anim.frame_current = anim.frame_start;
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }
        if ui.button("⏭").on_hover_text("Jump to end").clicked() {
            anim.frame_current = anim.frame_end;
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }
        ui.separator();
        ui.label("Start");
        let mut fs = anim.frame_start;
        if ui.add(egui::DragValue::new(&mut fs).speed(1)).changed() {
            anim.frame_start = fs.min(anim.frame_end);
            actions.push(Action::AnimScrub);
        }
        ui.label("End");
        let mut fe = anim.frame_end;
        if ui.add(egui::DragValue::new(&mut fe).speed(1)).changed() {
            anim.frame_end = fe.max(anim.frame_start);
            actions.push(Action::AnimScrub);
        }
        ui.separator();
        ui.label("Frame");
        let mut fc = anim.frame_current;
        if ui.add(egui::DragValue::new(&mut fc).speed(1)).changed() {
            anim.frame_current = fc.clamp(anim.frame_start, anim.frame_end);
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }
        ui.label(format!("/ {}", anim.frame_end));
        ui.separator();
        ui.label("FPS");
        let mut fps = anim.fps;
        if ui
            .add(egui::DragValue::new(&mut fps).range(1..=120).speed(1))
            .changed()
        {
            anim.fps = fps.max(1);
        }
        ui.separator();
        let auto = anim.auto_key;
        if ui
            .selectable_label(auto, "● Auto Key")
            .on_hover_text("Automatically insert keys when transforming")
            .clicked()
        {
            actions.push(Action::ToggleAutoKey);
        }
        if ui
            .button("Key LocRotScale")
            .on_hover_text("Insert location, rotation & scale keys")
            .clicked()
        {
            actions.push(Action::InsertKeyframeLocRotScale);
        }
    });
}

/// Right-side Render + Output properties (Blender Properties editor in Rendering workspace).
fn draw_rendering_properties(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
) {
    if st.render_width < 64 {
        st.render_width = 1920;
    }
    if st.render_height < 64 {
        st.render_height = 1080;
    }
    if st.render_percent == 0 {
        st.render_percent = 100;
    }
    if st.render_samples_viewport == 0 {
        st.render_samples_viewport = 16;
    }
    if st.render_samples == 0 {
        st.render_samples = 64;
    }

    ui.horizontal(|ui| {
        ui.label(RichText::new("Render").strong().size(14.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .small_button("Anim")
                .on_hover_text("Render Animation")
                .clicked()
            {
                actions.push(Action::RenderAnimation);
            }
            if ui
                .small_button("Image")
                .on_hover_text("Render Image (F12)")
                .clicked()
            {
                actions.push(Action::RenderImage);
            }
        });
    });
    ui.add_space(2.0);

    // Engine
    ui.label(RichText::new("Render Engine").strong());
    ui.horizontal(|ui| {
        for (i, name) in ["Workbench", "EEVEE", "Cycles"].iter().enumerate() {
            if ui
                .selectable_label(st.render_engine as usize == i, *name)
                .clicked()
            {
                st.render_engine = i as u8;
            }
        }
    });
    ui.add_space(4.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            egui::CollapsingHeader::new(RichText::new("Sampling").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("Viewport").weak().size(11.0));
                    ui.horizontal(|ui| {
                        ui.label("Samples");
                        ui.add(
                            egui::DragValue::new(&mut st.render_samples_viewport).range(1..=4096),
                        );
                    });
                    ui.checkbox(
                        &mut st.render_temporal_reprojection,
                        "Temporal Reprojection",
                    );
                    ui.checkbox(&mut st.render_jittered_shadows, "Jittered Shadows");
                    ui.separator();
                    ui.label(RichText::new("Render").weak().size(11.0));
                    ui.horizontal(|ui| {
                        ui.label("Samples");
                        ui.add(egui::DragValue::new(&mut st.render_samples).range(1..=4096));
                    });
                });

            egui::CollapsingHeader::new(RichText::new("Shadows").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_shadows, "Shadows");
                });

            egui::CollapsingHeader::new(RichText::new("Raytracing").strong())
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_raytracing, "Raytracing");
                });

            egui::CollapsingHeader::new(RichText::new("Film").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_film_transparent, "Transparent");
                    ui.checkbox(&mut st.render_use_viewport, "Match viewport size");
                });

            egui::CollapsingHeader::new("Depth of Field")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_dof, "Depth of Field");
                });

            egui::CollapsingHeader::new("Motion Blur")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_motion_blur, "Motion Blur");
                });

            egui::CollapsingHeader::new("Simplify")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut st.render_simplify, "Simplify");
                });

            ui.add_space(8.0);
            ui.label(RichText::new("Output").strong().size(14.0));
            ui.add_space(2.0);

            egui::CollapsingHeader::new(RichText::new("Format").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Resolution X");
                        ui.add(
                            egui::DragValue::new(&mut st.render_width)
                                .range(64..=7680)
                                .speed(16),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Resolution Y");
                        ui.add(
                            egui::DragValue::new(&mut st.render_height)
                                .range(64..=4320)
                                .speed(16),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("%");
                        ui.add(
                            egui::DragValue::new(&mut st.render_percent)
                                .range(1..=200)
                                .suffix("%"),
                        );
                    });
                    let (ew, eh) = effective_render_size(st);
                    ui.label(format!("Effective: {ew} × {eh}"));
                    ui.horizontal(|ui| {
                        for (label, w, h) in [
                            ("720p", 1280u32, 720),
                            ("1080p", 1920, 1080),
                            ("4K", 3840, 2160),
                        ] {
                            if ui.small_button(label).clicked() {
                                st.render_width = w;
                                st.render_height = h;
                                st.render_use_viewport = false;
                                st.render_percent = 100;
                            }
                        }
                    });
                });

            egui::CollapsingHeader::new(RichText::new("Frame Range").strong())
                .default_open(true)
                .show(ui, |ui| {
                    let anim = &mut doc.scene.animation;
                    ui.horizontal(|ui| {
                        ui.label("Start");
                        let mut fs = anim.frame_start;
                        if ui.add(egui::DragValue::new(&mut fs).speed(1)).changed() {
                            anim.frame_start = fs.min(anim.frame_end);
                        }
                        ui.label("End");
                        let mut fe = anim.frame_end;
                        if ui.add(egui::DragValue::new(&mut fe).speed(1)).changed() {
                            anim.frame_end = fe.max(anim.frame_start);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Frame Rate");
                        let mut fps = anim.fps;
                        if ui
                            .add(egui::DragValue::new(&mut fps).range(1..=120).suffix(" fps"))
                            .changed()
                        {
                            anim.fps = fps.max(1);
                        }
                    });
                });

            egui::CollapsingHeader::new(RichText::new("Output").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("File Format");
                        for (i, name) in ["PNG", "JPEG", "EXR"].iter().enumerate() {
                            if ui
                                .selectable_label(st.render_format as usize == i, *name)
                                .clicked()
                            {
                                st.render_format = i as u8;
                            }
                        }
                    });
                    ui.weak("Path: chosen when rendering");
                });

            egui::CollapsingHeader::new(RichText::new("Color Management").strong())
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak("Display Device: sRGB");
                    ui.weak("View Transform: Standard");
                    ui.weak("Look: None");
                });

            egui::CollapsingHeader::new("Performance")
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak("Threads: Auto");
                });
        });
}

/// Center editor: Blender Image / Render Result (no 3D viewport).
fn draw_scripting_workspace(ui: &mut egui::Ui, st: &mut UiState, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Text Editor").strong());
        ui.separator();
        if ui
            .add_enabled(
                !st.python_script_running,
                egui::Button::new("Run Script  (Alt-P)"),
            )
            .clicked()
        {
            actions.push(Action::RunPythonScript);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if ui.button("Open").clicked() {
                actions.push(Action::OpenPythonScript);
            }
            if ui.button("Save").clicked() {
                actions.push(Action::SavePythonScript);
            }
        }
        if let Some(path) = st.python_script_path.as_ref() {
            ui.weak(path.display().to_string());
        }
        if st.python_script_running {
            ui.spinner();
            ui.weak("Running");
        }
    });
    ui.separator();
    let editor_height = (ui.available_height() * 0.68).max(180.0);
    ui.add_sized(
        [ui.available_width(), editor_height],
        egui::TextEdit::multiline(&mut st.python_script)
            .font(egui::TextStyle::Monospace)
            .code_editor()
            .desired_width(f32::INFINITY),
    );
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Python Console").strong());
        ui.weak("Embedded Python · Rust scene compatibility subset");
    });
    egui::ScrollArea::vertical()
        .max_height(ui.available_height())
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if st.python_script_output.is_empty() {
                ui.weak("Script output and errors will appear here.");
            } else {
                ui.label(RichText::new(&st.python_script_output).monospace().color(
                    if st.python_script_output.contains("Traceback")
                        || st.python_script_output.starts_with("Could not")
                    {
                        egui::Color32::LIGHT_RED
                    } else {
                        ui.visuals().text_color()
                    },
                ));
            }
        });
}

fn poll_python_script(doc: &mut Document, st: &mut UiState) {
    let completion = match st.python_script_rx.as_ref() {
        Some(receiver) => match receiver.try_recv() {
            Ok(completion) => Some(completion),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                st.python_script_output = "Python worker stopped without a result.".to_owned();
                st.python_script_rx = None;
                st.python_script_running = false;
                return;
            }
        },
        None => None,
    };
    let Some(completion) = completion else {
        return;
    };
    st.python_script_rx = None;
    st.python_script_running = false;
    match completion.result {
        Ok(result) if completion.document_version != doc.version => {
            st.python_script_output = format!(
                "{}{}Scene changed while the script was running; its changes were discarded.",
                result.output,
                if result.output.is_empty() { "" } else { "\n" }
            );
        }
        Ok(result) => match apply_script_result(doc, &result) {
            Ok(changed) => {
                st.python_script_output = if result.output.is_empty() {
                    if changed {
                        "Script completed; scene changes applied.".to_owned()
                    } else {
                        "Script completed successfully.".to_owned()
                    }
                } else {
                    result.output
                };
            }
            Err(error) => {
                st.python_script_output = format!(
                    "{}{}{}",
                    result.output,
                    if result.output.is_empty() { "" } else { "\n" },
                    error
                );
            }
        },
        Err(error) => {
            st.python_script_output = format!(
                "{}{}{}",
                error.output,
                if error.output.is_empty() { "" } else { "\n" },
                error.message
            );
        }
    }
}

fn apply_script_result(doc: &mut Document, result: &ScriptResult) -> Result<bool, String> {
    let initial = ScriptSnapshot::from_document(doc);
    if initial.objects == result.objects
        && initial.materials == result.materials
        && initial.collections == result.collections
        && initial.keyframes == result.keyframes
        && initial.frame_current == result.frame_current
        && initial.active == result.active
    {
        return Ok(false);
    }
    if result.objects.iter().any(|object| {
        object
            .location
            .iter()
            .chain(object.rotation_euler.iter())
            .chain(object.scale.iter())
            .any(|value| !value.is_finite())
    }) {
        return Err(
            "Script produced a non-finite transform; no scene changes were applied.".into(),
        );
    }
    for object in &result.objects {
        let is_light_or_camera = object.object_type == "LIGHT" || object.object_type == "CAMERA" || object.object_type == "EMPTY";
        if object.id.is_none() && !is_light_or_camera && primitive_from_script(object).is_none() {
            return Err(format!(
                "BlendCraft scripting does not support creating {} objects yet.",
                if object.primitive.is_empty() {
                    object.object_type.as_str()
                } else {
                    object.primitive.as_str()
                }
            ));
        }
        if let Some(p) = object.parent {
            if p >= result.objects.len() {
                return Err(format!(
                    "Script parent index {p} for object {} is out of range.",
                    object.name
                ));
            }
        }
        if object
            .material
            .is_some_and(|index| index >= result.materials.len())
        {
            return Err(format!(
                "Script assigned an invalid material to object {}; no scene changes were applied.",
                object.name
            ));
        }
    }
    if result.materials.iter().any(|material| {
        material
            .diffuse_color
            .iter()
            .chain(material.emissive.iter())
            .chain([&material.metallic, &material.roughness])
            .any(|value| !value.is_finite())
    }) {
        return Err(
            "Script produced non-finite material values; no scene changes were applied.".into(),
        );
    }
    if result
        .active
        .is_some_and(|index| index >= result.objects.len())
    {
        return Err("Script returned an invalid active object.".into());
    }

    doc.checkpoint();
    for (index, scripted) in result.materials.iter().enumerate() {
        if let Some(material) = doc.scene.materials.get_mut(index) {
            material.name.clone_from(&scripted.name);
            material.base_color = scripted.diffuse_color;
            material.metallic = scripted.metallic;
            material.roughness = scripted.roughness;
            material.emissive = scripted.emissive;
        } else {
            let mut material = forge_core::Material::default();
            material.name.clone_from(&scripted.name);
            material.base_color = scripted.diffuse_color;
            material.metallic = scripted.metallic;
            material.roughness = scripted.roughness;
            material.emissive = scripted.emissive;
            doc.scene.materials.push(material);
        }
    }
    let retained_ids = result
        .objects
        .iter()
        .filter_map(|object| object.id)
        .collect::<std::collections::HashSet<_>>();
    let removed_ids = doc
        .scene
        .objects
        .iter()
        .filter(|object| !retained_ids.contains(&object.id))
        .map(|object| object.id)
        .collect::<std::collections::HashSet<_>>();
    doc.scene
        .objects
        .retain(|object| !removed_ids.contains(&object.id));
    for object in &mut doc.scene.objects {
        if object
            .parent
            .is_some_and(|parent| removed_ids.contains(&parent))
        {
            object.parent = None;
        }
    }

    let mut result_ids = Vec::with_capacity(result.objects.len());
    for scripted in &result.objects {
        let id = if let Some(id) = scripted.id {
            id
        } else if scripted.object_type == "LIGHT" {
            let lt = match scripted.light_type.as_deref().unwrap_or("POINT") {
                "SUN" => forge_core::LightType::Sun,
                "SPOT" => forge_core::LightType::Spot,
                "AREA" => forge_core::LightType::Area,
                _ => forge_core::LightType::Point,
            };
            let mut ld = forge_core::LightData::new(lt);
            if let Some(e) = scripted.light_energy {
                ld.intensity = e;
            }
            if let Some(c) = scripted.light_color {
                ld.color = c;
            }
            let mut object =
                forge_core::scene::Object::new(scripted.name.clone(), forge_core::MeshData::default());
            let id = doc.scene.alloc_id();
            object.id = id;
            object.kind = forge_core::ObjectKind::Light(ld);
            object.collection = doc.active_collection.filter(|collection| {
                doc.scene
                    .collections
                    .iter()
                    .any(|item| item.id == *collection)
            });
            doc.scene.objects.push(object);
            id
        } else if scripted.object_type == "CAMERA" {
            let mut cd = forge_core::CameraData::default();
            if let Some(f) = scripted.camera_fov {
                cd.fov_deg = f;
            }
            if let Some(o) = scripted.camera_ortho {
                cd.ortho = o;
            }
            if let Some(s) = scripted.camera_ortho_size {
                cd.ortho_size = s;
            }
            if let Some(n) = scripted.camera_near {
                cd.near = n;
            }
            if let Some(f) = scripted.camera_far {
                cd.far = f;
            }
            let mut object =
                forge_core::scene::Object::new(scripted.name.clone(), forge_core::MeshData::default());
            let id = doc.scene.alloc_id();
            object.id = id;
            object.kind = forge_core::ObjectKind::Camera(cd);
            object.collection = doc.active_collection.filter(|collection| {
                doc.scene
                    .collections
                    .iter()
                    .any(|item| item.id == *collection)
            });
            doc.scene.objects.push(object);
            id
        } else if scripted.object_type == "EMPTY" {
            let shape = match scripted.empty_shape.as_deref().unwrap_or("Plain Axes") {
                "Arrows" => forge_core::EmptyShape::Arrows,
                "Single Arrow" => forge_core::EmptyShape::SingleArrow,
                "Circle" => forge_core::EmptyShape::Circle,
                "Cube" => forge_core::EmptyShape::Cube,
                "Sphere" => forge_core::EmptyShape::Sphere,
                "Cone" => forge_core::EmptyShape::Cone,
                _ => forge_core::EmptyShape::PlainAxes,
            };
            let size = scripted.empty_size.unwrap_or(1.0);
            let mut object =
                forge_core::scene::Object::new(scripted.name.clone(), forge_core::MeshData::default());
            let id = doc.scene.alloc_id();
            object.id = id;
            object.kind = forge_core::ObjectKind::Empty { shape, size };
            object.collection = doc.active_collection.filter(|collection| {
                doc.scene
                    .collections
                    .iter()
                    .any(|item| item.id == *collection)
            });
            doc.scene.objects.push(object);
            id
        } else {
            let primitive = primitive_from_script(scripted)
                .expect("script primitives were validated before scene mutation");
            let mut object =
                forge_core::scene::Object::new(scripted.name.clone(), primitive.build());
            let id = doc.scene.alloc_id();
            object.id = id;
            object.primitive = Some(primitive);
            object.collection = doc.active_collection.filter(|collection| {
                doc.scene
                    .collections
                    .iter()
                    .any(|item| item.id == *collection)
            });
            doc.scene.objects.push(object);
            id
        };
        let object = doc
            .scene
            .objects
            .iter_mut()
            .find(|object| object.id == id)
            .ok_or_else(|| format!("Script object {} no longer exists.", scripted.name))?;
        object.name.clone_from(&scripted.name);
        object.translation = Vec3::from_array(scripted.location);
        object.rotation = crate::scripting::rotation_quaternion(scripted.rotation_euler);
        object.scale = Vec3::from_array(scripted.scale);
        object.material = scripted.material;
        object.visible = scripted.visible;
        // Update light/camera data on existing objects too
        match scripted.object_type.as_str() {
            "LIGHT" => {
                let lt = match scripted.light_type.as_deref().unwrap_or("POINT") {
                    "SUN" => forge_core::LightType::Sun,
                    "SPOT" => forge_core::LightType::Spot,
                    "AREA" => forge_core::LightType::Area,
                    _ => forge_core::LightType::Point,
                };
                let mut ld = match &object.kind {
                    forge_core::ObjectKind::Light(l) => *l,
                    _ => forge_core::LightData::new(lt),
                };
                ld.kind = lt;
                if let Some(e) = scripted.light_energy {
                    ld.intensity = e;
                }
                if let Some(c) = scripted.light_color {
                    ld.color = c;
                }
                if let Some(r) = scripted.light_range {
                    ld.range = r;
                }
                if let Some(s) = scripted.light_spot_size {
                    ld.spot_angle_deg = s;
                }
                if let Some(b) = scripted.light_spot_blend {
                    ld.spot_blend = b;
                }
                object.kind = forge_core::ObjectKind::Light(ld);
            }
            "CAMERA" => {
                let mut cd = match &object.kind {
                    forge_core::ObjectKind::Camera(c) => *c,
                    _ => forge_core::CameraData::default(),
                };
                if let Some(f) = scripted.camera_fov {
                    cd.fov_deg = f;
                }
                if let Some(o) = scripted.camera_ortho {
                    cd.ortho = o;
                }
                if let Some(s) = scripted.camera_ortho_size {
                    cd.ortho_size = s;
                }
                if let Some(n) = scripted.camera_near {
                    cd.near = n;
                }
                if let Some(f) = scripted.camera_far {
                    cd.far = f;
                }
                object.kind = forge_core::ObjectKind::Camera(cd);
            }
            "EMPTY" => {
                let shape = match scripted.empty_shape.as_deref().unwrap_or("Plain Axes") {
                    "Arrows" => forge_core::EmptyShape::Arrows,
                    "Single Arrow" => forge_core::EmptyShape::SingleArrow,
                    "Circle" => forge_core::EmptyShape::Circle,
                    "Cube" => forge_core::EmptyShape::Cube,
                    "Sphere" => forge_core::EmptyShape::Sphere,
                    "Cone" => forge_core::EmptyShape::Cone,
                    _ => forge_core::EmptyShape::PlainAxes,
                };
                let size = scripted.empty_size.unwrap_or(1.0);
                object.kind = forge_core::ObjectKind::Empty { shape, size };
            }
            _ => {}
        }
        // Material slots: slot 0 drives object.material
        if let Some(Some(mi)) = scripted.material_slots.get(scripted.active_material_index) {
            object.material = Some(*mi);
        } else if let Some(Some(mi)) = scripted.material_slots.first() {
            object.material = Some(*mi);
        }
        result_ids.push(id);
    }
    // Second pass: parenting (indices → ids)
    for (scripted, id) in result.objects.iter().zip(result_ids.iter().copied()) {
        if let Some(pidx) = scripted.parent {
            if let Some(&pid) = result_ids.get(pidx) {
                if pid != id {
                    if let Some(object) = doc.scene.objects.iter_mut().find(|o| o.id == id) {
                        object.parent = Some(pid);
                    }
                }
            }
        } else if let Some(object) = doc.scene.objects.iter_mut().find(|o| o.id == id) {
            object.parent = None;
        }
    }

    // Collections
    let mut col_ids = Vec::with_capacity(result.collections.len());
    for scripted in &result.collections {
        let id = if let Some(id) = scripted.id {
            if let Some(c) = doc.scene.collections.iter_mut().find(|c| c.id == id) {
                c.name.clone_from(&scripted.name);
                c.visible = scripted.visible;
            }
            id
        } else {
            let id = doc.scene.alloc_id();
            doc.scene.collections.push(forge_core::Collection {
                id,
                name: scripted.name.clone(),
                visible: scripted.visible,
            });
            id
        };
        col_ids.push(id);
    }
    // Link objects to collections
    for (scripted, id) in result.objects.iter().zip(result_ids.iter().copied()) {
        if let Some(object) = doc.scene.objects.iter_mut().find(|o| o.id == id) {
            object.collection = scripted
                .collection
                .and_then(|i| col_ids.get(i).copied());
        }
    }

    // Keyframes
    doc.scene.animation.frame_current = result.frame_current;
    for kf in &result.keyframes {
        let Some(&oid) = result_ids.get(kf.object) else {
            continue;
        };
        let anim = doc.scene.animation.objects.entry(oid).or_default();
        match kf.data_path.as_str() {
            "location" if kf.value.len() >= 3 => {
                anim.insert_or_merge(
                    kf.frame,
                    Some(Vec3::new(kf.value[0], kf.value[1], kf.value[2])),
                    None,
                    None,
                );
            }
            "rotation_euler" if kf.value.len() >= 3 => {
                let q = crate::scripting::rotation_quaternion([
                    kf.value[0],
                    kf.value[1],
                    kf.value[2],
                ]);
                anim.insert_or_merge(kf.frame, None, Some(q), None);
            }
            "scale" if kf.value.len() >= 3 => {
                anim.insert_or_merge(
                    kf.frame,
                    None,
                    None,
                    Some(Vec3::new(kf.value[0], kf.value[1], kf.value[2])),
                );
            }
            _ => {}
        }
    }

    doc.selection.clear();
    for (scripted, id) in result.objects.iter().zip(result_ids.iter().copied()) {
        if scripted.selected {
            doc.selection.insert(id);
        }
    }
    doc.active = result
        .active
        .and_then(|index| result_ids.get(index).copied());
    doc.touch();
    Ok(true)
}

fn primitive_from_script(object: &crate::scripting::ScriptObject) -> Option<Primitive> {
    let parameter = |index: usize, default: f32| {
        object
            .primitive_params
            .get(index)
            .copied()
            .unwrap_or(default)
    };
    use Primitive as P;
    Some(match object.primitive.as_str() {
        "Cube" => P::Cube {
            size: parameter(0, 2.0),
        },
        "Plane" => P::Plane {
            size: parameter(0, 2.0),
        },
        "Sphere" => P::Sphere {
            radius: parameter(0, 1.0),
            segments: parameter(1, 32.0) as u32,
            rings: parameter(2, 16.0) as u32,
        },
        "Ico Sphere" | "IcoSphere" => P::IcoSphere {
            radius: parameter(0, 1.0),
            subdivisions: parameter(1, 2.0) as u32,
        },
        "Cylinder" => P::Cylinder {
            radius: parameter(0, 1.0),
            height: parameter(1, 2.0),
            segments: parameter(2, 32.0) as u32,
        },
        "Cone" => P::Cone {
            radius: parameter(0, 1.0),
            height: parameter(1, 2.0),
            segments: parameter(2, 32.0) as u32,
        },
        "Torus" => P::Torus {
            major_radius: parameter(0, 1.0),
            minor_radius: parameter(1, 0.25),
            major_segments: parameter(2, 48.0) as u32,
            minor_segments: parameter(3, 12.0) as u32,
        },
        "Monkey" => P::Monkey {
            size: parameter(0, 2.0),
        },
        "Circle" => P::Circle {
            radius: parameter(0, 1.0),
            segments: parameter(1, 32.0) as u32,
        },
        "Grid" => P::Grid {
            size: parameter(0, 2.0),
            x_div: parameter(1, 10.0) as u32,
            y_div: parameter(2, 10.0) as u32,
        },
        "Capsule" => P::Capsule {
            radius: parameter(0, 0.5),
            height: parameter(1, 2.0),
            segments: parameter(2, 16.0) as u32,
            rings: parameter(3, 8.0) as u32,
        },
        "Pyramid" => P::Pyramid {
            size: parameter(0, 2.0),
            height: parameter(1, 2.0),
        },
        "Tube" => P::Tube {
            radius: parameter(0, 1.0),
            height: parameter(1, 2.0),
            segments: parameter(2, 32.0) as u32,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod scripting_apply_tests {
    use super::{apply_script_result, poll_python_script, ScriptResult, UiState};
    use crate::scripting::{ScriptCompletion, ScriptMaterial, ScriptObject};
    use forge_core::Document;
    use glam::Vec3;

    #[test]
    fn scene_mutations_are_applied_as_one_undoable_change() {
        let mut doc = Document::default();
        doc.scene.objects.clear();
        doc.selection.clear();
        doc.active = None;
        let result = ScriptResult {
            objects: vec![ScriptObject {
                id: None,
                name: "Script Cube".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Cube".to_owned(),
                primitive_params: vec![2.0],
                material: None,
                location: [1.0, 2.0, 3.0],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: true,
                visible: true,
                parent: None,
                light_type: None,
                light_energy: None,
                light_color: None,
                camera_fov: None,
                camera_ortho: None,
                camera_ortho_size: None,
                camera_near: None,
                camera_far: None,
                empty_shape: None,
                empty_size: None,
                material_slots: Vec::new(),
                active_material_index: 0,
                collection: None,
                light_range: None,
                light_spot_size: None,
                light_spot_blend: None,
                vertices: Vec::new(),
            }],
            materials: Vec::new(),
            collections: Vec::new(),
            keyframes: Vec::new(),
            frame_current: 1,
            active: Some(0),
            output: String::new(),
        };

        assert!(apply_script_result(&mut doc, &result).unwrap());
        assert_eq!(doc.scene.objects.len(), 1);
        assert_eq!(doc.scene.objects[0].name, "Script Cube");
        assert_eq!(doc.scene.objects[0].translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(doc.active, Some(doc.scene.objects[0].id));
        assert!(doc.selection.contains(&doc.scene.objects[0].id));
        assert!(doc.undo());
        assert!(doc.scene.objects.is_empty());
    }

    #[test]
    fn invalid_script_transforms_do_not_mutate_the_scene() {
        let mut doc = Document::default();
        doc.scene.objects.clear();
        doc.active = None;
        let result = ScriptResult {
            objects: vec![ScriptObject {
                id: None,
                name: "Invalid Cube".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Cube".to_owned(),
                primitive_params: vec![2.0],
                material: None,
                location: [f32::NAN, 0.0, 0.0],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: false,
                visible: true,
                parent: None,
                light_type: None,
                light_energy: None,
                light_color: None,
                camera_fov: None,
                camera_ortho: None,
                camera_ortho_size: None,
                camera_near: None,
                camera_far: None,
                empty_shape: None,
                empty_size: None,
                material_slots: Vec::new(),
                active_material_index: 0,
                collection: None,
                light_range: None,
                light_spot_size: None,
                light_spot_blend: None,
                vertices: Vec::new(),
            }],
            materials: Vec::new(),
            collections: Vec::new(),
            keyframes: Vec::new(),
            frame_current: 1,
            active: None,
            output: String::new(),
        };

        assert!(apply_script_result(&mut doc, &result).is_err());
        assert!(doc.scene.objects.is_empty());
        assert_eq!(doc.undo_len(), 0);
    }

    #[test]
    fn generated_mesh_primitives_are_built_with_their_parameters() {
        let mut doc = Document::default();
        doc.scene.objects.clear();
        doc.active = None;
        let result = ScriptResult {
            objects: vec![ScriptObject {
                id: None,
                name: "Sphere".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Sphere".to_owned(),
                primitive_params: vec![1.5, 12.0, 8.0],
                material: None,
                location: [0.0; 3],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: true,
                visible: true,
                parent: None,
                light_type: None,
                light_energy: None,
                light_color: None,
                camera_fov: None,
                camera_ortho: None,
                camera_ortho_size: None,
                camera_near: None,
                camera_far: None,
                empty_shape: None,
                empty_size: None,
                material_slots: Vec::new(),
                active_material_index: 0,
                collection: None,
                light_range: None,
                light_spot_size: None,
                light_spot_blend: None,
                vertices: Vec::new(),
            }],
            materials: Vec::new(),
            collections: Vec::new(),
            keyframes: Vec::new(),
            frame_current: 1,
            active: Some(0),
            output: String::new(),
        };

        apply_script_result(&mut doc, &result).unwrap();

        assert_eq!(
            doc.scene.objects[0].primitive,
            Some(forge_core::Primitive::Sphere {
                radius: 1.5,
                segments: 12,
                rings: 8
            })
        );
    }

    #[test]
    fn script_materials_are_applied_and_assigned_without_losing_other_fields() {
        let mut doc = Document::default();
        doc.scene.objects.clear();
        doc.scene.materials.clear();
        let mut existing_material = forge_core::Material::default();
        existing_material.alpha_mode = forge_core::AlphaMode::Blend;
        existing_material.normal_scale = 0.25;
        doc.scene.materials.push(existing_material);
        doc.active = None;
        let result = ScriptResult {
            objects: vec![ScriptObject {
                id: None,
                name: "Colored Cube".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Cube".to_owned(),
                primitive_params: vec![2.0],
                material: Some(0),
                location: [0.0; 3],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: true,
                visible: true,
                parent: None,
                light_type: None,
                light_energy: None,
                light_color: None,
                camera_fov: None,
                camera_ortho: None,
                camera_ortho_size: None,
                camera_near: None,
                camera_far: None,
                empty_shape: None,
                empty_size: None,
                material_slots: Vec::new(),
                active_material_index: 0,
                collection: None,
                light_range: None,
                light_spot_size: None,
                light_spot_blend: None,
                vertices: Vec::new(),
            }],
            materials: vec![ScriptMaterial {
                name: "Script Material".to_owned(),
                diffuse_color: [1.0, 0.0, 0.0, 1.0],
                metallic: 0.8,
                roughness: 0.2,
                emissive: [0.1, 0.0, 0.0],
            }],
            collections: Vec::new(),
            keyframes: Vec::new(),
            frame_current: 1,
            active: Some(0),
            output: String::new(),
        };

        assert!(apply_script_result(&mut doc, &result).unwrap());

        assert_eq!(doc.scene.objects[0].material, Some(0));
        assert_eq!(doc.scene.materials[0].name, "Script Material");
        assert_eq!(doc.scene.materials[0].base_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(doc.scene.materials[0].metallic, 0.8);
        assert_eq!(doc.scene.materials[0].roughness, 0.2);
        assert_eq!(doc.scene.materials[0].emissive, [0.1, 0.0, 0.0]);
        assert_eq!(
            doc.scene.materials[0].alpha_mode,
            forge_core::AlphaMode::Blend
        );
        assert_eq!(doc.scene.materials[0].normal_scale, 0.25);
    }

    #[test]
    fn stale_script_results_are_discarded_after_scene_edits() {
        let mut doc = Document::default();
        doc.scene.objects.clear();
        doc.active = None;
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut ui = UiState::default();
        ui.python_script_rx = Some(receiver);
        ui.python_script_running = true;
        sender
            .send(ScriptCompletion {
                document_version: doc.version.wrapping_sub(1),
                result: Ok(ScriptResult {
                    objects: vec![ScriptObject {
                        id: None,
                        name: "Stale Cube".to_owned(),
                        object_type: "MESH".to_owned(),
                        primitive: "Cube".to_owned(),
                        primitive_params: vec![2.0],
                        material: None,
                        material_slots: Vec::new(),
                        active_material_index: 0,
                        location: [0.0; 3],
                        rotation_euler: [0.0; 3],
                        scale: [1.0; 3],
                        selected: true,
                        visible: true,
                        parent: None,
                        collection: None,
                        light_type: None,
                        light_energy: None,
                        light_color: None,
                        light_range: None,
                        light_spot_size: None,
                        light_spot_blend: None,
                        camera_fov: None,
                        camera_ortho: None,
                        camera_ortho_size: None,
                        camera_near: None,
                        camera_far: None,
                        empty_shape: None,
                        empty_size: None,
                        vertices: Vec::new(),
                    }],
                    materials: Vec::new(),
                    collections: Vec::new(),
                    keyframes: Vec::new(),
                    frame_current: 1,
                    active: Some(0),
                    output: String::new(),
                }),
            })
            .unwrap();

        poll_python_script(&mut doc, &mut ui);

        assert!(doc.scene.objects.is_empty());
        assert!(ui
            .python_script_output
            .contains("its changes were discarded"));
    }
}

fn draw_render_result_view(ui: &mut egui::Ui, st: &mut UiState, actions: &mut Vec<Action>) {
    // Header like Blender Image editor
    ui.horizontal(|ui| {
        ui.label(RichText::new("View").weak());
        ui.separator();
        ui.label(RichText::new("Image").strong());
        ui.separator();
        ui.label(RichText::new("Render Result").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Render Animation").clicked() {
                actions.push(Action::RenderAnimation);
            }
            if ui.button("Render Image").on_hover_text("F12").clicked() {
                actions.push(Action::RenderImage);
            }
            let (ew, eh) = effective_render_size(st);
            ui.weak(format!("{ew} × {eh}"));
        });
    });
    ui.add_space(4.0);

    // Fill remaining space with checkerboard "empty render"
    let avail = ui.available_size();
    let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));

    // Aspect-correct frame for output resolution
    let (ew, eh) = effective_render_size(st);
    let aspect = ew as f32 / eh.max(1) as f32;
    let pad = 24.0;
    let max_w = (rect.width() - pad * 2.0).max(32.0);
    let max_h = (rect.height() - pad * 2.0).max(32.0);
    let (fw, fh) = if max_w / max_h > aspect {
        (max_h * aspect, max_h)
    } else {
        (max_w, max_w / aspect)
    };
    let frame = egui::Rect::from_center_size(rect.center(), egui::vec2(fw, fh));

    // Checkerboard inside frame
    let cell = 16.0;
    let light = egui::Color32::from_rgb(0x3a, 0x3a, 0x3a);
    let dark = egui::Color32::from_rgb(0x2c, 0x2c, 0x2c);
    painter.rect_filled(frame, 2.0, dark);
    let cols_n = (frame.width() / cell).ceil() as i32;
    let rows_n = (frame.height() / cell).ceil() as i32;
    for row in 0..rows_n {
        for col in 0..cols_n {
            if (row + col) % 2 == 0 {
                let x0 = frame.left() + col as f32 * cell;
                let y0 = frame.top() + row as f32 * cell;
                let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell))
                    .intersect(frame);
                painter.rect_filled(r, 0.0, light);
            }
        }
    }
    painter.rect_stroke(
        frame,
        2.0,
        egui::Stroke::new(1.5_f32, egui::Color32::from_gray(70)),
    );
    painter.text(
        frame.center(),
        egui::Align2::CENTER_CENTER,
        "Render Result\n(empty — press Render Image)",
        egui::FontId::proportional(14.0),
        egui::Color32::from_gray(170),
    );
}

fn draw_compositor_preview(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
) {
    poll_compositor_evaluation(ctx, doc, st);
    let output_current = st.compositor_output_key
        == Some((
            st.compositor_render_serial,
            graph_fingerprint(&doc.scene.compositor_graph),
        ));
    ui.horizontal(|ui| {
        ui.label(RichText::new("Compositor").strong());
        ui.separator();
        if ui
            .button("Render")
            .on_hover_text("Render the current frame and process the compositor graph")
            .clicked()
        {
            actions.push(Action::RenderCompositor);
        }
        if ui
            .add_enabled(
                output_current && !st.compositor_output_rgba.is_empty(),
                egui::Button::new("Save PNG…"),
            )
            .clicked()
        {
            actions.push(Action::SaveCompositor);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if st.compositor_eval_error {
                ui.colored_label(egui::Color32::LIGHT_RED, "Graph evaluation failed");
            } else if st.compositor_eval_busy
                || (st.compositor_render_serial > 0 && !output_current)
            {
                ui.spinner();
                ui.weak("Compositing…");
            } else if st.compositor_render_serial > 0 {
                ui.weak(format!(
                    "{} × {}",
                    st.compositor_render_width, st.compositor_render_height
                ));
            } else {
                ui.weak("Render a frame to preview the node graph");
            }
        });
    });

    let avail = ui.available_size();
    let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));
    if let Some(texture) = st.compositor_output_texture.as_ref() {
        let aspect =
            st.compositor_render_width.max(1) as f32 / st.compositor_render_height.max(1) as f32;
        let pad = 24.0;
        let max_w = (rect.width() - pad * 2.0).max(32.0);
        let max_h = (rect.height() - pad * 2.0).max(32.0);
        let (w, h) = if max_w / max_h > aspect {
            (max_h * aspect, max_h)
        } else {
            (max_w, max_w / aspect)
        };
        let image_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(w, h));
        painter.image(
            texture.id(),
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        painter.rect_stroke(
            image_rect,
            0.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(65)),
        );
    } else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Render a frame to see the composited result",
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(170),
        );
    }
}

fn poll_compositor_evaluation(ctx: &egui::Context, doc: &mut Document, st: &mut UiState) {
    if let Some(rx) = st.compositor_eval_rx.as_ref() {
        match rx.try_recv() {
            Ok(done) => {
                st.compositor_eval_rx = None;
                st.compositor_eval_busy = false;
                if graph_fingerprint(&done.graph) == graph_fingerprint(&doc.scene.compositor_graph)
                    && done.render_serial == st.compositor_render_serial
                {
                    if let Some(rgba) = done.rgba {
                        st.compositor_eval_error = false;
                        st.compositor_output_rgba = rgba;
                        st.compositor_output_key =
                            Some((done.render_serial, graph_fingerprint(&done.graph)));
                        let image = egui::ColorImage::from_rgba_unmultiplied(
                            [done.width as usize, done.height as usize],
                            &st.compositor_output_rgba,
                        );
                        if let Some(texture) = st.compositor_output_texture.as_mut() {
                            texture.set(image, egui::TextureOptions::LINEAR);
                        } else {
                            st.compositor_output_texture = Some(ctx.load_texture(
                                "compositor-output",
                                image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    } else {
                        st.compositor_eval_error = true;
                    }
                }
            }
            Err(TryRecvError::Disconnected) => {
                st.compositor_eval_rx = None;
                st.compositor_eval_busy = false;
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(30));
            }
        }
    }

    if st.compositor_render_serial == 0 {
        return;
    }
    let graph = doc.scene.compositor_graph.clone();
    let needs_eval =
        st.compositor_output_key != Some((st.compositor_render_serial, graph_fingerprint(&graph)));
    if !needs_eval || st.compositor_eval_busy {
        return;
    }
    if st
        .compositor_eval_requested_graph
        .as_ref()
        .is_none_or(|requested| graph_fingerprint(requested) != graph_fingerprint(&graph))
        || st.compositor_eval_requested_serial != st.compositor_render_serial
    {
        st.compositor_eval_error = false;
        st.compositor_eval_requested_graph = Some(graph.clone());
        st.compositor_eval_requested_serial = st.compositor_render_serial;
        st.compositor_eval_requested_at = Some(std::time::Instant::now());
    }
    if st
        .compositor_eval_requested_at
        .is_some_and(|time| time.elapsed() < Duration::from_millis(120))
    {
        ctx.request_repaint_after(Duration::from_millis(120));
        return;
    }

    let graph = graph.clone();
    let source = st.compositor_render_rgba.clone();
    let width = st.compositor_render_width;
    let height = st.compositor_render_height;
    let render_serial = st.compositor_render_serial;
    let (tx, rx) = mpsc::channel();
    st.compositor_eval_rx = Some(rx);
    st.compositor_eval_busy = true;
    std::thread::spawn(move || {
        let rgba = graph.evaluate(width, height, &source);
        let _ = tx.send(CompositorEvalCompletion {
            graph,
            render_serial,
            width,
            height,
            rgba,
        });
    });
    ctx.request_repaint_after(Duration::from_millis(30));
}

fn graph_fingerprint(graph: &CompositorGraph) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for node in &graph.nodes {
        node.id.hash(&mut hasher);
        (node.kind as u8).hash(&mut hasher);
        node.value.to_bits().hash(&mut hasher);
        node.secondary.to_bits().hash(&mut hasher);
    }
    for link in &graph.links {
        link.from_node.hash(&mut hasher);
        link.to_node.hash(&mut hasher);
    }
    hasher.finish()
}

fn draw_compositing_editor(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    doc: &mut Document,
    st: &mut UiState,
    _actions: &mut Vec<Action>,
) {
    let before = doc.scene.compositor_graph.clone();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Compositor").strong());
        ui.separator();
        ui.menu_button("Add Node", |ui| {
            for kind in [
                CompositorNodeKind::BrightnessContrast,
                CompositorNodeKind::Exposure,
                CompositorNodeKind::Blur,
                CompositorNodeKind::Glare,
                CompositorNodeKind::Invert,
            ] {
                if ui.button(kind.title()).clicked() {
                    let graph = &mut doc.scene.compositor_graph;
                    let pos = [220.0, graph.nodes.len() as f32 * 32.0];
                    let id = graph.add_node(kind, pos);
                    insert_compositor_effect(graph, id);
                    ui.close_menu();
                }
            }
        });
        ui.separator();
        ui.weak("Click output then input to connect · drag empty space to pan · scroll to zoom · right-click a wire to remove it");
    });

    let (canvas, canvas_response) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas);
    painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(0x1b, 0x1b, 0x1b));
    if st.compositor_zoom < 0.2 {
        st.compositor_zoom = 1.0;
    }
    if canvas_response.hovered() {
        let scroll = ctx.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let local = ctx
                .input(|i| i.pointer.hover_pos())
                .unwrap_or(canvas.center())
                - canvas.min;
            let before_zoom = st.compositor_zoom;
            let after_zoom = (before_zoom * (scroll * 0.001).exp()).clamp(0.25, 2.5);
            let graph_x = (local.x - st.compositor_pan[0]) / before_zoom;
            let graph_y = (local.y - st.compositor_pan[1]) / before_zoom;
            st.compositor_zoom = after_zoom;
            st.compositor_pan = [
                local.x - graph_x * after_zoom,
                local.y - graph_y * after_zoom,
            ];
        }
    }

    let zoom = st.compositor_zoom.clamp(0.25, 2.5);
    let origin = canvas.min + egui::vec2(st.compositor_pan[0], st.compositor_pan[1]);
    let rect_for = |node: &forge_core::CompositorNode| {
        let size = egui::vec2(230.0, compositor_node_height(node.kind)) * zoom;
        egui::Rect::from_min_size(origin + egui::vec2(node.pos[0], node.pos[1]) * zoom, size)
    };

    st.compositor_last_node_rects = doc
        .scene
        .compositor_graph
        .nodes
        .iter()
        .map(|node| (node.id, rect_for(node)))
        .collect();
    let socket_y = |rect: egui::Rect| rect.top() + 42.0 * zoom;
    let node_rects = st.compositor_last_node_rects.clone();
    let wire_painter = painter.with_clip_rect(canvas);
    let mut wire_to_remove = None;
    let pointer = ctx.input(|i| i.pointer.interact_pos());
    for (index, link) in doc.scene.compositor_graph.links.iter().enumerate() {
        let Some((_, from)) = node_rects.iter().find(|(id, _)| *id == link.from_node) else {
            continue;
        };
        let Some((_, to)) = node_rects.iter().find(|(id, _)| *id == link.to_node) else {
            continue;
        };
        let a = egui::pos2(from.right(), socket_y(*from));
        let b = egui::pos2(to.left(), socket_y(*to));
        let points = shader_wire_points(a, b);
        let hovered = canvas.contains(pointer.unwrap_or_default())
            && pointer.is_some_and(|pos| shader_wire_hit(pos, points, 7.0));
        wire_painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
            points,
            false,
            egui::Color32::TRANSPARENT,
            egui::Stroke::new(
                if hovered { 3.5 } else { 2.5_f32 },
                if hovered {
                    egui::Color32::LIGHT_BLUE
                } else {
                    egui::Color32::from_rgb(0xc0, 0xb0, 0x55)
                },
            ),
        ));
        if hovered && ctx.input(|i| i.pointer.secondary_clicked()) {
            wire_to_remove = Some(index);
        }
    }
    if let (Some(from_id), Some(pointer)) = (st.compositor_link_from, pointer) {
        if let Some((_, from)) = node_rects.iter().find(|(id, _)| *id == from_id) {
            let a = egui::pos2(from.right(), socket_y(*from));
            let points = shader_wire_points(a, pointer);
            wire_painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                points,
                false,
                egui::Color32::TRANSPARENT,
                egui::Stroke::new(2.0_f32, egui::Color32::LIGHT_BLUE),
            ));
        }
    }
    if let Some(index) = wire_to_remove {
        doc.scene.compositor_graph.links.remove(index);
    }

    let mut node_rects = Vec::new();
    let node_count = doc.scene.compositor_graph.nodes.len();
    for index in 0..node_count {
        let kind = doc.scene.compositor_graph.nodes[index].kind;
        let id = doc.scene.compositor_graph.nodes[index].id;
        let rect = rect_for(&doc.scene.compositor_graph.nodes[index]);
        node_rects.push((id, rect));
        let title_rect =
            egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), rect.top() + 30.0 * zoom));
        let body_rect = egui::Rect::from_min_max(title_rect.max, rect.max);
        painter.rect_filled(rect, 5.0, egui::Color32::from_rgb(0x33, 0x33, 0x33));
        painter.rect_filled(title_rect, 5.0, egui::Color32::from_rgb(0x50, 0x47, 0x29));
        painter.text(
            title_rect.center() - egui::vec2(8.0, 0.0),
            egui::Align2::CENTER_CENTER,
            kind.title(),
            egui::FontId::proportional((13.0 * zoom).max(9.0)),
            egui::Color32::WHITE,
        );
        let header_response = ui.interact(
            title_rect,
            egui::Id::new(("compositor_node_header", id)),
            egui::Sense::click_and_drag(),
        );
        if header_response.dragged() {
            let delta = ctx.input(|i| i.pointer.delta()) / zoom;
            doc.scene.compositor_graph.nodes[index].pos[0] += delta.x;
            doc.scene.compositor_graph.nodes[index].pos[1] += delta.y;
        }
        if matches!(
            kind,
            CompositorNodeKind::BrightnessContrast
                | CompositorNodeKind::Exposure
                | CompositorNodeKind::Blur
                | CompositorNodeKind::Glare
                | CompositorNodeKind::Invert
        ) {
            let delete_rect = egui::Rect::from_center_size(
                egui::pos2(title_rect.right() - 12.0 * zoom, title_rect.center().y),
                egui::vec2(20.0, 20.0) * zoom,
            );
            painter.text(
                delete_rect.center(),
                egui::Align2::CENTER_CENTER,
                "×",
                egui::FontId::proportional((15.0 * zoom).max(10.0)),
                egui::Color32::from_gray(200),
            );
            if ui
                .interact(
                    delete_rect,
                    egui::Id::new(("compositor_delete", id)),
                    egui::Sense::click(),
                )
                .on_hover_text("Delete node")
                .clicked()
            {
                doc.scene.compositor_graph.remove_node(id);
                break;
            }
        }
        let mut input_pos = None;
        if kind.has_image_input() {
            let center = egui::pos2(rect.left(), socket_y(rect));
            let socket = egui::Rect::from_center_size(center, egui::vec2(16.0, 16.0));
            painter.circle_filled(center, 5.0, egui::Color32::from_rgb(0x9a, 0xc8, 0x70));
            let response = ui.interact(
                socket,
                egui::Id::new(("compositor_input", id)),
                egui::Sense::click(),
            );
            if response.clicked()
                || (response.hovered()
                    && ctx.input(|i| i.pointer.any_released())
                    && st.compositor_link_from.is_some())
            {
                if let Some(from) = st.compositor_link_from.take() {
                    if doc.scene.compositor_graph.connect(from, id) {
                        doc.touch();
                    }
                }
            }
            input_pos = Some(center);
        }
        if !kind.is_output() {
            let center = egui::pos2(rect.right(), socket_y(rect));
            let socket = egui::Rect::from_center_size(center, egui::vec2(16.0, 16.0));
            painter.circle_filled(center, 5.0, egui::Color32::from_rgb(0x9a, 0xc8, 0x70));
            let response = ui.interact(
                socket,
                egui::Id::new(("compositor_output", id)),
                egui::Sense::click_and_drag(),
            );
            if response.clicked() || response.drag_started() {
                st.compositor_link_from = Some(id);
            }
        }
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(body_rect.shrink(8.0)), |ui| {
            ui.set_clip_rect(canvas);
            if zoom < 0.65 {
                ui.weak("Zoom in to edit node values");
            } else {
                match kind {
                    CompositorNodeKind::RenderLayers => {
                        ui.weak("Current rendered frame");
                    }
                    CompositorNodeKind::BrightnessContrast => {
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].value,
                                -100.0..=100.0,
                            )
                            .text("Brightness"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].secondary,
                                -100.0..=100.0,
                            )
                            .text("Contrast"),
                        );
                    }
                    CompositorNodeKind::Exposure => {
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].value,
                                -8.0..=8.0,
                            )
                            .text("Stops"),
                        );
                    }
                    CompositorNodeKind::Blur => {
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].value,
                                0.0..=32.0,
                            )
                            .text("Radius"),
                        );
                    }
                    CompositorNodeKind::Glare => {
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].value,
                                0.0..=1.0,
                            )
                            .text("Threshold"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut doc.scene.compositor_graph.nodes[index].secondary,
                                0.0..=4.0,
                            )
                            .text("Strength"),
                        );
                    }
                    CompositorNodeKind::Invert => {
                        ui.weak("Invert RGB; preserve alpha");
                    }
                    CompositorNodeKind::Composite | CompositorNodeKind::Viewer => {
                        ui.weak(if input_pos.is_some() {
                            "Final image output"
                        } else {
                            "Connect an image input"
                        });
                    }
                }
            }
        });
    }

    if canvas_response.hovered()
        && ctx.input(|i| i.pointer.button_down(egui::PointerButton::Middle))
    {
        let delta = ctx.input(|i| i.pointer.delta());
        st.compositor_pan[0] += delta.x;
        st.compositor_pan[1] += delta.y;
    } else if canvas_response.dragged()
        && !st
            .compositor_last_node_rects
            .iter()
            .any(|(_, rect)| rect.contains(pointer.unwrap_or_default()))
    {
        let delta = ctx.input(|i| i.pointer.delta());
        st.compositor_pan[0] += delta.x;
        st.compositor_pan[1] += delta.y;
    }
    commit_compositor_graph(doc, before);
}

fn compositor_node_height(kind: CompositorNodeKind) -> f32 {
    match kind {
        CompositorNodeKind::BrightnessContrast | CompositorNodeKind::Glare => 130.0,
        CompositorNodeKind::Exposure | CompositorNodeKind::Blur => 100.0,
        _ => 75.0,
    }
}

fn insert_compositor_effect(graph: &mut CompositorGraph, id: u64) {
    let Some(source) = graph
        .nodes
        .iter()
        .find(|node| node.kind == CompositorNodeKind::RenderLayers)
        .map(|node| node.id)
    else {
        return;
    };
    let outgoing = graph
        .links
        .iter()
        .filter(|link| link.from_node == source)
        .map(|link| link.to_node)
        .collect::<Vec<_>>();
    if outgoing.is_empty() {
        return;
    }
    graph.links.retain(|link| link.from_node != source);
    graph.links.push(forge_core::CompositorLink {
        from_node: source,
        to_node: id,
    });
    for to_node in outgoing {
        graph.links.push(forge_core::CompositorLink {
            from_node: id,
            to_node,
        });
    }
}

fn commit_compositor_graph(doc: &mut Document, before: CompositorGraph) {
    if before == doc.scene.compositor_graph {
        return;
    }
    let after = doc.scene.compositor_graph.clone();
    if !doc.edit_open {
        doc.scene.compositor_graph = before;
        doc.checkpoint();
        doc.edit_open = true;
        doc.scene.compositor_graph = after;
    }
    doc.touch();
}

fn draw_timeline(
    ui: &mut egui::Ui,
    doc: &mut Document,
    _st: &mut UiState,
    actions: &mut Vec<Action>,
) {
    let anim = &mut doc.scene.animation;
    // Ensure sane range
    if anim.frame_end < anim.frame_start {
        anim.frame_end = anim.frame_start;
    }
    anim.frame_current = anim.frame_current.clamp(anim.frame_start, anim.frame_end);

    ui.horizontal(|ui| {
        // Playback controls
        let play_label = if anim.playing { "⏸" } else { "▶" };
        if ui
            .button(play_label)
            .on_hover_text("Play / Pause animation")
            .clicked()
        {
            actions.push(Action::ToggleAnimationPlay);
        }
        if ui.button("⏮").on_hover_text("Jump to start").clicked() {
            anim.frame_current = anim.frame_start;
            anim.playing = false;
            actions.push(Action::AnimScrub); // force eval
        }
        if ui.button("⏭").on_hover_text("Jump to end").clicked() {
            anim.frame_current = anim.frame_end;
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }

        ui.separator();
        ui.label("Start");
        let mut fs = anim.frame_start;
        if ui.add(egui::DragValue::new(&mut fs).speed(1)).changed() {
            anim.frame_start = fs.min(anim.frame_end);
            actions.push(Action::AnimScrub);
        }
        ui.label("End");
        let mut fe = anim.frame_end;
        if ui.add(egui::DragValue::new(&mut fe).speed(1)).changed() {
            anim.frame_end = fe.max(anim.frame_start);
            actions.push(Action::AnimScrub);
        }

        ui.separator();
        ui.label("Frame");
        let mut fc = anim.frame_current;
        if ui.add(egui::DragValue::new(&mut fc).speed(1)).changed() {
            anim.frame_current = fc.clamp(anim.frame_start, anim.frame_end);
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }
        ui.label(format!("/ {}", anim.frame_end));
        ui.separator();
        ui.label("FPS");
        let mut fps = anim.fps;
        if ui
            .add(egui::DragValue::new(&mut fps).range(1..=120).speed(1))
            .changed()
        {
            anim.fps = fps.max(1);
        }
        ui.weak("playback & render");

        ui.separator();
        let auto = anim.auto_key;
        if ui
            .selectable_label(auto, "● Auto Key")
            .on_hover_text("Automatically insert keys when transforming")
            .clicked()
        {
            actions.push(Action::ToggleAutoKey);
        }
        if ui
            .button("Key LocRotScale")
            .on_hover_text("Insert location, rotation & scale key (I / K)")
            .clicked()
        {
            actions.push(Action::InsertKeyframeLocRotScale);
        }
        if ui.button("Key Loc").clicked() {
            actions.push(Action::InsertKeyframeLocation);
        }
        if ui.button("Key Rot").clicked() {
            actions.push(Action::InsertKeyframeRotation);
        }
        if ui.button("Key Scale").clicked() {
            actions.push(Action::InsertKeyframeScale);
        }
        if ui
            .button("Delete Key")
            .on_hover_text("Delete keys on current frame for selection")
            .clicked()
        {
            actions.push(Action::DeleteKeyframe);
        }
    });

    // Scrubbable timeline ruler with keyframe diamonds
    let full = ui.available_rect_before_wrap();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(full.width(), 28.0),
        egui::Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(0x1e, 0x1e, 0x1e));

    let start = anim.frame_start as f32;
    let end = anim.frame_end as f32;
    let span = (end - start).max(1.0);
    let x_of = |f: f32| rect.left() + ((f - start) / span) * rect.width();

    // Frame ticks
    let step = if span > 200.0 {
        20.0
    } else if span > 80.0 {
        10.0
    } else {
        5.0
    };
    let mut f = (start / step).floor() * step;
    while f <= end {
        let x = x_of(f);
        let major = (f as i32) % (step as i32 * 2) == 0;
        painter.line_segment(
            [
                egui::pos2(x, rect.bottom()),
                egui::pos2(x, rect.bottom() - if major { 10.0 } else { 5.0 }),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(if major { 120 } else { 70 })),
        );
        if major {
            painter.text(
                egui::pos2(x, rect.top() + 2.0),
                egui::Align2::CENTER_TOP,
                format!("{}", f as i32),
                egui::FontId::proportional(10.0),
                egui::Color32::from_gray(160),
            );
        }
        f += step;
    }

    // Keyframe markers for selected objects (or all)
    let ids: Vec<u64> = if doc.selection.is_empty() {
        doc.scene.objects.iter().map(|o| o.id).collect()
    } else {
        doc.selection.iter().copied().collect()
    };
    for id in ids {
        if let Some(oa) = anim.objects.get(&id) {
            for k in &oa.keys {
                let x = x_of(k.frame as f32);
                let col = if k.frame == anim.frame_current {
                    egui::Color32::from_rgb(0xff, 0xcc, 0x33)
                } else {
                    egui::Color32::from_rgb(0x55, 0xaa, 0xff)
                };
                let c = egui::pos2(x, rect.center().y);
                let s = 5.0;
                let diamond = [
                    egui::pos2(c.x, c.y - s),
                    egui::pos2(c.x + s, c.y),
                    egui::pos2(c.x, c.y + s),
                    egui::pos2(c.x - s, c.y),
                ];
                painter.add(egui::Shape::convex_polygon(
                    diamond.to_vec(),
                    col,
                    egui::Stroke::NONE,
                ));
            }
        }
    }

    // Playhead
    let px = x_of(anim.frame_current as f32);
    painter.line_segment(
        [egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())],
        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(0xff, 0x66, 0x44)),
    );

    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let frame = (start + t * span).round() as i32;
            anim.frame_current = frame.clamp(anim.frame_start, anim.frame_end);
            anim.playing = false;
            actions.push(Action::AnimScrub);
        }
    }
}

fn menu_bar(
    ui: &mut egui::Ui,
    doc: &mut Document,
    settings: &mut ViewSettings,
    ts: &mut ToolState,
    st: &mut UiState,
    b: &[Binding],
    recent: &[std::path::PathBuf],
    actions: &mut Vec<Action>,
) {
    let has_sel = !doc.selection.is_empty();
    let has_active = doc.active_index().is_some();
    egui::menu::bar(ui, |ui| {
        ui.menu_button("File", |ui| {
            mi(ui, actions, b, "New", Action::New, true);
            mi(ui, actions, b, "Open…", Action::Open, true);
            ui.menu_button("Open Recent", |ui| {
                if recent.is_empty() {
                    ui.add_enabled(false, egui::Button::new("(no recent files)"));
                }
                for p in recent {
                    let label = p
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| p.display().to_string());
                    let r = ui.button(label).on_hover_text(p.display().to_string());
                    if r.clicked() {
                        actions.push(Action::OpenRecent(p.clone()));
                        ui.close_menu();
                    }
                }
                ui.separator();
                mi(
                    ui,
                    actions,
                    b,
                    "Clear Recent",
                    Action::ClearRecent,
                    !recent.is_empty(),
                );
            });
            ui.separator();
            mi(ui, actions, b, "Save", Action::Save, true);
            mi(ui, actions, b, "Save As…", Action::SaveAs, true);
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Import… (glTF, GLB, OBJ, FBX, Unity package)",
                Action::Import,
                true,
            );
            if ui.button("Font Library…").clicked() {
                st.show_font_library = true;
                ui.close_menu();
            }
            ui.menu_button("Export", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "glTF Binary (.glb)",
                    Action::ExportGlb,
                    true,
                );
                mi(ui, actions, b, "Wavefront (.obj)", Action::ExportObj, true);
                mi(
                    ui,
                    actions,
                    b,
                    "Autodesk FBX (.fbx)",
                    Action::ExportFbx,
                    true,
                );
                ui.separator();
                mi(
                    ui,
                    actions,
                    b,
                    "glTF Binary, selected only",
                    Action::ExportGlbSelected,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Wavefront, selected only",
                    Action::ExportObjSelected,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Autodesk FBX, selected only",
                    Action::ExportFbxSelected,
                    has_sel,
                );
            });
            ui.separator();
            mi(ui, actions, b, "Quit", Action::Quit, true);
        });

        ui.menu_button("Edit", |ui| {
            mi(ui, actions, b, "Undo", Action::Undo, doc.can_undo());
            mi(ui, actions, b, "Redo", Action::Redo, doc.can_redo());
            ui.separator();
            mi(ui, actions, b, "Duplicate", Action::Duplicate, has_sel);
            mi(
                ui,
                actions,
                b,
                "Duplicate Linked",
                Action::DuplicateLinked,
                has_sel,
            );
            mi(ui, actions, b, "Delete", Action::Delete, has_sel);
            mi(ui, actions, b, "Rename Active", Action::Rename, has_active);
            mi(
                ui,
                actions,
                b,
                "Make Single User",
                Action::MakeSingleUser,
                has_sel,
            );
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Keyboard Shortcuts…",
                Action::ToggleShortcuts,
                true,
            );
        });

        ui.menu_button("Select", |ui| {
            mi(ui, actions, b, "All", Action::SelectAll, true);
            mi(ui, actions, b, "None", Action::SelectNone, true);
            mi(ui, actions, b, "Invert", Action::SelectInvert, true);
            mi(ui, actions, b, "Random", Action::SelectRandom, true);
            ui.separator();
            mi(ui, actions, b, "Children", Action::SelectChildren, has_sel);
            mi(ui, actions, b, "Parent", Action::SelectParent, has_sel);
            ui.menu_button("Mode", |ui| {
                let in_edit = doc.in_edit_mode();
                mi(
                    ui,
                    actions,
                    b,
                    if in_edit { "Object Mode" } else { "Edit Mode" },
                    Action::ToggleEditMode,
                    true,
                );
                mi(
                    ui,
                    actions,
                    b,
                    if doc.sculpt_mode {
                        "Exit Sculpt Mode"
                    } else {
                        "Sculpt Mode"
                    },
                    Action::ToggleSculptMode,
                    true,
                );
                mi(
                    ui,
                    actions,
                    b,
                    if doc.uv_mode {
                        "Exit UV Editing"
                    } else {
                        "UV Editing"
                    },
                    Action::ToggleUvMode,
                    true,
                );
                ui.separator();
                ui.add_enabled_ui(in_edit, |ui| {
                    mi(
                        ui,
                        actions,
                        b,
                        "Vertex (points)",
                        Action::EditDomainVertex,
                        true,
                    );
                    mi(ui, actions, b, "Edge (lines)", Action::EditDomainEdge, true);
                    mi(
                        ui,
                        actions,
                        b,
                        "Face (surfaces)",
                        Action::EditDomainFace,
                        true,
                    );
                });
                if !in_edit {
                    ui.weak("Select a mesh object, then Edit Mode.");
                }
            });
            ui.menu_button("Select Similar", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Same Type",
                    Action::SelectSameType,
                    has_active,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Same Collection",
                    Action::SelectSameCollection,
                    has_active,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Same Material",
                    Action::SelectSameMaterial,
                    has_active,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Linked Duplicates",
                    Action::SelectLinked,
                    has_active,
                );
            });
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Select Tool",
                Action::SetTool(Tool::Select),
                true,
            );
            mi(
                ui,
                actions,
                b,
                "Box Select Tool",
                Action::SetTool(Tool::BoxSelect),
                true,
            );
            mi(
                ui,
                actions,
                b,
                "Lasso Select Tool",
                Action::SetTool(Tool::Lasso),
                true,
            );
            mi(
                ui,
                actions,
                b,
                "Circle Select Tool",
                Action::SetTool(Tool::Circle),
                true,
            );
        });

        ui.menu_button("Add", |ui| {
            ui.menu_button("Mesh", |ui| {
                for p in Primitive::all() {
                    let label = p.label();
                    mi(ui, actions, b, label, Action::Add(p), true);
                }
            });
            ui.menu_button("Curve", |ui| {
                for p in Primitive::curves() {
                    mi(ui, actions, b, p.label(), Action::Add(p), true);
                }
                soon(ui, "Nurbs Curve");
            });
            mi(
                ui,
                actions,
                b,
                "Surface",
                Action::AddKind(ObjectKind::Surface(SurfaceData::default())),
                true,
            );
            mi(
                ui,
                actions,
                b,
                "Volume",
                Action::AddKind(ObjectKind::Volume(VolumeData::default())),
                true,
            );
            mi(
                ui,
                actions,
                b,
                "Grease Pencil",
                Action::AddKind(ObjectKind::GreasePencil(GreasePencilData::default())),
                true,
            );
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Armature",
                Action::AddKind(ObjectKind::Armature(ArmatureData::default())),
                true,
            );
            soon(ui, "Lattice");
            ui.separator();
            ui.menu_button("Empty", |ui| {
                for sh in EmptyShape::ALL {
                    mi(
                        ui,
                        actions,
                        b,
                        sh.name(),
                        Action::AddKind(ObjectKind::Empty {
                            shape: sh,
                            size: 1.0,
                        }),
                        true,
                    );
                }
            });
            ui.menu_button("Image", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Image as Plane…",
                    Action::AddImagePlane,
                    true,
                );
                soon(ui, "Reference image");
            });
            ui.separator();
            ui.menu_button("Light", |ui| {
                for t in [
                    LightType::Point,
                    LightType::Sun,
                    LightType::Spot,
                    LightType::Area,
                ] {
                    mi(
                        ui,
                        actions,
                        b,
                        t.name(),
                        Action::AddKind(ObjectKind::Light(LightData::new(t))),
                        true,
                    );
                }
            });
            ui.menu_button("Light Probe", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Reflection",
                    Action::AddKind(ObjectKind::LightProbe(LightProbeData {
                        kind: LightProbeType::Reflection,
                        ..Default::default()
                    })),
                    true,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Irradiance Volume",
                    Action::AddKind(ObjectKind::LightProbe(LightProbeData {
                        kind: LightProbeType::Irradiance,
                        ..Default::default()
                    })),
                    true,
                );
            });
            ui.menu_button("Camera", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Perspective",
                    Action::AddKind(ObjectKind::Camera(CameraData::default())),
                    true,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Orthographic",
                    Action::AddKind(ObjectKind::Camera(CameraData {
                        ortho: true,
                        ..CameraData::default()
                    })),
                    true,
                );
            });
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Speaker",
                Action::AddKind(ObjectKind::Speaker(SpeakerData::default())),
                true,
            );
            ui.menu_button("Force Field", |ui| {
                for t in ForceFieldType::ALL {
                    mi(
                        ui,
                        actions,
                        b,
                        t.name(),
                        Action::AddKind(ObjectKind::ForceField(ForceFieldData {
                            kind: t,
                            ..Default::default()
                        })),
                        true,
                    );
                }
            });
            soon(ui, "Collection Instance");
        });

        ui.menu_button("Object", |ui| {
            ui.menu_button("Transform", |ui| {
                mi(ui, actions, b, "Move", Action::ModalMove, has_sel);
                mi(ui, actions, b, "Rotate", Action::ModalRotate, has_sel);
                mi(ui, actions, b, "Scale", Action::ModalScale, has_sel);
                ui.separator();
                mi(
                    ui,
                    actions,
                    b,
                    "Clear Location",
                    Action::ClearLocation,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Clear Rotation",
                    Action::ClearRotation,
                    has_sel,
                );
                mi(ui, actions, b, "Clear Scale", Action::ClearScale, has_sel);
                ui.separator();
                ui.menu_button("Align Objects", |ui| {
                    for (tn, to) in [
                        ("To Active", AlignTo::Active),
                        ("To Selection", AlignTo::Selection),
                        ("To 3D Cursor", AlignTo::Cursor),
                    ] {
                        ui.menu_button(tn, |ui| {
                            for (ai, an) in ["X", "Y", "Z"].iter().enumerate() {
                                ui.menu_button(*an, |ui| {
                                    for (mn, mode) in [
                                        ("Min", AlignMode::Min),
                                        ("Center", AlignMode::Center),
                                        ("Max", AlignMode::Max),
                                    ] {
                                        mi(
                                            ui,
                                            actions,
                                            b,
                                            mn,
                                            Action::Align(ai, mode, to),
                                            has_sel,
                                        );
                                    }
                                });
                            }
                        });
                    }
                });
            });
            ui.menu_button("Apply", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "All Transforms",
                    Action::Apply {
                        loc: true,
                        rot: true,
                        scale: true,
                    },
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Location",
                    Action::Apply {
                        loc: true,
                        rot: false,
                        scale: false,
                    },
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Rotation",
                    Action::Apply {
                        loc: false,
                        rot: true,
                        scale: false,
                    },
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Scale",
                    Action::Apply {
                        loc: false,
                        rot: false,
                        scale: true,
                    },
                    has_sel,
                );
            });
            ui.menu_button("Set Origin", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Origin to Geometry",
                    Action::SetOrigin(OriginMode::Geometry),
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Origin to Center of Mass (vertex average)",
                    Action::SetOrigin(OriginMode::Median),
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Origin to 3D Cursor",
                    Action::SetOrigin(OriginMode::Cursor),
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Geometry to Origin",
                    Action::GeometryToOrigin,
                    has_sel,
                );
            });
            ui.menu_button("Snap", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Selection to Cursor",
                    Action::SnapSelToCursor,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Selection to Active",
                    Action::SnapSelToActive,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Cursor to Selected",
                    Action::SnapCursorToSel,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Cursor to Active",
                    Action::SnapCursorToActive,
                    has_active,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Cursor to World Origin",
                    Action::SnapCursorToOrigin,
                    true,
                );
            });
            ui.menu_button("Parent", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "Set (selected to active)",
                    Action::ParentSet,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "Clear and Keep Transform",
                    Action::ParentClearKeep,
                    has_sel,
                );
                mi(ui, actions, b, "Clear", Action::ParentClear, has_sel);
            });
            ui.menu_button("Collection", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "New Collection",
                    Action::NewCollection,
                    true,
                );
                ui.separator();
                ui.label("Move selection to:");
                mi(
                    ui,
                    actions,
                    b,
                    "Scene Collection",
                    Action::MoveToCollection(None),
                    has_sel,
                );
                for c in &doc.scene.collections {
                    mi(
                        ui,
                        actions,
                        b,
                        &c.name,
                        Action::MoveToCollection(Some(c.id)),
                        has_sel,
                    );
                }
            });
            ui.menu_button("Constraints", |ui| {
                for k in ConstraintKind::all_defaults() {
                    let n = k.name();
                    mi(ui, actions, b, n, Action::AddConstraint(k), has_active);
                }
                ui.separator();
                mi(
                    ui,
                    actions,
                    b,
                    "Clear Constraints",
                    Action::ClearConstraints,
                    has_active,
                );
            });
            ui.separator();
            mi(ui, actions, b, "Join", Action::Join, has_sel);
            ui.separator();
            mi(ui, actions, b, "Shade Smooth", Action::ShadeSmooth, has_sel);
            mi(ui, actions, b, "Shade Flat", Action::ShadeFlat, has_sel);
            ui.separator();
            ui.menu_button("Separate", |ui| {
                mi(
                    ui,
                    actions,
                    b,
                    "By Loose Parts",
                    Action::SeparateLoose,
                    has_sel,
                );
                mi(
                    ui,
                    actions,
                    b,
                    "By Material",
                    Action::SeparateMaterial,
                    has_sel,
                );
            });
            ui.separator();
            mi(ui, actions, b, "Hide Selected", Action::Hide, has_sel);
            mi(
                ui,
                actions,
                b,
                "Hide Unselected",
                Action::HideUnselected,
                has_sel,
            );
            mi(ui, actions, b, "Show All", Action::ShowAll, true);
        });

        ui.menu_button("View", |ui| {
            ui.menu_button("Shading", |ui| {
                for m in [
                    ShadingMode::Rendered,
                    ShadingMode::Albedo,
                    ShadingMode::Normals,
                    ShadingMode::Roughness,
                    ShadingMode::Metallic,
                    ShadingMode::UvChecker,
                ] {
                    mi(ui, actions, b, m.name(), Action::SetShading(m), true);
                }
            });
            ui.menu_button("Wireframe", |ui| {
                for m in [WireMode::Off, WireMode::Overlay, WireMode::Only] {
                    ui.radio_value(&mut settings.wireframe, m, m.name());
                }
            });
            ui.checkbox(&mut settings.show_background, "Show environment background");
            ui.checkbox(&mut settings.shadows, "Real-time shadows");
            if settings.shadows {
                ui.add(
                    egui::Slider::new(&mut settings.shadow_softness, 0.0..=4.0)
                        .text("Shadow softness"),
                );
            }
            ui.separator();
            ui.add(
                egui::Slider::new(&mut settings.exposure, 0.1..=8.0)
                    .logarithmic(true)
                    .text("Exposure"),
            );
            ui.add(
                egui::Slider::new(&mut settings.background_blur, 0.0..=1.0).text("Background blur"),
            );
            ui.add(
                egui::Slider::new(&mut settings.env_strength, 0.0..=2.0)
                    .text("Environment strength"),
            );
            ui.add(
                egui::Slider::new(&mut settings.env_rotation, -3.14..=3.14)
                    .text("Environment rotation"),
            );
            ui.separator();
            mi(ui, actions, b, "Load HDRI…", Action::LoadHdri, true);
            mi(ui, actions, b, "Reset sky", Action::ResetSky, true);
            ui.separator();
            mi(
                ui,
                actions,
                b,
                "Frame selected",
                Action::FrameSelected,
                true,
            );
            mi(ui, actions, b, "Frame all", Action::FrameAll, true);
            mi(ui, actions, b, "Camera view", Action::CameraView, true);
            ui.separator();
            mi(ui, actions, b, "Render Image…", Action::RenderImage, true);
            mi(
                ui,
                actions,
                b,
                "Render Animation…",
                Action::RenderAnimation,
                true,
            );
            ui.weak("Output settings → Properties · Output");
            ui.separator();
            mi(ui, actions, b, "Toggle snapping", Action::ToggleSnap, true);
            ui.checkbox(&mut ts.snap.enabled, "Snapping enabled");
        });

        ui.menu_button("Tools", |ui| {
            soon(ui, "Modifiers");
            if ui.button("UV Editing").clicked() {
                actions.push(Action::ToggleUvMode);
                ui.close_menu();
            }
            soon(ui, "Texture painting");
            if ui.button("Sculpt Mode").clicked() {
                actions.push(Action::ToggleSculptMode);
                ui.close_menu();
            }
            soon(ui, "Animation / rigging");
        });
        ui.menu_button("Game Dev", |ui| {
            soon(ui, "Asset validation");
            soon(ui, "Export for Unity");
            soon(ui, "Export for Godot");
            soon(ui, "Generate LODs");
            soon(ui, "Generate colliders");
            soon(ui, "Lightmap UVs");
        });
        ui.menu_button("Help", |ui| {
            mi(
                ui,
                actions,
                b,
                "Keyboard Shortcuts",
                Action::ToggleShortcuts,
                true,
            );
            if ui.button("Font Library…").clicked() {
                st.show_font_library = true;
                ui.close_menu();
            }
            if ui.button("About").clicked() {
                st.show_about = true;
                ui.close_menu();
            }
        });
    });
}

// ------------------------------------------------------------------ outliner
fn outliner(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.heading("Scene");
        if ui.button("+ Collection").clicked() {
            actions.push(Action::NewCollection);
        }
    });
    ui.add(
        egui::TextEdit::singleline(&mut st.outliner_filter)
            .hint_text("Filter by name…")
            .desired_width(f32::INFINITY),
    );
    ui.separator();
    if doc.scene.objects.is_empty() {
        ui.label("Empty. Use the Add menu or File > Import.");
    }
    egui::ScrollArea::vertical().show(ui, |ui| {
        let filter = st.outliner_filter.to_lowercase();
        if !filter.is_empty() {
            for i in 0..doc.scene.objects.len() {
                if doc.scene.objects[i].name.to_lowercase().contains(&filter) {
                    ui.horizontal(|ui| object_row(ui, doc, st, actions, i));
                }
            }
            return;
        }
        // Scene Collection (root)
        let root_label = if doc.active_collection.is_none() {
            RichText::new("Scene Collection").strong()
        } else {
            RichText::new("Scene Collection")
        };
        egui::CollapsingHeader::new(root_label)
            .default_open(true)
            .show(ui, |ui| {
                for i in 0..doc.scene.objects.len() {
                    if doc.scene.objects[i].collection.is_none() && is_tree_root(doc, i) {
                        object_tree(ui, doc, st, actions, i, None, 0);
                    }
                }
                let cols: Vec<u64> = doc.scene.collections.iter().map(|c| c.id).collect();
                for cid in cols {
                    collection_node(ui, doc, st, actions, cid);
                }
            });
    });
}

fn is_tree_root(doc: &Document, i: usize) -> bool {
    let o = &doc.scene.objects[i];
    match o.parent.and_then(|p| doc.scene.index_of(p)) {
        None => true,
        Some(pi) => doc.scene.objects[pi].collection != o.collection,
    }
}

fn collection_node(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
    cid: u64,
) {
    let Some(ci) = doc.scene.collections.iter().position(|c| c.id == cid) else {
        return;
    };
    let id = ui.make_persistent_id(("collection", cid));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let header = state.show_header(ui, |ui| {
        let mut vis = doc.scene.collections[ci].visible;
        let r = ui.checkbox(&mut vis, "").on_hover_text("Visible");
        commit(doc, &r, |d| d.scene.collections[ci].visible = vis);
        if st.renaming == Some(Rename::Collection(cid)) {
            rename_field(ui, doc, st, |d, name| d.scene.collections[ci].name = name);
        } else {
            let name = doc.scene.collections[ci].name.clone();
            let active = doc.active_collection == Some(cid);
            let resp = ui.selectable_label(
                active,
                if active {
                    RichText::new(&name).strong()
                } else {
                    RichText::new(&name)
                },
            );
            if resp.clicked() {
                doc.active_collection = Some(cid);
            }
            if resp.double_clicked() {
                st.renaming = Some(Rename::Collection(cid));
                st.rename_buf = name;
                st.rename_focus = true;
            }
            resp.context_menu(|ui| {
                if ui.button("Rename").clicked() {
                    st.renaming = Some(Rename::Collection(cid));
                    st.rename_buf = doc.scene.collections[ci].name.clone();
                    st.rename_focus = true;
                    ui.close_menu();
                }
                if ui.button("Select Objects").clicked() {
                    for i in 0..doc.scene.objects.len() {
                        if doc.scene.objects[i].collection == Some(cid) {
                            doc.select_add(i);
                        }
                    }
                    ui.close_menu();
                }
                if ui.button("Delete Collection (keep objects)").clicked() {
                    doc.delete_collection(cid);
                    ui.close_menu();
                }
            });
        }
    });
    header.body(|ui| {
        for i in 0..doc.scene.objects.len() {
            if doc.scene.objects[i].collection == Some(cid) && is_tree_root(doc, i) {
                object_tree(ui, doc, st, actions, i, Some(cid), 0);
            }
        }
    });
}

fn object_tree(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
    idx: usize,
    coll: Option<u64>,
    depth: usize,
) {
    if depth > 32 || idx >= doc.scene.objects.len() {
        return;
    }
    let id = doc.scene.objects[idx].id;
    let kids: Vec<usize> = doc
        .scene
        .children_of(id)
        .into_iter()
        .filter(|&c| doc.scene.objects[c].collection == coll)
        .collect();
    if kids.is_empty() {
        ui.horizontal(|ui| object_row(ui, doc, st, actions, idx));
    } else {
        let cid = ui.make_persistent_id(("object", id));
        let state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), cid, true);
        let header = state.show_header(ui, |ui| object_row(ui, doc, st, actions, idx));
        header.body(|ui| {
            for k in kids {
                object_tree(ui, doc, st, actions, k, coll, depth + 1);
            }
        });
    }
}

fn rename_field(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    apply: impl FnOnce(&mut Document, String),
) {
    let te = ui.add(egui::TextEdit::singleline(&mut st.rename_buf).desired_width(120.0));
    if st.rename_focus {
        te.request_focus();
        st.rename_focus = false;
    }
    if te.lost_focus() {
        let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
        if !esc && !st.rename_buf.trim().is_empty() {
            doc.checkpoint();
            apply(doc, st.rename_buf.trim().to_string());
        }
        st.renaming = None;
    }
}

/// Full Object Mode context menu shared by Outliner rows and the 3D viewport.
fn object_context_menu(ui: &mut egui::Ui, doc: &mut Document, actions: &mut Vec<Action>) {
    let mut push = |ui: &mut egui::Ui, label: &str, a: Action| {
        if ui.button(label).clicked() {
            actions.push(a);
            ui.close_menu();
        }
    };
    push(ui, "Rename", Action::Rename);
    push(ui, "Duplicate", Action::Duplicate);
    push(ui, "Duplicate Linked", Action::DuplicateLinked);
    push(ui, "Delete", Action::Delete);
    push(ui, "Make Single User", Action::MakeSingleUser);
    ui.separator();
    push(ui, "Select Children", Action::SelectChildren);
    push(ui, "Select Parent", Action::SelectParent);
    ui.separator();
    ui.menu_button("Transform", |ui| {
        push(ui, "Move", Action::ModalMove);
        push(ui, "Rotate", Action::ModalRotate);
        push(ui, "Scale", Action::ModalScale);
        ui.separator();
        push(ui, "Clear Location", Action::ClearLocation);
        push(ui, "Clear Rotation", Action::ClearRotation);
        push(ui, "Clear Scale", Action::ClearScale);
        ui.separator();
        ui.menu_button("Align Objects", |ui| {
            for (axis, axis_name) in ["X", "Y", "Z"].iter().enumerate() {
                ui.menu_button(*axis_name, |ui| {
                    for (label, mode) in [
                        ("Min", AlignMode::Min),
                        ("Center", AlignMode::Center),
                        ("Max", AlignMode::Max),
                    ] {
                        for (target, to) in [
                            ("To Active", AlignTo::Active),
                            ("To Selection", AlignTo::Selection),
                            ("To 3D Cursor", AlignTo::Cursor),
                        ] {
                            push(
                                ui,
                                &format!("{label} — {target}"),
                                Action::Align(axis, mode, to),
                            );
                        }
                    }
                });
            }
        });
    });
    ui.menu_button("Apply", |ui| {
        push(
            ui,
            "All Transforms",
            Action::Apply {
                loc: true,
                rot: true,
                scale: true,
            },
        );
        push(
            ui,
            "Location",
            Action::Apply {
                loc: true,
                rot: false,
                scale: false,
            },
        );
        push(
            ui,
            "Rotation",
            Action::Apply {
                loc: false,
                rot: true,
                scale: false,
            },
        );
        push(
            ui,
            "Scale",
            Action::Apply {
                loc: false,
                rot: false,
                scale: true,
            },
        );
    });
    ui.menu_button("Set Origin", |ui| {
        push(
            ui,
            "Origin to Geometry",
            Action::SetOrigin(OriginMode::Geometry),
        );
        push(
            ui,
            "Origin to Center of Mass",
            Action::SetOrigin(OriginMode::Median),
        );
        push(
            ui,
            "Origin to 3D Cursor",
            Action::SetOrigin(OriginMode::Cursor),
        );
        push(ui, "Geometry to Origin", Action::GeometryToOrigin);
    });
    ui.menu_button("Snap", |ui| {
        push(ui, "Selection to Cursor", Action::SnapSelToCursor);
        push(ui, "Selection to Active", Action::SnapSelToActive);
        push(ui, "Cursor to Selected", Action::SnapCursorToSel);
        push(ui, "Cursor to Active", Action::SnapCursorToActive);
        push(ui, "Cursor to World Origin", Action::SnapCursorToOrigin);
    });
    ui.separator();
    ui.menu_button("Parent", |ui| {
        push(ui, "Set Parent (to Active)", Action::ParentSet);
        push(ui, "Clear Parent (keep transform)", Action::ParentClearKeep);
        push(ui, "Clear Parent", Action::ParentClear);
    });
    ui.menu_button("Collection", |ui| {
        push(ui, "New Collection", Action::NewCollection);
        ui.separator();
        push(ui, "Move to Collection…", Action::OpenMoveToCollection);
        push(
            ui,
            "Move to Scene Collection",
            Action::MoveToCollection(None),
        );
        for c in &doc.scene.collections {
            push(
                ui,
                &format!("Move to {}", c.name),
                Action::MoveToCollection(Some(c.id)),
            );
        }
    });
    ui.menu_button("Constraints", |ui| {
        for kind in ConstraintKind::all_defaults() {
            push(ui, kind.name(), Action::AddConstraint(kind));
        }
        ui.separator();
        push(ui, "Clear Constraints", Action::ClearConstraints);
    });
    ui.separator();
    push(ui, "Join", Action::Join);
    ui.menu_button("Separate", |ui| {
        push(ui, "By Loose Parts", Action::SeparateLoose);
        push(ui, "By Material", Action::SeparateMaterial);
    });
    ui.separator();
    push(ui, "Shade Smooth", Action::ShadeSmooth);
    push(ui, "Shade Flat", Action::ShadeFlat);
    ui.separator();
    push(ui, "Hide Selected", Action::Hide);
    push(ui, "Hide Unselected", Action::HideUnselected);
    push(ui, "Show All", Action::ShowAll);
    ui.separator();
    push(ui, "Select Same Type", Action::SelectSameType);
    push(ui, "Select Same Collection", Action::SelectSameCollection);
    push(ui, "Select Same Material", Action::SelectSameMaterial);
    push(ui, "Select Linked", Action::SelectLinked);
}

fn object_row(
    ui: &mut egui::Ui,
    doc: &mut Document,
    st: &mut UiState,
    actions: &mut Vec<Action>,
    idx: usize,
) {
    let o = &doc.scene.objects[idx];
    let id = o.id;
    let name = o.name.clone();
    let mut vis = o.visible;
    let tag = kind_tag(&o.kind);
    let r = ui.checkbox(&mut vis, "").on_hover_text("Visible");
    commit(doc, &r, |d| d.scene.objects[idx].visible = vis);
    ui.weak(tag);
    if st.renaming == Some(Rename::Object(id)) {
        rename_field(ui, doc, st, |d, n| d.scene.objects[idx].name = n);
        return;
    }
    let sel = doc.is_selected(idx);
    let active = doc.active == Some(id);
    let resp = ui.selectable_label(
        sel,
        if active && sel {
            RichText::new(&name).strong()
        } else {
            RichText::new(&name)
        },
    );
    if resp.clicked() {
        let m = ui.input(|i| i.modifiers);
        if m.shift || m.ctrl || m.command {
            doc.select_toggle(idx);
        } else {
            doc.select_only(idx);
        }
    }
    if resp.double_clicked() {
        st.renaming = Some(Rename::Object(id));
        st.rename_buf = name;
        st.rename_focus = true;
    }
    if resp.secondary_clicked() && !sel {
        doc.select_only(idx);
    }
    resp.context_menu(|ui| object_context_menu(ui, doc, actions));
}

fn move_popup(ctx: &egui::Context, doc: &Document, st: &mut UiState, actions: &mut Vec<Action>) {
    if !st.show_move_popup {
        return;
    }
    let mut open = true;
    let mut close = false;
    egui::Window::new("Move to Collection")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            if ui.button("Scene Collection").clicked() {
                actions.push(Action::MoveToCollection(None));
                close = true;
            }
            for c in &doc.scene.collections {
                if ui.button(&c.name).clicked() {
                    actions.push(Action::MoveToCollection(Some(c.id)));
                    close = true;
                }
            }
            ui.separator();
            if ui.button("+ New Collection").clicked() {
                actions.push(Action::NewCollection);
                close = true;
            }
        });
    st.show_move_popup = open && !close;
}

// ------------------------------------------------------------------ shortcuts window

fn font_library_window(ctx: &egui::Context, st: &mut UiState) {
    if !st.show_font_library {
        return;
    }
    let mut open = st.show_font_library;
    egui::Window::new("Font Library")
        .open(&mut open)
        .default_width(520.0)
        .default_height(480.0)
        .show(ctx, |ui| {
            ui.label("Browse free fonts (download once, then offline) and fonts already on this computer.");
            ui.add(
                egui::TextEdit::singleline(&mut st.font_filter)
                    .hint_text("Filter fonts…")
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Import .ttf / .otf…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Fonts", &["ttf", "otf"])
                        .pick_file()
                    {
                        match crate::fonts_io::import_font_file(&path) {
                            Ok(name) => st.font_status = format!("Imported {name}"),
                            Err(e) => st.font_status = format!("Import failed: {e}"),
                        }
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Rescan system fonts").clicked() {
                    crate::fonts_io::load_cached_fonts();
                    st.font_status = "System fonts rescanned".into();
                }
            });
            if !st.font_status.is_empty() {
                ui.weak(&st.font_status);
            }
            ui.separator();
            ui.heading("Font catalog (offline reference)");
            let filter = st.font_filter.to_ascii_lowercase();
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                egui::Grid::new("catalog_fonts").num_columns(4).striped(true).show(ui, |ui| {
                    ui.strong("Name");
                    ui.strong("Family");
                    ui.strong("License");
                    ui.strong("");
                    ui.end_row();
                    for f in free_font_catalog() {
                        if !filter.is_empty()
                            && !f.name.to_ascii_lowercase().contains(&filter)
                            && !f.family.to_ascii_lowercase().contains(&filter)
                        {
                            continue;
                        }
                        let installed = crate::fonts_io::is_catalog_installed(f.name);
                        ui.label(f.name);
                        ui.label(f.family);
                        ui.label(f.license);
                        if installed {
                            ui.label("✓ Installed");
                        } else {
                            ui.label("Import a local font file to use");
                        }
                        ui.end_row();
                    }
                });
            });
            ui.separator();
            ui.heading("Installed (ready to use on Text objects)");
            let names = font_registry()
                .lock()
                .map(|r| r.names())
                .unwrap_or_default();
            egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                for n in names {
                    if !filter.is_empty() && !n.to_ascii_lowercase().contains(&filter) {
                        continue;
                    }
                    ui.label(n);
                }
            });
        });
    st.show_font_library = open;
}

fn shortcuts_window(ctx: &egui::Context, st: &mut UiState, bindings: &[Binding]) {
    let mut open = st.show_shortcuts;
    egui::Window::new("Keyboard Shortcuts")
        .open(&mut open)
        .default_size([640.0, 640.0])
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.add(
                    egui::TextEdit::singleline(&mut st.shortcut_filter)
                        .hint_text("e.g. move, Ctrl+S, collection")
                        .desired_width(340.0),
                );
                if ui.button("Clear").clicked() {
                    st.shortcut_filter.clear();
                }
            });
            ui.weak("Press F1 to open or close this window. These are the live key bindings.");
            ui.separator();
            let f = st.shortcut_filter.to_lowercase();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut cats: Vec<&str> = Vec::new();
                for b in bindings {
                    if !cats.contains(&b.category) {
                        cats.push(b.category);
                    }
                }
                for cat in cats {
                    let rows: Vec<(String, &str)> = bindings
                        .iter()
                        .filter(|b| b.category == cat)
                        .map(|b| (keymap::describe(b), b.label))
                        .filter(|(k, l)| {
                            f.is_empty()
                                || k.to_lowercase().contains(&f)
                                || l.to_lowercase().contains(&f)
                                || cat.to_lowercase().contains(&f)
                        })
                        .collect();
                    if rows.is_empty() {
                        continue;
                    }
                    egui::CollapsingHeader::new(RichText::new(cat).strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::Grid::new(("sc", cat))
                                .striped(true)
                                .num_columns(2)
                                .show(ui, |ui| {
                                    for (k, l) in rows {
                                        ui.label(RichText::new(k).monospace().strong());
                                        ui.label(l);
                                        ui.end_row();
                                    }
                                });
                        });
                }
                let mut ecats: Vec<&str> = Vec::new();
                for (c, _, _) in keymap::EXTRA {
                    if !ecats.contains(c) {
                        ecats.push(*c);
                    }
                }
                for cat in ecats {
                    let rows: Vec<(&str, &str)> = keymap::EXTRA
                        .iter()
                        .filter(|(c, _, _)| *c == cat)
                        .map(|(_, k, l)| (*k, *l))
                        .filter(|(k, l)| {
                            f.is_empty()
                                || k.to_lowercase().contains(&f)
                                || l.to_lowercase().contains(&f)
                                || cat.to_lowercase().contains(&f)
                        })
                        .collect();
                    if rows.is_empty() {
                        continue;
                    }
                    egui::CollapsingHeader::new(RichText::new(cat).strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::Grid::new(("scx", cat))
                                .striped(true)
                                .num_columns(2)
                                .show(ui, |ui| {
                                    for (k, l) in rows {
                                        ui.label(RichText::new(k).monospace().strong());
                                        ui.label(l);
                                        ui.end_row();
                                    }
                                });
                        });
                }
            });
        });
    st.show_shortcuts = open;
}

// ------------------------------------------------------------------ properties
fn target_combo(ui: &mut egui::Ui, t: &mut Option<u64>, names: &[(u64, String)]) -> bool {
    let mut ch = false;
    let cur = t
        .and_then(|id| names.iter().find(|(i, _)| *i == id))
        .map_or("None".to_string(), |(_, n)| n.clone());
    egui::ComboBox::from_label("Target")
        .selected_text(cur)
        .show_ui(ui, |ui| {
            if ui.selectable_label(t.is_none(), "None").clicked() {
                *t = None;
                ch = true;
            }
            for (id, n) in names {
                if ui.selectable_label(*t == Some(*id), n).clicked() {
                    *t = Some(*id);
                    ch = true;
                }
            }
        });
    ch
}

fn limit_rows(
    ui: &mut egui::Ui,
    label: &str,
    min: &mut [f32; 3],
    max: &mut [f32; 3],
    enable: &mut [bool; 3],
    speed: f64,
) -> bool {
    let mut ch = false;
    for (k, ax) in ["X", "Y", "Z"].iter().enumerate() {
        ui.horizontal(|ui| {
            ch |= ui
                .checkbox(&mut enable[k], format!("{label} {ax}"))
                .changed();
            ch |= ui
                .add(
                    egui::DragValue::new(&mut min[k])
                        .speed(speed)
                        .prefix("min "),
                )
                .changed();
            ch |= ui
                .add(
                    egui::DragValue::new(&mut max[k])
                        .speed(speed)
                        .prefix("max "),
                )
                .changed();
        });
    }
    ch
}

/// Edits one constraint in place; returns true if anything changed.
fn constraint_body(ui: &mut egui::Ui, c: &mut Constraint, names: &[(u64, String)]) -> bool {
    let mut ch = false;
    ch |= ui.checkbox(&mut c.enabled, "Enabled").changed();
    ch |= ui
        .add(egui::Slider::new(&mut c.influence, 0.0..=1.0).text("Influence"))
        .changed();
    if let Some(t) = c.kind.target_mut() {
        ch |= target_combo(ui, t, names);
    }
    match &mut c.kind {
        ConstraintKind::CopyLocation { axes, .. } => {
            ui.horizontal(|ui| {
                for (k, l) in ["X", "Y", "Z"].iter().enumerate() {
                    ch |= ui.checkbox(&mut axes[k], *l).changed();
                }
            });
        }
        ConstraintKind::LimitLocation { min, max, enable } => {
            ch |= limit_rows(ui, "Loc", min, max, enable, 0.05)
        }
        ConstraintKind::LimitRotation {
            min_deg,
            max_deg,
            enable,
        } => ch |= limit_rows(ui, "Rot", min_deg, max_deg, enable, 0.5),
        ConstraintKind::LimitScale { min, max, enable } => {
            ch |= limit_rows(ui, "Scale", min, max, enable, 0.01)
        }
        ConstraintKind::LimitDistance { distance, .. } => {
            ch |= ui
                .add(
                    egui::DragValue::new(distance)
                        .speed(0.05)
                        .range(0.0..=10000.0)
                        .prefix("Distance "),
                )
                .changed();
        }
        ConstraintKind::Floor { height } => {
            ch |= ui
                .add(egui::DragValue::new(height).speed(0.05).prefix("Height "))
                .changed();
        }
        ConstraintKind::TrackTo { .. } => {
            ui.weak("Local -Z points at the target (Y up).");
        }
        _ => {}
    }
    ch
}

fn material_ui(ui: &mut egui::Ui, doc: &mut Document, i: usize) {
    let cur = doc.scene.objects[i]
        .material
        .filter(|&m| m < doc.scene.materials.len());
    let mut pick: Option<Option<usize>> = None;
    egui::ComboBox::from_label("Slot")
        .selected_text(cur.map_or("None (default)".to_string(), |m| {
            doc.scene.materials[m].name.clone()
        }))
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(cur.is_none(), "None (default)")
                .clicked()
            {
                pick = Some(None);
            }
            for (mi, m) in doc.scene.materials.iter().enumerate() {
                if ui.selectable_label(cur == Some(mi), &m.name).clicked() {
                    pick = Some(Some(mi));
                }
            }
        });
    if let Some(newm) = pick {
        doc.checkpoint();
        doc.scene.objects[i].material = newm;
    }
    if ui.button("New material").clicked() {
        doc.checkpoint();
        let mut m = Material::default();
        m.name = format!("Material.{:03}", doc.scene.materials.len() + 1);
        doc.scene.materials.push(m);
        doc.scene.objects[i].material = Some(doc.scene.materials.len() - 1);
    }

    let Some(mi) = cur else { return };
    let m = doc.scene.materials[mi].clone();
    let mut mname = m.name.clone();
    let r = ui.text_edit_singleline(&mut mname);
    commit(doc, &r, |d| d.scene.materials[mi].name = mname.clone());

    ui.horizontal(|ui| {
        ui.label("Base color");
        let mut c = egui::Color32::from_rgba_unmultiplied(
            lin_to_srgb8(m.base_color[0]),
            lin_to_srgb8(m.base_color[1]),
            lin_to_srgb8(m.base_color[2]),
            (m.base_color[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        );
        let r = ui.color_edit_button_srgba(&mut c);
        commit(doc, &r, |d| {
            let a = c.a() as f32 / 255.0;
            d.scene.materials[mi].base_color = [
                srgb8_to_lin(c.r()),
                srgb8_to_lin(c.g()),
                srgb8_to_lin(c.b()),
                a,
            ];
        });
    });
    let mut metallic = m.metallic;
    let r = ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0).text("Metallic"));
    commit(doc, &r, |d| d.scene.materials[mi].metallic = metallic);
    let mut rough = m.roughness;
    let r = ui.add(egui::Slider::new(&mut rough, 0.0..=1.0).text("Roughness"));
    commit(doc, &r, |d| d.scene.materials[mi].roughness = rough);
    ui.horizontal(|ui| {
        ui.label("Emission");
        let mut c = [
            lin_to_srgb8(m.emissive[0]),
            lin_to_srgb8(m.emissive[1]),
            lin_to_srgb8(m.emissive[2]),
        ];
        let r = ui.color_edit_button_srgb(&mut c);
        commit(doc, &r, |d| {
            d.scene.materials[mi].emissive =
                [srgb8_to_lin(c[0]), srgb8_to_lin(c[1]), srgb8_to_lin(c[2])]
        });
    });
    let mut mode = m.alpha_mode;
    let before = mode;
    egui::ComboBox::from_label("Alpha")
        .selected_text(format!("{mode:?}"))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut mode, AlphaMode::Opaque, "Opaque");
            ui.selectable_value(&mut mode, AlphaMode::Mask, "Mask");
            ui.selectable_value(&mut mode, AlphaMode::Blend, "Blend");
        });
    if mode != before {
        doc.checkpoint();
        doc.scene.materials[mi].alpha_mode = mode;
    }
    let tex = [
        ("base", m.base_color_tex),
        ("normal", m.normal_tex),
        ("metal/rough", m.metallic_roughness_tex),
        ("occlusion", m.occlusion_tex),
        ("emissive", m.emissive_tex),
    ];
    let used: Vec<&str> = tex
        .iter()
        .filter(|(_, t)| t.is_some())
        .map(|(n, _)| *n)
        .collect();
    if !used.is_empty() {
        ui.weak(format!("Textures: {}", used.join(", ")));
    }
}

fn effective_render_size(st: &UiState) -> (u32, u32) {
    let pct = st.render_percent.clamp(1, 200) as f32 / 100.0;
    let w = ((st.render_width.max(64) as f32) * pct).round() as u32;
    let h = ((st.render_height.max(64) as f32) * pct).round() as u32;
    (w.clamp(64, 7680), h.clamp(64, 4320))
}

fn draw_render_output_props(
    ui: &mut egui::Ui,
    st: &mut UiState,
    anim: &mut AnimationData,
    actions: &mut Vec<Action>,
) {
    if st.render_width < 64 {
        st.render_width = 1920;
    }
    if st.render_height < 64 {
        st.render_height = 1080;
    }
    if st.render_percent == 0 {
        st.render_percent = 100;
    }
    if st.render_anim_frames == 0 {
        st.render_anim_frames = 36;
    }
    if st.render_fps == 0 {
        st.render_fps = 24;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        if st.properties_tab == PropertiesTab::Render {
            ui.label(RichText::new("Render Engine").strong());
            ui.horizontal(|ui| {
                ui.selectable_label(true, "Workbench");
                ui.add_enabled(false, egui::Button::new("Cycles"))
                    .on_hover_text("Not available yet");
                ui.add_enabled(false, egui::Button::new("EEVEE"))
                    .on_hover_text("Not available yet");
            });
            ui.separator();
            ui.label(RichText::new("Sampling").strong());
            ui.weak("Realtime viewport path · offline capture uses 1 sample");
            ui.separator();
            ui.label(RichText::new("Film").strong());
            ui.checkbox(
                &mut st.render_use_viewport,
                "Match viewport size (ignore resolution below)",
            );
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button("Render Image")
                    .on_hover_text("Render still at Output resolution")
                    .clicked()
                {
                    actions.push(Action::RenderImage);
                }
                if ui
                    .button("Render Animation")
                    .on_hover_text("Turntable frame sequence")
                    .clicked()
                {
                    actions.push(Action::RenderAnimation);
                }
            });
        }

        // Shared format / dimensions (Blender Output properties)
        ui.label(RichText::new("Format").strong());
        ui.horizontal(|ui| {
            ui.label("Resolution X");
            ui.add(
                egui::DragValue::new(&mut st.render_width)
                    .range(64..=7680)
                    .speed(16),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Resolution Y");
            ui.add(
                egui::DragValue::new(&mut st.render_height)
                    .range(64..=4320)
                    .speed(16),
            );
        });
        ui.horizontal(|ui| {
            ui.label("%");
            ui.add(
                egui::DragValue::new(&mut st.render_percent)
                    .range(1..=200)
                    .suffix("%"),
            );
            ui.weak("scale");
        });
        let (ew, eh) = effective_render_size(st);
        ui.label(format!("Effective: {ew} × {eh} px"));

        ui.horizontal(|ui| {
            ui.label("Presets");
            for (label, w, h) in [
                ("720p", 1280u32, 720u32),
                ("1080p", 1920, 1080),
                ("1440p", 2560, 1440),
                ("4K", 3840, 2160),
            ] {
                if ui.small_button(label).clicked() {
                    st.render_width = w;
                    st.render_height = h;
                    st.render_use_viewport = false;
                    st.render_percent = 100;
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.small_button("16:9").clicked() {
                st.render_height = ((st.render_width as f32) * 9.0 / 16.0).round().max(64.0) as u32;
            }
            if ui.small_button("Square").clicked() {
                let s = st.render_width.max(st.render_height);
                st.render_width = s;
                st.render_height = s;
            }
            if ui.small_button("50%").clicked() {
                st.render_percent = 50;
            }
            if ui.small_button("100%").clicked() {
                st.render_percent = 100;
            }
        });

        ui.separator();
        ui.label(RichText::new("Frame Range (from Timeline)").strong());
        ui.horizontal(|ui| {
            ui.label("Start");
            let mut fs = anim.frame_start;
            if ui.add(egui::DragValue::new(&mut fs).speed(1)).changed() {
                anim.frame_start = fs.min(anim.frame_end);
            }
            ui.label("End");
            let mut fe = anim.frame_end;
            if ui.add(egui::DragValue::new(&mut fe).speed(1)).changed() {
                anim.frame_end = fe.max(anim.frame_start);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Frame Rate");
            let mut fps = anim.fps;
            if ui
                .add(egui::DragValue::new(&mut fps).range(1..=120).suffix(" fps"))
                .changed()
            {
                anim.fps = fps.max(1);
            }
        });
        let nframes = (anim.frame_end - anim.frame_start + 1).max(1) as f32;
        let secs = nframes / anim.fps.max(1) as f32;
        ui.weak(format!(
            "{nframes:.0} frames at {fps} fps → video ≈ {secs:.2}s",
            fps = anim.fps.max(1)
        ));
        ui.weak(format!(
            "Same FPS as Timeline · 1 frame = 1/{} s of time",
            anim.fps.max(1)
        ));
        ui.weak("Tip: use 720p or 50% for faster animation bakes.");

        ui.separator();
        ui.label(RichText::new("Output").strong());
        ui.horizontal(|ui| {
            ui.label("File Format");
            ui.selectable_label(true, "PNG");
            ui.add_enabled(false, egui::Button::new("JPEG"));
            ui.add_enabled(false, egui::Button::new("EXR"));
        });
        ui.weak("Color: RGBA 8-bit · no grid · no UI overlays");
    });
}

fn draw_material_props_summary(ui: &mut egui::Ui, doc: &mut Document, object_index: usize) {
    let Some(mi) = doc.scene.objects[object_index]
        .material
        .filter(|m| *m < doc.scene.materials.len())
    else {
        ui.weak("No material on this object.\\nAssign one in the Object panel or open the Shading workspace.");
        return;
    };
    let name = doc.scene.materials[mi].name.clone();
    ui.label(RichText::new(name).strong());
    ui.weak("Full node graph → Shading workspace");
    ui.separator();
    let mat = &doc.scene.materials[mi];
    ui.label(format!(
        "Metallic {:.2} · Roughness {:.2}",
        mat.metallic, mat.roughness
    ));
    if let Some(ti) = mat.base_color_tex {
        if let Some(tex) = doc.scene.textures.get(ti) {
            ui.label(format!(
                "Base Color Tex: {} ({}×{})",
                tex.name, tex.width, tex.height
            ));
        }
    } else {
        ui.label("Base Color Tex: None");
    }
    ui.weak("Use the Shader Editor for graph, Apply Graph, and Bake.");
}

fn properties(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, actions: &mut Vec<Action>) {
    // Blender-style context tabs
    ui.horizontal(|ui| {
        for (tab, icon, tip) in [
            (PropertiesTab::Object, "◎", "Object"),
            (PropertiesTab::Render, "📷", "Render"),
            (PropertiesTab::Output, "🖼", "Output"),
            (PropertiesTab::Material, "🎨", "Material"),
        ] {
            let selected = st.properties_tab == tab;
            let btn = ui.selectable_label(selected, icon).on_hover_text(tip);
            if btn.clicked() {
                st.properties_tab = tab;
            }
        }
        ui.separator();
        ui.label(
            RichText::new(match st.properties_tab {
                PropertiesTab::Object => "Object Properties",
                PropertiesTab::Render => "Render Properties",
                PropertiesTab::Output => "Output Properties",
                PropertiesTab::Material => "Material Properties",
            })
            .strong(),
        );
    });
    ui.separator();

    // --- Render / Output tabs do not need a selection ---
    if st.properties_tab == PropertiesTab::Render || st.properties_tab == PropertiesTab::Output {
        draw_render_output_props(ui, st, &mut doc.scene.animation, actions);
        return;
    }

    let Some(i) = doc.active_index() else {
        ui.label("Nothing selected.\nClick an object in the viewport or the outliner.");
        return;
    };
    if doc.selection.len() > 1 {
        ui.weak(format!(
            "{} objects selected (editing the active one)",
            doc.selection.len()
        ));
    }

    if st.properties_tab == PropertiesTab::Material {
        draw_material_props_summary(ui, doc, i);
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        // name
        let mut name = doc.scene.objects[i].name.clone();
        let r = ui.text_edit_singleline(&mut name);
        commit(doc, &r, |d| d.scene.objects[i].name = name.clone());

        // transform
        ui.add_space(6.0);
        ui.label(RichText::new("Transform").strong());
        let o = &doc.scene.objects[i];
        let mut loc = o.translation.to_array();
        let (rx, ry, rz) = o.rotation.to_euler(EulerRot::XYZ);
        let mut rot = [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()];
        let mut scl = o.scale.to_array();
        let row = |ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f64, suffix: &str| -> Vec<egui::Response> {
            let mut rs = Vec::new();
            ui.horizontal(|ui| {
                ui.label(format!("{label:<9}"));
                for c in v.iter_mut() {
                    rs.push(ui.add(egui::DragValue::new(c).speed(speed).suffix(suffix)));
                }
            });
            rs
        };
        for r in row(ui, "Location", &mut loc, 0.02, " m") {
            commit(doc, &r, |d| d.scene.objects[i].translation = Vec3::from(loc));
        }
        for r in row(ui, "Rotation", &mut rot, 0.5, "°") {
            commit(doc, &r, |d| d.scene.objects[i].rotation = Quat::from_euler(EulerRot::XYZ, rot[0].to_radians(), rot[1].to_radians(), rot[2].to_radians()));
        }
        for r in row(ui, "Scale", &mut scl, 0.01, "") {
            commit(doc, &r, |d| {
                let fix = |v: f32| if v.abs() < 1e-4 { 1e-4 } else { v };
                d.scene.objects[i].scale = Vec3::new(fix(scl[0]), fix(scl[1]), fix(scl[2]));
            });
        }

        // relations
        ui.add_space(6.0);
        ui.label(RichText::new("Relations").strong());
        let cur_parent = doc.scene.objects[i].parent;
        let mut new_parent: Option<Option<u64>> = None;
        egui::ComboBox::from_label("Parent")
            .selected_text(cur_parent.and_then(|p| doc.scene.index_of(p)).map_or("None".to_string(), |pi| doc.scene.objects[pi].name.clone()))
            .show_ui(ui, |ui| {
                if ui.selectable_label(cur_parent.is_none(), "None").clicked() {
                    new_parent = Some(None);
                }
                for (j, oj) in doc.scene.objects.iter().enumerate() {
                    if j == i || doc.scene.is_ancestor(i, j) {
                        continue;
                    }
                    if ui.selectable_label(cur_parent == Some(oj.id), &oj.name).clicked() {
                        new_parent = Some(Some(oj.id));
                    }
                }
            });
        if let Some(np) = new_parent {
            if np != cur_parent {
                let _ = doc.set_parent_of(i, np);
            }
        }
        let cur_col = doc.scene.objects[i].collection;
        let mut new_col: Option<Option<u64>> = None;
        egui::ComboBox::from_label("Collection")
            .selected_text(cur_col.and_then(|c| doc.scene.collections.iter().find(|x| x.id == c)).map_or("Scene Collection".to_string(), |c| c.name.clone()))
            .show_ui(ui, |ui| {
                if ui.selectable_label(cur_col.is_none(), "Scene Collection").clicked() {
                    new_col = Some(None);
                }
                for c in &doc.scene.collections {
                    if ui.selectable_label(cur_col == Some(c.id), &c.name).clicked() {
                        new_col = Some(Some(c.id));
                    }
                }
            });
        if let Some(nc) = new_col {
            if nc != cur_col {
                doc.checkpoint();
                doc.scene.objects[i].collection = nc;
            }
        }

        // constraints
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Constraints").strong());
            ui.menu_button("Add", |ui| {
                for k in ConstraintKind::all_defaults() {
                    if ui.button(k.name()).clicked() {
                        actions.push(Action::AddConstraint(k));
                        ui.close_menu();
                    }
                }
            });
        });
        let my_id = doc.scene.objects[i].id;
        let names: Vec<(u64, String)> = doc.scene.objects.iter().filter(|o| o.id != my_id).map(|o| (o.id, o.name.clone())).collect();
        let mut remove: Option<usize> = None;
        for ci in 0..doc.scene.objects[i].constraints.len() {
            let mut c = doc.scene.objects[i].constraints[ci].clone();
            let mut changed = false;
            ui.push_id(ci, |ui| {
                egui::CollapsingHeader::new(c.kind.name()).default_open(true).show(ui, |ui| {
                    changed = constraint_body(ui, &mut c, &names);
                    if ui.button("Remove").clicked() {
                        remove = Some(ci);
                    }
                });
            });
            commit_if(doc, changed, |d| d.scene.objects[i].constraints[ci] = c.clone());
        }
        if let Some(ci) = remove {
            doc.checkpoint();
            doc.scene.objects[i].constraints.remove(ci);
        }

        // primitive parameters
        if let Some(mut p) = doc.scene.objects[i].primitive.clone() {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("{} parameters", p.name())).strong());
            if doc.scene.objects[i].link_group.is_some() {
                ui.weak("Linked duplicate: edits apply to all linked objects.");
            }
            let mut rs: Vec<egui::Response> = Vec::new();
            let mut f = |ui: &mut egui::Ui, label: &str, v: &mut f32| {
                ui.horizontal(|ui| {
                    ui.label(label);
                    rs.push(ui.add(egui::DragValue::new(v).speed(0.01).range(0.001..=1000.0)));
                });
            };
            let mut rs_i: Vec<egui::Response> = Vec::new();
            let mut n = |ui: &mut egui::Ui, label: &str, v: &mut u32, min: u32, max: u32| {
                ui.horizontal(|ui| {
                    ui.label(label);
                    rs_i.push(ui.add(egui::DragValue::new(v).speed(0.2).range(min..=max)));
                });
            };
            match &mut p {
                Primitive::Plane { size } | Primitive::Cube { size } => f(ui, "Size", size),
                Primitive::Circle { radius, segments } => {
                    f(ui, "Radius", radius);
                    n(ui, "Segments", segments, 3, 256);
                }
                Primitive::Grid { size, x_div, y_div } => {
                    f(ui, "Size", size);
                    n(ui, "X divisions", x_div, 1, 256);
                    n(ui, "Y divisions", y_div, 1, 256);
                }
                Primitive::IcoSphere { radius, subdivisions } => {
                    f(ui, "Radius", radius);
                    n(ui, "Subdivisions", subdivisions, 1, 6);
                }
                Primitive::Capsule { radius, height, segments, rings } => {
                    f(ui, "Radius", radius);
                    f(ui, "Height", height);
                    n(ui, "Segments", segments, 3, 256);
                    n(ui, "Rings", rings, 2, 64);
                }
                Primitive::Sphere { radius, segments, rings } => {
                    f(ui, "Radius", radius);
                    n(ui, "Segments", segments, 3, 256);
                    n(ui, "Rings", rings, 2, 256);
                }
                Primitive::Cylinder { radius, height, segments } | Primitive::Cone { radius, height, segments } => {
                    f(ui, "Radius", radius);
                    f(ui, "Height", height);
                    n(ui, "Segments", segments, 3, 256);
                }
                Primitive::Torus { major_radius, minor_radius, major_segments, minor_segments } => {
                    f(ui, "Major radius", major_radius);
                    f(ui, "Minor radius", minor_radius);
                    n(ui, "Major segments", major_segments, 3, 256);
                    n(ui, "Minor segments", minor_segments, 3, 256);
                }
                Primitive::Monkey { size } => f(ui, "Size", size),
                Primitive::Text { content, size, font, font_name, extrude, depth } => {
                    ui.label("Content");
                    let mut text = content.clone();
                    let te = ui.add(
                        egui::TextEdit::multiline(&mut text)
                            .desired_width(180.0)
                            .desired_rows(2)
                            .hint_text("Enter text…"),
                    );
                    if te.changed() {
                        *content = text;
                    }
                    ui.label("Font file");
                    let loaded: Vec<String> = font_registry()
                        .lock()
                        .map(|r| r.names())
                        .unwrap_or_default();
                    let label = if font_name.is_empty() {
                        format!("Built-in ({})", font.name())
                    } else {
                        font_name.clone()
                    };
                    egui::ComboBox::from_id_salt("text_font_file")
                        .selected_text(label)
                        .width(160.0)
                        .show_ui(ui, |ui| {
                            if ui.selectable_label(font_name.is_empty(), "Built-in procedural").clicked() {
                                font_name.clear();
                            }
                            for n in &loaded {
                                if ui.selectable_label(*font_name == *n, n).clicked() {
                                    *font_name = n.clone();
                                }
                            }
                        });
                    if font_name.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label("Built-in style");
                            egui::ComboBox::from_id_salt("text_font_style")
                                .selected_text(font.name())
                                .width(100.0)
                                .show_ui(ui, |ui| {
                                    for fnt in TextFont::all() {
                                        ui.selectable_value(font, fnt, fnt.name());
                                    }
                                });
                        });
                    }
                    ui.horizontal(|ui| {
                        #[cfg(not(target_arch = "wasm32"))]
                        if ui.button("Import font…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Fonts", &["ttf", "otf"])
                                .pick_file()
                            {
                                match crate::fonts_io::import_font_file(&path) {
                                    Ok(name) => {
                                        *font_name = name;
                                    }
                                    Err(e) => log::warn!("font import: {e}"),
                                }
                            }
                        }
                    });
                    f(ui, "Size", size);
                    ui.checkbox(extrude, "3D (extrude)");
                    if *extrude {
                        f(ui, "Depth", depth);
                    }
                }
                Primitive::Pyramid { size, height } => {
                    f(ui, "Base size", size);
                    f(ui, "Height", height);
                }
                Primitive::Prism { radius, height, segments } | Primitive::Tube { radius, height, segments } => {
                    f(ui, "Radius", radius);
                    f(ui, "Height", height);
                    n(ui, "Segments", segments, 3, 256);
                }
                Primitive::CurveBezier { size, segments } => {
                    f(ui, "Size", size);
                    n(ui, "Segments", segments, 4, 256);
                }
                Primitive::CurveCircle { radius, segments } => {
                    f(ui, "Radius", radius);
                    n(ui, "Segments", segments, 8, 256);
                }
                Primitive::CurvePath { length, segments } => {
                    f(ui, "Length", length);
                    n(ui, "Segments", segments, 2, 256);
                }
                Primitive::Metaball { radius, subdivisions } => {
                    f(ui, "Radius", radius);
                    n(ui, "Subdivisions", subdivisions, 1, 5);
                }
            }
            // Rebuild mesh when any parameter response changed OR primitive data diverged
            let orig = doc.scene.objects[i].primitive.clone();
            let data_changed = orig.as_ref() != Some(&p);
            let ui_changed = rs.iter().chain(rs_i.iter()).any(|r| r.changed());
            if ui_changed || data_changed {
                let p2 = p.clone();
                // Use first response if any, else force commit
                if ui_changed {
                    for r in rs.iter().chain(rs_i.iter()) {
                        if r.changed() {
                            commit(doc, r, |d| {
                                let mesh = Arc::new(p2.build());
                                let group = d.scene.objects[i].link_group;
                                for (j, o) in d.scene.objects.iter_mut().enumerate() {
                                    if j == i || (group.is_some() && o.link_group == group && o.primitive.is_some()) {
                                        o.mesh = mesh.clone();
                                        o.primitive = Some(p2.clone());
                                    }
                                }
                            });
                            break;
                        }
                    }
                } else if data_changed {
                    // Font / text content changed without a DragValue response
                    doc.checkpoint();
                    let mesh = Arc::new(p2.build());
                    let group = doc.scene.objects[i].link_group;
                    for (j, o) in doc.scene.objects.iter_mut().enumerate() {
                        if j == i || (group.is_some() && o.link_group == group && o.primitive.is_some()) {
                            o.mesh = mesh.clone();
                            o.primitive = Some(p2.clone());
                        }
                    }
                    doc.version = doc.version.wrapping_add(1);
                    doc.dirty = true;
                }
            }
        }

        // kind-specific settings (empty / camera / light)
        match doc.scene.objects[i].kind.clone() {
            ObjectKind::Mesh => {}
            ObjectKind::Empty { shape, size } => {
                ui.add_space(6.0);
                ui.label(RichText::new("Empty").strong());
                let mut sh = shape;
                egui::ComboBox::from_label("Display").selected_text(sh.name()).show_ui(ui, |ui| {
                    for x in EmptyShape::ALL {
                        ui.selectable_value(&mut sh, x, x.name());
                    }
                });
                if sh != shape {
                    doc.checkpoint();
                    doc.scene.objects[i].kind = ObjectKind::Empty { shape: sh, size };
                }
                let mut sz = size;
                let r = ui.add(egui::DragValue::new(&mut sz).speed(0.01).range(0.01..=100.0).prefix("Size "));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Empty { shape: sh, size: sz });
            }
            ObjectKind::Camera(c) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Camera").strong());
                let mut c2 = c;
                let r = ui.checkbox(&mut c2.ortho, "Orthographic");
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Camera(c2));
                if c2.ortho {
                    let r = ui.add(egui::DragValue::new(&mut c2.ortho_size).speed(0.05).range(0.1..=500.0).prefix("Size "));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Camera(c2));
                } else {
                    let r = ui.add(egui::Slider::new(&mut c2.fov_deg, 5.0..=150.0).text("Field of view"));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Camera(c2));
                }
                ui.horizontal(|ui| {
                    let r = ui.add(egui::DragValue::new(&mut c2.near).speed(0.01).range(0.001..=1000.0).prefix("Near "));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Camera(c2));
                    let r = ui.add(egui::DragValue::new(&mut c2.far).speed(1.0).range(0.1..=100000.0).prefix("Far "));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Camera(c2));
                });
                if ui.button("Look through camera (0)").clicked() {
                    actions.push(Action::CameraView);
                }
            }
            ObjectKind::Light(l) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Light").strong());
                let mut l2 = l;
                egui::ComboBox::from_label("Type").selected_text(l2.kind.name()).show_ui(ui, |ui| {
                    for t in [LightType::Point, LightType::Sun, LightType::Spot, LightType::Area] {
                        ui.selectable_value(&mut l2.kind, t, t.name());
                    }
                });
                if l2.kind != l.kind {
                    doc.checkpoint();
                    doc.scene.objects[i].kind = ObjectKind::Light(l2);
                }
                ui.horizontal(|ui| {
                    ui.label("Color");
                    let mut c = [lin_to_srgb8(l2.color[0]), lin_to_srgb8(l2.color[1]), lin_to_srgb8(l2.color[2])];
                    let r = ui.color_edit_button_srgb(&mut c);
                    commit(doc, &r, |d| {
                        let mut nl = l2;
                        nl.color = [srgb8_to_lin(c[0]), srgb8_to_lin(c[1]), srgb8_to_lin(c[2])];
                        d.scene.objects[i].kind = ObjectKind::Light(nl);
                    });
                });
                let r = ui.add(egui::Slider::new(&mut l2.intensity, 0.01..=500.0).logarithmic(true).text("Intensity"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Light(l2));
                if l2.kind != LightType::Sun {
                    let r = ui.add(egui::DragValue::new(&mut l2.range).speed(0.1).range(0.0..=1000.0).prefix("Range ")).on_hover_text("0 = unlimited");
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Light(l2));
                }
                if l2.kind == LightType::Spot {
                    let r = ui.add(egui::Slider::new(&mut l2.spot_angle_deg, 1.0..=170.0).text("Cone angle"));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Light(l2));
                    let r = ui.add(egui::Slider::new(&mut l2.spot_blend, 0.0..=1.0).text("Edge softness"));
                    commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Light(l2));
                }
                ui.weak("Lights shine along their local -Z axis. No shadows yet.");
            }
            ObjectKind::Surface(s) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Surface").strong());
                let mut s2 = s;
                let r = ui.add(egui::Slider::new(&mut s2.size, 0.1..=20.0).text("Size"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Surface(s2));
                let r = ui.add(egui::Slider::new(&mut s2.u_segments, 1..=32).text("U segments"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Surface(s2));
                let r = ui.add(egui::Slider::new(&mut s2.v_segments, 1..=32).text("V segments"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Surface(s2));
                ui.weak("NURBS surface placeholder — grid patch in the viewport.");
            }
            ObjectKind::Volume(v) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Volume").strong());
                let mut v2 = v;
                ui.horizontal(|ui| {
                    ui.label("Size");
                    let r = ui.add(egui::DragValue::new(&mut v2.size[0]).speed(0.05).prefix("X "));
                    let r2 = ui.add(egui::DragValue::new(&mut v2.size[1]).speed(0.05).prefix("Y "));
                    let r3 = ui.add(egui::DragValue::new(&mut v2.size[2]).speed(0.05).prefix("Z "));
                    if r.changed() || r2.changed() || r3.changed() {
                        doc.checkpoint();
                        doc.scene.objects[i].kind = ObjectKind::Volume(v2);
                    }
                });
                let r = ui.add(egui::Slider::new(&mut v2.density, 0.0..=10.0).text("Density"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Volume(v2));
                ui.weak("Volume domain box (fog / SDF placeholder).");
            }
            ObjectKind::GreasePencil(g) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Grease Pencil").strong());
                let mut g2 = g;
                let r = ui.add(egui::Slider::new(&mut g2.thickness, 0.001..=0.2).logarithmic(true).text("Stroke thickness"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::GreasePencil(g2));
                let r = ui.add(egui::Slider::new(&mut g2.layers, 1..=8).text("Layers"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::GreasePencil(g2));
                ui.weak("2D drawing object — strokes editable later.");
            }
            ObjectKind::Armature(a) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Armature").strong());
                ui.label(format!("{} bones", a.bones.len()));
                ui.horizontal(|ui| {
                    if ui.button("Add bone").clicked() {
                        actions.push(crate::keymap::Action::ArmatureAddBone);
                    }
                    if ui.button("Extrude").on_hover_text("New bone from selected tip (E)").clicked() {
                        actions.push(crate::keymap::Action::ArmatureExtrude);
                    }
                    if ui.button("Delete").clicked() {
                        actions.push(crate::keymap::Action::ArmatureDeleteBones);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Clear pose").clicked() {
                        actions.push(crate::keymap::Action::ArmatureClearPose);
                    }
                    if ui.button("Bind selected meshes").on_hover_text(
                        "Select Armature + Mesh(es), then bind for skinning"
                    ).clicked() {
                        actions.push(crate::keymap::Action::ArmatureBindMeshes);
                    }
                });
                ui.add_space(4.0);
                ui.label(RichText::new("Bones").strong());
                egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                    for (bi, bone) in a.bones.iter().enumerate() {
                        let sel = a.selected.contains(&(bi as u32));
                        let label = format!(
                            "{}{}  ({:.2})",
                            if a.active == Some(bi as u32) { "• " } else { "  " },
                            bone.name,
                            bone.length()
                        );
                        if ui.selectable_label(sel, label).clicked() {
                            if let ObjectKind::Armature(ref mut am) = doc.scene.objects[i].kind {
                                am.select_bone(bi as u32, ui.input(|inp| inp.modifiers.shift));
                            }
                        }
                    }
                });
                if let Some(ai) = a.active {
                    if let Some(bone) = a.bones.get(ai as usize) {
                        ui.add_space(4.0);
                        ui.label(format!("Active: {}", bone.name));
                        ui.weak("Pose: R rotate · G move  |  Edit Mode (Tab): structure");
                        ui.weak("E extrude · X delete · Bind meshes to deform");
                    }
                }
            }
            ObjectKind::LightProbe(p) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Light Probe").strong());
                let mut p2 = p;
                egui::ComboBox::from_label("Type").selected_text(p2.kind.name()).show_ui(ui, |ui| {
                    for t in [LightProbeType::Reflection, LightProbeType::Irradiance] {
                        ui.selectable_value(&mut p2.kind, t, t.name());
                    }
                });
                if p2.kind != p.kind {
                    doc.checkpoint();
                    doc.scene.objects[i].kind = ObjectKind::LightProbe(p2);
                }
                let r = ui.add(egui::Slider::new(&mut p2.radius, 0.1..=20.0).text("Radius"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::LightProbe(p2));
                let r = ui.add(egui::Slider::new(&mut p2.intensity, 0.0..=5.0).text("Intensity"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::LightProbe(p2));
                ui.weak("Reflection / irradiance probe (viewport helper for now).");
            }
            ObjectKind::Speaker(s) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Speaker").strong());
                let mut s2 = s;
                let r = ui.add(egui::Slider::new(&mut s2.volume, 0.0..=2.0).text("Volume"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Speaker(s2));
                let r = ui.add(egui::Slider::new(&mut s2.distance, 0.1..=100.0).text("Distance"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Speaker(s2));
                let r = ui.checkbox(&mut s2.muted, "Muted");
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::Speaker(s2));
                ui.weak("3D audio source placeholder (no audio playback yet).");
            }
            ObjectKind::ForceField(f) => {
                ui.add_space(6.0);
                ui.label(RichText::new("Force Field").strong());
                let mut f2 = f;
                egui::ComboBox::from_label("Type").selected_text(f2.kind.name()).show_ui(ui, |ui| {
                    for t in ForceFieldType::ALL {
                        ui.selectable_value(&mut f2.kind, t, t.name());
                    }
                });
                if f2.kind != f.kind {
                    doc.checkpoint();
                    doc.scene.objects[i].kind = ObjectKind::ForceField(f2);
                }
                let r = ui.add(egui::Slider::new(&mut f2.strength, -10.0..=10.0).text("Strength"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::ForceField(f2));
                let r = ui.add(egui::Slider::new(&mut f2.size, 0.1..=20.0).text("Size"));
                commit(doc, &r, |d| d.scene.objects[i].kind = ObjectKind::ForceField(f2));
                ui.weak("Physics force field helper (no simulation yet).");
            }
        }
        if !doc.scene.objects[i].kind.is_mesh() {
            return;
        }

        // mesh info + materials
        let o = &doc.scene.objects[i];
        ui.add_space(4.0);
        ui.weak(format!("{} vertices, {} triangles", o.mesh.vertices.len(), o.mesh.indices.len() / 3));
        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("Material").strong());
        let ngroups = doc.scene.objects[i].mesh.groups.len();
        if ngroups == 0 {
            material_ui(ui, doc, i);
        } else {
            ui.weak(format!("{ngroups} material slots (joined mesh). Use Object > Separate > By Material to split."));
            for gi in 0..ngroups {
                let cur = doc.scene.objects[i].mesh.groups[gi].material.filter(|&m| m < doc.scene.materials.len());
                let mut pick: Option<Option<usize>> = None;
                egui::ComboBox::from_label(format!("Slot {}", gi + 1))
                    .selected_text(cur.map_or("None (default)".to_string(), |m| doc.scene.materials[m].name.clone()))
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(cur.is_none(), "None (default)").clicked() {
                            pick = Some(None);
                        }
                        for (mi, m) in doc.scene.materials.iter().enumerate() {
                            if ui.selectable_label(cur == Some(mi), &m.name).clicked() {
                                pick = Some(Some(mi));
                            }
                        }
                    });
                if let Some(nm) = pick {
                    doc.checkpoint();
                    Arc::make_mut(&mut doc.scene.objects[i].mesh).groups[gi].material = nm;
                }
            }
        }
    });
}
