//! egui interface: menu bar, outliner (collections + hierarchy), properties, status bar, shortcuts window.
use crate::keymap::{self, Action, Binding};
use crate::tools::{Tool, ToolState};
use egui::RichText;
use forge_core::{font_registry, free_font_catalog, TextFont, 
    AlignMode, AlignTo, AlphaMode, ArmatureData, CameraData, Constraint, ConstraintKind, Document, EmptyShape, ForceFieldData, ForceFieldType,
    GreasePencilData, LightData, LightProbeData, LightProbeType, LightType, Material, ObjectKind, SpeakerData, SurfaceData, VolumeData,
    OriginMode, Primitive,
};
use forge_render::{ShadingMode, ViewSettings, WireMode};
use glam::{EulerRot, Quat, Vec3};
use std::sync::Arc;

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
    pub status: String,
    pub shortcut_filter: String,
    pub outliner_filter: String,
    pub renaming: Option<Rename>,
    pub rename_buf: String,
    pub rename_focus: bool,
}

// ------------------------------------------------------------------ helpers
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
    let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
fn srgb8_to_lin(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Menu entry that shows its keyboard shortcut automatically.
fn mi(ui: &mut egui::Ui, actions: &mut Vec<Action>, bindings: &[Binding], label: &str, action: Action, enabled: bool) {
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
        visuals.override_text_color = Some(egui::Color32::from_rgb(0xe0, 0xe0, 0xe0));
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
            let mesh_ok = doc.active_index().map(|i| doc.scene.objects[i].kind.is_mesh()).unwrap_or(false)
                || doc.selection.iter().any(|id| {
                    doc.scene.objects.iter().any(|o| o.id == *id && o.kind.is_mesh())
                });

            // Mode selector (Blender style)
            let in_sculpt = doc.sculpt_mode;
            if ui.add_enabled(mesh_ok || in_edit || in_sculpt, egui::SelectableLabel::new(!in_edit && !in_sculpt, "Object Mode")).clicked() {
                if in_edit {
                    actions.push(Action::ToggleEditMode);
                }
                if in_sculpt {
                    actions.push(Action::ToggleSculptMode);
                }
            }
            let edit_label = if doc.armature_edit { "Armature Edit" } else { "Edit Mode" };
            if ui.add_enabled(mesh_ok || in_edit || doc.active_armature().is_some(), egui::SelectableLabel::new(in_edit, edit_label)).clicked() {
                if in_sculpt {
                    actions.push(Action::ToggleSculptMode);
                }
                if !in_edit {
                    actions.push(Action::ToggleEditMode);
                }
            }
            if ui.add_enabled(mesh_ok || in_sculpt, egui::SelectableLabel::new(in_sculpt, "Sculpt Mode")).clicked() {
                if in_edit {
                    actions.push(Action::ToggleEditMode);
                }
                if !in_sculpt {
                    actions.push(Action::ToggleSculptMode);
                }
            }

            ui.separator();

            ui.add_enabled_ui(in_edit, |ui| {
                let domain = doc.edit.as_ref().map(|e| e.selection.domain);
                if ui.selectable_label(domain == Some(forge_core::EditDomain::Vertex), "Vertex").clicked() {
                    actions.push(Action::EditDomainVertex);
                }
                if ui.selectable_label(domain == Some(forge_core::EditDomain::Edge), "Edge").clicked() {
                    actions.push(Action::EditDomainEdge);
                }
                if ui.selectable_label(domain == Some(forge_core::EditDomain::Face), "Face").clicked() {
                    actions.push(Action::EditDomainFace);
                }
            });

            if !mesh_ok && !in_edit && !in_sculpt {
                ui.weak("Select a mesh to edit");
            }
        });
    });

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
                Some(a) if n > 0 => ui.label(format!("{n} selected, active: {}", doc.scene.objects[a].name)),
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

    // Left: Toolbar – drawn icons, screen-aware height, scroll if needed
    let screen_h = ctx.screen_rect().height();
    let toolbar_w = if screen_h < 700.0 { 36.0 } else { 40.0 };
    let btn = if screen_h < 700.0 { 26.0 } else { 30.0 };
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
                        let tools = if doc.sculpt_mode {
                            &Tool::SCULPT_MODE[..]
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
                            draw_tool_icon(ui, tool, icon_rect, selected);
                            if resp.clicked() {
                                ts.tool = tool;
                                // Mesh ops are interactive via Ctrl+LMB in Edit Mode.
                                // Toolbar only selects the tool (except Add Cube / Cursor).
                                match tool {
                                    Tool::AddCube if !doc.in_edit_mode() => {
                                        actions.push(Action::Add(forge_core::Primitive::Cube { size: 1.0 }));
                                    }
                                    Tool::Cursor => actions.push(Action::SnapCursorToSel),
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
                            resp.on_hover_text(tool.tooltip());
                        }
                        ui.add_space(3.0);
                    });
                });
        });

    // Right: Outliner (top) + Properties (bottom) – Blender default arrangement
    egui::SidePanel::right("right_panel")
    .default_width(300.0)
    .width_range(220.0..=480.0)
    .show(ctx, |ui| {
        // Outliner takes the upper part and is resizable
        egui::TopBottomPanel::top("outliner_panel")
            .resizable(true)
            .default_height(280.0)
            .height_range(120.0..=600.0)
            .show_inside(ui, |ui| {
                ui.heading("Outliner");
                ui.separator();
                outliner(ui, doc, st, &mut actions);
            });

        // Properties fill the remaining space
        ui.heading("Properties");
        ui.separator();
        properties(ui, doc, &mut actions);
    });

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

fn draw_tool_icon(ui: &mut egui::Ui, tool: Tool, rect: egui::Rect, selected: bool) {
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

    match tool {
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
                painter.add(egui::Shape::convex_polygon(vec![tip, p1, p2], col, egui::Stroke::NONE));
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
                let raw = b - a; let len = (raw.x*raw.x + raw.y*raw.y).sqrt().max(1e-5); let n = egui::vec2(-raw.y, raw.x) / len * s * 0.2;
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
                egui::Rect::from_center_size(c + egui::vec2(0.0, s * 0.25), egui::vec2(s * 1.2, s * 0.7)),
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
                vec![
                    b,
                    b + egui::vec2(-s * 0.25, 0.05),
                    a + (b - a) * 0.3,
                ],
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
            painter.line_segment(
                [c + egui::vec2(-s, 0.0), c + egui::vec2(s, 0.0)],
                thick,
            );
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
            painter.line_segment(
                [c + egui::vec2(0.0, -s), c + egui::vec2(0.0, s)],
                thick,
            );
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
                [c + egui::vec2(-s * 0.7, -s * 0.8), c + egui::vec2(-s * 0.2, s * 0.8)],
                thick,
            );
            painter.line_segment(
                [c + egui::vec2(s * 0.2, -s * 0.8), c + egui::vec2(s * 0.7, s * 0.8)],
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
                egui::Rect::from_center_size(c + egui::vec2(0.0, s * 0.15), egui::vec2(s * 1.3, s * 0.55)),
                2.0,
                col,
            );
            painter.circle_filled(c + egui::vec2(0.0, -s * 0.25), s * 0.35, col);
        }
        Tool::SculptLayer => {
            painter.rect_stroke(egui::Rect::from_center_size(c, egui::vec2(s * 1.4, s * 0.4)), 1.0, stroke);
            painter.rect_stroke(egui::Rect::from_center_size(c + egui::vec2(0.0, -s * 0.45), egui::vec2(s * 1.4, s * 0.4)), 1.0, stroke);
        }
        Tool::SculptInflate | Tool::SculptBlob => {
            painter.circle_stroke(c, s * 0.5, stroke);
            painter.circle_stroke(c, s * 0.85, stroke);
        }
        Tool::SculptCrease => {
            painter.line_segment([c + egui::vec2(-s * 0.8, s * 0.5), c + egui::vec2(0.0, -s * 0.7)], thick);
            painter.line_segment([c + egui::vec2(s * 0.8, s * 0.5), c + egui::vec2(0.0, -s * 0.7)], thick);
        }
        Tool::SculptSmooth => {
            let mut pts = Vec::new();
            for i in 0..=10 {
                let t = i as f32 / 10.0;
                pts.push(c + egui::vec2(-s + 2.0 * s * t, (t * std::f32::consts::TAU).sin() * s * 0.3));
            }
            painter.add(egui::Shape::line(pts, thick));
        }
        Tool::SculptFlatten | Tool::SculptFill | Tool::SculptScrape => {
            painter.line_segment([c + egui::vec2(-s, 0.0), c + egui::vec2(s, 0.0)], thick);
            painter.line_segment([c + egui::vec2(0.0, -s * 0.6), c + egui::vec2(0.0, s * 0.6)], stroke);
        }
        Tool::SculptPinch => {
            painter.line_segment([c + egui::vec2(-s, -s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(s, -s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(-s, s * 0.5), c], thick);
            painter.line_segment([c + egui::vec2(s, s * 0.5), c], thick);
        }
        Tool::SculptGrab | Tool::SculptSnakeHook | Tool::SculptNudge | Tool::SculptThumb => {
            painter.line_segment([c + egui::vec2(-s * 0.5, 0.0), c + egui::vec2(s * 0.5, 0.0)], thick);
            painter.line_segment([c + egui::vec2(0.0, -s * 0.5), c + egui::vec2(0.0, s * 0.5)], thick);
            painter.circle_stroke(c, s * 0.75, stroke);
        }
    }
}


// ------------------------------------------------------------------ menu bar
fn menu_bar(ui: &mut egui::Ui, doc: &mut Document, settings: &mut ViewSettings, ts: &mut ToolState, st: &mut UiState, b: &[Binding], recent: &[std::path::PathBuf], actions: &mut Vec<Action>) {
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
                    let label = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string());
                    let r = ui.button(label).on_hover_text(p.display().to_string());
                    if r.clicked() {
                        actions.push(Action::OpenRecent(p.clone()));
                        ui.close_menu();
                    }
                }
                ui.separator();
                mi(ui, actions, b, "Clear Recent", Action::ClearRecent, !recent.is_empty());
            });
            ui.separator();
            mi(ui, actions, b, "Save", Action::Save, true);
            mi(ui, actions, b, "Save As…", Action::SaveAs, true);
            ui.separator();
            mi(ui, actions, b, "Import… (glTF, GLB, OBJ, FBX)", Action::Import, true);
            if ui.button("Font Library…").clicked() {
                st.show_font_library = true;
                ui.close_menu();
            }
            ui.menu_button("Export", |ui| {
                mi(ui, actions, b, "glTF Binary (.glb)", Action::ExportGlb, true);
                mi(ui, actions, b, "Wavefront (.obj)", Action::ExportObj, true);
                mi(ui, actions, b, "Autodesk FBX (.fbx)", Action::ExportFbx, true);
                ui.separator();
                mi(ui, actions, b, "glTF Binary, selected only", Action::ExportGlbSelected, has_sel);
                mi(ui, actions, b, "Wavefront, selected only", Action::ExportObjSelected, has_sel);
                mi(ui, actions, b, "Autodesk FBX, selected only", Action::ExportFbxSelected, has_sel);
            });
            ui.separator();
            mi(ui, actions, b, "Quit", Action::Quit, true);
        });

        ui.menu_button("Edit", |ui| {
            mi(ui, actions, b, "Undo", Action::Undo, doc.can_undo());
            mi(ui, actions, b, "Redo", Action::Redo, doc.can_redo());
            ui.separator();
            mi(ui, actions, b, "Duplicate", Action::Duplicate, has_sel);
            mi(ui, actions, b, "Duplicate Linked", Action::DuplicateLinked, has_sel);
            mi(ui, actions, b, "Delete", Action::Delete, has_sel);
            mi(ui, actions, b, "Rename Active", Action::Rename, has_active);
            mi(ui, actions, b, "Make Single User", Action::MakeSingleUser, has_sel);
            ui.separator();
            mi(ui, actions, b, "Keyboard Shortcuts…", Action::ToggleShortcuts, true);
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
            mi(ui, actions, b, if in_edit { "Object Mode" } else { "Edit Mode" }, Action::ToggleEditMode, true);
            mi(ui, actions, b, if doc.sculpt_mode { "Exit Sculpt Mode" } else { "Sculpt Mode" }, Action::ToggleSculptMode, true);
            ui.separator();
            ui.add_enabled_ui(in_edit, |ui| {
                mi(ui, actions, b, "Vertex (points)", Action::EditDomainVertex, true);
                mi(ui, actions, b, "Edge (lines)", Action::EditDomainEdge, true);
                mi(ui, actions, b, "Face (surfaces)", Action::EditDomainFace, true);
            });
            if !in_edit {
                ui.weak("Select a mesh object, then Edit Mode.");
            }
        });
        ui.menu_button("Select Similar", |ui| {
                mi(ui, actions, b, "Same Type", Action::SelectSameType, has_active);
                mi(ui, actions, b, "Same Collection", Action::SelectSameCollection, has_active);
                mi(ui, actions, b, "Same Material", Action::SelectSameMaterial, has_active);
                mi(ui, actions, b, "Linked Duplicates", Action::SelectLinked, has_active);
            });
            ui.separator();
            mi(ui, actions, b, "Select Tool", Action::SetTool(Tool::Select), true);
            mi(ui, actions, b, "Box Select Tool", Action::SetTool(Tool::BoxSelect), true);
            mi(ui, actions, b, "Lasso Select Tool", Action::SetTool(Tool::Lasso), true);
            mi(ui, actions, b, "Circle Select Tool", Action::SetTool(Tool::Circle), true);
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
            mi(ui, actions, b, "Surface", Action::AddKind(ObjectKind::Surface(SurfaceData::default())), true);
            mi(ui, actions, b, "Volume", Action::AddKind(ObjectKind::Volume(VolumeData::default())), true);
            mi(ui, actions, b, "Grease Pencil", Action::AddKind(ObjectKind::GreasePencil(GreasePencilData::default())), true);
            ui.separator();
            mi(ui, actions, b, "Armature", Action::AddKind(ObjectKind::Armature(ArmatureData::default())), true);
            soon(ui, "Lattice");
            ui.separator();
            ui.menu_button("Empty", |ui| {
                for sh in EmptyShape::ALL {
                    mi(ui, actions, b, sh.name(), Action::AddKind(ObjectKind::Empty { shape: sh, size: 1.0 }), true);
                }
            });
            ui.menu_button("Image", |ui| {
                mi(ui, actions, b, "Image as Plane…", Action::AddImagePlane, true);
                soon(ui, "Reference image");
            });
            ui.separator();
            ui.menu_button("Light", |ui| {
                for t in [LightType::Point, LightType::Sun, LightType::Spot, LightType::Area] {
                    mi(ui, actions, b, t.name(), Action::AddKind(ObjectKind::Light(LightData::new(t))), true);
                }
            });
            ui.menu_button("Light Probe", |ui| {
                mi(ui, actions, b, "Reflection", Action::AddKind(ObjectKind::LightProbe(LightProbeData {
                    kind: LightProbeType::Reflection,
                    ..Default::default()
                })), true);
                mi(ui, actions, b, "Irradiance Volume", Action::AddKind(ObjectKind::LightProbe(LightProbeData {
                    kind: LightProbeType::Irradiance,
                    ..Default::default()
                })), true);
            });
            ui.menu_button("Camera", |ui| {
                mi(ui, actions, b, "Perspective", Action::AddKind(ObjectKind::Camera(CameraData::default())), true);
                mi(ui, actions, b, "Orthographic", Action::AddKind(ObjectKind::Camera(CameraData { ortho: true, ..CameraData::default() })), true);
            });
            ui.separator();
            mi(ui, actions, b, "Speaker", Action::AddKind(ObjectKind::Speaker(SpeakerData::default())), true);
            ui.menu_button("Force Field", |ui| {
                for t in ForceFieldType::ALL {
                    mi(ui, actions, b, t.name(), Action::AddKind(ObjectKind::ForceField(ForceFieldData {
                        kind: t,
                        ..Default::default()
                    })), true);
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
                mi(ui, actions, b, "Clear Location", Action::ClearLocation, has_sel);
                mi(ui, actions, b, "Clear Rotation", Action::ClearRotation, has_sel);
                mi(ui, actions, b, "Clear Scale", Action::ClearScale, has_sel);
                ui.separator();
                ui.menu_button("Align Objects", |ui| {
                    for (tn, to) in [("To Active", AlignTo::Active), ("To Selection", AlignTo::Selection), ("To 3D Cursor", AlignTo::Cursor)] {
                        ui.menu_button(tn, |ui| {
                            for (ai, an) in ["X", "Y", "Z"].iter().enumerate() {
                                ui.menu_button(*an, |ui| {
                                    for (mn, mode) in [("Min", AlignMode::Min), ("Center", AlignMode::Center), ("Max", AlignMode::Max)] {
                                        mi(ui, actions, b, mn, Action::Align(ai, mode, to), has_sel);
                                    }
                                });
                            }
                        });
                    }
                });
            });
            ui.menu_button("Apply", |ui| {
                mi(ui, actions, b, "All Transforms", Action::Apply { loc: true, rot: true, scale: true }, has_sel);
                mi(ui, actions, b, "Location", Action::Apply { loc: true, rot: false, scale: false }, has_sel);
                mi(ui, actions, b, "Rotation", Action::Apply { loc: false, rot: true, scale: false }, has_sel);
                mi(ui, actions, b, "Scale", Action::Apply { loc: false, rot: false, scale: true }, has_sel);
            });
            ui.menu_button("Set Origin", |ui| {
                mi(ui, actions, b, "Origin to Geometry", Action::SetOrigin(OriginMode::Geometry), has_sel);
                mi(ui, actions, b, "Origin to Center of Mass (vertex average)", Action::SetOrigin(OriginMode::Median), has_sel);
                mi(ui, actions, b, "Origin to 3D Cursor", Action::SetOrigin(OriginMode::Cursor), has_sel);
                mi(ui, actions, b, "Geometry to Origin", Action::GeometryToOrigin, has_sel);
            });
            ui.menu_button("Snap", |ui| {
                mi(ui, actions, b, "Selection to Cursor", Action::SnapSelToCursor, has_sel);
                mi(ui, actions, b, "Selection to Active", Action::SnapSelToActive, has_sel);
                mi(ui, actions, b, "Cursor to Selected", Action::SnapCursorToSel, has_sel);
                mi(ui, actions, b, "Cursor to Active", Action::SnapCursorToActive, has_active);
                mi(ui, actions, b, "Cursor to World Origin", Action::SnapCursorToOrigin, true);
            });
            ui.menu_button("Parent", |ui| {
                mi(ui, actions, b, "Set (selected to active)", Action::ParentSet, has_sel);
                mi(ui, actions, b, "Clear and Keep Transform", Action::ParentClearKeep, has_sel);
                mi(ui, actions, b, "Clear", Action::ParentClear, has_sel);
            });
            ui.menu_button("Collection", |ui| {
                mi(ui, actions, b, "New Collection", Action::NewCollection, true);
                ui.separator();
                ui.label("Move selection to:");
                mi(ui, actions, b, "Scene Collection", Action::MoveToCollection(None), has_sel);
                for c in &doc.scene.collections {
                    mi(ui, actions, b, &c.name, Action::MoveToCollection(Some(c.id)), has_sel);
                }
            });
            ui.menu_button("Constraints", |ui| {
                for k in ConstraintKind::all_defaults() {
                    let n = k.name();
                    mi(ui, actions, b, n, Action::AddConstraint(k), has_active);
                }
                ui.separator();
                mi(ui, actions, b, "Clear Constraints", Action::ClearConstraints, has_active);
            });
            ui.separator();
            mi(ui, actions, b, "Join", Action::Join, has_sel);
            ui.separator();
            mi(ui, actions, b, "Shade Smooth", Action::ShadeSmooth, has_sel);
            mi(ui, actions, b, "Shade Flat", Action::ShadeFlat, has_sel);
            ui.separator();
            ui.menu_button("Separate", |ui| {
                mi(ui, actions, b, "By Loose Parts", Action::SeparateLoose, has_sel);
                mi(ui, actions, b, "By Material", Action::SeparateMaterial, has_sel);
            });
            ui.separator();
            mi(ui, actions, b, "Hide Selected", Action::Hide, has_sel);
            mi(ui, actions, b, "Hide Unselected", Action::HideUnselected, has_sel);
            mi(ui, actions, b, "Show All", Action::ShowAll, true);
        });

        ui.menu_button("View", |ui| {
            ui.menu_button("Shading", |ui| {
                for m in [ShadingMode::Rendered, ShadingMode::Albedo, ShadingMode::Normals, ShadingMode::Roughness, ShadingMode::Metallic, ShadingMode::UvChecker] {
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
                ui.add(egui::Slider::new(&mut settings.shadow_softness, 0.0..=4.0).text("Shadow softness"));
            }
            ui.separator();
            ui.add(egui::Slider::new(&mut settings.exposure, 0.1..=8.0).logarithmic(true).text("Exposure"));
            ui.add(egui::Slider::new(&mut settings.background_blur, 0.0..=1.0).text("Background blur"));
            ui.add(egui::Slider::new(&mut settings.env_strength, 0.0..=2.0).text("Environment strength"));
            ui.add(egui::Slider::new(&mut settings.env_rotation, -3.14..=3.14).text("Environment rotation"));
            ui.separator();
            mi(ui, actions, b, "Load HDRI…", Action::LoadHdri, true);
            mi(ui, actions, b, "Reset sky", Action::ResetSky, true);
            ui.separator();
            mi(ui, actions, b, "Frame selected", Action::FrameSelected, true);
            mi(ui, actions, b, "Frame all", Action::FrameAll, true);
            mi(ui, actions, b, "Camera view", Action::CameraView, true);
            ui.separator();
            mi(ui, actions, b, "Render Image…", Action::RenderImage, true);
            ui.separator();
            mi(ui, actions, b, "Toggle snapping", Action::ToggleSnap, true);
            ui.checkbox(&mut ts.snap.enabled, "Snapping enabled");
        });

        ui.menu_button("Tools", |ui| {
            soon(ui, "Modifiers");
            soon(ui, "UV editor");
            soon(ui, "Texture painting");
            if ui.button("Sculpt Mode").clicked() { actions.push(Action::ToggleSculptMode); ui.close_menu(); }
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
            mi(ui, actions, b, "Keyboard Shortcuts", Action::ToggleShortcuts, true);
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
    ui.add(egui::TextEdit::singleline(&mut st.outliner_filter).hint_text("Filter by name…").desired_width(f32::INFINITY));
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
        let root_label = if doc.active_collection.is_none() { RichText::new("Scene Collection").strong() } else { RichText::new("Scene Collection") };
        egui::CollapsingHeader::new(root_label).default_open(true).show(ui, |ui| {
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

fn collection_node(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, actions: &mut Vec<Action>, cid: u64) {
    let Some(ci) = doc.scene.collections.iter().position(|c| c.id == cid) else { return };
    let id = ui.make_persistent_id(("collection", cid));
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let header = state.show_header(ui, |ui| {
        let mut vis = doc.scene.collections[ci].visible;
        let r = ui.checkbox(&mut vis, "").on_hover_text("Visible");
        commit(doc, &r, |d| d.scene.collections[ci].visible = vis);
        if st.renaming == Some(Rename::Collection(cid)) {
            rename_field(ui, doc, st, |d, name| d.scene.collections[ci].name = name);
        } else {
            let name = doc.scene.collections[ci].name.clone();
            let active = doc.active_collection == Some(cid);
            let resp = ui.selectable_label(active, if active { RichText::new(&name).strong() } else { RichText::new(&name) });
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

fn object_tree(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, actions: &mut Vec<Action>, idx: usize, coll: Option<u64>, depth: usize) {
    if depth > 32 || idx >= doc.scene.objects.len() {
        return;
    }
    let id = doc.scene.objects[idx].id;
    let kids: Vec<usize> = doc.scene.children_of(id).into_iter().filter(|&c| doc.scene.objects[c].collection == coll).collect();
    if kids.is_empty() {
        ui.horizontal(|ui| object_row(ui, doc, st, actions, idx));
    } else {
        let cid = ui.make_persistent_id(("object", id));
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), cid, true);
        let header = state.show_header(ui, |ui| object_row(ui, doc, st, actions, idx));
        header.body(|ui| {
            for k in kids {
                object_tree(ui, doc, st, actions, k, coll, depth + 1);
            }
        });
    }
}

fn rename_field(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, apply: impl FnOnce(&mut Document, String)) {
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

fn object_row(ui: &mut egui::Ui, doc: &mut Document, st: &mut UiState, actions: &mut Vec<Action>, idx: usize) {
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
    let resp = ui.selectable_label(sel, if active && sel { RichText::new(&name).strong() } else { RichText::new(&name) });
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
    resp.context_menu(|ui| {
        let mut push = |ui: &mut egui::Ui, label: &str, a: Action| {
            if ui.button(label).clicked() {
                actions.push(a);
                ui.close_menu();
            }
        };
        push(ui, "Rename", Action::Rename);
        push(ui, "Duplicate", Action::Duplicate);
        push(ui, "Delete", Action::Delete);
        push(ui, "Hide", Action::Hide);
        ui.separator();
        push(ui, "Select Children", Action::SelectChildren);
        push(ui, "Select Parent", Action::SelectParent);
        ui.separator();
        // Parenting (Blender-style): make the active object the parent of the selection,
        // or clear the parent relationship.
        push(ui, "Set Parent (to Active)", Action::ParentSet);
        push(ui, "Clear Parent (keep transform)", Action::ParentClearKeep);
        push(ui, "Clear Parent", Action::ParentClear);
        ui.separator();
        push(ui, "Move to Collection…", Action::OpenMoveToCollection);
    });
}

fn move_popup(ctx: &egui::Context, doc: &Document, st: &mut UiState, actions: &mut Vec<Action>) {
    if !st.show_move_popup {
        return;
    }
    let mut open = true;
    let mut close = false;
    egui::Window::new("Move to Collection").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
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
            ui.heading("Downloadable (free licenses)");
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
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.button("Download").clicked() {
                                match crate::fonts_io::download_catalog_font(f.name) {
                                    Ok(n) => st.font_status = format!("Downloaded {n}"),
                                    Err(e) => st.font_status = format!("Failed: {e}"),
                                }
                            }
                            #[cfg(target_arch = "wasm32")]
                            ui.label("—");
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
    egui::Window::new("Keyboard Shortcuts").open(&mut open).default_size([640.0, 640.0]).resizable(true).collapsible(false).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.add(egui::TextEdit::singleline(&mut st.shortcut_filter).hint_text("e.g. move, Ctrl+S, collection").desired_width(340.0));
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
                    .filter(|(k, l)| f.is_empty() || k.to_lowercase().contains(&f) || l.to_lowercase().contains(&f) || cat.to_lowercase().contains(&f))
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                egui::CollapsingHeader::new(RichText::new(cat).strong()).default_open(true).show(ui, |ui| {
                    egui::Grid::new(("sc", cat)).striped(true).num_columns(2).show(ui, |ui| {
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
                    .filter(|(k, l)| f.is_empty() || k.to_lowercase().contains(&f) || l.to_lowercase().contains(&f) || cat.to_lowercase().contains(&f))
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                egui::CollapsingHeader::new(RichText::new(cat).strong()).default_open(true).show(ui, |ui| {
                    egui::Grid::new(("scx", cat)).striped(true).num_columns(2).show(ui, |ui| {
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
    let cur = t.and_then(|id| names.iter().find(|(i, _)| *i == id)).map_or("None".to_string(), |(_, n)| n.clone());
    egui::ComboBox::from_label("Target").selected_text(cur).show_ui(ui, |ui| {
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

fn limit_rows(ui: &mut egui::Ui, label: &str, min: &mut [f32; 3], max: &mut [f32; 3], enable: &mut [bool; 3], speed: f64) -> bool {
    let mut ch = false;
    for (k, ax) in ["X", "Y", "Z"].iter().enumerate() {
        ui.horizontal(|ui| {
            ch |= ui.checkbox(&mut enable[k], format!("{label} {ax}")).changed();
            ch |= ui.add(egui::DragValue::new(&mut min[k]).speed(speed).prefix("min ")).changed();
            ch |= ui.add(egui::DragValue::new(&mut max[k]).speed(speed).prefix("max ")).changed();
        });
    }
    ch
}

/// Edits one constraint in place; returns true if anything changed.
fn constraint_body(ui: &mut egui::Ui, c: &mut Constraint, names: &[(u64, String)]) -> bool {
    let mut ch = false;
    ch |= ui.checkbox(&mut c.enabled, "Enabled").changed();
    ch |= ui.add(egui::Slider::new(&mut c.influence, 0.0..=1.0).text("Influence")).changed();
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
        ConstraintKind::LimitLocation { min, max, enable } => ch |= limit_rows(ui, "Loc", min, max, enable, 0.05),
        ConstraintKind::LimitRotation { min_deg, max_deg, enable } => ch |= limit_rows(ui, "Rot", min_deg, max_deg, enable, 0.5),
        ConstraintKind::LimitScale { min, max, enable } => ch |= limit_rows(ui, "Scale", min, max, enable, 0.01),
        ConstraintKind::LimitDistance { distance, .. } => {
            ch |= ui.add(egui::DragValue::new(distance).speed(0.05).range(0.0..=10000.0).prefix("Distance ")).changed();
        }
        ConstraintKind::Floor { height } => {
            ch |= ui.add(egui::DragValue::new(height).speed(0.05).prefix("Height ")).changed();
        }
        ConstraintKind::TrackTo { .. } => {
            ui.weak("Local -Z points at the target (Y up).");
        }
        _ => {}
    }
    ch
}

fn material_ui(ui: &mut egui::Ui, doc: &mut Document, i: usize) {
    let cur = doc.scene.objects[i].material.filter(|&m| m < doc.scene.materials.len());
    let mut pick: Option<Option<usize>> = None;
    egui::ComboBox::from_label("Slot")
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
        let mut c = [lin_to_srgb8(m.base_color[0]), lin_to_srgb8(m.base_color[1]), lin_to_srgb8(m.base_color[2]), (m.base_color[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8];
        let r = ui.color_edit_button_srgba_unmultiplied(&mut c);
        commit(doc, &r, |d| d.scene.materials[mi].base_color = [srgb8_to_lin(c[0]), srgb8_to_lin(c[1]), srgb8_to_lin(c[2]), c[3] as f32 / 255.0]);
    });
    let mut metallic = m.metallic;
    let r = ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0).text("Metallic"));
    commit(doc, &r, |d| d.scene.materials[mi].metallic = metallic);
    let mut rough = m.roughness;
    let r = ui.add(egui::Slider::new(&mut rough, 0.0..=1.0).text("Roughness"));
    commit(doc, &r, |d| d.scene.materials[mi].roughness = rough);
    ui.horizontal(|ui| {
        ui.label("Emission");
        let mut c = [lin_to_srgb8(m.emissive[0]), lin_to_srgb8(m.emissive[1]), lin_to_srgb8(m.emissive[2])];
        let r = ui.color_edit_button_srgb(&mut c);
        commit(doc, &r, |d| d.scene.materials[mi].emissive = [srgb8_to_lin(c[0]), srgb8_to_lin(c[1]), srgb8_to_lin(c[2])]);
    });
    let mut mode = m.alpha_mode;
    let before = mode;
    egui::ComboBox::from_label("Alpha").selected_text(format!("{mode:?}")).show_ui(ui, |ui| {
        ui.selectable_value(&mut mode, AlphaMode::Opaque, "Opaque");
        ui.selectable_value(&mut mode, AlphaMode::Mask, "Mask");
        ui.selectable_value(&mut mode, AlphaMode::Blend, "Blend");
    });
    if mode != before {
        doc.checkpoint();
        doc.scene.materials[mi].alpha_mode = mode;
    }
    let tex = [("base", m.base_color_tex), ("normal", m.normal_tex), ("metal/rough", m.metallic_roughness_tex), ("occlusion", m.occlusion_tex), ("emissive", m.emissive_tex)];
    let used: Vec<&str> = tex.iter().filter(|(_, t)| t.is_some()).map(|(n, _)| *n).collect();
    if !used.is_empty() {
        ui.weak(format!("Textures: {}", used.join(", ")));
    }
}

fn properties(ui: &mut egui::Ui, doc: &mut Document, actions: &mut Vec<Action>) {
    ui.heading("Properties");
    ui.separator();
    let Some(i) = doc.active_index() else {
        ui.label("Nothing selected.\nClick an object in the viewport or the outliner.");
        return;
    };
    if doc.selection.len() > 1 {
        ui.weak(format!("{} objects selected (editing the active one)", doc.selection.len()));
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
                        #[cfg(not(target_arch = "wasm32"))]
                        if ui.button("Download free fonts…").clicked() {
                            for f in free_font_catalog() {
                                match crate::fonts_io::download_catalog_font(f.name) {
                                    Ok(name) => {
                                        if font_name.is_empty() {
                                            *font_name = name;
                                        }
                                    }
                                    Err(e) => log::warn!("font download {}: {e}", f.name),
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
