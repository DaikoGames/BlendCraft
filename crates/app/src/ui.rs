//! egui interface: menu bar, outliner (collections + hierarchy), properties, status bar, shortcuts window.
use crate::keymap::{self, Action, Binding};
use crate::tools::{Tool, ToolState};
use egui::RichText;
use forge_core::{font_registry, free_font_catalog, TextFont, 
    AlignMode, AlignTo, AlphaMode, CameraData, Constraint, ConstraintKind, Document, EmptyShape, LightData, LightType, Material, ObjectKind,
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

    egui::TopBottomPanel::top("menu").show(ctx, |ui| menu_bar(ui, doc, settings, ts, st, bindings, recent, &mut actions));

    egui::TopBottomPanel::top("mode_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let in_edit = doc.in_edit_mode();
            let mesh_ok = doc.active_index().map(|i| doc.scene.objects[i].kind.is_mesh()).unwrap_or(false)
                || doc.selection.iter().any(|id| {
                    doc.scene.objects.iter().any(|o| o.id == *id && o.kind.is_mesh())
                });
            if ui.add_enabled(mesh_ok || in_edit, egui::SelectableLabel::new(!in_edit, "Object Mode")).clicked() {
                if in_edit {
                    actions.push(Action::ToggleEditMode);
                }
            }
            if ui.add_enabled(mesh_ok || in_edit, egui::SelectableLabel::new(in_edit, "Edit Mode")).clicked() {
                if !in_edit {
                    actions.push(Action::ToggleEditMode);
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
            if !mesh_ok && !in_edit {
                ui.weak("Select a mesh to edit");
            }
        });
    });
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let tris: usize = (0..doc.scene.objects.len()).filter(|&i| doc.scene.is_visible(i)).map(|i| doc.scene.objects[i].mesh.indices.len() / 3).sum();
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

    egui::SidePanel::left("outliner").default_width(250.0).show(ctx, |ui| outliner(ui, doc, st, &mut actions));
    egui::SidePanel::right("properties").default_width(320.0).show(ctx, |ui| properties(ui, doc, &mut actions));

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
            ui.separator();
            ui.add_enabled_ui(in_edit, |ui| {
                mi(ui, actions, b, "Vertex", Action::EditDomainVertex, true);
                mi(ui, actions, b, "Edge", Action::EditDomainEdge, true);
                mi(ui, actions, b, "Face", Action::EditDomainFace, true);
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
            soon(ui, "Surface");
            // Metaball + Text are mesh approximations in Primitive::all()
            soon(ui, "Volume");
            soon(ui, "Grease Pencil");
            ui.separator();
            soon(ui, "Armature");
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
            soon(ui, "Light Probe");
            ui.menu_button("Camera", |ui| {
                mi(ui, actions, b, "Perspective", Action::AddKind(ObjectKind::Camera(CameraData::default())), true);
                mi(ui, actions, b, "Orthographic", Action::AddKind(ObjectKind::Camera(CameraData { ortho: true, ..CameraData::default() })), true);
            });
            ui.separator();
            soon(ui, "Speaker");
            soon(ui, "Force Field");
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
            soon(ui, "Edit mode (vertices / edges / faces)");
            soon(ui, "Modifiers");
            soon(ui, "UV editor");
            soon(ui, "Texture painting");
            soon(ui, "Sculpt");
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
