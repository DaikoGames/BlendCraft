//! Tool state (selection tools, gizmo tools, pivot, orientation, snapping) and selection helpers.
use forge_core::{Document, OrbitCamera};
use glam::{Vec2, Vec3};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    // Selection
    Select,
    BoxSelect,
    Lasso,
    Circle,
    // Transform
    Cursor,
    Move,
    Rotate,
    Scale,
    Transform,
    // Annotate / Measure
    Annotate,
    Measure,
    // Edit Mode mesh tools
    AddCube,
    ExtrudeRegion,
    InsetFaces,
    Bevel,
    LoopCut,
    Knife,
    PolyBuild,
    Spin,
    Smooth,
    EdgeSlide,
    ShrinkFatten,
    Shear,
    RipRegion,
    // Sculpt Mode brushes (Blender toolbar set)
    SculptDraw,
    SculptDrawSharp,
    SculptClay,
    SculptClayStrips,
    SculptLayer,
    SculptInflate,
    SculptBlob,
    SculptCrease,
    SculptSmooth,
    SculptFlatten,
    SculptFill,
    SculptScrape,
    SculptPinch,
    SculptGrab,
    SculptSnakeHook,
    SculptNudge,
    SculptThumb,
}

impl Tool {
    /// Tools shown in Object Mode (left toolbar)
    pub const OBJECT_MODE: [Tool; 11] = [
        Tool::Select,
        Tool::Cursor,
        Tool::Move,
        Tool::Rotate,
        Tool::Scale,
        Tool::Transform,
        Tool::Annotate,
        Tool::Measure,
        Tool::AddCube,
        Tool::BoxSelect,
        Tool::Lasso,
    ];

    /// Tools shown in Edit Mode (left toolbar) – matches the Blender list you provided
    pub const SCULPT_MODE: [Tool; 17] = [
        Tool::SculptDraw,
        Tool::SculptDrawSharp,
        Tool::SculptClay,
        Tool::SculptClayStrips,
        Tool::SculptLayer,
        Tool::SculptInflate,
        Tool::SculptBlob,
        Tool::SculptCrease,
        Tool::SculptSmooth,
        Tool::SculptFlatten,
        Tool::SculptFill,
        Tool::SculptScrape,
        Tool::SculptPinch,
        Tool::SculptGrab,
        Tool::SculptSnakeHook,
        Tool::SculptNudge,
        Tool::SculptThumb,
    ];

    pub const EDIT_MODE: [Tool; 21] = [
        Tool::Select,
        Tool::Cursor,
        Tool::Move,
        Tool::Rotate,
        Tool::Scale,
        Tool::Transform,
        Tool::Annotate,
        Tool::Measure,
        Tool::AddCube,
        Tool::ExtrudeRegion,
        Tool::InsetFaces,
        Tool::Bevel,
        Tool::LoopCut,
        Tool::Knife,
        Tool::PolyBuild,
        Tool::Spin,
        Tool::Smooth,
        Tool::EdgeSlide,
        Tool::ShrinkFatten,
        Tool::Shear,
        Tool::RipRegion,
    ];

    pub const ALL: [Tool; 24] = [
        Tool::Select,
        Tool::BoxSelect,
        Tool::Lasso,
        Tool::Circle,
        Tool::Cursor,
        Tool::Move,
        Tool::Rotate,
        Tool::Scale,
        Tool::Transform,
        Tool::Annotate,
        Tool::Measure,
        Tool::AddCube,
        Tool::ExtrudeRegion,
        Tool::InsetFaces,
        Tool::Bevel,
        Tool::LoopCut,
        Tool::Knife,
        Tool::PolyBuild,
        Tool::Spin,
        Tool::Smooth,
        Tool::EdgeSlide,
        Tool::ShrinkFatten,
        Tool::Shear,
        Tool::RipRegion,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::BoxSelect => "Box Select",
            Tool::Lasso => "Lasso",
            Tool::Circle => "Circle Select",
            Tool::Cursor => "Cursor",
            Tool::Move => "Move",
            Tool::Rotate => "Rotate",
            Tool::Scale => "Scale",
            Tool::Transform => "Transform",
            Tool::Annotate => "Annotate",
            Tool::Measure => "Measure",
            Tool::AddCube => "Add Cube",
            Tool::ExtrudeRegion => "Extrude Region",
            Tool::InsetFaces => "Inset Faces",
            Tool::Bevel => "Bevel",
            Tool::LoopCut => "Loop Cut",
            Tool::Knife => "Knife",
            Tool::PolyBuild => "Poly Build",
            Tool::Spin => "Spin – Ctrl+LMB drag",
            Tool::Smooth => "Smooth",
            Tool::EdgeSlide => "Edge Slide – Ctrl+LMB drag",
            Tool::ShrinkFatten => "Shrink/Fatten",
            Tool::Shear => "Shear – Ctrl+LMB drag",
            Tool::RipRegion => "Rip Region",
            Tool::SculptDraw => "Draw",
            Tool::SculptDrawSharp => "Draw Sharp",
            Tool::SculptClay => "Clay",
            Tool::SculptClayStrips => "Clay Strips",
            Tool::SculptLayer => "Layer",
            Tool::SculptInflate => "Inflate",
            Tool::SculptBlob => "Blob",
            Tool::SculptCrease => "Crease",
            Tool::SculptSmooth => "Smooth",
            Tool::SculptFlatten => "Flatten",
            Tool::SculptFill => "Fill",
            Tool::SculptScrape => "Scrape",
            Tool::SculptPinch => "Pinch",
            Tool::SculptGrab => "Grab",
            Tool::SculptSnakeHook => "Snake Hook",
            Tool::SculptNudge => "Nudge",
            Tool::SculptThumb => "Thumb",
        }
    }

    /// Compact toolbar label – ASCII only so it always renders.
    pub fn icon(self) -> &'static str {
        match self {
            Tool::Select => "Sel",
            Tool::BoxSelect => "Box",
            Tool::Lasso => "Las",
            Tool::Circle => "Cir",
            Tool::Cursor => "Cur",
            Tool::Move => "Mov",
            Tool::Rotate => "Rot",
            Tool::Scale => "Scl",
            Tool::Transform => "Xform",
            Tool::Annotate => "Ann",
            Tool::Measure => "Mea",
            Tool::AddCube => "Cube",
            Tool::ExtrudeRegion => "Ext",
            Tool::InsetFaces => "Ins",
            Tool::Bevel => "Bev",
            Tool::LoopCut => "Loop",
            Tool::Knife => "Knf",
            Tool::PolyBuild => "Poly",
            Tool::Spin => "Spin – Ctrl+LMB drag",
            Tool::Smooth => "Smo",
            Tool::EdgeSlide => "Slide",
            Tool::ShrinkFatten => "S/F",
            Tool::Shear => "Shr",
            Tool::RipRegion => "Rip",
            Tool::SculptDraw => "Drw",
            Tool::SculptDrawSharp => "Shp",
            Tool::SculptClay => "Clay",
            Tool::SculptClayStrips => "ClS",
            Tool::SculptLayer => "Lay",
            Tool::SculptInflate => "Inf",
            Tool::SculptBlob => "Blob",
            Tool::SculptCrease => "Crs",
            Tool::SculptSmooth => "Smo",
            Tool::SculptFlatten => "Flt",
            Tool::SculptFill => "Fill",
            Tool::SculptScrape => "Scr",
            Tool::SculptPinch => "Pch",
            Tool::SculptGrab => "Grb",
            Tool::SculptSnakeHook => "Snk",
            Tool::SculptNudge => "Ndg",
            Tool::SculptThumb => "Thm",
        }
    }

    /// Tooltip text including Blender-like shortcut hints.
    pub fn tooltip(self) -> &'static str {
        match self {
            Tool::Select => "Select (W)",
            Tool::BoxSelect => "Box Select (B)",
            Tool::Lasso => "Lasso Select",
            Tool::Circle => "Circle Select (C)",
            Tool::Cursor => "3D Cursor",
            Tool::Move => "Move (G)",
            Tool::Rotate => "Rotate (R)",
            Tool::Scale => "Scale (S)",
            Tool::Transform => "Transform",
            Tool::Annotate => "Annotate – draw strokes",
            Tool::Measure => "Measure – click two points",
            Tool::AddCube => "Add Cube",
            Tool::ExtrudeRegion => "Extrude Region (E) – Ctrl+LMB drag",
            Tool::InsetFaces => "Inset Faces (I) – Ctrl+LMB drag",
            Tool::Bevel => "Bevel (Ctrl+B) – Ctrl+LMB drag",
            Tool::LoopCut => "Loop Cut (Ctrl+R) – Ctrl+LMB",
            Tool::Knife => "Knife (K) – Ctrl+LMB",
            Tool::PolyBuild => "Poly Build (P) – Ctrl+LMB",
            Tool::Spin => "Spin – Ctrl+LMB drag",
            Tool::Smooth => "Smooth – Ctrl+LMB drag",
            Tool::EdgeSlide => "Edge Slide – Ctrl+LMB drag",
            Tool::ShrinkFatten => "Shrink/Fatten (Alt+S) – Ctrl+LMB drag",
            Tool::Shear => "Shear – Ctrl+LMB drag",
            Tool::RipRegion => "Rip Region (V) – Ctrl+LMB",
            Tool::SculptDraw => "Draw – LMB, Ctrl invert",
            Tool::SculptDrawSharp => "Draw Sharp – sharp falloff",
            Tool::SculptClay => "Clay – build volume",
            Tool::SculptClayStrips => "Clay Strips – aggressive clay",
            Tool::SculptLayer => "Layer – fixed height",
            Tool::SculptInflate => "Inflate – Ctrl deflate",
            Tool::SculptBlob => "Blob – spherical push",
            Tool::SculptCrease => "Crease – sharp indents",
            Tool::SculptSmooth => "Smooth – Shift enhance details",
            Tool::SculptFlatten => "Flatten – average plane",
            Tool::SculptFill => "Fill – raise valleys",
            Tool::SculptScrape => "Scrape – lower peaks",
            Tool::SculptPinch => "Pinch – pull to center",
            Tool::SculptGrab => "Grab – LMB drag",
            Tool::SculptSnakeHook => "Snake Hook – pull along stroke",
            Tool::SculptNudge => "Nudge – slide along stroke",
            Tool::SculptThumb => "Thumb – flatten + slide",
        }
    }

    pub fn is_gizmo(self) -> bool {
        matches!(self, Tool::Move | Tool::Rotate | Tool::Scale | Tool::Transform)
    }

    pub fn is_select(self) -> bool {
        matches!(self, Tool::Select | Tool::BoxSelect | Tool::Lasso | Tool::Circle)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pivot {
    Median,
    Active,
    Individual,
    Cursor,
    BoundsCenter,
}

impl Pivot {
    pub const ALL: [Pivot; 5] = [Pivot::Median, Pivot::Active, Pivot::Individual, Pivot::Cursor, Pivot::BoundsCenter];
    pub fn name(self) -> &'static str {
        match self {
            Pivot::Median => "Median Point",
            Pivot::Active => "Active Element",
            Pivot::Individual => "Individual Origins",
            Pivot::Cursor => "3D Cursor",
            Pivot::BoundsCenter => "Bounding Box Center",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Orientation {
    Global,
    Local,
    View,
}

impl Orientation {
    pub const ALL: [Orientation; 3] = [Orientation::Global, Orientation::Local, Orientation::View];
    pub fn name(self) -> &'static str {
        match self {
            Orientation::Global => "Global",
            Orientation::Local => "Local",
            Orientation::View => "View",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapMode {
    Increment,
    Vertex,
}

#[derive(Clone, Debug)]
pub struct SnapSettings {
    pub enabled: bool,
    pub mode: SnapMode,
    pub move_step: f32,
    pub rotate_step_deg: f32,
    pub scale_step: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SnapMode::Increment,
            move_step: 0.25,
            rotate_step_deg: 15.0,
            scale_step: 0.1,
        }
    }
}

pub struct ToolState {
    pub tool: Tool,
    pub pivot: Pivot,
    pub orient: Orientation,
    pub snap: SnapSettings,
    pub circle_radius: f32,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            tool: Tool::Select,
            pivot: Pivot::Median,
            orient: Orientation::Global,
            snap: SnapSettings::default(),
            circle_radius: 40.0,
        }
    }
}

impl ToolState {
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Selection helpers (original API, kept for compatibility with main.rs)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelectOp {
    Replace,
    Extend,
    Subtract,
}

/// A shape being dragged with the Box / Lasso tool (physical pixels).
pub enum SelShape {
    Box { a: Vec2, b: Vec2 },
    Lasso(Vec<Vec2>),
}

/// World point -> physical pixel position in the window. `vp` = viewport x, y, w, h.
pub fn project_px(cam: &OrbitCamera, vp: [f32; 4], p: Vec3) -> Option<Vec2> {
    let c = cam.view_proj(vp[2] / vp[3]) * p.extend(1.0);
    if c.w <= 1e-5 {
        return None;
    }
    Some(Vec2::new(
        vp[0] + (c.x / c.w * 0.5 + 0.5) * vp[2],
        vp[1] + (1.0 - (c.y / c.w * 0.5 + 0.5)) * vp[3],
    ))
}

/// Screen position (bounding-box center) of every visible object.
pub fn anchors(doc: &Document, cam: &OrbitCamera, vp: [f32; 4]) -> Vec<(usize, Vec2)> {
    let worlds = doc.scene.world_matrices();
    (0..doc.scene.objects.len())
        .filter(|&i| doc.scene.is_visible(i))
        .filter_map(|i| {
            let b = doc.scene.object_world_bounds(i, &worlds[i])?;
            project_px(cam, vp, b.center()).map(|p| (i, p))
        })
        .collect()
}

pub fn apply_selection(doc: &mut Document, hits: &[usize], op: SelectOp) {
    match op {
        SelectOp::Replace => {
            doc.select_none();
            for &i in hits {
                doc.select_add(i);
            }
        }
        SelectOp::Extend => {
            for &i in hits {
                doc.select_add(i);
            }
        }
        SelectOp::Subtract => {
            for &i in hits {
                doc.select_remove(i);
            }
        }
    }
}

pub fn point_in_polygon(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The tool bar floating in the top-left corner of the 3D viewport.
pub fn draw_toolbar(ctx: &egui::Context, rect: egui::Rect, ts: &mut ToolState) {
    egui::Area::new(egui::Id::new("tool_bar"))
        .fixed_pos(rect.min + egui::vec2(8.0, 8.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    for t in [Tool::Select, Tool::Move, Tool::Rotate, Tool::Scale] {
                        ui.selectable_value(&mut ts.tool, t, t.name());
                    }
                });
                ui.horizontal(|ui| {
                    egui::ComboBox::from_label("Pivot")
                        .selected_text(ts.pivot.name())
                        .show_ui(ui, |ui| {
                            for p in Pivot::ALL {
                                ui.selectable_value(&mut ts.pivot, p, p.name());
                            }
                        });
                    egui::ComboBox::from_label("Orientation")
                        .selected_text(ts.orient.name())
                        .show_ui(ui, |ui| {
                            for o in Orientation::ALL {
                                ui.selectable_value(&mut ts.orient, o, o.name());
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut ts.snap.enabled, "Snap")
                        .on_hover_text("Shift+Tab. Hold Ctrl to toggle while transforming.");
                    egui::ComboBox::from_label("Snap to")
                        .selected_text(match ts.snap.mode {
                            SnapMode::Increment => "Increment",
                            SnapMode::Vertex => "Vertex",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut ts.snap.mode, SnapMode::Increment, "Increment");
                            ui.selectable_value(&mut ts.snap.mode, SnapMode::Vertex, "Vertex");
                        });
                    if ts.snap.mode == SnapMode::Increment {
                        ui.add(
                            egui::DragValue::new(&mut ts.snap.move_step)
                                .speed(0.01)
                                .range(0.001..=100.0)
                                .prefix("move "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut ts.snap.rotate_step_deg)
                                .speed(0.5)
                                .range(0.1..=180.0)
                                .suffix("°"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut ts.snap.scale_step)
                                .speed(0.01)
                                .range(0.001..=10.0)
                                .prefix("scale "),
                        );
                    }
                });
            });
        });
}
