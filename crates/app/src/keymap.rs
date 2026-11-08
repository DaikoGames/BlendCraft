//! All actions and the keyboard bindings. This table is the single source of truth:
//! key handling, menu shortcut hints and the "Keyboard Shortcuts" window all read from it.
use crate::tools::Tool;
use forge_core::{AlignMode, AlignTo, ConstraintKind, ObjectKind, OriginMode, Primitive};
use forge_render::ShadingMode;
use std::path::PathBuf;
use winit::keyboard::KeyCode;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    // file
    New,
    Open,
    OpenRecent(PathBuf),
    ClearRecent,
    Save,
    SaveAs,
    Import,
    ExportGlb,
    ExportObj,
    ExportGlbSelected,
    ExportObjSelected,
    ExportFbx,
    ExportFbxSelected,
    Quit,
    // edit
    Undo,
    Redo,
    Duplicate,
    DuplicateLinked,
    Delete,
    Extrude,
    InsetFaces,
    Bevel,
    LoopCut,
    Knife,
    PolyBuild,
    Spin,
    SmoothVerts,
    EdgeSlide,
    ShrinkFatten,
    Shear,
    RipRegion,
    MergeVerts,
    Rename,
    MakeSingleUser,
    // select
    SelectAll,
    SelectNone,
    SelectInvert,
    SelectRandom,
    SelectChildren,
    SelectParent,
    SelectSameType,
    SelectSameCollection,
    SelectSameMaterial,
    SelectLinked,
    // tools
    SetTool(Tool),
    CycleTool,
    ToggleSnap,
    // modal transforms
    ModalMove,
    ModalRotate,
    ModalScale,
    // object
    Hide,
    HideUnselected,
    ShowAll,
    ParentSet,
    ParentClear,
    ParentClearKeep,
    NewCollection,
    OpenMoveToCollection,
    MoveToCollection(Option<u64>),
    Join,
    ShadeSmooth,
    ShadeFlat,
    SeparateLoose,
    SeparateMaterial,
    SetOrigin(OriginMode),
    GeometryToOrigin,
    Apply { loc: bool, rot: bool, scale: bool },
    ClearLocation,
    ClearRotation,
    ClearScale,
    Align(usize, AlignMode, AlignTo),
    SnapSelToCursor,
    SnapSelToActive,
    SnapCursorToSel,
    SnapCursorToActive,
    SnapCursorToOrigin,
    AddConstraint(ConstraintKind),
    ClearConstraints,
    // add
    Add(Primitive),
    AddKind(ObjectKind),
    AddImagePlane,
    LoadHdri,
    ResetSky,
    // view
    FrameAll,
    FrameSelected,
    CameraView,
    SetShading(ShadingMode),
    CycleWire,
    ToggleBackground,
    ExposureDown,
    ExposureUp,
    EnvRotate(i8),
    RenderImage,
    // help
    ToggleShortcuts,
    ShowAbout,
    ToggleEditMode,
    ToggleSculptMode,
    ArmatureAddBone,
    ArmatureExtrude,
    ArmatureDeleteBones,
    ArmatureClearPose,
    ArmatureBindMeshes,
    EditDomainVertex,
    EditDomainEdge,
    EditDomainFace,
}

pub struct Binding {
    pub key: KeyCode,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub action: Action,
    pub category: &'static str,
    pub label: &'static str,
}

/// Mouse / modal-mode shortcuts that are not simple key bindings (shown in the shortcuts window).
pub const EXTRA: &[(&str, &str, &str)] = &[
    ("Mouse", "Left click", "Select object (empty space = deselect)"),
    ("Mouse", "Shift + Left click", "Add to selection / toggle / make active"),
    ("Mouse", "Left drag (Select tool)", "Orbit the view"),
    ("Mouse", "Middle drag", "Orbit"),
    ("Mouse", "Shift + Middle drag", "Pan"),
    ("Mouse", "Right drag", "Pan"),
    ("Mouse", "Shift + Right click", "Place the 3D cursor"),
    ("Mouse", "Mouse wheel", "Zoom (Circle select tool: change brush radius)"),
    ("Select tools", "Box tool: drag", "Select everything inside the rectangle"),
    ("Select tools", "Lasso tool: drag", "Select everything inside the drawn shape"),
    ("Select tools", "Circle tool: paint", "Select everything under the brush"),
    ("Select tools", "Shift / Ctrl while selecting", "Extend / subtract from the selection"),
    ("Gizmo tools", "Drag a colored handle", "Move / rotate / scale along that axis"),
    ("Gizmo tools", "Ctrl while dragging", "Toggle snapping for this drag"),
    ("While moving / rotating / scaling (G R S)", "Mouse", "Transform the selection"),
    ("While moving / rotating / scaling (G R S)", "X / Y / Z", "Constrain to axis (press again: local axis, again: free)"),
    ("While moving / rotating / scaling (G R S)", "Shift + X / Y / Z", "Constrain to plane (move only)"),
    ("While moving / rotating / scaling (G R S)", "0-9 . - Backspace", "Type an exact value (distance, degrees, factor)"),
    ("While moving / rotating / scaling (G R S)", "Ctrl", "Toggle snapping while held"),
    ("While moving / rotating / scaling (G R S)", "Enter or Left click", "Confirm"),
    ("While moving / rotating / scaling (G R S)", "Esc or Right click", "Cancel"),
];

pub fn bindings() -> Vec<Binding> {
    use Action as A;
    use KeyCode as K;
    let mut v: Vec<Binding> = Vec::new();
    // mods: any of 'c' (Ctrl / Cmd), 's' (Shift), 'a' (Alt)
    let mut b = |category: &'static str, label: &'static str, mods: &str, key: KeyCode, action: Action| {
        v.push(Binding { key, ctrl: mods.contains('c'), shift: mods.contains('s'), alt: mods.contains('a'), action, category, label });
    };
    b("File", "New scene", "c", K::KeyN, A::New);
    b("File", "Open project", "c", K::KeyO, A::Open);
    b("File", "Save", "c", K::KeyS, A::Save);
    b("File", "Save as", "cs", K::KeyS, A::SaveAs);
    b("File", "Quit", "c", K::KeyQ, A::Quit);

    b("Edit", "Undo", "c", K::KeyZ, A::Undo);
    b("Edit", "Redo", "cs", K::KeyZ, A::Redo);
    b("Edit", "Redo", "c", K::KeyY, A::Redo);
    b("Edit", "Duplicate (then move)", "s", K::KeyD, A::Duplicate);
    b("Edit", "Duplicate linked (then move)", "a", K::KeyD, A::DuplicateLinked);
    b("Edit", "Delete", "", K::KeyX, A::Delete);
    b("Edit", "Delete", "", K::Delete, A::Delete);
    b("Edit", "Extrude", "", K::KeyE, A::Extrude);
    b("Edit", "Inset Faces", "", K::KeyI, A::InsetFaces);
    b("Edit", "Bevel", "c", K::KeyB, A::Bevel);
    b("Edit", "Loop Cut", "c", K::KeyR, A::LoopCut);
    b("Edit", "Knife bisect", "", K::KeyK, A::Knife);
    b("Edit", "Shrink/Fatten", "a", K::KeyS, A::ShrinkFatten);
    b("Edit", "Smooth verts", "", K::KeyO, A::SmoothVerts);
    b("Edit", "Rip Region", "", K::KeyV, A::RipRegion);
    b("Edit", "Poly Build", "", K::KeyP, A::PolyBuild);
    b("Edit", "Spin", "a", K::KeyE, A::Spin);
    b("Edit", "Merge verts", "", K::KeyM, A::MergeVerts);
    b("Edit", "Rename active object", "", K::F2, A::Rename);

    b("Select", "Select all", "", K::KeyA, A::SelectAll);
    b("Select", "Deselect all", "a", K::KeyA, A::SelectNone);
    b("Select", "Invert selection", "c", K::KeyI, A::SelectInvert);
    b("Select", "Select children (recursive)", "s", K::BracketRight, A::SelectChildren);
    b("Select", "Select parent", "s", K::BracketLeft, A::SelectParent);
    b("Select", "Select linked duplicates", "c", K::KeyL, A::SelectLinked);

    b("Tools", "Select tool", "", K::KeyQ, A::SetTool(Tool::Select));
    b("Tools", "Box select tool", "", K::KeyB, A::SetTool(Tool::BoxSelect));
    b("Tools", "Circle select tool", "", K::KeyC, A::SetTool(Tool::Circle));
    b("Tools", "Lasso select tool", "", K::KeyL, A::SetTool(Tool::Lasso));
    b("Tools", "Cycle gizmo tool (Move / Rotate / Scale)", "", K::KeyT, A::CycleTool);
    b("Tools", "Toggle snapping", "s", K::Tab, A::ToggleSnap);
    b("Mode", "Toggle Edit Mode", "", K::Tab, A::ToggleEditMode);
    b("Mode", "Toggle Sculpt Mode", "c", K::Tab, A::ToggleSculptMode);
    b("Mode", "Vertex select", "", K::Digit1, A::EditDomainVertex);
    b("Mode", "Edge select", "", K::Digit2, A::EditDomainEdge);
    b("Mode", "Face select", "", K::Digit3, A::EditDomainFace);

    b("Transform", "Move (modal)", "", K::KeyG, A::ModalMove);
    b("Transform", "Rotate (modal)", "", K::KeyR, A::ModalRotate);
    b("Transform", "Scale (modal)", "", K::KeyS, A::ModalScale);
    b("Transform", "Clear location", "a", K::KeyG, A::ClearLocation);
    b("Transform", "Clear rotation", "a", K::KeyR, A::ClearRotation);
    b("Transform", "Clear scale", "a", K::KeyS, A::ClearScale);
    b("Transform", "Apply all transforms", "c", K::KeyA, A::Apply { loc: true, rot: true, scale: true });
    b("Transform", "Cursor to world origin", "s", K::KeyC, A::SnapCursorToOrigin);

    b("Object", "Hide selected", "", K::KeyH, A::Hide);
    b("Object", "Hide unselected", "s", K::KeyH, A::HideUnselected);
    b("Object", "Show all", "a", K::KeyH, A::ShowAll);
    b("Object", "Parent (selected to active)", "c", K::KeyP, A::ParentSet);
    b("Object", "Clear parent, keep transform", "a", K::KeyP, A::ParentClearKeep);
    b("Object", "Clear parent", "as", K::KeyP, A::ParentClear);
    b("Object", "Move to collection", "", K::KeyM, A::OpenMoveToCollection);
    b("Object", "New collection", "c", K::KeyG, A::NewCollection);
    b("Object", "Join selected meshes", "c", K::KeyJ, A::Join);
    b("Object", "Shade Smooth", "", K::F3, A::ShadeSmooth);
    b("Object", "Shade Flat", "s", K::F3, A::ShadeFlat);

    b("View", "Frame selected", "", K::KeyF, A::FrameSelected);
    b("View", "Frame all", "", K::Home, A::FrameAll);
    b("View", "Look through camera", "", K::Digit0, A::CameraView);
    b("View", "Shading: Rendered", "", K::Digit1, A::SetShading(ShadingMode::Rendered));
    b("View", "Shading: Albedo", "", K::Digit2, A::SetShading(ShadingMode::Albedo));
    b("View", "Shading: Normals", "", K::Digit3, A::SetShading(ShadingMode::Normals));
    b("View", "Shading: Roughness", "", K::Digit4, A::SetShading(ShadingMode::Roughness));
    b("View", "Shading: Metallic", "", K::Digit5, A::SetShading(ShadingMode::Metallic));
    b("View", "Shading: UV checker", "", K::Digit6, A::SetShading(ShadingMode::UvChecker));
    b("View", "Cycle wireframe (off / overlay / only)", "s", K::KeyZ, A::CycleWire);
    b("View", "Toggle environment background", "s", K::KeyB, A::ToggleBackground);
    b("View", "Exposure down", "", K::Minus, A::ExposureDown);
    b("View", "Exposure up", "", K::Equal, A::ExposureUp);
    b("View", "Rotate environment left", "", K::BracketLeft, A::EnvRotate(-1));
    b("View", "Rotate environment right", "", K::BracketRight, A::EnvRotate(1));

    b("Help", "Show / hide this shortcuts window", "", K::F1, A::ToggleShortcuts);
    v
}

pub fn key_name(k: KeyCode) -> String {
    match k {
        KeyCode::Minus => "-".into(),
        KeyCode::Equal => "=".into(),
        KeyCode::BracketLeft => "[".into(),
        KeyCode::BracketRight => "]".into(),
        KeyCode::Slash => "/".into(),
        KeyCode::Escape => "Esc".into(),
        KeyCode::Numpad0 => "Numpad 0".into(),
        _ => {
            let d = format!("{k:?}");
            d.strip_prefix("Key").or_else(|| d.strip_prefix("Digit")).unwrap_or(&d).to_string()
        }
    }
}

pub fn describe(b: &Binding) -> String {
    let mut s = String::new();
    if b.ctrl {
        s.push_str("Ctrl+");
    }
    if b.alt {
        s.push_str("Alt+");
    }
    if b.shift {
        s.push_str("Shift+");
    }
    s.push_str(&key_name(b.key));
    s
}

/// Shortcut text for a menu item (empty if the action has no key).
pub fn shortcut_text(bindings: &[Binding], a: &Action) -> String {
    bindings.iter().find(|b| &b.action == a).map(describe).unwrap_or_default()
}

pub fn find_action(bindings: &[Binding], ctrl: bool, shift: bool, alt: bool, key: KeyCode) -> Option<Action> {
    bindings.iter().find(|b| b.key == key && b.ctrl == ctrl && b.shift == shift && b.alt == alt).map(|b| b.action.clone())
}
