//! Platform-independent core: no windowing, no GPU. Compiles to WASM unchanged.
pub mod camera;
pub mod command;
pub mod constraints;
pub mod document;
pub mod environment;
pub mod animation;
pub mod armature;
pub mod kinds;
pub mod material;
pub mod shader_graph;
pub mod mesh;
mod edit;
pub mod sculpt;
pub mod uv;
pub mod primitives;
pub mod fonts;
pub mod project;
pub mod scene;

pub use camera::{FixedView, OrbitCamera};
pub use animation::{AnimationData, ObjectAnimation, TransformKey};
pub use armature::{ArmatureData, Bone, SkinBinding};
pub use kinds::{helper_lines, helper_radius, CameraData, EmptyShape, ForceFieldData, ForceFieldType, GreasePencilData, LightData, LightProbeData, LightProbeType, LightType, ObjectKind, SpeakerData, SurfaceData, VolumeData};
pub use command::{Command, History};
pub use constraints::{Constraint, ConstraintKind};
pub use document::{AlignMode, AlignTo, Document, OriginMode, SameKind};
pub use environment::{sh_basis, EnvironmentMap};
pub use material::{AlphaMode, Material, TextureData};
pub use shader_graph::{apply_eval_to_material, bake_graph_texture, color_ramp_eval_stops, color_ramp_from_node, evaluate_graph, evaluate_graph_ex, ramp_get_stops, ramp_stop_count, COLOR_RAMP_MAX_STOPS, ShaderGraph, ShaderLink, ShaderNode, ShaderNodeKind, SocketDef, SocketType};
pub use mesh::{Aabb, MeshData, MeshGroup, Vertex};
pub use primitives::{Primitive, TextFont};
pub use fonts::{font_registry, free_font_catalog, mesh_from_font, system_font_dirs, FontRegistry, CatalogFont};
pub use edit::{EditDomain, EditPick, EditSelection, EditSession, edge_key, world_positions};
pub use sculpt::{apply_brush, dyntopo_refine, raycast_mesh, SculptBrush, SculptSettings};
pub use uv::{
    normalize_uvs, pack_islands, pick_uv, unwrap_cube, unwrap_planar, unwrap_reset, unwrap_smart,
    uv_islands, uv_rotate, uv_scale, uv_selection_center, uv_translate,
};
pub use scene::{Collection, Object, Scene};
