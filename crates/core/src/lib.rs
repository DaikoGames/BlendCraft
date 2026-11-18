//! Platform-independent core: no windowing, no GPU. Compiles to WASM unchanged.
pub mod animation;
pub mod armature;
pub mod camera;
pub mod cloth;
pub mod command;
pub mod compositor;
pub mod geometry_nodes;
pub mod constraints;
pub mod document;
mod edit;
pub mod environment;
pub mod fluid;
pub mod fonts;
pub mod kinds;
pub mod material;
pub mod meshcache;
pub mod mesh;
pub mod modifiers;
pub mod primitives;
pub mod physics;
pub mod project;
pub mod scene;
pub mod sculpt;
pub mod simulation;
pub mod shader_graph;
pub mod uv;

pub use animation::{AnimationData, ObjectAnimation, TransformKey};
pub use armature::{ArmatureData, Bone, SkinBinding};
pub use camera::{FixedView, OrbitCamera};
pub use cloth::{ClothBaker, ClothPin, FabricPreset, FabricType};
pub use command::{Command, History};
pub use compositor::{CompositorGraph, CompositorLink, CompositorNode, CompositorNodeKind};
pub use geometry_nodes::{
    GeoLink, GeoNode, GeoNodeKind, GeoSocketType, GeometryNodeGraph,
};
pub use constraints::{Constraint, ConstraintKind};
pub use document::{AlignMode, AlignTo, Document, OriginMode, SameKind};
pub use edit::{edge_key, world_positions, EditDomain, EditPick, EditSelection, EditSession};
pub use environment::{sh_basis, EnvironmentMap};
pub use fluid::{
    apply_domain_display, domain_material_index, fluid_bake_frames, free_fluid_bake,
    mesh_from_particles, migrate_legacy_fluids, scene_at_frame, FlowBehavior, FluidBaker,
    FluidDomain, FluidEffector, FluidFlow, FluidObject, FluidSolver, FluidType,
};
pub use meshcache::{apply_mesh_caches, MeshSequence};
pub use fonts::{
    font_registry, free_font_catalog, mesh_from_font, system_font_dirs, CatalogFont, FontRegistry,
};
pub use kinds::{
    helper_lines, helper_radius, CameraData, EmptyShape, ForceFieldData, ForceFieldType,
    GreasePencilData, LightData, LightProbeData, LightProbeType, LightType, ObjectKind,
    SpeakerData, SurfaceData, VolumeData,
};
pub use material::{AlphaMode, Material, TextureData};
pub use mesh::{Aabb, MeshData, MeshGroup, Vertex};
pub use modifiers::{
    evaluate_object_modifiers, evaluate_stack, BooleanOperation, Modifier, ModifierKind,
};
pub use primitives::{Primitive, TextFont};
pub use physics::{
    apply_baked_frame, bake_physics_to_keyframes, bake_scene, CollisionShape, PhysicsCache, PhysicsRuntime,
    PhysicsTransform, PhysicsWorldSettings, RigidBody, RigidBodyConstraint,
    RigidBodyConstraintType, RigidBodyType,
};
pub use scene::{Collection, Object, Scene};
pub use simulation::{
    bake_non_rigid_simulations, free_sim_bake, promote_legacy_fluid_components, ClothSettings,
    FluidSettings, ObjectSimulation, ParticleSettings, SimulationSettings, SoftBodySettings,
};
pub use sculpt::{apply_brush, dyntopo_refine, raycast_mesh, SculptBrush, SculptSettings};
pub use shader_graph::{
    apply_eval_to_material, bake_graph_texture, color_ramp_eval_stops, color_ramp_from_node,
    evaluate_graph, evaluate_graph_ex, evaluate_surface_shading, ramp_get_stops,
    ramp_hue_interpolation, ramp_set_stops, ramp_stop_count, ColorRampStop, ShaderGraph, ShaderLink,
    ShaderNode, ShaderNodeKind, SocketDef, SocketType, COLOR_RAMP_MAX_STOPS,
};
pub use uv::{
    normalize_uvs, pack_islands, pick_uv, unwrap_cube, unwrap_planar, unwrap_reset, unwrap_smart,
    uv_islands, uv_rotate, uv_scale, uv_selection_center, uv_translate,
};
