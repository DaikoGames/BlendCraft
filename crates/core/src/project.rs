//! Native project format (.forge): magic + zlib(bincode(Scene)).
use crate::{
    AnimationData, Collection, ColorRampStop, Material, Object, Scene, ShaderGraph, ShaderLink,
    ShaderNode, ShaderNodeKind, TextureData, CompositorGraph, RigidBody,
};
use anyhow::{anyhow, bail, Result};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"FRGA";
const V9_MAGIC: &[u8; 4] = b"FRG9";
const V8_MAGIC: &[u8; 4] = b"FRG8";
const V7_MAGIC: &[u8; 4] = b"FRG7";
const V6_MAGIC: &[u8; 4] = b"FRG6";
const V5_MAGIC: &[u8; 4] = b"FRG5";
const V4_MAGIC: &[u8; 4] = b"FRG4";
const LEGACY_MAGIC: &[u8; 4] = b"FRG3";



/// Cloth settings as stored by FRG7-FRG9 (before fabrics, collisions and XPBD).
#[derive(Serialize, Deserialize)]
struct V9ClothSettings {
    enabled: bool, quality_steps: u32, vertex_mass: f32, tension_stiffness: f32,
    bending_stiffness: f32, damping: f32, air_damping: f32, gravity_scale: f32, pin_boundary: bool,
}
impl From<V9ClothSettings> for crate::ClothSettings {
    fn from(old: V9ClothSettings) -> Self {
        // The old unit-less stiffness values do not map onto the physical ones, so migrated
        // cloth keeps the default (cotton) material marked as Custom and the old behaviour flags.
        let mut s = crate::ClothSettings::default();
        s.enabled = old.enabled;
        s.quality_steps = old.quality_steps.clamp(1, 64);
        s.damping = old.damping.clamp(0.0, 1.0);
        s.air_damping = old.air_damping.max(0.0);
        s.gravity_scale = old.gravity_scale;
        s.pin = if old.pin_boundary { crate::ClothPin::Boundary } else { crate::ClothPin::None };
        s.fabric = crate::FabricType::Custom;
        s
    }
}
#[derive(Serialize, Deserialize)]
struct V9SimulationSettings {
    cloth: V9ClothSettings,
    soft_body: crate::SoftBodySettings,
    fluid: crate::FluidSettings,
    particles: crate::ParticleSettings,
}
impl From<V9SimulationSettings> for crate::SimulationSettings {
    fn from(old: V9SimulationSettings) -> Self {
        Self { cloth: old.cloth.into(), soft_body: old.soft_body, fluid: old.fluid, particles: old.particles }
    }
}
#[derive(Serialize, Deserialize)]
enum V9ObjectSimulation {
    Cloth(V9ClothSettings),
    SoftBody(crate::SoftBodySettings),
    Fluid(crate::FluidSettings),
    Particles(crate::ParticleSettings),
}
impl From<V9ObjectSimulation> for crate::ObjectSimulation {
    fn from(old: V9ObjectSimulation) -> Self {
        match old {
            V9ObjectSimulation::Cloth(c) => crate::ObjectSimulation::Cloth(c.into()),
            V9ObjectSimulation::SoftBody(c) => crate::ObjectSimulation::SoftBody(c),
            V9ObjectSimulation::Fluid(c) => crate::ObjectSimulation::Fluid(c),
            V9ObjectSimulation::Particles(c) => crate::ObjectSimulation::Particles(c),
        }
    }
}
fn upgrade_object_sims(
    old: std::collections::HashMap<u64, V9ObjectSimulation>,
) -> std::collections::HashMap<u64, crate::ObjectSimulation> {
    old.into_iter().map(|(k, v)| (k, v.into())).collect()
}

/// Previous current layout (FRG9), before fabric presets / XPBD cloth settings.
#[derive(Serialize, Deserialize)]
struct Version9Scene {
    objects: Vec<Object>, materials: Vec<Material>, textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>, cursor: Vec3, next_id: u64, animation: AnimationData,
    compositor_graph: CompositorGraph,
    rigid_bodies: std::collections::HashMap<u64, crate::RigidBody>,
    physics_world: crate::PhysicsWorldSettings,
    rigid_body_constraints: Vec<crate::RigidBodyConstraint>,
    physics_cache: Option<crate::PhysicsCache>,
    simulations: V9SimulationSettings,
    object_simulations: std::collections::HashMap<u64, V9ObjectSimulation>,
    fluids: std::collections::HashMap<u64, crate::FluidObject>,
    mesh_caches: std::collections::HashMap<u64, crate::MeshSequence>,
}
impl From<Version9Scene> for Scene {
    fn from(scene: Version9Scene) -> Self {
        Self { objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: scene.rigid_bodies, physics_world: scene.physics_world,
            rigid_body_constraints: scene.rigid_body_constraints, physics_cache: scene.physics_cache,
            simulations: scene.simulations.into(),
            object_simulations: upgrade_object_sims(scene.object_simulations),
            fluids: scene.fluids, mesh_caches: scene.mesh_caches, type_hidden: 0, type_unselectable: 0, }
    }
}

/// Previous current layout (FRG8), before water physics and mesh sequence caches.
#[derive(Serialize, Deserialize)]
struct Version8Scene {
    objects: Vec<Object>, materials: Vec<Material>, textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>, cursor: Vec3, next_id: u64, animation: AnimationData,
    compositor_graph: CompositorGraph,
    rigid_bodies: std::collections::HashMap<u64, crate::RigidBody>,
    physics_world: crate::PhysicsWorldSettings,
    rigid_body_constraints: Vec<crate::RigidBodyConstraint>,
    physics_cache: Option<crate::PhysicsCache>,
    simulations: V9SimulationSettings,
    object_simulations: std::collections::HashMap<u64, V9ObjectSimulation>,
}
impl From<Version8Scene> for Scene {
    fn from(scene: Version8Scene) -> Self {
        Self { objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: scene.rigid_bodies, physics_world: scene.physics_world,
            rigid_body_constraints: scene.rigid_body_constraints, physics_cache: scene.physics_cache,
            simulations: scene.simulations.into(), object_simulations: upgrade_object_sims(scene.object_simulations),
            fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0, }
    }
}

/// Previous current layout (FRG7), before per-object simulation components.
#[derive(Serialize, Deserialize)]
struct Version7Scene {
    objects: Vec<Object>, materials: Vec<Material>, textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>, cursor: Vec3, next_id: u64, animation: AnimationData,
    compositor_graph: CompositorGraph,
    rigid_bodies: std::collections::HashMap<u64, crate::RigidBody>,
    physics_world: crate::PhysicsWorldSettings,
    rigid_body_constraints: Vec<crate::RigidBodyConstraint>,
    physics_cache: Option<crate::PhysicsCache>,
    simulations: V9SimulationSettings,
}
impl From<Version7Scene> for Scene {
    fn from(scene: Version7Scene) -> Self {
        Self { objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: scene.rigid_bodies, physics_world: scene.physics_world,
            rigid_body_constraints: scene.rigid_body_constraints, physics_cache: scene.physics_cache,
            simulations: scene.simulations.into(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0, }
    }
}

/// Previous current layout (FRG6), before non-rigid simulation settings.
#[derive(Serialize, Deserialize)]
struct Version6Scene {
    objects: Vec<Object>, materials: Vec<Material>, textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>, cursor: Vec3, next_id: u64, animation: AnimationData,
    compositor_graph: CompositorGraph,
    rigid_bodies: std::collections::HashMap<u64, crate::RigidBody>,
    physics_world: crate::PhysicsWorldSettings,
    rigid_body_constraints: Vec<crate::RigidBodyConstraint>,
    physics_cache: Option<crate::PhysicsCache>,
}
impl From<Version6Scene> for Scene {
    fn from(scene: Version6Scene) -> Self {
        Self { objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: scene.rigid_bodies, physics_world: scene.physics_world,
            rigid_body_constraints: scene.rigid_body_constraints, physics_cache: scene.physics_cache,
            simulations: Default::default(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0, }
    }
}

/// Previous project layout. Keeping this decoder lets existing FRG5 projects
/// load after the full rigid-body world and bake data were added to Scene.
#[derive(Serialize, Deserialize)]
struct Version5Scene {
    objects: Vec<Object>,
    materials: Vec<Material>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
    compositor_graph: CompositorGraph,
    rigid_bodies: std::collections::HashMap<u64, Version5RigidBody>,
}

#[derive(Serialize, Deserialize)]
struct Version5RigidBody {
    body_type: crate::RigidBodyType,
    collision_shape: crate::CollisionShape,
    mass: f32,
    friction: f32,
    restitution: f32,
    linear_damping: f32,
    enabled: bool,
}

impl From<Version5Scene> for Scene {
    fn from(scene: Version5Scene) -> Self {
        Self {
            objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: scene.rigid_bodies.into_iter().map(|(id, b)| (id, RigidBody {
                body_type: b.body_type, collision_shape: b.collision_shape, mass: b.mass,
                friction: b.friction, restitution: b.restitution, linear_damping: b.linear_damping,
                enabled: b.enabled, ..RigidBody::default()
            })).collect(), physics_world: Default::default(),
            rigid_body_constraints: Vec::new(), physics_cache: None, simulations: Default::default(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Version4Scene {
    objects: Vec<Object>,
    materials: Vec<Material>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
    compositor_graph: CompositorGraph,
}

impl From<Version4Scene> for Scene {
    fn from(scene: Version4Scene) -> Self {
        Self {
            objects: scene.objects, materials: scene.materials, textures: scene.textures,
            collections: scene.collections, cursor: scene.cursor, next_id: scene.next_id,
            animation: scene.animation, compositor_graph: scene.compositor_graph,
            rigid_bodies: Default::default(), physics_world: Default::default(),
            rigid_body_constraints: Vec::new(), physics_cache: None, simulations: Default::default(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Version3Scene {
    objects: Vec<Object>,
    materials: Vec<Version3Material>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
}

#[derive(Serialize, Deserialize)]
struct Version3Material {
    name: String,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    alpha_mode: crate::AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    normal_scale: f32,
    occlusion_strength: f32,
    base_color_tex: Option<usize>,
    metallic_roughness_tex: Option<usize>,
    normal_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    shader_graph: Option<Version3ShaderGraph>,
    displacement_scale: f32,
    volume_density: f32,
}

#[derive(Serialize, Deserialize)]
struct Version3ShaderGraph {
    nodes: Vec<Version3ShaderNode>,
    links: Vec<ShaderLink>,
    next_id: u64,
}

#[derive(Serialize, Deserialize)]
struct Version3ShaderNode {
    id: u64,
    kind: ShaderNodeKind,
    pos: [f32; 2],
    image_index: Option<usize>,
    value: f32,
    color: [f32; 4],
    props: [f32; 16],
    input_defaults: Vec<[f32; 4]>,
    mode: u8,
    name_prop: String,
    ui_size: [f32; 2],
    ramp_stops: Vec<ColorRampStop>,
}

impl From<Version3Scene> for Scene {
    fn from(scene: Version3Scene) -> Self {
        Self {
            objects: scene.objects,
            materials: scene
                .materials
                .into_iter()
                .map(|material| Material {
                    name: material.name,
                    base_color: material.base_color,
                    metallic: material.metallic,
                    roughness: material.roughness,
                    emissive: material.emissive,
                    alpha_mode: material.alpha_mode,
                    alpha_cutoff: material.alpha_cutoff,
                    double_sided: material.double_sided,
                    normal_scale: material.normal_scale,
                    occlusion_strength: material.occlusion_strength,
                    base_color_tex: material.base_color_tex,
                    metallic_roughness_tex: material.metallic_roughness_tex,
                    normal_tex: material.normal_tex,
                    occlusion_tex: material.occlusion_tex,
                    emissive_tex: material.emissive_tex,
                    shader_graph: material.shader_graph.map(|graph| ShaderGraph {
                        nodes: graph
                            .nodes
                            .into_iter()
                            .map(|node| ShaderNode {
                                id: node.id,
                                kind: node.kind,
                                pos: node.pos,
                                image_index: node.image_index,
                                value: node.value,
                                color: node.color,
                                props: node.props,
                                input_defaults: node.input_defaults,
                                mode: node.mode,
                                name_prop: node.name_prop,
                                ui_size: node.ui_size,
                                ramp_stops: node.ramp_stops,
                            })
                            .collect(),
                        links: graph.links,
                        next_id: graph.next_id,
                    }),
                    displacement_scale: material.displacement_scale,
                    volume_density: material.volume_density,
                })
                .collect(),
            textures: scene.textures,
            collections: scene.collections,
            cursor: scene.cursor,
            next_id: scene.next_id,
            animation: scene.animation,
            compositor_graph: Default::default(),
            rigid_bodies: Default::default(), physics_world: Default::default(),
            rigid_body_constraints: Vec::new(), physics_cache: None, simulations: Default::default(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct LegacyScene {
    objects: Vec<Object>,
    materials: Vec<LegacyMaterial>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
}

#[derive(Serialize, Deserialize)]
struct LegacyMaterial {
    name: String,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    alpha_mode: crate::AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    normal_scale: f32,
    occlusion_strength: f32,
    base_color_tex: Option<usize>,
    metallic_roughness_tex: Option<usize>,
    normal_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    shader_graph: Option<LegacyShaderGraph>,
    displacement_scale: f32,
    volume_density: f32,
}

#[derive(Serialize, Deserialize)]
struct LegacyShaderGraph {
    nodes: Vec<LegacyShaderNode>,
    links: Vec<ShaderLink>,
    next_id: u64,
}

#[derive(Serialize, Deserialize)]
struct LegacyShaderNode {
    id: u64,
    kind: ShaderNodeKind,
    pos: [f32; 2],
    image_index: Option<usize>,
    value: f32,
    color: [f32; 4],
    props: [f32; 16],
    input_defaults: Vec<[f32; 4]>,
    mode: u8,
    name_prop: String,
}

impl From<LegacyShaderNode> for ShaderNode {
    fn from(node: LegacyShaderNode) -> Self {
        Self {
            id: node.id,
            kind: node.kind,
            pos: node.pos,
            image_index: node.image_index,
            value: node.value,
            color: node.color,
            props: node.props,
            input_defaults: node.input_defaults,
            mode: node.mode,
            name_prop: node.name_prop,
            ui_size: [0.0; 2],
            ramp_stops: Vec::new(),
        }
    }
}

impl From<LegacyScene> for Scene {
    fn from(scene: LegacyScene) -> Self {
        Self {
            objects: scene.objects,
            materials: scene
                .materials
                .into_iter()
                .map(|material| Material {
                    name: material.name,
                    base_color: material.base_color,
                    metallic: material.metallic,
                    roughness: material.roughness,
                    emissive: material.emissive,
                    alpha_mode: material.alpha_mode,
                    alpha_cutoff: material.alpha_cutoff,
                    double_sided: material.double_sided,
                    normal_scale: material.normal_scale,
                    occlusion_strength: material.occlusion_strength,
                    base_color_tex: material.base_color_tex,
                    metallic_roughness_tex: material.metallic_roughness_tex,
                    normal_tex: material.normal_tex,
                    occlusion_tex: material.occlusion_tex,
                    emissive_tex: material.emissive_tex,
                    shader_graph: material.shader_graph.map(|graph| ShaderGraph {
                        nodes: graph.nodes.into_iter().map(ShaderNode::from).collect(),
                        links: graph.links,
                        next_id: graph.next_id,
                    }),
                    displacement_scale: material.displacement_scale,
                    volume_density: material.volume_density,
                })
                .collect(),
            textures: scene.textures,
            collections: scene.collections,
            cursor: scene.cursor,
            next_id: scene.next_id,
            animation: scene.animation,
            compositor_graph: Default::default(),
            rigid_bodies: Default::default(), physics_world: Default::default(),
            rigid_body_constraints: Vec::new(), physics_cache: None, simulations: Default::default(), object_simulations: Default::default(), fluids: Default::default(), mesh_caches: Default::default(), type_hidden: 0, type_unselectable: 0,
        }
    }
}

pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>> {
    let raw = bincode::serialize(scene)?;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(&raw)?;
    let mut out = MAGIC.to_vec();
    out.extend(enc.finish()?);
    Ok(out)
}

pub fn from_bytes(bytes: &[u8]) -> Result<Scene> {
    if bytes.len() < 4 || (&bytes[..4] != MAGIC && &bytes[..4] != V9_MAGIC && &bytes[..4] != V8_MAGIC && &bytes[..4] != V7_MAGIC && &bytes[..4] != V6_MAGIC && &bytes[..4] != V5_MAGIC && &bytes[..4] != V4_MAGIC && &bytes[..4] != LEGACY_MAGIC) {
        bail!("not a Forge3D project file");
    }
    let is_current = &bytes[..4] == MAGIC;
    let is_v9 = &bytes[..4] == V9_MAGIC;
    let is_v8 = &bytes[..4] == V8_MAGIC;
    let is_v7 = &bytes[..4] == V7_MAGIC;
    let is_v6 = &bytes[..4] == V6_MAGIC;
    let is_v5 = &bytes[..4] == V5_MAGIC;
    let is_v4 = &bytes[..4] == V4_MAGIC;
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(&bytes[4..]).read_to_end(&mut raw)?;
    if is_current { return Ok(bincode::deserialize(&raw)?); }
    if is_v9 {
        let scene: Version9Scene = bincode::deserialize(&raw)?;
        return Ok(Scene::from(scene));
    }
    if is_v8 {
        let scene: Version8Scene = bincode::deserialize(&raw)?;
        let mut scene = Scene::from(scene);
        crate::fluid::migrate_legacy_fluids(&mut scene);
        return Ok(scene);
    }
    if is_v7 {
        let scene: Version7Scene = bincode::deserialize(&raw)?;
        return Ok(Scene::from(scene));
    }
    if is_v6 {
        let scene: Version6Scene = bincode::deserialize(&raw)?;
        return Ok(Scene::from(scene));
    }
    if is_v5 {
        let scene: Version5Scene = bincode::deserialize(&raw)?;
        return Ok(Scene::from(scene));
    }
    if is_v4 {
        let scene: Version4Scene = bincode::deserialize(&raw)?;
        return Ok(Scene::from(scene));
    }
    let version3_error = match bincode::deserialize::<Version3Scene>(&raw) {
        Ok(scene) => return Ok(Scene::from(scene)),
        Err(error) => error,
    };
    bincode::deserialize::<LegacyScene>(&raw)
        .map(Scene::from)
        .map_err(|legacy_error| {
            anyhow!("could not decode FRG3 project: {version3_error}; {legacy_error}")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[test]
    fn uv_coordinates_survive_project_roundtrip() {
        let mut d = Document::new();
        let obj = &mut d.scene.objects[0];
        let mesh = std::sync::Arc::make_mut(&mut obj.mesh);
        mesh.vertices[0].uv = [0.123, 0.875];
        let bytes = to_bytes(&d.scene).unwrap();
        assert_eq!(&bytes[..4], MAGIC);
        let loaded = from_bytes(&bytes).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.123).abs() < 1e-6);
        assert!((uv[1] - 0.875).abs() < 1e-6);
    }

    #[test]
    fn active_edit_session_uvs_are_committed_before_project_save() {
        let mut d = Document::new();
        d.select_only(0);
        d.enter_edit_mode().unwrap();
        d.edit.as_mut().unwrap().mesh.vertices[0].uv = [0.314, 0.728];
        // This is the same commit performed by the application save command.
        d.sync_edit_mesh();
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.314).abs() < 1e-6);
        assert!((uv[1] - 0.728).abs() < 1e-6);
    }

    #[test]
    fn painted_texture_survives_project_roundtrip() {
        let mut d = Document::new();
        d.select_only(0);
        d.uv_unwrap_smart();
        d.enter_paint_mode().unwrap();
        assert!(d.paint_at_uv(glam::Vec2::new(0.5, 0.5), [1.0, 0.0, 0.0, 1.0], 0.025, 1.0));
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let material = &loaded.materials[loaded.objects[0].material.unwrap()];
        let tex = &loaded.textures[material.base_color_tex.unwrap()];
        let center = ((tex.height / 2 * tex.width + tex.width / 2) * 4) as usize;
        assert!(tex.rgba8[center] > 240);
        assert!(tex.rgba8[center + 1] < 20);
        assert!(tex.rgba8[center + 2] < 20);
    }

    #[test]
    fn frg9_projects_with_old_cloth_settings_still_load() {
        let mut d = Document::new();
        let id = 7;
        // Hand-build an FRG9 file: the old cloth layout lives inside Scene.
        let v9 = Version9Scene {
            objects: d.scene.objects.clone(), materials: d.scene.materials.clone(),
            textures: d.scene.textures.clone(), collections: d.scene.collections.clone(),
            cursor: d.scene.cursor, next_id: d.scene.next_id, animation: d.scene.animation.clone(),
            compositor_graph: d.scene.compositor_graph.clone(),
            rigid_bodies: Default::default(), physics_world: d.scene.physics_world.clone(),
            rigid_body_constraints: Vec::new(), physics_cache: None,
            simulations: V9SimulationSettings {
                cloth: V9ClothSettings { enabled: true, quality_steps: 9, vertex_mass: 0.3,
                    tension_stiffness: 15.0, bending_stiffness: 0.4, damping: 0.02,
                    air_damping: 0.1, gravity_scale: 1.0, pin_boundary: true },
                soft_body: Default::default(), fluid: Default::default(), particles: Default::default(),
            },
            object_simulations: std::iter::once((id, V9ObjectSimulation::Cloth(V9ClothSettings {
                enabled: true, quality_steps: 13, vertex_mass: 0.3, tension_stiffness: 15.0,
                bending_stiffness: 0.4, damping: 0.02, air_damping: 0.1, gravity_scale: 1.0,
                pin_boundary: false }))).collect(),
            fluids: Default::default(), mesh_caches: Default::default(),
        };
        let raw = bincode::serialize(&v9).unwrap();
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&raw).unwrap();
        let mut bytes = V9_MAGIC.to_vec();
        bytes.extend(enc.finish().unwrap());
        let loaded = from_bytes(&bytes).unwrap();
        assert!(loaded.simulations.cloth.enabled);
        assert_eq!(loaded.simulations.cloth.quality_steps, 9);
        assert_eq!(loaded.simulations.cloth.pin, crate::ClothPin::Boundary);
        match loaded.object_simulations.get(&id).unwrap() {
            crate::ObjectSimulation::Cloth(c) => {
                assert_eq!(c.quality_steps, 13);
                assert_eq!(c.pin, crate::ClothPin::None);
                assert_eq!(c.fabric, crate::FabricType::Custom);
            }
            _ => panic!("wrong simulation component type"),
        }
        let _ = &mut d;
    }

    #[test]
    fn object_simulation_components_survive_project_roundtrip() {
        let mut d = Document::new();
        let id = 42;
        d.scene.object_simulations.insert(id, crate::ObjectSimulation::Cloth(
            crate::ClothSettings { enabled: true, quality_steps: 17, ..Default::default() }
        ));
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        match loaded.object_simulations.get(&id).unwrap() {
            crate::ObjectSimulation::Cloth(settings) => {
                assert!(settings.enabled);
                assert_eq!(settings.quality_steps, 17);
            }
            _ => panic!("wrong simulation component type"),
        }
    }

    #[test]
    fn non_rigid_simulation_settings_survive_project_roundtrip() {
        let mut d = Document::new();
        d.scene.simulations.cloth.enabled = true;
        d.scene.simulations.cloth.quality_steps = 24;
        d.scene.simulations.soft_body.edge_stiffness = 0.42;
        d.scene.simulations.fluid.domain_resolution = 96;
        d.scene.simulations.particles.count = 12_345;
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert!(loaded.simulations.cloth.enabled);
        assert_eq!(loaded.simulations.cloth.quality_steps, 24);
        assert!((loaded.simulations.soft_body.edge_stiffness - 0.42).abs() < 1e-6);
        assert_eq!(loaded.simulations.fluid.domain_resolution, 96);
        assert_eq!(loaded.simulations.particles.count, 12_345);
    }

    #[test]
    fn rigid_body_settings_survive_project_roundtrip() {
        let mut d = Document::new();
        let id = d.scene.objects[0].id;
        d.scene.rigid_bodies.insert(id, crate::RigidBody {
            body_type: crate::RigidBodyType::Passive,
            collision_shape: crate::CollisionShape::Box,
            mass: 7.5,
            friction: 0.8,
            restitution: 0.25,
            linear_damping: 0.1,
            enabled: true,
            ..crate::RigidBody::default()
        });
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let body = loaded.rigid_bodies.get(&id).unwrap();
        assert_eq!(body.body_type, crate::RigidBodyType::Passive);
        assert_eq!(body.collision_shape, crate::CollisionShape::Box);
        assert!((body.mass - 7.5).abs() < 1e-6);
    }

    #[test]
    fn rigid_body_world_constraints_and_bake_survive_project_roundtrip() {
        let mut d = Document::new();
        let a = d.scene.objects[0].id;
        let b = d.scene.objects[1].id;
        d.scene.rigid_bodies.insert(a, crate::RigidBody::default());
        d.scene.rigid_bodies.insert(b, crate::RigidBody {
            body_type: crate::RigidBodyType::Passive,
            collision_shape: crate::CollisionShape::Mesh,
            ..crate::RigidBody::default()
        });
        d.scene.physics_world.gravity = Vec3::new(0.0, -3.0, 0.0);
        d.scene.physics_world.substeps_per_frame = 24;
        d.scene.physics_world.solver_iterations = 18;
        d.scene.rigid_body_constraints.push(crate::RigidBodyConstraint {
            object_a: a,
            object_b: b,
            kind: crate::RigidBodyConstraintType::Hinge,
            anchor_a: Vec3::new(0.1, 0.2, 0.3),
            anchor_b: Vec3::new(-0.1, 0.0, 0.2),
            ..crate::RigidBodyConstraint::default()
        });
        d.scene.physics_cache = Some(crate::PhysicsCache {
            frames: [(
                1,
                vec![crate::PhysicsTransform {
                    object_id: a,
                    translation: Vec3::new(1.0, 2.0, 3.0),
                    rotation: glam::Quat::IDENTITY,
                }],
            )].into_iter().collect(),
        });

        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(loaded.physics_world.substeps_per_frame, 24);
        assert!((loaded.physics_world.gravity.y + 3.0).abs() < 1e-6);
        assert_eq!(loaded.rigid_body_constraints.len(), 1);
        assert_eq!(loaded.rigid_body_constraints[0].kind, crate::RigidBodyConstraintType::Hinge);
        assert_eq!(loaded.physics_cache.as_ref().unwrap().frames[&1].len(), 1);
        assert_eq!(loaded.rigid_bodies[&b].collision_shape, crate::CollisionShape::Mesh);
    }

    #[test]
    fn frg4_projects_load_without_rigid_body_settings() {
        let d = Document::new();
        let old = Version4Scene {
            objects: d.scene.objects.clone(),
            materials: d.scene.materials.clone(),
            textures: d.scene.textures.clone(),
            collections: d.scene.collections.clone(),
            cursor: d.scene.cursor,
            next_id: d.scene.next_id,
            animation: d.scene.animation.clone(),
            compositor_graph: d.scene.compositor_graph.clone(),
        };
        let raw = bincode::serialize(&old).unwrap();
        let mut encoded = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoded.write_all(&raw).unwrap();
        let mut bytes = V4_MAGIC.to_vec();
        bytes.extend(encoded.finish().unwrap());
        let loaded = from_bytes(&bytes).unwrap();
        assert!(loaded.rigid_bodies.is_empty());
        assert_eq!(loaded.objects.len(), d.scene.objects.len());
    }

    #[test]
    fn roundtrip() {
        let d = Document::new();
        let back = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(back.objects.len(), 3); // Cube + Sun + Camera
        assert_eq!(
            back.objects[0].mesh.vertices.len(),
            d.scene.objects[0].mesh.vertices.len()
        );
    }

    #[test]
    fn compositor_graph_survives_project_roundtrip() {
        let mut d = Document::new();
        let effect = d
            .scene
            .compositor_graph
            .add_node(crate::CompositorNodeKind::Exposure, [260.0, 40.0]);
        d.scene
            .compositor_graph
            .nodes
            .iter_mut()
            .find(|node| node.id == effect)
            .unwrap()
            .value = 1.5;
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(loaded.compositor_graph, d.scene.compositor_graph);
    }

    #[test]
    fn legacy_nodes_load_with_default_editor_state() {
        let legacy = LegacyScene {
            objects: Vec::new(),
            materials: vec![LegacyMaterial {
                name: "Legacy".into(),
                base_color: [1.0; 4],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                alpha_mode: crate::AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                double_sided: false,
                normal_scale: 1.0,
                occlusion_strength: 1.0,
                base_color_tex: None,
                metallic_roughness_tex: None,
                normal_tex: None,
                occlusion_tex: None,
                emissive_tex: None,
                shader_graph: Some(LegacyShaderGraph {
                    nodes: vec![LegacyShaderNode {
                        id: 7,
                        kind: ShaderNodeKind::ColorRamp,
                        pos: [12.0, 34.0],
                        image_index: None,
                        value: 2.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                        props: [0.0; 16],
                        input_defaults: Vec::new(),
                        mode: 0,
                        name_prop: String::new(),
                    }],
                    links: Vec::new(),
                    next_id: 8,
                }),
                displacement_scale: 0.0,
                volume_density: 0.0,
            }],
            textures: Arc::new(Vec::new()),
            collections: Vec::new(),
            cursor: Vec3::ZERO,
            next_id: 1,
            animation: AnimationData::default(),
        };
        let raw = bincode::serialize(&legacy).unwrap();
        let mut encoded = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoded.write_all(&raw).unwrap();
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend(encoded.finish().unwrap());

        let scene = from_bytes(&bytes).unwrap();
        let node = &scene.materials[0].shader_graph.as_ref().unwrap().nodes[0];
        assert_eq!(node.id, 7);
        assert_eq!(node.pos, [12.0, 34.0]);
        assert_eq!(node.ui_size, [0.0; 2]);
        assert!(node.ramp_stops.is_empty());
    }

    #[test]
    fn frg3_project_loads_with_a_default_compositor_graph() {
        let legacy = Version3Scene {
            objects: Vec::new(),
            materials: vec![Version3Material {
                name: "Old material".into(),
                base_color: [1.0; 4],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                alpha_mode: crate::AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                double_sided: false,
                normal_scale: 1.0,
                occlusion_strength: 1.0,
                base_color_tex: None,
                metallic_roughness_tex: None,
                normal_tex: None,
                occlusion_tex: None,
                emissive_tex: None,
                shader_graph: Some(Version3ShaderGraph {
                    nodes: vec![Version3ShaderNode {
                        id: 8,
                        kind: ShaderNodeKind::ColorRamp,
                        pos: [1.0, 2.0],
                        image_index: None,
                        value: 1.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                        props: [0.0; 16],
                        input_defaults: Vec::new(),
                        mode: 0,
                        name_prop: String::new(),
                        ui_size: [190.0, 230.0],
                        ramp_stops: vec![ColorRampStop {
                            color: [1.0; 4],
                            position: 0.5,
                        }],
                    }],
                    links: Vec::new(),
                    next_id: 9,
                }),
                displacement_scale: 0.0,
                volume_density: 0.0,
            }],
            textures: Arc::new(Vec::new()),
            collections: Vec::new(),
            cursor: Vec3::ZERO,
            next_id: 1,
            animation: AnimationData::default(),
        };
        let raw = bincode::serialize(&legacy).unwrap();
        let mut encoded = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoded.write_all(&raw).unwrap();
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend(encoded.finish().unwrap());

        let scene = from_bytes(&bytes).unwrap();
        assert_eq!(scene.compositor_graph.nodes.len(), 3);
        let node = &scene.materials[0].shader_graph.as_ref().unwrap().nodes[0];
        assert_eq!(node.id, 8);
        assert_eq!(node.ui_size, [190.0, 230.0]);
        assert_eq!(node.ramp_stops.len(), 1);
    }

    #[test]
    fn fluids_and_mesh_caches_survive_project_roundtrip() {
        let mut d = Document::new();
        let id = d.scene.objects[0].id;
        d.scene.fluids.insert(id, crate::FluidObject::Domain(crate::FluidDomain { resolution: 48, ..Default::default() }));
        let frames = vec![d.scene.objects[0].mesh.clone(), d.scene.objects[0].mesh.clone()];
        d.scene.mesh_caches.insert(id, crate::MeshSequence::new("test", frames, 1.0, 24.0));
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        match loaded.fluids.get(&id) {
            Some(crate::FluidObject::Domain(domain)) => assert_eq!(domain.resolution, 48),
            _ => panic!("fluid domain lost"),
        }
        assert_eq!(loaded.mesh_caches.get(&id).unwrap().len(), 2);
    }
}
