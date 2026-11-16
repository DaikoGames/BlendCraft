//! Cloth, soft-body and particle solvers + bake-to-mesh-sequence.
//!
//! Rigid bodies stay in `physics.rs` (Rapier). Fluids stay in `fluid.rs` (PBF).
//! This module owns the deformable / emitter systems that write `MeshSequence`
//! caches so timeline playback works the same way as fluid bakes.
use crate::mesh::{MeshData, Vertex};
use crate::meshcache::MeshSequence;
use crate::scene::{Object, Scene};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Settings (serialized with the project)
// ---------------------------------------------------------------------------

pub use crate::cloth::ClothSettings;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftBodySettings {
    pub enabled: bool,
    pub mass: f32,
    pub edge_stiffness: f32,
    pub bending_stiffness: f32,
    pub damping: f32,
    pub goal_strength: f32,
    pub goal_friction: f32,
    pub gravity_scale: f32,
    pub use_self_collision: bool,
}
impl Default for SoftBodySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            mass: 1.0,
            edge_stiffness: 0.8,
            bending_stiffness: 0.2,
            damping: 0.02,
            goal_strength: 0.1,
            goal_friction: 0.0,
            gravity_scale: 1.0,
            use_self_collision: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidSettings {
    pub enabled: bool,
    pub domain_resolution: u32,
    pub particle_radius: f32,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub time_scale: f32,
    pub gravity_scale: f32,
    pub max_particles: u32,
    pub use_adaptive_time: bool,
}
impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            domain_resolution: 32,
            particle_radius: 0.1,
            viscosity: 0.03,
            surface_tension: 0.2,
            time_scale: 1.0,
            gravity_scale: 1.0,
            max_particles: 50_000,
            use_adaptive_time: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleSettings {
    pub enabled: bool,
    pub count: u32,
    pub emit_rate: f32,
    pub lifetime: f32,
    pub frame_start: f32,
    pub frame_end: f32,
    pub initial_velocity: Vec3,
    pub velocity_randomness: f32,
    pub gravity_scale: f32,
    pub size: f32,
    pub size_randomness: f32,
}
impl Default for ParticleSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            count: 1000,
            emit_rate: 100.0,
            lifetime: 50.0,
            frame_start: 1.0,
            frame_end: 200.0,
            initial_velocity: Vec3::new(0.0, 2.0, 0.0),
            velocity_randomness: 0.2,
            gravity_scale: 1.0,
            size: 0.05,
            size_randomness: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SimulationSettings {
    #[serde(default)]
    pub cloth: ClothSettings,
    #[serde(default)]
    pub soft_body: SoftBodySettings,
    #[serde(default)]
    pub fluid: FluidSettings,
    #[serde(default)]
    pub particles: ParticleSettings,
}

/// A simulation component assigned to one mesh object (Blender-style Physics tab).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectSimulation {
    Cloth(ClothSettings),
    SoftBody(SoftBodySettings),
    Fluid(FluidSettings),
    Particles(ParticleSettings),
}

impl ObjectSimulation {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Cloth(_) => "Cloth",
            Self::SoftBody(_) => "Soft Body",
            Self::Fluid(_) => "Fluid (legacy — use Domain)",
            Self::Particles(_) => "Particle System",
        }
    }

    pub fn enabled(&self) -> bool {
        match self {
            Self::Cloth(s) => s.enabled,
            Self::SoftBody(s) => s.enabled,
            Self::Fluid(s) => s.enabled,
            Self::Particles(s) => s.enabled,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared constraint graph
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DistConstraint {
    i: usize,
    j: usize,
    rest: f32,
    stiffness: f32,
}

struct DeformState {
    pos: Vec3,
    prev: Vec3,
    pinned: bool,
    inv_mass: f32,
    goal: Vec3,
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn build_edges(indices: &[u32]) -> Vec<(u32, u32)> {
    let mut set = HashSet::new();
    for tri in indices.chunks_exact(3) {
        set.insert(edge_key(tri[0], tri[1]));
        set.insert(edge_key(tri[1], tri[2]));
        set.insert(edge_key(tri[2], tri[0]));
    }
    set.into_iter().collect()
}

/// Opposite-vertex pairs across each shared edge → bending springs.
fn build_bending_pairs(indices: &[u32]) -> Vec<(u32, u32)> {
    let mut edge_tris: HashMap<(u32, u32), Vec<[u32; 3]>> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let t = [tri[0], tri[1], tri[2]];
        for &(a, b) in &[(0, 1), (1, 2), (2, 0)] {
            edge_tris.entry(edge_key(t[a], t[b])).or_default().push(t);
        }
    }
    let mut pairs = HashSet::new();
    for ((a, b), tris) in edge_tris {
        if tris.len() < 2 {
            continue;
        }
        let opposite = |t: [u32; 3]| -> u32 {
            for v in t {
                if v != a && v != b {
                    return v;
                }
            }
            a
        };
        pairs.insert(edge_key(opposite(tris[0]), opposite(tris[1])));
    }
    pairs.into_iter().collect()
}

pub(crate) fn boundary_verts(indices: &[u32], n: usize) -> HashSet<usize> {
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        for e in [edge_key(tri[0], tri[1]), edge_key(tri[1], tri[2]), edge_key(tri[2], tri[0])] {
            *count.entry(e).or_default() += 1;
        }
    }
    let mut out = HashSet::new();
    for ((a, b), c) in count {
        if c == 1 {
            out.insert(a as usize);
            out.insert(b as usize);
        }
    }
    // Degenerate / empty: pin nothing special, but clamp indices.
    out.retain(|&i| i < n);
    out
}

fn project_distance(states: &mut [DeformState], c: &DistConstraint) {
    let (pi, pj) = (states[c.i].pos, states[c.j].pos);
    let delta = pj - pi;
    let len = delta.length();
    if len < 1e-8 {
        return;
    }
    let diff = (len - c.rest) / len;
    let w_i = states[c.i].inv_mass;
    let w_j = states[c.j].inv_mass;
    let w = w_i + w_j;
    if w < 1e-12 {
        return;
    }
    let corr = delta * (diff * c.stiffness);
    if !states[c.i].pinned {
        states[c.i].pos += corr * (w_i / w);
    }
    if !states[c.j].pinned {
        states[c.j].pos -= corr * (w_j / w);
    }
}

pub(crate) fn recompute_normals(mesh: &mut MeshData) {
    for v in &mut mesh.vertices {
        v.normal = [0.0, 0.0, 0.0];
    }
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
        let n = (b - a).cross(c - a);
        for &idx in tri {
            let vn = &mut mesh.vertices[idx as usize].normal;
            vn[0] += n.x;
            vn[1] += n.y;
            vn[2] += n.z;
        }
    }
    for v in &mut mesh.vertices {
        let n = Vec3::from(v.normal).normalize_or_zero();
        v.normal = n.to_array();
    }
}

fn mesh_from_positions(base: &MeshData, positions: &[Vec3]) -> MeshData {
    let mut mesh = MeshData {
        name: base.name.clone(),
        vertices: base.vertices.clone(),
        indices: base.indices.clone(),
        groups: base.groups.clone(),
    };
    for (i, p) in positions.iter().enumerate() {
        if i < mesh.vertices.len() {
            mesh.vertices[i].position = p.to_array();
        }
    }
    recompute_normals(&mut mesh);
    mesh
}

// ---------------------------------------------------------------------------
// Soft body
// ---------------------------------------------------------------------------

struct SoftBodySim {
    states: Vec<DeformState>,
    edges: Vec<DistConstraint>,
    bend: Vec<DistConstraint>,
    settings: SoftBodySettings,
    rest_mesh: MeshData,
}

impl SoftBodySim {
    fn new(mesh: &MeshData, settings: SoftBodySettings, world_positions: &[Vec3]) -> Self {
        let n = mesh.vertices.len();
        let inv = if settings.mass > 1e-6 {
            1.0 / (settings.mass / n.max(1) as f32).max(1e-6)
        } else {
            1.0
        };
        let states: Vec<DeformState> = (0..n)
            .map(|i| {
                let p = world_positions.get(i).copied().unwrap_or(Vec3::ZERO);
                DeformState {
                    pos: p,
                    prev: p,
                    pinned: false,
                    inv_mass: inv,
                    goal: p,
                }
            })
            .collect();
        let edges: Vec<DistConstraint> = build_edges(&mesh.indices)
            .into_iter()
            .map(|(a, b)| {
                let i = a as usize;
                let j = b as usize;
                DistConstraint {
                    i,
                    j,
                    rest: (states[i].pos - states[j].pos).length().max(1e-5),
                    stiffness: settings.edge_stiffness.clamp(0.0, 1.0),
                }
            })
            .collect();
        let bend: Vec<DistConstraint> = build_bending_pairs(&mesh.indices)
            .into_iter()
            .map(|(a, b)| {
                let i = a as usize;
                let j = b as usize;
                DistConstraint {
                    i,
                    j,
                    rest: (states[i].pos - states[j].pos).length().max(1e-5),
                    stiffness: settings.bending_stiffness.clamp(0.0, 1.0) * 0.5,
                }
            })
            .collect();
        Self {
            states,
            edges,
            bend,
            settings,
            rest_mesh: mesh.clone(),
        }
    }

    fn step(&mut self, dt: f32, gravity: Vec3) {
        let dt = dt.clamp(1e-5, 0.05);
        let damp = (1.0 - self.settings.damping).clamp(0.0, 1.0);
        let g = gravity * self.settings.gravity_scale;
        let goal = self.settings.goal_strength.clamp(0.0, 1.0);
        for s in &mut self.states {
            let vel = (s.pos - s.prev) * damp;
            let mut accel = g;
            // Goal spring pulls toward the rest pose (Blender soft-body goal).
            if goal > 0.0 {
                accel += (s.goal - s.pos) * (goal * 40.0);
            }
            let next = s.pos + vel + accel * (dt * dt);
            s.prev = s.pos;
            s.pos = next;
        }
        for _ in 0..8 {
            for c in &self.edges {
                project_distance(&mut self.states, c);
            }
            for c in &self.bend {
                project_distance(&mut self.states, c);
            }
        }
        // Optional cheap self-collision: push vertices that get too close apart.
        if self.settings.use_self_collision {
            let min_d = 0.02_f32;
            let n = self.states.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = self.states[j].pos - self.states[i].pos;
                    let len = d.length();
                    if len < min_d && len > 1e-8 {
                        let push = d * ((min_d - len) * 0.5 / len);
                        self.states[i].pos -= push;
                        self.states[j].pos += push;
                    }
                }
            }
        }
    }

    fn to_mesh(&self) -> MeshData {
        let positions: Vec<Vec3> = self.states.iter().map(|s| s.pos).collect();
        mesh_from_positions(&self.rest_mesh, &positions)
    }
}

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

struct Particle {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    lifetime: f32,
    size: f32,
}

struct ParticleSim {
    particles: Vec<Particle>,
    settings: ParticleSettings,
    emit_origin: Vec3,
    emit_accum: f32,
    rng: u32,
}

impl ParticleSim {
    fn new(settings: ParticleSettings, origin: Vec3) -> Self {
        Self {
            particles: Vec::new(),
            settings,
            emit_origin: origin,
            emit_accum: 0.0,
            rng: 0xC0FFEE,
        }
    }

    fn rand01(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }

    fn step(&mut self, dt: f32, frame: f32, gravity: Vec3) {
        if !self.settings.enabled {
            return;
        }
        // Copy fields first so `rand01` can mutably borrow `self` without clashing
        // with an active `&self.settings` borrow.
        let frame_start = self.settings.frame_start;
        let frame_end = self.settings.frame_end;
        let emit_rate = self.settings.emit_rate;
        let max_count = self.settings.count as usize;
        let vel_rand = self.settings.velocity_randomness;
        let base_size = self.settings.size;
        let size_rand = self.settings.size_randomness;
        let init_vel = self.settings.initial_velocity;
        let lifetime = self.settings.lifetime.max(1.0) / 24.0;
        let gravity_scale = self.settings.gravity_scale;
        let origin = self.emit_origin;

        if frame >= frame_start && frame <= frame_end {
            self.emit_accum += emit_rate * dt;
            while self.emit_accum >= 1.0 && self.particles.len() < max_count {
                self.emit_accum -= 1.0;
                let jx = (self.rand01() - 0.5) * 2.0 * vel_rand;
                let jy = (self.rand01() - 0.5) * 2.0 * vel_rand;
                let jz = (self.rand01() - 0.5) * 2.0 * vel_rand;
                let size = (base_size * (1.0 + (self.rand01() - 0.5) * 2.0 * size_rand)).max(0.001);
                let ox = (self.rand01() - 0.5) * 0.1;
                let oy = (self.rand01() - 0.5) * 0.1;
                let oz = (self.rand01() - 0.5) * 0.1;
                self.particles.push(Particle {
                    pos: origin + Vec3::new(ox, oy, oz),
                    vel: init_vel + Vec3::new(jx, jy, jz),
                    age: 0.0,
                    lifetime,
                    size,
                });
            }
        }
        let g = gravity * gravity_scale;
        for p in &mut self.particles {
            p.vel += g * dt;
            p.pos += p.vel * dt;
            p.age += dt;
        }
        self.particles.retain(|p| p.age < p.lifetime);
    }

    fn to_mesh(&self) -> MeshData {
        // Each particle becomes a small camera-facing tetrahedron so it shows up
        // without a special particle shader.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for p in &self.particles {
            let base = vertices.len() as u32;
            let s = p.size;
            let offs = [
                Vec3::new(0.0, s, 0.0),
                Vec3::new(s * 0.866, -s * 0.5, 0.0),
                Vec3::new(-s * 0.866, -s * 0.5, 0.0),
                Vec3::new(0.0, 0.0, s),
            ];
            for o in &offs {
                let pos = p.pos + *o;
                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal: o.normalize_or_zero().to_array(),
                    uv: [0.0, 0.0],
                });
            }
            for tri in [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]] {
                indices.push(base + tri[0]);
                indices.push(base + tri[1]);
                indices.push(base + tri[2]);
            }
        }
        MeshData {
            name: "Particles".into(),
            vertices,
            indices,
            groups: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// World-space positions helper
// ---------------------------------------------------------------------------

fn world_vertex_positions(obj: &Object, scene: &Scene) -> Vec<Vec3> {
    let worlds = scene.world_matrices();
    let idx = scene.index_of(obj.id).unwrap_or(0);
    let m = worlds.get(idx).copied().unwrap_or(glam::Mat4::IDENTITY);
    obj.mesh
        .vertices
        .iter()
        .map(|v| m.transform_point3(Vec3::from(v.position)))
        .collect()
}

// ---------------------------------------------------------------------------
// Public bake API
// ---------------------------------------------------------------------------

/// Bake every enabled cloth / soft-body / particle component into mesh caches.
/// Returns how many object caches were written.
pub fn bake_non_rigid_simulations(scene: &mut Scene) -> usize {
    let fps = scene.animation.fps.max(1) as f32;
    let start = scene.animation.frame_start;
    let end = scene.animation.frame_end.max(start);
    let dt = 1.0 / fps;
    let gravity = scene.physics_world.gravity;

    // Collect targets first (id, kind) so we can mutably borrow scene later.
    let targets: Vec<(u64, ObjectSimulation)> = scene
        .object_simulations
        .iter()
        .filter(|(_, sim)| sim.enabled())
        .map(|(id, sim)| (*id, sim.clone()))
        .collect();

    let mut baked = 0usize;
    for (id, sim) in targets {
        let Some(obj_index) = scene.index_of(id) else {
            continue;
        };
        let obj = scene.objects[obj_index].clone();
        if !obj.kind.is_mesh() {
            continue;
        }

        match sim {
            ObjectSimulation::Cloth(settings) => {
                if crate::cloth::bake_cloth(scene, id, &settings) {
                    baked += 1;
                }
            }
            ObjectSimulation::SoftBody(settings) => {
                let world_pos = world_vertex_positions(&obj, scene);
                let mut soft = SoftBodySim::new(&obj.mesh, settings, &world_pos);
                let mut frames = Vec::with_capacity((end - start + 1) as usize);
                for _frame in start..=end {
                    soft.step(dt, gravity);
                    let worlds = scene.world_matrices();
                    let m = worlds.get(obj_index).copied().unwrap_or(glam::Mat4::IDENTITY);
                    let inv = m.inverse();
                    let mut mesh = soft.to_mesh();
                    for v in &mut mesh.vertices {
                        let p = inv.transform_point3(Vec3::from(v.position));
                        v.position = p.to_array();
                    }
                    recompute_normals(&mut mesh);
                    frames.push(Arc::new(mesh));
                }
                scene.mesh_caches.insert(
                    id,
                    MeshSequence::new(format!("Soft body bake of {}", obj.name), frames, start as f32, fps),
                );
                baked += 1;
            }
            ObjectSimulation::Particles(settings) => {
                let worlds = scene.world_matrices();
                let m = worlds.get(obj_index).copied().unwrap_or(glam::Mat4::IDENTITY);
                let origin = m.transform_point3(Vec3::ZERO);
                let mut parts = ParticleSim::new(settings, origin);
                let mut frames = Vec::with_capacity((end - start + 1) as usize);
                for frame in start..=end {
                    parts.step(dt, frame as f32, gravity);
                    // Particle positions are world-space; convert to object local.
                    let inv = m.inverse();
                    let mut mesh = parts.to_mesh();
                    for v in &mut mesh.vertices {
                        let p = inv.transform_point3(Vec3::from(v.position));
                        v.position = p.to_array();
                    }
                    recompute_normals(&mut mesh);
                    frames.push(Arc::new(mesh));
                }
                scene.mesh_caches.insert(
                    id,
                    MeshSequence::new(format!("Particle bake of {}", obj.name), frames, start as f32, fps),
                );
                baked += 1;
            }
            ObjectSimulation::Fluid(_) => {
                // Legacy fluid component: point the user at Domain/Flow/Effector.
                // `migrate_legacy_fluids` converts these into the real fluid system.
            }
        }
    }
    baked
}

/// Free mesh caches produced by cloth / soft-body / particle bakes for one object.
pub fn free_sim_bake(scene: &mut Scene, object_id: u64) -> bool {
    let Some(seq) = scene.mesh_caches.remove(&object_id) else {
        return false;
    };
    // The timeline left the last cached frame in the object: put the cloth's rest pose back.
    if seq.source.starts_with("Cloth bake") {
        if let (Some(index), Some(first)) = (scene.index_of(object_id), seq.frames.first()) {
            scene.objects[index].mesh = first.clone();
        }
    }
    true
}

/// Convert legacy `ObjectSimulation::Fluid` + scene-level fluid flags into the
/// Domain / Flow fluid objects used by `fluid.rs`. Returns how many were migrated.
pub fn promote_legacy_fluid_components(scene: &mut Scene) -> usize {
    crate::fluid::migrate_legacy_fluids(scene);
    // Count remaining legacy fluid components after migration.
    scene
        .object_simulations
        .values()
        .filter(|s| matches!(s, ObjectSimulation::Fluid(_)))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::plane;
    use crate::scene::Object;

    #[test]
    fn cloth_bake_moves_vertices() {
        let mut scene = Scene::default();
        scene.animation.frame_start = 1;
        scene.animation.frame_end = 5;
        scene.animation.fps = 24;
        let mut o = Object::new("Flag", plane(1.0));
        o.id = scene.alloc_id();
        let id = o.id;
        scene.objects.push(o);
        scene.object_simulations.insert(
            id,
            ObjectSimulation::Cloth(ClothSettings {
                enabled: true,
                use_collision: false,
                ..Default::default()
            }),
        );
        let n = bake_non_rigid_simulations(&mut scene);
        assert_eq!(n, 1);
        let seq = scene.mesh_caches.get(&id).expect("cache");
        assert_eq!(seq.len(), 5);
        let moved = seq.frames[0]
            .vertices
            .iter()
            .zip(seq.frames[4].vertices.iter())
            .any(|(a, b)| (Vec3::from(a.position) - Vec3::from(b.position)).length() > 1e-3);
        assert!(moved, "free cloth must fall under gravity");
    }

    #[test]
    fn freeing_a_cloth_bake_restores_the_rest_pose() {
        let mut scene = Scene::default();
        scene.animation.frame_start = 1;
        scene.animation.frame_end = 12;
        scene.animation.fps = 24;
        let mut o = Object::new("Cloth", plane(1.0));
        o.id = scene.alloc_id();
        let id = o.id;
        scene.objects.push(o);
        scene.object_simulations.insert(
            id,
            ObjectSimulation::Cloth(ClothSettings {
                enabled: true,
                use_collision: false,
                ..Default::default()
            }),
        );
        bake_non_rigid_simulations(&mut scene);
        let last = scene.mesh_caches[&id].frames.last().unwrap().clone();
        scene.objects[0].mesh = last;
        assert!(free_sim_bake(&mut scene, id));
        let y = scene.objects[0].mesh.vertices[0].position[1];
        assert!(y.abs() < 1e-4, "rest pose expected, got y={y}");
    }
}
