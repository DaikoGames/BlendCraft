//! Scene-level configuration for the non-rigid simulation systems.
//!
//! These settings are serialized with the project. The actual cloth, soft-body,
//! fluid and particle solvers are deliberately kept separate from Rapier rigid
//! bodies so each can evolve without coupling their runtime state to the scene.
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClothSettings {
    pub enabled: bool,
    pub quality_steps: u32,
    pub vertex_mass: f32,
    pub tension_stiffness: f32,
    pub bending_stiffness: f32,
    pub damping: f32,
    pub air_damping: f32,
    pub gravity_scale: f32,
    pub pin_boundary: bool,
}
impl Default for ClothSettings {
    fn default() -> Self { Self { enabled: false, quality_steps: 8, vertex_mass: 0.3, tension_stiffness: 15.0, bending_stiffness: 0.5, damping: 0.02, air_damping: 1.0, gravity_scale: 1.0, pin_boundary: false } }
}

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
    fn default() -> Self { Self { enabled: false, mass: 1.0, edge_stiffness: 0.8, bending_stiffness: 0.2, damping: 0.02, goal_strength: 0.0, goal_friction: 0.0, gravity_scale: 1.0, use_self_collision: false } }
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
    fn default() -> Self { Self { enabled: false, domain_resolution: 32, particle_radius: 0.1, viscosity: 0.03, surface_tension: 0.2, time_scale: 1.0, gravity_scale: 1.0, max_particles: 50_000, use_adaptive_time: true } }
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
    fn default() -> Self { Self { enabled: false, count: 1000, emit_rate: 100.0, lifetime: 50.0, frame_start: 1.0, frame_end: 200.0, initial_velocity: Vec3::new(0.0, 2.0, 0.0), velocity_randomness: 0.2, gravity_scale: 1.0, size: 0.05, size_randomness: 0.0 } }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SimulationSettings {
    #[serde(default)] pub cloth: ClothSettings,
    #[serde(default)] pub soft_body: SoftBodySettings,
    #[serde(default)] pub fluid: FluidSettings,
    #[serde(default)] pub particles: ParticleSettings,
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
        match self { Self::Cloth(_) => "Cloth", Self::SoftBody(_) => "Soft Body", Self::Fluid(_) => "Fluid", Self::Particles(_) => "Particle System" }
    }
}
