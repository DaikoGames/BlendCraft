//! Blender-style rigid-body simulation backed by Rapier 3D.
//!
//! Scene settings and baked transforms are serializable; Rapier's transient
//! broad/narrow phase and solver state live only in `PhysicsRuntime`.
use crate::{Scene};
use glam::{Mat4, Quat, Vec3};
use rapier3d::na::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use rapier3d::prelude::{
    BroadPhaseMultiSap, CCDSolver, ColliderBuilder, ColliderSet,
    FixedJointBuilder, ImpulseJointSet, IntegrationParameters, IslandManager,
    MultibodyJointSet, NarrowPhase, PhysicsPipeline, PrismaticJointBuilder,
    RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle, RigidBodySet,
    SharedShape, SphericalJointBuilder,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigidBodyType {
    #[default]
    Active,
    Passive,
}

/// Collision geometry available in Blender's rigid-body object workflow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollisionShape {
    #[default]
    ConvexHull,
    Box,
    Sphere,
    Capsule,
    Cylinder,
    Cone,
    Mesh,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub body_type: RigidBodyType,
    pub collision_shape: CollisionShape,
    pub mass: f32,
    pub friction: f32,
    pub restitution: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub enabled: bool,
    /// Active but kinematic: follows object animation and pushes dynamic bodies.
    pub animated: bool,
    /// Blender's Dynamic toggle. When disabled, an active body is kinematic.
    pub dynamic: bool,
    pub use_deactivation: bool,
    pub use_ccd: bool,
    pub collision_margin: f32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Active,
            collision_shape: CollisionShape::ConvexHull,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            linear_damping: 0.04,
            angular_damping: 0.1,
            enabled: true,
            animated: false,
            dynamic: true,
            use_deactivation: true,
            use_ccd: false,
            collision_margin: 0.04,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsWorldSettings {
    pub enabled: bool,
    pub gravity: Vec3,
    pub time_scale: f32,
    pub substeps_per_frame: u32,
    pub solver_iterations: u32,
    pub split_impulse: bool,
    pub cache_start: i32,
    pub cache_end: i32,
}
impl Default for PhysicsWorldSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            time_scale: 1.0,
            substeps_per_frame: 10,
            solver_iterations: 10,
            split_impulse: false,
            cache_start: 1,
            cache_end: 250,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigidBodyConstraintType {
    #[default]
    Fixed,
    Point,
    Hinge,
    Slider,
}

/// A constraint connects two rigid-body objects by stable object IDs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyConstraint {
    pub name: String,
    pub object_a: u64,
    pub object_b: u64,
    pub kind: RigidBodyConstraintType,
    pub enabled: bool,
    pub disable_collisions: bool,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub use_limits: bool,
    pub lower_limit: f32,
    pub upper_limit: f32,
}
impl Default for RigidBodyConstraint {
    fn default() -> Self {
        Self {
            name: "Rigid Body Constraint".into(),
            object_a: 0,
            object_b: 0,
            kind: RigidBodyConstraintType::Fixed,
            enabled: true,
            disable_collisions: true,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            axis: Vec3::Y,
            use_limits: false,
            lower_limit: -std::f32::consts::FRAC_PI_2,
            upper_limit: std::f32::consts::FRAC_PI_2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsTransform {
    pub object_id: u64,
    pub translation: Vec3,
    pub rotation: Quat,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsCache {
    pub frames: BTreeMap<i32, Vec<PhysicsTransform>>,
}

/// Transient simulation state. Rebuilt from Scene when playback starts.
pub struct PhysicsRuntime {
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseMultiSap,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    handles: HashMap<u64, RigidBodyHandle>,
    settings: PhysicsWorldSettings,
}

fn rapier_iso(position: Vec3, rotation: Quat) -> Isometry3<f32> {
    let q = rotation.normalize();
    Isometry3::from_parts(
        Translation3::new(position.x, position.y, position.z),
        UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z)),
    )
}
fn glam_pose(iso: &Isometry3<f32>) -> (Vec3, Quat) {
    let t = iso.translation.vector;
    let q = iso.rotation.quaternion();
    (Vec3::new(t.x, t.y, t.z), Quat::from_xyzw(q.i, q.j, q.k, q.w).normalize())
}

fn collider_shape(scene: &Scene, index: usize, body: &RigidBody, world_scale: Vec3) -> Option<(SharedShape, Vec3)> {
    let object = scene.objects.get(index)?;
    let bounds = object.mesh.bounds()?;
    let scale = world_scale.abs();
    let signed_scale = world_scale;
    let center = bounds.center() * signed_scale;
    let half = (bounds.max - bounds.min) * 0.5 * scale;
    let shape = match body.collision_shape {
        CollisionShape::Box => SharedShape::cuboid(half.x.max(0.0001), half.y.max(0.0001), half.z.max(0.0001)),
        CollisionShape::Sphere => SharedShape::ball(half.max_element().max(0.0001)),
        CollisionShape::Capsule => SharedShape::capsule_y((half.y - half.x.max(half.z)).max(0.0), half.x.max(half.z).max(0.0001)),
        CollisionShape::Cylinder => SharedShape::cylinder(half.y.max(0.0001), half.x.max(half.z).max(0.0001)),
        CollisionShape::Cone => SharedShape::cone(half.y.max(0.0001), half.x.max(half.z).max(0.0001)),
        CollisionShape::ConvexHull | CollisionShape::Mesh => {
            let mesh_center = bounds.center();
            let points: Vec<Point3<f32>> = object.mesh.vertices.iter().map(|v| {
                // Shapes are centered around their collider origin; the center
                // offset is applied once through ColliderBuilder::translation.
                let p = (Vec3::from(v.position) - mesh_center) * signed_scale;
                Point3::new(p.x, p.y, p.z)
            }).collect();
            if body.collision_shape == CollisionShape::Mesh && body.body_type == RigidBodyType::Passive {
                let triangles: Vec<[u32; 3]> = object.mesh.indices.chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]]).collect();
                // Rapier 0.22 returns SharedShape directly from `trimesh`.
                // Keep the mesh collider native instead of treating it as a Result.
                SharedShape::trimesh(points, triangles)
            } else {
                SharedShape::convex_hull(&points).unwrap_or_else(|| SharedShape::cuboid(half.x.max(0.0001), half.y.max(0.0001), half.z.max(0.0001)))
            }
        }
    };
    Some((shape, center))
}

impl PhysicsRuntime {
    pub fn new(scene: &Scene) -> Self {
        let mut runtime = Self {
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseMultiSap::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            handles: HashMap::new(),
            settings: scene.physics_world.clone(),
        };
        if !runtime.settings.enabled { return runtime; }
        let worlds = scene.world_matrices();
        for (index, object) in scene.objects.iter().enumerate() {
            let Some(body) = scene.rigid_bodies.get(&object.id).filter(|b| b.enabled) else { continue; };
            let (scale, rotation, translation) = worlds[index].to_scale_rotation_translation();
            let pose = rapier_iso(translation, rotation);
            let builder = match body.body_type {
                RigidBodyType::Passive if body.animated => RigidBodyBuilder::kinematic_position_based(),
                RigidBodyType::Passive => RigidBodyBuilder::fixed(),
                RigidBodyType::Active if body.animated || !body.dynamic => RigidBodyBuilder::kinematic_position_based(),
                RigidBodyType::Active => RigidBodyBuilder::dynamic(),
            }
            .position(pose)
            .linear_damping(body.linear_damping.max(0.0))
            .angular_damping(body.angular_damping.max(0.0))
            .can_sleep(body.use_deactivation)
            .ccd_enabled(body.use_ccd);
            let handle = runtime.bodies.insert(builder.build());
            runtime.handles.insert(object.id, handle);
            if let Some((shape, center)) = collider_shape(scene, index, body, scale) {
                let collider = ColliderBuilder::new(shape)
                    .translation(Vector3::new(center.x, center.y, center.z))
                    .friction(body.friction.clamp(0.0, 10.0))
                    .restitution(body.restitution.clamp(0.0, 1.0))
                    .contact_skin(body.collision_margin.max(0.0))
                    .mass(body.mass.max(0.001))
                    .build();
                runtime.colliders.insert_with_parent(collider, handle, &mut runtime.bodies);
            }
        }
        for constraint in scene.rigid_body_constraints.iter().filter(|c| c.enabled) {
            let (Some(&a), Some(&b)) = (runtime.handles.get(&constraint.object_a), runtime.handles.get(&constraint.object_b)) else { continue; };
            if a == b { continue; }
            let axis = if constraint.axis.length_squared() < 1.0e-8 { Vec3::Y } else { constraint.axis.normalize() };
            let axis = rapier3d::na::Unit::new_normalize(Vector3::new(axis.x, axis.y, axis.z));
            let anchor_a = Point3::new(constraint.anchor_a.x, constraint.anchor_a.y, constraint.anchor_a.z);
            let anchor_b = Point3::new(constraint.anchor_b.x, constraint.anchor_b.y, constraint.anchor_b.z);
            match constraint.kind {
                RigidBodyConstraintType::Fixed => {
                    let joint = FixedJointBuilder::new().local_anchor1(anchor_a).local_anchor2(anchor_b)
                        .contacts_enabled(!constraint.disable_collisions).build();
                    runtime.impulse_joints.insert(a, b, joint, true);
                }
                RigidBodyConstraintType::Point => {
                    let joint = SphericalJointBuilder::new().local_anchor1(anchor_a).local_anchor2(anchor_b)
                        .contacts_enabled(!constraint.disable_collisions).build();
                    runtime.impulse_joints.insert(a, b, joint, true);
                }
                RigidBodyConstraintType::Hinge => {
                    let mut joint = RevoluteJointBuilder::new(axis).local_anchor1(anchor_a).local_anchor2(anchor_b)
                        .contacts_enabled(!constraint.disable_collisions);
                    if constraint.use_limits { joint = joint.limits([constraint.lower_limit, constraint.upper_limit]); }
                    runtime.impulse_joints.insert(a, b, joint.build(), true);
                }
                RigidBodyConstraintType::Slider => {
                    let mut joint = PrismaticJointBuilder::new(axis).local_anchor1(anchor_a).local_anchor2(anchor_b)
                        .contacts_enabled(!constraint.disable_collisions);
                    if constraint.use_limits { joint = joint.limits([constraint.lower_limit, constraint.upper_limit]); }
                    runtime.impulse_joints.insert(a, b, joint.build(), true);
                }
            }
        }
        runtime
    }

    pub fn step_scene(&mut self, scene: &mut Scene, frame_dt: f32) -> bool {
        if !self.settings.enabled || !frame_dt.is_finite() || frame_dt <= 0.0
            || !self.settings.time_scale.is_finite() || self.settings.time_scale <= 0.0
            || self.handles.is_empty()
        {
            return false;
        }
        let substeps = self.settings.substeps_per_frame.clamp(1, 128);
        let mut params = IntegrationParameters::default();
        params.dt = (frame_dt * self.settings.time_scale.max(0.0)) / substeps as f32;
        params.num_solver_iterations = NonZeroUsize::new(self.settings.solver_iterations.clamp(1, 128) as usize)
            .expect("solver iterations are clamped to a non-zero value");
        let gravity = Vector3::new(self.settings.gravity.x, self.settings.gravity.y, self.settings.gravity.z);

        // Animated rigid bodies are kinematic: move them to their evaluated
        // object transform over the frame so they can push dynamic bodies.
        let worlds = scene.world_matrices();
        let mut kinematic_targets = Vec::new();
        for (id, handle) in &self.handles {
            let Some(index) = scene.index_of(*id) else { continue; };
            let Some(body) = scene.rigid_bodies.get(id) else { continue; };
            let is_kinematic = body.animated || (body.body_type == RigidBodyType::Active && !body.dynamic);
            if !is_kinematic { continue; }
            let (_, target_rotation, target_translation) = worlds[index].to_scale_rotation_translation();
            if let Some(rigid_body) = self.bodies.get(*handle) {
                let (start_translation, start_rotation) = glam_pose(rigid_body.position());
                kinematic_targets.push((*handle, start_translation, start_rotation, target_translation, target_rotation));
            }
        }

        for step in 0..substeps {
            let t = (step + 1) as f32 / substeps as f32;
            for (handle, start_translation, start_rotation, target_translation, target_rotation) in &kinematic_targets {
                if let Some(body) = self.bodies.get_mut(*handle) {
                    let translation = start_translation.lerp(*target_translation, t);
                    let rotation = start_rotation.slerp(*target_rotation, t).normalize();
                    body.set_next_kinematic_position(rapier_iso(translation, rotation));
                }
            }
            self.pipeline.step(
                &gravity, &params, &mut self.islands, &mut self.broad_phase,
                &mut self.narrow_phase, &mut self.bodies, &mut self.colliders,
                &mut self.impulse_joints, &mut self.multibody_joints,
                &mut self.ccd_solver, None, &(), &(),
            );
        }

        let mut changed = false;
        let mut worlds = scene.world_matrices();
        for (id, handle) in &self.handles {
            let Some(index) = scene.index_of(*id) else { continue; };
            let Some(body) = scene.rigid_bodies.get(id) else { continue; };
            if body.body_type != RigidBodyType::Active || body.animated || !body.dynamic { continue; }
            let Some(rigid_body) = self.bodies.get(*handle) else { continue; };
            let (translation, rotation) = glam_pose(rigid_body.position());
            let (scale, old_rotation, old_translation) = worlds[index].to_scale_rotation_translation();
            if (translation - old_translation).length_squared() > 1.0e-12
                || rotation.dot(old_rotation).abs() < 0.999999
            {
                let world = Mat4::from_scale_rotation_translation(scale, rotation, translation);
                scene.set_world(index, world, &worlds);
                changed = true;
                worlds = scene.world_matrices();
            }
        }
        changed
    }
}

/// Bake the rigid-body world from cache_start through cache_end.
pub fn bake_scene(scene: &Scene) -> PhysicsCache {
    let mut cache = PhysicsCache::default();
    let mut working = scene.clone();
    let start = working.physics_world.cache_start;
    let end = working.physics_world.cache_end.max(start);
    let fps = working.animation.fps.max(1) as f32;
    let animated_objects: Vec<_> = working.animation.objects.iter()
        .map(|(id, animation)| (*id, animation.clone()))
        .collect();
    let mut runtime: Option<PhysicsRuntime> = None;

    for frame in start..=end {
        // Evaluate keyframes before each frame's simulation step. Dynamic rigid
        // bodies remain solver-driven unless their Blender-style Animated flag
        // is enabled; other objects can still animate normally.
        for (id, animation) in &animated_objects {
            let driven_by_physics = working.rigid_bodies.get(id).is_some_and(|body|
                body.enabled && body.body_type == RigidBodyType::Active
                    && body.dynamic && !body.animated
            );
            if driven_by_physics { continue; }
            let Some(index) = working.index_of(*id) else { continue; };
            let (location, rotation, scale) = animation.eval(frame as f32);
            let object = &mut working.objects[index];
            if let Some(value) = location { object.translation = value; }
            if let Some(value) = rotation { object.rotation = value; }
            if let Some(value) = scale { object.scale = value; }
        }

        if runtime.is_none() {
            runtime = Some(PhysicsRuntime::new(&working));
        } else if let Some(simulation) = runtime.as_mut() {
            simulation.step_scene(&mut working, 1.0 / fps);
        }

        let worlds = working.world_matrices();
        let mut transforms = Vec::new();
        for (index, object) in working.objects.iter().enumerate() {
            if working.rigid_bodies.get(&object.id).is_some_and(|body|
                body.enabled && body.body_type == RigidBodyType::Active
            ) {
                let (_, rotation, translation) = worlds[index].to_scale_rotation_translation();
                transforms.push(PhysicsTransform { object_id: object.id, translation, rotation });
            }
        }
        cache.frames.insert(frame, transforms);
    }
    cache
}

/// Convert the currently baked rigid-body cache into ordinary object transform
/// keyframes. This mirrors Blender's "Bake to Keyframes" workflow: after the
/// conversion, the motion is editable through the regular animation system and
/// no longer depends on a live physics runtime.
///
/// Returns the number of object keyframes written. Existing keys at the same
/// frames are merged/replaced for the three transform channels.
pub fn bake_physics_to_keyframes(scene: &mut Scene) -> usize {
    let Some(cache) = scene.physics_cache.as_ref() else { return 0; };
    if cache.frames.is_empty() { return 0; }

    // Work on a clone so cached world-space transforms can be converted through
    // Scene::set_world into the object's local transform, including parenting.
    let frames: Vec<(i32, Vec<PhysicsTransform>)> = cache.frames
        .iter()
        .map(|(frame, transforms)| (*frame, transforms.clone()))
        .collect();
    let mut working = scene.clone();
    let animated_objects: Vec<_> = working.animation.objects.iter()
        .map(|(id, animation)| (*id, animation.clone()))
        .collect();
    let mut written = 0usize;

    for (frame, transforms) in frames {
        // Re-evaluate non-physics animation first, matching the bake pass. This
        // matters when a rigid body is parented to an animated object.
        for (id, animation) in &animated_objects {
            let driven_by_physics = working.rigid_bodies.get(id).is_some_and(|body|
                body.enabled && body.body_type == RigidBodyType::Active
                    && body.dynamic && !body.animated
            );
            if driven_by_physics { continue; }
            let Some(index) = working.index_of(*id) else { continue; };
            let (location, rotation, scale) = animation.eval(frame as f32);
            let object = &mut working.objects[index];
            if let Some(value) = location { object.translation = value; }
            if let Some(value) = rotation { object.rotation = value; }
            if let Some(value) = scale { object.scale = value; }
        }
        let mut worlds = working.world_matrices();
        for transform in transforms {
            let Some(index) = working.index_of(transform.object_id) else { continue; };
            let (scale, _, _) = worlds[index].to_scale_rotation_translation();
            let world = Mat4::from_scale_rotation_translation(
                scale,
                transform.rotation,
                transform.translation,
            );
            working.set_world(index, world, &worlds);
            worlds = working.world_matrices();
            let object = &working.objects[index];
            let animation = scene.animation.objects
                .entry(transform.object_id)
                .or_default();
            animation.insert_or_merge(
                frame,
                Some(object.translation),
                Some(object.rotation),
                Some(object.scale),
            );
            written += 1;
        }
    }
    written
}

pub fn apply_baked_frame(scene: &mut Scene, frame: i32) -> bool {
    let Some(transforms) = scene.physics_cache.as_ref().and_then(|cache| cache.frames.get(&frame)).cloned() else { return false; };
    let mut worlds = scene.world_matrices();
    let mut changed = false;
    for transform in transforms {
        let Some(index) = scene.index_of(transform.object_id) else { continue; };
        let (scale, _, _) = worlds[index].to_scale_rotation_translation();
        let world = Mat4::from_scale_rotation_translation(scale, transform.rotation, transform.translation);
        scene.set_world(index, world, &worlds);
        changed = true;
        worlds = scene.world_matrices();
    }
    changed
}
