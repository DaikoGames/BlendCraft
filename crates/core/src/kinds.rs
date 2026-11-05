//! Non-mesh object kinds (Empty, Camera, Light) and their viewport helper geometry.
use crate::scene::Object;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmptyShape {
    PlainAxes,
    Arrows,
    SingleArrow,
    Circle,
    Cube,
    Sphere,
    Cone,
}

impl EmptyShape {
    pub const ALL: [EmptyShape; 7] = [
        EmptyShape::PlainAxes,
        EmptyShape::Arrows,
        EmptyShape::SingleArrow,
        EmptyShape::Circle,
        EmptyShape::Cube,
        EmptyShape::Sphere,
        EmptyShape::Cone,
    ];
    pub fn name(self) -> &'static str {
        match self {
            EmptyShape::PlainAxes => "Plain Axes",
            EmptyShape::Arrows => "Arrows",
            EmptyShape::SingleArrow => "Single Arrow",
            EmptyShape::Circle => "Circle",
            EmptyShape::Cube => "Cube",
            EmptyShape::Sphere => "Sphere",
            EmptyShape::Cone => "Cone",
        }
    }
}

/// Cameras look along local -Z (same as glTF / Blender).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraData {
    pub ortho: bool,
    pub fov_deg: f32,
    pub ortho_size: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraData {
    fn default() -> Self {
        Self { ortho: false, fov_deg: 40.0, ortho_size: 6.0, near: 0.1, far: 200.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightType {
    Point,
    Sun,
    Spot,
}

impl LightType {
    pub fn name(self) -> &'static str {
        match self {
            LightType::Point => "Point",
            LightType::Sun => "Sun",
            LightType::Spot => "Spot",
        }
    }
}

/// Lights shine along local -Z. Intensity: Point/Spot in candela-like units, Sun in lux-like units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightData {
    pub kind: LightType,
    pub color: [f32; 3],
    pub intensity: f32,
    /// 0 = unlimited
    pub range: f32,
    /// full cone angle in degrees (Spot)
    pub spot_angle_deg: f32,
    /// 0..1 soft edge (Spot)
    pub spot_blend: f32,
}

impl LightData {
    pub fn new(kind: LightType) -> Self {
        let intensity = match kind {
            LightType::Point => 20.0,
            LightType::Sun => 3.0,
            LightType::Spot => 30.0,
        };
        Self { kind, color: [1.0, 1.0, 1.0], intensity, range: 0.0, spot_angle_deg: 45.0, spot_blend: 0.15 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectKind {
    Mesh,
    Empty { shape: EmptyShape, size: f32 },
    Camera(CameraData),
    Light(LightData),
}

impl ObjectKind {
    pub fn is_mesh(&self) -> bool {
        matches!(self, ObjectKind::Mesh)
    }

    pub fn default_name(&self) -> &'static str {
        match self {
            ObjectKind::Mesh => "Mesh",
            ObjectKind::Empty { .. } => "Empty",
            ObjectKind::Camera(_) => "Camera",
            ObjectKind::Light(l) => match l.kind {
                LightType::Point => "Point Light",
                LightType::Sun => "Sun Light",
                LightType::Spot => "Spot Light",
            },
        }
    }
}

/// Radius (world units) used for click-picking and framing of non-mesh objects.
pub fn helper_radius(o: &Object, world: &Mat4) -> f32 {
    match &o.kind {
        ObjectKind::Mesh => 0.0,
        ObjectKind::Empty { size, .. } => {
            let (s, _, _) = world.to_scale_rotation_translation();
            size * s.abs().max_element().max(0.01)
        }
        ObjectKind::Camera(_) => 0.6,
        ObjectKind::Light(_) => 0.45,
    }
}

type Seg = (Vec3, Vec3);

fn ring(out: &mut Vec<Seg>, center: Vec3, u: Vec3, v: Vec3, r: f32, n: usize) {
    for k in 0..n {
        let a0 = k as f32 / n as f32 * TAU;
        let a1 = (k + 1) as f32 / n as f32 * TAU;
        out.push((center + r * (a0.cos() * u + a0.sin() * v), center + r * (a1.cos() * u + a1.sin() * v)));
    }
}

fn arrow(out: &mut Vec<Seg>, axis: Vec3, len: f32) {
    let tip = axis * len;
    out.push((Vec3::ZERO, tip));
    let (p1, p2) = if axis.abs().dot(Vec3::Y) > 0.9 { (Vec3::X, Vec3::Z) } else { (axis.cross(Vec3::Y).normalize(), axis.cross(axis.cross(Vec3::Y)).normalize()) };
    let back = tip - axis * len * 0.2;
    for p in [p1, -p1, p2, -p2] {
        out.push((tip, back + p * len * 0.08));
    }
}

fn empty_lines(shape: EmptyShape, s: f32) -> Vec<Seg> {
    let mut v = Vec::new();
    match shape {
        EmptyShape::PlainAxes => {
            for a in [Vec3::X, Vec3::Y, Vec3::Z] {
                v.push((-a * s, a * s));
            }
        }
        EmptyShape::Arrows => {
            for a in [Vec3::X, Vec3::Y, Vec3::Z] {
                arrow(&mut v, a, s);
            }
        }
        EmptyShape::SingleArrow => arrow(&mut v, Vec3::Z, s),
        EmptyShape::Circle => ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, s, 48),
        EmptyShape::Cube => {
            for i in 0..8u32 {
                let p = |k: u32| Vec3::new(if k & 1 == 0 { -s } else { s }, if k & 2 == 0 { -s } else { s }, if k & 4 == 0 { -s } else { s });
                for b in [1u32, 2, 4] {
                    if i & b == 0 {
                        v.push((p(i), p(i | b)));
                    }
                }
            }
        }
        EmptyShape::Sphere => {
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, s, 40);
            ring(&mut v, Vec3::ZERO, Vec3::Y, Vec3::Z, s, 40);
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Z, s, 40);
        }
        EmptyShape::Cone => {
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Z, s, 32);
            let apex = Vec3::new(0.0, s * 2.0, 0.0);
            for p in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
                v.push((p * s, apex));
            }
        }
    }
    v
}

fn camera_lines(c: &CameraData) -> Vec<Seg> {
    let d = 1.0;
    let mut v = Vec::new();
    if c.ortho {
        let hh = (c.ortho_size * 0.5).min(3.0);
        let hw = hh * 16.0 / 9.0;
        let corners = |z: f32| [Vec3::new(-hw, -hh, z), Vec3::new(hw, -hh, z), Vec3::new(hw, hh, z), Vec3::new(-hw, hh, z)];
        for z in [0.0, -d] {
            let q = corners(z);
            for k in 0..4 {
                v.push((q[k], q[(k + 1) % 4]));
            }
        }
        for k in 0..4 {
            v.push((corners(0.0)[k], corners(-d)[k]));
        }
    } else {
        let hh = (c.fov_deg.to_radians() * 0.5).tan() * d;
        let hw = hh * 16.0 / 9.0;
        let q = [Vec3::new(-hw, -hh, -d), Vec3::new(hw, -hh, -d), Vec3::new(hw, hh, -d), Vec3::new(-hw, hh, -d)];
        for k in 0..4 {
            v.push((Vec3::ZERO, q[k]));
            v.push((q[k], q[(k + 1) % 4]));
        }
        // "up" triangle
        let top = Vec3::new(0.0, hh * 1.45, -d);
        v.push((Vec3::new(-hw * 0.4, hh, -d), top));
        v.push((Vec3::new(hw * 0.4, hh, -d), top));
    }
    v
}

fn light_lines(l: &LightData) -> Vec<Seg> {
    let mut v = Vec::new();
    match l.kind {
        LightType::Point => {
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, 0.2, 24);
            ring(&mut v, Vec3::ZERO, Vec3::Y, Vec3::Z, 0.2, 24);
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Z, 0.2, 24);
        }
        LightType::Sun => {
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, 0.25, 32);
            for k in 0..8 {
                let a = k as f32 / 8.0 * TAU;
                let d = Vec3::new(a.cos(), a.sin(), 0.0);
                v.push((d * 0.33, d * 0.5));
            }
            v.push((Vec3::ZERO, Vec3::new(0.0, 0.0, -1.6)));
        }
        LightType::Spot => {
            let d = 1.2;
            let outer = (l.spot_angle_deg * 0.5).to_radians().clamp(0.02, FRAC_PI_2 - 0.05);
            let r = (d * outer.tan()).min(d * 3.0);
            let c = Vec3::new(0.0, 0.0, -d);
            ring(&mut v, c, Vec3::X, Vec3::Y, r, 40);
            for p in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y] {
                v.push((Vec3::ZERO, c + p * r));
            }
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, 0.08, 16);
        }
    }
    v
}

/// World-space line segments to draw for a non-mesh object (empty for meshes).
pub fn helper_lines(o: &Object, world: &Mat4) -> Vec<(Vec3, Vec3)> {
    let (local, with_scale) = match &o.kind {
        ObjectKind::Mesh => return Vec::new(),
        ObjectKind::Empty { shape, size } => (empty_lines(*shape, *size), true),
        ObjectKind::Camera(c) => (camera_lines(c), false),
        ObjectKind::Light(l) => (light_lines(l), false),
    };
    let m = if with_scale {
        *world
    } else {
        let (_, r, t) = world.to_scale_rotation_translation();
        Mat4::from_rotation_translation(r, t)
    };
    local.into_iter().map(|(a, b)| (m.transform_point3(a), m.transform_point3(b))).collect()
}
