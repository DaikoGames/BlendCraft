//! Non-mesh object kinds (Empty, Camera, Light) and their viewport helper geometry.
use crate::armature::ArmatureData;
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
        Self {
            ortho: false,
            fov_deg: 40.0,
            ortho_size: 6.0,
            near: 0.1,
            far: 200.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightType {
    Point,
    Sun,
    Spot,
    Area,
}

impl LightType {
    pub fn name(self) -> &'static str {
        match self {
            LightType::Point => "Point",
            LightType::Sun => "Sun",
            LightType::Spot => "Spot",
            LightType::Area => "Area",
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
        // Point/Spot: candela-like; Sun: lux-like. Values kept moderate so a white
        // material next to the light does not blow out to pure white after tonemapping.
        // Range 0 = unlimited; a finite default range gives natural falloff for lamps.
        let (intensity, range) = match kind {
            LightType::Point => (20.0, 25.0),
            LightType::Sun => (5.0, 0.0),
            LightType::Spot => (25.0, 30.0),
            LightType::Area => (30.0, 40.0),
        };
        Self {
            kind,
            color: [1.0, 1.0, 1.0],
            intensity,
            range,
            spot_angle_deg: 45.0,
            spot_blend: 0.15,
        }
    }
}

/// NURBS-style surface placeholder (viewport grid patch).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceData {
    pub size: f32,
    pub u_segments: u32,
    pub v_segments: u32,
}

impl Default for SurfaceData {
    fn default() -> Self {
        Self {
            size: 2.0,
            u_segments: 8,
            v_segments: 8,
        }
    }
}

/// Volumetric domain box (smoke / fog / SDF placeholder).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VolumeData {
    pub size: [f32; 3],
    pub density: f32,
}

impl Default for VolumeData {
    fn default() -> Self {
        Self {
            size: [2.0, 2.0, 2.0],
            density: 1.0,
        }
    }
}

/// 2D drawing layer object (Grease Pencil style).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GreasePencilData {
    pub thickness: f32,
    pub layers: u32,
}

impl Default for GreasePencilData {
    fn default() -> Self {
        Self {
            thickness: 0.02,
            layers: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightProbeType {
    Reflection,
    Irradiance,
}

impl LightProbeType {
    pub fn name(self) -> &'static str {
        match self {
            LightProbeType::Reflection => "Reflection",
            LightProbeType::Irradiance => "Irradiance",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightProbeData {
    pub kind: LightProbeType,
    pub radius: f32,
    pub intensity: f32,
}

impl Default for LightProbeData {
    fn default() -> Self {
        Self {
            kind: LightProbeType::Reflection,
            radius: 1.0,
            intensity: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeakerData {
    pub volume: f32,
    pub distance: f32,
    pub muted: bool,
}

impl Default for SpeakerData {
    fn default() -> Self {
        Self {
            volume: 1.0,
            distance: 10.0,
            muted: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForceFieldType {
    Force,
    Wind,
    Vortex,
    Magnetic,
    Turbulence,
}

impl ForceFieldType {
    pub fn name(self) -> &'static str {
        match self {
            ForceFieldType::Force => "Force",
            ForceFieldType::Wind => "Wind",
            ForceFieldType::Vortex => "Vortex",
            ForceFieldType::Magnetic => "Magnetic",
            ForceFieldType::Turbulence => "Turbulence",
        }
    }
    pub const ALL: [ForceFieldType; 5] = [
        ForceFieldType::Force,
        ForceFieldType::Wind,
        ForceFieldType::Vortex,
        ForceFieldType::Magnetic,
        ForceFieldType::Turbulence,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForceFieldData {
    pub kind: ForceFieldType,
    pub strength: f32,
    pub size: f32,
}

impl Default for ForceFieldData {
    fn default() -> Self {
        Self {
            kind: ForceFieldType::Force,
            strength: 1.0,
            size: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectKind {
    Mesh,
    Empty { shape: EmptyShape, size: f32 },
    Camera(CameraData),
    Light(LightData),
    Surface(SurfaceData),
    Volume(VolumeData),
    GreasePencil(GreasePencilData),
    Armature(ArmatureData),
    LightProbe(LightProbeData),
    Speaker(SpeakerData),
    ForceField(ForceFieldData),
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
                LightType::Area => "Area Light",
            },
            ObjectKind::Surface(_) => "Surface",
            ObjectKind::Volume(_) => "Volume",
            ObjectKind::GreasePencil(_) => "Grease Pencil",
            ObjectKind::Armature(_) => "Armature",
            ObjectKind::LightProbe(p) => match p.kind {
                LightProbeType::Reflection => "Reflection Probe",
                LightProbeType::Irradiance => "Irradiance Volume",
            },
            ObjectKind::Speaker(_) => "Speaker",
            ObjectKind::ForceField(f) => match f.kind {
                ForceFieldType::Force => "Force Field",
                ForceFieldType::Wind => "Wind",
                ForceFieldType::Vortex => "Vortex",
                ForceFieldType::Magnetic => "Magnetic",
                ForceFieldType::Turbulence => "Turbulence",
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
        ObjectKind::Surface(s) => s.size * 0.7,
        ObjectKind::Volume(v) => v.size[0].max(v.size[1]).max(v.size[2]) * 0.6,
        ObjectKind::GreasePencil(_) => 0.8,
        ObjectKind::Armature(a) => {
            let n = a.bones.len().max(1) as f32;
            a.bones.first().map(|b| b.length()).unwrap_or(0.5) * n * 0.5
        }
        ObjectKind::LightProbe(p) => p.radius,
        ObjectKind::Speaker(_) => 0.5,
        ObjectKind::ForceField(f) => f.size,
    }
}

type Seg = (Vec3, Vec3);

fn ring(out: &mut Vec<Seg>, center: Vec3, u: Vec3, v: Vec3, r: f32, n: usize) {
    for k in 0..n {
        let a0 = k as f32 / n as f32 * TAU;
        let a1 = (k + 1) as f32 / n as f32 * TAU;
        out.push((
            center + r * (a0.cos() * u + a0.sin() * v),
            center + r * (a1.cos() * u + a1.sin() * v),
        ));
    }
}

fn arrow(out: &mut Vec<Seg>, axis: Vec3, len: f32) {
    let tip = axis * len;
    out.push((Vec3::ZERO, tip));
    let (p1, p2) = if axis.abs().dot(Vec3::Y) > 0.9 {
        (Vec3::X, Vec3::Z)
    } else {
        (
            axis.cross(Vec3::Y).normalize(),
            axis.cross(axis.cross(Vec3::Y)).normalize(),
        )
    };
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
                let p = |k: u32| {
                    Vec3::new(
                        if k & 1 == 0 { -s } else { s },
                        if k & 2 == 0 { -s } else { s },
                        if k & 4 == 0 { -s } else { s },
                    )
                };
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
        let corners = |z: f32| {
            [
                Vec3::new(-hw, -hh, z),
                Vec3::new(hw, -hh, z),
                Vec3::new(hw, hh, z),
                Vec3::new(-hw, hh, z),
            ]
        };
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
        let q = [
            Vec3::new(-hw, -hh, -d),
            Vec3::new(hw, -hh, -d),
            Vec3::new(hw, hh, -d),
            Vec3::new(-hw, hh, -d),
        ];
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
            let outer = (l.spot_angle_deg * 0.5)
                .to_radians()
                .clamp(0.02, FRAC_PI_2 - 0.05);
            let r = (d * outer.tan()).min(d * 3.0);
            let c = Vec3::new(0.0, 0.0, -d);
            ring(&mut v, c, Vec3::X, Vec3::Y, r, 40);
            for p in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y] {
                v.push((Vec3::ZERO, c + p * r));
            }
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, 0.08, 16);
        }
        LightType::Area => {
            // Rectangle helper in the XY plane facing -Z
            let hx = 0.5;
            let hy = 0.35;
            let corners = [
                Vec3::new(-hx, -hy, 0.0),
                Vec3::new(hx, -hy, 0.0),
                Vec3::new(hx, hy, 0.0),
                Vec3::new(-hx, hy, 0.0),
            ];
            for i in 0..4 {
                v.push((corners[i], corners[(i + 1) % 4]));
            }
            // Direction indicator
            v.push((Vec3::ZERO, Vec3::new(0.0, 0.0, -0.6)));
            // Cross
            v.push((
                Vec3::new(-hx * 0.3, 0.0, 0.0),
                Vec3::new(hx * 0.3, 0.0, 0.0),
            ));
            v.push((
                Vec3::new(0.0, -hy * 0.3, 0.0),
                Vec3::new(0.0, hy * 0.3, 0.0),
            ));
        }
    }
    v
}

/// World-space line segments to draw for a non-mesh object (empty for meshes).

fn surface_lines(s: &SurfaceData) -> Vec<Seg> {
    let mut v = Vec::new();
    let h = s.size * 0.5;
    let nu = s.u_segments.max(1);
    let nv = s.v_segments.max(1);
    for i in 0..=nu {
        let u = -h + 2.0 * h * (i as f32 / nu as f32);
        v.push((Vec3::new(u, 0.0, -h), Vec3::new(u, 0.0, h)));
    }
    for j in 0..=nv {
        let w = -h + 2.0 * h * (j as f32 / nv as f32);
        v.push((Vec3::new(-h, 0.0, w), Vec3::new(h, 0.0, w)));
    }
    v
}

fn volume_lines(vol: &VolumeData) -> Vec<Seg> {
    let hx = vol.size[0] * 0.5;
    let hy = vol.size[1] * 0.5;
    let hz = vol.size[2] * 0.5;
    let corners = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    edges
        .iter()
        .map(|&(a, b)| (corners[a], corners[b]))
        .collect()
}

fn grease_pencil_lines(g: &GreasePencilData) -> Vec<Seg> {
    let mut v = Vec::new();
    // Frame + a simple stroke path
    let s = 0.8;
    let frame = [
        Vec3::new(-s, -s, 0.0),
        Vec3::new(s, -s, 0.0),
        Vec3::new(s, s, 0.0),
        Vec3::new(-s, s, 0.0),
    ];
    for i in 0..4 {
        v.push((frame[i], frame[(i + 1) % 4]));
    }
    // Sample stroke (sine wave)
    let n = 24;
    for i in 0..n {
        let t0 = i as f32 / n as f32;
        let t1 = (i + 1) as f32 / n as f32;
        let y0 = (t0 * TAU).sin() * 0.35;
        let y1 = (t1 * TAU).sin() * 0.35;
        v.push((
            Vec3::new(-0.6 + t0 * 1.2, y0, g.thickness),
            Vec3::new(-0.6 + t1 * 1.2, y1, g.thickness),
        ));
    }
    v
}

fn armature_lines(a: &ArmatureData) -> Vec<Seg> {
    a.lines()
}

fn light_probe_lines(p: &LightProbeData) -> Vec<Seg> {
    let mut v = Vec::new();
    let r = p.radius.max(0.1);
    match p.kind {
        LightProbeType::Reflection => {
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, r, 32);
            ring(&mut v, Vec3::ZERO, Vec3::Y, Vec3::Z, r, 32);
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Z, r, 32);
        }
        LightProbeType::Irradiance => {
            // grid box
            let s = r;
            let corners = [
                Vec3::new(-s, -s, -s),
                Vec3::new(s, -s, -s),
                Vec3::new(s, s, -s),
                Vec3::new(-s, s, -s),
                Vec3::new(-s, -s, s),
                Vec3::new(s, -s, s),
                Vec3::new(s, s, s),
                Vec3::new(-s, s, s),
            ];
            for (a, b) in [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ] {
                v.push((corners[a], corners[b]));
            }
            // inner grid
            v.push((Vec3::new(-s, 0.0, -s), Vec3::new(s, 0.0, -s)));
            v.push((Vec3::new(-s, 0.0, s), Vec3::new(s, 0.0, s)));
            v.push((Vec3::new(0.0, -s, -s), Vec3::new(0.0, s, -s)));
            v.push((Vec3::new(0.0, -s, s), Vec3::new(0.0, s, s)));
        }
    }
    v
}

fn speaker_lines(s: &SpeakerData) -> Vec<Seg> {
    let mut v = Vec::new();
    // Cone + body
    let body_r = 0.15;
    let cone_r = 0.35;
    let depth = 0.25;
    ring(
        &mut v,
        Vec3::new(0.0, 0.0, -depth),
        Vec3::X,
        Vec3::Y,
        body_r,
        16,
    );
    ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, cone_r, 24);
    for k in 0..8 {
        let a = k as f32 / 8.0 * TAU;
        let p0 = Vec3::new(body_r * a.cos(), body_r * a.sin(), -depth);
        let p1 = Vec3::new(cone_r * a.cos(), cone_r * a.sin(), 0.0);
        v.push((p0, p1));
    }
    // Sound waves
    let reach = s.distance.min(5.0) * 0.15;
    for i in 1..=3 {
        let rr = cone_r + reach * (i as f32 / 3.0);
        ring(
            &mut v,
            Vec3::new(0.0, 0.0, 0.05 * i as f32),
            Vec3::X,
            Vec3::Y,
            rr,
            20,
        );
    }
    v
}

fn force_field_lines(f: &ForceFieldData) -> Vec<Seg> {
    let mut v = Vec::new();
    let s = f.size.max(0.2);
    match f.kind {
        ForceFieldType::Force => {
            arrow(&mut v, Vec3::Z, s);
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, s * 0.3, 16);
        }
        ForceFieldType::Wind => {
            for i in 0..5 {
                let y = -s * 0.5 + s * (i as f32 / 4.0);
                v.push((Vec3::new(-s * 0.3, y, 0.0), Vec3::new(s * 0.5, y, 0.0)));
                v.push((
                    Vec3::new(s * 0.35, y + 0.08, 0.0),
                    Vec3::new(s * 0.5, y, 0.0),
                ));
                v.push((
                    Vec3::new(s * 0.35, y - 0.08, 0.0),
                    Vec3::new(s * 0.5, y, 0.0),
                ));
            }
        }
        ForceFieldType::Vortex => {
            for i in 0..3 {
                let r = s * (0.3 + 0.25 * i as f32);
                ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Z, r, 24);
            }
            arrow(&mut v, Vec3::Y, s * 0.8);
        }
        ForceFieldType::Magnetic => {
            // horseshoe-ish
            ring(
                &mut v,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::X,
                Vec3::Y,
                s * 0.5,
                20,
            );
            v.push((Vec3::new(-s * 0.5, 0.0, 0.0), Vec3::new(-s * 0.5, -s, 0.0)));
            v.push((Vec3::new(s * 0.5, 0.0, 0.0), Vec3::new(s * 0.5, -s, 0.0)));
        }
        ForceFieldType::Turbulence => {
            for i in 0..6 {
                let a = i as f32 / 6.0 * TAU;
                let p = Vec3::new(a.cos(), a.sin(), (a * 2.0).sin()) * s * 0.5;
                v.push((Vec3::ZERO, p));
            }
            ring(&mut v, Vec3::ZERO, Vec3::X, Vec3::Y, s * 0.4, 16);
        }
    }
    v
}

pub fn helper_lines(o: &Object, world: &Mat4) -> Vec<(Vec3, Vec3)> {
    let (local, with_scale) = match &o.kind {
        ObjectKind::Mesh => return Vec::new(),
        ObjectKind::Empty { shape, size } => (empty_lines(*shape, *size), true),
        ObjectKind::Camera(c) => (camera_lines(c), false),
        ObjectKind::Light(l) => (light_lines(l), false),
        ObjectKind::Surface(s) => (surface_lines(s), true),
        ObjectKind::Volume(v) => (volume_lines(v), true),
        ObjectKind::GreasePencil(g) => (grease_pencil_lines(g), true),
        ObjectKind::Armature(a) => (armature_lines(a), true),
        ObjectKind::LightProbe(p) => (light_probe_lines(p), true),
        ObjectKind::Speaker(s) => (speaker_lines(s), true),
        ObjectKind::ForceField(f) => (force_field_lines(f), true),
    };
    let m = if with_scale {
        *world
    } else {
        let (_, r, t) = world.to_scale_rotation_translation();
        Mat4::from_rotation_translation(r, t)
    };
    local
        .into_iter()
        .map(|(a, b)| (m.transform_point3(a), m.transform_point3(b)))
        .collect()
}
