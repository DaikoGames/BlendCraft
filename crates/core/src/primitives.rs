//! Parametric primitives. Objects remember their parameters so they can be edited later.
use crate::mesh::{MeshData, Vertex};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Primitive {
    Plane { size: f32 },
    Cube { size: f32 },
    Circle { radius: f32, segments: u32 },
    Grid { size: f32, x_div: u32, y_div: u32 },
    IcoSphere { radius: f32, subdivisions: u32 },
    Capsule { radius: f32, height: f32, segments: u32, rings: u32 },
    Sphere { radius: f32, segments: u32, rings: u32 },
    Cylinder { radius: f32, height: f32, segments: u32 },
    Cone { radius: f32, height: f32, segments: u32 },
    Torus { major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32 },
    /// Stylized monkey head (Suzanne-inspired)
    Monkey { size: f32 },
    /// Square-base pyramid
    Pyramid { size: f32, height: f32 },
    /// N-gonal prism (default hexagonal)
    Prism { radius: f32, height: f32, segments: u32 },
    /// Open cylinder (no caps)
    Tube { radius: f32, height: f32, segments: u32 },
    /// Bezier-like curved tube
    CurveBezier { size: f32, segments: u32 },
    /// Planar curve circle (ring)
    CurveCircle { radius: f32, segments: u32 },
    /// Straight path extruded as a thin tube
    CurvePath { length: f32, segments: u32 },
    /// Soft-ball approximation (metaball look)
    Metaball { radius: f32, subdivisions: u32 },
    /// Simple 3D text block
    Text { size: f32 },
}

impl Primitive {
    pub fn name(&self) -> &'static str {
        match self {
            Primitive::Plane { .. } => "Plane",
            Primitive::Cube { .. } => "Cube",
            Primitive::Circle { .. } => "Circle",
            Primitive::Grid { .. } => "Grid",
            Primitive::IcoSphere { .. } => "Ico Sphere",
            Primitive::Capsule { .. } => "Capsule",
            Primitive::Sphere { .. } => "Sphere",
            Primitive::Cylinder { .. } => "Cylinder",
            Primitive::Cone { .. } => "Cone",
            Primitive::Torus { .. } => "Torus",
            Primitive::Monkey { .. } => "Monkey",
            Primitive::Pyramid { .. } => "Pyramid",
            Primitive::Prism { .. } => "Prism",
            Primitive::Tube { .. } => "Tube",
            Primitive::CurveBezier { .. } => "Bezier",
            Primitive::CurveCircle { .. } => "Curve Circle",
            Primitive::CurvePath { .. } => "Path",
            Primitive::Metaball { .. } => "Metaball",
            Primitive::Text { .. } => "Text",
        }
    }

    /// Name shown in menus.
    pub fn label(&self) -> &'static str {
        match self {
            Primitive::Sphere { .. } => "UV Sphere",
            Primitive::Monkey { .. } => "Monkey",
            Primitive::CurveBezier { .. } => "Bezier",
            Primitive::CurveCircle { .. } => "Circle",
            Primitive::CurvePath { .. } => "Path",
            _ => self.name(),
        }
    }

    /// One default instance of every primitive (for the Add menu).
    pub fn all() -> Vec<Primitive> {
        vec![
            Primitive::Plane { size: 2.0 },
            Primitive::Cube { size: 1.0 },
            Primitive::Circle { radius: 0.5, segments: 32 },
            Primitive::Sphere { radius: 0.5, segments: 32, rings: 16 },
            Primitive::IcoSphere { radius: 0.5, subdivisions: 3 },
            Primitive::Cylinder { radius: 0.5, height: 1.0, segments: 32 },
            Primitive::Cone { radius: 0.5, height: 1.0, segments: 32 },
            Primitive::Torus { major_radius: 0.5, minor_radius: 0.15, major_segments: 48, minor_segments: 16 },
            Primitive::Grid { size: 2.0, x_div: 10, y_div: 10 },
            Primitive::Capsule { radius: 0.25, height: 0.5, segments: 24, rings: 8 },
            Primitive::Monkey { size: 1.0 },
            Primitive::Pyramid { size: 1.0, height: 1.0 },
            Primitive::Prism { radius: 0.5, height: 1.0, segments: 6 },
            Primitive::Tube { radius: 0.5, height: 1.0, segments: 32 },
            Primitive::Metaball { radius: 0.5, subdivisions: 3 },
            Primitive::Text { size: 0.5 },
        ]
    }

    /// Curve-like mesh objects (Add > Curve)
    pub fn curves() -> Vec<Primitive> {
        vec![
            Primitive::CurveBezier { size: 1.0, segments: 32 },
            Primitive::CurveCircle { radius: 0.5, segments: 48 },
            Primitive::CurvePath { length: 2.0, segments: 16 },
        ]
    }

    pub fn build(&self) -> MeshData {
        let mut m = match *self {
            Primitive::Plane { size } => plane(size),
            Primitive::Cube { size } => cube(size),
            Primitive::Circle { radius, segments } => circle(radius, segments),
            Primitive::Grid { size, x_div, y_div } => grid(size, x_div, y_div),
            Primitive::IcoSphere { radius, subdivisions } => icosphere(radius, subdivisions),
            Primitive::Capsule { radius, height, segments, rings } => capsule(radius, height, segments, rings),
            Primitive::Sphere { radius, segments, rings } => sphere(radius, segments, rings),
            Primitive::Cylinder { radius, height, segments } => cylinder(radius, height, segments),
            Primitive::Cone { radius, height, segments } => cone(radius, height, segments),
            Primitive::Torus { major_radius, minor_radius, major_segments, minor_segments } => {
                torus(major_radius, minor_radius, major_segments, minor_segments)
            }
            Primitive::Monkey { size } => monkey(size),
            Primitive::Pyramid { size, height } => pyramid(size, height),
            Primitive::Prism { radius, height, segments } => prism(radius, height, segments),
            Primitive::Tube { radius, height, segments } => tube(radius, height, segments),
            Primitive::CurveBezier { size, segments } => curve_bezier(size, segments),
            Primitive::CurveCircle { radius, segments } => curve_circle(radius, segments),
            Primitive::CurvePath { length, segments } => curve_path(length, segments),
            Primitive::Metaball { radius, subdivisions } => icosphere(radius, subdivisions),
            Primitive::Text { size } => text_mesh(size),
        };
        m.name = self.name().into();
        fix_winding(&mut m);
        // Curved primitives default to smooth shading
        match self {
            Primitive::Sphere { .. }
            | Primitive::IcoSphere { .. }
            | Primitive::Cylinder { .. }
            | Primitive::Cone { .. }
            | Primitive::Torus { .. }
            | Primitive::Capsule { .. }
            | Primitive::Monkey { .. }
            | Primitive::Metaball { .. }
            | Primitive::Tube { .. }
            | Primitive::CurveBezier { .. }
            | Primitive::CurveCircle { .. }
            | Primitive::CurvePath { .. } => m.shade_smooth(),
            _ => {}
        }
        m
    }
}

fn v(p: Vec3, n: Vec3, uv: [f32; 2]) -> Vertex {
    Vertex { position: p.to_array(), normal: n.to_array(), uv }
}

/// Make triangle winding (CCW = front) agree with the vertex normals.
fn fix_winding(m: &mut MeshData) {
    let verts = &m.vertices;
    for t in m.indices.chunks_exact_mut(3) {
        let p = |i: u32| Vec3::from(verts[i as usize].position);
        let n = |i: u32| Vec3::from(verts[i as usize].normal);
        let face = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        if face.dot(n(t[0]) + n(t[1]) + n(t[2])) < 0.0 {
            t.swap(1, 2);
        }
    }
}

pub fn plane(size: f32) -> MeshData {
    let h = size * 0.5;
    let n = Vec3::Y;
    MeshData {
        name: "Plane".into(),
        vertices: vec![
            v(Vec3::new(-h, 0.0, -h), n, [0.0, 0.0]),
            v(Vec3::new(h, 0.0, -h), n, [1.0, 0.0]),
            v(Vec3::new(h, 0.0, h), n, [1.0, 1.0]),
            v(Vec3::new(-h, 0.0, h), n, [0.0, 1.0]),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        ..Default::default()
    }
}

pub fn cube(size: f32) -> MeshData {
    let h = size * 0.5;
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    let corners = [(-1.0, -1.0, [0.0, 0.0]), (1.0, -1.0, [1.0, 0.0]), (1.0, 1.0, [1.0, 1.0]), (-1.0, 1.0, [0.0, 1.0])];
    let mut m = MeshData { name: "Cube".into(), ..Default::default() };
    for (n, t, b) in faces {
        let base = m.vertices.len() as u32;
        for (s, u, uv) in corners {
            m.vertices.push(v(n * h + t * (h * s) + b * (h * u), n, uv));
        }
        m.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    m
}

fn sphere(r: f32, seg: u32, rings: u32) -> MeshData {
    let (seg, rings) = (seg.max(3), rings.max(2));
    let mut m = MeshData::default();
    for ri in 0..=rings {
        let tv = ri as f32 / rings as f32;
        let th = tv * PI;
        for si in 0..=seg {
            let tu = si as f32 / seg as f32;
            let ph = tu * TAU;
            let n = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
            m.vertices.push(v(n * r, n, [tu, tv]));
        }
    }
    for ri in 0..rings {
        for si in 0..seg {
            let a = ri * (seg + 1) + si;
            let b = a + seg + 1;
            m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    m
}

fn cap(m: &mut MeshData, r: f32, y: f32, ny: f32, seg: u32) {
    let center = m.vertices.len() as u32;
    let n = Vec3::new(0.0, ny, 0.0);
    m.vertices.push(v(Vec3::new(0.0, y, 0.0), n, [0.5, 0.5]));
    let ring = m.vertices.len() as u32;
    for si in 0..=seg {
        let ph = si as f32 / seg as f32 * TAU;
        m.vertices.push(v(Vec3::new(r * ph.cos(), y, r * ph.sin()), n, [0.5 + 0.5 * ph.cos(), 0.5 + 0.5 * ph.sin()]));
    }
    for si in 0..seg {
        m.indices.extend_from_slice(&[center, ring + si, ring + si + 1]);
    }
}

fn cylinder(r: f32, h: f32, seg: u32) -> MeshData {
    let seg = seg.max(3);
    let mut m = MeshData::default();
    for si in 0..=seg {
        let u = si as f32 / seg as f32;
        let ph = u * TAU;
        let n = Vec3::new(ph.cos(), 0.0, ph.sin());
        m.vertices.push(v(Vec3::new(r * ph.cos(), h * 0.5, r * ph.sin()), n, [u, 0.0]));
        m.vertices.push(v(Vec3::new(r * ph.cos(), -h * 0.5, r * ph.sin()), n, [u, 1.0]));
    }
    for si in 0..seg {
        let (a, b, c, d) = (2 * si, 2 * si + 1, 2 * si + 2, 2 * si + 3);
        m.indices.extend_from_slice(&[a, b, c, c, b, d]);
    }
    cap(&mut m, r, h * 0.5, 1.0, seg);
    cap(&mut m, r, -h * 0.5, -1.0, seg);
    m
}

fn cone(r: f32, h: f32, seg: u32) -> MeshData {
    let seg = seg.max(3);
    let mut m = MeshData::default();
    for si in 0..=seg {
        let u = si as f32 / seg as f32;
        let ph = u * TAU;
        let n = Vec3::new(h * ph.cos(), r, h * ph.sin()).normalize_or_zero();
        m.vertices.push(v(Vec3::new(0.0, h * 0.5, 0.0), n, [u, 0.0]));
        m.vertices.push(v(Vec3::new(r * ph.cos(), -h * 0.5, r * ph.sin()), n, [u, 1.0]));
    }
    for si in 0..seg {
        m.indices.extend_from_slice(&[2 * si, 2 * si + 1, 2 * si + 3]);
    }
    cap(&mut m, r, -h * 0.5, -1.0, seg);
    m
}

fn torus(big: f32, small: f32, maj: u32, min: u32) -> MeshData {
    let (maj, min) = (maj.max(3), min.max(3));
    let mut m = MeshData::default();
    for i in 0..=maj {
        let tu = i as f32 / maj as f32;
        let u = tu * TAU;
        for j in 0..=min {
            let tv = j as f32 / min as f32;
            let w = tv * TAU;
            let n = Vec3::new(w.cos() * u.cos(), w.sin(), w.cos() * u.sin());
            let center = Vec3::new(big * u.cos(), 0.0, big * u.sin());
            m.vertices.push(v(center + n * small, n, [tu, tv]));
        }
    }
    for i in 0..maj {
        for j in 0..min {
            let a = i * (min + 1) + j;
            let b = a + min + 1;
            m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    m
}

fn circle(r: f32, seg: u32) -> MeshData {
    let seg = seg.max(3);
    let mut m = MeshData::default();
    let n = Vec3::Y;
    m.vertices.push(v(Vec3::ZERO, n, [0.5, 0.5]));
    for si in 0..=seg {
        let ph = si as f32 / seg as f32 * TAU;
        m.vertices.push(v(Vec3::new(r * ph.cos(), 0.0, r * ph.sin()), n, [0.5 + 0.5 * ph.cos(), 0.5 + 0.5 * ph.sin()]));
    }
    for si in 0..seg {
        m.indices.extend_from_slice(&[0, 1 + si, 2 + si]);
    }
    m
}

fn grid(size: f32, xd: u32, yd: u32) -> MeshData {
    let (xd, yd) = (xd.max(1), yd.max(1));
    let mut m = MeshData::default();
    for j in 0..=yd {
        for i in 0..=xd {
            let (u, w) = (i as f32 / xd as f32, j as f32 / yd as f32);
            m.vertices.push(v(Vec3::new((u - 0.5) * size, 0.0, (w - 0.5) * size), Vec3::Y, [u, w]));
        }
    }
    for j in 0..yd {
        for i in 0..xd {
            let a = j * (xd + 1) + i;
            let b = a + xd + 1;
            m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    m
}

fn icosphere(r: f32, sub: u32) -> MeshData {
    let sub = sub.clamp(1, 6);
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut pts: Vec<Vec3> = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..sub {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut mid = |a: u32, b: u32, pts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *cache.entry(key).or_insert_with(|| {
                pts.push(((pts[a as usize] + pts[b as usize]) * 0.5).normalize());
                (pts.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = mid(a, b, &mut pts);
            let bc = mid(b, c, &mut pts);
            let ca = mid(c, a, &mut pts);
            next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let mut m = MeshData::default();
    for d in &pts {
        // note: spherical UVs have a seam and pole distortion (same as most ico-sphere generators)
        let u = d.z.atan2(d.x) / TAU + 0.5;
        let w = d.y.clamp(-1.0, 1.0).acos() / PI;
        m.vertices.push(v(*d * r, *d, [u, w]));
    }
    for t in tris {
        m.indices.extend_from_slice(&t);
    }
    m
}

fn capsule(r: f32, h: f32, seg: u32, rings: u32) -> MeshData {
    let (seg, rings) = (seg.max(3), rings.max(2));
    // profile: (y, ring radius, normal.xz scale, normal.y)
    let mut prof: Vec<(f32, f32, f32, f32)> = Vec::new();
    for i in 0..=rings {
        let th = i as f32 / rings as f32 * (PI * 0.5);
        prof.push((h * 0.5 + r * th.cos(), r * th.sin(), th.sin(), th.cos()));
    }
    for i in 0..=rings {
        let th = PI * 0.5 + i as f32 / rings as f32 * (PI * 0.5);
        prof.push((-h * 0.5 + r * th.cos(), r * th.sin(), th.sin(), th.cos()));
    }
    let mut m = MeshData::default();
    let last = (prof.len() - 1) as f32;
    for (pi, &(y, rr, nxz, ny)) in prof.iter().enumerate() {
        for si in 0..=seg {
            let u = si as f32 / seg as f32;
            let ph = u * TAU;
            m.vertices.push(v(
                Vec3::new(rr * ph.cos(), y, rr * ph.sin()),
                Vec3::new(nxz * ph.cos(), ny, nxz * ph.sin()),
                [u, pi as f32 / last],
            ));
        }
    }
    for pi in 0..prof.len() as u32 - 1 {
        for si in 0..seg {
            let a = pi * (seg + 1) + si;
            let b = a + seg + 1;
            m.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_shapes_face_outward() {
        for p in [
            Primitive::Cube { size: 1.0 },
            Primitive::Sphere { radius: 1.0, segments: 16, rings: 8 },
            Primitive::Cylinder { radius: 1.0, height: 2.0, segments: 16 },
            Primitive::Cone { radius: 1.0, height: 2.0, segments: 16 },
            Primitive::IcoSphere { radius: 1.0, subdivisions: 2 },
            Primitive::Capsule { radius: 0.5, height: 1.0, segments: 12, rings: 4 },
        ] {
            let m = p.build();
            for t in m.indices.chunks_exact(3) {
                let q = |i: u32| Vec3::from(m.vertices[i as usize].position);
                let n = (q(t[1]) - q(t[0])).cross(q(t[2]) - q(t[0]));
                if n.length() < 1e-6 {
                    continue;
                }
                let c = (q(t[0]) + q(t[1]) + q(t[2])) / 3.0;
                assert!(n.dot(c) > -1e-4, "{} has an inward-facing triangle", p.name());
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Extra Blender-style primitives
// ---------------------------------------------------------------------------

fn pyramid(size: f32, height: f32) -> MeshData {
    let h = size * 0.5;
    let apex = Vec3::new(0.0, height, 0.0);
    let base = [
        Vec3::new(-h, 0.0, -h),
        Vec3::new(h, 0.0, -h),
        Vec3::new(h, 0.0, h),
        Vec3::new(-h, 0.0, h),
    ];
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    // base (down)
    let bn = Vec3::NEG_Y;
    let bi = vertices.len() as u32;
    for (i, p) in base.iter().enumerate() {
        let uu = if i == 1 || i == 2 { 1.0 } else { 0.0 };
        let vv = if i >= 2 { 1.0 } else { 0.0 };
        vertices.push(v(*p, bn, [uu, vv]));
    }
    indices.extend_from_slice(&[bi, bi + 2, bi + 1, bi, bi + 3, bi + 2]);
    // sides
    for i in 0..4 {
        let a = base[i];
        let b = base[(i + 1) % 4];
        let n = (b - a).cross(apex - a).normalize_or_zero();
        let i0 = vertices.len() as u32;
        vertices.push(v(a, n, [0.0, 0.0]));
        vertices.push(v(b, n, [1.0, 0.0]));
        vertices.push(v(apex, n, [0.5, 1.0]));
        indices.extend_from_slice(&[i0, i0 + 1, i0 + 2]);
    }
    MeshData { name: "Pyramid".into(), vertices, indices, ..Default::default() }
}

fn prism(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let hy = height * 0.5;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    // side faces
    for i in 0..seg {
        let a0 = i as f32 / seg as f32 * TAU;
        let a1 = (i + 1) as f32 / seg as f32 * TAU;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();
        let p0 = Vec3::new(c0 * radius, -hy, s0 * radius);
        let p1 = Vec3::new(c1 * radius, -hy, s1 * radius);
        let p2 = Vec3::new(c1 * radius, hy, s1 * radius);
        let p3 = Vec3::new(c0 * radius, hy, s0 * radius);
        let n = Vec3::new((c0 + c1) * 0.5, 0.0, (s0 + s1) * 0.5).normalize_or_zero();
        let i0 = vertices.len() as u32;
        vertices.extend([v(p0, n, [0.0, 0.0]), v(p1, n, [1.0, 0.0]), v(p2, n, [1.0, 1.0]), v(p3, n, [0.0, 1.0])]);
        indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]);
    }
    // caps
    for (y, n, flip) in [(-hy, Vec3::NEG_Y, true), (hy, Vec3::Y, false)] {
        let center = vertices.len() as u32;
        vertices.push(v(Vec3::new(0.0, y, 0.0), n, [0.5, 0.5]));
        for i in 0..=seg {
            let a = i as f32 / seg as f32 * TAU;
            let (s, c) = a.sin_cos();
            vertices.push(v(Vec3::new(c * radius, y, s * radius), n, [c * 0.5 + 0.5, s * 0.5 + 0.5]));
        }
        for i in 0..seg {
            let a = center + 1 + i;
            let b = center + 2 + i;
            if flip {
                indices.extend_from_slice(&[center, b, a]);
            } else {
                indices.extend_from_slice(&[center, a, b]);
            }
        }
    }
    MeshData { name: "Prism".into(), vertices, indices, ..Default::default() }
}

fn tube(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let hy = height * 0.5;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..seg {
        let a0 = i as f32 / seg as f32 * TAU;
        let a1 = (i + 1) as f32 / seg as f32 * TAU;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();
        let n0 = Vec3::new(c0, 0.0, s0);
        let n1 = Vec3::new(c1, 0.0, s1);
        let p0 = Vec3::new(c0 * radius, -hy, s0 * radius);
        let p1 = Vec3::new(c1 * radius, -hy, s1 * radius);
        let p2 = Vec3::new(c1 * radius, hy, s1 * radius);
        let p3 = Vec3::new(c0 * radius, hy, s0 * radius);
        let i0 = vertices.len() as u32;
        vertices.extend([
            v(p0, n0, [0.0, 0.0]),
            v(p1, n1, [1.0, 0.0]),
            v(p2, n1, [1.0, 1.0]),
            v(p3, n0, [0.0, 1.0]),
        ]);
        indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]);
    }
    MeshData { name: "Tube".into(), vertices, indices, ..Default::default() }
}

fn tube_along(points: &[Vec3], radius: f32, radial: u32) -> MeshData {
    let radial = radial.max(3);
    if points.len() < 2 {
        return MeshData { name: "Curve".into(), ..Default::default() };
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut frames = Vec::new();
    // Parallel transport frames
    let mut t = (points[1] - points[0]).normalize_or_zero();
    let mut n = if t.y.abs() < 0.9 { t.cross(Vec3::Y).normalize_or_zero() } else { t.cross(Vec3::X).normalize_or_zero() };
    let mut b = t.cross(n);
    for i in 0..points.len() {
        if i > 0 {
            let t2 = if i + 1 < points.len() {
                (points[i + 1] - points[i]).normalize_or_zero()
            } else {
                (points[i] - points[i - 1]).normalize_or_zero()
            };
            if t2.length_squared() > 1e-8 && t.length_squared() > 1e-8 {
                let axis = t.cross(t2);
                let axis_len = axis.length();
                if axis_len > 1e-6 {
                    let ang = t.dot(t2).clamp(-1.0, 1.0).acos();
                    let q = Quat::from_axis_angle(axis / axis_len, ang);
                    n = q * n;
                    b = q * b;
                }
                t = t2;
            }
        }
        frames.push((n, b));
    }
    for (i, p) in points.iter().enumerate() {
        let (fn_, fb) = frames[i];
        let v_coord = i as f32 / (points.len() - 1) as f32;
        for j in 0..radial {
            let a = j as f32 / radial as f32 * TAU;
            let (s, c) = a.sin_cos();
            let offset = (fn_ * c + fb * s) * radius;
            let normal = offset.normalize_or_zero();
            vertices.push(v(*p + offset, normal, [j as f32 / radial as f32, v_coord]));
        }
    }
    let radial_us = radial as usize;
    for i in 0..points.len() - 1 {
        for j in 0..radial_us {
            let j2 = (j + 1) % radial_us;
            let a = (i * radial_us + j) as u32;
            let b = (i * radial_us + j2) as u32;
            let c = ((i + 1) * radial_us + j2) as u32;
            let d = ((i + 1) * radial_us + j) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    MeshData { name: "Curve".into(), vertices, indices, ..Default::default() }
}

fn curve_bezier(size: f32, segments: u32) -> MeshData {
    let seg = segments.max(4);
    let mut pts = Vec::new();
    // Classic S-shaped cubic bezier
    let p0 = Vec3::new(-size, 0.0, 0.0);
    let p1 = Vec3::new(-size * 0.3, size, 0.0);
    let p2 = Vec3::new(size * 0.3, -size, 0.0);
    let p3 = Vec3::new(size, 0.0, 0.0);
    for i in 0..=seg {
        let t = i as f32 / seg as f32;
        let u = 1.0 - t;
        let p = p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t);
        pts.push(p);
    }
    let mut m = tube_along(&pts, size * 0.04, 8);
    m.name = "Bezier".into();
    m
}

fn curve_circle(radius: f32, segments: u32) -> MeshData {
    let seg = segments.max(8);
    let mut pts = Vec::new();
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        pts.push(Vec3::new(c * radius, 0.0, s * radius));
    }
    let mut m = tube_along(&pts, radius * 0.04, 8);
    m.name = "Curve Circle".into();
    m
}

fn curve_path(length: f32, segments: u32) -> MeshData {
    let seg = segments.max(2);
    let mut pts = Vec::new();
    for i in 0..=seg {
        let t = i as f32 / seg as f32;
        pts.push(Vec3::new((t - 0.5) * length, 0.0, 0.0));
    }
    let mut m = tube_along(&pts, length * 0.02, 8);
    m.name = "Path".into();
    m
}

fn text_mesh(size: f32) -> MeshData {
    // Simple block letter "A" as a stand-in for Text objects
    let s = size;
    let t = size * 0.18; // thickness
    let d = size * 0.2;  // depth
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    // Helper to add an axis-aligned box
    let mut add_box = |min: Vec3, max: Vec3| {
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let faces = [
            ([0, 1, 2, 3], Vec3::NEG_Z),
            ([4, 7, 6, 5], Vec3::Z),
            ([0, 4, 5, 1], Vec3::NEG_Y),
            ([2, 6, 7, 3], Vec3::Y),
            ([0, 3, 7, 4], Vec3::NEG_X),
            ([1, 5, 6, 2], Vec3::X),
        ];
        for (idx, n) in faces {
            let i0 = vertices.len() as u32;
            for &k in &idx {
                vertices.push(v(corners[k], n, [0.0, 0.0]));
            }
            indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]);
        }
    };
    // Left leg
    add_box(Vec3::new(-s * 0.45, 0.0, -d * 0.5), Vec3::new(-s * 0.45 + t, s, d * 0.5));
    // Right leg
    add_box(Vec3::new(s * 0.45 - t, 0.0, -d * 0.5), Vec3::new(s * 0.45, s, d * 0.5));
    // Crossbar
    add_box(Vec3::new(-s * 0.35, s * 0.4, -d * 0.5), Vec3::new(s * 0.35, s * 0.4 + t, d * 0.5));
    // Top bar
    add_box(Vec3::new(-s * 0.45, s - t, -d * 0.5), Vec3::new(s * 0.45, s, d * 0.5));
    MeshData { name: "Text".into(), vertices, indices, ..Default::default() }
}

fn monkey(size: f32) -> MeshData {
    // Stylized low-poly monkey head (Suzanne-inspired): head sphere + ears + snout
    let mut m = icosphere(size * 0.45, 2);
    // Scale head slightly
    for v in &mut m.vertices {
        v.position[0] *= 1.05;
        v.position[1] *= 0.95;
        v.position[2] *= 0.9;
        v.position[1] += size * 0.1;
    }
    // Ears as flattened spheres on the sides
    for side in [-1.0f32, 1.0] {
        let ear = icosphere(size * 0.22, 1);
        let ox = side * size * 0.48;
        let oy = size * 0.2;
        let oz = size * -0.05;
        let base = m.vertices.len() as u32;
        for mut v in ear.vertices {
            v.position[0] = v.position[0] * 0.45 + ox;
            v.position[1] = v.position[1] + oy;
            v.position[2] = v.position[2] * 1.1 + oz;
            m.vertices.push(v);
        }
        for idx in ear.indices {
            m.indices.push(base + idx);
        }
    }
    // Snout
    let snout = icosphere(size * 0.22, 1);
    let base = m.vertices.len() as u32;
    for mut v in snout.vertices {
        v.position[0] *= 0.9;
        v.position[1] = v.position[1] * 0.7 + size * 0.0;
        v.position[2] = v.position[2] * 0.85 + size * 0.38;
        m.vertices.push(v);
    }
    for idx in snout.indices {
        m.indices.push(base + idx);
    }
    m.name = "Monkey".into();
    m
}
