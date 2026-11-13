//! Sculpt Mode brushes (Blender-inspired set).
use crate::mesh::MeshData;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptBrush {
    Draw,
    DrawSharp,
    Clay,
    ClayStrips,
    Layer,
    Inflate,
    Blob,
    Crease,
    Smooth,
    Flatten,
    Fill,
    Scrape,
    Pinch,
    Grab,
    SnakeHook,
    Nudge,
    Thumb,
}

impl SculptBrush {
    pub fn name(self) -> &'static str {
        match self {
            Self::Draw => "Draw",
            Self::DrawSharp => "Draw Sharp",
            Self::Clay => "Clay",
            Self::ClayStrips => "Clay Strips",
            Self::Layer => "Layer",
            Self::Inflate => "Inflate",
            Self::Blob => "Blob",
            Self::Crease => "Crease",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Fill => "Fill",
            Self::Scrape => "Scrape",
            Self::Pinch => "Pinch",
            Self::Grab => "Grab",
            Self::SnakeHook => "Snake Hook",
            Self::Nudge => "Nudge",
            Self::Thumb => "Thumb",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SculptSettings {
    pub brush: SculptBrush,
    pub radius: f32,
    pub strength: f32,
    pub invert: bool,
    /// Dynamic topology: subdivide under the brush while sculpting.
    pub dyntopo: bool,
    /// Target edge length in local units. Smaller = more vertices.
    pub dyntopo_detail: f32,
}

impl Default for SculptSettings {
    fn default() -> Self {
        Self {
            brush: SculptBrush::Draw,
            radius: 0.35,
            strength: 0.35,
            invert: false,
            dyntopo: false,
            // Relative detail: fraction of brush radius (Blender-like).
            // 0.4 ≈ medium; smaller = denser. Typical range 0.2 .. 0.8
            dyntopo_detail: 0.4,
        }
    }
}

fn falloff(dist: f32, radius: f32) -> f32 {
    if dist >= radius || radius < 1e-8 {
        return 0.0;
    }
    let t = 1.0 - dist / radius;
    t * t * (3.0 - 2.0 * t)
}

/// Sharper falloff for Draw Sharp / Crease.
fn falloff_sharp(dist: f32, radius: f32) -> f32 {
    if dist >= radius || radius < 1e-8 {
        return 0.0;
    }
    let t = 1.0 - dist / radius;
    let t = t * t;
    t * t
}

/// Squarer falloff for Clay Strips.
fn falloff_square(dist: f32, radius: f32) -> f32 {
    if dist >= radius || radius < 1e-8 {
        return 0.0;
    }
    let t = 1.0 - dist / radius;
    t
}

fn ray_tri(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 1e-5 {
        Some(t)
    } else {
        None
    }
}

pub fn raycast_mesh(mesh: &MeshData, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3, f32)> {
    let dir = dir.normalize_or_zero();
    if dir.length_squared() < 1e-12 {
        return None;
    }
    let mut best: Option<(f32, Vec3, Vec3)> = None;
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
        if let Some(t) = ray_tri(origin, dir, a, b, c) {
            let n = (b - a).cross(c - a).normalize_or_zero();
            if best.map(|(bt, _, _)| t < bt).unwrap_or(true) {
                best = Some((t, origin + dir * t, n));
            }
        }
    }
    best.map(|(t, p, n)| (p, n, t))
}

/// Subdivide edges under the brush (Blender-like relative detail).
/// `detail_relative` is a fraction of the brush radius (e.g. 0.4).
/// Hard-capped so a single sample cannot explode the mesh.
pub fn dyntopo_refine(mesh: &mut MeshData, hit: Vec3, radius: f32, detail_relative: f32) -> u32 {
    const MAX_SPLITS_PER_SAMPLE: usize = 64;
    const MAX_VERTS: usize = 80_000;

    if mesh.vertices.len() >= MAX_VERTS {
        return 0;
    }

    let radius = radius.max(1e-4);
    // Blender Relative Detail: target edge length ≈ radius * relative
    let detail = (radius * detail_relative.clamp(0.15, 1.5)).max(radius * 0.15);
    let mut added = 0u32;

    // Single pass only — never multi-pass per sample (that was exploding density)
    let edges = {
        let mut set = std::collections::BTreeSet::new();
        for tri in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            set.insert(if a < b { (a, b) } else { (b, a) });
            set.insert(if b < c { (b, c) } else { (c, b) });
            set.insert(if c < a { (c, a) } else { (a, c) });
        }
        set
    };

    let mut splits: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for &(a, b) in &edges {
        if splits.len() >= MAX_SPLITS_PER_SAMPLE {
            break;
        }
        if mesh.vertices.len() + splits.len() >= MAX_VERTS {
            break;
        }
        let pa = Vec3::from(mesh.vertices[a as usize].position);
        let pb = Vec3::from(mesh.vertices[b as usize].position);
        let mid = (pa + pb) * 0.5;
        // Only edges whose midpoint is inside the brush
        if (mid - hit).length() > radius {
            continue;
        }
        // Only split if clearly longer than target detail
        let len = (pb - pa).length();
        if len < detail * 1.5 {
            continue;
        }
        let ni = mesh.vertices.len() as u32 + splits.len() as u32;
        splits.insert((a, b), ni);
    }

    if splits.is_empty() {
        return 0;
    }

    // Materialize new vertices in stable order of map iteration is not sorted —
    // assign indices sequentially.
    let mut ordered: Vec<((u32, u32), u32)> = splits.into_iter().collect();
    ordered.sort_by_key(|(_, ni)| *ni);
    let base = mesh.vertices.len() as u32;
    let mut remap = std::collections::HashMap::new();
    for (i, ((a, b), _)) in ordered.iter().enumerate() {
        let pa = Vec3::from(mesh.vertices[*a as usize].position);
        let pb = Vec3::from(mesh.vertices[*b as usize].position);
        let mid = (pa + pb) * 0.5;
        let mut v = mesh.vertices[*a as usize].clone();
        v.position = mid.to_array();
        mesh.vertices.push(v);
        remap.insert((*a, *b), base + i as u32);
        added += 1;
    }

    let old = mesh.indices.clone();
    let mut new_idx = Vec::with_capacity(old.len() + added as usize * 6);
    for tri in old.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let key = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };
        let ab = remap.get(&key(a, b)).copied();
        let bc = remap.get(&key(b, c)).copied();
        let ca = remap.get(&key(c, a)).copied();
        match (ab, bc, ca) {
            (None, None, None) => new_idx.extend_from_slice(tri),
            (Some(m), None, None) => new_idx.extend_from_slice(&[a, m, c, m, b, c]),
            (None, Some(m), None) => new_idx.extend_from_slice(&[a, b, m, a, m, c]),
            (None, None, Some(m)) => new_idx.extend_from_slice(&[a, b, m, b, c, m]),
            (Some(m0), Some(m1), None) => {
                new_idx.extend_from_slice(&[a, m0, c, m0, b, m1, m0, m1, c])
            }
            (None, Some(m0), Some(m1)) => {
                new_idx.extend_from_slice(&[a, b, m0, a, m0, m1, m0, c, m1])
            }
            (Some(m0), None, Some(m1)) => {
                new_idx.extend_from_slice(&[a, m0, m1, m0, b, c, m0, c, m1])
            }
            (Some(m0), Some(m1), Some(m2)) => {
                new_idx.extend_from_slice(&[a, m0, m2, m0, b, m1, m2, m1, c, m0, m1, m2])
            }
        }
    }
    mesh.indices = new_idx;
    mesh.compute_normals();
    added
}

/// Optional stroke direction (world/local) for Nudge / Thumb / SnakeHook.
pub fn apply_brush(
    mesh: &mut MeshData,
    settings: &SculptSettings,
    hit: Vec3,
    hit_normal: Vec3,
    grab_delta: Option<Vec3>,
) {
    let radius = settings.radius.max(1e-5);
    let strength = settings.strength.clamp(0.0, 1.0);
    let sign = if settings.invert { -1.0 } else { 1.0 };
    let n_hit = hit_normal.normalize_or_zero();

    // Average plane height in brush for Flatten/Fill/Scrape
    let mut plane_h = 0.0;
    let mut plane_n = 0u32;
    for v in &mesh.vertices {
        let p = Vec3::from(v.position);
        if (p - hit).length() < radius {
            plane_h += (p - hit).dot(n_hit);
            plane_n += 1;
        }
    }
    let avg_h = if plane_n > 0 {
        plane_h / plane_n as f32
    } else {
        0.0
    };

    match settings.brush {
        SculptBrush::Draw
        | SculptBrush::DrawSharp
        | SculptBrush::Clay
        | SculptBrush::ClayStrips
        | SculptBrush::Layer => {
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = match settings.brush {
                    SculptBrush::DrawSharp => falloff_sharp(d, radius),
                    SculptBrush::ClayStrips => falloff_square(d, radius),
                    _ => falloff(d, radius),
                };
                if w <= 0.0 {
                    continue;
                }
                let mut disp = n_hit * (strength * w * radius * 0.35 * sign);
                // Clay: slightly flatten toward plane while drawing
                if matches!(settings.brush, SculptBrush::Clay | SculptBrush::ClayStrips) {
                    let h = (p - hit).dot(n_hit);
                    let flatten = n_hit * ((avg_h - h) * 0.25 * w * strength);
                    disp = disp * 0.85 + flatten;
                }
                // Layer: cap displacement magnitude
                if settings.brush == SculptBrush::Layer {
                    let max_h = radius * 0.25 * strength;
                    let h = (p - hit).dot(n_hit);
                    let target = max_h * sign;
                    let step = (target - h).clamp(-max_h * 0.15, max_h * 0.15) * w;
                    disp = n_hit * step;
                }
                v.position = (p + disp).to_array();
            }
        }
        SculptBrush::Inflate => {
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let n = Vec3::from(v.normal).normalize_or_zero();
                v.position = (p + n * (strength * w * radius * 0.35 * sign)).to_array();
            }
        }
        SculptBrush::Blob => {
            // Push toward/away from a sphere centered above the hit
            let center = hit + n_hit * (radius * 0.5 * sign);
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let dir = (p - center).normalize_or_zero();
                v.position = (p + dir * (strength * w * radius * 0.3 * sign)).to_array();
            }
        }
        SculptBrush::Crease => {
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff_sharp(d, radius);
                if w <= 0.0 {
                    continue;
                }
                // Draw component
                let draw = n_hit * (strength * w * radius * 0.3 * sign);
                // Pinch toward center
                let to_c = (hit - p).normalize_or_zero();
                let pinch = to_c * (strength * w * radius * 0.2);
                v.position = (p + draw + pinch).to_array();
            }
        }
        SculptBrush::Smooth => {
            let nverts = mesh.vertices.len();
            let mut sum = vec![Vec3::ZERO; nverts];
            let mut count = vec![0u32; nverts];
            for tri in mesh.indices.chunks_exact(3) {
                for i in 0..3 {
                    let a = tri[i] as usize;
                    let b = tri[(i + 1) % 3] as usize;
                    sum[a] += Vec3::from(mesh.vertices[b].position);
                    count[a] += 1;
                }
            }
            let targets: Vec<Vec3> = (0..nverts)
                .map(|i| {
                    if count[i] > 0 {
                        sum[i] / count[i] as f32
                    } else {
                        Vec3::from(mesh.vertices[i].position)
                    }
                })
                .collect();
            for (i, v) in mesh.vertices.iter_mut().enumerate() {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let t = (strength * w).clamp(0.0, 1.0);
                // invert = enhance details
                let target = if settings.invert {
                    p + (p - targets[i])
                } else {
                    targets[i]
                };
                v.position = p.lerp(target, t).to_array();
            }
        }
        SculptBrush::Flatten | SculptBrush::Fill | SculptBrush::Scrape => {
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let h = (p - hit).dot(n_hit);
                let delta_h = avg_h - h;
                let apply = match settings.brush {
                    SculptBrush::Fill => delta_h > 0.0,   // only raise low points
                    SculptBrush::Scrape => delta_h < 0.0, // only lower high points
                    _ => true,                            // Flatten both ways
                };
                if apply {
                    v.position = (p + n_hit * (delta_h * strength * w)).to_array();
                }
            }
        }
        SculptBrush::Pinch => {
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let to_c = (hit - p) * (strength * w * sign);
                v.position = (p + to_c).to_array();
            }
        }
        SculptBrush::Grab | SculptBrush::SnakeHook => {
            let delta = grab_delta.unwrap_or(Vec3::ZERO);
            if delta.length_squared() < 1e-12 {
                return;
            }
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                v.position = (p + delta * w * strength).to_array();
            }
        }
        SculptBrush::Nudge => {
            let delta = grab_delta.unwrap_or(n_hit * 0.0);
            let dir = if delta.length_squared() > 1e-12 {
                delta.normalize_or_zero()
            } else {
                n_hit
            };
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                v.position = (p + dir * (strength * w * radius * 0.25)).to_array();
            }
        }
        SculptBrush::Thumb => {
            // Flatten a bit + slide along stroke
            let delta = grab_delta.unwrap_or(Vec3::ZERO);
            for v in &mut mesh.vertices {
                let p = Vec3::from(v.position);
                let d = (p - hit).length();
                let w = falloff(d, radius);
                if w <= 0.0 {
                    continue;
                }
                let h = (p - hit).dot(n_hit);
                let flatten = n_hit * ((avg_h - h) * 0.5 * w * strength);
                let slide = delta * (w * strength * 0.5);
                v.position = (p + flatten + slide).to_array();
            }
        }
    }
    mesh.compute_normals();
}
