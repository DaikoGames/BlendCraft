//! Progressive CPU path tracer used when the render engine is set to Cycles.
//!
//! A real Monte-Carlo path tracer that accumulates samples over frames, like Blender's
//! progressive viewport:
//!
//! * **Binned-SAH BVH** (iterative, front-to-back traversal) instead of testing every triangle,
//!   so millions of triangles are fine.
//! * **Multi-threaded**: image rows are handed out to all CPU cores.
//! * **Resolution up to 1920 px** on the long side, with an adaptive scheme: a quick low-res
//!   preview while the camera moves, then full resolution; on slow machines the resolution
//!   divisor adapts automatically so the viewport stays responsive.
//! * **Better shading**: smooth (interpolated) normals, GGX microfacet specular with Smith
//!   shadowing and Fresnel, one-sample MIS between the diffuse and specular lobes, soft shadows
//!   from light radius, real spot cones, per-group materials, emissive surfaces.
//! * **Camera matches the viewport**: field of view, fixed (camera object) views and orthographic.
//! * Correct **sRGB output** (the preview texture is `Rgba8UnormSrgb`).
use forge_core::{
    evaluate_surface_shading, LightType, Material, MeshData, ObjectKind, OrbitCamera, Scene,
    ShaderGraph, TextureData,
};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use web_time::Instant;

/// Path depth for viewport and offline (same look when samples converge).
const MAX_BOUNCES: u32 = 5;
/// Match offline so the live view is not a softer lighting model than F12.
const INTERACTIVE_BOUNCES: u32 = 5;
const MAX_TRIANGLES: usize = 4_000_000;
/// Path-trace at most this many pixels on the long side (viewport / progressive).
const MAX_DIM: u32 = 2560;
/// Time budget per displayed frame in the interactive viewport.
const FRAME_BUDGET_MS: u128 = 48;
/// Per-contribution clamp for indirect light (suppresses fireflies).
const INDIRECT_CLAMP: f32 = 10.0;
const LEAF_SIZE: usize = 4;
const BINS: usize = 16;
const STACK: usize = 128;

// ---------------------------------------------------------------------------
// Scene data
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Tri {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    /// Per-vertex UVs for texture sampling (barycentric interpolation).
    uv0: [f32; 2],
    uv1: [f32; 2],
    uv2: [f32; 2],
    mat: u32,
}

#[derive(Clone, Copy)]
struct Mat {
    albedo: Vec3,
    emission: Vec3,
    metallic: f32,
    roughness: f32,
    /// Index into Geometry.textures, or -1 if constant albedo only.
    tex: i32,
    /// Index into Geometry.graphs for live per-hit evaluation, or -1.
    graph: i32,
}

#[derive(Clone, Copy)]
struct Light {
    /// 0 point, 1 sun, 2 spot
    kind: u8,
    pos: Vec3,
    dir: Vec3,
    color: Vec3,
    range: f32,
    /// Point/spot: sphere radius (m). Sun: tangent of the angular radius.
    radius: f32,
    cos_outer: f32,
    cos_inner: f32,
}

#[derive(Clone, Copy)]
struct Node {
    bmin: Vec3,
    bmax: Vec3,
    /// Leaf: first triangle index. Interior: index of the right child (left = self + 1).
    a: u32,
    /// Leaf: triangle count (> 0). Interior: 0.
    count: u32,
}

#[derive(Clone)]
struct TexImage {
    width: u32,
    height: u32,
    /// RGBA8, row-major, length = width * height * 4.
    rgba: Arc<[u8]>,
}

struct GraphEvalData {
    graph: ShaderGraph,
    material: Material,
    /// Shared scene texture snapshot (not cloned per material).
    textures: Arc<Vec<TextureData>>,
}

#[derive(Default)]
struct Geometry {
    tris: Vec<Tri>,
    nodes: Vec<Node>,
    mats: Vec<Mat>,
    lights: Vec<Light>,
    textures: Vec<TexImage>,
    graphs: Vec<GraphEvalData>,
    eps: f32,
}

#[derive(Clone, Copy)]
struct Hit {
    t: f32,
    u: f32,
    v: f32,
    tri: u32,
}

fn box_area(lo: Vec3, hi: Vec3) -> f32 {
    let d = (hi - lo).max(Vec3::ZERO);
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

struct Builder {
    bmin: Vec<Vec3>,
    bmax: Vec<Vec3>,
    cen: Vec<Vec3>,
    idx: Vec<u32>,
    nodes: Vec<Node>,
}

impl Builder {
    fn make_leaf(&mut self, node: usize, start: usize, n: usize) {
        self.nodes[node].a = start as u32;
        self.nodes[node].count = n as u32;
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let n = end - start;
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        let mut clo = Vec3::splat(f32::MAX);
        let mut chi = Vec3::splat(f32::MIN);
        for k in start..end {
            let i = self.idx[k] as usize;
            lo = lo.min(self.bmin[i]);
            hi = hi.max(self.bmax[i]);
            clo = clo.min(self.cen[i]);
            chi = chi.max(self.cen[i]);
        }
        let node = self.nodes.len();
        self.nodes.push(Node {
            bmin: lo,
            bmax: hi,
            a: 0,
            count: 0,
        });
        if n <= 2 {
            self.make_leaf(node, start, n);
            return node;
        }
        let ext = chi - clo;
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let extent = ext[axis];
        let mut mid = (start + end) / 2;
        if extent > 1e-9 {
            let mut counts = [0usize; BINS];
            let mut bmin = [Vec3::splat(f32::MAX); BINS];
            let mut bmax = [Vec3::splat(f32::MIN); BINS];
            let bin_of = |c: f32| -> usize {
                (((c - clo[axis]) / extent * BINS as f32) as usize).min(BINS - 1)
            };
            for k in start..end {
                let i = self.idx[k] as usize;
                let b = bin_of(self.cen[i][axis]);
                counts[b] += 1;
                bmin[b] = bmin[b].min(self.bmin[i]);
                bmax[b] = bmax[b].max(self.bmax[i]);
            }
            // Sweep: cost of splitting after bin `i`.
            let mut left_cost = [0.0_f32; BINS - 1];
            let mut acc_lo = Vec3::splat(f32::MAX);
            let mut acc_hi = Vec3::splat(f32::MIN);
            let mut acc_n = 0usize;
            for i in 0..BINS - 1 {
                acc_lo = acc_lo.min(bmin[i]);
                acc_hi = acc_hi.max(bmax[i]);
                acc_n += counts[i];
                left_cost[i] = acc_n as f32 * box_area(acc_lo, acc_hi);
            }
            let mut best = f32::MAX;
            let mut best_i = 0usize;
            acc_lo = Vec3::splat(f32::MAX);
            acc_hi = Vec3::splat(f32::MIN);
            acc_n = 0;
            for i in (1..BINS).rev() {
                acc_lo = acc_lo.min(bmin[i]);
                acc_hi = acc_hi.max(bmax[i]);
                acc_n += counts[i];
                let cost = left_cost[i - 1] + acc_n as f32 * box_area(acc_lo, acc_hi);
                if cost < best {
                    best = cost;
                    best_i = i - 1;
                }
            }
            let node_area = box_area(lo, hi).max(1e-20);
            let split_cost = 0.125 + best / node_area;
            if n <= LEAF_SIZE && split_cost >= n as f32 {
                self.make_leaf(node, start, n);
                return node;
            }
            // Partition by bin.
            let mut i = start;
            let mut j = end;
            while i < j {
                let t = self.idx[i] as usize;
                if bin_of(self.cen[t][axis]) <= best_i {
                    i += 1;
                } else {
                    j -= 1;
                    self.idx.swap(i, j);
                }
            }
            if i > start && i < end {
                mid = i;
            }
        } else if n <= LEAF_SIZE {
            self.make_leaf(node, start, n);
            return node;
        }
        let _left = self.build(start, mid);
        let right = self.build(mid, end);
        self.nodes[node].a = right as u32;
        node
    }
}

/// Build a BVH; returns the nodes and the triangles reordered to match the leaves.
fn build_bvh(tris: Vec<Tri>) -> (Vec<Node>, Vec<Tri>) {
    if tris.is_empty() {
        return (Vec::new(), tris);
    }
    let n = tris.len();
    let mut bmin = Vec::with_capacity(n);
    let mut bmax = Vec::with_capacity(n);
    let mut cen = Vec::with_capacity(n);
    for t in &tris {
        let p1 = t.v0 + t.e1;
        let p2 = t.v0 + t.e2;
        let lo = t.v0.min(p1).min(p2);
        let hi = t.v0.max(p1).max(p2);
        bmin.push(lo);
        bmax.push(hi);
        cen.push((lo + hi) * 0.5);
    }
    let mut b = Builder {
        bmin,
        bmax,
        cen,
        idx: (0..n as u32).collect(),
        nodes: Vec::with_capacity(n / 2 + 1),
    };
    b.build(0, n);
    let ordered: Vec<Tri> = b.idx.iter().map(|&i| tris[i as usize]).collect();
    (b.nodes, ordered)
}

#[inline]
fn node_entry(node: &Node, o: Vec3, inv: Vec3, tmax: f32) -> f32 {
    let t1 = (node.bmin - o) * inv;
    let t2 = (node.bmax - o) * inv;
    let tn = t1.min(t2).max_element().max(0.0);
    let tf = t1.max(t2).min_element().min(tmax);
    if tn <= tf {
        tn
    } else {
        f32::INFINITY
    }
}

#[inline]
fn tri_hit(t: &Tri, o: Vec3, d: Vec3, tmax: f32) -> Option<(f32, f32, f32)> {
    let p = d.cross(t.e2);
    let det = t.e1.dot(p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - t.v0;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(t.e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let tt = t.e2.dot(q) * inv;
    if tt > 1e-7 && tt < tmax {
        Some((tt, u, v))
    } else {
        None
    }
}

impl Geometry {
    fn closest(&self, o: Vec3, d: Vec3, tmax: f32) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best = tmax;
        let mut hit = None;
        if node_entry(&self.nodes[0], o, inv, best) == f32::INFINITY {
            return None;
        }
        let mut stack = [0u32; STACK];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let node = &self.nodes[ni];
            if node.count > 0 {
                let first = node.a as usize;
                for k in first..first + node.count as usize {
                    if let Some((t, u, v)) = tri_hit(&self.tris[k], o, d, best) {
                        best = t;
                        hit = Some(Hit {
                            t,
                            u,
                            v,
                            tri: k as u32,
                        });
                    }
                }
            } else {
                let l = ni + 1;
                let r = node.a as usize;
                let tl = node_entry(&self.nodes[l], o, inv, best);
                let tr = node_entry(&self.nodes[r], o, inv, best);
                if sp + 2 >= STACK {
                    continue;
                }
                if tl < f32::INFINITY && tr < f32::INFINITY {
                    if tl <= tr {
                        stack[sp] = r as u32;
                        stack[sp + 1] = l as u32;
                    } else {
                        stack[sp] = l as u32;
                        stack[sp + 1] = r as u32;
                    }
                    sp += 2;
                } else if tl < f32::INFINITY {
                    stack[sp] = l as u32;
                    sp += 1;
                } else if tr < f32::INFINITY {
                    stack[sp] = r as u32;
                    sp += 1;
                }
            }
        }
        hit
    }

    fn occluded(&self, o: Vec3, d: Vec3, tmax: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        if node_entry(&self.nodes[0], o, inv, tmax) == f32::INFINITY {
            return false;
        }
        let mut stack = [0u32; STACK];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let node = &self.nodes[ni];
            if node.count > 0 {
                let first = node.a as usize;
                for k in first..first + node.count as usize {
                    if tri_hit(&self.tris[k], o, d, tmax).is_some() {
                        return true;
                    }
                }
            } else {
                if sp + 2 >= STACK {
                    continue;
                }
                let l = ni + 1;
                let r = node.a as usize;
                if node_entry(&self.nodes[l], o, inv, tmax) < f32::INFINITY {
                    stack[sp] = l as u32;
                    sp += 1;
                }
                if node_entry(&self.nodes[r], o, inv, tmax) < f32::INFINITY {
                    stack[sp] = r as u32;
                    sp += 1;
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Random numbers and sampling helpers
// ---------------------------------------------------------------------------

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// PCG output function; returns a float in [0, 1).
fn rand01(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((*s >> ((*s >> 28) + 4)) ^ *s).wrapping_mul(277_803_737);
    let r = (w >> 22) ^ w;
    (r >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Orthonormal basis around `n` (Duff et al.).
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

fn cosine_hemisphere(n: Vec3, r1: f32, r2: f32) -> Vec3 {
    let phi = 2.0 * PI * r1;
    let r = r2.sqrt();
    let (t, b) = basis(n);
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - r2).sqrt()).normalize_or_zero()
}

fn unit_sphere(r1: f32, r2: f32) -> Vec3 {
    let z = 1.0 - 2.0 * r1;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * r2;
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

// ---------------------------------------------------------------------------
// BSDF: Lambert diffuse + GGX specular, one-sample MIS between the lobes
// ---------------------------------------------------------------------------

struct Bsdf {
    albedo: Vec3,
    f0: Vec3,
    metallic: f32,
    a2: f32,
    /// Probability of sampling the specular lobe.
    ps: f32,
}

#[inline]
fn ggx_d(nh: f32, a2: f32) -> f32 {
    let d = nh * nh * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1e-12)
}

#[inline]
fn smith_g1(nx: f32, a2: f32) -> f32 {
    2.0 * nx / (nx + (a2 + (1.0 - a2) * nx * nx).sqrt()).max(1e-12)
}

impl Bsdf {
    fn new(m: &Mat) -> Self {
        let a = (m.roughness * m.roughness).max(0.0025);
        let metallic = m.metallic.clamp(0.0, 1.0);
        Self {
            albedo: m.albedo,
            f0: Vec3::splat(0.04).lerp(m.albedo, metallic),
            metallic,
            a2: a * a,
            ps: (0.25 + 0.75 * metallic).clamp(0.1, 0.9),
        }
    }

    /// BSDF value (without the cosine term) and the mixture pdf for direction `l`.
    fn eval_pdf(&self, n: Vec3, v: Vec3, l: Vec3) -> (Vec3, f32) {
        let nl = n.dot(l);
        let nv = n.dot(v);
        if nl <= 0.0 || nv <= 0.0 {
            return (Vec3::ZERO, 0.0);
        }
        let h = (v + l).normalize_or_zero();
        let nh = n.dot(h).max(0.0);
        let vh = v.dot(h).max(0.0);
        let d = ggx_d(nh, self.a2);
        let g = smith_g1(nl, self.a2) * smith_g1(nv, self.a2);
        let fres = self.f0 + (Vec3::ONE - self.f0) * (1.0 - vh).powi(5);
        let spec = fres * (d * g / (4.0 * nl * nv).max(1e-12));
        let diff = self.albedo * ((1.0 - self.metallic) / PI);
        let pdf_spec = d * nh / (4.0 * vh).max(1e-12);
        let pdf_diff = nl / PI;
        (diff + spec, (1.0 - self.ps) * pdf_diff + self.ps * pdf_spec)
    }

    /// Sample a direction; returns `(direction, f * cos / pdf)`.
    fn sample(&self, n: Vec3, v: Vec3, rng: &mut u32) -> Option<(Vec3, Vec3)> {
        let r1 = rand01(rng);
        let r2 = rand01(rng);
        let r3 = rand01(rng);
        let l = if r3 < self.ps {
            let phi = 2.0 * PI * r1;
            let cos_t = ((1.0 - r2) / (1.0 + (self.a2 - 1.0) * r2)).max(0.0).sqrt();
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let (t, b) = basis(n);
            let h = (t * (sin_t * phi.cos()) + b * (sin_t * phi.sin()) + n * cos_t)
                .normalize_or_zero();
            let l = h * (2.0 * v.dot(h)) - v;
            l.normalize_or_zero()
        } else {
            cosine_hemisphere(n, r1, r2)
        };
        if l == Vec3::ZERO {
            return None;
        }
        let (f, pdf) = self.eval_pdf(n, v, l);
        if pdf <= 1e-8 {
            return None;
        }
        Some((l, f * (n.dot(l) / pdf)))
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Cam {
    cam_to_world: Mat4,
    tan_half: f32,
    aspect: f32,
    ortho_h: Option<f32>,
}

impl Cam {
    fn new(camera: &OrbitCamera, aspect: f32) -> Self {
        let cam_to_world = camera.view().inverse();
        let (fov, ortho_h) = match &camera.fixed {
            Some(f) => (f.fov_y, f.ortho_height),
            None => (
                camera.fov_y,
                if camera.ortho {
                    Some(camera.distance * 2.0 * (camera.fov_y * 0.5).tan())
                } else {
                    None
                },
            ),
        };
        Self {
            cam_to_world,
            tan_half: (fov * 0.5).tan().max(1e-4),
            aspect,
            ortho_h,
        }
    }

    /// Ray for normalized device coordinates in [-1, 1] (y up).
    fn ray(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        match self.ortho_h {
            Some(h) => {
                let o = self
                    .cam_to_world
                    .transform_point3(Vec3::new(x * self.aspect * h * 0.5, y * h * 0.5, 0.0));
                let d = self
                    .cam_to_world
                    .transform_vector3(Vec3::NEG_Z)
                    .normalize_or_zero();
                (o, d)
            }
            None => {
                let local = Vec3::new(x * self.aspect * self.tan_half, y * self.tan_half, -1.0);
                let o = self.cam_to_world.transform_point3(Vec3::ZERO);
                let d = self.cam_to_world.transform_vector3(local).normalize_or_zero();
                (o, d)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

fn sky(dir: Vec3) -> Vec3 {
    Vec3::new(0.4, 0.6, 0.9) * (dir.y * 0.5 + 0.5).clamp(0.05, 1.0) + Vec3::splat(0.05)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 - e0).abs() < 1e-6 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Next-event estimation towards one randomly chosen light.
fn direct_light(geo: &Geometry, p: Vec3, ng: Vec3, ns: Vec3, v: Vec3, bsdf: &Bsdf, rng: &mut u32) -> Vec3 {
    let count = geo.lights.len();
    if count == 0 {
        return Vec3::ZERO;
    }
    let li = ((rand01(rng) * count as f32) as usize).min(count - 1);
    let l = &geo.lights[li];
    let (wi, dist, radiance) = if l.kind == 1 {
        // Sun: jitter inside the angular disk for soft shadows.
        let axis = (-l.dir).normalize_or_zero();
        let (t, b) = basis(axis);
        let a = 2.0 * PI * rand01(rng);
        let r = rand01(rng).sqrt() * l.radius;
        let wi = (axis + t * (r * a.cos()) + b * (r * a.sin())).normalize_or_zero();
        (wi, f32::INFINITY, l.color)
    } else {
        let target = l.pos + unit_sphere(rand01(rng), rand01(rng)) * l.radius;
        let to = target - p;
        let dist = to.length().max(1e-4);
        let wi = to / dist;
        let mut a = 1.0 / (dist * dist);
        if l.range > 0.0 {
            let x = (1.0 - (dist / l.range).clamp(0.0, 1.0)).max(0.0);
            a *= x * x;
        }
        if l.kind == 2 {
            let c = (-wi).dot(l.dir);
            a *= smoothstep(l.cos_outer, l.cos_inner, c);
        }
        (wi, dist, l.color * a)
    };
    let nl = ns.dot(wi);
    if nl <= 0.0 || ng.dot(wi) <= 0.0 || radiance == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let tmax = if dist.is_finite() { dist - geo.eps * 2.0 } else { 1.0e30 };
    if geo.occluded(p + ng * geo.eps, wi, tmax) {
        return Vec3::ZERO;
    }
    let (f, _) = bsdf.eval_pdf(ns, v, wi);
    f * radiance * (nl * count as f32)
}

fn trace(geo: &Geometry, mut o: Vec3, mut d: Vec3, rng: &mut u32, max_bounces: u32) -> Vec3 {
    let mut thr = Vec3::ONE;
    let mut col = Vec3::ZERO;
    let max_bounces = max_bounces.clamp(1, MAX_BOUNCES);
    for bounce in 0..max_bounces {
        let clamp = |c: Vec3| -> Vec3 {
            if bounce > 0 {
                c.min(Vec3::splat(INDIRECT_CLAMP))
            } else {
                c
            }
        };
        let Some(hit) = geo.closest(o, d, f32::INFINITY) else {
            col += clamp(thr * sky(d));
            break;
        };
        let tri = &geo.tris[hit.tri as usize];
        let base_mat = &geo.mats[tri.mat as usize];
        let p = o + d * hit.t;
        let mut ng = tri.e1.cross(tri.e2).normalize_or_zero();
        let w = 1.0 - hit.u - hit.v;
        let mut ns = (tri.n0 * w + tri.n1 * hit.u + tri.n2 * hit.v).normalize_or_zero();
        let uv_u = tri.uv0[0] * w + tri.uv1[0] * hit.u + tri.uv2[0] * hit.v;
        let uv_v = tri.uv0[1] * w + tri.uv1[1] * hit.u + tri.uv2[1] * hit.v;
        let mat = shade_mat(geo, base_mat, uv_u, uv_v, p);
        if ng == Vec3::ZERO {
            ng = -d;
        }
        if ns == Vec3::ZERO {
            ns = ng;
        }
        if ng.dot(d) > 0.0 {
            ng = -ng;
            ns = -ns;
        }
        if ns.dot(ng) < 0.0 {
            ns = ng;
        }
        let v = -d;

        col += clamp(thr * mat.emission);
        let bsdf = Bsdf::new(&mat);
        col += clamp(thr * direct_light(geo, p, ng, ns, v, &bsdf, rng));

        let Some((l, weight)) = bsdf.sample(ns, v, rng) else {
            break;
        };
        thr *= weight;
        // Russian roulette earlier → fewer long paths, more samples per ms.
        if bounce >= 1 {
            let survive = thr.max_element().clamp(0.05, 0.9);
            if rand01(rng) > survive {
                break;
            }
            thr /= survive;
        }
        let side = if l.dot(ng) >= 0.0 { 1.0 } else { -1.0 };
        o = p + ng * (geo.eps * side);
        d = l;
    }
    col
}

fn render_pass(
    geo: &Geometry,
    cam: &Cam,
    w: usize,
    h: usize,
    accum: &mut [[f32; 3]],
    sample: u32,
    threads: usize,
    max_bounces: u32,
) {
    // Atomic row counter avoids Mutex contention between workers.
    let next_row = AtomicUsize::new(0);
    // SAFETY: each row index is claimed at most once via the atomic counter.
    let accum_ptr = accum.as_mut_ptr() as usize;
    let work = || loop {
        let y = next_row.fetch_add(1, Ordering::Relaxed);
        if y >= h {
            break;
        }
        let row = unsafe {
            let base = accum_ptr as *mut [f32; 3];
            std::slice::from_raw_parts_mut(base.add(y * w), w)
        };
        for (x, px) in row.iter_mut().enumerate() {
            let seed = hash_u32(
                (x as u32).wrapping_mul(1973)
                    ^ (y as u32).wrapping_mul(9277)
                    ^ sample.wrapping_mul(26_699)
                    ^ 0xA511_E9B3,
            );
            let mut rng = hash_u32(seed ^ 0x68E3_1DA4);
            let jx = rand01(&mut rng);
            let jy = rand01(&mut rng);
            let nx = (x as f32 + jx) / w as f32 * 2.0 - 1.0;
            let ny = 1.0 - (y as f32 + jy) / h as f32 * 2.0;
            let (o, d) = cam.ray(nx, ny);
            let c = trace(geo, o, d, &mut rng, max_bounces);
            if c.is_finite() {
                px[0] += c.x;
                px[1] += c.y;
                px[2] += c.z;
            }
        }
    };
    if threads > 1 {
        std::thread::scope(|sc| {
            for _ in 1..threads {
                sc.spawn(&work);
            }
            work();
        });
    } else {
        work();
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct CyclesState {
    geo: Geometry,
    accum: Vec<[f32; 3]>,
    width: u32,
    height: u32,
    samples: u32,
    /// Fingerprint of camera + scene so we know when to reset.
    fingerprint: u64,
    /// Latest tonemapped sRGB RGBA8 preview (width*height*4).
    preview: Vec<u8>,
    exposure: f32,
    threads: usize,
    /// Interactive viewport (low-res preview while moving, time budget) vs. final render.
    interactive: bool,
    preview_mode: bool,
    /// Resolution divisor that adapts to the machine speed.
    res_div: u32,
    measured: bool,
    srgb_lut: Vec<u8>,
    /// Fingerprint of materials + textures + mesh topology used in the last rebuild.
    /// When this differs from the live scene, Cycles must rebuild so shading updates appear.
    scene_fp: u64,
    /// Last user resolution controls — change forces an immediate buffer resize.
    last_max_dim: u32,
    last_pixel_size: u32,
}

fn build_srgb_lut() -> Vec<u8> {
    (0..1024)
        .map(|i| {
            let x = i as f32 / 1023.0;
            let y = if x <= 0.003_130_8 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (y * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        })
        .collect()
}

impl Default for CyclesState {
    fn default() -> Self {
        Self {
            geo: Geometry::default(),
            accum: Vec::new(),
            width: 0,
            height: 0,
            samples: 0,
            fingerprint: 0,
            preview: Vec::new(),
            exposure: 1.0,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .clamp(1, 32),
            interactive: true,
            preview_mode: false,
            res_div: 1,
            measured: false,
            srgb_lut: build_srgb_lut(),
            scene_fp: 0,
            last_max_dim: 0,
            last_pixel_size: 0,
        }
    }
}

/// Hash of materials, textures and mesh ids — used to know when Cycles must rebuild.
pub fn scene_shading_fingerprint(scene: &Scene) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    mix(scene.materials.len() as u64);
    mix(scene.textures.len() as u64);
    mix(scene.objects.len() as u64);
    for (mi, m) in scene.materials.iter().enumerate() {
        mix(mi as u64);
        for c in m.base_color {
            mix(c.to_bits() as u64);
        }
        mix(m.metallic.to_bits() as u64);
        mix(m.roughness.to_bits() as u64);
        for c in m.emissive {
            mix(c.to_bits() as u64);
        }
        if let Some(i) = m.base_color_tex {
            mix(i as u64 + 1);
            if let Some(t) = scene.textures.get(i) {
                mix(t.width as u64);
                mix(t.height as u64);
                // Sample a few texels so content changes invalidate the fingerprint.
                let n = t.rgba8.len();
                if n >= 4 {
                    mix(t.rgba8[0] as u64);
                    mix(t.rgba8[n / 2] as u64);
                    mix(t.rgba8[n - 1] as u64);
                }
            }
        }
        if let Some(g) = m.shader_graph.as_ref() {
            mix(graph_fingerprint(g));
        } else {
            mix(0);
        }
    }
    for (i, o) in scene.objects.iter().enumerate() {
        if !o.kind.is_mesh() {
            continue;
        }
        mix(i as u64);
        mix(o.id);
        mix(Arc::as_ptr(&o.mesh) as u64);
        mix(o.mesh.vertices.len() as u64);
        mix(o.mesh.indices.len() as u64);
        if let Some(mi) = o.material {
            mix(mi as u64 + 1);
        }
    }
    h
}

impl CyclesState {
    pub fn samples(&self) -> u32 {
        self.samples
    }

    pub fn reset_accumulation(&mut self) {
        self.fingerprint = 0;
        self.samples = 0;
        self.accum.clear();
        self.preview.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Rebuild geometry when materials / graphs / textures / meshes changed since last time.
    /// Returns true if a rebuild happened.
    pub fn ensure_scene(&mut self, scene: &Scene) -> bool {
        let fp = scene_shading_fingerprint(scene);
        if fp == self.scene_fp && !self.geo.tris.is_empty() {
            return false;
        }
        self.rebuild_scene(scene);
        self.scene_fp = fp;
        true
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn preview_rgba(&self) -> &[u8] {
        &self.preview
    }

    /// Force the next `accumulate` to rebuild the buffer (e.g. user lowered Max Size).
    pub fn invalidate_resolution(&mut self) {
        self.last_max_dim = 0;
        self.last_pixel_size = 0;
        self.width = 0;
        self.height = 0;
        self.samples = 0;
        self.accum.clear();
        self.preview.clear();
        self.fingerprint = 0;
    }

    /// `true` = interactive viewport behaviour (default); `false` = final render at full
    /// resolution without the low-res preview or the per-frame time budget.
    pub fn set_interactive(&mut self, interactive: bool) {
        self.interactive = interactive;
        if !interactive {
            self.res_div = 1;
            self.preview_mode = false;
        }
    }

    /// Rebuild geometry and BVH from the scene (call from `set_scene`).
    pub fn rebuild_scene(&mut self, scene: &Scene) {
        let mut geo = Geometry::default();
        let mut tris: Vec<Tri> = Vec::new();
        geo.mats.push(convert_material(&Material::default(), -1, -1));

        // One shared texture snapshot for all graph materials (avoids N clones).
        let shared_textures: Arc<Vec<TextureData>> = Arc::new(scene.textures.as_ref().clone());

        // Keep texture indices identical to Scene::textures so material.base_color_tex
        // (and other slots) map 1:1. Previously only used albedos were compacted into
        // a new list, which was easy to desync from material indices on import.
        for t in scene.textures.iter() {
            geo.textures.push(tex_from_data(t));
        }

        // Resolve every material once. Prefer live shader-graph evaluation so procedural
        // shading works even on meshes with broken/missing UVs.
        let mut mat_map: HashMap<usize, u32> = HashMap::new();
        for (mi, mat) in scene.materials.iter().enumerate() {
            let tex_idx = mat
                .base_color_tex
                .filter(|&ti| {
                    scene.textures.get(ti).map_or(false, |t| {
                        t.width > 0 && t.height > 0 && !t.rgba8.is_empty()
                    })
                })
                .map(|ti| ti as i32)
                .unwrap_or(-1);

            let graph_idx = if let Some(graph) = mat.shader_graph.as_ref() {
                // Don't let an empty Principled graph hide an imported albedo map.
                if graph.is_trivial_untextured() && mat.base_color_tex.is_some() {
                    -1
                } else {
                    let idx = geo.graphs.len() as i32;
                    geo.graphs.push(GraphEvalData {
                        graph: graph.clone(),
                        material: mat.clone(),
                        textures: Arc::clone(&shared_textures),
                    });
                    idx
                }
            } else {
                -1
            };

            let mut m = convert_material(mat, tex_idx, graph_idx);
            // Pre-average a few graph samples so secondary bounces have a cheap albedo.
            if graph_idx >= 0 {
                if let Some(g) = geo.graphs.last() {
                    let mut acc = Vec3::ZERO;
                    let probes = [
                        [0.2, 0.3, 0.1],
                        [0.7, 0.4, 0.6],
                        [0.4, 0.8, 0.3],
                        [0.5, 0.5, 0.5],
                    ];
                    for p in probes {
                        let (c, _, _, _) =
                            evaluate_surface_shading(&g.graph, &g.material, g.textures.as_ref(), [p[0], p[1]], p);
                        acc += Vec3::new(c[0], c[1], c[2]).max(Vec3::ZERO);
                    }
                    m.albedo = (acc / probes.len() as f32).max(Vec3::splat(0.01));
                }
            }
            geo.mats.push(m);
            mat_map.insert(mi, (geo.mats.len() - 1) as u32);
        }

        let material_for = |index: Option<usize>| -> u32 {
            index
                .and_then(|mi| mat_map.get(&mi).copied())
                .unwrap_or(0)
        };

        let worlds = scene.world_matrices();
        for (i, o) in scene.objects.iter().enumerate() {
            if !scene.is_visible(i) {
                continue;
            }
            let m = worlds.get(i).copied().unwrap_or(Mat4::IDENTITY);
            if let ObjectKind::Light(l) = &o.kind {
                let pos = m.w_axis.truncate();
                let dir = m.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                let color = Vec3::new(l.color[0], l.color[1], l.color[2]) * l.intensity.max(0.0);
                let outer = (l.spot_angle_deg.clamp(1.0, 179.0) * 0.5).to_radians();
                let inner = outer * (1.0 - l.spot_blend.clamp(0.0, 1.0));
                let (kind, radius) = match l.kind {
                    LightType::Sun => (1, 0.02),
                    LightType::Spot => (2, 0.05),
                    _ => (0, 0.05),
                };
                geo.lights.push(Light {
                    kind,
                    pos,
                    dir,
                    color,
                    range: l.range.max(0.0),
                    radius,
                    cos_outer: outer.cos(),
                    cos_inner: inner.cos(),
                });
                continue;
            }
            if !o.kind.is_mesh() || o.mesh.indices.is_empty() {
                continue;
            }
            let default_mat = material_for(o.material);
            let mut group_mats: Vec<(usize, usize, u32)> = Vec::new();
            for g in &o.mesh.groups {
                let id = match g.material {
                    Some(mi) => material_for(Some(mi)),
                    None => default_mat,
                };
                group_mats.push((
                    g.first_index as usize,
                    (g.first_index + g.index_count) as usize,
                    id,
                ));
            }
            push_mesh_tris(&mut tris, &o.mesh, m, default_mat, &group_mats);
            if tris.len() >= MAX_TRIANGLES {
                break;
            }
        }
        // Ray offset scaled to the scene size.
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for t in &tris {
            lo = lo.min(t.v0);
            hi = hi.max(t.v0 + t.e1).max(t.v0 + t.e2);
        }
        let diag = if tris.is_empty() { 1.0 } else { (hi - lo).length() };
        geo.eps = (diag * 2.0e-5).clamp(1.0e-4, 5.0e-3);
        let (nodes, ordered) = build_bvh(tris);
        geo.nodes = nodes;
        geo.tris = ordered;
        self.geo = geo;
        self.scene_fp = scene_shading_fingerprint(scene);

        // Changing geometry always invalidates the accumulation.
        self.fingerprint = self.fingerprint.wrapping_add(0x9E37_79B9);
        self.samples = 0;
        self.accum.clear();
        self.preview.clear();
    }

    /// Add samples for the current camera / viewport size. Returns true when the
    /// preview buffer was updated.
    ///
    /// * `max_dim` — long-edge pixel cap from the user (720 / 1080 / 1920 / 2560 / 4096…).
    /// * `pixel_size` — 1 = full, 2 = half, 4 = quarter (Blender-style). Forced, not adaptive.
    pub fn accumulate(
        &mut self,
        camera: &OrbitCamera,
        view_w: u32,
        view_h: u32,
        exposure: f32,
        target_samples: u32,
        max_dim: u32,
        pixel_size: u32,
    ) -> bool {
        if view_w < 2 || view_h < 2 {
            return false;
        }
        self.exposure = exposure.max(0.01);
        let target = target_samples.max(1);
        // Resolution model (Blender-like, predictable):
        //   1) Start from the *viewport panel* size
        //   2) Divide by Pixel Size (1× full, 2× half, 4× quarter, 8× eighth)
        //   3) Cap the long edge at Max Size
        // Never supersample above the panel — that made "lower Max Size" look unchanged.
        // Allow full viewport size offline (up to 8K); only clamp extremes.
        let max_dim = max_dim.max(64).min(7680);
        let div = match pixel_size {
            0 | 1 => 1,
            2 => 2,
            3 | 4 => 4,
            _ => 8,
        };
        let base_w = (view_w.max(1) + div - 1) / div;
        let base_h = (view_h.max(1) + div - 1) / div;
        let (tw, th) = fit_size(base_w, base_h, max_dim);
        // Settings changes must rebuild the buffer this frame.
        let settings_changed =
            self.last_max_dim != max_dim || self.last_pixel_size != div || self.last_max_dim == 0;
        self.last_max_dim = max_dim;
        self.last_pixel_size = div;
        let fp = camera_fingerprint(camera, tw, th)
            ^ ((max_dim as u64) << 17)
            ^ ((div as u64) << 9)
            ^ ((tw as u64) << 3)
            ^ (th as u64);
        if fp != self.fingerprint || settings_changed || self.width != tw || self.height != th {
            self.fingerprint = fp;
            self.preview_mode = false;
            self.measured = false;
            self.samples = 0;
            self.accum.clear();
            self.res_div = div;
            self.width = tw;
            self.height = th;
            self.accum = vec![[0.0; 3]; (tw * th) as usize];
            self.preview = vec![0u8; (tw * th * 4) as usize];
        } else if self.accum.is_empty() {
            self.width = tw;
            self.height = th;
            self.samples = 0;
            self.accum = vec![[0.0; 3]; (tw * th) as usize];
            self.preview = vec![0u8; (tw * th * 4) as usize];
        }
        if self.samples >= target {
            return false;
        }

        let w = tw as usize;
        let h = th as usize;
        let cam = Cam::new(camera, tw as f32 / th as f32);
        let max_bounces = if self.interactive {
            INTERACTIVE_BOUNCES
        } else {
            MAX_BOUNCES
        };
        let start = Instant::now();
        loop {
            render_pass(
                &self.geo,
                &cam,
                w,
                h,
                &mut self.accum,
                self.samples,
                self.threads,
                max_bounces,
            );
            self.samples += 1;
            if !self.interactive {
                break;
            }
            if self.samples >= target || start.elapsed().as_millis() >= FRAME_BUDGET_MS {
                break;
            }
        }
        self.tonemap();
        true
    }

    fn tonemap(&mut self) {
        let scale = self.exposure / self.samples.max(1) as f32;
        let lut = &self.srgb_lut;
        let enc = |x: f32| -> u8 { lut[((x.clamp(0.0, 1.0) * 1023.0) + 0.5) as usize] };
        for (i, c) in self.accum.iter().enumerate() {
            let rgb = aces_tonemap(Vec3::new(c[0], c[1], c[2]) * scale);
            let o = i * 4;
            self.preview[o] = enc(rgb.x);
            self.preview[o + 1] = enc(rgb.y);
            self.preview[o + 2] = enc(rgb.z);
            self.preview[o + 3] = 255;
        }
    }
}

fn convert_material(m: &Material, tex_index: i32, graph_index: i32) -> Mat {
    // Always keep the base-color *factor* as albedo. Textured materials multiply
    // sample_tex(...) * albedo in shade_mat (glTF convention). Previously albedo
    // was forced to ONE whenever a texture index was set, so a missing/empty
    // texture or bad UVs produced a washed-out white surface in Cycles.
    Mat {
        albedo: Vec3::new(m.base_color[0], m.base_color[1], m.base_color[2]).max(Vec3::ZERO),
        emission: Vec3::new(m.emissive[0], m.emissive[1], m.emissive[2]),
        metallic: m.metallic,
        roughness: m.roughness.max(0.02),
        tex: tex_index,
        graph: graph_index,
    }
}

/// Cheap fingerprint of a shader graph so we can cache auto-bakes across rebuilds.
fn graph_fingerprint(graph: &ShaderGraph) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    mix(graph.nodes.len() as u64);
    mix(graph.links.len() as u64);
    for n in &graph.nodes {
        mix(n.id);
        mix(n.kind as u8 as u64);
        mix(n.mode as u64);
        mix(n.value.to_bits() as u64);
        for c in n.color {
            mix(c.to_bits() as u64);
        }
        for p in n.props {
            mix(p.to_bits() as u64);
        }
        for stop in &n.ramp_stops {
            mix(stop.position.to_bits() as u64);
            for c in stop.color {
                mix(c.to_bits() as u64);
            }
        }
        if let Some(i) = n.image_index {
            mix(i as u64 + 1);
        }
    }
    for l in &graph.links {
        mix(l.from);
        mix(l.from_socket as u64);
        mix(l.to);
        mix(l.to_socket as u64);
    }
    h
}

fn tex_from_data(t: &TextureData) -> TexImage {
    TexImage {
        width: t.width,
        height: t.height,
        rgba: Arc::from(t.rgba8.as_slice()),
    }
}

/// Bilinear sample of an RGBA8 texture; returns linear-ish RGB (input is treated as sRGB-ish).
fn sample_tex(tex: &TexImage, u: f32, v: f32) -> Vec3 {
    if tex.width == 0 || tex.height == 0 || tex.rgba.is_empty() {
        return Vec3::ONE;
    }
    let w = tex.width as f32;
    let h = tex.height as f32;
    // Repeat wrap
    let u = u - u.floor();
    let v = v - v.floor();
    let x = u * (w - 1.0).max(0.0);
    let y = v * (h - 1.0).max(0.0); // glTF UV origin = top-left, matches row-major RGBA
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(tex.width.saturating_sub(1));
    let y1 = (y0 + 1).min(tex.height.saturating_sub(1));
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let fetch = |px: u32, py: u32| -> Vec3 {
        let i = ((py * tex.width + px) * 4) as usize;
        if i + 2 >= tex.rgba.len() {
            return Vec3::ONE;
        }
        Vec3::new(
            tex.rgba[i] as f32 / 255.0,
            tex.rgba[i + 1] as f32 / 255.0,
            tex.rgba[i + 2] as f32 / 255.0,
        )
    };
    let c00 = fetch(x0, y0);
    let c10 = fetch(x1, y0);
    let c01 = fetch(x0, y1);
    let c11 = fetch(x1, y1);
    let c0 = c00.lerp(c10, fx);
    let c1 = c01.lerp(c11, fx);
    c0.lerp(c1, fy)
}

fn shade_mat(geo: &Geometry, mat: &Mat, u: f32, v: f32, position: Vec3) -> Mat {
    // Evaluate shader graphs on every hit so multi-bounce light matches surface shading.
    if mat.graph >= 0 {
        if let Some(g) = geo.graphs.get(mat.graph as usize) {
            let (base, metallic, roughness, emissive) = evaluate_surface_shading(
                &g.graph,
                &g.material,
                g.textures.as_ref(),
                [u, v],
                position.to_array(),
            );
            return Mat {
                albedo: Vec3::new(base[0], base[1], base[2]).max(Vec3::ZERO),
                emission: Vec3::new(emissive[0], emissive[1], emissive[2]).max(Vec3::ZERO),
                metallic: metallic.clamp(0.0, 1.0),
                roughness: roughness.clamp(0.02, 1.0),
                tex: -1,
                graph: -1,
            };
        }
    }
    if mat.tex >= 0 {
        if let Some(tex) = geo.textures.get(mat.tex as usize) {
            if tex.width > 0 && tex.height > 0 && !tex.rgba.is_empty() {
                let albedo = sample_tex(tex, u, v) * mat.albedo;
                return Mat {
                    albedo,
                    emission: mat.emission,
                    metallic: mat.metallic,
                    roughness: mat.roughness,
                    tex: -1,
                    graph: -1,
                };
            }
        }
        // Texture index set but data missing — keep factor color, not white.
        return *mat;
    }
    *mat
}

fn push_mesh_tris(
    out: &mut Vec<Tri>,
    mesh: &Arc<MeshData>,
    m: Mat4,
    default_mat: u32,
    groups: &[(usize, usize, u32)],
) {
    let nmat = glam::Mat3::from_mat4(m).inverse().transpose();
    let verts = &mesh.vertices;
    for (ti, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if out.len() >= MAX_TRIANGLES {
            return;
        }
        if tri.iter().any(|&k| k as usize >= verts.len()) {
            continue;
        }
        let first = ti * 3;
        let mat = groups
            .iter()
            .find(|g| first >= g.0 && first < g.1)
            .map(|g| g.2)
            .unwrap_or(default_mat);
        let p0 = m.transform_point3(Vec3::from(verts[tri[0] as usize].position));
        let p1 = m.transform_point3(Vec3::from(verts[tri[1] as usize].position));
        let p2 = m.transform_point3(Vec3::from(verts[tri[2] as usize].position));
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let flat = e1.cross(e2).normalize_or_zero();
        if flat == Vec3::ZERO {
            continue;
        }
        let vn = |k: u32| -> Vec3 {
            let n = (nmat * Vec3::from(verts[k as usize].normal)).normalize_or_zero();
            // Keep interpolated normals on the same side as the geometric normal.
            if n == Vec3::ZERO || n.dot(flat) < 0.0 {
                flat
            } else {
                n
            }
        };
        let uv = |k: u32| verts[k as usize].uv;
        out.push(Tri {
            v0: p0,
            e1,
            e2,
            n0: vn(tri[0]),
            n1: vn(tri[1]),
            n2: vn(tri[2]),
            uv0: uv(tri[0]),
            uv1: uv(tri[1]),
            uv2: uv(tri[2]),
            mat,
        });
    }
}

fn fit_size(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= max_dim {
        return (w.max(1), h.max(1));
    }
    let s = max_dim as f32 / long as f32;
    (
        ((w as f32 * s).round() as u32).max(1),
        ((h as f32 * s).round() as u32).max(1),
    )
}

/// Buffer whose long edge is exactly `long`, aspect matched to the panel.
fn fit_to_long_edge(aspect_w: f32, aspect_h: f32, long: u32) -> (u32, u32) {
    let long = long.max(1);
    let aw = aspect_w.max(1.0);
    let ah = aspect_h.max(1.0);
    if aw >= ah {
        let w = long;
        let h = ((long as f32) * ah / aw).round().max(1.0) as u32;
        (w, h)
    } else {
        let h = long;
        let w = ((long as f32) * aw / ah).round().max(1.0) as u32;
        (w, h)
    }
}

fn camera_fingerprint(cam: &OrbitCamera, w: u32, h: u32) -> u64 {
    let mut hsh: u64 = 0xcbf2_9ce4_8422_2325;
    // Quantize so tiny float jitter does not reset accumulation every frame.
    let mut mix = |x: f32| {
        let q = (x * 1024.0).round() as i32;
        hsh ^= q as u64;
        hsh = hsh.wrapping_mul(0x1000_0000_01b3);
    };
    for x in cam.view().to_cols_array() {
        mix(x);
    }
    mix(w as f32);
    mix(h as f32);
    mix(cam.distance);
    mix(cam.ortho as u32 as f32);
    match &cam.fixed {
        Some(f) => {
            mix(f.fov_y);
            mix(f.ortho_height.unwrap_or(-1.0));
        }
        None => mix(cam.fov_y),
    }
    hsh
}

fn aces_tonemap(x: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (x * a + Vec3::splat(b))) / (x * (x * c + Vec3::splat(d)) + Vec3::splat(e)))
        .clamp(Vec3::ZERO, Vec3::ONE)
}

/// Path-trace a full-resolution still (used for F12 when engine is Cycles).
pub fn render_still(
    scene: &Scene,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
    samples: u32,
    exposure: f32,
) -> Vec<u8> {
    let mut state = CyclesState::default();
    state.set_interactive(false);
    state.rebuild_scene(scene);
    let target = samples.max(1);
    while state.samples < target {
        state.accumulate(
            camera,
            width,
            height,
            exposure,
            target,
            width.max(height).max(256),
            1,
        );
    }
    // Resample to the requested size if the tracer used a smaller buffer.
    let (sw, sh) = state.size();
    if sw == width && sh == height {
        return state.preview;
    }
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let sx = (x as f32 * sw as f32 / width as f32) as u32;
            let sy = (y as f32 * sh as f32 / height as f32) as u32;
            let si = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4) as usize;
            let di = ((y * width + x) * 4) as usize;
            out[di..di + 4].copy_from_slice(&state.preview[si..si + 4]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_tris(n: usize, seed: u32) -> Vec<Tri> {
        let mut s = seed;
        let mut r = |scale: f32| (rand01(&mut s) - 0.5) * scale;
        (0..n)
            .map(|_| {
                let c = Vec3::new(r(20.0), r(20.0), r(20.0));
                let v0 = c + Vec3::new(r(1.0), r(1.0), r(1.0));
                let e1 = Vec3::new(r(1.5), r(1.5), r(1.5));
                let e2 = Vec3::new(r(1.5), r(1.5), r(1.5));
                Tri {
                    v0,
                    e1,
                    e2,
                    n0: Vec3::Y,
                    n1: Vec3::Y,
                    n2: Vec3::Y,
                    uv0: [0.0, 0.0],
                    uv1: [1.0, 0.0],
                    uv2: [0.0, 1.0],
                    mat: 0,
                }
            })
            .collect()
    }

    #[test]
    fn bvh_matches_brute_force() {
        let tris = random_tris(3000, 7);
        let (nodes, ordered) = build_bvh(tris);
        let geo = Geometry {
            tris: ordered,
            nodes,
            mats: vec![convert_material(&Material::default(), -1, -1)],
            lights: Vec::new(),
            textures: Vec::new(),
            graphs: Vec::new(),
            eps: 1e-4,
        };
        let mut s = 99u32;
        for _ in 0..400 {
            let o = unit_sphere(rand01(&mut s), rand01(&mut s)) * 18.0;
            let target = Vec3::new(
                (rand01(&mut s) - 0.5) * 10.0,
                (rand01(&mut s) - 0.5) * 10.0,
                (rand01(&mut s) - 0.5) * 10.0,
            );
            let d = (target - o).normalize();
            let fast = geo.closest(o, d, f32::INFINITY);
            let mut brute: Option<f32> = None;
            for t in &geo.tris {
                if let Some((tt, _, _)) = tri_hit(t, o, d, brute.unwrap_or(f32::INFINITY)) {
                    brute = Some(tt);
                }
            }
            match (fast, brute) {
                (None, None) => {}
                (Some(h), Some(b)) => assert!((h.t - b).abs() < 1e-3, "{} vs {}", h.t, b),
                (a, b) => panic!("mismatch: bvh {:?} brute {:?}", a.map(|h| h.t), b),
            }
            assert_eq!(
                geo.occluded(o, d, f32::INFINITY),
                brute.is_some(),
                "occlusion must agree with closest hit"
            );
        }
    }

    #[test]
    fn bsdf_sampling_weights_are_finite_and_bounded() {
        let m = Mat {
            albedo: Vec3::new(0.8, 0.6, 0.4),
            emission: Vec3::ZERO,
            metallic: 0.5,
            roughness: 0.4,
            tex: -1,
            graph: -1,
        };
        let b = Bsdf::new(&m);
        let n = Vec3::Y;
        let v = Vec3::new(0.3, 0.8, 0.2).normalize();
        let mut s = 5u32;
        let mut sum = Vec3::ZERO;
        let mut count = 0;
        for _ in 0..2000 {
            if let Some((l, w)) = b.sample(n, v, &mut s) {
                assert!(l.dot(n) > 0.0);
                assert!(w.is_finite());
                sum += w;
                count += 1;
            }
        }
        assert!(count > 1000);
        // Energy conservation: the average weight (albedo) must not exceed ~1.
        let avg = sum / 2000.0;
        assert!(avg.max_element() < 1.2, "avg weight {avg:?}");
    }

    #[test]
    fn empty_scene_renders_sky_and_keeps_requested_size() {
        let scene = Scene::default();
        let cam = OrbitCamera::default();
        let out = render_still(&scene, &cam, 64, 48, 2, 1.0);
        assert_eq!(out.len(), 64 * 48 * 4);
        assert!(out.chunks_exact(4).any(|p| p[0] > 10 || p[1] > 10 || p[2] > 10));
        assert!(out.chunks_exact(4).all(|p| p[3] == 255));
    }

    #[test]
    fn camera_center_ray_points_at_the_target() {
        let cam = OrbitCamera::default();
        let c = Cam::new(&cam, 1.0);
        let (o, d) = c.ray(0.0, 0.0);
        let expect = (cam.target - cam.eye()).normalize();
        assert!((o - cam.eye()).length() < 1e-3);
        assert!(d.dot(expect) > 0.999);
    }
}
