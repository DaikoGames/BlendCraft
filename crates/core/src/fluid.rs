//! Water / liquid simulation with Blender-style Domain, Flow and Effector roles.
//!
//! The solver is a local, CPU-based Position Based Fluids (PBF) implementation
//! (Macklin & Mueller, 2013): incompressible particle liquid with XSPH viscosity and a
//! simple cohesion term for surface tension. It is not Mantaflow/FLIP and will not
//! reproduce Blender's results exactly, but the workflow is the same:
//!
//! * **Domain**  - a box object that bounds the simulation and receives the bake.
//! * **Flow**    - a mesh that emits liquid (Geometry = initial volume, Inflow =
//!                 continuous source, Outflow = sink).
//! * **Effector**- a mesh that collides with the liquid (animated / rigid-body baked
//!                 objects move the water).
//!
//! Baking produces one surface mesh per frame (marching tetrahedra over a particle
//! density field). The result is stored as a `MeshSequence` on a "Fluid Mesh" object,
//! so it plays back on the timeline and can be exported to Alembic.
use crate::material::{AlphaMode, Material};
use crate::mesh::{MeshData, Vertex};
use crate::meshcache::{apply_mesh_caches, MeshSequence};
use crate::physics::apply_baked_frame;
use crate::scene::{Object, Scene};
use crate::simulation::FluidSettings;
use glam::{Mat3, Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use web_time::Instant;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluidType {
    #[default]
    Domain,
    Flow,
    Effector,
}

impl FluidType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Domain => "Domain",
            Self::Flow => "Flow",
            Self::Effector => "Effector",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowBehavior {
    /// Liquid exists inside the mesh at the first frame only.
    #[default]
    Geometry,
    /// The mesh keeps emitting liquid while it is active.
    Inflow,
    /// Liquid entering the mesh is removed.
    Outflow,
}

impl FlowBehavior {
    pub fn label(self) -> &'static str {
        match self {
            Self::Geometry => "Geometry",
            Self::Inflow => "Inflow",
            Self::Outflow => "Outflow",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidDomain {
    /// Grid divisions along the longest domain axis (Blender: Resolution Divisions).
    pub resolution: u32,
    /// Particles per cell and axis (Blender: Particles). 2 means 8 per cell.
    pub particles_per_cell: u32,
    /// Strength of XSPH velocity smoothing, 0..1.
    pub viscosity: f32,
    /// Cohesion strength, 0..1.
    pub surface_tension: f32,
    pub use_scene_gravity: bool,
    pub gravity: Vec3,
    /// Playback speed of the simulation clock.
    pub time_scale: f32,
    /// Minimum solver substeps per frame (more are added automatically for fast motion).
    pub substeps: u32,
    pub solver_iterations: u32,
    pub max_particles: u32,
    /// Generate a surface mesh from the particles.
    pub use_mesh: bool,
    /// Mesh grid refinement relative to the simulation grid (Blender: Mesh Scale).
    pub mesh_scale: u32,
    /// Laplacian smoothing passes on the surface mesh.
    pub mesh_smoothing: u32,
    /// Walls: -X, +X, -Y, +Y, -Z, +Z. An open side lets liquid leave the domain.
    pub border_collisions: [bool; 6],
    pub cache_frame_start: i32,
    pub cache_frame_end: i32,
    /// Object that shows the baked surface ("Fluid Mesh"), created by the first bake.
    pub mesh_object: Option<u64>,
}

impl Default for FluidDomain {
    fn default() -> Self {
        Self {
            resolution: 32,
            particles_per_cell: 2,
            viscosity: 0.03,
            surface_tension: 0.1,
            use_scene_gravity: true,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            time_scale: 1.0,
            substeps: 2,
            solver_iterations: 4,
            max_particles: 100_000,
            use_mesh: true,
            mesh_scale: 2,
            mesh_smoothing: 2,
            border_collisions: [true; 6],
            cache_frame_start: 1,
            cache_frame_end: 100,
            mesh_object: None,
        }
    }
}

impl FluidDomain {
    /// Convert the old settings-only placeholder into a real domain.
    pub fn from_legacy(old: &FluidSettings) -> Self {
        let mut d = Self::default();
        d.resolution = old.domain_resolution.clamp(8, 256);
        d.viscosity = old.viscosity.clamp(0.0, 1.0);
        d.surface_tension = old.surface_tension.clamp(0.0, 1.0);
        d.time_scale = old.time_scale.max(0.01);
        d.max_particles = old.max_particles.max(100);
        if old.gravity_scale != 1.0 {
            d.use_scene_gravity = false;
            d.gravity = Vec3::new(0.0, -9.81 * old.gravity_scale, 0.0);
        }
        d
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidFlow {
    pub enabled: bool,
    pub behavior: FlowBehavior,
    /// Treat the mesh as an open surface and fill everything below it
    /// (Blender: Use Plane Initialization).
    pub is_planar: bool,
    pub use_initial_velocity: bool,
    pub initial_velocity: Vec3,
    /// Inflow is active from `frame_start` to `frame_end`.
    pub frame_start: i32,
    pub frame_end: i32,
}

impl Default for FluidFlow {
    fn default() -> Self {
        Self {
            enabled: true,
            behavior: FlowBehavior::Geometry,
            is_planar: false,
            use_initial_velocity: false,
            initial_velocity: Vec3::ZERO,
            frame_start: 1,
            frame_end: 250,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidEffector {
    pub enabled: bool,
    /// Extra collision thickness in world units.
    pub surface_thickness: f32,
}

impl Default for FluidEffector {
    fn default() -> Self {
        Self { enabled: true, surface_thickness: 0.0 }
    }
}

/// One fluid role assigned to an object (Blender: Physics > Fluid > Type).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FluidObject {
    Domain(FluidDomain),
    Flow(FluidFlow),
    Effector(FluidEffector),
}

impl FluidObject {
    pub fn kind(&self) -> FluidType {
        match self {
            Self::Domain(_) => FluidType::Domain,
            Self::Flow(_) => FluidType::Flow,
            Self::Effector(_) => FluidType::Effector,
        }
    }

    pub fn with_kind(kind: FluidType) -> Self {
        match kind {
            FluidType::Domain => Self::Domain(FluidDomain::default()),
            FluidType::Flow => Self::Flow(FluidFlow::default()),
            FluidType::Effector => Self::Effector(FluidEffector::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Small parallel helper (std only; serial on wasm)
// ---------------------------------------------------------------------------

fn par_fill<T: Send, F: Fn(usize) -> T + Sync>(out: &mut [T], f: F) {
    let n = out.len();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1).min(16);
        if threads > 1 && n >= 2048 {
            let chunk = (n + threads - 1) / threads;
            std::thread::scope(|scope| {
                for (ci, part) in out.chunks_mut(chunk).enumerate() {
                    let f = &f;
                    scope.spawn(move || {
                        for (k, slot) in part.iter_mut().enumerate() {
                            *slot = f(ci * chunk + k);
                        }
                    });
                }
            });
            return;
        }
    }
    for i in 0..n {
        out[i] = f(i);
    }
}

// ---------------------------------------------------------------------------
// SPH kernels and neighbor search
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Kernels {
    h: f32,
    h2: f32,
    poly6: f32,
    spiky: f32,
}

impl Kernels {
    fn new(h: f32) -> Self {
        let pi = std::f32::consts::PI;
        Self { h, h2: h * h, poly6: 315.0 / (64.0 * pi * h.powi(9)), spiky: 45.0 / (pi * h.powi(6)) }
    }
    #[inline]
    fn w(&self, r2: f32) -> f32 {
        if r2 >= self.h2 {
            0.0
        } else {
            let d = self.h2 - r2;
            self.poly6 * d * d * d
        }
    }
    /// Gradient of W(|d|) with respect to the first particle (d = p_i - p_j).
    #[inline]
    fn grad(&self, d: Vec3, r2: f32) -> Vec3 {
        if r2 >= self.h2 || r2 < 1e-12 {
            return Vec3::ZERO;
        }
        let r = r2.sqrt();
        let t = self.h - r;
        d * (-self.spiky * t * t / r)
    }
}

struct Grid {
    min: Vec3,
    cell: f32,
    dims: [usize; 3],
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Grid {
    fn build(points: &[Vec3], min: Vec3, max: Vec3, cell: f32) -> Grid {
        let ext = (max - min).max(Vec3::splat(cell));
        let dims = [
            ((ext.x / cell).ceil() as usize).max(1) + 1,
            ((ext.y / cell).ceil() as usize).max(1) + 1,
            ((ext.z / cell).ceil() as usize).max(1) + 1,
        ];
        let total = dims[0] * dims[1] * dims[2];
        let mut grid = Grid { min, cell, dims, start: vec![0u32; total + 1], items: vec![0u32; points.len()] };
        let mut cell_of: Vec<u32> = Vec::with_capacity(points.len());
        for p in points {
            let c = grid.cell_of(*p);
            let idx = ((c[2] * dims[1] + c[1]) * dims[0] + c[0]) as u32;
            cell_of.push(idx);
            grid.start[idx as usize + 1] += 1;
        }
        for i in 0..total {
            grid.start[i + 1] += grid.start[i];
        }
        let mut fill = grid.start.clone();
        for (i, idx) in cell_of.iter().enumerate() {
            let slot = fill[*idx as usize];
            grid.items[slot as usize] = i as u32;
            fill[*idx as usize] += 1;
        }
        grid
    }

    #[inline]
    fn cell_of(&self, p: Vec3) -> [usize; 3] {
        let q = (p - self.min) / self.cell;
        [
            (q.x.max(0.0) as usize).min(self.dims[0] - 1),
            (q.y.max(0.0) as usize).min(self.dims[1] - 1),
            (q.z.max(0.0) as usize).min(self.dims[2] - 1),
        ]
    }
}

fn build_neighbors(points: &[Vec3], grid: &Grid, h2: f32, nbr: &mut Vec<u32>, nbr_start: &mut Vec<u32>) {
    nbr.clear();
    nbr_start.clear();
    nbr_start.push(0);
    for (i, p) in points.iter().enumerate() {
        let c = grid.cell_of(*p);
        let z0 = c[2].saturating_sub(1);
        let z1 = (c[2] + 1).min(grid.dims[2] - 1);
        let y0 = c[1].saturating_sub(1);
        let y1 = (c[1] + 1).min(grid.dims[1] - 1);
        let x0 = c[0].saturating_sub(1);
        let x1 = (c[0] + 1).min(grid.dims[0] - 1);
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let ci = (z * grid.dims[1] + y) * grid.dims[0] + x;
                    let (a, b) = (grid.start[ci] as usize, grid.start[ci + 1] as usize);
                    for k in a..b {
                        let j = grid.items[k] as usize;
                        if j != i && (points[j] - *p).length_squared() < h2 {
                            nbr.push(j as u32);
                        }
                    }
                }
            }
        }
        nbr_start.push(nbr.len() as u32);
    }
}

// ---------------------------------------------------------------------------
// Effector collision geometry
// ---------------------------------------------------------------------------

fn closest_point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Uniform grid over triangle indices. Built once per frame over the volume swept
/// by the (linearly interpolated) effector.
struct TriGrid {
    min: Vec3,
    cell: f32,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

impl TriGrid {
    fn build(a: &[[Vec3; 3]], b: &[[Vec3; 3]], pad: f32) -> TriGrid {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for t in a.iter().chain(b.iter()) {
            for v in t.iter() {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if a.is_empty() {
            lo = Vec3::ZERO;
            hi = Vec3::ZERO;
        }
        lo -= Vec3::splat(pad);
        hi += Vec3::splat(pad);
        let ext = hi - lo;
        let cell = (ext.max_element() / 24.0).max(pad).max(1e-4);
        let dims = [
            ((ext.x / cell).ceil() as usize).max(1),
            ((ext.y / cell).ceil() as usize).max(1),
            ((ext.z / cell).ceil() as usize).max(1),
        ];
        let mut grid = TriGrid { min: lo, cell, dims, cells: vec![Vec::new(); dims[0] * dims[1] * dims[2]] };
        for (ti, ta) in a.iter().enumerate() {
            let tb = b.get(ti).unwrap_or(ta);
            let mut tlo = Vec3::splat(f32::MAX);
            let mut thi = Vec3::splat(f32::MIN);
            for v in ta.iter().chain(tb.iter()) {
                tlo = tlo.min(*v);
                thi = thi.max(*v);
            }
            tlo -= Vec3::splat(pad);
            thi += Vec3::splat(pad);
            let c0 = grid.index_of(tlo);
            let c1 = grid.index_of(thi);
            for z in c0[2]..=c1[2] {
                for y in c0[1]..=c1[1] {
                    for x in c0[0]..=c1[0] {
                        grid.cells[(z * dims[1] + y) * dims[0] + x].push(ti as u32);
                    }
                }
            }
        }
        grid
    }

    fn index_of(&self, p: Vec3) -> [usize; 3] {
        let q = (p - self.min) / self.cell;
        [
            (q.x.max(0.0) as usize).min(self.dims[0] - 1),
            (q.y.max(0.0) as usize).min(self.dims[1] - 1),
            (q.z.max(0.0) as usize).min(self.dims[2] - 1),
        ]
    }

    fn inside(&self, p: Vec3) -> bool {
        let q = (p - self.min) / self.cell;
        q.x >= 0.0
            && q.y >= 0.0
            && q.z >= 0.0
            && q.x < self.dims[0] as f32
            && q.y < self.dims[1] as f32
            && q.z < self.dims[2] as f32
    }

    fn candidates(&self, p: Vec3) -> &[u32] {
        let c = self.index_of(p);
        &self.cells[(c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]]
    }
}

pub struct EffectorGeo {
    tris_a: Vec<[Vec3; 3]>,
    tris_b: Vec<[Vec3; 3]>,
    thickness: f32,
    grid: TriGrid,
}

impl EffectorGeo {
    fn new(tris_a: Vec<[Vec3; 3]>, tris_b: Vec<[Vec3; 3]>, thickness: f32, pad: f32) -> Self {
        // Without matching topology there is no usable motion: treat as static.
        let tris_b = if tris_b.len() == tris_a.len() { tris_b } else { tris_a.clone() };
        let grid = TriGrid::build(&tris_a, &tris_b, pad + thickness);
        Self { tris_a, tris_b, thickness, grid }
    }

    fn lerped(&self, t: f32) -> Vec<[Vec3; 3]> {
        self.tris_a
            .iter()
            .zip(self.tris_b.iter())
            .map(|(a, b)| [a[0].lerp(b[0], t), a[1].lerp(b[1], t), a[2].lerp(b[2], t)])
            .collect()
    }

    /// Push `p` out of the effector (position-based collision response).
    fn push_out(&self, tris: &[[Vec3; 3]], p: &mut Vec3, margin: f32) {
        if !self.grid.inside(*p) {
            return;
        }
        let mut best_d2 = f32::MAX;
        let mut best_cp = Vec3::ZERO;
        let mut best_n = Vec3::Y;
        for &ti in self.grid.candidates(*p) {
            let Some(t) = tris.get(ti as usize) else { continue };
            let cp = closest_point_triangle(*p, t[0], t[1], t[2]);
            let d2 = (*p - cp).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best_cp = cp;
                best_n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            }
        }
        if best_d2 == f32::MAX || best_n == Vec3::ZERO {
            return;
        }
        let signed = (*p - best_cp).dot(best_n);
        let wanted = self.thickness + margin;
        if signed < wanted {
            *p = best_cp + best_n * wanted;
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice fill of closed meshes
// ---------------------------------------------------------------------------

/// Lattice cells (ix, iy, iz) whose centers lie inside the closed mesh. The lattice is
/// anchored at the world origin so separate flow objects interleave cleanly.
/// With `planar` the mesh is treated as an open surface: everything below it is filled.
fn fill_lattice(tris: &[[Vec3; 3]], planar: bool, h0: f32, lo: Vec3, hi: Vec3) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    if tris.is_empty() || h0 <= 0.0 {
        return out;
    }
    let nx = 1.234e-4 * h0;
    let nz = 2.345e-4 * h0;
    let ix_lo = ((lo.x - nx) / h0 - 0.5).ceil() as i32;
    let ix_hi = ((hi.x - nx) / h0 - 0.5).floor() as i32;
    let iz_lo = ((lo.z - nz) / h0 - 0.5).ceil() as i32;
    let iz_hi = ((hi.z - nz) / h0 - 0.5).floor() as i32;
    let iy_lo = (lo.y / h0 - 0.5).ceil() as i32;
    let iy_hi = (hi.y / h0 - 0.5).floor() as i32;
    let mut columns: HashMap<(i32, i32), Vec<f32>> = HashMap::new();
    for t in tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        let det = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z);
        if det.abs() < 1e-14 {
            continue;
        }
        let min_x = a.x.min(b.x).min(c.x);
        let max_x = a.x.max(b.x).max(c.x);
        let min_z = a.z.min(b.z).min(c.z);
        let max_z = a.z.max(b.z).max(c.z);
        let x0 = (((min_x - nx) / h0 - 0.5).ceil() as i32).max(ix_lo);
        let x1 = (((max_x - nx) / h0 - 0.5).floor() as i32).min(ix_hi);
        let z0 = (((min_z - nz) / h0 - 0.5).ceil() as i32).max(iz_lo);
        let z1 = (((max_z - nz) / h0 - 0.5).floor() as i32).min(iz_hi);
        if x0 > x1 || z0 > z1 {
            continue;
        }
        for ix in x0..=x1 {
            let x = (ix as f32 + 0.5) * h0 + nx;
            for iz in z0..=z1 {
                let z = (iz as f32 + 0.5) * h0 + nz;
                let l1 = ((b.z - c.z) * (x - c.x) + (c.x - b.x) * (z - c.z)) / det;
                let l2 = ((c.z - a.z) * (x - c.x) + (a.x - c.x) * (z - c.z)) / det;
                let l3 = 1.0 - l1 - l2;
                if l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0 {
                    columns.entry((ix, iz)).or_default().push(l1 * a.y + l2 * b.y + l3 * c.y);
                }
            }
        }
    }
    for ((ix, iz), mut ys) in columns {
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if planar {
            let top = ys[ys.len() - 1];
            let top_i = ((top / h0 - 0.5).floor() as i32).min(iy_hi);
            for iy in iy_lo..=top_i {
                out.push((ix, iy, iz));
            }
        } else {
            let mut k = 0;
            while k + 1 < ys.len() {
                let i0 = ((ys[k] / h0 - 0.5).ceil() as i32).max(iy_lo);
                let i1 = ((ys[k + 1] / h0 - 0.5).floor() as i32).min(iy_hi);
                for iy in i0..=i1 {
                    out.push((ix, iy, iz));
                }
                k += 2;
            }
        }
    }
    out
}

fn jitter(ix: i32, iy: i32, iz: i32) -> Vec3 {
    let mut h = (ix as u32).wrapping_mul(73856093) ^ (iy as u32).wrapping_mul(19349663) ^ (iz as u32).wrapping_mul(83492791);
    let mut next = || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        (h % 2001) as f32 / 1000.0 - 1.0
    };
    Vec3::new(next(), next(), next())
}

fn lattice_position(ix: i32, iy: i32, iz: i32, h0: f32) -> Vec3 {
    Vec3::new((ix as f32 + 0.5) * h0, (iy as f32 + 0.5) * h0, (iz as f32 + 0.5) * h0) + jitter(ix, iy, iz) * (0.04 * h0)
}

fn lattice_key(p: Vec3, h0: f32) -> (i32, i32, i32) {
    ((p.x / h0).floor() as i32, (p.y / h0).floor() as i32, (p.z / h0).floor() as i32)
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

pub struct FluidSolver {
    pub h0: f32,
    kernels: Kernels,
    rho0: f32,
    eps: f32,
    bmin: Vec3,
    bmax: Vec3,
    pub pos: Vec<Vec3>,
    vel: Vec<Vec3>,
    pred: Vec<Vec3>,
    lambda: Vec<f32>,
    delta: Vec<Vec3>,
    vel_next: Vec<Vec3>,
    nbr: Vec<u32>,
    nbr_start: Vec<u32>,
}

impl FluidSolver {
    pub fn new(h0: f32, bmin: Vec3, bmax: Vec3) -> Self {
        let h = 2.0 * h0;
        let kernels = Kernels::new(h);
        // Rest density of a perfect lattice with spacing h0.
        let mut rho0 = 0.0f32;
        for x in -3i32..=3 {
            for y in -3i32..=3 {
                for z in -3i32..=3 {
                    let r2 = (x * x + y * y + z * z) as f32 * h0 * h0;
                    rho0 += kernels.w(r2);
                }
            }
        }
        let g_unit = kernels.spiky * h * h / rho0.max(1e-9);
        Self {
            h0,
            kernels,
            rho0,
            eps: 0.5 * g_unit * g_unit,
            bmin,
            bmax,
            pos: Vec::new(),
            vel: Vec::new(),
            pred: Vec::new(),
            lambda: Vec::new(),
            delta: Vec::new(),
            vel_next: Vec::new(),
            nbr: Vec::new(),
            nbr_start: Vec::new(),
        }
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn add_particle(&mut self, p: Vec3, v: Vec3) {
        self.pos.push(p);
        self.vel.push(v);
    }

    fn remove_where<F: Fn(Vec3) -> bool>(&mut self, f: F) {
        let mut i = 0;
        while i < self.pos.len() {
            if f(self.pos[i]) {
                self.pos.swap_remove(i);
                self.vel.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn clamp_to_domain(&self, p: &mut Vec3, borders: &[bool; 6]) {
        let margin = 0.5 * self.h0;
        for a in 0..3 {
            let lo = self.bmin[a] + margin;
            let hi = self.bmax[a] - margin;
            if p[a] < lo && borders[2 * a] {
                p[a] = lo;
            }
            if p[a] > hi && borders[2 * a + 1] {
                p[a] = hi;
            }
        }
    }

    pub fn advance_frame(&mut self, frame_dt: f32, gravity: Vec3, domain: &FluidDomain, effectors: &[EffectorGeo]) {
        if self.pos.is_empty() || frame_dt <= 0.0 {
            return;
        }
        let vmax = self.vel.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let needed = ((vmax * frame_dt) / (0.5 * self.h0)).ceil() as u32;
        let substeps = domain.substeps.max(1).max(needed).min(24);
        let dt = frame_dt / substeps as f32;
        for s in 0..substeps {
            let t0 = s as f32 / substeps as f32;
            let t1 = (s + 1) as f32 / substeps as f32;
            let _ = t0;
            self.substep(dt, gravity, domain, effectors, t1);
        }
    }

    fn substep(&mut self, dt: f32, gravity: Vec3, domain: &FluidDomain, effectors: &[EffectorGeo], t_end: f32) {
        let n = self.pos.len();
        if n == 0 {
            return;
        }
        let borders = domain.border_collisions;
        self.pred.resize(n, Vec3::ZERO);
        self.lambda.resize(n, 0.0);
        self.delta.resize(n, Vec3::ZERO);
        self.vel_next.resize(n, Vec3::ZERO);

        for i in 0..n {
            self.vel[i] += gravity * dt;
            let mut p = self.pos[i] + self.vel[i] * dt;
            self.clamp_to_domain(&mut p, &borders);
            self.pred[i] = p;
        }

        let lerped: Vec<Vec<[Vec3; 3]>> = effectors.iter().map(|e| e.lerped(t_end)).collect();
        let margin = 0.5 * self.h0;

        let grid = Grid::build(&self.pred, self.bmin, self.bmax, self.kernels.h);
        build_neighbors(&self.pred, &grid, self.kernels.h2, &mut self.nbr, &mut self.nbr_start);

        let kernels = self.kernels;
        let rho0 = self.rho0;
        let eps = self.eps;
        let max_step = 0.5 * self.h0;
        for _ in 0..domain.solver_iterations.max(1) {
            {
                let pred = &self.pred;
                let nbr = &self.nbr;
                let nbr_start = &self.nbr_start;
                par_fill(&mut self.lambda, |i| {
                    let pi = pred[i];
                    let (a, b) = (nbr_start[i] as usize, nbr_start[i + 1] as usize);
                    let mut rho = kernels.w(0.0);
                    let mut grad_i = Vec3::ZERO;
                    let mut sum_sq = 0.0f32;
                    for k in a..b {
                        let d = pi - pred[nbr[k] as usize];
                        let r2 = d.length_squared();
                        rho += kernels.w(r2);
                        let g = kernels.grad(d, r2) / rho0;
                        grad_i += g;
                        sum_sq += g.length_squared();
                    }
                    let c = rho / rho0 - 1.0;
                    if c <= 0.0 {
                        0.0
                    } else {
                        -c / (sum_sq + grad_i.length_squared() + eps)
                    }
                });
            }
            {
                let pred = &self.pred;
                let lambda = &self.lambda;
                let nbr = &self.nbr;
                let nbr_start = &self.nbr_start;
                par_fill(&mut self.delta, |i| {
                    let pi = pred[i];
                    let (a, b) = (nbr_start[i] as usize, nbr_start[i + 1] as usize);
                    let mut dp = Vec3::ZERO;
                    for k in a..b {
                        let j = nbr[k] as usize;
                        let d = pi - pred[j];
                        let r2 = d.length_squared();
                        dp += kernels.grad(d, r2) * (lambda[i] + lambda[j]);
                    }
                    let dp = dp / rho0;
                    let len = dp.length();
                    if len > max_step {
                        dp * (max_step / len)
                    } else {
                        dp
                    }
                });
            }
            for i in 0..n {
                let mut p = self.pred[i] + self.delta[i];
                self.clamp_to_domain(&mut p, &borders);
                for (e, tris) in effectors.iter().zip(lerped.iter()) {
                    e.push_out(tris, &mut p, margin);
                }
                self.pred[i] = p;
            }
        }

        let inv_dt = 1.0 / dt;
        let v_cap = 1.0 * self.h0 * inv_dt;
        for i in 0..n {
            let mut v = (self.pred[i] - self.pos[i]) * inv_dt;
            let len = v.length();
            if len > v_cap {
                v *= v_cap / len;
            }
            self.vel[i] = v;
            self.pos[i] = self.pred[i];
        }

        // XSPH viscosity and cohesion (surface tension) on the new state.
        let visc = domain.viscosity.clamp(0.0, 1.0);
        let tension = domain.surface_tension.clamp(0.0, 1.0);
        let gmag = gravity.length().max(1.0);
        let h = self.kernels.h;
        {
            let pos = &self.pos;
            let vel = &self.vel;
            let nbr = &self.nbr;
            let nbr_start = &self.nbr_start;
            par_fill(&mut self.vel_next, |i| {
                let (a, b) = (nbr_start[i] as usize, nbr_start[i + 1] as usize);
                let mut vsum = Vec3::ZERO;
                let mut csum = Vec3::ZERO;
                let mut wsum = 0.0f32;
                for k in a..b {
                    let j = nbr[k] as usize;
                    let d = pos[j] - pos[i];
                    let w = kernels.w(d.length_squared());
                    wsum += w;
                    vsum += (vel[j] - vel[i]) * w;
                    csum += d * w;
                }
                let mut v = vel[i] + vsum * (visc / rho0);
                if wsum > 0.0 {
                    v += (csum / wsum) / h * (tension * 3.0 * gmag * dt);
                }
                v
            });
        }
        std::mem::swap(&mut self.vel, &mut self.vel_next);
        for v in self.vel.iter_mut() {
            let len = v.length();
            if len > v_cap {
                *v *= v_cap / len;
            }
        }

        // Particles that left through an open wall are deleted.
        let (bmin, bmax, h0) = (self.bmin, self.bmax, self.h0);
        self.remove_where(|p| {
            for a in 0..3 {
                if (!borders[2 * a] && p[a] < bmin[a] + 0.55 * h0) || (!borders[2 * a + 1] && p[a] > bmax[a] - 0.55 * h0) {
                    return true;
                }
            }
            false
        });
    }
}

// ---------------------------------------------------------------------------
// Surface meshing (marching tetrahedra on a particle density field)
// ---------------------------------------------------------------------------

const CUBE_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];
const CUBE_TETS: [[usize; 4]; 6] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

pub fn mesh_from_particles(particles: &[Vec3], bmin: Vec3, bmax: Vec3, h0: f32, mesh_cell: f32, smoothing: u32, name: &str) -> MeshData {
    let mut mesh = MeshData { name: name.to_string(), ..Default::default() };
    if particles.is_empty() || h0 <= 0.0 {
        return mesh;
    }
    let radius = 2.0 * h0;
    let mut mc = mesh_cell.max(0.25 * h0);
    let ext = (bmax - bmin).max(Vec3::splat(mc));
    // Keep the grid at a sane size.
    loop {
        let cells = (ext.x / mc).ceil() * (ext.y / mc).ceil() * (ext.z / mc).ceil();
        if cells <= 6.0e6 {
            break;
        }
        mc *= 1.2;
    }
    let dims = [
        (ext.x / mc).ceil() as usize + 1,
        (ext.y / mc).ceil() as usize + 1,
        (ext.z / mc).ceil() as usize + 1,
    ];
    let idx = |x: usize, y: usize, z: usize| (z * dims[1] + y) * dims[0] + x;
    let mut field = vec![0.0f32; dims[0] * dims[1] * dims[2]];

    let mut plo = Vec3::splat(f32::MAX);
    let mut phi = Vec3::splat(f32::MIN);
    let reach = (radius / mc).ceil() as i32;
    for p in particles {
        plo = plo.min(*p);
        phi = phi.max(*p);
        let q = (*p - bmin) / mc;
        let cx = q.x.floor() as i32;
        let cy = q.y.floor() as i32;
        let cz = q.z.floor() as i32;
        for z in (cz - reach)..=(cz + reach + 1) {
            if z < 0 || z as usize >= dims[2] {
                continue;
            }
            for y in (cy - reach)..=(cy + reach + 1) {
                if y < 0 || y as usize >= dims[1] {
                    continue;
                }
                for x in (cx - reach)..=(cx + reach + 1) {
                    if x < 0 || x as usize >= dims[0] {
                        continue;
                    }
                    let gp = bmin + Vec3::new(x as f32, y as f32, z as f32) * mc;
                    let r2 = (gp - *p).length_squared() / (radius * radius);
                    if r2 < 1.0 {
                        let t = 1.0 - r2;
                        field[idx(x as usize, y as usize, z as usize)] += t * t * t;
                    }
                }
            }
        }
    }

    // Field value inside a perfectly packed block, used to place the iso surface.
    let mut interior = 0.0f32;
    for x in -3i32..=3 {
        for y in -3i32..=3 {
            for z in -3i32..=3 {
                let r2 = (x * x + y * y + z * z) as f32 * h0 * h0 / (radius * radius);
                if r2 < 1.0 {
                    let t = 1.0 - r2;
                    interior += t * t * t;
                }
            }
        }
    }
    let iso = 0.4 * interior;

    let lo = ((plo - bmin) / mc - Vec3::splat(reach as f32 + 1.0)).max(Vec3::ZERO);
    let hi = ((phi - bmin) / mc + Vec3::splat(reach as f32 + 1.0)).max(Vec3::ZERO);
    let (x0, y0, z0) = (lo.x as usize, lo.y as usize, lo.z as usize);
    let x1 = (hi.x as usize).min(dims[0] - 2);
    let y1 = (hi.y as usize).min(dims[1] - 2);
    let z1 = (hi.z as usize).min(dims[2] - 2);

    let mut weld: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut corner_idx = [0u32; 8];
                let mut corner_val = [0.0f32; 8];
                let mut inside = 0;
                for (k, c) in CUBE_CORNERS.iter().enumerate() {
                    let i = idx(x + c[0], y + c[1], z + c[2]);
                    corner_idx[k] = i as u32;
                    corner_val[k] = field[i];
                    if corner_val[k] >= iso {
                        inside += 1;
                    }
                }
                if inside == 0 || inside == 8 {
                    continue;
                }
                let corner_pos = |k: usize| -> Vec3 {
                    let c = CUBE_CORNERS[k];
                    bmin + Vec3::new((x + c[0]) as f32, (y + c[1]) as f32, (z + c[2]) as f32) * mc
                };
                for tet in CUBE_TETS.iter() {
                    polygonise_tet(tet, &corner_idx, &corner_val, &corner_pos, iso, &mut weld, &mut vertices, &mut indices);
                }
            }
        }
    }

    // Laplacian smoothing.
    if smoothing > 0 && !vertices.is_empty() {
        for _ in 0..smoothing {
            let mut sum = vec![Vec3::ZERO; vertices.len()];
            let mut cnt = vec![0.0f32; vertices.len()];
            for t in indices.chunks_exact(3) {
                for e in 0..3 {
                    let a = t[e] as usize;
                    let b = t[(e + 1) % 3] as usize;
                    sum[a] += vertices[b];
                    cnt[a] += 1.0;
                    sum[b] += vertices[a];
                    cnt[b] += 1.0;
                }
            }
            for i in 0..vertices.len() {
                if cnt[i] > 0.0 {
                    vertices[i] = vertices[i].lerp(sum[i] / cnt[i], 0.5);
                }
            }
        }
    }

    mesh.vertices = vertices
        .iter()
        .map(|p| Vertex { position: p.to_array(), normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        })
        .collect();
    mesh.indices = indices;
    mesh.compute_normals();
    mesh
}

fn polygonise_tet<F: Fn(usize) -> Vec3>(
    tet: &[usize; 4],
    corner_idx: &[u32; 8],
    corner_val: &[f32; 8],
    corner_pos: &F,
    iso: f32,
    weld: &mut HashMap<(u32, u32), u32>,
    vertices: &mut Vec<Vec3>,
    indices: &mut Vec<u32>,
) {
    let mut ins: Vec<usize> = Vec::with_capacity(4);
    let mut out: Vec<usize> = Vec::with_capacity(4);
    for &k in tet.iter() {
        if corner_val[k] >= iso {
            ins.push(k);
        } else {
            out.push(k);
        }
    }
    if ins.is_empty() || out.is_empty() {
        return;
    }
    let mut edge_vertex = |a: usize, b: usize, vertices: &mut Vec<Vec3>| -> u32 {
        let (ia, ib) = (corner_idx[a], corner_idx[b]);
        let key = if ia < ib { (ia, ib) } else { (ib, ia) };
        if let Some(v) = weld.get(&key) {
            return *v;
        }
        let (fa, fb) = (corner_val[a], corner_val[b]);
        let t = if (fb - fa).abs() < 1e-12 { 0.5 } else { ((iso - fa) / (fb - fa)).clamp(0.0, 1.0) };
        let p = corner_pos(a).lerp(corner_pos(b), t);
        let v = vertices.len() as u32;
        vertices.push(p);
        weld.insert(key, v);
        v
    };
    let mut emit = |tri: [u32; 3], outward: Vec3, vertices: &Vec<Vec3>| {
        let (a, b, c) = (vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]);
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1e-20 {
            return;
        }
        if n.dot(outward) >= 0.0 {
            indices.extend_from_slice(&[tri[0], tri[1], tri[2]]);
        } else {
            indices.extend_from_slice(&[tri[0], tri[2], tri[1]]);
        }
    };
    match ins.len() {
        1 => {
            let a = ins[0];
            let v0 = edge_vertex(a, out[0], &mut *vertices);
            let v1 = edge_vertex(a, out[1], &mut *vertices);
            let v2 = edge_vertex(a, out[2], &mut *vertices);
            let centroid = (corner_pos(out[0]) + corner_pos(out[1]) + corner_pos(out[2])) / 3.0;
            emit([v0, v1, v2], centroid - corner_pos(a), &*vertices);
        }
        3 => {
            let d = out[0];
            let v0 = edge_vertex(d, ins[0], &mut *vertices);
            let v1 = edge_vertex(d, ins[1], &mut *vertices);
            let v2 = edge_vertex(d, ins[2], &mut *vertices);
            let centroid = (corner_pos(ins[0]) + corner_pos(ins[1]) + corner_pos(ins[2])) / 3.0;
            emit([v0, v1, v2], corner_pos(d) - centroid, &*vertices);
        }
        2 => {
            let (a1, a2, b1, b2) = (ins[0], ins[1], out[0], out[1]);
            let v0 = edge_vertex(a1, b1, &mut *vertices);
            let v1 = edge_vertex(a1, b2, &mut *vertices);
            let v2 = edge_vertex(a2, b2, &mut *vertices);
            let v3 = edge_vertex(a2, b1, &mut *vertices);
            let dir = (corner_pos(b1) + corner_pos(b2)) * 0.5 - (corner_pos(a1) + corner_pos(a2)) * 0.5;
            emit([v0, v1, v2], dir, &*vertices);
            emit([v0, v2, v3], dir, &*vertices);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Baking
// ---------------------------------------------------------------------------

fn world_triangles(scene: &Scene, index: usize, world: &Mat4) -> Vec<[Vec3; 3]> {
    let mesh = &scene.objects[index].mesh;
    let pts: Vec<Vec3> = mesh.vertices.iter().map(|v| world.transform_point3(Vec3::from(v.position))).collect();
    let flip = Mat3::from_mat4(*world).determinant() < 0.0;
    let mut out = Vec::with_capacity(mesh.indices.len() / 3);
    for t in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if a < pts.len() && b < pts.len() && c < pts.len() {
            out.push(if flip { [pts[a], pts[c], pts[b]] } else { [pts[a], pts[b], pts[c]] });
        }
    }
    out
}

/// Copy of the scene with animation, rigid-body bakes and mesh caches evaluated at `frame`.
pub fn scene_at_frame(base: &Scene, frame: i32) -> Scene {
    let mut s = base.clone();
    let f = frame as f32;
    let keys: Vec<(u64, (Option<Vec3>, Option<glam::Quat>, Option<Vec3>))> =
        s.animation.objects.iter().map(|(id, oa)| (*id, oa.eval(f))).collect();
    for (id, (loc, rot, scl)) in keys {
        let Some(index) = s.index_of(id) else { continue };
        let driven_by_physics = s.rigid_bodies.get(&id).map_or(false, |b| {
            b.enabled && b.body_type == crate::physics::RigidBodyType::Active && b.dynamic && !b.animated
        });
        if driven_by_physics {
            continue;
        }
        let o = &mut s.objects[index];
        if let Some(v) = loc {
            o.translation = v;
        }
        if let Some(v) = rot {
            o.rotation = v;
        }
        if let Some(v) = scl {
            o.scale = v;
        }
    }
    apply_baked_frame(&mut s, frame);
    apply_mesh_caches(&mut s, f);
    s
}

pub struct FluidBaker {
    domain_id: u64,
    domain_name: String,
    base: Scene,
    domain: FluidDomain,
    solver: FluidSolver,
    bmin: Vec3,
    bmax: Vec3,
    mesh_cell: f32,
    gravity: Vec3,
    fps: f32,
    start: i32,
    end: i32,
    frame: i32,
    prev_scene: Option<Scene>,
    frames: Vec<Arc<MeshData>>,
    pub particle_counts: Vec<u32>,
    pub warnings: Vec<String>,
}

impl FluidBaker {
    pub fn new(scene: &Scene, domain_id: u64) -> Result<Self, String> {
        let Some(FluidObject::Domain(domain)) = scene.fluids.get(&domain_id).cloned() else {
            return Err("The selected object is not a fluid domain.".to_string());
        };
        let index = scene.index_of(domain_id).ok_or_else(|| "The domain object no longer exists.".to_string())?;
        let worlds = scene.world_matrices();
        let bounds = scene.objects[index].mesh.bounds().ok_or_else(|| "The domain has no mesh.".to_string())?;
        let mut bmin = Vec3::splat(f32::MAX);
        let mut bmax = Vec3::splat(f32::MIN);
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { bounds.min.x } else { bounds.max.x },
                if i & 2 == 0 { bounds.min.y } else { bounds.max.y },
                if i & 4 == 0 { bounds.min.z } else { bounds.max.z },
            );
            let w = worlds[index].transform_point3(corner);
            bmin = bmin.min(w);
            bmax = bmax.max(w);
        }
        let longest = (bmax - bmin).max_element();
        if longest <= 1e-4 {
            return Err("The domain is too small.".to_string());
        }
        let start = domain.cache_frame_start;
        let end = domain.cache_frame_end.max(start);
        let cell = longest / domain.resolution.max(4) as f32;
        let mut h0 = cell / domain.particles_per_cell.clamp(1, 4) as f32;

        let gravity = if domain.use_scene_gravity { scene.physics_world.gravity } else { domain.gravity };
        let fps = scene.animation.fps.max(1) as f32;
        let first = scene_at_frame(scene, start);

        // Pick a particle spacing that respects the particle budget.
        let mut warnings = Vec::new();
        for attempt in 0..4 {
            let count = Self::initial_cells(&first, domain_id, h0, bmin, bmax).len();
            if count as u32 <= domain.max_particles {
                break;
            }
            let factor = ((count as f32) / (domain.max_particles as f32)).cbrt() * 1.02;
            h0 *= factor;
            if attempt == 0 {
                warnings.push(format!(
                    "Max Particles ({}) is lower than the requested fill: particles are made larger, so detail is reduced.",
                    domain.max_particles
                ));
            }
        }
        let mesh_cell = (cell / domain.mesh_scale.clamp(1, 4) as f32).max(0.5 * h0);
        let solver = FluidSolver::new(h0, bmin, bmax);
        let name = scene.objects[index].name.clone();
        let has_flow = scene.fluids.iter().any(|(id, f)| {
            matches!(f, FluidObject::Flow(fl) if fl.enabled && fl.behavior != FlowBehavior::Outflow) && scene.index_of(*id).is_some()
        });
        if !has_flow {
            warnings.push("No enabled Flow object (Geometry or Inflow) was found: the bake will be empty.".to_string());
        }
        Ok(Self {
            domain_id,
            domain_name: name,
            base: scene.clone(),
            domain,
            solver,
            bmin,
            bmax,
            mesh_cell,
            gravity,
            fps,
            start,
            end,
            frame: start,
            prev_scene: None,
            frames: Vec::new(),
            particle_counts: Vec::new(),
            warnings,
        })
    }

    /// Lattice cells for every enabled Geometry flow at the evaluated scene's frame.
    fn initial_cells(scene: &Scene, domain_id: u64, h0: f32, bmin: Vec3, bmax: Vec3) -> Vec<(i32, i32, i32)> {
        let worlds = scene.world_matrices();
        let mut cells: Vec<(i32, i32, i32)> = Vec::new();
        for (id, f) in scene.fluids.iter() {
            let FluidObject::Flow(flow) = f else { continue };
            if !flow.enabled || flow.behavior == FlowBehavior::Outflow || *id == domain_id {
                continue;
            }
            let Some(index) = scene.index_of(*id) else { continue };
            let tris = world_triangles(scene, index, &worlds[index]);
            let lo = bmin + Vec3::splat(0.5 * h0);
            let hi = bmax - Vec3::splat(0.5 * h0);
            cells.extend(fill_lattice(&tris, flow.is_planar, h0, lo, hi));
        }
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    pub fn domain_id(&self) -> u64 {
        self.domain_id
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn progress(&self) -> (i32, i32, i32) {
        (self.frame, self.start, self.end)
    }

    pub fn is_finished(&self) -> bool {
        self.frame > self.end
    }

    pub fn last_particle_count(&self) -> usize {
        self.solver.particle_count()
    }

    /// Simulate frames until `budget` is used up. Returns true when the bake is complete.
    pub fn step(&mut self, budget: Duration) -> bool {
        let started = Instant::now();
        while self.frame <= self.end {
            self.bake_frame();
            self.frame += 1;
            if started.elapsed() >= budget {
                break;
            }
        }
        self.frame > self.end
    }

    fn bake_frame(&mut self) {
        let frame = self.frame;
        let scene = scene_at_frame(&self.base, frame);
        let worlds = scene.world_matrices();
        let h0 = self.solver.h0;
        let lo = self.bmin + Vec3::splat(0.5 * h0);
        let hi = self.bmax - Vec3::splat(0.5 * h0);
        let budget = self.domain.max_particles as usize;

        // Flows.
        let mut outflow_cells: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut occupied: Option<HashSet<(i32, i32, i32)>> = None;
        let flow_ids: Vec<u64> = scene.fluids.keys().copied().collect();
        for id in flow_ids {
            if id == self.domain_id {
                continue;
            }
            let Some(FluidObject::Flow(flow)) = scene.fluids.get(&id).cloned() else { continue };
            if !flow.enabled {
                continue;
            }
            let Some(index) = scene.index_of(id) else { continue };
            let tris = world_triangles(&scene, index, &worlds[index]);
            let velocity = if flow.use_initial_velocity { flow.initial_velocity } else { Vec3::ZERO };
            match flow.behavior {
                FlowBehavior::Outflow => {
                    for c in fill_lattice(&tris, flow.is_planar, h0, lo - Vec3::splat(h0), hi + Vec3::splat(h0)) {
                        outflow_cells.insert(c);
                    }
                }
                FlowBehavior::Geometry => {
                    if frame == self.start {
                        for (ix, iy, iz) in fill_lattice(&tris, flow.is_planar, h0, lo, hi) {
                            if self.solver.particle_count() >= budget {
                                break;
                            }
                            self.solver.add_particle(lattice_position(ix, iy, iz, h0), velocity);
                        }
                    }
                }
                FlowBehavior::Inflow => {
                    if frame >= flow.frame_start && frame <= flow.frame_end {
                        let occ = occupied.get_or_insert_with(|| self.solver.pos.iter().map(|p| lattice_key(*p, h0)).collect());
                        for (ix, iy, iz) in fill_lattice(&tris, flow.is_planar, h0, lo, hi) {
                            if self.solver.particle_count() >= budget {
                                break;
                            }
                            if occ.insert((ix, iy, iz)) {
                                self.solver.add_particle(lattice_position(ix, iy, iz, h0), velocity);
                            }
                        }
                    }
                }
            }
        }
        if !outflow_cells.is_empty() {
            self.solver.remove_where(|p| outflow_cells.contains(&lattice_key(p, h0)));
        }

        // Simulate (the first frame only shows the initial state).
        if frame > self.start {
            if let Some(prev) = &self.prev_scene {
                let prev_worlds = prev.world_matrices();
                let mut effectors: Vec<EffectorGeo> = Vec::new();
                for (id, f) in scene.fluids.iter() {
                    let FluidObject::Effector(eff) = f else { continue };
                    if !eff.enabled {
                        continue;
                    }
                    let (Some(index), Some(prev_index)) = (scene.index_of(*id), prev.index_of(*id)) else { continue };
                    let now = world_triangles(&scene, index, &worlds[index]);
                    let before = world_triangles(prev, prev_index, &prev_worlds[prev_index]);
                    effectors.push(EffectorGeo::new(before, now, eff.surface_thickness.max(0.0), 1.5 * h0));
                }
                let dt = self.domain.time_scale.max(0.001) / self.fps;
                self.solver.advance_frame(dt, self.gravity, &self.domain, &effectors);
            }
        }

        let mesh = if self.domain.use_mesh {
            mesh_from_particles(
                &self.solver.pos,
                self.bmin,
                self.bmax,
                h0,
                self.mesh_cell,
                self.domain.mesh_smoothing,
                &format!("{} Fluid Mesh", self.domain_name),
            )
        } else {
            particles_as_mesh(&self.solver.pos, 0.35 * h0, &format!("{} Particles", self.domain_name))
        };
        self.frames.push(Arc::new(mesh));
        self.particle_counts.push(self.solver.particle_count() as u32);
        self.prev_scene = Some(scene);
    }

    /// Install the finished bake into the scene. Returns the number of frames stored.
    pub fn finish(self, scene: &mut Scene) -> usize {
        let count = self.frames.len();
        if count == 0 {
            return 0;
        }
        let existing = self
            .domain
            .mesh_object
            .filter(|id| scene.index_of(*id).is_some());
        let mesh_id = match existing {
            Some(id) => id,
            None => {
                let material = water_material_index(scene);
                let mut object = Object::new(format!("{} Fluid Mesh", self.domain_name), MeshData::default());
                object.material = Some(material);
                let id = scene.alloc_id();
                object.id = id;
                scene.objects.push(object);
                id
            }
        };
        if let Some(FluidObject::Domain(d)) = scene.fluids.get_mut(&self.domain_id) {
            d.mesh_object = Some(mesh_id);
        }
        let first = self.frames[0].clone();
        scene.mesh_caches.insert(
            mesh_id,
            MeshSequence::new(format!("Fluid bake of {}", self.domain_name), self.frames, self.start as f32, self.fps),
        );
        if let Some(index) = scene.index_of(mesh_id) {
            scene.objects[index].mesh = first;
            scene.objects[index].primitive = None;
        }
        count
    }
}

fn water_material_index(scene: &mut Scene) -> usize {
    if let Some(i) = scene.materials.iter().position(|m| m.name == "Water") {
        return i;
    }
    let mut m = Material::default();
    m.name = "Water".into();
    m.base_color = [0.10, 0.34, 0.62, 0.6];
    m.metallic = 0.0;
    m.roughness = 0.04;
    m.alpha_mode = AlphaMode::Blend;
    m.double_sided = true;
    scene.materials.push(m);
    scene.materials.len() - 1
}

/// Semi-transparent material used for the Domain box so the simulation volume
/// is visible without occluding the water (Blender-style domain display).
pub fn domain_material_index(scene: &mut Scene) -> usize {
    if let Some(i) = scene.materials.iter().position(|m| m.name == "Fluid Domain") {
        return i;
    }
    let mut m = Material::default();
    m.name = "Fluid Domain".into();
    // Light cyan, mostly see-through so the domain bounds stay readable.
    m.base_color = [0.35, 0.75, 0.95, 0.18];
    m.metallic = 0.0;
    m.roughness = 0.9;
    m.alpha_mode = AlphaMode::Blend;
    m.double_sided = true;
    scene.materials.push(m);
    scene.materials.len() - 1
}

/// Apply the transparent domain look to an object that was just set as Domain.
/// Call this when the user assigns the Domain role so the box no longer looks solid.
pub fn apply_domain_display(scene: &mut Scene, object_id: u64) {
    let mat = domain_material_index(scene);
    if let Some(index) = scene.index_of(object_id) {
        scene.objects[index].material = Some(mat);
    }
}

/// Debug display: every particle as a small octahedron.
fn particles_as_mesh(points: &[Vec3], radius: f32, name: &str) -> MeshData {
    let mut mesh = MeshData { name: name.to_string(), ..Default::default() };
    let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let faces: [[usize; 3]; 8] = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
    for p in points {
        let base = mesh.vertices.len() as u32;
        for d in dirs.iter() {
            mesh.vertices.push(Vertex { position: (*p + *d * radius).to_array(), normal: d.to_array(), uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        });
        }
        for f in faces.iter() {
            mesh.indices.extend_from_slice(&[base + f[0] as u32, base + f[1] as u32, base + f[2] as u32]);
        }
    }
    mesh
}

/// Remove a domain's baked surface (the Fluid Mesh object stays, empty).
pub fn free_fluid_bake(scene: &mut Scene, domain_id: u64) -> bool {
    let Some(FluidObject::Domain(d)) = scene.fluids.get(&domain_id) else { return false };
    let Some(mesh_id) = d.mesh_object else { return false };
    let removed = scene.mesh_caches.remove(&mesh_id).is_some();
    if let Some(index) = scene.index_of(mesh_id) {
        scene.objects[index].mesh = Arc::new(MeshData::default());
    }
    removed
}

/// Number of frames in a domain's bake (0 when none).
pub fn fluid_bake_frames(scene: &Scene, domain_id: u64) -> usize {
    match scene.fluids.get(&domain_id) {
        Some(FluidObject::Domain(d)) => d
            .mesh_object
            .and_then(|id| scene.mesh_caches.get(&id))
            .map_or(0, |seq| seq.len()),
        _ => 0,
    }
}

/// Convert the old placeholder components (settings only) into real fluid domains.
pub fn migrate_legacy_fluids(scene: &mut Scene) {
    let legacy: Vec<(u64, FluidSettings)> = scene
        .object_simulations
        .iter()
        .filter_map(|(id, sim)| match sim {
            crate::simulation::ObjectSimulation::Fluid(f) => Some((*id, f.clone())),
            _ => None,
        })
        .collect();
    for (id, old) in legacy {
        scene.object_simulations.remove(&id);
        scene.fluids.entry(id).or_insert_with(|| FluidObject::Domain(FluidDomain::from_legacy(&old)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    fn box_object(name: &str, min: Vec3, max: Vec3) -> Object {
        let mut m = cube(1.0);
        let b = m.bounds().unwrap();
        let src_center = b.center();
        let src_size = b.max - b.min;
        for v in m.vertices.iter_mut() {
            let p = (Vec3::from(v.position) - src_center) / src_size;
            v.position = (min + (p + Vec3::splat(0.5)) * (max - min)).to_array();
        }
        Object::new(name, m)
    }

    #[test]
    fn water_block_falls_and_stays_inside_the_domain() {
        let mut scene = Scene::default();
        let mut domain = box_object("Domain", Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
        domain.id = scene.alloc_id();
        let domain_id = domain.id;
        scene.objects.push(domain);
        let mut flow = box_object("Water", Vec3::new(-0.5, 0.8, -0.5), Vec3::new(0.5, 1.6, 0.5));
        flow.id = scene.alloc_id();
        let flow_id = flow.id;
        scene.objects.push(flow);
        let mut d = FluidDomain::default();
        d.resolution = 12;
        d.cache_frame_start = 1;
        d.cache_frame_end = 6;
        scene.fluids.insert(domain_id, FluidObject::Domain(d));
        scene.fluids.insert(flow_id, FluidObject::Flow(FluidFlow::default()));

        let mut baker = FluidBaker::new(&scene, domain_id).expect("baker");
        while !baker.step(Duration::from_millis(500)) {}
        assert!(baker.last_particle_count() > 100, "block should be filled with particles");
        for p in baker.solver.pos.iter() {
            assert!(p.x >= -1.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 2.0 && p.z >= -1.0 && p.z <= 1.0);
            assert!(p.is_finite());
        }
        let first_y: f32 = baker.solver.pos.iter().map(|p| p.y).sum::<f32>() / baker.solver.pos.len() as f32;
        assert!(first_y < 1.2, "water should have fallen below its start height, mean y = {first_y}");
        let frames = baker.finish(&mut scene);
        assert_eq!(frames, 6);
        assert_eq!(fluid_bake_frames(&scene, domain_id), 6);
        let mesh_id = match scene.fluids.get(&domain_id) {
            Some(FluidObject::Domain(d)) => d.mesh_object.unwrap(),
            _ => panic!("domain lost"),
        };
        let seq = scene.mesh_caches.get(&mesh_id).unwrap();
        assert!(!seq.frames[0].indices.is_empty(), "first frame should have a surface");
    }

    #[test]
    fn lattice_fill_of_a_box_matches_its_volume() {
        let obj = box_object("Box", Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let tris = world_triangles(
            &Scene { objects: vec![obj], ..Default::default() },
            0,
            &Mat4::IDENTITY,
        );
        let cells = fill_lattice(&tris, false, 0.1, Vec3::splat(-1.0), Vec3::splat(2.0));
        assert!((cells.len() as i32 - 1000).abs() < 150, "got {}", cells.len());
    }

    #[test]
    fn legacy_fluid_components_become_domains() {
        let mut scene = Scene::default();
        scene.object_simulations.insert(
            7,
            crate::simulation::ObjectSimulation::Fluid(FluidSettings { enabled: true, ..Default::default() }),
        );
        migrate_legacy_fluids(&mut scene);
        assert!(scene.object_simulations.is_empty());
        assert!(matches!(scene.fluids.get(&7), Some(FluidObject::Domain(_))));
    }
}
