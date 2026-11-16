//! Cloth simulation.
//!
//! * **Fabric presets** (silk, cotton, linen, wool, denim, leather, rubber) that set every
//!   physical parameter at once; any value can still be edited afterwards (-> "Custom").
//! * **XPBD solver with sub-stepping** ("small steps"): stretch/compression edge constraints
//!   with physical stiffness (N/m), dihedral-angle bending constraints driven by a real bending
//!   rigidity (mN*m), aerodynamic drag and a hard strain limit.
//! * **Collisions** with every other visible mesh in the scene (triangle meshes, spatial hash,
//!   friction), optional **self collision**, and reaction to **Wind / Force / Vortex /
//!   Turbulence** force-field objects.
//! * **Pinning** (none / boundary / top edge / top corners) with a configurable pin depth, which
//!   clamps rows of vertices so stiff fabrics can stick out of their pins.
//! * **Adaptive refinement**: low-resolution meshes (like the default 4-vertex plane) are
//!   subdivided automatically to a target edge length, so cloth drapes without manual subdividing.
//!
//! Stiffness values are resolution independent: the stretch stiffness is a membrane stiffness in
//! N/m and the hinge stiffness is derived from the bending rigidity `B` as
//! `k = 3 * B * e^2 / (A_a + A_b)`.
use crate::kinds::{ForceFieldType, ObjectKind};
use crate::mesh::{MeshData, MeshGroup, Vertex};
use crate::meshcache::MeshSequence;
use crate::scene::Scene;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Hard cap on simulated vertices (keeps bakes responsive).
const MAX_CLOTH_VERTICES: usize = 40_000;
/// Hard cap on collider triangles gathered from the scene.
const MAX_COLLIDER_TRIS: usize = 600_000;

// ---------------------------------------------------------------------------
// Fabric presets
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FabricType {
    Silk,
    #[default]
    Cotton,
    Linen,
    Wool,
    Denim,
    Leather,
    Rubber,
    Custom,
}

/// Physical parameters a fabric sets.
#[derive(Clone, Copy, Debug)]
pub struct FabricPreset {
    /// Areal density in kg/m^2.
    pub mass: f32,
    /// Membrane stretch stiffness in N/m.
    pub tension: f32,
    /// Fraction of the tension stiffness that resists compression (low = wrinkles easily).
    pub compression: f32,
    /// Bending rigidity in mN*m.
    pub bending: f32,
    pub damping: f32,
    pub air: f32,
    pub friction: f32,
    /// Maximum strain (0.05 = 5 %) before the hard limit engages.
    pub max_stretch: f32,
    /// Collision thickness in metres.
    pub thickness: f32,
}

impl FabricType {
    pub const ALL: [FabricType; 8] = [
        FabricType::Silk,
        FabricType::Cotton,
        FabricType::Linen,
        FabricType::Wool,
        FabricType::Denim,
        FabricType::Leather,
        FabricType::Rubber,
        FabricType::Custom,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FabricType::Silk => "Silk",
            FabricType::Cotton => "Cotton",
            FabricType::Linen => "Linen",
            FabricType::Wool => "Wool",
            FabricType::Denim => "Denim",
            FabricType::Leather => "Leather",
            FabricType::Rubber => "Rubber",
            FabricType::Custom => "Custom",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            FabricType::Silk => "Very light and floppy; flutters in the slightest breeze and folds in tiny wrinkles.",
            FabricType::Cotton => "Everyday fabric: light, soft, medium friction.",
            FabricType::Linen => "Crisper than cotton; holds larger folds.",
            FabricType::Wool => "Heavy, thick and stretchy with high friction.",
            FabricType::Denim => "Heavy and stiff; forms wide, sturdy folds.",
            FabricType::Leather => "Very heavy and stiff; barely stretches and keeps its shape.",
            FabricType::Rubber => "Heavy, elastic and grippy; stretches a lot and springs back.",
            FabricType::Custom => "Hand-tuned values.",
        }
    }

    pub fn preset(self) -> Option<FabricPreset> {
        let p = |mass, tension, compression, bending, damping, friction, max_stretch, thickness| {
            Some(FabricPreset {
                mass,
                tension,
                compression,
                bending,
                damping,
                air: 1.0,
                friction,
                max_stretch,
                thickness,
            })
        };
        match self {
            FabricType::Silk => p(0.06, 600.0, 0.02, 0.05, 0.02, 0.15, 0.04, 0.008),
            FabricType::Cotton => p(0.20, 800.0, 0.05, 1.0, 0.03, 0.45, 0.05, 0.010),
            FabricType::Linen => p(0.25, 1500.0, 0.10, 5.0, 0.03, 0.50, 0.03, 0.010),
            FabricType::Wool => p(0.35, 400.0, 0.08, 3.0, 0.05, 0.60, 0.12, 0.012),
            FabricType::Denim => p(0.40, 2000.0, 0.12, 15.0, 0.04, 0.50, 0.02, 0.012),
            FabricType::Leather => p(0.80, 3000.0, 0.15, 200.0, 0.06, 0.60, 0.015, 0.015),
            FabricType::Rubber => p(1.20, 120.0, 0.50, 60.0, 0.08, 0.90, 0.60, 0.015),
            FabricType::Custom => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClothPin {
    #[default]
    None,
    /// Every vertex on an open boundary.
    Boundary,
    /// The highest row of vertices (curtains, banners).
    TopEdge,
    /// The two outermost vertices of the top row (hanging sheet, hammock).
    TopCorners,
}

impl ClothPin {
    pub const ALL: [ClothPin; 4] = [
        ClothPin::None,
        ClothPin::Boundary,
        ClothPin::TopEdge,
        ClothPin::TopCorners,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ClothPin::None => "None",
            ClothPin::Boundary => "Boundary",
            ClothPin::TopEdge => "Top Edge",
            ClothPin::TopCorners => "Top Corners",
        }
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClothSettings {
    pub enabled: bool,
    pub fabric: FabricType,
    /// Sub-steps per frame (more = stiffer, more accurate, slower).
    pub quality_steps: u32,
    /// Areal density, kg/m^2.
    pub mass: f32,
    /// Stretch stiffness, N/m.
    pub tension_stiffness: f32,
    /// Compression stiffness as a fraction of the tension stiffness.
    pub compression_ratio: f32,
    /// Bending rigidity, mN*m.
    pub bending_stiffness: f32,
    pub damping: f32,
    pub air_damping: f32,
    pub gravity_scale: f32,
    pub friction: f32,
    /// Hard strain limit (0 = off).
    pub max_stretch: f32,
    pub pin: ClothPin,
    /// Also pin this many rings of neighbours around the pinned vertices (clamping).
    pub pin_depth: u32,
    pub use_collision: bool,
    pub collision_distance: f32,
    pub use_self_collision: bool,
    pub self_distance: f32,
    /// React to Wind / Force / Vortex / Turbulence objects.
    pub use_force_fields: bool,
    /// Subdivide the mesh during simulation until edges are about this long (0 = off).
    pub target_edge_length: f32,
}

impl ClothSettings {
    pub fn with_fabric(fabric: FabricType) -> Self {
        let mut s = Self {
            enabled: false,
            fabric: FabricType::Cotton,
            quality_steps: 10,
            mass: 0.2,
            tension_stiffness: 800.0,
            compression_ratio: 0.05,
            bending_stiffness: 1.0,
            damping: 0.03,
            air_damping: 1.0,
            gravity_scale: 1.0,
            friction: 0.45,
            max_stretch: 0.05,
            pin: ClothPin::None,
            pin_depth: 1,
            use_collision: true,
            collision_distance: 0.01,
            use_self_collision: false,
            self_distance: 0.02,
            use_force_fields: true,
            target_edge_length: 0.04,
        };
        s.apply_fabric(fabric);
        s
    }

    /// Copy every physical value of a fabric preset (Custom keeps the current values).
    pub fn apply_fabric(&mut self, fabric: FabricType) {
        if let Some(p) = fabric.preset() {
            self.mass = p.mass;
            self.tension_stiffness = p.tension;
            self.compression_ratio = p.compression;
            self.bending_stiffness = p.bending;
            self.damping = p.damping;
            self.air_damping = p.air;
            self.friction = p.friction;
            self.max_stretch = p.max_stretch;
            self.collision_distance = p.thickness;
        }
        self.fabric = fabric;
    }
}

impl Default for ClothSettings {
    fn default() -> Self {
        Self::with_fabric(FabricType::Cotton)
    }
}

// ---------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Map every vertex to the first vertex at the same position (welds UV / hard-edge seams).
fn weld_map(mesh: &MeshData) -> Vec<u32> {
    let q = 1.0e4_f32;
    let mut seen: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(mesh.vertices.len());
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let key = (
                (v.position[0] * q).round() as i64,
                (v.position[1] * q).round() as i64,
                (v.position[2] * q).round() as i64,
            );
            *seen.entry(key).or_insert(i as u32)
        })
        .collect()
}

fn mean_edge_length(mesh: &MeshData) -> (f32, usize) {
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for t in mesh.indices.chunks_exact(3) {
        edges.insert(edge_key(t[0], t[1]));
        edges.insert(edge_key(t[1], t[2]));
        edges.insert(edge_key(t[2], t[0]));
    }
    if edges.is_empty() {
        return (0.0, 0);
    }
    let mut sum = 0.0_f32;
    for &(a, b) in &edges {
        let pa = mesh.vertices.get(a as usize).map(|v| Vec3::from(v.position));
        let pb = mesh.vertices.get(b as usize).map(|v| Vec3::from(v.position));
        if let (Some(pa), Some(pb)) = (pa, pb) {
            sum += (pa - pb).length();
        }
    }
    (sum / edges.len() as f32, edges.len())
}

/// Split every triangle into four (midpoint subdivision), keeping UVs and material groups.
fn subdivide_once(mesh: &MeshData) -> MeshData {
    let mut vertices = mesh.vertices.clone();
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut indices: Vec<u32> = Vec::with_capacity(mesh.indices.len() * 4);
    let mut midpoint = |vertices: &mut Vec<Vertex>, a: u32, b: u32| -> u32 {
        let key = edge_key(a, b);
        if let Some(&m) = mid.get(&key) {
            return m;
        }
        let va = vertices[a as usize];
        let vb = vertices[b as usize];
        let n = (Vec3::from(va.normal) + Vec3::from(vb.normal)).normalize_or_zero();
        let v = Vertex {
            position: ((Vec3::from(va.position) + Vec3::from(vb.position)) * 0.5).to_array(),
            normal: n.to_array(),
            uv: [(va.uv[0] + vb.uv[0]) * 0.5, (va.uv[1] + vb.uv[1]) * 0.5],
        };
        let index = vertices.len() as u32;
        vertices.push(v);
        mid.insert(key, index);
        index
    };
    for t in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (t[0], t[1], t[2]);
        let ab = midpoint(&mut vertices, a, b);
        let bc = midpoint(&mut vertices, b, c);
        let ca = midpoint(&mut vertices, c, a);
        indices.extend_from_slice(&[a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]);
    }
    let groups: Vec<MeshGroup> = mesh
        .groups
        .iter()
        .map(|g| MeshGroup {
            first_index: g.first_index * 4,
            index_count: g.index_count * 4,
            material: g.material,
        })
        .collect();
    MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups,
    }
}

/// Subdivide until the mean edge length reaches `target` (or the vertex cap is hit).
fn refine_for_cloth(mut mesh: MeshData, target: f32) -> MeshData {
    if target <= 0.0 {
        return mesh;
    }
    for _ in 0..6 {
        let (mean, edges) = mean_edge_length(&mesh);
        if edges == 0 || mean <= target * 1.25 {
            break;
        }
        if mesh.vertices.len() + edges > MAX_CLOTH_VERTICES {
            break;
        }
        mesh = subdivide_once(&mesh);
    }
    mesh
}

// ---------------------------------------------------------------------------
// Colliders
// ---------------------------------------------------------------------------

struct CollTri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    n: Vec3,
}

struct Colliders {
    tris: Vec<CollTri>,
    cell: f32,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    /// Triangles too large for the grid (floors, walls) are tested for every particle.
    big: Vec<u32>,
}

const NO_TRIS: &[u32] = &[];

fn cell_of(p: Vec3, cell: f32) -> (i32, i32, i32) {
    (
        (p.x / cell).floor() as i32,
        (p.y / cell).floor() as i32,
        (p.z / cell).floor() as i32,
    )
}

impl Colliders {
    fn new(tris: Vec<CollTri>, margin: f32) -> Self {
        let mut extent_sum = 0.0_f32;
        for t in &tris {
            let mn = t.a.min(t.b).min(t.c);
            let mx = t.a.max(t.b).max(t.c);
            extent_sum += (mx - mn).max_element();
        }
        let avg = if tris.is_empty() {
            1.0
        } else {
            extent_sum / tris.len() as f32
        };
        let cell = avg.clamp(0.05, 2.0).max(margin * 2.0);
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        let mut big = Vec::new();
        for (i, t) in tris.iter().enumerate() {
            let mn = t.a.min(t.b).min(t.c) - Vec3::splat(margin);
            let mx = t.a.max(t.b).max(t.c) + Vec3::splat(margin);
            let lo = cell_of(mn, cell);
            let hi = cell_of(mx, cell);
            let count = (hi.0 - lo.0 + 1).max(1) as i64
                * (hi.1 - lo.1 + 1).max(1) as i64
                * (hi.2 - lo.2 + 1).max(1) as i64;
            if count > 512 {
                big.push(i as u32);
                continue;
            }
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        grid.entry((x, y, z)).or_default().push(i as u32);
                    }
                }
            }
        }
        Self {
            tris,
            cell,
            grid,
            big,
        }
    }

    fn candidates(&self, p: Vec3) -> (&[u32], &[u32]) {
        let key = cell_of(p, self.cell);
        let cells = self.grid.get(&key).map(|v| v.as_slice()).unwrap_or(NO_TRIS);
        (cells, self.big.as_slice())
    }
}

/// Closest point on triangle `abc` to `p` (Ericson, Real-Time Collision Detection).
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

fn collect_colliders(
    scene: &Scene,
    worlds: &[Mat4],
    skip_id: u64,
    distance: f32,
) -> Option<Colliders> {
    let mut tris: Vec<CollTri> = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if o.id == skip_id || !o.kind.is_mesh() || !scene.is_visible(i) {
            continue;
        }
        if scene.fluids.contains_key(&o.id) {
            continue;
        }
        if scene
            .object_simulations
            .get(&o.id)
            .map_or(false, |s| s.enabled())
        {
            continue;
        }
        if scene
            .mesh_caches
            .get(&o.id)
            .map_or(false, |s| s.source.to_lowercase().contains("fluid"))
        {
            continue;
        }
        let m = worlds.get(i).copied().unwrap_or(Mat4::IDENTITY);
        let verts = &o.mesh.vertices;
        for t in o.mesh.indices.chunks_exact(3) {
            if tris.len() >= MAX_COLLIDER_TRIS {
                break;
            }
            if t.iter().any(|&k| k as usize >= verts.len()) {
                continue;
            }
            let a = m.transform_point3(Vec3::from(verts[t[0] as usize].position));
            let b = m.transform_point3(Vec3::from(verts[t[1] as usize].position));
            let c = m.transform_point3(Vec3::from(verts[t[2] as usize].position));
            let n = (b - a).cross(c - a).normalize_or_zero();
            if n == Vec3::ZERO {
                continue;
            }
            tris.push(CollTri { a, b, c, n });
        }
    }
    if tris.is_empty() {
        None
    } else {
        Some(Colliders::new(tris, distance + 0.03))
    }
}

// ---------------------------------------------------------------------------
// Force fields
// ---------------------------------------------------------------------------

struct Field {
    kind: ForceFieldType,
    pos: Vec3,
    axis: Vec3,
    strength: f32,
    size: f32,
}

fn collect_fields(scene: &Scene, worlds: &[Mat4]) -> Vec<Field> {
    let mut out = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_visible(i) {
            continue;
        }
        if let ObjectKind::ForceField(f) = &o.kind {
            let m = worlds.get(i).copied().unwrap_or(Mat4::IDENTITY);
            let axis = m.transform_vector3(Vec3::Z).normalize_or_zero();
            out.push(Field {
                kind: f.kind,
                pos: m.w_axis.truncate(),
                axis: if axis == Vec3::ZERO { Vec3::Z } else { axis },
                strength: f.strength,
                size: f.size.max(0.01),
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

struct Stretch {
    a: u32,
    b: u32,
    rest: f32,
}

struct Hinge {
    v: [u32; 4],
    theta0: f32,
    /// Angular stiffness in N*m/rad.
    k: f32,
}

fn hinge_angle(x1: Vec3, x2: Vec3, x3: Vec3, x4: Vec3) -> f32 {
    let e = x2 - x1;
    let eh = e.normalize_or_zero();
    let na = e.cross(x3 - x1);
    let nb = (x4 - x2).cross(e);
    let la = na.length();
    let lb = nb.length();
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    let s = na.cross(nb).dot(eh) / (la * lb);
    let c = na.dot(nb) / (la * lb);
    s.atan2(c)
}

pub(crate) struct ClothSim {
    pos: Vec<Vec3>,
    prev: Vec<Vec3>,
    vel: Vec<Vec3>,
    rest: Vec<Vec3>,
    normals: Vec<Vec3>,
    inv_mass: Vec<f32>,
    /// Vertex takes part in the simulation (is a welded representative used by a triangle).
    active: Vec<bool>,
    rep: Vec<u32>,
    tris: Vec<[u32; 3]>,
    stretch: Vec<Stretch>,
    hinges: Vec<Hinge>,
    s: ClothSettings,
    template: MeshData,
    colliders: Option<Colliders>,
    fields: Vec<Field>,
    time: f32,
}

impl ClothSim {
    fn new(
        world_mesh: MeshData,
        s: &ClothSettings,
        colliders: Option<Colliders>,
        fields: Vec<Field>,
    ) -> Self {
        let n = world_mesh.vertices.len();
        let rep = weld_map(&world_mesh);
        let pos: Vec<Vec3> = world_mesh
            .vertices
            .iter()
            .map(|v| Vec3::from(v.position))
            .collect();
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(world_mesh.indices.len() / 3);
        for t in world_mesh.indices.chunks_exact(3) {
            if t.iter().any(|&k| k as usize >= n) {
                continue;
            }
            let (a, b, c) = (rep[t[0] as usize], rep[t[1] as usize], rep[t[2] as usize]);
            if a == b || b == c || c == a {
                continue;
            }
            tris.push([a, b, c]);
        }

        // Vertex areas.
        let mut area = vec![0.0_f32; n];
        let mut active = vec![false; n];
        for t in &tris {
            let a = pos[t[0] as usize];
            let b = pos[t[1] as usize];
            let c = pos[t[2] as usize];
            let tri_area = 0.5 * (b - a).cross(c - a).length();
            for &k in t {
                area[k as usize] += tri_area / 3.0;
                active[k as usize] = true;
            }
        }

        // Stretch constraints.
        let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
        for t in &tris {
            edge_set.insert(edge_key(t[0], t[1]));
            edge_set.insert(edge_key(t[1], t[2]));
            edge_set.insert(edge_key(t[2], t[0]));
        }
        let mut edge_list: Vec<(u32, u32)> = edge_set.into_iter().collect();
        edge_list.sort_unstable();
        let stretch: Vec<Stretch> = edge_list
            .iter()
            .map(|&(a, b)| Stretch {
                a,
                b,
                rest: (pos[a as usize] - pos[b as usize]).length().max(1e-6),
            })
            .collect();

        // Bending hinges: edges shared by exactly two triangles.
        let mut edge_tris: HashMap<(u32, u32), Vec<(usize, u32, u32)>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                edge_tris.entry(edge_key(a, b)).or_default().push((ti, a, b));
            }
        }
        let bending_nm = (s.bending_stiffness * 1.0e-3).max(1.0e-9);
        let mut hinges: Vec<Hinge> = Vec::new();
        let mut hinge_keys: Vec<(u32, u32)> = edge_tris.keys().copied().collect();
        hinge_keys.sort_unstable();
        for key in hinge_keys {
            let list = &edge_tris[&key];
            if list.len() != 2 {
                continue;
            }
            let (t0, x1, x2) = list[0];
            let (t1, _, _) = list[1];
            let apex = |t: &[u32; 3]| -> Option<u32> {
                t.iter().copied().find(|&v| v != key.0 && v != key.1)
            };
            let (Some(x3), Some(x4)) = (apex(&tris[t0]), apex(&tris[t1])) else {
                continue;
            };
            let tri_area = |t: &[u32; 3]| -> f32 {
                let a = pos[t[0] as usize];
                let b = pos[t[1] as usize];
                let c = pos[t[2] as usize];
                0.5 * (b - a).cross(c - a).length()
            };
            let a_sum = tri_area(&tris[t0]) + tri_area(&tris[t1]);
            let e_len = (pos[x2 as usize] - pos[x1 as usize]).length();
            if a_sum < 1e-12 || e_len < 1e-9 {
                continue;
            }
            let theta0 = hinge_angle(
                pos[x1 as usize],
                pos[x2 as usize],
                pos[x3 as usize],
                pos[x4 as usize],
            );
            hinges.push(Hinge {
                v: [x1, x2, x3, x4],
                theta0,
                k: 3.0 * bending_nm * e_len * e_len / a_sum,
            });
        }

        // Pinning.
        let pinned = pick_pins(&pos, &tris, &stretch, &active, s);

        // Masses.
        let rho = s.mass.max(1.0e-3);
        let inv_mass: Vec<f32> = (0..n)
            .map(|i| {
                if !active[i] || pinned[i] {
                    0.0
                } else {
                    1.0 / (rho * area[i]).max(1.0e-9)
                }
            })
            .collect();

        Self {
            prev: pos.clone(),
            rest: pos.clone(),
            vel: vec![Vec3::ZERO; n],
            normals: vec![Vec3::Y; n],
            pos,
            inv_mass,
            active,
            rep,
            tris,
            stretch,
            hinges,
            s: s.clone(),
            template: world_mesh,
            colliders,
            fields,
            time: 0.0,
        }
    }

    fn compute_normals(&mut self) {
        for n in &mut self.normals {
            *n = Vec3::ZERO;
        }
        for t in &self.tris {
            let a = self.pos[t[0] as usize];
            let b = self.pos[t[1] as usize];
            let c = self.pos[t[2] as usize];
            let n = (b - a).cross(c - a);
            for &k in t {
                self.normals[k as usize] += n;
            }
        }
        for n in &mut self.normals {
            *n = n.normalize_or_zero();
        }
    }

    /// Wind velocity (m/s) and extra acceleration (m/s^2) at `p`.
    fn field_at(&self, p: Vec3) -> (Vec3, Vec3) {
        let mut wind = Vec3::ZERO;
        let mut acc = Vec3::ZERO;
        for f in &self.fields {
            match f.kind {
                ForceFieldType::Wind => {
                    let phase = self.time * 1.7 + p.dot(Vec3::new(0.8, 0.3, 0.6)) * 2.0;
                    wind += f.axis * (f.strength * (1.0 + 0.25 * phase.sin()));
                }
                ForceFieldType::Force => {
                    let d = f.pos - p;
                    let dist = d.length().max(1.0e-3);
                    let fall = 1.0 / (1.0 + (dist / f.size).powi(2));
                    acc += d / dist * (f.strength * fall);
                }
                ForceFieldType::Vortex => {
                    let r = p - f.pos;
                    let t = f.axis.cross(r);
                    let l = t.length();
                    if l > 1.0e-6 {
                        let fall = 1.0 / (1.0 + (l / f.size).powi(2));
                        acc += t / l * (f.strength * fall);
                    }
                }
                ForceFieldType::Turbulence => {
                    let q = p * 2.0 + Vec3::splat(self.time * 1.3);
                    acc += Vec3::new(
                        q.x.sin() * q.y.cos(),
                        q.y.sin() * q.z.cos(),
                        q.z.sin() * q.x.cos(),
                    ) * f.strength;
                }
                ForceFieldType::Magnetic => {}
            }
        }
        (wind, acc)
    }

    fn solve_stretch(&mut self, h: f32) {
        let h2 = h * h;
        let k_t = self.s.tension_stiffness.max(1.0e-3);
        let k_c = (k_t * self.s.compression_ratio.clamp(0.001, 1.0)).max(1.0e-4);
        for i in 0..self.stretch.len() {
            let (a, b, rest) = {
                let c = &self.stretch[i];
                (c.a as usize, c.b as usize, c.rest)
            };
            let d = self.pos[b] - self.pos[a];
            let l = d.length();
            if l < 1.0e-9 {
                continue;
            }
            let wa = self.inv_mass[a];
            let wb = self.inv_mass[b];
            let ws = wa + wb;
            if ws == 0.0 {
                continue;
            }
            let k = if l >= rest { k_t } else { k_c };
            let alpha = 1.0 / (k * h2);
            let dl = -(l - rest) / (ws + alpha);
            let u = d / l;
            self.pos[a] -= u * (wa * dl);
            self.pos[b] += u * (wb * dl);
        }
    }

    fn solve_bending(&mut self, h: f32) {
        let h2 = h * h;
        for i in 0..self.hinges.len() {
            let (v, theta0, k) = {
                let hg = &self.hinges[i];
                (hg.v, hg.theta0, hg.k)
            };
            let [i1, i2, i3, i4] = [v[0] as usize, v[1] as usize, v[2] as usize, v[3] as usize];
            let x1 = self.pos[i1];
            let x2 = self.pos[i2];
            let x3 = self.pos[i3];
            let x4 = self.pos[i4];
            let e = x2 - x1;
            let el = e.length();
            if el < 1.0e-9 {
                continue;
            }
            let eh = e / el;
            let na = e.cross(x3 - x1);
            let nb = (x4 - x2).cross(e);
            let la2 = na.length_squared();
            let lb2 = nb.length_squared();
            if la2 < 1.0e-16 || lb2 < 1.0e-16 {
                continue;
            }
            let den = (la2 * lb2).sqrt();
            let theta = (na.cross(nb).dot(eh) / den).atan2(na.dot(nb) / den);
            let mut c = theta - theta0;
            if c > std::f32::consts::PI {
                c -= 2.0 * std::f32::consts::PI;
            }
            if c < -std::f32::consts::PI {
                c += 2.0 * std::f32::consts::PI;
            }
            let g3 = na * (-el / la2);
            let g4 = nb * (-el / lb2);
            let a1 = (x3 - x2).dot(eh) / el;
            let a2 = (x3 - x1).dot(eh) / el;
            let b1 = (x4 - x2).dot(eh) / el;
            let b2 = (x4 - x1).dot(eh) / el;
            let g1 = g3 * a1 + g4 * b1;
            let g2 = g3 * (-a2) + g4 * (-b2);
            let (w1, w2, w3, w4) = (
                self.inv_mass[i1],
                self.inv_mass[i2],
                self.inv_mass[i3],
                self.inv_mass[i4],
            );
            let ws = w1 * g1.length_squared()
                + w2 * g2.length_squared()
                + w3 * g3.length_squared()
                + w4 * g4.length_squared();
            let alpha = 1.0 / (k.max(1.0e-9) * h2);
            let dl = -c / (ws + alpha);
            self.pos[i1] += g1 * (w1 * dl);
            self.pos[i2] += g2 * (w2 * dl);
            self.pos[i3] += g3 * (w3 * dl);
            self.pos[i4] += g4 * (w4 * dl);
        }
    }

    fn limit_strain(&mut self) {
        let lim = self.s.max_stretch;
        if lim <= 0.0 {
            return;
        }
        for i in 0..self.stretch.len() {
            let (a, b, rest) = {
                let c = &self.stretch[i];
                (c.a as usize, c.b as usize, c.rest)
            };
            let d = self.pos[b] - self.pos[a];
            let l = d.length();
            let mx = rest * (1.0 + lim);
            if l > mx && l > 1.0e-9 {
                let ws = self.inv_mass[a] + self.inv_mass[b];
                if ws == 0.0 {
                    continue;
                }
                let u = d / l;
                let c = (l - mx) / ws;
                self.pos[a] += u * (self.inv_mass[a] * c);
                self.pos[b] -= u * (self.inv_mass[b] * c);
            }
        }
    }

    fn collide(&mut self) {
        let Some(col) = self.colliders.as_ref() else {
            return;
        };
        let r = self.s.collision_distance.max(1.0e-4);
        let mu = self.s.friction.clamp(0.0, 1.0);
        for i in 0..self.pos.len() {
            if self.inv_mass[i] == 0.0 {
                continue;
            }
            let prev = self.prev[i];
            let mut p = self.pos[i];
            let (cells, big) = col.candidates(p);
            let mut contact: Option<Vec3> = None;
            for &ti in cells.iter().chain(big.iter()) {
                let t = &col.tris[ti as usize];
                let q = closest_point_triangle(p, t.a, t.b, t.c);
                let d = p - q;
                let dist = d.length();
                if dist >= r {
                    continue;
                }
                let side = if (prev - t.a).dot(t.n) >= 0.0 { 1.0 } else { -1.0 };
                let nn = if dist > 1.0e-7 && d.dot(t.n) * side >= 0.0 {
                    d / dist
                } else {
                    t.n * side
                };
                p = q + nn * r;
                contact = Some(nn);
            }
            if let Some(n) = contact {
                let dx = p - prev;
                let tang = dx - n * dx.dot(n);
                p -= tang * mu;
                self.pos[i] = p;
            }
        }
    }

    fn self_collide(&mut self) {
        let d = self.s.self_distance.max(1.0e-4);
        let n = self.pos.len();
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::with_capacity(n);
        for i in 0..n {
            if self.active[i] {
                grid.entry(cell_of(self.pos[i], d)).or_default().push(i as u32);
            }
        }
        let skip2 = 4.0 * d * d;
        for i in 0..n {
            if !self.active[i] {
                continue;
            }
            let c = cell_of(self.pos[i], d);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) else {
                            continue;
                        };
                        for &j in list {
                            let j = j as usize;
                            if j <= i {
                                continue;
                            }
                            let wi = self.inv_mass[i];
                            let wj = self.inv_mass[j];
                            let ws = wi + wj;
                            if ws == 0.0 {
                                continue;
                            }
                            if (self.rest[i] - self.rest[j]).length_squared() < skip2 {
                                continue;
                            }
                            let delta = self.pos[j] - self.pos[i];
                            let dist = delta.length();
                            if dist >= d || dist < 1.0e-7 {
                                continue;
                            }
                            let corr = delta / dist * ((d - dist) / ws);
                            self.pos[i] -= corr * wi;
                            self.pos[j] += corr * wj;
                        }
                    }
                }
            }
        }
    }

    fn substep(&mut self, h: f32, gravity: Vec3) {
        let n = self.pos.len();
        self.prev.copy_from_slice(&self.pos);
        self.compute_normals();
        let rho = self.s.mass.max(1.0e-3);
        let air = self.s.air_damping.max(0.0);
        let use_fields = !self.fields.is_empty();
        for i in 0..n {
            if self.inv_mass[i] == 0.0 {
                continue;
            }
            let p = self.pos[i];
            let (wind, acc) = if use_fields {
                self.field_at(p)
            } else {
                (Vec3::ZERO, Vec3::ZERO)
            };
            let nrm = self.normals[i];
            let vn = (wind - self.vel[i]).dot(nrm);
            // Quadratic aerodynamic drag on the normal component, integrated implicitly.
            let kd = 0.6 * air * vn.abs() / rho;
            let dvn = vn * (h * kd) / (1.0 + h * kd);
            let mut v = self.vel[i] + (gravity + acc) * h + nrm * dvn;
            let speed = v.length();
            if speed > 60.0 {
                v *= 60.0 / speed;
            }
            self.vel[i] = v;
            self.pos[i] = p + v * h;
        }

        self.solve_stretch(h);
        self.solve_bending(h);
        self.limit_strain();
        if self.s.use_self_collision {
            self.self_collide();
        }
        self.collide();
        self.limit_strain();
        self.collide();

        let damp = (-self.s.damping.max(0.0) * 10.0 * h).exp();
        for i in 0..n {
            if self.inv_mass[i] == 0.0 {
                self.vel[i] = Vec3::ZERO;
                continue;
            }
            if !self.pos[i].is_finite() {
                self.pos[i] = self.prev[i];
                self.vel[i] = Vec3::ZERO;
                continue;
            }
            self.vel[i] = (self.pos[i] - self.prev[i]) / h * damp;
        }
        self.time += h;
    }

    fn step(&mut self, dt: f32, gravity: Vec3) {
        let sub = self.s.quality_steps.clamp(1, 64);
        let h = dt.clamp(1.0e-5, 0.1) / sub as f32;
        let g = gravity * self.s.gravity_scale;
        for _ in 0..sub {
            self.substep(h, g);
        }
    }

    /// Current state as an object-local mesh (`inv_world` maps world space to object space).
    fn local_mesh(&self, inv_world: &Mat4) -> MeshData {
        let mut mesh = MeshData {
            name: self.template.name.clone(),
            vertices: self.template.vertices.clone(),
            indices: self.template.indices.clone(),
            groups: self.template.groups.clone(),
        };
        for (i, v) in mesh.vertices.iter_mut().enumerate() {
            let src = self.pos[self.rep[i] as usize];
            v.position = inv_world.transform_point3(src).to_array();
        }
        crate::simulation::recompute_normals(&mut mesh);
        mesh
    }
}

fn pick_pins(
    pos: &[Vec3],
    tris: &[[u32; 3]],
    stretch: &[Stretch],
    active: &[bool],
    s: &ClothSettings,
) -> Vec<bool> {
    let n = pos.len();
    let mut pinned = vec![false; n];
    if s.pin == ClothPin::None {
        return pinned;
    }
    match s.pin {
        ClothPin::None => {}
        ClothPin::Boundary => {
            let flat: Vec<u32> = tris.iter().flat_map(|t| t.iter().copied()).collect();
            for i in crate::simulation::boundary_verts(&flat, n) {
                pinned[i] = true;
            }
        }
        ClothPin::TopEdge | ClothPin::TopCorners => {
            let mut max_y = f32::MIN;
            for i in 0..n {
                if active[i] {
                    max_y = max_y.max(pos[i].y);
                }
            }
            let mean_edge = if stretch.is_empty() {
                0.01
            } else {
                stretch.iter().map(|e| e.rest).sum::<f32>() / stretch.len() as f32
            };
            let tol = (mean_edge * 0.5).max(1.0e-3);
            let top: Vec<usize> = (0..n)
                .filter(|&i| active[i] && pos[i].y >= max_y - tol)
                .collect();
            if s.pin == ClothPin::TopEdge {
                for &i in &top {
                    pinned[i] = true;
                }
            } else if !top.is_empty() {
                let (mut min_x, mut max_x, mut min_z, mut max_z) =
                    (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for &i in &top {
                    min_x = min_x.min(pos[i].x);
                    max_x = max_x.max(pos[i].x);
                    min_z = min_z.min(pos[i].z);
                    max_z = max_z.max(pos[i].z);
                }
                let along_x = (max_x - min_x) >= (max_z - min_z);
                let key = |i: usize| if along_x { pos[i].x } else { pos[i].z };
                let lo = top
                    .iter()
                    .copied()
                    .min_by(|&a, &b| key(a).total_cmp(&key(b)));
                let hi = top
                    .iter()
                    .copied()
                    .max_by(|&a, &b| key(a).total_cmp(&key(b)));
                if let Some(i) = lo {
                    pinned[i] = true;
                }
                if let Some(i) = hi {
                    pinned[i] = true;
                }
            }
        }
    }
    // Clamp extra rings of neighbours.
    if s.pin_depth > 1 {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for e in stretch {
            adj[e.a as usize].push(e.b);
            adj[e.b as usize].push(e.a);
        }
        for _ in 1..s.pin_depth.min(8) {
            let mut add: Vec<usize> = Vec::new();
            for i in 0..n {
                if pinned[i] {
                    for &j in &adj[i] {
                        if !pinned[j as usize] {
                            add.push(j as usize);
                        }
                    }
                }
            }
            for j in add {
                pinned[j] = true;
            }
        }
    }
    pinned
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

/// Simulate one cloth object over the timeline and store a mesh sequence cache.
/// The first cache frame is the rest pose. Returns false if the object is gone.
pub(crate) fn bake_cloth(scene: &mut Scene, id: u64, settings: &ClothSettings) -> bool {
    let Some(obj_index) = scene.index_of(id) else {
        return false;
    };
    let fps = scene.animation.fps.max(1) as f32;
    let start = scene.animation.frame_start;
    let end = scene.animation.frame_end.max(start);
    let count = (end - start + 1).max(1) as usize;
    let gravity = scene.physics_world.gravity;
    let worlds = scene.world_matrices();
    let m = worlds.get(obj_index).copied().unwrap_or(Mat4::IDENTITY);
    let inv = m.inverse();

    // Re-baking starts again from the rest pose stored in the previous cache.
    let (source, name): (Arc<MeshData>, String) = {
        let object = &scene.objects[obj_index];
        let src = match scene.mesh_caches.get(&id) {
            Some(seq) if seq.source.starts_with("Cloth bake") && !seq.frames.is_empty() => {
                seq.frames[0].clone()
            }
            _ => object.mesh.clone(),
        };
        (src, object.name.clone())
    };
    let world_mesh = refine_for_cloth(source.transformed(m), settings.target_edge_length);
    let colliders = if settings.use_collision {
        collect_colliders(scene, &worlds, id, settings.collision_distance)
    } else {
        None
    };
    let fields = if settings.use_force_fields {
        collect_fields(scene, &worlds)
    } else {
        Vec::new()
    };

    let mut sim = ClothSim::new(world_mesh, settings, colliders, fields);
    let dt = 1.0 / fps;
    let mut frames: Vec<Arc<MeshData>> = Vec::with_capacity(count);
    for k in 0..count {
        if k > 0 {
            sim.step(dt, gravity);
        }
        frames.push(Arc::new(sim.local_mesh(&inv)));
    }
    scene.mesh_caches.insert(
        id,
        MeshSequence::new(format!("Cloth bake of {}", name), frames, start as f32, fps),
    );
    true
}

// ---------------------------------------------------------------------------
// Progressive baker (keeps the UI responsive, same pattern as FluidBaker)
// ---------------------------------------------------------------------------

/// Stepped cloth bake: advance a few simulation frames per UI tick so the app does not freeze.
pub struct ClothBaker {
    object_id: u64,
    object_name: String,
    sim: ClothSim,
    inv_world: Mat4,
    gravity: Vec3,
    fps: f32,
    start: i32,
    end: i32,
    /// Next frame index relative to start (0 = rest pose already stored).
    next_k: usize,
    frames: Vec<Arc<MeshData>>,
}

impl ClothBaker {
    /// Prepare a progressive bake for one cloth object. Uses the scene's gravity.
    pub fn new(scene: &Scene, id: u64, settings: &ClothSettings) -> Result<Self, String> {
        let Some(obj_index) = scene.index_of(id) else {
            return Err("Cloth object no longer exists.".into());
        };
        let object = &scene.objects[obj_index];
        if !object.kind.is_mesh() {
            return Err("Cloth requires a mesh object.".into());
        }
        let fps = scene.animation.fps.max(1) as f32;
        let start = scene.animation.frame_start;
        let end = scene.animation.frame_end.max(start);
        let count = (end - start + 1).max(1) as usize;
        let gravity = scene.physics_world.gravity;
        let worlds = scene.world_matrices();
        let m = worlds.get(obj_index).copied().unwrap_or(Mat4::IDENTITY);
        let inv = m.inverse();

        let source: Arc<MeshData> = match scene.mesh_caches.get(&id) {
            Some(seq) if seq.source.starts_with("Cloth bake") && !seq.frames.is_empty() => {
                seq.frames[0].clone()
            }
            _ => object.mesh.clone(),
        };
        let world_mesh = refine_for_cloth(source.transformed(m), settings.target_edge_length);
        let colliders = if settings.use_collision {
            collect_colliders(scene, &worlds, id, settings.collision_distance)
        } else {
            None
        };
        let fields = if settings.use_force_fields {
            collect_fields(scene, &worlds)
        } else {
            Vec::new()
        };
        let sim = ClothSim::new(world_mesh, settings, colliders, fields);
        let mut frames = Vec::with_capacity(count);
        // Frame 0 is the rest pose (matches bake_cloth).
        frames.push(Arc::new(sim.local_mesh(&inv)));
        Ok(Self {
            object_id: id,
            object_name: object.name.clone(),
            sim,
            inv_world: inv,
            gravity,
            fps,
            start,
            end,
            next_k: 1,
            frames,
        })
    }

    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    pub fn progress(&self) -> (i32, i32, i32) {
        let cur = self.start + self.next_k as i32 - 1;
        (cur.min(self.end), self.start, self.end)
    }

    pub fn is_finished(&self) -> bool {
        let count = (self.end - self.start + 1).max(1) as usize;
        self.next_k >= count
    }

    /// Advance simulation for up to `budget`. Returns true when the bake is complete.
    pub fn step(&mut self, budget: Duration) -> bool {
        let count = (self.end - self.start + 1).max(1) as usize;
        if self.next_k >= count {
            return true;
        }
        let dt = 1.0 / self.fps;
        let deadline = Instant::now() + budget;
        while self.next_k < count {
            self.sim.step(dt, self.gravity);
            self.frames
                .push(Arc::new(self.sim.local_mesh(&self.inv_world)));
            self.next_k += 1;
            if Instant::now() >= deadline {
                break;
            }
        }
        self.next_k >= count
    }

    /// Write the finished mesh sequence into the scene and return the frame count.
    pub fn finish(self, scene: &mut Scene) -> usize {
        let n = self.frames.len();
        scene.mesh_caches.insert(
            self.object_id,
            MeshSequence::new(
                format!("Cloth bake of {}", self.object_name),
                self.frames,
                self.start as f32,
                self.fps,
            ),
        );
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::plane;
    use crate::scene::Object;

    fn scene_with_cloth(settings: ClothSettings) -> (Scene, u64) {
        let mut scene = Scene::default();
        scene.animation.frame_start = 1;
        scene.animation.frame_end = 24;
        scene.animation.fps = 24;
        let mut o = Object::new("Cloth", plane(1.0));
        o.id = scene.alloc_id();
        let id = o.id;
        scene.objects.push(o);
        scene
            .object_simulations
            .insert(id, crate::ObjectSimulation::Cloth(settings));
        (scene, id)
    }

    fn lowest_y(mesh: &MeshData) -> f32 {
        mesh.vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::MAX, f32::min)
    }

    #[test]
    fn presets_exist_for_every_fabric_except_custom() {
        for f in FabricType::ALL {
            assert_eq!(f.preset().is_some(), f != FabricType::Custom);
        }
        let leather = FabricType::Leather.preset().unwrap();
        let silk = FabricType::Silk.preset().unwrap();
        assert!(leather.bending > silk.bending * 100.0);
        assert!(leather.mass > silk.mass);
    }

    #[test]
    fn applying_a_fabric_sets_the_values() {
        let mut s = ClothSettings::default();
        s.apply_fabric(FabricType::Leather);
        assert_eq!(s.fabric, FabricType::Leather);
        assert_eq!(s.bending_stiffness, FabricType::Leather.preset().unwrap().bending);
    }

    #[test]
    fn low_resolution_plane_is_refined_and_falls() {
        let mut s = ClothSettings::with_fabric(FabricType::Cotton);
        s.enabled = true;
        s.use_collision = false;
        let (mut scene, id) = scene_with_cloth(s.clone());
        assert!(bake_cloth(&mut scene, id, &s));
        let seq = scene.mesh_caches.get(&id).expect("cache");
        assert_eq!(seq.len(), 24);
        assert!(seq.frames[0].vertices.len() > 100, "plane should be refined");
        let first = lowest_y(&seq.frames[0]);
        let last = lowest_y(&seq.frames[23]);
        assert!(last < first - 0.5, "unpinned cloth must fall: {first} -> {last}");
        assert!(seq.frames[23]
            .vertices
            .iter()
            .all(|v| v.position.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn pinned_top_edge_stays_put() {
        let mut s = ClothSettings::with_fabric(FabricType::Denim);
        s.enabled = true;
        s.use_collision = false;
        s.pin = ClothPin::TopEdge;
        let (mut scene, id) = scene_with_cloth(s.clone());
        // Stand the plane up so its top row is a clear edge.
        let mesh = scene.objects[0].mesh.transformed(Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2));
        scene.objects[0].mesh = Arc::new(mesh);
        assert!(bake_cloth(&mut scene, id, &s));
        let seq = scene.mesh_caches.get(&id).unwrap();
        let top = |m: &MeshData| m.vertices.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);
        assert!((top(&seq.frames[0]) - top(&seq.frames[23])).abs() < 1e-3);
        assert!(lowest_y(&seq.frames[23]) > lowest_y(&seq.frames[0]) - 0.1, "hangs, does not fall");
    }

    #[test]
    fn cloth_rests_on_a_floor_collider() {
        let mut s = ClothSettings::with_fabric(FabricType::Silk);
        s.enabled = true;
        s.use_collision = true;
        let (mut scene, id) = scene_with_cloth(s.clone());
        scene.objects[0].translation = Vec3::new(0.0, 0.5, 0.0);
        let mut floor = Object::new("Floor", plane(6.0));
        floor.id = scene.alloc_id();
        scene.objects.push(floor);
        assert!(bake_cloth(&mut scene, id, &s));
        let seq = scene.mesh_caches.get(&id).unwrap();
        let low = lowest_y(&seq.frames[23]);
        assert!(low > -0.01, "cloth must not pass through the floor: {low}");
    }

    #[test]
    fn rebake_starts_from_the_rest_pose() {
        let mut s = ClothSettings::with_fabric(FabricType::Cotton);
        s.enabled = true;
        s.use_collision = false;
        let (mut scene, id) = scene_with_cloth(s.clone());
        assert!(bake_cloth(&mut scene, id, &s));
        let first_a = lowest_y(&scene.mesh_caches[&id].frames[0]);
        // Simulate the timeline having swapped in the last frame, then bake again.
        let last = scene.mesh_caches[&id].frames[23].clone();
        scene.objects[0].mesh = last;
        assert!(bake_cloth(&mut scene, id, &s));
        let first_b = lowest_y(&scene.mesh_caches[&id].frames[0]);
        assert!((first_a - first_b).abs() < 1e-4);
    }
}
