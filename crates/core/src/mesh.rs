use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Linear RGBA vertex color (1,1,1,1 = white / unpainted).
    pub color: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Named per-vertex weight group (Blender-style vertex group).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VertexGroup {
    pub name: String,
    /// Weight per mesh vertex in [0, 1]. Length matches `MeshData::vertices` when active.
    pub weights: Vec<f32>,
    /// Whether this group is locked against paint/edits.
    pub locked: bool,
}

impl VertexGroup {
    pub fn new(name: impl Into<String>, vertex_count: usize) -> Self {
        Self {
            name: name.into(),
            weights: vec![0.0; vertex_count],
            locked: false,
        }
    }

    pub fn resize(&mut self, vertex_count: usize) {
        self.weights.resize(vertex_count, 0.0);
    }

    pub fn normalize_all(groups: &mut [VertexGroup], vertex_count: usize) {
        for vi in 0..vertex_count {
            let mut sum = 0.0f32;
            for g in groups.iter() {
                if vi < g.weights.len() {
                    sum += g.weights[vi].max(0.0);
                }
            }
            if sum > 1e-8 {
                for g in groups.iter_mut() {
                    if vi < g.weights.len() {
                        g.weights[vi] = (g.weights[vi].max(0.0) / sum).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn union(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn radius(&self) -> f32 {
        (self.max - self.min).length() * 0.5
    }
}

/// A run of indices drawn with its own material (None = default material).
/// When a mesh has no groups, the object's single material applies to everything.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshGroup {
    pub first_index: u32,
    pub index_count: u32,
    pub material: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshData {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub groups: Vec<MeshGroup>,
    /// Optional named vertex groups for weight paint / armature assignment.
    #[serde(default)]
    pub vertex_groups: Vec<VertexGroup>,
    /// Index of the active vertex group for Weight Paint (-1 = none).
    #[serde(default = "default_active_vertex_group")]
    pub active_vertex_group: i32,
}

fn default_active_vertex_group() -> i32 {
    -1
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            name: String::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            groups: Vec::new(),
            vertex_groups: Vec::new(),
            active_vertex_group: -1,
        }
    }
}

impl MeshData {
    /// Returns a copy with `m` baked into positions and normals.
    pub fn transformed(&self, m: Mat4) -> MeshData {
        let nm = Mat3::from_mat4(m).inverse().transpose();
        let vertices = self
            .vertices
            .iter()
            .map(|v| Vertex {
                position: m.transform_point3(Vec3::from(v.position)).to_array(),
                normal: (nm * Vec3::from(v.normal)).normalize_or_zero().to_array(),
                uv: v.uv,
                color: v.color,
            })
            .collect();
        let mut indices = self.indices.clone();
        if Mat3::from_mat4(m).determinant() < 0.0 {
            for t in indices.chunks_exact_mut(3) {
                t.swap(1, 2); // mirrored transform: keep front faces front
            }
        }
        MeshData {
            name: self.name.clone(),
            vertices,
            indices,
            groups: self.groups.clone(),
            vertex_groups: self.vertex_groups.clone(),
            active_vertex_group: self.active_vertex_group,
        }
    }

    /// Reconstruct quad faces from pairs of consistently wound, coplanar
    /// render triangles. The GPU/index buffer remains triangulated, while
    /// modeling and UV tools can reason about the original four-sided face.
    /// Triangles with a boundary edge, a non-manifold edge, or a sharp fold
    /// are deliberately left as triangles.
    pub fn quad_faces(&self) -> Vec<[u32; 4]> {
        let mut edge_owners: HashMap<(u32, u32), Vec<(usize, u32, u32, u32)>> = HashMap::new();
        for (face, tri) in self.indices.chunks_exact(3).enumerate() {
            let [a, b, c] = [tri[0], tri[1], tri[2]];
            for (from, to, other) in [(a, b, c), (b, c, a), (c, a, b)] {
                let edge = if from < to { (from, to) } else { (to, from) };
                edge_owners
                    .entry(edge)
                    .or_default()
                    .push((face, from, to, other));
            }
        }
        let mut paired = std::collections::HashSet::new();
        let mut quads = Vec::new();
        for owners in edge_owners.values() {
            if owners.len() != 2 {
                continue;
            }
            let (f0, a0, b0, x) = owners[0];
            let (f1, a1, b1, y) = owners[1];
            if f0 == f1 || a0 != b1 || b0 != a1 {
                continue;
            }
            if !paired.insert((f0.min(f1), f0.max(f1))) {
                continue;
            }
            let p = |i: u32| Vec3::from(self.vertices[i as usize].position);
            // Both triangles: normal = (to - from) × (other - from) using the
            // oriented shared-edge endpoints stored above. The previous n1 formula
            // used (from - to) and accidentally flipped one normal, so planar
            // quads (e.g. primitives::plane) were rejected.
            let n0 = (p(b0) - p(a0)).cross(p(x) - p(a0)).normalize_or_zero();
            let n1 = (p(b1) - p(a1)).cross(p(y) - p(a1)).normalize_or_zero();
            if n0.dot(n1) < 0.999 {
                continue;
            }
            // Boundary order: first triangle's outer corner, shared edge,
            // second triangle's outer corner, then the other shared corner.
            quads.push([x, a0, y, b0]);
        }
        quads
    }

    pub fn bounds(&self) -> Option<Aabb> {
        let mut it = self.vertices.iter().map(|v| Vec3::from(v.position));
        let first = it.next()?;
        let (min, max) = it.fold((first, first), |(a, b), p| (a.min(p), b.max(p)));
        Some(Aabb { min, max })
    }

    /// Smooth vertex normals from triangle geometry (area weighted).
    pub fn compute_normals(&mut self) {
        let mut acc = vec![Vec3::ZERO; self.vertices.len()];
        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (pa, pb, pc) = (
                Vec3::from(self.vertices[a].position),
                Vec3::from(self.vertices[b].position),
                Vec3::from(self.vertices[c].position),
            );
            let n = (pb - pa).cross(pc - pa); // length = 2*area
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
        for (v, n) in self.vertices.iter_mut().zip(acc) {
            v.normal = n.normalize_or_zero().to_array();
        }
    }

    /// Weld vertices that share the same position, then average normals.
    /// Makes spheres/cylinders look smooth even if they were built with split verts.
    pub fn shade_smooth(&mut self) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            return;
        }
        // Quantize positions so "same" points weld reliably
        let key = |p: [f32; 3]| -> (i32, i32, i32) {
            (
                (p[0] * 1_000_000.0).round() as i32,
                (p[1] * 1_000_000.0).round() as i32,
                (p[2] * 1_000_000.0).round() as i32,
            )
        };
        let mut map: std::collections::HashMap<(i32, i32, i32), u32> =
            std::collections::HashMap::new();
        let mut new_verts: Vec<Vertex> = Vec::new();
        let mut remap = vec![0u32; self.vertices.len()];
        for (i, v) in self.vertices.iter().enumerate() {
            let k = key(v.position);
            if let Some(&ni) = map.get(&k) {
                remap[i] = ni;
            } else {
                let ni = new_verts.len() as u32;
                map.insert(k, ni);
                new_verts.push(*v);
                remap[i] = ni;
            }
        }
        for idx in &mut self.indices {
            *idx = remap[*idx as usize];
        }
        self.vertices = new_verts;
        self.compute_normals();
    }

    /// Flat shading: each triangle gets its own vertices with the face normal.
    pub fn shade_flat(&mut self) {
        if self.indices.is_empty() {
            return;
        }
        let mut new_verts = Vec::with_capacity(self.indices.len());
        let mut new_idx = Vec::with_capacity(self.indices.len());
        for tri in self.indices.chunks_exact(3) {
            let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let pa = Vec3::from(self.vertices[ia].position);
            let pb = Vec3::from(self.vertices[ib].position);
            let pc = Vec3::from(self.vertices[ic].position);
            let n = (pb - pa).cross(pc - pa).normalize_or_zero().to_array();
            let base = new_verts.len() as u32;
            for &src in &[ia, ib, ic] {
                let mut v = self.vertices[src];
                v.normal = n;
                new_verts.push(v);
            }
            new_idx.extend_from_slice(&[base, base + 1, base + 2]);
        }
        self.vertices = new_verts;
        self.indices = new_idx;
        // Rebuild groups as one group if needed
        if !self.groups.is_empty() {
            let mat = self.groups[0].material;
            self.groups.clear();
            self.groups.push(MeshGroup {
                first_index: 0,
                index_count: self.indices.len() as u32,
                material: mat,
            });
        }
    }
}

impl MeshData {
    pub fn centroid(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.vertices.iter().map(|v| Vec3::from(v.position)).sum();
        sum / self.vertices.len() as f32
    }

    /// Make sure the mesh has explicit material groups (needed before appending other meshes).

    /// Ensure every vertex has a valid white color if colors were zeroed by Default/Pod.
    pub fn ensure_vertex_colors(&mut self) {
        for v in &mut self.vertices {
            if v.color[0] == 0.0 && v.color[1] == 0.0 && v.color[2] == 0.0 && v.color[3] == 0.0 {
                v.color = [1.0, 1.0, 1.0, 1.0];
            }
        }
    }

    pub fn ensure_vertex_group_sizes(&mut self) {
        let n = self.vertices.len();
        for g in &mut self.vertex_groups {
            g.resize(n);
        }
    }

    pub fn add_vertex_group(&mut self, name: impl Into<String>) -> usize {
        let n = self.vertices.len();
        self.vertex_groups.push(VertexGroup::new(name, n));
        let idx = self.vertex_groups.len() - 1;
        self.active_vertex_group = idx as i32;
        idx
    }

    pub fn remove_vertex_group(&mut self, index: usize) {
        if index >= self.vertex_groups.len() {
            return;
        }
        self.vertex_groups.remove(index);
        if self.vertex_groups.is_empty() {
            self.active_vertex_group = -1;
        } else if self.active_vertex_group as usize >= self.vertex_groups.len() {
            self.active_vertex_group = (self.vertex_groups.len() - 1) as i32;
        }
    }

    pub fn active_group_mut(&mut self) -> Option<&mut VertexGroup> {
        let i = self.active_vertex_group;
        if i >= 0 {
            self.vertex_groups.get_mut(i as usize)
        } else {
            None
        }
    }

    pub fn normalize_vertex_groups(&mut self) {
        let n = self.vertices.len();
        VertexGroup::normalize_all(&mut self.vertex_groups, n);
    }

    /// Blender-style weight paint heat map:
    /// 0 blue → cyan → green → yellow → 1 red.
    pub fn weight_to_color(w: f32) -> [f32; 4] {
        // Matches Blender's common weight-paint ramp (smoothstep blends between stops).
        let t = w.clamp(0.0, 1.0);
        // Stops: (t, r, g, b)
        const STOPS: [(f32, f32, f32, f32); 5] = [
            (0.00, 0.00, 0.00, 1.00), // blue
            (0.25, 0.00, 1.00, 1.00), // cyan
            (0.50, 0.00, 1.00, 0.00), // green
            (0.75, 1.00, 1.00, 0.00), // yellow
            (1.00, 1.00, 0.00, 0.00), // red
        ];
        let mut i = 0;
        while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
            i += 1;
        }
        if i + 1 >= STOPS.len() {
            return [1.0, 0.0, 0.0, 1.0];
        }
        let (t0, r0, g0, b0) = STOPS[i];
        let (t1, r1, g1, b1) = STOPS[i + 1];
        let u = if (t1 - t0).abs() < 1e-8 {
            0.0
        } else {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        };
        // smoothstep for a softer Blender-like blend
        let u = u * u * (3.0 - 2.0 * u);
        [
            r0 + (r1 - r0) * u,
            g0 + (g1 - g0) * u,
            b0 + (b1 - b0) * u,
            1.0,
        ]
    }

    /// Apply active vertex-group weights as vertex colors for visualization.
    pub fn apply_weight_visualization(&mut self) {
        let n = self.vertices.len();
        let weights: Vec<f32> = if let Some(g) = self.active_group_mut() {
            let mut w = g.weights.clone();
            w.resize(n, 0.0);
            w
        } else {
            vec![0.0; n]
        };
        for (v, w) in self.vertices.iter_mut().zip(weights.iter()) {
            v.color = Self::weight_to_color(*w);
        }
    }

    /// Restore vertex colors to white (exit weight-viz).
    pub fn clear_weight_visualization(&mut self) {
        for v in &mut self.vertices {
            v.color = [1.0, 1.0, 1.0, 1.0];
        }
    }

    pub fn ensure_groups(&mut self, material: Option<usize>) {
        if self.groups.is_empty() && !self.indices.is_empty() {
            self.groups.push(MeshGroup {
                first_index: 0,
                index_count: self.indices.len() as u32,
                material,
            });
        }
    }

    pub fn merge_adjacent_groups(&mut self) {
        let mut out: Vec<MeshGroup> = Vec::new();
        for g in &self.groups {
            if let Some(last) = out.last_mut() {
                if last.material == g.material
                    && last.first_index + last.index_count == g.first_index
                {
                    last.index_count += g.index_count;
                    continue;
                }
            }
            out.push(*g);
        }
        self.groups = out;
    }

    /// Append `other` (transformed by `m`). `fallback` is the material for other's ungrouped triangles.
    /// Call `ensure_groups` on self first.
    pub fn append(&mut self, other: &MeshData, m: Mat4, fallback: Option<usize>) {
        let o = other.transformed(m);
        let vbase = self.vertices.len() as u32;
        let ibase = self.indices.len() as u32;
        self.vertices.extend_from_slice(&o.vertices);
        self.indices.extend(o.indices.iter().map(|i| i + vbase));
        if o.groups.is_empty() {
            self.groups.push(MeshGroup {
                first_index: ibase,
                index_count: o.indices.len() as u32,
                material: fallback,
            });
        } else {
            for g in &o.groups {
                self.groups.push(MeshGroup {
                    first_index: g.first_index + ibase,
                    index_count: g.index_count,
                    material: g.material,
                });
            }
        }
    }

    fn tri_materials(&self) -> Option<Vec<Option<usize>>> {
        if self.groups.is_empty() {
            return None;
        }
        let mut tm = vec![None; self.indices.len() / 3];
        for g in &self.groups {
            let (a, b) = (
                g.first_index as usize / 3,
                (g.first_index + g.index_count) as usize / 3,
            );
            for t in a..b.min(tm.len()) {
                tm[t] = g.material;
            }
        }
        Some(tm)
    }

    /// New mesh made from the given triangles (vertices are re-indexed).
    fn subset(&self, tris: &[usize], tm: &Option<Vec<Option<usize>>>, name: String) -> MeshData {
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut m = MeshData {
            name,
            ..Default::default()
        };
        for &t in tris {
            for k in 0..3 {
                let old = self.indices[t * 3 + k];
                let ni = match remap.get(&old) {
                    Some(&n) => n,
                    None => {
                        m.vertices.push(self.vertices[old as usize]);
                        let n = (m.vertices.len() - 1) as u32;
                        remap.insert(old, n);
                        n
                    }
                };
                m.indices.push(ni);
            }
            if let Some(tm) = tm {
                let mat = tm[t];
                match m.groups.last_mut() {
                    Some(g) if g.material == mat => g.index_count += 3,
                    _ => m.groups.push(MeshGroup {
                        first_index: (m.indices.len() - 3) as u32,
                        index_count: 3,
                        material: mat,
                    }),
                }
            }
        }
        m
    }

    /// Split into connected pieces (triangles sharing vertices).
    pub fn loose_parts(&self) -> Vec<MeshData> {
        let n = self.vertices.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut c = x;
            while p[c] != r {
                let nx = p[c];
                p[c] = r;
                c = nx;
            }
            r
        }
        // vertices at the same position count as connected (flat-shaded meshes duplicate vertices per face)
        let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
        for (vi, v) in self.vertices.iter().enumerate() {
            let key = (
                (v.position[0] * 1e5).round() as i64,
                (v.position[1] * 1e5).round() as i64,
                (v.position[2] * 1e5).round() as i64,
            );
            match weld.get(&key) {
                Some(&first) => {
                    let (ra, rb) = (find(&mut parent, vi), find(&mut parent, first));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
                None => {
                    weld.insert(key, vi);
                }
            }
        }
        for t in self.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
            let (rb, rc) = (find(&mut parent, b), find(&mut parent, c));
            if rb != rc {
                parent[rb] = rc;
            }
        }
        let mut order: Vec<usize> = Vec::new();
        let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ti, t) in self.indices.chunks_exact(3).enumerate() {
            let r = find(&mut parent, t[0] as usize);
            comps
                .entry(r)
                .or_insert_with(|| {
                    order.push(r);
                    Vec::new()
                })
                .push(ti);
        }
        if order.len() < 2 {
            return vec![self.clone()];
        }
        let tm = self.tri_materials();
        order
            .iter()
            .enumerate()
            .map(|(k, r)| {
                let name = if k == 0 {
                    self.name.clone()
                } else {
                    format!("{}.{:03}", self.name, k)
                };
                self.subset(&comps[r], &tm, name)
            })
            .collect()
    }

    /// One mesh per material group (ungrouped meshes return themselves).
    pub fn split_by_material(&self) -> Vec<(Option<usize>, MeshData)> {
        let Some(tm) = self.tri_materials() else {
            return vec![(None, self.clone())];
        };
        let mut mats: Vec<Option<usize>> = Vec::new();
        for m in &tm {
            if !mats.contains(m) {
                mats.push(*m);
            }
        }
        mats.into_iter()
            .enumerate()
            .map(|(k, mat)| {
                let tris: Vec<usize> = (0..tm.len()).filter(|&t| tm[t] == mat).collect();
                let name = if k == 0 {
                    self.name.clone()
                } else {
                    format!("{}.{:03}", self.name, k)
                };
                (mat, self.subset(&tris, &None, name))
            })
            .collect()
    }
}

#[cfg(test)]
mod quad_tests {
    use super::*;

    #[test]
    fn plane_render_triangles_reconstruct_one_quad() {
        let mesh = crate::primitives::plane(2.0);
        let quads = mesh.quad_faces();
        assert_eq!(quads.len(), 1);
        let unique: std::collections::HashSet<u32> = quads[0].into_iter().collect();
        assert_eq!(unique.len(), 4);
    }
}
