//! Full Edit Mode: selection domains, transforms, delete, extrude, dissolve.
use crate::mesh::MeshData;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditDomain {
    Vertex,
    Edge,
    Face,
}

impl Default for EditDomain {
    fn default() -> Self {
        Self::Vertex
    }
}

impl EditDomain {
    pub fn name(self) -> &'static str {
        match self {
            EditDomain::Vertex => "Vertex",
            EditDomain::Edge => "Edge",
            EditDomain::Face => "Face",
        }
    }
}

pub type EdgeKey = (u32, u32);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EditSelection {
    pub domain: EditDomain,
    pub verts: BTreeSet<u32>,
    pub edges: BTreeSet<EdgeKey>,
    pub faces: BTreeSet<u32>,
}

impl EditSelection {
    pub fn clear(&mut self) {
        self.verts.clear();
        self.edges.clear();
        self.faces.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty() && self.edges.is_empty() && self.faces.is_empty()
    }

    pub fn select_all_verts(&mut self, count: u32) {
        self.clear();
        self.domain = EditDomain::Vertex;
        self.verts = (0..count).collect();
    }

    pub fn select_all_edges(&mut self, edges: &[EdgeKey]) {
        self.clear();
        self.domain = EditDomain::Edge;
        self.edges = edges.iter().copied().collect();
    }

    pub fn select_all_faces(&mut self, face_count: u32) {
        self.clear();
        self.domain = EditDomain::Face;
        self.faces = (0..face_count).collect();
    }

    pub fn invert_verts(&mut self, count: u32) {
        let all: BTreeSet<u32> = (0..count).collect();
        self.verts = all.difference(&self.verts).copied().collect();
        self.domain = EditDomain::Vertex;
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EditPick {
    Vertex(u32),
    Edge(EdgeKey),
    Face(u32),
}

pub fn edge_key(a: u32, b: u32) -> EdgeKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Clone, Debug)]
pub struct EditSession {
    pub object_id: u64,
    pub mesh: MeshData,
    pub original: MeshData,
    pub selection: EditSelection,
}

impl EditSession {
    pub fn new(object_id: u64, mesh: MeshData) -> Self {
        Self {
            object_id,
            original: mesh.clone(),
            mesh,
            selection: EditSelection::default(),
        }
    }

    pub fn face_count(&self) -> u32 {
        (self.mesh.indices.len() / 3) as u32
    }

    pub fn build_edges(&self) -> Vec<EdgeKey> {
        let mut set = BTreeSet::new();
        for tri in self.mesh.indices.chunks_exact(3) {
            set.insert(edge_key(tri[0], tri[1]));
            set.insert(edge_key(tri[1], tri[2]));
            set.insert(edge_key(tri[2], tri[0]));
        }
        set.into_iter().collect()
    }

    pub fn selected_vert_indices(&self) -> BTreeSet<u32> {
        let mut verts = self.selection.verts.clone();
        match self.selection.domain {
            EditDomain::Vertex => {}
            EditDomain::Edge => {
                for &(a, b) in &self.selection.edges {
                    verts.insert(a);
                    verts.insert(b);
                }
            }
            EditDomain::Face => {
                for &f in &self.selection.faces {
                    let i = (f as usize) * 3;
                    if i + 2 < self.mesh.indices.len() {
                        verts.insert(self.mesh.indices[i]);
                        verts.insert(self.mesh.indices[i + 1]);
                        verts.insert(self.mesh.indices[i + 2]);
                    }
                }
            }
        }
        verts
    }

    pub fn selection_center(&self) -> Vec3 {
        let verts = self.selected_vert_indices();
        if verts.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        let mut n = 0u32;
        for vi in verts {
            if let Some(v) = self.mesh.vertices.get(vi as usize) {
                sum += Vec3::from(v.position);
                n += 1;
            }
        }
        if n == 0 {
            Vec3::ZERO
        } else {
            sum / n as f32
        }
    }

    pub fn snapshot_positions(&self) -> Vec<(u32, Vec3)> {
        self.selected_vert_indices()
            .into_iter()
            .filter_map(|vi| {
                self.mesh
                    .vertices
                    .get(vi as usize)
                    .map(|v| (vi, Vec3::from(v.position)))
            })
            .collect()
    }

    pub fn set_vert_positions(&mut self, positions: &[(u32, Vec3)]) {
        for &(vi, p) in positions {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                v.position = p.to_array();
            }
        }
        self.mesh.compute_normals();
    }

    pub fn translate_selection(&mut self, delta: Vec3) {
        for vi in self.selected_vert_indices() {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                v.position[0] += delta.x;
                v.position[1] += delta.y;
                v.position[2] += delta.z;
            }
        }
        self.mesh.compute_normals();
    }

    /// Rotate selected verts around `center` by `quat` (object space).
    pub fn rotate_selection(&mut self, center: Vec3, quat: Quat) {
        for vi in self.selected_vert_indices() {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                let p = Vec3::from(v.position);
                v.position = (center + quat * (p - center)).to_array();
            }
        }
        self.mesh.compute_normals();
    }

    /// Scale selected verts around `center` by `factor` (uniform or per-axis).
    pub fn scale_selection(&mut self, center: Vec3, factor: Vec3) {
        for vi in self.selected_vert_indices() {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                let p = Vec3::from(v.position);
                let d = p - center;
                v.position = (center + d * factor).to_array();
            }
        }
        self.mesh.compute_normals();
    }

    /// Delete faces or remove verts (and incident faces).
    pub fn delete_selection(&mut self) {
        match self.selection.domain {
            EditDomain::Face => self.delete_faces(),
            EditDomain::Edge => self.delete_edges(),
            EditDomain::Vertex => self.delete_verts(),
        }
    }

    fn delete_faces(&mut self) {
        if self.selection.faces.is_empty() {
            return;
        }
        let mut new_idx = Vec::new();
        for (fi, tri) in self.mesh.indices.chunks_exact(3).enumerate() {
            if !self.selection.faces.contains(&(fi as u32)) {
                new_idx.extend_from_slice(tri);
            }
        }
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.mesh.compute_normals();
    }

    fn delete_edges(&mut self) {
        // Remove faces that use any selected edge
        if self.selection.edges.is_empty() {
            return;
        }
        let edges = &self.selection.edges;
        let mut new_idx = Vec::new();
        for tri in self.mesh.indices.chunks_exact(3) {
            let e0 = edge_key(tri[0], tri[1]);
            let e1 = edge_key(tri[1], tri[2]);
            let e2 = edge_key(tri[2], tri[0]);
            if edges.contains(&e0) || edges.contains(&e1) || edges.contains(&e2) {
                continue;
            }
            new_idx.extend_from_slice(tri);
        }
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.mesh.compute_normals();
    }

    fn delete_verts(&mut self) {
        if self.selection.verts.is_empty() {
            return;
        }
        let kill = &self.selection.verts;
        // Drop triangles that reference deleted verts
        let mut new_idx = Vec::new();
        for tri in self.mesh.indices.chunks_exact(3) {
            if kill.contains(&tri[0]) || kill.contains(&tri[1]) || kill.contains(&tri[2]) {
                continue;
            }
            new_idx.extend_from_slice(tri);
        }
        // Compact vertices
        let mut remap = vec![None; self.mesh.vertices.len()];
        let mut new_verts = Vec::new();
        for (i, v) in self.mesh.vertices.iter().enumerate() {
            if kill.contains(&(i as u32)) {
                continue;
            }
            remap[i] = Some(new_verts.len() as u32);
            new_verts.push(v.clone());
        }
        for idx in &mut new_idx {
            *idx = remap[*idx as usize].unwrap_or(0);
        }
        self.mesh.vertices = new_verts;
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.mesh.compute_normals();
    }

    /// Extrude selected faces along their average normal by `amount`.
    /// Creates a rim of side faces and new front faces.
    pub fn extrude_faces(&mut self, amount: f32) {
        if self.selection.domain != EditDomain::Face || self.selection.faces.is_empty() {
            // If verts/edges selected, extrude along view is handled by caller with translate
            // For verts-only: duplicate and translate is a simple "extrude verts"
            if self.selection.domain == EditDomain::Vertex && !self.selection.verts.is_empty() {
                self.extrude_verts(amount);
            } else if self.selection.domain == EditDomain::Edge && !self.selection.edges.is_empty()
            {
                self.extrude_edges(amount);
            }
            return;
        }

        let faces: Vec<u32> = self.selection.faces.iter().copied().collect();
        let mut new_face_ids = BTreeSet::new();
        let mut vert_map: HashMap<u32, u32> = HashMap::new(); // old -> new extruded vert

        // Average normal of selected faces
        let mut n_acc = Vec3::ZERO;
        for &f in &faces {
            let i = (f as usize) * 3;
            if i + 2 >= self.mesh.indices.len() {
                continue;
            }
            let a = Vec3::from(self.mesh.vertices[self.mesh.indices[i] as usize].position);
            let b = Vec3::from(self.mesh.vertices[self.mesh.indices[i + 1] as usize].position);
            let c = Vec3::from(self.mesh.vertices[self.mesh.indices[i + 2] as usize].position);
            n_acc += (b - a).cross(c - a).normalize_or_zero();
        }
        let n = n_acc.normalize_or_zero();
        if n.length_squared() < 1e-8 {
            return;
        }
        let offset = n * amount;

        // Create new verts
        let old_verts: BTreeSet<u32> = faces
            .iter()
            .flat_map(|&f| {
                let i = (f as usize) * 3;
                [
                    self.mesh.indices[i],
                    self.mesh.indices[i + 1],
                    self.mesh.indices[i + 2],
                ]
            })
            .collect();

        for &vi in &old_verts {
            let v = self.mesh.vertices[vi as usize].clone();
            let mut nv = v;
            nv.position[0] += offset.x;
            nv.position[1] += offset.y;
            nv.position[2] += offset.z;
            let ni = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(nv);
            vert_map.insert(vi, ni);
        }

        // Boundary edges of selection (appear once)
        let mut edge_count: HashMap<EdgeKey, (u32, u32)> = HashMap::new();
        for &f in &faces {
            let i = (f as usize) * 3;
            let a = self.mesh.indices[i];
            let b = self.mesh.indices[i + 1];
            let c = self.mesh.indices[i + 2];
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let k = edge_key(x, y);
                edge_count
                    .entry(k)
                    .and_modify(|c| c.0 += 1)
                    .or_insert((1, if x < y { 0 } else { 1 }));
                // store oriented pair as (x,y) in values differently - use separate
                let _ = (x, y);
            }
        }
        // Rebuild boundary with orientation from faces
        let mut boundary: Vec<(u32, u32)> = Vec::new();
        for &f in &faces {
            let i = (f as usize) * 3;
            let a = self.mesh.indices[i];
            let b = self.mesh.indices[i + 1];
            let c = self.mesh.indices[i + 2];
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let k = edge_key(x, y);
                let cnt = edge_count.get(&k).map(|c| c.0).unwrap_or(0);
                if cnt == 1 {
                    boundary.push((x, y));
                }
            }
        }

        // Side faces (quads as 2 tris)
        for (a, b) in boundary {
            let a2 = vert_map[&a];
            let b2 = vert_map[&b];
            // a,b,b2 and a,b2,a2
            self.mesh.indices.extend_from_slice(&[a, b, b2]);
            self.mesh.indices.extend_from_slice(&[a, b2, a2]);
        }

        // Replace original faces with extruded front
        for &f in &faces {
            let i = (f as usize) * 3;
            let a = self.mesh.indices[i];
            let b = self.mesh.indices[i + 1];
            let c = self.mesh.indices[i + 2];
            self.mesh.indices[i] = vert_map[&a];
            self.mesh.indices[i + 1] = vert_map[&b];
            self.mesh.indices[i + 2] = vert_map[&c];
            new_face_ids.insert(f);
        }

        self.selection.clear();
        self.selection.domain = EditDomain::Face;
        self.selection.faces = new_face_ids;
        // Also select the new verts
        self.selection.verts = vert_map.values().copied().collect();
        self.mesh.compute_normals();
    }

    fn extrude_verts(&mut self, amount: f32) {
        // Simple: move selected verts along average normal of incident faces
        let mut n_acc = Vec3::ZERO;
        let sel = self.selection.verts.clone();
        for tri in self.mesh.indices.chunks_exact(3) {
            if sel.contains(&tri[0]) || sel.contains(&tri[1]) || sel.contains(&tri[2]) {
                let a = Vec3::from(self.mesh.vertices[tri[0] as usize].position);
                let b = Vec3::from(self.mesh.vertices[tri[1] as usize].position);
                let c = Vec3::from(self.mesh.vertices[tri[2] as usize].position);
                n_acc += (b - a).cross(c - a).normalize_or_zero();
            }
        }
        let n = n_acc.normalize_or_zero();
        if n.length_squared() < 1e-8 {
            self.translate_selection(Vec3::Y * amount);
        } else {
            self.translate_selection(n * amount);
        }
    }

    fn extrude_edges(&mut self, amount: f32) {
        // Move edge verts along average normal
        self.selection.domain = EditDomain::Edge;
        let verts = self.selected_vert_indices();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts = verts;
        self.extrude_verts(amount);
        self.selection.domain = EditDomain::Edge;
    }

    /// Duplicate selected faces (in place offset slightly) — like Shift+D without modal.
    pub fn duplicate_faces(&mut self, offset: Vec3) {
        if self.selection.faces.is_empty() {
            return;
        }
        let faces: Vec<u32> = self.selection.faces.iter().copied().collect();
        let mut vert_map: HashMap<u32, u32> = HashMap::new();
        let mut used = BTreeSet::new();
        for &f in &faces {
            let i = (f as usize) * 3;
            used.insert(self.mesh.indices[i]);
            used.insert(self.mesh.indices[i + 1]);
            used.insert(self.mesh.indices[i + 2]);
        }
        for &vi in &used {
            let mut v = self.mesh.vertices[vi as usize].clone();
            v.position[0] += offset.x;
            v.position[1] += offset.y;
            v.position[2] += offset.z;
            let ni = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(v);
            vert_map.insert(vi, ni);
        }
        let mut new_faces = BTreeSet::new();
        for &f in &faces {
            let i = (f as usize) * 3;
            let a = vert_map[&self.mesh.indices[i]];
            let b = vert_map[&self.mesh.indices[i + 1]];
            let c = vert_map[&self.mesh.indices[i + 2]];
            let fi = (self.mesh.indices.len() / 3) as u32;
            self.mesh.indices.extend_from_slice(&[a, b, c]);
            new_faces.insert(fi);
        }
        self.selection.clear();
        self.selection.domain = EditDomain::Face;
        self.selection.faces = new_faces;
        self.mesh.compute_normals();
    }

    /// Merge selected verts to their median point.
    pub fn merge_verts_at_center(&mut self) {
        let verts: Vec<u32> = self.selected_vert_indices().into_iter().collect();
        if verts.len() < 2 {
            return;
        }
        let center = self.selection_center();
        let target = verts[0];
        if let Some(v) = self.mesh.vertices.get_mut(target as usize) {
            v.position = center.to_array();
        }
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        for &vi in &verts {
            remap.insert(vi, target);
        }
        for idx in &mut self.mesh.indices {
            if let Some(&t) = remap.get(idx) {
                *idx = t;
            }
        }
        // Remove degenerate tris
        let mut new_idx = Vec::new();
        for tri in self.mesh.indices.chunks_exact(3) {
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                new_idx.extend_from_slice(tri);
            }
        }
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts.insert(target);
        self.mesh.compute_normals();
    }

    /// Merge vertices that share the same position (within epsilon).
    /// Fixes cubes built with per-face split verts so corners are single points.
    pub fn weld_coincident(&mut self, eps: f32) {
        if self.mesh.vertices.is_empty() {
            return;
        }
        let quant = |p: [f32; 3]| -> (i32, i32, i32) {
            let s = 1.0 / eps.max(1e-8);
            (
                (p[0] * s).round() as i32,
                (p[1] * s).round() as i32,
                (p[2] * s).round() as i32,
            )
        };
        let mut map: std::collections::HashMap<(i32, i32, i32), u32> =
            std::collections::HashMap::new();
        let mut new_verts: Vec<crate::mesh::Vertex> = Vec::new();
        let mut remap = vec![0u32; self.mesh.vertices.len()];
        for (i, v) in self.mesh.vertices.iter().enumerate() {
            let k = quant(v.position);
            if let Some(&ni) = map.get(&k) {
                remap[i] = ni;
            } else {
                let ni = new_verts.len() as u32;
                map.insert(k, ni);
                remap[i] = ni;
                new_verts.push(v.clone());
            }
        }
        if new_verts.len() == self.mesh.vertices.len() {
            return; // nothing to weld
        }
        let mut new_idx = Vec::with_capacity(self.mesh.indices.len());
        for tri in self.mesh.indices.chunks_exact(3) {
            let a = remap[tri[0] as usize];
            let b = remap[tri[1] as usize];
            let c = remap[tri[2] as usize];
            if a != b && b != c && a != c {
                new_idx.extend_from_slice(&[a, b, c]);
            }
        }
        self.mesh.vertices = new_verts;
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.mesh.compute_normals();
    }

    /// When selecting a vertex, also select every vertex at the same position
    /// (handles residual split verts on older meshes).
    pub fn expand_vert_selection_by_position(&mut self, eps: f32) {
        if self.selection.domain != EditDomain::Vertex || self.selection.verts.is_empty() {
            return;
        }
        let selected: Vec<[f32; 3]> = self
            .selection
            .verts
            .iter()
            .filter_map(|&i| self.mesh.vertices.get(i as usize).map(|v| v.position))
            .collect();
        let eps2 = eps * eps;
        for (i, v) in self.mesh.vertices.iter().enumerate() {
            let p = v.position;
            for s in &selected {
                let dx = p[0] - s[0];
                let dy = p[1] - s[1];
                let dz = p[2] - s[2];
                if dx * dx + dy * dy + dz * dz <= eps2 {
                    self.selection.verts.insert(i as u32);
                    break;
                }
            }
        }
    }

    // ------------------------------------------------------------------ new Edit Mode tools

    /// Inset selected faces toward their center by `factor` (0..1).
    /// Blender-like: creates an inner face and side walls; original face is replaced.
    pub fn inset_faces(&mut self, factor: f32) {
        // Accept face selection, or promote from verts/edges that form faces
        if self.selection.faces.is_empty() {
            // Promote: any face whose 3 verts are all selected
            if self.selection.domain == EditDomain::Vertex && !self.selection.verts.is_empty() {
                let nfaces = self.mesh.indices.len() / 3;
                for fi in 0..nfaces {
                    let i = fi * 3;
                    let a = self.mesh.indices[i];
                    let b = self.mesh.indices[i + 1];
                    let c = self.mesh.indices[i + 2];
                    if self.selection.verts.contains(&a)
                        && self.selection.verts.contains(&b)
                        && self.selection.verts.contains(&c)
                    {
                        self.selection.faces.insert(fi as u32);
                    }
                }
            }
        }
        if self.selection.faces.is_empty() {
            return;
        }
        self.selection.domain = EditDomain::Face;
        let factor = factor.clamp(0.01, 0.49);
        let faces: Vec<u32> = self.selection.faces.iter().copied().collect();
        let face_set: BTreeSet<u32> = faces.iter().copied().collect();

        // Keep all non-selected faces
        let mut new_indices: Vec<u32> = Vec::new();
        for (fi, tri) in self.mesh.indices.chunks_exact(3).enumerate() {
            if !face_set.contains(&(fi as u32)) {
                new_indices.extend_from_slice(tri);
            }
        }

        let mut new_face_sel = BTreeSet::new();

        for &f in &faces {
            let i = (f as usize) * 3;
            if i + 2 >= self.mesh.indices.len() {
                continue;
            }
            let ia = self.mesh.indices[i];
            let ib = self.mesh.indices[i + 1];
            let ic = self.mesh.indices[i + 2];
            let a = Vec3::from(self.mesh.vertices[ia as usize].position);
            let b = Vec3::from(self.mesh.vertices[ib as usize].position);
            let c = Vec3::from(self.mesh.vertices[ic as usize].position);
            let center = (a + b + c) / 3.0;

            let na = a.lerp(center, factor);
            let nb = b.lerp(center, factor);
            let nc = c.lerp(center, factor);

            let base = self.mesh.vertices.len() as u32;
            for p in [na, nb, nc] {
                let mut v = self.mesh.vertices[ia as usize].clone();
                v.position = p.to_array();
                self.mesh.vertices.push(v);
            }

            // Side walls (outer -> inner)
            new_indices.extend_from_slice(&[ia, ib, base + 1]);
            new_indices.extend_from_slice(&[ia, base + 1, base]);
            new_indices.extend_from_slice(&[ib, ic, base + 2]);
            new_indices.extend_from_slice(&[ib, base + 2, base + 1]);
            new_indices.extend_from_slice(&[ic, ia, base]);
            new_indices.extend_from_slice(&[ic, base, base + 2]);

            // Inner face
            let fi = (new_indices.len() / 3) as u32;
            new_indices.extend_from_slice(&[base, base + 1, base + 2]);
            new_face_sel.insert(fi);
        }

        self.mesh.indices = new_indices;
        self.selection.clear();
        self.selection.domain = EditDomain::Face;
        self.selection.faces = new_face_sel;
        self.mesh.compute_normals();
    }

    /// Bevel: for selected edges/verts, offset along average normals (chamfer-like).
    /// Works with vertex, edge, or face selection (faces → boundary verts).
    pub fn bevel(&mut self, amount: f32) {
        let amount = amount.max(0.0);
        if amount < 1e-8 {
            return;
        }
        // Build affected vertex set from any domain
        let mut verts: BTreeSet<u32> = self.selected_vert_indices();
        if verts.is_empty() && !self.selection.faces.is_empty() {
            for &f in &self.selection.faces {
                let i = (f as usize) * 3;
                if i + 2 < self.mesh.indices.len() {
                    verts.insert(self.mesh.indices[i]);
                    verts.insert(self.mesh.indices[i + 1]);
                    verts.insert(self.mesh.indices[i + 2]);
                }
            }
        }
        if verts.is_empty() && !self.selection.edges.is_empty() {
            for &(a, b) in &self.selection.edges {
                verts.insert(a);
                verts.insert(b);
            }
        }
        if verts.is_empty() {
            return;
        }
        let mut normals = vec![Vec3::ZERO; self.mesh.vertices.len()];
        for tri in self.mesh.indices.chunks_exact(3) {
            let a = Vec3::from(self.mesh.vertices[tri[0] as usize].position);
            let b = Vec3::from(self.mesh.vertices[tri[1] as usize].position);
            let c = Vec3::from(self.mesh.vertices[tri[2] as usize].position);
            let n = (b - a).cross(c - a).normalize_or_zero();
            for &vi in tri {
                normals[vi as usize] += n;
            }
        }
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }
        for &vi in &verts {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                let n = normals[vi as usize];
                v.position[0] += n.x * amount;
                v.position[1] += n.y * amount;
                v.position[2] += n.z * amount;
            }
        }
        self.mesh.compute_normals();
    }

    /// Laplacian smooth on selected vertices.
    pub fn smooth_verts(&mut self, factor: f32) {
        let factor = factor.clamp(0.0, 1.0);
        let verts: Vec<u32> = self.selected_vert_indices().into_iter().collect();
        if verts.is_empty() {
            return;
        }
        // Build neighbor map
        let mut neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
        for tri in self.mesh.indices.chunks_exact(3) {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                neighbors.entry(a).or_default().push(b);
                neighbors.entry(b).or_default().push(a);
            }
        }
        let mut new_pos: HashMap<u32, Vec3> = HashMap::new();
        for &vi in &verts {
            let Some(neigh) = neighbors.get(&vi) else {
                continue;
            };
            if neigh.is_empty() {
                continue;
            }
            let mut acc = Vec3::ZERO;
            for &n in neigh {
                acc += Vec3::from(self.mesh.vertices[n as usize].position);
            }
            let avg = acc / neigh.len() as f32;
            let cur = Vec3::from(self.mesh.vertices[vi as usize].position);
            new_pos.insert(vi, cur.lerp(avg, factor));
        }
        for (vi, p) in new_pos {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                v.position = p.to_array();
            }
        }
        self.mesh.compute_normals();
    }

    /// Move selected verts along their normals (shrink/fatten).
    pub fn shrink_fatten(&mut self, amount: f32) {
        let verts: BTreeSet<u32> = self.selected_vert_indices();
        if verts.is_empty() {
            return;
        }
        let mut normals = vec![Vec3::ZERO; self.mesh.vertices.len()];
        for tri in self.mesh.indices.chunks_exact(3) {
            let a = Vec3::from(self.mesh.vertices[tri[0] as usize].position);
            let b = Vec3::from(self.mesh.vertices[tri[1] as usize].position);
            let c = Vec3::from(self.mesh.vertices[tri[2] as usize].position);
            let n = (b - a).cross(c - a).normalize_or_zero();
            for &vi in tri {
                normals[vi as usize] += n;
            }
        }
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }
        for &vi in &verts {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                let n = normals[vi as usize];
                v.position[0] += n.x * amount;
                v.position[1] += n.y * amount;
                v.position[2] += n.z * amount;
            }
        }
        self.mesh.compute_normals();
    }

    /// Shear selected verts along an axis.
    pub fn shear(&mut self, axis: usize, amount: f32) {
        let center = self.selection_center();
        let verts: BTreeSet<u32> = self.selected_vert_indices();
        for &vi in &verts {
            if let Some(v) = self.mesh.vertices.get_mut(vi as usize) {
                let p = Vec3::from(v.position) - center;
                match axis {
                    0 => {
                        // shear X by Y
                        v.position[0] += p.y * amount;
                    }
                    1 => {
                        // shear Y by X
                        v.position[1] += p.x * amount;
                    }
                    _ => {
                        // shear X by Z
                        v.position[0] += p.z * amount;
                    }
                }
            }
        }
        self.mesh.compute_normals();
    }

    /// Edge slide: move selected verts along the average direction of connected selected edges.
    pub fn edge_slide(&mut self, amount: f32) {
        if self.selection.domain != EditDomain::Edge || self.selection.edges.is_empty() {
            // Fall back to sliding selected verts toward neighbors
            self.shrink_fatten(amount * 0.1);
            return;
        }
        for &(a, b) in &self.selection.edges.clone() {
            let pa = Vec3::from(self.mesh.vertices[a as usize].position);
            let pb = Vec3::from(self.mesh.vertices[b as usize].position);
            let dir = (pb - pa).normalize_or_zero();
            if let Some(va) = self.mesh.vertices.get_mut(a as usize) {
                let p = Vec3::from(va.position) + dir * amount;
                va.position = p.to_array();
            }
            if let Some(vb) = self.mesh.vertices.get_mut(b as usize) {
                let p = Vec3::from(vb.position) - dir * amount;
                vb.position = p.to_array();
            }
        }
        self.mesh.compute_normals();
    }

    /// Rip: duplicate selected verts and disconnect them from unselected geometry.
    pub fn rip_region(&mut self) {
        let verts: BTreeSet<u32> = self.selected_vert_indices();
        if verts.is_empty() {
            return;
        }
        let mut remap: HashMap<u32, u32> = HashMap::new();
        for &vi in &verts {
            let v = self.mesh.vertices[vi as usize].clone();
            let ni = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(v);
            remap.insert(vi, ni);
        }
        // Update indices: faces that use only selected verts keep old; mixed faces get new verts for selected
        for tri in self.mesh.indices.chunks_exact_mut(3) {
            let sel_count = tri.iter().filter(|i| verts.contains(i)).count();
            if sel_count > 0 && sel_count < 3 {
                for idx in tri.iter_mut() {
                    if let Some(&ni) = remap.get(idx) {
                        *idx = ni;
                    }
                }
            }
        }
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts = remap.values().copied().collect();
        self.mesh.compute_normals();
    }

    /// Loop cut (simplified): subdivide selected edges at midpoint and split adjacent faces.
    pub fn loop_cut(&mut self) {
        let edges: Vec<EdgeKey> =
            if self.selection.domain == EditDomain::Edge && !self.selection.edges.is_empty() {
                self.selection.edges.iter().copied().collect()
            } else {
                // If faces selected, cut all their edges
                let mut e = BTreeSet::new();
                for &f in &self.selection.faces {
                    let i = (f as usize) * 3;
                    if i + 2 < self.mesh.indices.len() {
                        e.insert(edge_key(self.mesh.indices[i], self.mesh.indices[i + 1]));
                        e.insert(edge_key(self.mesh.indices[i + 1], self.mesh.indices[i + 2]));
                        e.insert(edge_key(self.mesh.indices[i + 2], self.mesh.indices[i]));
                    }
                }
                e.into_iter().collect()
            };
        if edges.is_empty() {
            return;
        }
        let mut mid_map: HashMap<EdgeKey, u32> = HashMap::new();
        for &(a, b) in &edges {
            let pa = Vec3::from(self.mesh.vertices[a as usize].position);
            let pb = Vec3::from(self.mesh.vertices[b as usize].position);
            let mid = (pa + pb) * 0.5;
            let mut v = self.mesh.vertices[a as usize].clone();
            v.position = mid.to_array();
            let ni = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(v);
            mid_map.insert(edge_key(a, b), ni);
        }
        // Split faces that contain a cut edge
        let old_indices = self.mesh.indices.clone();
        let mut new_indices = Vec::new();
        for tri in old_indices.chunks_exact(3) {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let mab = mid_map.get(&edge_key(a, b)).copied();
            let mbc = mid_map.get(&edge_key(b, c)).copied();
            let mca = mid_map.get(&edge_key(c, a)).copied();
            match (mab, mbc, mca) {
                (Some(m), None, None) => {
                    new_indices.extend_from_slice(&[a, m, c]);
                    new_indices.extend_from_slice(&[m, b, c]);
                }
                (None, Some(m), None) => {
                    new_indices.extend_from_slice(&[a, b, m]);
                    new_indices.extend_from_slice(&[a, m, c]);
                }
                (None, None, Some(m)) => {
                    new_indices.extend_from_slice(&[a, b, m]);
                    new_indices.extend_from_slice(&[b, c, m]);
                }
                (Some(m0), Some(m1), None) => {
                    new_indices.extend_from_slice(&[a, m0, c]);
                    new_indices.extend_from_slice(&[m0, b, m1]);
                    new_indices.extend_from_slice(&[m0, m1, c]);
                }
                (None, Some(m0), Some(m1)) => {
                    new_indices.extend_from_slice(&[a, b, m0]);
                    new_indices.extend_from_slice(&[a, m0, m1]);
                    new_indices.extend_from_slice(&[m0, c, m1]);
                }
                (Some(m0), None, Some(m1)) => {
                    new_indices.extend_from_slice(&[a, m0, m1]);
                    new_indices.extend_from_slice(&[m0, b, c]);
                    new_indices.extend_from_slice(&[m0, c, m1]);
                }
                (Some(m0), Some(m1), Some(m2)) => {
                    // all three edges cut – make 4 tris
                    new_indices.extend_from_slice(&[a, m0, m2]);
                    new_indices.extend_from_slice(&[m0, b, m1]);
                    new_indices.extend_from_slice(&[m2, m1, c]);
                    new_indices.extend_from_slice(&[m0, m1, m2]);
                }
                _ => new_indices.extend_from_slice(tri),
            }
        }
        self.mesh.indices = new_indices;
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts = mid_map.values().copied().collect();
        self.mesh.compute_normals();
    }

    /// Spin: rotate selected geometry around Z axis and create connecting faces (simplified radial extrude).
    pub fn spin(&mut self, steps: u32, angle_deg: f32) {
        let steps = steps.clamp(2, 32);
        let verts: Vec<u32> = self.selected_vert_indices().into_iter().collect();
        if verts.is_empty() {
            return;
        }
        let angle = angle_deg.to_radians() / steps as f32;
        let center = self.selection_center();
        let mut rings: Vec<Vec<u32>> = vec![verts.clone()];
        for s in 1..=steps {
            let mut ring = Vec::new();
            let q = Quat::from_axis_angle(Vec3::Y, angle * s as f32);
            for &vi in &verts {
                let p = Vec3::from(self.mesh.vertices[vi as usize].position);
                let rotated = center + q * (p - center);
                let mut v = self.mesh.vertices[vi as usize].clone();
                v.position = rotated.to_array();
                let ni = self.mesh.vertices.len() as u32;
                self.mesh.vertices.push(v);
                ring.push(ni);
            }
            rings.push(ring);
        }
        // Bridge consecutive rings
        for r in 0..rings.len() - 1 {
            let a = &rings[r];
            let b = &rings[r + 1];
            for i in 0..a.len() {
                let j = (i + 1) % a.len();
                if a.len() < 2 {
                    break;
                }
                self.mesh.indices.extend_from_slice(&[a[i], a[j], b[j]]);
                self.mesh.indices.extend_from_slice(&[a[i], b[j], b[i]]);
            }
        }
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        if let Some(last) = rings.last() {
            self.selection.verts = last.iter().copied().collect();
        }
        self.mesh.compute_normals();
    }

    /// Knife / bisect: cut the mesh with a plane defined by point + normal.
    /// Splits edges that cross the plane and triangulates resulting faces.
    pub fn knife_bisect(&mut self, plane_point: Vec3, plane_normal: Vec3) {
        let n = plane_normal.normalize_or_zero();
        if n.length_squared() < 1e-8 {
            return;
        }
        let d = -n.dot(plane_point);
        // For each edge, if it crosses the plane, insert a vertex
        let edges = self.build_edges();
        let mut edge_cut: HashMap<EdgeKey, u32> = HashMap::new();
        for &(a, b) in &edges {
            let pa = Vec3::from(self.mesh.vertices[a as usize].position);
            let pb = Vec3::from(self.mesh.vertices[b as usize].position);
            let da = n.dot(pa) + d;
            let db = n.dot(pb) + d;
            if da * db < 0.0 {
                let t = da / (da - db);
                let p = pa + (pb - pa) * t;
                let mut v = self.mesh.vertices[a as usize].clone();
                v.position = p.to_array();
                let ni = self.mesh.vertices.len() as u32;
                self.mesh.vertices.push(v);
                edge_cut.insert(edge_key(a, b), ni);
            }
        }
        if edge_cut.is_empty() {
            return;
        }
        // Rebuild faces: split triangles that have cut edges
        let old = self.mesh.indices.clone();
        let mut new_idx = Vec::new();
        for tri in old.chunks_exact(3) {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let mab = edge_cut.get(&edge_key(a, b)).copied();
            let mbc = edge_cut.get(&edge_key(b, c)).copied();
            let mca = edge_cut.get(&edge_key(c, a)).copied();
            match (mab, mbc, mca) {
                (Some(m), None, None) => {
                    new_idx.extend_from_slice(&[a, m, c, m, b, c]);
                }
                (None, Some(m), None) => {
                    new_idx.extend_from_slice(&[a, b, m, a, m, c]);
                }
                (None, None, Some(m)) => {
                    new_idx.extend_from_slice(&[a, b, m, b, c, m]);
                }
                (Some(m0), Some(m1), None) => {
                    new_idx.extend_from_slice(&[a, m0, c, m0, b, m1, m0, m1, c]);
                }
                (None, Some(m0), Some(m1)) => {
                    new_idx.extend_from_slice(&[a, b, m0, a, m0, m1, m0, c, m1]);
                }
                (Some(m0), None, Some(m1)) => {
                    new_idx.extend_from_slice(&[a, m0, m1, m0, b, c, m0, c, m1]);
                }
                (Some(m0), Some(m1), Some(m2)) => {
                    new_idx.extend_from_slice(&[a, m0, m2, m0, b, m1, m2, m1, c, m0, m1, m2]);
                }
                _ => new_idx.extend_from_slice(tri),
            }
        }
        self.mesh.indices = new_idx;
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts = edge_cut.values().copied().collect();
        self.mesh.compute_normals();
    }

    /// Poly Build: from selected edge(s), extrude a new face outward along average normal.
    pub fn poly_build(&mut self, amount: f32) {
        let edges: Vec<EdgeKey> = if !self.selection.edges.is_empty() {
            self.selection.edges.iter().copied().collect()
        } else if !self.selection.verts.is_empty() && self.selection.verts.len() >= 2 {
            // connect selected verts as a chain
            let v: Vec<u32> = self.selection.verts.iter().copied().collect();
            let mut e = Vec::new();
            for i in 0..v.len() - 1 {
                e.push(edge_key(v[i], v[i + 1]));
            }
            e
        } else {
            return;
        };
        if edges.is_empty() {
            return;
        }
        // Average normal from adjacent faces
        let mut n_acc = Vec3::ZERO;
        for tri in self.mesh.indices.chunks_exact(3) {
            let a = Vec3::from(self.mesh.vertices[tri[0] as usize].position);
            let b = Vec3::from(self.mesh.vertices[tri[1] as usize].position);
            let c = Vec3::from(self.mesh.vertices[tri[2] as usize].position);
            n_acc += (b - a).cross(c - a).normalize_or_zero();
        }
        let n = n_acc.normalize_or_zero();
        let offset = n * amount;

        let mut new_verts_map: HashMap<u32, u32> = HashMap::new();
        let mut involved = BTreeSet::new();
        for &(a, b) in &edges {
            involved.insert(a);
            involved.insert(b);
        }
        for &vi in &involved {
            let mut v = self.mesh.vertices[vi as usize].clone();
            let p = Vec3::from(v.position) + offset;
            v.position = p.to_array();
            let ni = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(v);
            new_verts_map.insert(vi, ni);
        }
        for &(a, b) in &edges {
            let na = new_verts_map[&a];
            let nb = new_verts_map[&b];
            // quad as two tris
            self.mesh.indices.extend_from_slice(&[a, b, nb]);
            self.mesh.indices.extend_from_slice(&[a, nb, na]);
        }
        self.selection.clear();
        self.selection.domain = EditDomain::Vertex;
        self.selection.verts = new_verts_map.values().copied().collect();
        self.mesh.compute_normals();
    }

    /// Pick the edge whose screen projection is closest to the cursor (pixel distance).
    /// Much more reliable than world-space ray distance for thin lines.
    pub fn pick_edge_screen(
        &self,
        world: glam::Mat4,
        project: &dyn Fn(Vec3) -> Option<(f32, f32)>,
        cursor: (f32, f32),
        max_px: f32,
    ) -> Option<EditPick> {
        let mut best: Option<(EdgeKey, f32)> = None;
        for &(a, b) in &self.build_edges() {
            let wa = world.transform_point3(Vec3::from(self.mesh.vertices[a as usize].position));
            let wb = world.transform_point3(Vec3::from(self.mesh.vertices[b as usize].position));
            let (Some((x0, y0)), Some((x1, y1))) = (project(wa), project(wb)) else {
                continue;
            };
            let d = point_to_segment_px(cursor.0, cursor.1, x0, y0, x1, y1);
            if d <= max_px && best.map_or(true, |(_, bd)| d < bd) {
                best = Some(((a, b), d));
            }
        }
        best.map(|(e, _)| EditPick::Edge(e))
    }

    pub fn pick_component(&self, ray_o: Vec3, ray_d: Vec3, thresh: f32) -> Option<EditPick> {
        let dir = ray_d.normalize_or_zero();
        if dir.length_squared() < 1e-12 {
            return None;
        }
        match self.selection.domain {
            EditDomain::Vertex => {
                let mut best: Option<(u32, f32)> = None;
                for (i, v) in self.mesh.vertices.iter().enumerate() {
                    let p = Vec3::from(v.position);
                    let t = (p - ray_o).dot(dir);
                    if t < 0.0 {
                        continue;
                    }
                    let closest = ray_o + dir * t;
                    let dist = (p - closest).length();
                    if dist < thresh && best.map_or(true, |(_, bd)| dist < bd) {
                        best = Some((i as u32, dist));
                    }
                }
                best.map(|(i, _)| EditPick::Vertex(i))
            }
            EditDomain::Edge => {
                // Generous hit radius so clicking the visible line always works
                let edge_thresh = thresh.max(0.05) * 2.5;
                let mut best: Option<(EdgeKey, f32)> = None;
                for &(a, b) in &self.build_edges() {
                    let pa = Vec3::from(self.mesh.vertices[a as usize].position);
                    let pb = Vec3::from(self.mesh.vertices[b as usize].position);
                    if let Some(dist) = ray_segment_dist(ray_o, dir, pa, pb) {
                        if dist < edge_thresh && best.map_or(true, |(_, bd)| dist < bd) {
                            best = Some(((a, b), dist));
                        }
                    }
                }
                best.map(|(e, _)| EditPick::Edge(e))
            }
            EditDomain::Face => {
                let mut best: Option<(u32, f32)> = None;
                for (fi, tri) in self.mesh.indices.chunks_exact(3).enumerate() {
                    let p0 = Vec3::from(self.mesh.vertices[tri[0] as usize].position);
                    let p1 = Vec3::from(self.mesh.vertices[tri[1] as usize].position);
                    let p2 = Vec3::from(self.mesh.vertices[tri[2] as usize].position);
                    if let Some(t) = ray_tri(ray_o, dir, p0, p1, p2) {
                        if best.map_or(true, |(_, bt)| t < bt) {
                            best = Some((fi as u32, t));
                        }
                    }
                }
                best.map(|(f, _)| EditPick::Face(f))
            }
        }
    }

    /// Select components whose screen projection falls inside a rect (min,max in pixels).
    pub fn select_in_rect(
        &mut self,
        world: Mat4,
        project: &dyn Fn(Vec3) -> Option<(f32, f32)>,
        min: (f32, f32),
        max: (f32, f32),
        extend: bool,
    ) {
        if !extend {
            self.selection.clear();
        }
        let inside = |x: f32, y: f32| x >= min.0 && x <= max.0 && y >= min.1 && y <= max.1;
        match self.selection.domain {
            EditDomain::Vertex => {
                for (i, v) in self.mesh.vertices.iter().enumerate() {
                    let w = world.transform_point3(Vec3::from(v.position));
                    if let Some((x, y)) = project(w) {
                        if inside(x, y) {
                            self.selection.verts.insert(i as u32);
                        }
                    }
                }
            }
            EditDomain::Edge => {
                for (a, b) in self.build_edges() {
                    let wa =
                        world.transform_point3(Vec3::from(self.mesh.vertices[a as usize].position));
                    let wb =
                        world.transform_point3(Vec3::from(self.mesh.vertices[b as usize].position));
                    if let (Some((x0, y0)), Some((x1, y1))) = (project(wa), project(wb)) {
                        let mx = (x0 + x1) * 0.5;
                        let my = (y0 + y1) * 0.5;
                        if inside(mx, my) {
                            self.selection.edges.insert((a, b));
                        }
                    }
                }
            }
            EditDomain::Face => {
                for (fi, tri) in self.mesh.indices.chunks_exact(3).enumerate() {
                    let c = (Vec3::from(self.mesh.vertices[tri[0] as usize].position)
                        + Vec3::from(self.mesh.vertices[tri[1] as usize].position)
                        + Vec3::from(self.mesh.vertices[tri[2] as usize].position))
                        / 3.0;
                    let w = world.transform_point3(c);
                    if let Some((x, y)) = project(w) {
                        if inside(x, y) {
                            self.selection.faces.insert(fi as u32);
                        }
                    }
                }
            }
        }
    }
}

fn point_to_segment_px(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let abx = x1 - x0;
    let aby = y1 - y0;
    let ab2 = abx * abx + aby * aby;
    if ab2 < 1e-6 {
        let dx = px - x0;
        let dy = py - y0;
        return (dx * dx + dy * dy).sqrt();
    }
    let t = ((px - x0) * abx + (py - y0) * aby) / ab2;
    let t = t.clamp(0.0, 1.0);
    let cx = x0 + abx * t;
    let cy = y0 + aby * t;
    let dx = px - cx;
    let dy = py - cy;
    (dx * dx + dy * dy).sqrt()
}

fn ray_segment_dist(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3) -> Option<f32> {
    let ab = b - a;
    let ab2 = ab.length_squared();
    if ab2 < 1e-12 {
        let t = (a - ro).dot(rd);
        if t < 0.0 {
            return None;
        }
        return Some((ro + rd * t - a).length());
    }
    let ao = a - ro;
    let rd_ab = rd.dot(ab);
    let det = 1.0 - rd_ab * rd_ab / ab2;
    if det.abs() < 1e-8 {
        let t = ao.dot(rd);
        if t < 0.0 {
            return None;
        }
        let u = ((ro + rd * t - a).dot(ab) / ab2).clamp(0.0, 1.0);
        return Some((ro + rd * t - (a + ab * u)).length());
    }
    let t = (ao.dot(rd) - ao.dot(ab) * rd_ab / ab2) / det;
    let u = ((ao.dot(ab) - t * rd_ab) / ab2).clamp(0.0, 1.0);
    if t < 0.0 {
        return None;
    }
    Some((ro + rd * t - (a + ab * u)).length())
}

fn ray_tri(o: Vec3, d: Vec3, p0: Vec3, p1: Vec3, p2: Vec3) -> Option<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = o - p0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * d.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    (t > 1e-5).then_some(t)
}

pub fn world_positions(mesh: &MeshData, world: Mat4) -> Vec<Vec3> {
    mesh.vertices
        .iter()
        .map(|v| world.transform_point3(Vec3::from(v.position)))
        .collect()
}
