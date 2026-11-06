//! Full Edit Mode: selection domains, transforms, delete, extrude, dissolve.
use crate::mesh::{MeshData, Vertex};
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
            } else if self.selection.domain == EditDomain::Edge && !self.selection.edges.is_empty() {
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
        let mut map: std::collections::HashMap<(i32, i32, i32), u32> = std::collections::HashMap::new();
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
                let mut best: Option<(EdgeKey, f32)> = None;
                for &(a, b) in &self.build_edges() {
                    let pa = Vec3::from(self.mesh.vertices[a as usize].position);
                    let pb = Vec3::from(self.mesh.vertices[b as usize].position);
                    if let Some(dist) = ray_segment_dist(ray_o, dir, pa, pb) {
                        if dist < thresh && best.map_or(true, |(_, bd)| dist < bd) {
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
                    let wa = world.transform_point3(Vec3::from(
                        self.mesh.vertices[a as usize].position,
                    ));
                    let wb = world.transform_point3(Vec3::from(
                        self.mesh.vertices[b as usize].position,
                    ));
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
