//! UV editing: unwrap projections, island pack, basic UV transforms.
//! Unwrappers split shared vertices so each face corner can hold unique UVs
//! (required for correct cube/smart projection on closed meshes).
use crate::mesh::{MeshData, Vertex};
use glam::{Vec2, Vec3};
use std::collections::{BTreeSet, HashMap};

/// Duplicate vertices so every triangle corner is unique (enables per-face UVs).
fn split_faces(mesh: &mut MeshData) {
    let mut new_verts: Vec<Vertex> = Vec::with_capacity(mesh.indices.len());
    let mut new_idx: Vec<u32> = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks_exact(3) {
        for &vi in tri {
            let v = mesh.vertices[vi as usize].clone();
            let ni = new_verts.len() as u32;
            new_verts.push(v);
            new_idx.push(ni);
        }
    }
    mesh.vertices = new_verts;
    mesh.indices = new_idx;
    mesh.groups.clear();
}

/// Smart UV Project (simplified Blender):
/// group faces by connected angle threshold, project each group along avg normal, pack.
pub fn unwrap_smart(mesh: &mut MeshData) {
    if mesh.indices.is_empty() {
        return;
    }
    const ANGLE_LIMIT_DEG: f32 = 66.0; // Blender default-ish
    let cos_limit = ANGLE_LIMIT_DEG.to_radians().cos();

    let nfaces = mesh.indices.len() / 3;
    // Face normals
    let mut fnorm: Vec<Vec3> = Vec::with_capacity(nfaces);
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
        fnorm.push((b - a).cross(c - a).normalize_or_zero());
    }
    // Edge → faces adjacency
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, tri) in mesh.indices.chunks_exact(3).enumerate() {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let e = if a < b { (a, b) } else { (b, a) };
            edge_faces.entry(e).or_default().push(fi);
        }
    }
    // Face adjacency if angle between normals is within limit
    let mut face_adj: Vec<Vec<usize>> = vec![Vec::new(); nfaces];
    for faces in edge_faces.values() {
        if faces.len() < 2 {
            continue;
        }
        for i in 0..faces.len() {
            for j in i + 1..faces.len() {
                let a = faces[i];
                let b = faces[j];
                if fnorm[a].dot(fnorm[b]) >= cos_limit {
                    face_adj[a].push(b);
                    face_adj[b].push(a);
                }
            }
        }
    }
    // Connected components of faces
    let mut seen = vec![false; nfaces];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..nfaces {
        if seen[i] {
            continue;
        }
        let mut stack = vec![i];
        let mut group = Vec::new();
        seen[i] = true;
        while let Some(f) = stack.pop() {
            group.push(f);
            for &n in &face_adj[f] {
                if !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        groups.push(group);
    }

    // Split to unique corners so groups can have independent UVs
    split_faces(mesh);
    // After split, face i still maps to indices 3i..3i+3
    for group in &groups {
        // Average normal
        let mut n = Vec3::ZERO;
        for &fi in group {
            let tri = &mesh.indices[fi * 3..fi * 3 + 3];
            let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
            let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
            let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
            n += (b - a).cross(c - a);
        }
        let n = n.normalize_or_zero();
        // Build tangent frame
        let ax = if n.x.abs() < 0.9 {
            n.cross(Vec3::X).normalize_or_zero()
        } else {
            n.cross(Vec3::Y).normalize_or_zero()
        };
        let ay = n.cross(ax).normalize_or_zero();
        for &fi in group {
            let tri = &mesh.indices[fi * 3..fi * 3 + 3];
            for &vi in tri {
                let p = Vec3::from(mesh.vertices[vi as usize].position);
                mesh.vertices[vi as usize].uv = [p.dot(ax), p.dot(ay)];
            }
        }
    }
    normalize_per_face_then_pack(mesh, 0.02);
}

/// Cube Projection: each face → best cube side (Blender allows overlap; we pack after).
pub fn unwrap_cube(mesh: &mut MeshData) {
    if mesh.indices.is_empty() {
        return;
    }
    split_faces(mesh);
    for tri in mesh.indices.chunks_exact(3) {
        let ia = tri[0] as usize;
        let ib = tri[1] as usize;
        let ic = tri[2] as usize;
        let a = Vec3::from(mesh.vertices[ia].position);
        let b = Vec3::from(mesh.vertices[ib].position);
        let c = Vec3::from(mesh.vertices[ic].position);
        let n = (b - a).cross(c - a).normalize_or_zero();
        let an = n.abs();
        let (ax, ay) = if an.z >= an.x && an.z >= an.y {
            (Vec3::X, Vec3::Y)
        } else if an.y >= an.x && an.y >= an.z {
            (Vec3::X, Vec3::Z)
        } else {
            (Vec3::Z, Vec3::Y)
        };
        let flip = if n.dot(ax.cross(ay)) < 0.0 { -1.0 } else { 1.0 };
        for &vi in &[ia, ib, ic] {
            let p = Vec3::from(mesh.vertices[vi].position);
            mesh.vertices[vi].uv = [p.dot(ax), p.dot(ay) * flip];
        }
    }
    normalize_per_face_then_pack(mesh, 0.02);
}

/// Project from average face normal (simple planar).
pub fn unwrap_planar(mesh: &mut MeshData) {
    if mesh.indices.is_empty() {
        return;
    }
    // Keep shared verts; project whole mesh onto best-fit plane from bounds
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        let p = Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    let extent = max - min;
    // Largest plane
    let (ax, ay) = if extent.x <= extent.y && extent.x <= extent.z {
        (Vec3::Y, Vec3::Z)
    } else if extent.y <= extent.x && extent.y <= extent.z {
        (Vec3::X, Vec3::Z)
    } else {
        (Vec3::X, Vec3::Y)
    };
    for v in &mut mesh.vertices {
        let p = Vec3::from(v.position);
        v.uv = [p.dot(ax), p.dot(ay)];
    }
    normalize_uvs(mesh);
}

/// Each face gets its own triangle in UV space (Blender "Reset").
pub fn unwrap_reset(mesh: &mut MeshData) {
    if mesh.indices.is_empty() {
        return;
    }
    split_faces(mesh);
    for tri in mesh.indices.chunks_exact(3) {
        mesh.vertices[tri[0] as usize].uv = [0.0, 0.0];
        mesh.vertices[tri[1] as usize].uv = [1.0, 0.0];
        mesh.vertices[tri[2] as usize].uv = [0.5, 0.866];
    }
    pack_islands(mesh, 0.02);
}

fn normalize_per_face_then_pack(mesh: &mut MeshData, padding: f32) {
    // Preserve relative coordinates across neighboring triangles. Normalizing
    // each triangle independently tears every quad into separate UV pieces.
    // Normalize the projection as a whole, then pack connected UV islands.
    normalize_uvs(mesh);
    pack_islands(mesh, padding);
}

/// Scale/translate all UVs so they fit in 0..1 with a small margin.
pub fn normalize_uvs(mesh: &mut MeshData) {
    if mesh.vertices.is_empty() {
        return;
    }
    let mut min = Vec2::new(f32::MAX, f32::MAX);
    let mut max = Vec2::new(f32::MIN, f32::MIN);
    for v in &mesh.vertices {
        let u = Vec2::from(v.uv);
        min = min.min(u);
        max = max.max(u);
    }
    let size = (max - min).max(Vec2::splat(1e-6));
    let scale = 0.98 / size.x.max(size.y);
    for v in &mut mesh.vertices {
        let u = Vec2::from(v.uv);
        let n = (u - min) * scale + Vec2::splat(0.01);
        v.uv = n.to_array();
    }
}

/// Connected UV islands. UV unwrap duplicates vertices at seams, so index-based
/// adjacency is insufficient. Faces are joined only when a geometric edge and
/// both endpoint UVs match (within a small tolerance).
pub fn uv_islands(mesh: &MeshData) -> Vec<Vec<u32>> {
    use std::collections::HashMap;

    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    struct CornerKey(i32, i32, i32, i32, i32);
    fn key(mesh: &MeshData, vi: u32) -> CornerKey {
        let v = &mesh.vertices[vi as usize];
        CornerKey(
            (v.position[0] * 100_000.0).round() as i32,
            (v.position[1] * 100_000.0).round() as i32,
            (v.position[2] * 100_000.0).round() as i32,
            (v.uv[0] * 100_000.0).round() as i32,
            (v.uv[1] * 100_000.0).round() as i32,
        )
    }

    let count = mesh.vertices.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); count];
    let mut edge_owners: HashMap<(CornerKey, CornerKey), Vec<(u32, u32)>> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            if a as usize >= count || b as usize >= count {
                continue;
            }
            // A face is one connected UV component by itself.
            adj[a as usize].push(b);
            adj[b as usize].push(a);
            let ka = key(mesh, a);
            let kb = key(mesh, b);
            let edge = if ka <= kb { (ka, kb) } else { (kb, ka) };
            edge_owners.entry(edge).or_default().push((a, b));
        }
    }
    // Matching edges connect their corner vertices across split triangle faces.
    for owners in edge_owners.values() {
        if owners.len() < 2 {
            continue;
        }
        for i in 0..owners.len() {
            for j in (i + 1)..owners.len() {
                let (a0, a1) = owners[i];
                let (b0, b1) = owners[j];
                for (a, b) in [(a0, b0), (a0, b1), (a1, b0), (a1, b1)] {
                    if key(mesh, a) == key(mesh, b) {
                        adj[a as usize].push(b);
                        adj[b as usize].push(a);
                    }
                }
            }
        }
    }

    let mut used = vec![false; count];
    for &idx in &mesh.indices {
        if let Some(slot) = used.get_mut(idx as usize) {
            *slot = true;
        }
    }
    let mut seen = vec![false; count];
    let mut islands = Vec::new();
    for i in 0..count {
        if seen[i] {
            continue;
        }
        // Ignore unused vertices; they are not UV faces/islands.
        if !used[i] {
            seen[i] = true;
            continue;
        }
        let mut stack = vec![i as u32];
        let mut island = Vec::new();
        seen[i] = true;
        while let Some(v) = stack.pop() {
            island.push(v);
            for &n in &adj[v as usize] {
                if !seen[n as usize] {
                    seen[n as usize] = true;
                    stack.push(n);
                }
            }
        }
        islands.push(island);
    }
    islands
}

/// Pack islands into the unit square with padding.
pub fn pack_islands(mesh: &mut MeshData, padding: f32) {
    let islands = uv_islands(mesh);
    if islands.is_empty() {
        return;
    }
    struct Isle {
        verts: Vec<u32>,
        min: Vec2,
        size: Vec2,
    }
    let mut isles: Vec<Isle> = islands
        .into_iter()
        .map(|verts| {
            let mut min = Vec2::splat(f32::MAX);
            let mut max = Vec2::splat(f32::MIN);
            for &vi in &verts {
                let u = Vec2::from(mesh.vertices[vi as usize].uv);
                min = min.min(u);
                max = max.max(u);
            }
            let size = (max - min).max(Vec2::splat(1e-6));
            Isle { verts, min, size }
        })
        .collect();
    isles.sort_by(|a, b| {
        (b.size.x * b.size.y)
            .partial_cmp(&(a.size.x * a.size.y))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Estimate a scale so all islands fit roughly in the unit square
    let total_area: f32 = isles.iter().map(|i| i.size.x * i.size.y).sum();
    let global = ((0.85f32) / total_area.max(1e-6).sqrt()).min(2.0);

    let mut x = padding;
    let mut y = padding;
    let mut row_h = 0.0f32;
    let max_w = 1.0 - padding;
    for isle in &isles {
        let s = global;
        let w = isle.size.x * s;
        let h = isle.size.y * s;
        if x + w > max_w && x > padding {
            x = padding;
            y += row_h + padding;
            row_h = 0.0;
        }
        for &vi in &isle.verts {
            let u = Vec2::from(mesh.vertices[vi as usize].uv);
            let local = (u - isle.min) * s + Vec2::new(x, y);
            mesh.vertices[vi as usize].uv = local.to_array();
        }
        x += w + padding;
        row_h = row_h.max(h);
    }
    // If we overflowed 0..1, normalize everything
    let mut max_v = 0.0f32;
    for v in &mesh.vertices {
        max_v = max_v.max(v.uv[0]).max(v.uv[1]);
    }
    if max_v > 1.0 {
        let s = 0.98 / max_v;
        for v in &mut mesh.vertices {
            v.uv[0] = v.uv[0] * s + 0.01;
            v.uv[1] = v.uv[1] * s + 0.01;
        }
    }
}

pub fn uv_translate(mesh: &mut MeshData, selected: &BTreeSet<u32>, delta: Vec2) {
    for &vi in selected {
        if let Some(v) = mesh.vertices.get_mut(vi as usize) {
            v.uv[0] += delta.x;
            v.uv[1] += delta.y;
        }
    }
}

pub fn uv_scale(mesh: &mut MeshData, selected: &BTreeSet<u32>, pivot: Vec2, factor: f32) {
    for &vi in selected {
        if let Some(v) = mesh.vertices.get_mut(vi as usize) {
            let u = Vec2::from(v.uv);
            let n = pivot + (u - pivot) * factor;
            v.uv = n.to_array();
        }
    }
}

pub fn uv_rotate(mesh: &mut MeshData, selected: &BTreeSet<u32>, pivot: Vec2, angle: f32) {
    let (s, c) = angle.sin_cos();
    for &vi in selected {
        if let Some(v) = mesh.vertices.get_mut(vi as usize) {
            let u = Vec2::from(v.uv) - pivot;
            let r = Vec2::new(u.x * c - u.y * s, u.x * s + u.y * c) + pivot;
            v.uv = r.to_array();
        }
    }
}

pub fn uv_selection_center(mesh: &MeshData, selected: &BTreeSet<u32>) -> Vec2 {
    if selected.is_empty() {
        return Vec2::splat(0.5);
    }
    let mut acc = Vec2::ZERO;
    let mut n = 0.0;
    for &vi in selected {
        if let Some(v) = mesh.vertices.get(vi as usize) {
            acc += Vec2::from(v.uv);
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        Vec2::splat(0.5)
    }
}

pub fn pick_uv(mesh: &MeshData, uv: Vec2, max_dist: f32) -> Option<u32> {
    let mut best: Option<(f32, u32)> = None;
    for (i, v) in mesh.vertices.iter().enumerate() {
        let d = (Vec2::from(v.uv) - uv).length();
        if d < max_dist {
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, i as u32));
            }
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;

    fn vertex(position: [f32; 3], uv: [f32; 2]) -> Vertex {
        Vertex {
            position,
            normal: [0.0, 0.0, 1.0],
            uv,
        }
    }

    #[test]
    fn split_quad_triangles_remain_one_uv_island() {
        // Two triangles with duplicated corners and a shared, matching diagonal.
        let mesh = MeshData {
            vertices: vec![
                vertex([0.0, 0.0, 0.0], [0.0, 0.0]),
                vertex([1.0, 0.0, 0.0], [1.0, 0.0]),
                vertex([1.0, 1.0, 0.0], [1.0, 1.0]),
                vertex([0.0, 0.0, 0.0], [0.0, 0.0]),
                vertex([1.0, 1.0, 0.0], [1.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.0, 1.0]),
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
            ..Default::default()
        };
        let islands = uv_islands(&mesh);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].len(), 6);
    }

    #[test]
    fn uv_seam_splits_islands() {
        let mesh = MeshData {
            vertices: vec![
                vertex([0.0, 0.0, 0.0], [0.0, 0.0]),
                vertex([1.0, 0.0, 0.0], [1.0, 0.0]),
                vertex([0.0, 1.0, 0.0], [0.0, 1.0]),
                vertex([0.0, 0.0, 0.0], [0.2, 0.0]),
                vertex([1.0, 0.0, 0.0], [0.8, 0.0]),
                vertex([1.0, -1.0, 0.0], [1.0, -1.0]),
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
            ..Default::default()
        };
        assert_eq!(uv_islands(&mesh).len(), 2);
    }
}
