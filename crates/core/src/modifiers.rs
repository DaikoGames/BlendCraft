//! Classic non-destructive modifier stack (Blender-style).
//!
//! Each modifier is applied in order to the object's base mesh (or the result of
//! the previous modifier). Geometry Nodes, when present, runs after the stack.
//!
//! Supported first set:
//! - Subdivision Surface (simple midpoint subdivision)
//! - Array
//! - Mirror
//! - Solidify
//! - Bevel (edge chamfer approximation)
//! - Boolean (union / difference / intersection on triangle meshes)

use crate::mesh::{MeshData, Vertex};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BooleanOperation {
    Union,
    Difference,
    Intersect,
}

impl BooleanOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Union => "Union",
            Self::Difference => "Difference",
            Self::Intersect => "Intersect",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModifierKind {
    SubdivisionSurface {
        levels: u32,
        render_levels: u32,
    },
    Array {
        count: u32,
        offset: [f32; 3],
        relative: bool,
        /// When true, merge vertices closer than merge_distance.
        merge: bool,
        merge_distance: f32,
    },
    Mirror {
        axis_x: bool,
        axis_y: bool,
        axis_z: bool,
        /// Merge vertices near the mirror plane.
        merge: bool,
        merge_threshold: f32,
    },
    Solidify {
        thickness: f32,
        offset: f32,
        /// Even thickness (approximate).
        even: bool,
    },
    Bevel {
        amount: f32,
        segments: u32,
        /// Limit to angle (radians). Faces whose dihedral is sharper get bevelled.
        angle_limit: f32,
    },
    Boolean {
        operation: BooleanOperation,
        /// Object id of the operand (must be a mesh in the same scene).
        object: Option<u64>,
    },
}

impl ModifierKind {
    pub fn title(&self) -> &'static str {
        match self {
            Self::SubdivisionSurface { .. } => "Subdivision Surface",
            Self::Array { .. } => "Array",
            Self::Mirror { .. } => "Mirror",
            Self::Solidify { .. } => "Solidify",
            Self::Bevel { .. } => "Bevel",
            Self::Boolean { .. } => "Boolean",
        }
    }

    pub fn default_subdivision() -> Self {
        Self::SubdivisionSurface {
            levels: 1,
            render_levels: 2,
        }
    }

    pub fn default_array() -> Self {
        Self::Array {
            count: 2,
            offset: [1.0, 0.0, 0.0],
            relative: true,
            merge: false,
            merge_distance: 0.01,
        }
    }

    pub fn default_mirror() -> Self {
        Self::Mirror {
            axis_x: true,
            axis_y: false,
            axis_z: false,
            merge: true,
            merge_threshold: 0.001,
        }
    }

    pub fn default_solidify() -> Self {
        Self::Solidify {
            thickness: 0.05,
            offset: -1.0,
            even: true,
        }
    }

    pub fn default_bevel() -> Self {
        Self::Bevel {
            amount: 0.05,
            segments: 1,
            angle_limit: 30f32.to_radians(),
        }
    }

    pub fn default_boolean() -> Self {
        Self::Boolean {
            operation: BooleanOperation::Difference,
            object: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Modifier {
    pub name: String,
    pub enabled: bool,
    pub show_viewport: bool,
    pub show_render: bool,
    pub kind: ModifierKind,
}

impl Modifier {
    pub fn new(kind: ModifierKind) -> Self {
        let name = kind.title().to_string();
        Self {
            name,
            enabled: true,
            show_viewport: true,
            show_render: true,
            kind,
        }
    }
}

/// Apply the full modifier stack to a base mesh.
///
/// `operand_meshes` maps object id → world-space mesh for Boolean operands.
/// When `for_render` is true, Subdivision uses `render_levels`.
pub fn evaluate_stack(
    base: &MeshData,
    modifiers: &[Modifier],
    operand_meshes: &HashMap<u64, MeshData>,
    for_render: bool,
) -> MeshData {
    let mut current = base.clone();
    for m in modifiers {
        if !m.enabled {
            continue;
        }
        if for_render && !m.show_render {
            continue;
        }
        if !for_render && !m.show_viewport {
            continue;
        }
        current = apply_one(&current, &m.kind, operand_meshes, for_render);
    }
    current
}

fn apply_one(
    mesh: &MeshData,
    kind: &ModifierKind,
    operands: &HashMap<u64, MeshData>,
    for_render: bool,
) -> MeshData {
    match kind {
        ModifierKind::SubdivisionSurface {
            levels,
            render_levels,
        } => {
            let lv = if for_render {
                (*render_levels).min(6)
            } else {
                (*levels).min(6)
            };
            subdivide_simple(mesh, lv)
        }
        ModifierKind::Array {
            count,
            offset,
            relative,
            merge,
            merge_distance,
        } => apply_array(
            mesh,
            (*count).max(1),
            Vec3::from(*offset),
            *relative,
            *merge,
            *merge_distance,
        ),
        ModifierKind::Mirror {
            axis_x,
            axis_y,
            axis_z,
            merge,
            merge_threshold,
        } => apply_mirror(
            mesh,
            [*axis_x, *axis_y, *axis_z],
            *merge,
            *merge_threshold,
        ),
        ModifierKind::Solidify {
            thickness,
            offset,
            even,
        } => apply_solidify(mesh, *thickness, *offset, *even),
        ModifierKind::Bevel {
            amount,
            segments,
            angle_limit,
        } => apply_bevel(mesh, *amount, (*segments).max(1).min(8), *angle_limit),
        ModifierKind::Boolean { operation, object } => {
            let Some(id) = object else {
                return mesh.clone();
            };
            let Some(other) = operands.get(id) else {
                return mesh.clone();
            };
            apply_boolean(mesh, other, *operation)
        }
    }
}

// ---------------------------------------------------------------------------
// Subdivision (reuses the simple midpoint scheme from geometry_nodes)
// ---------------------------------------------------------------------------

fn subdivide_simple(mesh: &MeshData, levels: u32) -> MeshData {
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current);
    }
    current
}

fn subdivide_once(mesh: &MeshData) -> MeshData {
    let mut vertices = mesh.vertices.clone();
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut mid = |a: u32, b: u32, vertices: &mut Vec<Vertex>| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&m) = edge_mid.get(&key) {
            return m;
        }
        let va = vertices[a as usize];
        let vb = vertices[b as usize];
        let pm = (Vec3::from(va.position) + Vec3::from(vb.position)) * 0.5;
        let nm = (Vec3::from(va.normal) + Vec3::from(vb.normal)).normalize_or_zero();
        let um = [(va.uv[0] + vb.uv[0]) * 0.5, (va.uv[1] + vb.uv[1]) * 0.5];
        let idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: pm.to_array(),
            normal: nm.to_array(),
            uv: um,
        });
        edge_mid.insert(key, idx);
        idx
    };
    let mut indices = Vec::with_capacity(mesh.indices.len() * 4);
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let ab = mid(a, b, &mut vertices);
        let bc = mid(b, c, &mut vertices);
        let ca = mid(c, a, &mut vertices);
        indices.extend_from_slice(&[a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]);
    }
    let mut out = MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: mesh.groups.clone(),
    };
    out.compute_normals();
    out
}

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

fn apply_array(
    mesh: &MeshData,
    count: u32,
    offset: Vec3,
    relative: bool,
    merge: bool,
    merge_distance: f32,
) -> MeshData {
    if count <= 1 || mesh.vertices.is_empty() {
        return mesh.clone();
    }

    let step = if relative {
        // Relative offset is in units of the bounding-box size along each axis.
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let size = (max - min).max(Vec3::splat(1e-6));
        offset * size
    } else {
        offset
    };

    let mut vertices = Vec::with_capacity(mesh.vertices.len() * count as usize);
    let mut indices = Vec::with_capacity(mesh.indices.len() * count as usize);

    for i in 0..count {
        let t = step * (i as f32);
        let base = vertices.len() as u32;
        for v in &mesh.vertices {
            let mut nv = *v;
            let p = Vec3::from(v.position) + t;
            nv.position = p.to_array();
            vertices.push(nv);
        }
        for &idx in &mesh.indices {
            indices.push(base + idx);
        }
    }

    let mut out = MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: vec![],
    };
    if merge && merge_distance > 0.0 {
        out = merge_by_distance(&out, merge_distance);
    }
    out.compute_normals();
    out
}

// ---------------------------------------------------------------------------
// Mirror
// ---------------------------------------------------------------------------

fn apply_mirror(
    mesh: &MeshData,
    axes: [bool; 3],
    merge: bool,
    threshold: f32,
) -> MeshData {
    if !axes[0] && !axes[1] && !axes[2] {
        return mesh.clone();
    }

    let mut vertices = mesh.vertices.clone();
    let mut indices = mesh.indices.clone();

    // Generate one mirrored copy per enabled axis (and combinations for multi-axis).
    // We mirror sequentially so X then Y produces the quadrant copy as well.
    let mut sources: Vec<(Vec<Vertex>, Vec<u32>)> = vec![(vertices.clone(), indices.clone())];

    for (axis, enabled) in axes.iter().enumerate() {
        if !enabled {
            continue;
        }
        let mut next = Vec::new();
        for (verts, inds) in &sources {
            // original
            next.push((verts.clone(), inds.clone()));
            // mirrored
            let mut mverts = verts.clone();
            for v in &mut mverts {
                v.position[axis] = -v.position[axis];
                v.normal[axis] = -v.normal[axis];
            }
            // Flip winding so normals stay outward after reflection.
            let mut minds = inds.clone();
            for tri in minds.chunks_exact_mut(3) {
                tri.swap(1, 2);
            }
            next.push((mverts, minds));
        }
        sources = next;
    }

    vertices.clear();
    indices.clear();
    for (verts, inds) in sources {
        let base = vertices.len() as u32;
        vertices.extend(verts);
        for i in inds {
            indices.push(base + i);
        }
    }

    let mut out = MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: vec![],
    };
    if merge && threshold > 0.0 {
        out = merge_by_distance(&out, threshold);
    }
    out.compute_normals();
    out
}

// ---------------------------------------------------------------------------
// Solidify
// ---------------------------------------------------------------------------

fn apply_solidify(mesh: &MeshData, thickness: f32, offset: f32, _even: bool) -> MeshData {
    if mesh.vertices.is_empty() || thickness.abs() < 1e-8 {
        return mesh.clone();
    }

    // offset: -1 = inward, 0 = centered, +1 = outward (Blender convention)
    let outer_shift = thickness * (0.5 + 0.5 * offset);
    let inner_shift = thickness * (-0.5 + 0.5 * offset);

    let n = mesh.vertices.len() as u32;
    let mut vertices = Vec::with_capacity(mesh.vertices.len() * 2);
    // Outer shell
    for v in &mesh.vertices {
        let nrm = Vec3::from(v.normal).normalize_or_zero();
        let mut nv = *v;
        nv.position = (Vec3::from(v.position) + nrm * outer_shift).to_array();
        vertices.push(nv);
    }
    // Inner shell (normals flipped later via winding)
    for v in &mesh.vertices {
        let nrm = Vec3::from(v.normal).normalize_or_zero();
        let mut nv = *v;
        nv.position = (Vec3::from(v.position) + nrm * inner_shift).to_array();
        nv.normal = (-nrm).to_array();
        vertices.push(nv);
    }

    let mut indices = Vec::with_capacity(mesh.indices.len() * 2 + mesh.indices.len() * 2);
    // Outer faces (same winding)
    for &i in &mesh.indices {
        indices.push(i);
    }
    // Inner faces (flipped winding)
    for tri in mesh.indices.chunks_exact(3) {
        indices.extend_from_slice(&[n + tri[0], n + tri[2], n + tri[1]]);
    }

    // Side walls: for every original edge, emit a quad (two triangles).
    // Build edge → opposite vertex map from the original mesh.
    let mut edge_faces: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_faces
                .entry(key)
                .and_modify(|cnt| *cnt += 1)
                .or_insert(1);
        }
    }
    // Boundary edges (appear once) get side walls.
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            if edge_faces.get(&key).copied().unwrap_or(0) == 1 {
                // Quad: outer u,v → inner v,u
                indices.extend_from_slice(&[u, v, n + v, u, n + v, n + u]);
            }
        }
    }

    let mut out = MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: vec![],
    };
    out.compute_normals();
    out
}

// ---------------------------------------------------------------------------
// Bevel (angle-limited edge offset – simple chamfer)
// ---------------------------------------------------------------------------

fn apply_bevel(mesh: &MeshData, amount: f32, segments: u32, angle_limit: f32) -> MeshData {
    if amount <= 1e-8 || mesh.indices.is_empty() {
        return mesh.clone();
    }
    // Very simple approximation: push vertices along the average of adjacent
    // face normals for edges whose dihedral angle is sharper than the limit.
    // Multi-segment bevel is approximated by repeated smaller offsets.
    let step = amount / segments as f32;
    let mut current = mesh.clone();
    for _ in 0..segments {
        current = bevel_once(&current, step, angle_limit);
    }
    current
}

fn bevel_once(mesh: &MeshData, amount: f32, angle_limit: f32) -> MeshData {
    let cos_limit = angle_limit.cos();
    // Face normals
    let mut face_nrm = Vec::with_capacity(mesh.indices.len() / 3);
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
        face_nrm.push((b - a).cross(c - a).normalize_or_zero());
    }
    // Per-vertex accumulated offset direction
    let mut offset = vec![Vec3::ZERO; mesh.vertices.len()];
    let mut weight = vec![0.0f32; mesh.vertices.len()];

    // For each edge, if the two adjacent faces form a sharp angle, push the
    // endpoints outward along the average of the two face normals.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_faces.entry(key).or_default().push(fi);
        }
    }
    for ((u, v), faces) in &edge_faces {
        if faces.len() != 2 {
            continue;
        }
        let n0 = face_nrm[faces[0]];
        let n1 = face_nrm[faces[1]];
        let d = n0.dot(n1);
        if d > cos_limit {
            continue; // too flat
        }
        let dir = (n0 + n1).normalize_or_zero();
        if dir.length_squared() < 1e-8 {
            continue;
        }
        for &vi in &[*u, *v] {
            offset[vi as usize] += dir;
            weight[vi as usize] += 1.0;
        }
    }

    let mut vertices = mesh.vertices.clone();
    for (i, v) in vertices.iter_mut().enumerate() {
        if weight[i] > 0.0 {
            let d = offset[i] / weight[i];
            let p = Vec3::from(v.position) + d.normalize_or_zero() * amount;
            v.position = p.to_array();
        }
    }
    let mut out = MeshData {
        name: mesh.name.clone(),
        vertices,
        indices: mesh.indices.clone(),
        groups: mesh.groups.clone(),
    };
    out.compute_normals();
    out
}

// ---------------------------------------------------------------------------
// Boolean (AABB-based CSG approximation – practical starter)
// ---------------------------------------------------------------------------

fn apply_boolean(a: &MeshData, b: &MeshData, op: BooleanOperation) -> MeshData {
    // Practical triangle-level boolean using containment tests.
    // Not a full robust CSG (no new intersection vertices), but useful for
    // many hard-surface cases and matches the "starter" quality of other systems.
    if a.indices.is_empty() {
        return match op {
            BooleanOperation::Union | BooleanOperation::Difference => MeshData::default(),
            BooleanOperation::Intersect => MeshData::default(),
        };
    }
    if b.indices.is_empty() {
        return match op {
            BooleanOperation::Union | BooleanOperation::Difference => a.clone(),
            BooleanOperation::Intersect => MeshData::default(),
        };
    }

    let inside_b = |p: Vec3| -> bool { point_in_mesh(p, b) };
    let inside_a = |p: Vec3| -> bool { point_in_mesh(p, a) };

    let keep_tri_a = |tri: &[u32]| -> bool {
        let c = centroid(a, tri);
        match op {
            BooleanOperation::Union => !inside_b(c),
            BooleanOperation::Difference => !inside_b(c),
            BooleanOperation::Intersect => inside_b(c),
        }
    };
    let keep_tri_b = |tri: &[u32]| -> bool {
        let c = centroid(b, tri);
        match op {
            BooleanOperation::Union => !inside_a(c),
            BooleanOperation::Difference => false, // never keep B in difference
            BooleanOperation::Intersect => inside_a(c),
        }
    };

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let append = |src: &MeshData,
                  tri: &[u32],
                  flip: bool,
                  vertices: &mut Vec<Vertex>,
                  indices: &mut Vec<u32>| {
        let base = vertices.len() as u32;
        for &i in tri {
            vertices.push(src.vertices[i as usize]);
        }
        if flip {
            indices.extend_from_slice(&[base, base + 2, base + 1]);
        } else {
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    };

    for tri in a.indices.chunks_exact(3) {
        if keep_tri_a(tri) {
            append(a, tri, false, &mut vertices, &mut indices);
        }
    }
    for tri in b.indices.chunks_exact(3) {
        if keep_tri_b(tri) {
            // For union/intersect the B contribution keeps original winding;
            // difference never takes B.
            append(b, tri, false, &mut vertices, &mut indices);
        }
    }

    let mut out = MeshData {
        name: a.name.clone(),
        vertices,
        indices,
        groups: vec![],
    };
    out.compute_normals();
    out
}

fn centroid(mesh: &MeshData, tri: &[u32]) -> Vec3 {
    let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
    let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
    let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
    (a + b + c) / 3.0
}

/// Ray-cast point-in-mesh (odd crossings along +X). Assumes roughly closed mesh.
fn point_in_mesh(p: Vec3, mesh: &MeshData) -> bool {
    let mut crossings = 0u32;
    let origin = p;
    let dir = Vec3::X;
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let b = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let c = Vec3::from(mesh.vertices[tri[2] as usize].position);
        if ray_triangle(origin, dir, a, b, c) {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

fn ray_triangle(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    const EPS: f32 = 1e-6;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a > -EPS && a < EPS {
        return false;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = f * e2.dot(q);
    t > EPS
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn merge_by_distance(mesh: &MeshData, dist: f32) -> MeshData {
    if mesh.vertices.is_empty() || dist <= 0.0 {
        return mesh.clone();
    }
    let dist2 = dist * dist;
    let mut map = vec![0u32; mesh.vertices.len()];
    let mut unique: Vec<Vertex> = Vec::new();
    'outer: for (i, v) in mesh.vertices.iter().enumerate() {
        let p = Vec3::from(v.position);
        for (j, u) in unique.iter().enumerate() {
            if (p - Vec3::from(u.position)).length_squared() <= dist2 {
                map[i] = j as u32;
                continue 'outer;
            }
        }
        map[i] = unique.len() as u32;
        unique.push(*v);
    }
    let mut indices = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks_exact(3) {
        let a = map[tri[0] as usize];
        let b = map[tri[1] as usize];
        let c = map[tri[2] as usize];
        if a != b && b != c && c != a {
            indices.extend_from_slice(&[a, b, c]);
        }
    }
    MeshData {
        name: mesh.name.clone(),
        vertices: unique,
        indices,
        groups: vec![],
    }
}

/// Convenience: evaluate modifiers for an object given a scene lookup for Boolean operands.
pub fn evaluate_object_modifiers(
    base: &MeshData,
    modifiers: &[Modifier],
    scene_objects: &[(u64, &MeshData, Mat4)],
    for_render: bool,
) -> MeshData {
    let mut operands = HashMap::new();
    for (id, mesh, world) in scene_objects {
        // Only include meshes that are referenced by a Boolean modifier.
        let needed = modifiers.iter().any(|m| {
            matches!(
                &m.kind,
                ModifierKind::Boolean {
                    object: Some(oid),
                    ..
                } if oid == id
            )
        });
        if needed {
            operands.insert(*id, mesh.transformed(*world));
        }
    }
    evaluate_stack(base, modifiers, &operands, for_render)
}
