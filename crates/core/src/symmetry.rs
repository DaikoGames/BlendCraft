//! Mesh Symmetry (Blender manual: Mesh Options → Mirror X/Y/Z).
//! Independent axis flags; transforms mirror exact local-space counterparts.
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Independent symmetry axes — same as Blender's header X/Y/Z toggles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshSymmetry {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl MeshSymmetry {
    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }

    pub fn toggle_x(mut self) -> Self {
        self.x = !self.x;
        self
    }
    pub fn toggle_y(mut self) -> Self {
        self.y = !self.y;
        self
    }
    pub fn toggle_z(mut self) -> Self {
        self.z = !self.z;
        self
    }

    /// All reflection combinations active for this symmetry (1, 3, or 7 mirrors).
    /// Order: single axes, then pairs, then all three — never includes identity.
    pub fn mirror_signs(self) -> Vec<Vec3> {
        let mut axes = Vec::new();
        if self.x {
            axes.push(0);
        }
        if self.y {
            axes.push(1);
        }
        if self.z {
            axes.push(2);
        }
        if axes.is_empty() {
            return Vec::new();
        }
        let n = axes.len();
        let mut out = Vec::new();
        // All non-empty subsets of enabled axes
        for mask in 1..(1 << n) {
            let mut s = Vec3::ONE;
            for (bit, &ax) in axes.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    match ax {
                        0 => s.x = -1.0,
                        1 => s.y = -1.0,
                        2 => s.z = -1.0,
                        _ => {}
                    }
                }
            }
            out.push(s);
        }
        out
    }
}

#[inline]
pub fn apply_signs(p: Vec3, signs: Vec3) -> Vec3 {
    Vec3::new(p.x * signs.x, p.y * signs.y, p.z * signs.z)
}

/// Reflect point for a single axis flag set (X and/or Y and/or Z).
#[inline]
pub fn mirror_point_sym(sym: MeshSymmetry, p: Vec3) -> Vec3 {
    let mut q = p;
    if sym.x {
        q.x = -q.x;
    }
    if sym.y {
        q.y = -q.y;
    }
    if sym.z {
        q.z = -q.z;
    }
    q
}

#[inline]
pub fn mirror_vec_sym(sym: MeshSymmetry, v: Vec3) -> Vec3 {
    mirror_point_sym(sym, v)
}

pub fn mirror_quat_sym(sym: MeshSymmetry, q: Quat) -> Quat {
    // Compose single-axis reflections
    let mut r = q;
    if sym.x {
        r = Quat::from_xyzw(r.x, -r.y, -r.z, r.w).normalize();
    }
    if sym.y {
        r = Quat::from_xyzw(-r.x, r.y, -r.z, r.w).normalize();
    }
    if sym.z {
        r = Quat::from_xyzw(-r.x, -r.y, r.z, r.w).normalize();
    }
    r
}

// ---- Back-compat (single-axis helpers) ----
#[inline]
pub fn mirror_x_point(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}
#[inline]
pub fn mirror_x_vec(v: Vec3) -> Vec3 {
    Vec3::new(-v.x, v.y, v.z)
}
#[inline]
pub fn mirror_x_quat(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.y, -q.z, q.w).normalize()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirrorAxis {
    #[default]
    None,
    X,
    Y,
    Z,
}

impl MirrorAxis {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "Off",
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
    pub fn is_on(self) -> bool {
        !matches!(self, Self::None)
    }
    pub fn cycle(self) -> Self {
        match self {
            Self::None => Self::X,
            Self::X => Self::Y,
            Self::Y => Self::Z,
            Self::Z => Self::None,
        }
    }
    pub fn to_sym(self) -> MeshSymmetry {
        match self {
            Self::None => MeshSymmetry::default(),
            Self::X => MeshSymmetry {
                x: true,
                ..Default::default()
            },
            Self::Y => MeshSymmetry {
                y: true,
                ..Default::default()
            },
            Self::Z => MeshSymmetry {
                z: true,
                ..Default::default()
            },
        }
    }
}

#[inline]
pub fn mirror_point(axis: MirrorAxis, p: Vec3) -> Vec3 {
    mirror_point_sym(axis.to_sym(), p)
}
#[inline]
pub fn mirror_vec(axis: MirrorAxis, v: Vec3) -> Vec3 {
    mirror_vec_sym(axis.to_sym(), v)
}
#[inline]
pub fn mirror_quat(axis: MirrorAxis, q: Quat) -> Quat {
    mirror_quat_sym(axis.to_sym(), q)
}

pub fn mirror_bone_name(name: &str) -> Option<String> {
    const PAIRS: &[(&str, &str)] = &[
        (".L", ".R"),
        (".R", ".L"),
        (".l", ".r"),
        (".r", ".l"),
        ("_L", "_R"),
        ("_R", "_L"),
        ("_l", "_r"),
        ("_r", "_l"),
        (".Left", ".Right"),
        (".Right", ".Left"),
        ("_Left", "_Right"),
        ("_Right", "_Left"),
        (".left", ".right"),
        (".right", ".left"),
    ];
    for &(a, b) in PAIRS {
        if let Some(prefix) = name.strip_suffix(a) {
            return Some(format!("{prefix}{b}"));
        }
    }
    None
}

pub fn find_near_vertex(
    positions: &[[f32; 3]],
    target: Vec3,
    threshold: f32,
    except: Option<usize>,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, p) in positions.iter().enumerate() {
        if except == Some(i) {
            continue;
        }
        let d = Vec3::from(*p).distance(target);
        if d <= threshold && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// Strict pair map for one sign combination (Blender: exact mirrored counterpart).
pub fn build_mirror_map_for_signs(
    positions: &[[f32; 3]],
    threshold: f32,
    signs: Vec3,
) -> Vec<Option<usize>> {
    let n = positions.len();
    let mut map = vec![None; n];
    for i in 0..n {
        let p = Vec3::from(positions[i]);
        let target = apply_signs(p, signs);
        // On the plane for this reflection → maps to self
        if p.distance(target) <= threshold * 0.5 {
            map[i] = Some(i);
            continue;
        }
        if let Some(j) = find_near_vertex(positions, target, threshold, Some(i)) {
            map[i] = Some(j);
        }
    }
    map
}

pub fn build_vertex_mirror_map_axis(
    positions: &[[f32; 3]],
    threshold: f32,
    axis: MirrorAxis,
) -> Vec<Option<usize>> {
    if !axis.is_on() {
        return vec![None; positions.len()];
    }
    let signs = match axis {
        MirrorAxis::X => Vec3::new(-1.0, 1.0, 1.0),
        MirrorAxis::Y => Vec3::new(1.0, -1.0, 1.0),
        MirrorAxis::Z => Vec3::new(1.0, 1.0, -1.0),
        MirrorAxis::None => Vec3::ONE,
    };
    build_mirror_map_for_signs(positions, threshold, signs)
}

pub fn build_vertex_mirror_map(positions: &[[f32; 3]], threshold: f32) -> Vec<Option<usize>> {
    build_vertex_mirror_map_axis(positions, threshold, MirrorAxis::X)
}




/// Returns (selected_vi, u32::MAX, signs) for each selected vert × enabled axis set.
pub fn bake_mirror_pairs(
    positions: &[[f32; 3]],
    selected: &[u32],
    sym: MeshSymmetry,
) -> Vec<(u32, u32, Vec3)> {
    if !sym.any() || selected.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for signs in sym.mirror_signs() {
        for &vi in selected {
            if (vi as usize) >= positions.len() {
                continue;
            }
            let orig = Vec3::from(positions[vi as usize]);
            if orig.distance(apply_signs(orig, signs)) < 1e-6 {
                continue;
            }
            out.push((vi, u32::MAX, signs));
        }
    }
    out
}

/// Apply mesh symmetry using a full mesh snapshot from transform-start.
/// `snap` = all vertex positions at the start of the transform.
/// `edited` = (vi, new_position) for selected verts.
/// `pairs` = from bake_mirror_pairs.
///
/// For face-split meshes (cube with 24 verts), every vertex that shared the
/// mirrored original location is moved to the mirrored *new* location — so a
/// corner does not split into two.
pub fn apply_mirror_pairs_full(
    mesh_positions: &mut [[f32; 3]],
    snap: &[[f32; 3]],
    edited: &[(u32, Vec3)],
    pairs: &[(u32, u32, Vec3)],
) {
    use std::collections::{HashMap, HashSet};
    let new_map: HashMap<u32, Vec3> = edited.iter().copied().collect();
    let selected: HashSet<u32> = new_map.keys().copied().collect();
    let n = mesh_positions.len().min(snap.len());
    let extent = snap
        .iter()
        .map(|p| Vec3::from(*p).length())
        .fold(0.0f32, f32::max)
        .max(1.0);
    let eps = (extent * 1e-4).max(1e-5) * 20.0; // ~coincident cluster

    for &(vi, _, signs) in pairs {
        let Some(new_p) = new_map.get(&vi).copied() else { continue };
        let Some(old_arr) = snap.get(vi as usize) else { continue };
        let old_p = Vec3::from(*old_arr);
        let m_old = apply_signs(old_p, signs);
        let m_new = apply_signs(new_p, signs);
        for j in 0..n {
            if selected.contains(&(j as u32)) {
                continue;
            }
            let p0 = Vec3::from(snap[j]);
            if p0.distance(m_old) <= eps {
                mesh_positions[j] = m_new.to_array();
            }
        }
    }
}

pub fn apply_mirror_pairs(
    _mesh_positions: &mut [[f32; 3]],
    _edited: &[(u32, Vec3)],
    _pairs: &[(u32, u32, Vec3)],
) {
    // Deprecated: use apply_mirror_pairs_full with a full snap.
}
