//! Real armatures: editable bone hierarchy, binding meshes, linear-blend skinning.

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    /// Parent bone index, if any.
    pub parent: Option<u32>,
    /// Rest-pose head (armature local space).
    pub head: [f32; 3],
    /// Rest-pose tail (armature local space).
    pub tail: [f32; 3],
    /// Pose rotation around the bone head (local).
    pub pose_rot: [f32; 4], // quat x,y,z,w
    /// Pose translation of the head (local offset from rest head).
    pub pose_loc: [f32; 3],
}

impl Bone {
    pub fn new(name: impl Into<String>, head: Vec3, tail: Vec3) -> Self {
        Self {
            name: name.into(),
            parent: None,
            head: head.to_array(),
            tail: tail.to_array(),
            pose_rot: [0.0, 0.0, 0.0, 1.0],
            pose_loc: [0.0, 0.0, 0.0],
        }
    }

    pub fn head_v(&self) -> Vec3 {
        Vec3::from(self.head)
    }
    pub fn tail_v(&self) -> Vec3 {
        Vec3::from(self.tail)
    }
    pub fn pose_rot_q(&self) -> Quat {
        Quat::from_xyzw(self.pose_rot[0], self.pose_rot[1], self.pose_rot[2], self.pose_rot[3])
            .normalize()
    }
    pub fn pose_loc_v(&self) -> Vec3 {
        Vec3::from(self.pose_loc)
    }

    pub fn length(&self) -> f32 {
        (self.tail_v() - self.head_v()).length()
    }

    pub fn direction(&self) -> Vec3 {
        (self.tail_v() - self.head_v()).normalize_or_zero()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmatureData {
    pub bones: Vec<Bone>,
    pub selected: BTreeSet<u32>,
    pub active: Option<u32>,
}

impl Default for ArmatureData {
    fn default() -> Self {
        // Default: a simple 3-bone chain along +Y
        let mut a = Self {
            bones: Vec::new(),
            selected: BTreeSet::new(),
            active: None,
        };
        a.add_bone_chain(3, 0.5);
        a
    }
}

impl ArmatureData {
    pub fn add_bone_chain(&mut self, count: u32, len: f32) {
        let count = count.max(1);
        let mut y = 0.0;
        for i in 0..count {
            let head = Vec3::new(0.0, y, 0.0);
            let tail = Vec3::new(0.0, y + len, 0.0);
            let mut b = Bone::new(format!("Bone.{}", i + 1), head, tail);
            if i > 0 {
                b.parent = Some(i - 1);
            }
            self.bones.push(b);
            y += len;
        }
        if let Some(last) = self.bones.len().checked_sub(1) {
            self.active = Some(last as u32);
            self.selected.clear();
            self.selected.insert(last as u32);
        }
    }

    pub fn add_bone(&mut self, parent: Option<u32>) -> u32 {
        let (head, tail) = if let Some(p) = parent.and_then(|i| self.bones.get(i as usize)) {
            let h = p.tail_v();
            let d = p.direction();
            let len = p.length().max(0.1);
            (h, h + d * len)
        } else {
            (Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0))
        };
        let idx = self.bones.len() as u32;
        let mut b = Bone::new(format!("Bone.{}", idx + 1), head, tail);
        b.parent = parent;
        self.bones.push(b);
        self.active = Some(idx);
        self.selected.clear();
        self.selected.insert(idx);
        idx
    }

    /// Extrude: new bone from each selected bone's tail.
    pub fn extrude_selected(&mut self) {
        let sel: Vec<u32> = self.selected.iter().copied().collect();
        if sel.is_empty() {
            if let Some(a) = self.active {
                self.selected.insert(a);
            } else {
                return;
            }
        }
        let sel: Vec<u32> = self.selected.iter().copied().collect();
        self.selected.clear();
        for &pi in &sel {
            let ni = self.add_bone(Some(pi));
            self.selected.insert(ni);
            self.active = Some(ni);
        }
    }

    pub fn delete_selected(&mut self) {
        if self.selected.is_empty() {
            return;
        }
        let kill: BTreeSet<u32> = self.selected.clone();
        // Also remove descendants
        let mut remove = kill.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for (i, b) in self.bones.iter().enumerate() {
                let i = i as u32;
                if remove.contains(&i) {
                    continue;
                }
                if let Some(p) = b.parent {
                    if remove.contains(&p) {
                        remove.insert(i);
                        changed = true;
                    }
                }
            }
        }
        let mut remap = vec![None; self.bones.len()];
        let mut new_bones = Vec::new();
        for (i, mut b) in self.bones.drain(..).enumerate() {
            if remove.contains(&(i as u32)) {
                continue;
            }
            let ni = new_bones.len() as u32;
            remap[i] = Some(ni);
            if let Some(p) = b.parent {
                b.parent = remap.get(p as usize).and_then(|x| *x);
            }
            new_bones.push(b);
        }
        self.bones = new_bones;
        self.selected.clear();
        self.active = self.bones.last().map(|_| (self.bones.len() - 1) as u32);
        if let Some(a) = self.active {
            self.selected.insert(a);
        }
    }

    /// Reset all pose transforms.
    pub fn clear_pose(&mut self) {
        for b in &mut self.bones {
            b.pose_rot = [0.0, 0.0, 0.0, 1.0];
            b.pose_loc = [0.0, 0.0, 0.0];
        }
    }

    /// World-space (armature-local) rest matrix for bone: maps bone-local to armature space.
    /// Bone local: origin at head, +Y toward tail.
    fn rest_matrix(bone: &Bone) -> Mat4 {
        let head = bone.head_v();
        let dir = bone.direction();
        let rot = if dir.length_squared() < 1e-10 {
            Quat::IDENTITY
        } else {
            Quat::from_rotation_arc(Vec3::Y, dir)
        };
        Mat4::from_rotation_translation(rot, head)
    }

    /// Pose matrix for a single bone (ignoring hierarchy for delta — hierarchy applied via heads).
    fn pose_matrix(bone: &Bone) -> Mat4 {
        let head = bone.head_v() + bone.pose_loc_v();
        let dir = bone.direction();
        let rest_rot = if dir.length_squared() < 1e-10 {
            Quat::IDENTITY
        } else {
            Quat::from_rotation_arc(Vec3::Y, dir)
        };
        let rot = rest_rot * bone.pose_rot_q();
        Mat4::from_rotation_translation(rot, head)
    }

    /// Deform matrices (armature local): pose * inverse(rest) for each bone.
    pub fn deform_matrices(&self) -> Vec<Mat4> {
        self.bones
            .iter()
            .map(|b| {
                let rest = Self::rest_matrix(b);
                let pose = Self::pose_matrix(b);
                // Keep children roughly attached: if parent moved, offset by parent delta at rest head
                pose * rest.inverse()
            })
            .collect()
    }

    /// Apply hierarchical pose so child bones follow parents.
    pub fn deform_matrices_hierarchical(&self) -> Vec<Mat4> {
        let n = self.bones.len();
        let mut rest_world = vec![Mat4::IDENTITY; n];
        let mut pose_world = vec![Mat4::IDENTITY; n];
        // Order: parents before children (bones are usually ordered that way; fix with topo)
        let order = self.topo_order();
        for &i in &order {
            let b = &self.bones[i];
            let local_rest = Self::rest_matrix(b);
            let local_pose = Self::pose_matrix(b);
            if let Some(p) = b.parent {
                let pi = p as usize;
                // Child rest is already in armature space (absolute head/tail).
                // For hierarchy: pose inherits parent pose delta.
                let parent_delta = pose_world[pi] * rest_world[pi].inverse();
                rest_world[i] = local_rest;
                // Pose: apply parent delta to rest head, then local pose offset
                let posed_head = parent_delta.transform_point3(b.head_v()) + b.pose_loc_v();
                let dir = b.direction();
                let rest_rot = if dir.length_squared() < 1e-10 {
                    Quat::IDENTITY
                } else {
                    Quat::from_rotation_arc(Vec3::Y, dir)
                };
                let parent_rot = {
                    let (_, r, _) = parent_delta.to_scale_rotation_translation();
                    r
                };
                let rot = parent_rot * rest_rot * b.pose_rot_q();
                pose_world[i] = Mat4::from_rotation_translation(rot, posed_head);
            } else {
                rest_world[i] = local_rest;
                pose_world[i] = local_pose;
            }
        }
        (0..n)
            .map(|i| pose_world[i] * rest_world[i].inverse())
            .collect()
    }

    fn topo_order(&self) -> Vec<usize> {
        let n = self.bones.len();
        let mut indeg = vec![0u32; n];
        for b in &self.bones {
            if let Some(p) = b.parent {
                if (p as usize) < n {
                    indeg[p as usize] = indeg[p as usize]; // parent has no indeg bump
                }
            }
        }
        // Simple: roots first by walking
        let mut order = Vec::new();
        let mut placed = vec![false; n];
        // repeatedly place bones whose parent is placed or none
        while order.len() < n {
            let mut progress = false;
            for i in 0..n {
                if placed[i] {
                    continue;
                }
                let ok = match self.bones[i].parent {
                    None => true,
                    Some(p) => (p as usize) < n && placed[p as usize],
                };
                if ok {
                    order.push(i);
                    placed[i] = true;
                    progress = true;
                }
            }
            if !progress {
                // cycle or invalid parent — push remaining
                for i in 0..n {
                    if !placed[i] {
                        order.push(i);
                    }
                }
                break;
            }
        }
        order
    }

    /// Viewport line segments for bones (armature local).
    pub fn lines(&self) -> Vec<(Vec3, Vec3)> {
        let mut v = Vec::new();
        let deform = self.deform_matrices_hierarchical();
        for (i, b) in self.bones.iter().enumerate() {
            let rest = Self::rest_matrix(b);
            let pose = deform[i] * rest;
            let head = pose.transform_point3(Vec3::ZERO);
            let len = b.length().max(0.05);
            let tip = pose.transform_point3(Vec3::new(0.0, len, 0.0));
            let mid = pose.transform_point3(Vec3::new(0.0, len * 0.25, 0.0));
            let r = len * 0.08;
            let ring = [
                pose.transform_point3(Vec3::new(r, len * 0.25, 0.0)),
                pose.transform_point3(Vec3::new(0.0, len * 0.25, r)),
                pose.transform_point3(Vec3::new(-r, len * 0.25, 0.0)),
                pose.transform_point3(Vec3::new(0.0, len * 0.25, -r)),
            ];
            for i in 0..4 {
                v.push((head, ring[i]));
                v.push((ring[i], tip));
                v.push((ring[i], ring[(i + 1) % 4]));
            }
            let _ = mid;
        }
        v
    }

    /// Pick closest bone to a ray (armature local). Returns bone index.
    pub fn pick_bone(&self, ray_o: Vec3, ray_d: Vec3, thresh: f32) -> Option<u32> {
        let dir = ray_d.normalize_or_zero();
        let deform = self.deform_matrices_hierarchical();
        let mut best: Option<(u32, f32)> = None;
        for (i, b) in self.bones.iter().enumerate() {
            let rest = Self::rest_matrix(b);
            let pose = deform[i] * rest;
            let head = pose.transform_point3(Vec3::ZERO);
            let tip = pose.transform_point3(Vec3::new(0.0, b.length().max(0.05), 0.0));
            if let Some(d) = ray_segment_dist(ray_o, dir, head, tip) {
                if d < thresh && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((i as u32, d));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    pub fn select_bone(&mut self, i: u32, extend: bool) {
        if i as usize >= self.bones.len() {
            return;
        }
        if !extend {
            self.selected.clear();
        }
        if extend && self.selected.contains(&i) {
            self.selected.remove(&i);
        } else {
            self.selected.insert(i);
        }
        self.active = Some(i);
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
    let (t, u) = if det.abs() < 1e-8 {
        let t = ao.dot(rd);
        let u = ((ro + rd * t - a).dot(ab) / ab2).clamp(0.0, 1.0);
        (t, u)
    } else {
        let t = (ao.dot(rd) - ao.dot(ab) * rd_ab / ab2) / det;
        let u = ((ao.dot(ab) - t * rd_ab) / ab2).clamp(0.0, 1.0);
        (t, u)
    };
    if t < 0.0 {
        return None;
    }
    Some((ro + rd * t - (a + ab * u)).length())
}

/// Skin weights binding a mesh to an armature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkinBinding {
    pub armature_id: u64,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub rest_positions: Vec<[f32; 3]>,
    pub rest_normals: Vec<[f32; 3]>,
}

impl SkinBinding {
    /// Automatic weights: each vertex weighted by inverse distance to bone segments.
    pub fn auto_weights(
        armature_id: u64,
        armature: &ArmatureData,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
    ) -> Self {
        let bones = &armature.bones;
        let mut joints = Vec::with_capacity(positions.len());
        let mut weights = Vec::with_capacity(positions.len());
        for p in positions {
            let p = Vec3::from(*p);
            let mut scores: Vec<(u16, f32)> = bones
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let d = dist_point_segment(p, b.head_v(), b.tail_v()).max(1e-4);
                    (i as u16, 1.0 / (d * d))
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(4);
            let sum: f32 = scores.iter().map(|(_, w)| *w).sum::<f32>().max(1e-8);
            let mut j = [0u16; 4];
            let mut w = [0f32; 4];
            for (k, (ji, wi)) in scores.iter().enumerate() {
                j[k] = *ji;
                w[k] = *wi / sum;
            }
            joints.push(j);
            weights.push(w);
        }
        Self {
            armature_id,
            joints,
            weights,
            rest_positions: positions.to_vec(),
            rest_normals: normals.to_vec(),
        }
    }

    /// Apply LBS; returns new positions (and optionally normals).
    pub fn deform(&self, deform_mats: &[Mat4]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut pos = Vec::with_capacity(self.rest_positions.len());
        let mut nrm = Vec::with_capacity(self.rest_normals.len());
        for i in 0..self.rest_positions.len() {
            let p = Vec3::from(self.rest_positions[i]);
            let n = Vec3::from(self.rest_normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]));
            let js = self.joints[i];
            let ws = self.weights[i];
            let mut out = Vec3::ZERO;
            let mut out_n = Vec3::ZERO;
            for k in 0..4 {
                let w = ws[k];
                if w < 1e-8 {
                    continue;
                }
                let bi = js[k] as usize;
                if bi >= deform_mats.len() {
                    continue;
                }
                let m = deform_mats[bi];
                out += m.transform_point3(p) * w;
                out_n += m.transform_vector3(n) * w;
            }
            pos.push(out.to_array());
            nrm.push(out_n.normalize_or_zero().to_array());
        }
        (pos, nrm)
    }
}

fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}
