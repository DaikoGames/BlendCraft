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
    /// Pose scale in bone-local axes. Defaults to identity for older project files.
    #[serde(default = "default_pose_scale")]
    pub pose_scale: [f32; 3],
}

fn default_pose_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
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
            pose_scale: [1.0, 1.0, 1.0],
        }
    }

    pub fn head_v(&self) -> Vec3 {
        Vec3::from(self.head)
    }
    pub fn tail_v(&self) -> Vec3 {
        Vec3::from(self.tail)
    }
    pub fn pose_rot_q(&self) -> Quat {
        Quat::from_xyzw(
            self.pose_rot[0],
            self.pose_rot[1],
            self.pose_rot[2],
            self.pose_rot[3],
        )
        .normalize()
    }
    pub fn pose_loc_v(&self) -> Vec3 {
        Vec3::from(self.pose_loc)
    }

    pub fn pose_scale_v(&self) -> Vec3 {
        Vec3::from(self.pose_scale)
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

    /// Create a connected child bone from either endpoint of an existing bone.
    /// `from_head=true` creates a branch from the parent's head; otherwise it
    /// continues from the parent's tail. The new bone becomes the active selection.
    pub fn extrude_from_joint(&mut self, parent: u32, from_head: bool) -> Option<u32> {
        let source = self.bones.get(parent as usize)?.clone();
        let head = if from_head {
            source.head_v()
        } else {
            source.tail_v()
        };
        let mut direction = source.direction();
        if from_head {
            direction = -direction;
        }
        if direction.length_squared() < 1e-8 {
            direction = Vec3::Y;
        }
        let length = source.length().max(0.25);
        let index = self.bones.len() as u32;
        let mut bone = Bone::new(
            format!("Bone.{}", index + 1),
            head,
            head + direction * length,
        );
        bone.parent = Some(parent);
        self.bones.push(bone);
        self.selected.clear();
        self.selected.insert(index);
        self.active = Some(index);
        Some(index)
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

    pub fn delete_selected(&mut self) -> Vec<Option<u32>> {
        if self.selected.is_empty() {
            return (0..self.bones.len()).map(|i| Some(i as u32)).collect();
        }
        let kill: BTreeSet<u32> = self.selected.clone();
        // Also remove descendants so no surviving bone keeps an invalid parent.
        let mut remove = kill;
        let mut changed = true;
        while changed {
            changed = false;
            for (i, b) in self.bones.iter().enumerate() {
                let i = i as u32;
                if remove.contains(&i) {
                    continue;
                }
                if b.parent.map_or(false, |p| remove.contains(&p)) {
                    remove.insert(i);
                    changed = true;
                }
            }
        }

        // Build the complete old -> new index map before remapping parents.
        let mut remap = vec![None; self.bones.len()];
        let mut next = 0u32;
        for i in 0..self.bones.len() {
            if !remove.contains(&(i as u32)) {
                remap[i] = Some(next);
                next += 1;
            }
        }
        let old_bones = std::mem::take(&mut self.bones);
        let mut new_bones = Vec::with_capacity(next as usize);
        for (i, mut b) in old_bones.into_iter().enumerate() {
            if remap[i].is_none() {
                continue;
            }
            b.parent = b
                .parent
                .and_then(|p| remap.get(p as usize).and_then(|mapped| *mapped));
            new_bones.push(b);
        }
        self.bones = new_bones;
        self.selected.clear();
        self.active = self.bones.last().map(|_| (self.bones.len() - 1) as u32);
        if let Some(a) = self.active {
            self.selected.insert(a);
        }
        remap
    }

    /// Reset pose transforms on the selected bones (or the active bone if selection is empty).
    pub fn clear_pose(&mut self) {
        let mut indices: Vec<u32> = self.selected.iter().copied().collect();
        if indices.is_empty() {
            if let Some(active) = self.active {
                indices.push(active);
            }
        }
        for i in indices {
            if let Some(b) = self.bones.get_mut(i as usize) {
                b.pose_rot = [0.0, 0.0, 0.0, 1.0];
                b.pose_loc = [0.0, 0.0, 0.0];
                b.pose_scale = [1.0, 1.0, 1.0];
            }
        }
    }

    /// Keep joints that were coincident in the original rest pose connected while
    /// editing bones. The selected side drives a shared joint: moving a parent
    /// moves its connected child head; moving a child moves the matching parent
    /// endpoint. This deliberately uses the original endpoint positions so the
    /// relationship remains stable throughout a modal drag.
    pub fn propagate_connected_rest_joints(
        &mut self,
        original: &[(Vec3, Vec3)],
        selected: &BTreeSet<u32>,
    ) {
        let order = self.topo_order();
        for child_idx in order {
            let Some(parent_idx) = self.bones[child_idx].parent.map(|p| p as usize) else {
                continue;
            };
            if parent_idx >= self.bones.len()
                || child_idx >= original.len()
                || parent_idx >= original.len()
            {
                continue;
            }

            let old_child_head = original[child_idx].0;
            let old_parent_head = original[parent_idx].0;
            let old_parent_tail = original[parent_idx].1;
            let parent_uses_tail = old_child_head.distance_squared(old_parent_tail) <= 1.0e-8;
            let parent_uses_head = old_child_head.distance_squared(old_parent_head) <= 1.0e-8;
            if !parent_uses_tail && !parent_uses_head {
                // Parented but deliberately unconnected bones retain their offset.
                continue;
            }

            let parent_selected = selected.contains(&(parent_idx as u32));
            let child_selected = selected.contains(&(child_idx as u32));
            let endpoint = if parent_uses_tail { 1 } else { 0 };

            if child_selected && !parent_selected {
                // The child is the edited side: move the parent's matching joint
                // to the child's new head. Other children connected there will
                // follow when their turn is processed.
                let child_head = self.bones[child_idx].head_v();
                if endpoint == 1 {
                    self.bones[parent_idx].tail = child_head.to_array();
                } else {
                    self.bones[parent_idx].head = child_head.to_array();
                }
            } else {
                // The parent drives the joint (also the deterministic choice when
                // both bones are selected). Translate the child as a whole so its
                // length and direction are preserved.
                let joint = if endpoint == 1 {
                    self.bones[parent_idx].tail_v()
                } else {
                    self.bones[parent_idx].head_v()
                };
                let old_head = self.bones[child_idx].head_v();
                let delta = joint - old_head;
                if delta.length_squared() > 1.0e-14 {
                    let new_tail = self.bones[child_idx].tail_v() + delta;
                    self.bones[child_idx].head = joint.to_array();
                    self.bones[child_idx].tail = new_tail.to_array();
                }
            }
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
        Mat4::from_scale_rotation_translation(bone.pose_scale_v(), rot, head)
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
                let (parent_scale, parent_rot, _) = parent_delta.to_scale_rotation_translation();
                let rot = parent_rot * rest_rot * b.pose_rot_q();
                pose_world[i] = Mat4::from_scale_rotation_translation(
                    parent_scale * b.pose_scale_v(),
                    rot,
                    posed_head,
                );
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

    /// Indexed viewport line segments for each bone (armature local).
    /// `posed=false` displays rest geometry for Armature Edit Mode.
    pub fn indexed_lines(&self, posed: bool) -> Vec<(u32, Vec3, Vec3)> {
        let mut v = Vec::new();
        let deform = if posed {
            Some(self.deform_matrices_hierarchical())
        } else {
            None
        };
        for (i, b) in self.bones.iter().enumerate() {
            let pose = if let Some(mats) = deform.as_ref() {
                mats[i] * Self::rest_matrix(b)
            } else {
                Self::rest_matrix(b)
            };
            let head = pose.transform_point3(Vec3::ZERO);
            let len = b.length().max(0.05);
            let tip = pose.transform_point3(Vec3::new(0.0, len, 0.0));
            let r = len * 0.08;
            let ring = [
                pose.transform_point3(Vec3::new(r, len * 0.25, 0.0)),
                pose.transform_point3(Vec3::new(0.0, len * 0.25, r)),
                pose.transform_point3(Vec3::new(-r, len * 0.25, 0.0)),
                pose.transform_point3(Vec3::new(0.0, len * 0.25, -r)),
            ];
            for j in 0..4 {
                v.push((i as u32, head, ring[j]));
                v.push((i as u32, ring[j], tip));
                v.push((i as u32, ring[j], ring[(j + 1) % 4]));
            }
        }
        v
    }

    /// Viewport line segments for bones (armature local, posed).
    pub fn lines(&self) -> Vec<(Vec3, Vec3)> {
        self.indexed_lines(true)
            .into_iter()
            .map(|(_, a, b)| (a, b))
            .collect()
    }

    /// Pick against rest-pose bone segments (Armature Edit Mode).
    pub fn pick_bone_rest(&self, ray_o: Vec3, ray_d: Vec3, thresh: f32) -> Option<u32> {
        let dir = ray_d.normalize_or_zero();
        let mut best: Option<(u32, f32)> = None;
        for (i, b) in self.bones.iter().enumerate() {
            if let Some(d) = ray_segment_dist(ray_o, dir, b.head_v(), b.tail_v()) {
                if d < thresh && best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((i as u32, d));
                }
            }
        }
        best.map(|(i, _)| i)
    }

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
            if self.active == Some(i) {
                self.active = self.selected.iter().next_back().copied();
            }
        } else {
            self.selected.insert(i);
            self.active = Some(i);
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
        if deform_mats.is_empty() {
            return (self.rest_positions.clone(), self.rest_normals.clone());
        }
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

#[cfg(test)]
mod connected_joint_tests {
    use super::{ArmatureData, Bone};
    use glam::Vec3;
    use std::collections::BTreeSet;

    fn connected_pair() -> ArmatureData {
        let mut arm = ArmatureData {
            bones: vec![
                Bone::new("Parent", Vec3::ZERO, Vec3::Y),
                Bone::new("Child", Vec3::Y, Vec3::Y * 2.0),
            ],
            selected: BTreeSet::new(),
            active: Some(0),
        };
        arm.bones[1].parent = Some(0);
        arm
    }

    #[test]
    fn moving_parent_moves_connected_child_joint_and_bone() {
        let mut arm = connected_pair();
        let original: Vec<_> = arm.bones.iter().map(|b| (b.head_v(), b.tail_v())).collect();
        arm.bones[0].head = (arm.bones[0].head_v() + Vec3::X).to_array();
        arm.bones[0].tail = (arm.bones[0].tail_v() + Vec3::X).to_array();
        arm.propagate_connected_rest_joints(&original, &BTreeSet::from([0]));
        assert_eq!(arm.bones[1].head_v(), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(arm.bones[1].tail_v(), Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn moving_connected_child_moves_parent_tip() {
        let mut arm = connected_pair();
        let original: Vec<_> = arm.bones.iter().map(|b| (b.head_v(), b.tail_v())).collect();
        arm.bones[1].head = (arm.bones[1].head_v() + Vec3::X).to_array();
        arm.bones[1].tail = (arm.bones[1].tail_v() + Vec3::X).to_array();
        arm.propagate_connected_rest_joints(&original, &BTreeSet::from([1]));
        assert_eq!(arm.bones[0].tail_v(), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(arm.bones[1].head_v(), arm.bones[0].tail_v());
    }
}
