use crate::constraints::Constraint;
use crate::armature::SkinBinding;
use crate::kinds::{helper_radius, ObjectKind};
use crate::material::{Material, TextureData};
use crate::mesh::{Aabb, MeshData};
use crate::primitives::Primitive;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Collection {
    pub id: u64,
    pub name: String,
    pub visible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    /// Stable identity (0 = not assigned yet). Parents, constraints and selection refer to ids.
    pub id: u64,
    pub name: String,
    /// Local transform (relative to the parent, if any).
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub visible: bool,
    /// Shared (cheap to clone for undo snapshots).
    pub mesh: Arc<MeshData>,
    /// Index into `Scene::materials`; None = default material. Ignored when the mesh has groups.
    pub material: Option<usize>,
    /// Set for parametric objects so their parameters stay editable.
    pub primitive: Option<Primitive>,
    pub kind: ObjectKind,
    pub parent: Option<u64>,
    pub collection: Option<u64>,
    /// Objects with the same link group share primitive parameters (linked duplicates).
    pub link_group: Option<u64>,
    pub constraints: Vec<Constraint>,
    /// Optional armature skinning binding (mesh objects only).
    pub skin: Option<SkinBinding>,
}

impl Object {
    pub fn new(name: impl Into<String>, mesh: MeshData) -> Self {
        Self {
            id: 0,
            name: name.into(),
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            mesh: Arc::new(mesh),
            material: None,
            primitive: None,
            kind: ObjectKind::Mesh,
            parent: None,
            collection: None,
            link_group: None,
            constraints: Vec::new(),
            skin: None,
        }
    }

    pub fn from_matrix(name: impl Into<String>, m: Mat4, mesh: Arc<MeshData>) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        let mut o = Self::new(name, MeshData::default());
        o.mesh = mesh;
        o.scale = scale;
        o.rotation = rotation;
        o.translation = translation;
        o
    }

    /// Local matrix (relative to the parent).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub materials: Vec<Material>,
    pub textures: Arc<Vec<TextureData>>,
    pub collections: Vec<Collection>,
    /// The 3D cursor: new objects appear here, can be used as a pivot / snap target.
    pub cursor: Vec3,
    pub next_id: u64,
}

struct Eval<'a> {
    s: &'a Scene,
    ids: HashMap<u64, usize>,
    out: Vec<Option<Mat4>>,
    state: Vec<u8>,
}

impl<'a> Eval<'a> {
    fn eval(&mut self, i: usize) -> Mat4 {
        if let Some(m) = self.out[i] {
            return m;
        }
        let s: &'a Scene = self.s;
        let o = &s.objects[i];
        if self.state[i] == 1 {
            return o.matrix(); // dependency loop: fall back to the local matrix
        }
        self.state[i] = 1;
        let mut m = o.matrix();
        if let Some(pid) = o.parent {
            if let Some(&pi) = self.ids.get(&pid) {
                if pi != i {
                    m = self.eval(pi) * m;
                }
            }
        }
        for c in &o.constraints {
            if !c.enabled {
                continue;
            }
            let tw = c
                .kind
                .target()
                .and_then(|t| self.ids.get(&t).copied())
                .filter(|&ti| ti != i)
                .map(|ti| self.eval(ti));
            m = c.apply(m, tw);
        }
        self.out[i] = Some(m);
        self.state[i] = 2;
        m
    }
}

impl Scene {
    pub fn alloc_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.objects.iter().position(|o| o.id == id)
    }

    /// Give every object a unique non-zero id.
    pub fn fix_ids(&mut self) {
        let max = self.objects.iter().map(|o| o.id).chain(self.collections.iter().map(|c| c.id)).max().unwrap_or(0);
        self.next_id = self.next_id.max(max);
        let mut used: HashSet<u64> = HashSet::new();
        for o in self.objects.iter_mut() {
            if o.id == 0 || !used.insert(o.id) {
                self.next_id += 1;
                o.id = self.next_id;
                used.insert(o.id);
            }
        }
    }

    /// World matrix of every object (parents and constraints applied).
    pub fn world_matrices(&self) -> Vec<Mat4> {
        let n = self.objects.len();
        let mut e = Eval {
            s: self,
            ids: self.objects.iter().enumerate().map(|(i, o)| (o.id, i)).collect(),
            out: vec![None; n],
            state: vec![0; n],
        };
        (0..n).map(|i| e.eval(i)).collect()
    }

    pub fn parent_world(&self, i: usize, worlds: &[Mat4]) -> Mat4 {
        self.objects[i]
            .parent
            .and_then(|p| self.index_of(p))
            .filter(|&pi| pi != i)
            .map_or(Mat4::IDENTITY, |pi| worlds[pi])
    }

    /// Set the local transform so the object ends up with `world` (given the current parent).
    pub fn set_world(&mut self, i: usize, world: Mat4, worlds: &[Mat4]) {
        let pw = self.parent_world(i, worlds);
        let local = if pw.determinant().abs() < 1e-12 { world } else { pw.inverse() * world };
        let (s, r, t) = local.to_scale_rotation_translation();
        let o = &mut self.objects[i];
        o.scale = s;
        o.rotation = r;
        o.translation = t;
    }

    /// Visible = object not hidden and its collection not hidden.
    pub fn is_visible(&self, i: usize) -> bool {
        let Some(o) = self.objects.get(i) else { return false };
        if !o.visible {
            return false;
        }
        match o.collection {
            None => true,
            Some(c) => self.collections.iter().find(|x| x.id == c).map_or(true, |x| x.visible),
        }
    }

    pub fn children_of(&self, id: u64) -> Vec<usize> {
        self.objects.iter().enumerate().filter(|(_, o)| o.parent == Some(id)).map(|(i, _)| i).collect()
    }

    /// Is object `anc` an ancestor of object `i`?
    pub fn is_ancestor(&self, anc: usize, i: usize) -> bool {
        let anc_id = self.objects[anc].id;
        let mut cur = self.objects[i].parent;
        for _ in 0..self.objects.len() + 1 {
            match cur {
                Some(p) if p == anc_id => return true,
                Some(p) => cur = self.index_of(p).and_then(|pi| self.objects[pi].parent),
                None => return false,
            }
        }
        false
    }

    /// Subset of `sel` whose ancestors are not in `sel` (moving these moves everything).
    pub fn roots(&self, sel: &[usize]) -> Vec<usize> {
        sel.iter().copied().filter(|&i| !sel.iter().any(|&a| a != i && self.is_ancestor(a, i))).collect()
    }

    pub fn object_world_bounds(&self, i: usize, world: &Mat4) -> Option<Aabb> {
        let o = &self.objects[i];
        if !o.kind.is_mesh() {
            let r = Vec3::splat(helper_radius(o, world));
            let t = world.w_axis.truncate();
            return Some(Aabb { min: t - r, max: t + r });
        }
        let b = o.mesh.bounds()?;
        let mut out: Option<Aabb> = None;
        for k in 0..8 {
            let c = Vec3::new(
                if k & 1 == 0 { b.min.x } else { b.max.x },
                if k & 2 == 0 { b.min.y } else { b.max.y },
                if k & 4 == 0 { b.min.z } else { b.max.z },
            );
            let p = world.transform_point3(c);
            let pb = Aabb { min: p, max: p };
            out = Some(out.map_or(pb, |a| a.union(pb)));
        }
        out
    }

    pub fn bounds(&self) -> Option<Aabb> {
        let w = self.world_matrices();
        (0..self.objects.len()).filter(|&i| self.is_visible(i)).filter_map(|i| self.object_world_bounds(i, &w[i])).reduce(Aabb::union)
    }

    /// Nearest visible object hit by a world-space ray, with the ray parameter.
    pub fn pick_hit(&self, origin: Vec3, dir: Vec3) -> Option<(usize, f32)> {
        let worlds = self.world_matrices();
        let mut best: Option<(usize, f32)> = None;
        for (i, o) in self.objects.iter().enumerate() {
            if !self.is_visible(i) {
                continue;
            }
            let m = worlds[i];
            if !o.kind.is_mesh() {
                if let Some(t) = ray_sphere(origin, dir, m.w_axis.truncate(), helper_radius(o, &m)) {
                    if best.map_or(true, |(_, bt)| t < bt) {
                        best = Some((i, t));
                    }
                }
                continue;
            }
            if m.determinant().abs() < 1e-12 {
                continue;
            }
            let inv = m.inverse();
            let ro = inv.transform_point3(origin);
            let rd = inv.transform_vector3(dir);
            if let Some(b) = o.mesh.bounds() {
                if !ray_aabb(ro, rd, b) {
                    continue;
                }
            }
            let verts = &o.mesh.vertices;
            for t in o.mesh.indices.chunks_exact(3) {
                let p = |k: u32| Vec3::from(verts[k as usize].position);
                if let Some(h) = ray_tri(ro, rd, p(t[0]), p(t[1]), p(t[2])) {
                    if best.map_or(true, |(_, bt)| h < bt) {
                        best = Some((i, h));
                    }
                }
            }
        }
        best
    }

    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<usize> {
        self.pick_hit(origin, dir).map(|h| h.0)
    }
}

fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let disc = b * b - (oc.dot(oc) - r * r);
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t = if -b - s > 0.0 { -b - s } else { -b + s };
    (t > 0.0).then_some(t)
}

fn ray_aabb(o: Vec3, d: Vec3, b: Aabb) -> bool {
    let inv = Vec3::ONE / d;
    let t1 = (b.min - o) * inv;
    let t2 = (b.max - o) * inv;
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    tmax >= tmin.max(0.0)
}

fn ray_tri(o: Vec3, d: Vec3, p0: Vec3, p1: Vec3, p2: Vec3) -> Option<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
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
    (t > 1e-6).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    #[test]
    fn pick_hits_cube_in_front() {
        let mut s = Scene::default();
        let mut o = Object::new("Cube", Primitive::Cube { size: 1.0 }.build());
        o.translation = Vec3::new(0.0, 0.0, -5.0);
        s.objects.push(o);
        s.fix_ids();
        assert_eq!(s.pick(Vec3::ZERO, Vec3::NEG_Z), Some(0));
        assert_eq!(s.pick(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Z), None);
    }

    #[test]
    fn child_follows_parent() {
        let mut s = Scene::default();
        let mut a = Object::new("A", MeshData::default());
        a.translation = Vec3::new(10.0, 0.0, 0.0);
        let mut b = Object::new("B", MeshData::default());
        b.translation = Vec3::new(1.0, 0.0, 0.0);
        s.objects.push(a);
        s.objects.push(b);
        s.fix_ids();
        let pid = s.objects[0].id;
        s.objects[1].parent = Some(pid);
        let w = s.world_matrices();
        assert!((w[1].w_axis.x - 11.0).abs() < 1e-5);
    }
}
